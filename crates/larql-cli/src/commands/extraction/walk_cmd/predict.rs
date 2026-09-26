//! Dense predict paths: local weights and remote FFN.

use larql_inference::{
    predict_with_ffn, predict_with_router, vindex::WalkFfn, LayerFfnRouter, LayerShardedBackend,
    ModelWeights, SparseFfn, WeightFfn,
};
use larql_vindex::{tokenizers, SilentLoadCallbacks, VectorIndex};
use std::time::Instant;

#[allow(unused_imports)]
use super::*;

/// Core predict logic shared by model and vindex paths.
pub(super) fn run_predict_inner(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    args: &WalkArgs,
    index: &VectorIndex,
) -> Result<(), Box<dyn std::error::Error>> {
    let verbose = args.verbose;

    let encoding = tokenizer
        .encode(args.prompt.as_str(), true)
        .map_err(|e| format!("tokenize error: {e}"))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();
    vlog!(
        verbose,
        "Prompt: {:?} ({} tokens)",
        args.prompt,
        token_ids.len()
    );

    // Remote FFN short-circuit: attention runs locally, FFN hits the server
    // per layer. Mutually exclusive with --compare (the comparison backends
    // need local FFN weights to diff against).
    if let Some(ref url) = args.ffn_remote {
        if args.compare {
            return Err("--compare is incompatible with --ffn-remote \
                       (comparison backends require local FFN)"
                .into());
        }
        return run_predict_remote(weights, tokenizer, &token_ids, args, url);
    }

    // Walk FFN forward pass (with trace for analysis output)
    let walk_ffn = WalkFfn::new_with_trace(weights, index, args.top_k);
    let start = Instant::now();

    // Autoregressive streaming path — default for `larql run`.
    // max_tokens == 1 preserves the legacy "show top-K predictions
    // for the next token" behavior of `dev walk --predict`.
    if args.max_tokens > 1 {
        generate_stream(weights, tokenizer, &walk_ffn, &token_ids, args, verbose);
        let walk_elapsed = start.elapsed();
        vlog!(
            verbose,
            "  Walk forward: {:.1}s",
            walk_elapsed.as_secs_f64()
        );
        return Ok(());
    }

    let result = predict_with_ffn(
        weights,
        tokenizer,
        &token_ids,
        args.predict_top_k,
        &walk_ffn,
    );
    let walk_elapsed = start.elapsed();

    let trace = walk_ffn.take_trace();

    if verbose {
        println!("\n── Walk Trace ──");
        print_walk_trace(&trace, args.down_top_k);
        println!();
    }

    print_predictions("walk", &result.predictions, verbose);
    vlog!(
        verbose,
        "  Walk forward: {:.1}s",
        walk_elapsed.as_secs_f64()
    );

    if args.compare {
        let start = Instant::now();
        let dense_result =
            larql_inference::predict(weights, tokenizer, &token_ids, args.predict_top_k);
        let dense_elapsed = start.elapsed();

        print_predictions("dense", &dense_result.predictions, verbose);
        vlog!(
            verbose,
            "  Dense forward: {:.1}s",
            dense_elapsed.as_secs_f64()
        );

        let sparse_ffn = SparseFfn {
            weights,
            top_k: args.top_k,
        };
        let start = Instant::now();
        let sparse_result = predict_with_ffn(
            weights,
            tokenizer,
            &token_ids,
            args.predict_top_k,
            &sparse_ffn,
        );
        let sparse_elapsed = start.elapsed();

        print_predictions(
            &format!("sparse:{}", args.top_k),
            &sparse_result.predictions,
            verbose,
        );
        vlog!(
            verbose,
            "  Sparse forward: {:.1}s",
            sparse_elapsed.as_secs_f64()
        );

        let weight_ffn = WeightFfn { weights };
        let walk_ffn2 = WalkFfn::new(weights, index, args.top_k);
        let num_layers = weights.num_layers;
        let switch = num_layers * 3 / 4;
        let mut backends: Vec<&dyn larql_inference::FfnBackend> = vec![&weight_ffn; num_layers];
        (switch..num_layers).for_each(|l| {
            backends[l] = &walk_ffn2;
        });
        let router = LayerFfnRouter::per_layer(backends);
        let start = Instant::now();
        let hybrid_result =
            predict_with_router(weights, tokenizer, &token_ids, args.predict_top_k, &router);
        let hybrid_elapsed = start.elapsed();

        print_predictions(
            &format!(
                "hybrid (dense:0-{}, walk:{}-{})",
                switch - 1,
                switch,
                num_layers - 1
            ),
            &hybrid_result.predictions,
            verbose,
        );
        vlog!(
            verbose,
            "  Hybrid forward: {:.1}s",
            hybrid_elapsed.as_secs_f64()
        );

        println!();
        println!(
            "{:<40} {:<15} {:>8} {:>8}",
            "Backend", "Top-1", "Prob", "Time"
        );
        println!("{}", "-".repeat(75));
        print_summary_row("walk", &result.predictions, walk_elapsed);
        print_summary_row("dense", &dense_result.predictions, dense_elapsed);
        print_summary_row(
            &format!("sparse:{}", args.top_k),
            &sparse_result.predictions,
            sparse_elapsed,
        );
        print_summary_row(
            &format!("dense:0-{},walk:{}-{}", switch - 1, switch, num_layers - 1),
            &hybrid_result.predictions,
            hybrid_elapsed,
        );
    }

    Ok(())
}

/// Remote FFN forward pass: attention local, FFN served over HTTP by
/// `larql-server`. See `crates/larql-inference/src/ffn/remote.rs` for the
/// backend and `crates/larql-server/src/routes/walk_ffn.rs` for the
/// server endpoint.
///
pub(super) fn run_predict_remote(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    args: &WalkArgs,
    url: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let verbose = args.verbose;
    let timeout = std::time::Duration::from_secs(args.ffn_remote_timeout_secs);

    vlog!(verbose, "Connecting to remote FFN: {url}");
    let remote = LayerShardedBackend::connect(url, timeout)?;
    if remote.hidden_size() != weights.hidden_size {
        return Err(format!(
            "remote hidden_size {} != local attention hidden_size {} \
             — client and server must be the same model",
            remote.hidden_size(),
            weights.hidden_size,
        )
        .into());
    }
    vlog!(
        verbose,
        "  connected: hidden={} primary={}",
        remote.hidden_size(),
        remote.primary_url()
    );

    let start = Instant::now();

    if args.max_tokens > 1 && args.ffn_dispatch == "batch" {
        // Batch predispatch: use Metal pipeline with parallel per-layer HTTP
        // requests. Requires the Q4K vindex with interleaved FFN mmap.
        use larql_inference::generate_with_remote_ffn_batch;
        let mut cb = SilentLoadCallbacks;
        let mut index = VectorIndex::load_vindex(
            args.index
                .as_deref()
                .expect("index required for batch dispatch"),
            &mut cb,
        )?;
        index.load_attn_kquant(
            args.index
                .as_deref()
                .expect("index required for batch dispatch"),
        )?;
        index.load_interleaved_kquant(
            args.index
                .as_deref()
                .expect("index required for batch dispatch"),
        )?;
        let _ = index.load_lm_head_kquant(
            args.index
                .as_deref()
                .expect("index required for batch dispatch"),
        );
        let backend = larql_compute::default_backend();
        let wrapped_prompt = larql_inference::chat::render_user_prompt(
            args.index.as_deref().expect("index required"),
            weights.arch.family(),
            args.prompt.as_str(),
        )?;
        let batch_ids = larql_inference::encode_prompt(tokenizer, &*weights.arch, &wrapped_prompt)
            .map_err(|e| format!("tokenize error: {e}"))?;
        let eos = larql_inference::layer_graph::generate::eos::EosConfig::from_vindex_dir(
            args.index.as_deref().expect("index required"),
        );
        let result = generate_with_remote_ffn_batch(
            weights,
            tokenizer,
            batch_ids,
            args.max_tokens,
            &index,
            &*backend,
            &remote,
            &eos,
            args.ffn_predispatch_iters,
        )
        .map_err(|e| format!("remote-ffn batch generate failed: {e}"))?;
        for tok in &result.tokens {
            print!("{tok}");
        }
        if !result.tokens.is_empty() {
            println!();
        }
        if verbose {
            eprintln!(
                "  Forward pass: {:.2}s  (FFN → {} batch)",
                start.elapsed().as_secs_f64(),
                url
            );
        }
        return Ok(());
    }

    if args.max_tokens > 1 {
        generate_stream(weights, tokenizer, &remote, token_ids, args, verbose);
        if verbose {
            eprintln!(
                "  Forward pass: {:.2}s  (FFN → {})",
                start.elapsed().as_secs_f64(),
                url
            );
        }
        return Ok(());
    }

    let result = predict_with_ffn(weights, tokenizer, token_ids, args.predict_top_k, &remote);
    let elapsed = start.elapsed();

    print_predictions("walk (ffn remote)", &result.predictions, verbose);
    if verbose {
        eprintln!(
            "  Forward pass: {:.2}s  (FFN → {})",
            elapsed.as_secs_f64(),
            url
        );
    }

    Ok(())
}
