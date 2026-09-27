//! Q4_K predict and generate paths: resident, uncached CPU, Metal, and remote FFN.

use larql_inference::{LayerShardedBackend, ModelWeights};
use larql_vindex::{tokenizers, VectorIndex};
use std::time::Instant;

#[allow(unused_imports)]
use super::*;

/// Build the Metal compute backend for `--metal`, or a clear error when the
/// binary lacks the backend or the host lacks a device. Delegates to the
/// shared registry-backed factory in `backend_select`.
pub(super) fn metal_backend_box(
) -> Result<Box<dyn larql_compute::ComputeBackend>, Box<dyn std::error::Error>> {
    crate::backend_select::backend_for_metal_flag(true)
}

/// Predict against a Q4_K / Q6_K vindex: dequantise each layer's attn + FFN
/// weights just-in-time, run the standard f32 forward block, drop, repeat.
/// Same observable output as [`run_predict_inner`] — just a different memory
/// profile (one layer's worth of f32 heap instead of the whole model).
pub(super) fn run_predict_q4k(
    weights: &mut ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    args: &WalkArgs,
    _index: &VectorIndex,
) -> Result<(), Box<dyn std::error::Error>> {
    let verbose = args.verbose;
    // Apply the same chat-template wrapping the gRPC path uses, so dense
    // Gemma 4 (and any other instruct family) doesn't see the raw user
    // prompt and fall into degenerate "answer-from-text" / "The answer is:"
    // loops. Falls back to raw prompt for vindexes without a chat template.
    let vindex_dir_for_chat = args.index.as_deref();
    let wrapped_prompt = match vindex_dir_for_chat {
        Some(dir) => larql_inference::chat::render_user_prompt(
            dir,
            weights.arch.family(),
            args.prompt.as_str(),
        )
        .unwrap_or_else(|e| {
            vlog!(
                verbose,
                "[chat] wrap failed ({e}) — falling back to raw prompt"
            );
            args.prompt.clone()
        }),
        None => args.prompt.clone(),
    };
    let token_ids =
        larql_inference::encode_prompt(tokenizer, &*weights.arch, wrapped_prompt.as_str())
            .map_err(|e| format!("tokenize error: {e}"))?;
    vlog!(
        verbose,
        "Prompt: {:?} (wrapped {} chars, {} tokens)",
        args.prompt,
        wrapped_prompt.len(),
        token_ids.len()
    );

    // The Q4 vindex we loaded already lives inside the VectorIndex used by
    // the walk caller, but we need our OWN VectorIndex with the Q4 mmaps
    // loaded (load_attn_kquant, load_interleaved_kquant) since the caller's index
    // might have been constructed without those accessors wired up.
    let vindex_path = args
        .index
        .as_deref()
        .ok_or("--index required for Q4 predict path")?;
    let mut cb = larql_vindex::SilentLoadCallbacks;
    let mut index = VectorIndex::load_vindex(vindex_path, &mut cb)?;
    index.load_attn_kquant(vindex_path)?;
    index.load_interleaved_kquant(vindex_path)?;
    let _ = index.load_lm_head_kquant(vindex_path);

    // Metal Q4K path (`--metal`) routes autoregressive generation through the
    // fused `full_pipeline_q4` prefill + `decode_token` KV-cached decode in
    // `layer_graph::generate`. Works for pre-norm (Llama/Mistral) and
    // post-norm + QK-norm (Gemma 3/4) architectures. CPU path below is the
    // fallback for when the backend is absent or for diffing.
    let start = Instant::now();

    // Autoregressive multi-token generation. For Q4K on CPU, we build
    // a per-layer CPU FfnBackend-compatible view and loop via the
    // generic `generate_stream`. Metal shader autoregressive generation
    // is a separate path (see `larql-inference/src/layer_graph/generate.rs`)
    // and is wired to `--metal`; that path is KV-cached and much faster.
    if args.max_tokens > 1 && !args.metal {
        // CPU Q4K autoregressive: per-step, dequantise layer weights
        // just-in-time (`predict_kquant` does this internally) and loop.
        // Not token-cached, so O(N²) but correct. For speed use --metal.
        //
        // This path has no KV cache and therefore no engine to select, so a
        // named `--engine` cannot be honoured here. Say so instead of running
        // something else under the caller's chosen label: silently dropping
        // the flag is what made every engine look identical through
        // `larql run` (issue #199), since they were all the same path.
        // Fast path first: a non-PLE hybrid-MoE model can run the resident,
        // KV-cached route instead of the O(N^2) re-dequantising loop below.
        //
        // Measured on gemma4-26b-a4b (M3 Max, 2026-08-08): 1745 ms/token on
        // the uncached path against 26.8 ms/token here — 65x — for 2.1 GB
        // more RSS (15.5 -> 17.6 GB). The two compute the same model: same
        // layers, same experts (which stay Q4_K in both), same lm_head. All
        // that differs is *when* attention and the dense FFN are dequantised:
        // once, up front, instead of on every token. `larql bench --cpu` has
        // always taken this route, which is why its numbers and `larql run`'s
        // disagreed by a factor nobody could place.
        // `LARQL_CPU_RESIDENT=0` forces the uncached route so the two can be
        // A/B'd in one binary under identical conditions. Without it the old
        // path becomes unreachable on MoE models the moment this lands, and a
        // before/after measured on two different builds — or, worse, two
        // different prompts — is not a comparison.
        let resident_allowed = std::env::var("LARQL_CPU_RESIDENT")
            .map(|v| v != "0")
            .unwrap_or(true);
        if resident_allowed
            && !arch_needs_per_layer_embeddings(weights)
            && weights.arch.is_hybrid_moe()
        {
            return run_q4k_generate_cpu_resident(weights, tokenizer, &token_ids, args, &index);
        }

        // Uncached fallback: PLE architectures and dense models. This path has
        // no KV cache and therefore no engine to select, so a named `--engine`
        // cannot be honoured — say so instead of running something else under
        // the caller's chosen label (issue #199).
        if let Some(spec) = requested_engine_spec(args) {
            return Err(engine_unsupported_on_uncached_path(&spec).into());
        }
        return run_q4k_generate_cpu(weights, tokenizer, &token_ids, args, &index);
    }

    let result = if args.metal {
        // `larql_compute::default_backend()` always returns CPU since
        // the GPU-backend extraction (see its doc-comment). GPU
        // selection is the caller's responsibility — mirror what
        // `bench/local_runtime.rs::build_runtime` does and reach for
        // `MetalBackend::new()` directly when `--metal` is set, so the
        // fused Q4 prefill + KV-cached decode kernels actually fire
        // here. The previous `default_backend()` call silently fell
        // through to CPU's `generate_via_cpu_q4k` fallback which
        // produces degenerate output ("ikea ikea ikea…"), masquerading
        // as a Granite/Gemma forward-path regression.
        let backend: Box<dyn larql_compute::ComputeBackend> = metal_backend_box()?;
        if !backend.supports_quant(::larql_compute::QuantFormat::Q4_K) {
            return Err("Metal backend doesn't report Q4_K support — \
                 check `larql diag <vindex>` for backend capabilities."
                .into());
        }
        vlog!(
            verbose,
            "Backend: {} (Metal Q4K prefill + KV-cached decode)",
            backend.name()
        );
        // --metal + --max-tokens > 1: route to the existing shader
        // autoregressive generate() in `larql-inference/src/layer_graph`
        // (GPU prefill + KV-cached decode). That function returns its
        // own tokens list; we stream them and exit.
        if args.max_tokens > 1 {
            use std::io::Write;
            let cached_layers =
                larql_inference::layer_graph::CachedLayerGraph::from_residuals(Vec::new());
            let num_layers = weights.num_layers;
            let result = larql_inference::layer_graph::generate(
                weights,
                tokenizer,
                &token_ids,
                args.max_tokens,
                &index,
                &*backend,
                &cached_layers,
                0..num_layers,
            );
            let mut stdout = std::io::stdout();
            for (tok, _) in &result.tokens {
                print!("{tok}");
                let _ = stdout.flush();
            }
            println!();
            if verbose {
                eprintln!(
                    "  prefill: {:.1}ms  decode avg: {:.1}ms/tok  ({:.1} tok/s)",
                    result.prefill_ms,
                    result.avg_decode_ms(),
                    result.decode_tok_s(),
                );
            }
            return Ok(());
        }
        larql_inference::vindex::predict_kquant_metal(
            weights,
            tokenizer,
            &token_ids,
            args.predict_top_k,
            &index,
            &*backend,
        )
    } else {
        vlog!(verbose, "Backend: CPU (Accelerate + dequantise-per-layer)");
        larql_inference::vindex::predict_kquant(
            weights,
            tokenizer,
            &token_ids,
            args.predict_top_k,
            &index,
        )
    };
    vlog!(
        verbose,
        "Q4 forward pass: {:.2}s",
        start.elapsed().as_secs_f64()
    );

    print_predictions("walk (q4k)", &result.predictions, verbose);

    Ok(())
}

/// Q4_K + remote FFN: local attention (dequant per layer), FFN over HTTP.
///
/// The existing `run_predict_remote` path expects attention tensors to live
/// inside `ModelWeights.tensors`, which is true only after the per-layer
/// Q4K dequant. So instead of routing through `run_predict_remote` we call
/// `predict_kquant_with_ffn` directly with a `RemoteWalkBackend` — that path
/// dequantises only Q/K/V/O per layer and skips the FFN dequant entirely.
pub(super) fn run_predict_q4k_remote(
    weights: &mut ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    args: &WalkArgs,
    vindex_path: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let verbose = args.verbose;
    let url = args.ffn_remote.as_ref().expect("ffn_remote is set");
    let timeout = std::time::Duration::from_secs(args.ffn_remote_timeout_secs);

    vlog!(verbose, "Connecting to remote FFN: {url}");
    let remote = LayerShardedBackend::connect(url, timeout)?;
    if remote.hidden_size() != weights.hidden_size {
        return Err(format!(
            "remote hidden_size {} != local hidden_size {} — client and server \
             must be the same model",
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

    // Build a fresh VectorIndex with the q4k attention mmap wired in.
    // Q4K FFN mmap is NOT loaded — FFN runs on the server.
    let mut cb = larql_vindex::SilentLoadCallbacks;
    let mut index = VectorIndex::load_vindex(vindex_path, &mut cb)?;
    index.load_attn_kquant(vindex_path)?;

    let token_ids = larql_inference::encode_prompt(tokenizer, &*weights.arch, args.prompt.as_str())
        .map_err(|e| format!("tokenize error: {e}"))?;
    vlog!(
        verbose,
        "Prompt: {:?} ({} tokens)",
        args.prompt,
        token_ids.len()
    );

    let start = Instant::now();
    // A refusal from the shards ends the command. Printing predictions built
    // without the layer that refused would report a walk the model never ran.
    let result = larql_inference::vindex::predict_kquant_with_ffn(
        weights,
        tokenizer,
        &token_ids,
        args.predict_top_k,
        &index,
        &remote,
    )
    .map_err(|refusal| format!("remote FFN refused ({}): {refusal}", refusal.kind()))?;
    let elapsed = start.elapsed();

    print_predictions("walk (q4k + ffn remote)", &result.predictions, verbose);
    if verbose {
        eprintln!(
            "  Forward pass: {:.2}s  (FFN → {})",
            elapsed.as_secs_f64(),
            url
        );
    }

    Ok(())
}

/// Whether this architecture applies Per-Layer Embeddings.
///
/// The resident engine path does not apply PLE, so those architectures
/// (Gemma 4 E-series) must keep the full-recompute route — the same guard
/// `bench`'s in-process MoE runner enforces, kept in both places because a
/// silent mismatch here is a wrong answer rather than a slow one.
pub(super) fn arch_needs_per_layer_embeddings(weights: &ModelWeights) -> bool {
    weights.arch.per_layer_input_gate_key(0).is_some()
}

/// CPU Q4K generation over resident attention + dense FFN, with a KV cache.
///
/// Dequantises attention and the dense FFN slab to f32 once and keeps them
/// resident for the whole run; experts stay Q4_K and are read directly by
/// `LocalMoeFfn`. Identical in what it computes to
/// [`run_q4k_generate_cpu`] — the difference is that the uncached loop redoes
/// that dequantisation, and re-runs the entire growing sequence, on every
/// token.
pub(super) fn run_q4k_generate_cpu_resident(
    weights: &mut ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    initial_ids: &[u32],
    args: &WalkArgs,
    index: &VectorIndex,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;
    let start = Instant::now();

    for layer in 0..weights.num_layers {
        larql_inference::vindex::insert_q4k_layer_tensors_resident(weights, index, layer)
            .map_err(|e| format!("failed to dequantise layer {layer} to f32: {e}"))?;
    }
    let resident_ms = start.elapsed().as_secs_f64() * 1000.0;

    // `--engine` is honoured here, unlike on the uncached path: this route
    // has a real KV engine to select.
    let engine_spec = requested_engine_spec(args);
    let spec = engine_spec.as_deref().unwrap_or("standard");
    let kind = larql_kv::EngineKind::from_name(spec)
        .ok_or_else(|| format!("--engine {spec:?}: unknown engine"))?;
    let engine_label = kind.display_name().to_string();
    let mut engine = kind.build(larql_inference::cpu_engine_backend());

    let weights_ref: &ModelWeights = weights;
    let moe_ffn = larql_inference::ffn::LocalMoeFfn {
        weights: weights_ref,
        index: Some(index),
    };

    let decode_start = Instant::now();
    let mut stdout = std::io::stdout();
    let mut emitted = 0usize;
    let ids = larql_kv::generation::generate_with_engine_resident(
        &mut engine,
        weights_ref,
        tokenizer,
        &moe_ffn,
        index,
        initial_ids,
        args.max_tokens,
        |_id, tok| {
            print!("{tok}");
            let _ = stdout.flush();
            emitted += 1;
        },
    );
    println!();

    if args.verbose {
        let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
        let n = ids.len().saturating_sub(initial_ids.len()).max(emitted);
        eprintln!(
            "  Q4K CPU generate (resident, {}): {:.2}s  ({} tokens, {:.1} ms/token)",
            engine_label,
            decode_ms / 1000.0,
            n,
            if n == 0 { 0.0 } else { decode_ms / n as f64 },
        );
        eprintln!("  f32-resident dequantisation: {resident_ms:.0} ms (once)");
    }
    Ok(())
}

/// CPU Q4K autoregressive generation. Per-step: dequantise the layer's
/// Q/K/V/O + gate/up/down weights (via `predict_kquant` internals), run
/// the forward pass, take argmax, append, repeat. Streams tokens.
pub(super) fn run_q4k_generate_cpu(
    weights: &mut ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    initial_ids: &[u32],
    args: &WalkArgs,
    index: &VectorIndex,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;
    let verbose = args.verbose;
    let mut ids = initial_ids.to_vec();
    let mut stdout = std::io::stdout();
    let start = Instant::now();

    for _step in 0..args.max_tokens {
        let result = larql_inference::vindex::predict_kquant(weights, tokenizer, &ids, 1, index);
        let next_id = match result.token_ids.first() {
            Some(&id) => id,
            None => break,
        };
        let tok_str = result
            .predictions
            .first()
            .map(|p| p.0.as_str())
            .unwrap_or("");
        print!("{tok_str}");
        let _ = stdout.flush();
        ids.push(next_id);
        if is_stop_token(tok_str) {
            break;
        }
    }
    println!();
    if verbose {
        eprintln!(
            "  Q4K CPU generate: {:.2}s  ({} tokens)",
            start.elapsed().as_secs_f64(),
            ids.len() - initial_ids.len(),
        );
    }
    Ok(())
}
