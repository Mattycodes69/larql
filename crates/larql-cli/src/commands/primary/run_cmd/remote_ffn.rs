//! `--ffn URL`: every layer's FFN is a round trip to a remote server.

#[allow(unused_imports)]
use super::*;

/// `--ffn URL` dispatch path for dense models.
///
/// Metal runs attention on the local GPU. Every layer's FFN is a round trip
/// to the remote server at `ffn_url` via `LayerShardedBackend`. The local
/// vindex supplies attention weights; the remote server supplies FFN outputs.
///
/// This is analogous to `run_with_moe_shards` for hybrid-MoE models, but
/// simpler: there is no local FFN and no router — every layer unconditionally
/// calls the remote server.
pub(super) fn run_with_remote_ffn(
    vindex_path: &std::path::Path,
    prompt: &str,
    ffn_url: &str,
    ffn_timeout_secs: u64,
    max_tokens: usize,
    dispatch: &str,
    predispatch_iters: usize,
    metal: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    use larql_inference::{
        generate_with_remote_ffn, generate_with_remote_ffn_batch, LayerShardedBackend,
    };
    use std::time::Duration;

    let timeout = Duration::from_secs(ffn_timeout_secs);
    let backend: Box<dyn larql_compute::ComputeBackend> =
        crate::backend_select::backend_for_metal_flag(metal)?;
    eprintln!("Connecting to remote FFN at {ffn_url}…");
    let remote = LayerShardedBackend::connect(ffn_url, timeout)
        .map_err(|e| format!("failed to connect to remote FFN server: {e}"))?;
    eprintln!("  Attention:  {} (local)", backend.name());
    eprintln!("  FFN:        remote  ({})  dispatch={dispatch}", ffn_url);

    let mut cb = larql_vindex::SilentLoadCallbacks;
    let weights = larql_vindex::load_model_weights_kquant(vindex_path, &mut cb)
        .map_err(|e| format!("failed to load client weights: {e}"))?;
    let tokenizer = larql_vindex::load_vindex_tokenizer(vindex_path)
        .map_err(|e| format!("failed to load tokenizer: {e}"))?;
    let mut index = larql_vindex::VectorIndex::load_vindex(vindex_path, &mut cb)
        .map_err(|e| format!("failed to load vindex: {e}"))?;
    index
        .load_attn_kquant(vindex_path)
        .map_err(|e| format!("failed to load attn Q4K: {e}"))?;
    index
        .load_interleaved_kquant(vindex_path)
        .map_err(|e| format!("failed to load interleaved Q4K: {e}"))?;
    let _ = index.load_lm_head_kquant(vindex_path);

    let wrapped_prompt =
        larql_inference::chat::render_user_prompt(vindex_path, weights.arch.family(), prompt)?;
    let prompt_ids = larql_inference::encode_prompt(&tokenizer, &*weights.arch, &wrapped_prompt)
        .map_err(|e| format!("failed to tokenise prompt: {e}"))?;
    eprintln!("[chat] tokenised to {} ids", prompt_ids.len());

    let eos = larql_inference::layer_graph::generate::eos::EosConfig::from_vindex_dir(vindex_path);
    let result = if dispatch == "batch" {
        generate_with_remote_ffn_batch(
            &weights,
            &tokenizer,
            prompt_ids,
            max_tokens,
            &index,
            &*backend,
            &remote,
            &eos,
            predispatch_iters,
        )
    } else {
        generate_with_remote_ffn(
            &weights, &tokenizer, prompt_ids, max_tokens, &index, &*backend, &remote, &eos,
        )
    }
    .map_err(|e| format!("remote-ffn generate failed ({dispatch}): {e}"))?;
    if result.wire_fallbacks > 0 {
        eprintln!(
            "[remote-ffn] warning: {} call(s) fell back from the Q8K wire to f32",
            result.wire_fallbacks
        );
    }

    if std::env::var("LARQL_DEBUG_TOKENS").is_ok() {
        eprintln!(
            "[debug] dispatch={dispatch} iters={predispatch_iters} n_tokens={} tokens={:?}",
            result.tokens.len(),
            result.tokens
        );
    }

    for tok in &result.tokens {
        print!("{tok}");
    }
    if !result.tokens.is_empty() {
        println!();
    }
    let _ = std::io::Write::flush(&mut std::io::stdout());

    let n = result.decode_ms.len();
    if n > 0 {
        let avg = result.decode_ms.iter().sum::<f64>() / n as f64;
        let tok_s = 1000.0 / avg;
        let num_layers = weights.num_layers;
        let hidden = weights.hidden_size;
        // One f32 residual in each direction per layer.
        let bytes_per_token = num_layers * hidden * std::mem::size_of::<f32>();
        let kb = |b: usize| b as f64 / 1024.0;
        eprintln!();
        eprintln!("  decode:     {tok_s:.1} tok/s");
        eprintln!(
            "  bytes sent: ~{:.0} KB  ({num_layers} layers × hidden × f32)",
            kb(bytes_per_token * n)
        );
        eprintln!(
            "  bytes recv: ~{:.0} KB  ({num_layers} layers × hidden × f32)",
            kb(bytes_per_token * n)
        );
    }
    Ok(())
}
