//! Q4K helpers — attention dequantisation re-export + WalkFfn-backed
//! forward paths.
//!
//! `ensure_attn_tensors_dequantised` moved to
//! [`larql_inference::vindex::dequant`] (2026-05-16) so the
//! `KvDispatch` trait impls in `larql-inference::kv_dispatch::*` can
//! call it without a `larql-kv → larql-inference → larql-kv` cycle.
//! Re-exported here to keep existing call sites compiling.

use larql_compute::ComputeBackend;
use larql_vindex::VectorIndex;
use ndarray::Array2;

use super::compute::{last_row, recompute_kv, RsPrefillResult};
use super::store::RsStore;
use crate::profiler::EngineProfiler;
use larql_inference::attention::run_attention_with_kv_backend;
use larql_inference::attention::SharedKV;
use larql_inference::forward::{embed_tokens_pub, run_ffn};
use larql_inference::model::ModelWeights;
use larql_inference::vindex::{WalkFfn, WalkFfnConfig};

/// Re-export — see [`larql_inference::vindex::dequant::ensure_attn_tensors_dequantised`].
pub use larql_inference::vindex::ensure_attn_tensors_dequantised;

/// Prefill using `WalkFfn` (Q4K FFN) instead of `BackendFfn` (f32 FFN).
pub(super) fn rs_prefill_walk(
    weights: &ModelWeights,
    index: &VectorIndex,
    token_ids: &[u32],
    max_window: Option<usize>,
    backend: &dyn ComputeBackend,
) -> RsPrefillResult {
    let num_layers = weights.num_layers;
    let seq_len = token_ids.len();
    let mut h = embed_tokens_pub(weights, token_ids);
    let mut stored: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
    let be = Some(backend);

    // Hoist WalkFfn construction out of the per-layer loop. Previously
    // this rebuilt the WalkFfn 34 times per prefill (once per layer);
    // now once total. WalkFfn carries no per-layer state — it's the
    // gate-index + backend pair, both stable across the loop.
    let walk_ffn = WalkFfn::from_config(weights, index, WalkFfnConfig::dense(num_layers))
        .with_backend(backend);

    // Capture per-layer K/V from each layer's attention block. These
    // are *already computed* by the forward pass; previously discarded
    // and re-derived from residuals on every decode step (W2 measured
    // 80% of decode time spent on this redundant recompute). Stashing
    // them here means decode_step appends one row per layer instead
    // of recomputing the entire hot tier.
    let mut hot_kv_captured: Vec<SharedKV> = Vec::with_capacity(num_layers);
    for layer in 0..num_layers {
        stored.push(h.clone());
        let (h_post_attn, k, v) = run_attention_with_kv_backend(weights, &h, layer, be)
            .expect("attention failed during MarkovRS Q4K prefill");
        hot_kv_captured.push((k, v));
        let (h_out, _) = run_ffn(weights, &h_post_attn, layer, &walk_ffn, false);
        h = h_out;
    }

    let mut rs = RsStore {
        hot_len: stored.first().map_or(0, |s| s.shape()[0]),
        stored,
        cold_residuals: None,
        cold_kv: None,
        hot_kv: Some(hot_kv_captured),
        cold_abs_start: 0,
        next_position: seq_len,
        max_window,
    };

    // Build pre-clip evicted-rows lookup so we can move the evicted
    // top of `hot_kv` directly into `cold_kv` without calling
    // `recompute_kv` (the K/V is already correct — we just need to
    // slice it).
    let pre_clip_hot_rows: Vec<usize> = if rs.hot_kv.is_some() {
        let window = max_window.unwrap_or(usize::MAX);
        rs.stored
            .iter()
            .map(|s| s.shape()[0].saturating_sub(window))
            .collect()
    } else {
        Vec::new()
    };
    let evicted_hot_kv = rs
        .hot_kv
        .as_ref()
        .filter(|_| pre_clip_hot_rows.iter().any(|&n| n > 0))
        .and_then(|hot_kv| RsStore::snapshot_evicted_hot_kv(hot_kv, &pre_clip_hot_rows));

    let mut cold: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
    for layer in 0..num_layers {
        rs.clip_layer(layer, &mut cold);
    }
    rs.finalise_hot_len_after_clip();
    if cold.first().map_or(0, |c| c.shape()[0]) > 0 {
        let cold_kv: Vec<SharedKV> = if let Some(evicted) = evicted_hot_kv {
            // Fast path: reuse the K/V we just computed during the
            // forward pass. No `recompute_kv` call needed — the
            // evicted slices ARE the cold K/V.
            evicted
        } else {
            // Fallback: forward pass didn't capture K/V (shouldn't
            // happen under current code, kept for safety).
            (0..num_layers)
                .map(|layer| {
                    recompute_kv(weights, &cold[layer], layer, 0, backend, Some(index))
                        .expect("cold K/V pre-computation failed")
                })
                .collect()
        };
        rs.cold_residuals = Some(cold);
        rs.cold_kv = Some(cold_kv);
        rs.cold_abs_start = 0;
    }
    let window_tokens = rs.window_tokens();
    let memory_bytes = rs.memory_bytes();
    RsPrefillResult {
        hidden: last_row(&h),
        store: rs,
        memory_bytes,
        window_tokens,
    }
}

/// Decode step using `WalkFfn` (Q4K FFN). Pass `Some(profiler)` to
/// accumulate per-stage wall-clock; pass `None` for the unprofiled
/// path. Sibling of [`super::compute::rs_decode_step_inner`] for the
/// Q4K side.
pub(super) fn rs_decode_step_walk(
    weights: &ModelWeights,
    index: &VectorIndex,
    new_token_id: u32,
    rs: RsStore,
    backend: &dyn ComputeBackend,
    mut profiler: Option<&mut EngineProfiler>,
) -> Option<(Array2<f32>, RsStore)> {
    use ndarray::s;
    use std::time::Instant;

    // Verbose env-var instrumentation is kept as an ad-hoc debug
    // channel (prints per-step lines to stderr). The structured
    // `profiler` accumulator is the supported path for
    // `larql bench --profile`.
    let instrument = std::env::var("LARQL_INSTRUMENT_MARKOV").is_ok();
    let timing = profiler.is_some() || instrument;

    let num_layers = weights.num_layers;
    let abs_position = rs.next_position;
    let t_step = if timing { Some(Instant::now()) } else { None };
    let t_embed_start = t_step;
    let mut h_new = embed_tokens_pub(weights, &[new_token_id]);
    let embed_us = t_embed_start
        .map(|t| t.elapsed().as_secs_f64() * 1e6)
        .unwrap_or(0.0);
    let mut new_stored: Vec<Array2<f32>> = Vec::with_capacity(num_layers);

    // Hoist WalkFfn out of the per-layer loop — see note in
    // `rs_prefill_walk`. Was 34× construction per decode step.
    let walk_ffn = WalkFfn::from_config(weights, index, WalkFfnConfig::dense(num_layers))
        .with_backend(backend);

    // Per-stage accumulators. With W2 caching landed, both
    // `recompute_*` timings should be near zero for cached-path decode
    // steps — they only fire on the fallback when hot_kv was dropped
    // (e.g. via_executor path doesn't cache yet, or post-overflow
    // first-step before cache rebuild). `recompute_hot` now also
    // measures the cheap "concat cached cold + cached hot" path.
    let mut recompute_cold_us = 0.0f64;
    let mut recompute_hot_us = 0.0f64;
    let mut attention_us = 0.0f64;
    let mut ffn_us = 0.0f64;
    let mut concat_us = 0.0f64;
    let mut attn_helper_hits = 0usize;
    let mut attn_helper_misses = 0usize;
    let mut s_hot_first_layer = 0usize;

    // Per-layer new hot_kv slices (post-attention), built up across
    // the layer loop and committed to the store at the end.
    let mut new_hot_kvs: Vec<SharedKV> = Vec::with_capacity(num_layers);

    for layer in 0..num_layers {
        let h_hot = &rs.stored[layer];
        let s_hot = h_hot.shape()[0];
        if layer == 0 {
            s_hot_first_layer = s_hot;
        }
        let hot_abs_start = abs_position.saturating_sub(s_hot);
        let c_rows = rs.cold_kv.as_ref().map_or(0, |kv| kv[layer].0.shape()[0]);

        // Build prior_kv. Three paths, in order of preference:
        //   1. Cached hot_kv (+ optional cached cold_kv) — concat is
        //      a memcpy; no projection work. **W2 fast path.**
        //   2. Cached cold_kv only — recompute hot from h_hot, concat
        //      with cold. (Hot K/V wasn't captured; falls back to the
        //      pre-W2 behaviour.)
        //   3. Neither cached — recompute everything from residuals.
        //      Slowest path; fires on first decode after overflow
        //      eviction (cache rebuilds during this step's tail
        //      processing) or on the via_executor path which doesn't
        //      capture K/V yet.
        let (k_full, v_full) =
            if let (Some(hot_kv), maybe_cold) = (rs.hot_kv.as_ref(), rs.cold_kv.as_ref()) {
                let (k_hot, v_hot) = &hot_kv[layer];
                let t_concat = if timing { Some(Instant::now()) } else { None };
                let pair = if let Some(cold_kv) = maybe_cold {
                    let (k_cold, v_cold) = &cold_kv[layer];
                    let c = k_cold.shape()[0];
                    let kv_dim = k_cold.shape()[1];
                    let mut k_combined = Array2::<f32>::zeros((c + s_hot, kv_dim));
                    k_combined.slice_mut(s![..c, ..]).assign(k_cold);
                    k_combined.slice_mut(s![c.., ..]).assign(k_hot);
                    let mut v_combined = Array2::<f32>::zeros((c + s_hot, kv_dim));
                    v_combined.slice_mut(s![..c, ..]).assign(v_cold);
                    v_combined.slice_mut(s![c.., ..]).assign(v_hot);
                    (k_combined, v_combined)
                } else {
                    (k_hot.clone(), v_hot.clone())
                };
                if let Some(t) = t_concat {
                    concat_us += t.elapsed().as_secs_f64() * 1e6;
                }
                pair
            } else if let Some(cold_kv) = &rs.cold_kv {
                let (k_cold, v_cold) = &cold_kv[layer];
                let t_hot = if timing { Some(Instant::now()) } else { None };
                let (k_hot, v_hot) =
                    recompute_kv(weights, h_hot, layer, hot_abs_start, backend, Some(index))?;
                if let Some(t) = t_hot {
                    recompute_hot_us += t.elapsed().as_secs_f64() * 1e6;
                }
                let t_concat = if timing { Some(Instant::now()) } else { None };
                let c = k_cold.shape()[0];
                let kv_dim = k_cold.shape()[1];
                let mut k_combined = Array2::<f32>::zeros((c + s_hot, kv_dim));
                k_combined.slice_mut(s![..c, ..]).assign(k_cold);
                k_combined.slice_mut(s![c.., ..]).assign(&k_hot);
                let mut v_combined = Array2::<f32>::zeros((c + s_hot, kv_dim));
                v_combined.slice_mut(s![..c, ..]).assign(v_cold);
                v_combined.slice_mut(s![c.., ..]).assign(&v_hot);
                if let Some(t) = t_concat {
                    concat_us += t.elapsed().as_secs_f64() * 1e6;
                }
                (k_combined, v_combined)
            } else {
                let (h_full, full_abs_start) = match &rs.cold_residuals {
                    Some(cold) if cold[layer].shape()[0] > 0 => {
                        let h_cold = &cold[layer];
                        let s_cold = h_cold.shape()[0];
                        let hidden = h_hot.shape()[1];
                        let mut combined = Array2::<f32>::zeros((s_cold + s_hot, hidden));
                        combined.slice_mut(s![..s_cold, ..]).assign(h_cold);
                        combined.slice_mut(s![s_cold.., ..]).assign(h_hot);
                        (combined, rs.cold_abs_start)
                    }
                    _ => (h_hot.clone(), hot_abs_start),
                };
                let t_cold = if timing { Some(Instant::now()) } else { None };
                let pair = recompute_kv(
                    weights,
                    &h_full,
                    layer,
                    full_abs_start,
                    backend,
                    Some(index),
                )?;
                if let Some(t) = t_cold {
                    recompute_cold_us += t.elapsed().as_secs_f64() * 1e6;
                }
                pair
            };

        new_stored.push(h_new.clone());

        let t_attn = if timing { Some(Instant::now()) } else { None };
        let kv_pair = (k_full, v_full);
        let native_result = larql_inference::vindex::attention_decode_step_native(
            weights,
            index,
            backend,
            &h_new,
            layer,
            Some(&kv_pair),
            abs_position,
        );
        if instrument {
            if native_result.is_some() {
                attn_helper_hits += 1;
            } else {
                attn_helper_misses += 1;
            }
        }
        let (h_post_attn, new_kv_full) = native_result.or_else(|| {
            larql_inference::attention::run_attention_block_decode_step_backend(
                weights,
                &h_new,
                layer,
                Some(&kv_pair),
                abs_position,
                Some(backend),
            )
        })?;
        if let Some(t) = t_attn {
            attention_us += t.elapsed().as_secs_f64() * 1e6;
        }

        // Capture the new hot_kv slice for this layer.
        // `new_kv_full` = (cold_kv ++ hot_kv ++ new_row). Slicing off
        // `c_rows` (the cold portion, unchanged this step) leaves
        // `hot_kv ++ new_row`, which is exactly the new hot K/V to
        // cache for the next step.
        let new_hot_kv = (
            new_kv_full.0.slice(s![c_rows.., ..]).to_owned(),
            new_kv_full.1.slice(s![c_rows.., ..]).to_owned(),
        );
        new_hot_kvs.push(new_hot_kv);

        let t_ffn = if timing { Some(Instant::now()) } else { None };
        // Try the production-path native-quantised FFN helper first —
        // direct Q4K/Q6K matvec on the vindex's compact gate/up/down
        // bytes. Falls back to WalkFfn (and then dense WeightFfn) when
        // the backend doesn't have native quant support or the layer
        // isn't direct-matvec-eligible.
        let h_out = larql_inference::vindex::ffn_decode_step_native(
            weights,
            index,
            backend,
            &h_post_attn,
            layer,
        )
        .unwrap_or_else(|| {
            let (h, _) = run_ffn(weights, &h_post_attn, layer, &walk_ffn, false);
            h
        });
        if let Some(t) = t_ffn {
            ffn_us += t.elapsed().as_secs_f64() * 1e6;
        }
        h_new = h_out;
    }

    if instrument {
        let total_ms =
            (embed_us + recompute_cold_us + recompute_hot_us + concat_us + attention_us + ffn_us)
                / 1e3;
        eprintln!(
            "[markov-rs/decode] s_hot={s_hot_first_layer} embed={:.2}ms \
             recompute_cold={:.2}ms recompute_hot={:.2}ms concat={:.2}ms \
             attention={:.2}ms ffn={:.2}ms total={:.2}ms \
             (attn_helper hits/miss={attn_helper_hits}/{attn_helper_misses})",
            embed_us / 1e3,
            recompute_cold_us / 1e3,
            recompute_hot_us / 1e3,
            concat_us / 1e3,
            attention_us / 1e3,
            ffn_us / 1e3,
            total_ms,
        );
    }

    if let (Some(prof), Some(t_step)) = (profiler.as_mut(), t_step) {
        prof.embed.total_us += embed_us;
        prof.embed.count += 1;
        prof.recompute_cold.total_us += recompute_cold_us;
        prof.recompute_cold.count += 1;
        prof.recompute_hot.total_us += recompute_hot_us;
        prof.recompute_hot.count += 1;
        prof.attention.total_us += attention_us;
        prof.attention.count += 1;
        prof.ffn.total_us += ffn_us;
        prof.ffn.count += 1;
        prof.decode_total.total_us += t_step.elapsed().as_secs_f64() * 1e6;
        prof.decode_total.count += 1;
    }

    let mut updated_stored: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
    for (stored, new_row) in rs.stored.iter().zip(new_stored.iter()) {
        let s_old = stored.shape()[0];
        let hidden_dim = stored.shape()[1];
        let mut combined = Array2::<f32>::zeros((s_old + 1, hidden_dim));
        combined.slice_mut(s![..s_old, ..]).assign(stored);
        combined.slice_mut(s![s_old.., ..]).assign(new_row);
        updated_stored.push(combined);
    }

    let mut updated_rs = RsStore {
        hot_len: updated_stored.first().map_or(0, |s| s.shape()[0]),
        stored: updated_stored,
        cold_residuals: rs.cold_residuals,
        cold_kv: rs.cold_kv,
        // Commit the new hot K/V slices (one row appended per layer
        // vs the pre-decode cache). Becomes the prior K/V for the
        // next step's fast path. Will be sliced by `clip_layer`
        // below if the window cap is exceeded.
        hot_kv: Some(new_hot_kvs),
        cold_abs_start: rs.cold_abs_start,
        next_position: abs_position + 1,
        max_window: rs.max_window,
    };

    // Pre-clip snapshot of how many hot_kv rows each layer would
    // evict — used below to move evicted K/V directly into cold_kv
    // (vs the pre-W2 behaviour of clearing cold_kv and recomputing
    // from cold_residuals on the next step).
    let pre_clip_evicted_rows: Vec<usize> = if updated_rs.hot_kv.is_some() {
        let window = updated_rs.max_window.unwrap_or(usize::MAX);
        updated_rs
            .stored
            .iter()
            .map(|s| s.shape()[0].saturating_sub(window))
            .collect()
    } else {
        Vec::new()
    };
    let evicted_hot_kv = updated_rs
        .hot_kv
        .as_ref()
        .filter(|_| pre_clip_evicted_rows.iter().any(|&n| n > 0))
        .and_then(|hot_kv| RsStore::snapshot_evicted_hot_kv(hot_kv, &pre_clip_evicted_rows));

    let mut overflow: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
    for layer in 0..num_layers {
        updated_rs.clip_layer(layer, &mut overflow);
    }
    updated_rs.finalise_hot_len_after_clip();
    if overflow.first().map_or(0, |c| c.shape()[0]) > 0 {
        match updated_rs.cold_residuals.as_mut() {
            Some(cold) => {
                for layer in 0..num_layers {
                    let hidden = cold[layer].shape()[1];
                    let c_old = cold[layer].shape()[0];
                    let c_new = overflow[layer].shape()[0];
                    let mut merged = Array2::<f32>::zeros((c_old + c_new, hidden));
                    merged.slice_mut(s![..c_old, ..]).assign(&cold[layer]);
                    merged.slice_mut(s![c_old.., ..]).assign(&overflow[layer]);
                    cold[layer] = merged;
                }
            }
            None => {
                updated_rs.cold_residuals = Some(overflow);
            }
        }

        // Merge evicted hot_kv into cold_kv. The evicted K/V rows
        // are the K/V projection of the evicted residual rows under
        // their original RoPE positions — i.e. they ARE the cold
        // K/V for those positions. Concat in place; no `recompute_kv`
        // call needed.
        if let Some(evicted) = evicted_hot_kv {
            match updated_rs.cold_kv.as_mut() {
                Some(cold_kv) => {
                    for (layer, (k_new, v_new)) in evicted.into_iter().enumerate() {
                        let (k_old, v_old) = &cold_kv[layer];
                        let kv_dim = k_old.shape()[1];
                        let c_old = k_old.shape()[0];
                        let c_new = k_new.shape()[0];
                        let mut k_merged = Array2::<f32>::zeros((c_old + c_new, kv_dim));
                        k_merged.slice_mut(s![..c_old, ..]).assign(k_old);
                        k_merged.slice_mut(s![c_old.., ..]).assign(&k_new);
                        let mut v_merged = Array2::<f32>::zeros((c_old + c_new, kv_dim));
                        v_merged.slice_mut(s![..c_old, ..]).assign(v_old);
                        v_merged.slice_mut(s![c_old.., ..]).assign(&v_new);
                        cold_kv[layer] = (k_merged, v_merged);
                    }
                }
                None => {
                    updated_rs.cold_kv = Some(evicted);
                }
            }
        } else {
            // No cached hot_kv to slice from (via_executor path, or
            // pre-W2 codepaths). Fall back to the legacy invalidation:
            // clear cold_kv so the next step recomputes from
            // cold_residuals. This keeps correctness when W2 caching
            // isn't engaged.
            updated_rs.cold_kv = None;
        }
    }

    Some((last_row(&h_new), updated_rs))
}
