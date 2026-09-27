//! `rope_scaling` / `rope_parameters` → [`RopeScaling`].

use crate::config::RopeScaling;

/// Parse one RoPE-scaling block. Four shapes appear in the wild:
///
/// 1. Flat with `factor` (Llama 2-style linear, simple `rope_type=linear`).
/// 2. `rope_type=llama3` with the four wavelength-band fields below.
/// 3. Gemma 3 structured per-layer-type,
///    `{full_attention: {rope_type: linear, factor: N, ...}, sliding_attention: {...}}`.
///    Only the `full_attention` slot carries a non-default scaling (sliding
///    layers use plain RoPE), so its `rope_type` and `factor` are lifted and
///    `gemma3_global_only` is set.
/// 4. Missing entirely (older Llama, Mistral) → `None`.
///
/// And two *homes* for any of those shapes: the legacy `rope_scaling`
/// key, and transformers-5.x's `rope_parameters`, which carries theta AND
/// scaling in one block (`{rope_theta, rope_type: "yarn", factor, …}`).
/// The parser's theta read prefers `rope_parameters`; the scaling
/// read must too, or a 5.x checkpoint's YaRN block is dropped at parse
/// while its theta is honoured — the §4.7.8 shape again, caught by the
/// VINDEX3 carriage test on a Glimmer-shaped fixture. A `rope_parameters`
/// block that declares no scaling (`rope_type: "default"`, no `factor`)
/// parses to `None` and the legacy key is consulted.
pub(super) fn parse_rope_scaling(rs: &serde_json::Value) -> Option<RopeScaling> {
    // Gemma 3 per-layer-type form.
    if let Some(full) = rs.get("full_attention") {
        let scaling_type = full
            .get("rope_type")
            .or_else(|| full.get("type"))
            .and_then(|v| v.as_str())?
            .to_string();
        let factor = full.get("factor")?.as_f64()?;
        return Some(RopeScaling {
            scaling_type,
            factor,
            llama3_low_freq_factor: None,
            llama3_high_freq_factor: None,
            llama3_original_max_position_embeddings: None,
            yarn_beta_fast: None,
            yarn_beta_slow: None,
            yarn_truncate: None,
            yarn_mscale: None,
            yarn_mscale_all_dim: None,
            gemma3_global_only: true,
        });
    }
    // Flat form (Llama, Mistral, Gemma 1/2, GPT-OSS, DeepSeek, etc.).
    let scaling_type = rs
        .get("type")
        .or_else(|| rs.get("rope_type"))
        .and_then(|v| v.as_str())?
        .to_string();
    let factor = rs.get("factor")?.as_f64()?;
    let llama3_low = rs.get("low_freq_factor").and_then(|v| v.as_f64());
    let llama3_high = rs.get("high_freq_factor").and_then(|v| v.as_f64());
    let llama3_old_ctx = rs
        .get("original_max_position_embeddings")
        .and_then(|v| v.as_f64());
    // YaRN band bounds. Absent means "use the paper's defaults" (32 / 1),
    // which is what `_compute_yarn_parameters` falls back to — so `None`
    // here is a real value downstream, not a missing one. `truncate`
    // decides whether the correction range is rounded outward to integer
    // dimensions; HF defaults it to true and GPT-OSS ships false.
    let yarn_beta_fast = rs.get("beta_fast").and_then(|v| v.as_f64());
    let yarn_beta_slow = rs.get("beta_slow").and_then(|v| v.as_f64());
    let yarn_truncate = rs.get("truncate").and_then(|v| v.as_bool());
    // DeepSeek's two extra amplitude knobs. They must be parsed even
    // though no R1 checkpoint uses them: when *both* are present HF
    // computes the attention factor as a *ratio* that typically collapses
    // to 1.0, where the single-argument form would give 1.35. Reading
    // yarn without reading these would newly apply a wrong amplitude to
    // every DeepSeek layer.
    let yarn_mscale = rs.get("mscale").and_then(|v| v.as_f64());
    let yarn_mscale_all_dim = rs.get("mscale_all_dim").and_then(|v| v.as_f64());
    Some(RopeScaling {
        scaling_type,
        factor,
        llama3_low_freq_factor: llama3_low,
        llama3_high_freq_factor: llama3_high,
        llama3_original_max_position_embeddings: llama3_old_ctx,
        yarn_beta_fast,
        yarn_beta_slow,
        yarn_truncate,
        yarn_mscale,
        yarn_mscale_all_dim,
        gemma3_global_only: false,
    })
}
