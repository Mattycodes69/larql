//! Full hybrid-MoE layer: CPU vs Metal per-layer residual diff.

#[allow(unused_imports)]
use super::*;

// ── layer: full hybrid-MoE layer CPU vs Metal residual diff ──────────────────
//
// Runs CPU `predict_kquant_hidden` and Metal `generate` on the same prompt with
// their respective dump hooks enabled, then compares per-layer residuals.
//
// CPU dumps:   LARQL_CPU_DUMP_LAYERS → cpu_layer_{LL}.f32 (last-position row)
//              LARQL_CPU_STAGE_DUMP  → cpu_L0_<stage>.f32
// Metal dump:  LARQL_DUMP_RESIDUALS  → binary (LARQL_RES_V2 header, then per-
//              layer records: u32 layer_idx, u32 hidden, f32[hidden] layer_in,
//              f32[hidden] h_post_attn, f32[hidden] layer_out)
//
// The comparison is decode-step vs prefill-last-token, so the two are in
// slightly different compute contexts (Metal uses KV cache; CPU re-processes
// the full sequence). This is sufficient to locate the first diverging layer
// but not to compute precise numeric agreement.

#[cfg(all(feature = "gpu", target_os = "macos"))]
pub(super) fn run_layer_diff(
    path: &std::path::Path,
    config: &larql_vindex::VindexConfig,
    args: &ParityArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    use larql_inference::layer_graph::{generate::generate, CachedLayerGraph};
    use larql_inference::vindex::predict_kquant_hidden;

    let num_layers = config.num_layers;
    let hidden = config.hidden_size;

    let prompt = args.prompt.as_deref().unwrap_or("The capital of France is");

    println!("Prompt:    {prompt:?}");
    println!("Backends:  metal (reference) → cpu");
    println!();

    // ── Set up temp dirs for dump files ─────────────────────────────────────
    let base = std::env::temp_dir().join(format!("larql_parity_{}", std::process::id()));
    let cpu_path_buf = base.join("cpu");
    let metal_path_buf = base.join("metal_residuals.bin");
    let metal_dense_dir = base.join("metal_dense");
    std::fs::create_dir_all(&cpu_path_buf)?;
    let cpu_path = cpu_path_buf.as_path();
    let metal_path = metal_path_buf.as_path();
    struct Cleanup(std::path::PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let _cleanup = Cleanup(base);

    // ── Load vindex (shared mmap; two weight copies for the two runs) ────────
    let mut cb = larql_vindex::SilentLoadCallbacks;
    let mut index = larql_vindex::VectorIndex::load_vindex(path, &mut cb)?;
    index.load_attn_kquant(path)?;
    index.load_interleaved_kquant(path)?;
    let _ = index.load_lm_head_kquant(path);
    let tokenizer = larql_vindex::load_vindex_tokenizer(path)?;
    let mut w_metal = larql_vindex::load_model_weights_kquant(path, &mut cb)?;
    let w_cpu = larql_vindex::load_model_weights_kquant(path, &mut cb)?;

    let wrapped = larql_inference::wrap_chat_prompt(path, Some(config.model.as_str()), prompt);
    let token_ids = larql_inference::encode_prompt(&tokenizer, &*w_metal.arch, &wrapped.prompt)?;
    println!("  seq_len: {} tokens post-template", token_ids.len());
    println!();

    // The MoE decode path writes a single LARQL_DUMP_RESIDUALS binary
    // covering every layer; the dense Metal decode path doesn't fire that
    // hook (it only runs in the MoE branch of decode_token_with_moe_split_fn).
    // For dense models we use LARQL_METAL_DUMP_LAYERS, which fires inside
    // prefill_kquant and writes one file per layer (metal_layer_NN_h_out.f32 +
    // metal_layer_NN_h_post_attn.f32). This aligns with the CPU dumps,
    // which are also captured during prefill.
    let is_moe = w_metal.arch.is_hybrid_moe();
    if !is_moe {
        std::fs::create_dir_all(&metal_dense_dir)?;
    }

    // ── Metal run (reference — produces correct output) ──────────────────────
    if is_moe {
        std::env::set_var("LARQL_DUMP_RESIDUALS", metal_path);
    } else {
        std::env::set_var("LARQL_METAL_DUMP_LAYERS", &metal_dense_dir);
    }
    println!("Running Metal…");
    let metal_result = {
        let backend = larql_compute_metal::MetalBackend::new()
            .ok_or("Metal backend unavailable — build with `--features gpu` on M-series Mac")?;
        let cache = CachedLayerGraph::from_residuals(Vec::new());
        generate(
            &mut w_metal,
            &tokenizer,
            &token_ids,
            1,
            &index,
            &backend,
            &cache,
            0..num_layers,
        )
    };
    std::env::remove_var("LARQL_DUMP_RESIDUALS");
    std::env::remove_var("LARQL_METAL_DUMP_LAYERS");
    println!("  Metal output: {:?}", metal_result.text().trim());

    // ── CPU run ──────────────────────────────────────────────────────────────
    std::env::set_var("LARQL_CPU_DUMP_LAYERS", cpu_path);
    std::env::set_var("LARQL_CPU_STAGE_DUMP", cpu_path);
    println!("Running CPU…");
    predict_kquant_hidden(&w_cpu, &token_ids, &index, None);
    std::env::remove_var("LARQL_CPU_DUMP_LAYERS");
    std::env::remove_var("LARQL_CPU_STAGE_DUMP");

    // ── Load per-layer Metal output ──────────────────────────────────────────
    // MoE: parse binary residual dump (richer — includes h_post_attn).
    // Dense: read decode_layer_NN.f32 written by LARQL_DECODE_DUMP_LAYERS.
    let metal_layers: std::collections::BTreeMap<usize, ResidualRecord> = if is_moe {
        let metal_bytes = std::fs::read(metal_path)?;
        let parsed = parse_residual_dump(&metal_bytes);
        if parsed.is_empty() {
            return Err(
                "Metal residual dump is empty — LARQL_DUMP_RESIDUALS may not have fired".into(),
            );
        }
        parsed.into_iter().collect()
    } else {
        // Prefill dumps: metal_layer_NN_h_out.f32 (post-FFN residual) and
        // metal_layer_NN_h_post_attn.f32 (post-attention residual).
        // Both have shape [seq_len * hidden]; we take the last position.
        let last_pos_slice = |v: Vec<f32>| -> Vec<f32> {
            let n = v.len() / hidden;
            if n == 0 {
                v
            } else {
                v[(n - 1) * hidden..].to_vec()
            }
        };
        let mut out = std::collections::BTreeMap::new();
        for l in 0..num_layers {
            let h_out_path = metal_dense_dir.join(format!("metal_layer_{l:02}_h_out.f32"));
            let h_pa_path = metal_dense_dir.join(format!("metal_layer_{l:02}_h_post_attn.f32"));
            let layer_out = match read_parity_f32(&h_out_path) {
                Some(v) => last_pos_slice(v),
                None => continue,
            };
            let h_post_attn = read_parity_f32(&h_pa_path)
                .map(last_pos_slice)
                .unwrap_or_default();
            out.insert(
                l,
                ResidualRecord {
                    h_post_attn,
                    layer_out,
                },
            );
        }
        if out.is_empty() {
            return Err(
                "Metal dense dump is empty — LARQL_METAL_DUMP_LAYERS may not have fired".into(),
            );
        }
        out
    };

    // ── Compare per layer ────────────────────────────────────────────────────
    println!();
    println!("━━━ Layer-by-layer residual diff (Metal = reference) ━━━━━━━━━━");
    println!(
        "  {:>3}  {:>10}  {:>10}  {:>10}  {:>12}  note",
        "L", "cos(h_pa)", "cos(h_out)", "‖cpu‖", "‖metal‖"
    );
    println!("  {}", "─".repeat(72));

    const DRIFT: f32 = 0.9999;
    let mut first_bad: Option<usize> = None;

    for l in 0..num_layers {
        let cpu_out_path = cpu_path.join(format!("cpu_layer_{l:02}.f32"));
        let cpu_pa_path = cpu_path.join(format!("cpu_layer_{l:02}_h_post_attn.f32"));

        let cpu_out = match read_parity_f32(&cpu_out_path) {
            Some(v) => v,
            None => {
                println!("  L{l:02}  <cpu dump missing>");
                continue;
            }
        };
        let metal_rec = match metal_layers.get(&l) {
            Some(r) => r,
            None => {
                println!("  L{l:02}  <metal dump missing>");
                continue;
            }
        };

        // CPU dump has (seq_len × hidden) elements; take the last position.
        let seq_positions = cpu_out.len() / hidden;
        let cpu_last = if seq_positions > 0 {
            cpu_out[(seq_positions - 1) * hidden..].to_vec()
        } else {
            cpu_out.clone()
        };

        let cos_out = naive_cos_sim(&cpu_last, &metal_rec.layer_out);
        let norm_cpu = naive_rms_mag(&cpu_last);
        let norm_mtl = naive_rms_mag(&metal_rec.layer_out);

        // Dense path doesn't capture h_post_attn separately, so cos(h_pa)
        // is only computed when we have it (MoE).
        let cos_pa = if metal_rec.h_post_attn.is_empty() {
            None
        } else {
            read_parity_f32(&cpu_pa_path).map(|v| {
                let n = v.len() / hidden;
                let last = if n > 0 {
                    v[(n - 1) * hidden..].to_vec()
                } else {
                    v
                };
                naive_cos_sim(&last, &metal_rec.h_post_attn)
            })
        };

        if cos_out < DRIFT && first_bad.is_none() {
            first_bad = Some(l);
        }
        let flag = if cos_out < DRIFT { " ←" } else { "" };
        let note = match cos_pa {
            Some(ca) if ca < DRIFT && cos_out < DRIFT => "attn+ffn",
            Some(ca) if ca < DRIFT => "attn",
            Some(_) if cos_out < DRIFT => "ffn/moe",
            Some(_) => "clean",
            None => "?",
        };
        let hpa_s = cos_pa
            .map(|c| format!("{c:>10.6}"))
            .unwrap_or_else(|| "         -".into());
        println!(
            "  L{l:02}  {hpa_s}  {cos_out:>10.6}  {norm_cpu:>10.4}  {norm_mtl:>12.4}  {note}{flag}"
        );
    }

    println!();
    match first_bad {
        Some(l) => {
            println!("First divergence at L{l} (cos < {DRIFT}).");
            let note = if l == 0 {
                "L0 drift — culprit is embedding, pre-norm, attention, or MoE combine."
            } else {
                "Earlier layers match; drift introduced at this layer."
            };
            println!("{note}");
        }
        None => {
            println!("All layers match within cos ≥ {DRIFT}.");
            println!("Note: Metal decode vs CPU prefill — slight positional mismatch expected.");
        }
    }

    Ok(())
}

/// Per-layer record from `LARQL_DUMP_RESIDUALS` binary.
pub(super) struct ResidualRecord {
    pub(super) h_post_attn: Vec<f32>,
    pub(super) layer_out: Vec<f32>,
}

/// Parse `LARQL_DUMP_RESIDUALS` binary (written by `moe_combine.rs / diag.rs`).
/// Returns a map from layer_idx → record. Skips the 16-byte magic header.
pub(super) fn parse_residual_dump(
    bytes: &[u8],
) -> std::collections::HashMap<usize, ResidualRecord> {
    let mut map = std::collections::HashMap::new();
    if bytes.len() < 16 {
        return map;
    }
    let mut pos = 16usize; // skip magic
    while pos + 8 <= bytes.len() {
        let layer_idx = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
        let hidden = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
        pos += 8;
        let n_bytes = hidden * 4;
        if pos + n_bytes * 3 > bytes.len() {
            break;
        }
        let layer_in: Vec<f32> = bytes[pos..pos + n_bytes]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        pos += n_bytes;
        let h_post_attn: Vec<f32> = bytes[pos..pos + n_bytes]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        pos += n_bytes;
        let layer_out: Vec<f32> = bytes[pos..pos + n_bytes]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        pos += n_bytes;
        let _ = layer_in; // used for format validation only
        map.insert(
            layer_idx,
            ResidualRecord {
                h_post_attn,
                layer_out,
            },
        );
    }
    map
}

pub(super) fn read_parity_f32(path: &std::path::Path) -> Option<Vec<f32>> {
    let bytes = std::fs::read(path).ok()?;
    if bytes.len() % 4 != 0 {
        return None;
    }
    Some(
        bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect(),
    )
}

pub(super) fn naive_cos_sim(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let dot: f32 = a[..n].iter().zip(&b[..n]).map(|(x, y)| x * y).sum();
    let na: f32 = a[..n].iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b[..n].iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb + 1e-10)
}

pub(super) fn naive_rms_mag(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt()
}
