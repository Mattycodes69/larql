//! Multi-modal orchestration for `larql run --image` (Phase 1d).
//!
//! `prepare_multimodal_input` is the single CLI-side composition point
//! for Phase 1: text + image paths → encoder forward → connector project
//! → `EmbeddingPlan` ready to feed into `embed_plan` and on to the
//! engine's `prefill_from_hidden` (the engine seam from ADR-0023).
//!
//! ## Phase 1d scope
//!
//! Gemma 3 specifically. The arguments take *concrete* SigLIP and
//! Gemma3 projector types — not a generic dispatch over
//! `arch.multimodal().vision_encoder()`. That dispatch layer becomes
//! worthwhile in Phase 2 when Granite Vision 4.1 lands a second
//! encoder family (also SigLIP-derivative but distinct) with AnyRes
//! tiling. Per the "no premature crate extraction" rule: one consumer
//! gets concrete types; the second consumer earns the abstraction.
//!
//! ## Plan shape
//!
//! For each image, the per-image fragment is:
//!
//! ```text
//! Tokens([<start_of_image>]),
//! Precomputed { 256 rows of projected vision embeddings, modality: Image },
//! Tokens([<end_of_image>]),
//! ```
//!
//! Then a single trailing `Tokens(text_token_ids)`. Phase 1 is
//! **prefix-only** — image chunks first, text after. Mid-sequence
//! interleaving is Phase 3, and the placeholder-token emission
//! belongs to ChatTemplate by Phase 3 too (see TODO at the splice
//! points below).

use std::path::Path;

use larql_compute::connectors::projector::VisionProjector;
use larql_compute::encoders::vision_tower::VisionEncoder;
use larql_compute::forward::{embed_plan, EmbeddingChunk, EmbeddingPlan, PositionScheme};
use larql_models::connectors::projector::load_projector_from_safetensors;
use larql_models::encoders::vision_tower::{load_vision_tower_from_safetensors, VisionConfig};
use larql_models::{MmConnector, ModalEncoder, ModalInput, Modality, ModelArchitecture};

use crate::anyres_tiler::AnyResTileSpec;
use crate::commands::primary::run_cmd::RunArgs;
use crate::image_input::decode_and_resize_square;

/// Compose a multi-modal input plan from images + text.
///
/// Dispatches on `TokenBudget`:
///   - `Fixed(n)` — Gemma 3 path: single square resize per image,
///     AvgPool connector. One Precomputed chunk per image.
///   - `PerTile { tokens_per_tile }` — Granite Vision path: AnyRes
///     tiling, MLP connector. **N+1 Precomputed chunks per image**
///     (1 base + N detail tiles). This is the Phase 2 splice stress test.
///   - `Dynamic` — Qwen-VL (Phase 4), rejected.
///
/// The `encoder` and `connector` are supplied as trait objects so the
/// caller can dispatch on `mm.vision_encoder()` when loading weights.
#[allow(clippy::too_many_arguments)]
pub fn prepare_multimodal_input(
    lm_arch: &dyn ModelArchitecture,
    encoder: &dyn ModalEncoder,
    connector: &dyn MmConnector,
    encoder_image_size: usize,
    image_paths: &[impl AsRef<Path>],
    text_token_ids: &[u32],
) -> Result<EmbeddingPlan, String> {
    let mm = lm_arch
        .multimodal()
        .ok_or_else(|| "model architecture does not declare multi-modal support".to_string())?;
    let placeholder = mm
        .image_placeholder()
        .ok_or_else(|| "model does not declare an image placeholder protocol".to_string())?;

    let mut chunks: Vec<EmbeddingChunk> = Vec::with_capacity(image_paths.len() * 3 + 1);

    match mm.image_token_budget() {
        larql_models::TokenBudget::Fixed(_) => {
            for path in image_paths {
                let path = path.as_ref();
                let rgb = decode_and_resize_square(path, encoder_image_size)?;

                let encoder_out = encoder.encode(ModalInput::Image {
                    rgb: &rgb,
                    width: encoder_image_size,
                    height: encoder_image_size,
                })?;
                let projected = connector.project(&encoder_out);

                if let Some(start) = placeholder.start {
                    chunks.push(EmbeddingChunk::Tokens(vec![start]));
                }
                chunks.push(EmbeddingChunk::Precomputed {
                    rows: projected,
                    modality: Modality::Image,
                });
                if let Some(end) = placeholder.end {
                    chunks.push(EmbeddingChunk::Tokens(vec![end]));
                }
            }
        }
        larql_models::TokenBudget::PerTile { .. } => {
            let tile_counts = mm.valid_tile_counts();
            if tile_counts.is_empty() {
                return Err("PerTile budget but valid_tile_counts is empty".to_string());
            }
            let spec = AnyResTileSpec {
                tile_size: encoder_image_size,
                valid_tile_counts: tile_counts.to_vec(),
            };
            for path in image_paths {
                let path = path.as_ref();
                let img = image::open(path)
                    .map_err(|e| format!("failed to open {}: {e}", path.display()))?;
                let (w, h) = (img.width() as usize, img.height() as usize);
                let rgb = img.to_rgb8().into_raw();
                let tiled = spec.tile(&rgb, w, h);

                let all_tiles = std::iter::once(&tiled.base_tile).chain(tiled.detail_tiles.iter());
                for tile_rgb in all_tiles {
                    let encoder_out = encoder.encode(ModalInput::Image {
                        rgb: tile_rgb,
                        width: encoder_image_size,
                        height: encoder_image_size,
                    })?;
                    let projected = connector.project(&encoder_out);

                    chunks.push(EmbeddingChunk::Tokens(vec![placeholder.fill]));
                    chunks.push(EmbeddingChunk::Precomputed {
                        rows: projected,
                        modality: Modality::Image,
                    });
                }
            }
        }
        larql_models::TokenBudget::Dynamic => {
            return Err("Dynamic token budget is Phase 4 (Qwen-VL); not yet supported".to_string());
        }
    }

    chunks.push(EmbeddingChunk::Tokens(text_token_ids.to_vec()));

    Ok(EmbeddingPlan {
        chunks,
        positions: PositionScheme::Sequential,
    })
}

// ─── Phase 1d.3c: capability check ─────────────────────────────────────
//
// Extracted to a standalone function so the capability semantics can
// be unit-tested without setting up a full LM runtime. The CLI's
// `run_with_images` must call this BEFORE doing any encoder work —
// see ADR-0023 §"Default-false debt" for the rationale (vision
// encoding can take minutes; failing fast on engine incompatibility
// is the point of the capability flag).

/// Verify the resolved engine supports multi-modal input. Returns
/// `Ok(())` if it does, or a `String` error naming both the
/// incapable engine and the recommended fix.
///
/// MUST be called before any vision encoding work. Currently
/// `StandardEngine` is the only MM-capable engine; other engines
/// inherit the default-false convention from `KvEngine::supports_multimodal`.
pub fn ensure_engine_supports_multimodal(
    engine: &larql_inference::kv_engine::AnyEngine,
) -> Result<(), String> {
    if engine.supports_multimodal() {
        return Ok(());
    }
    Err(format!(
        "engine {:?} does not support multi-modal input; use `--engine standard` \
         (the only MM-capable engine in Phase 1; other engines will gain support \
         as their use cases land — see ADR-0023)",
        engine.name()
    ))
}

/// Entry point for `larql run --image foo.jpg "describe"`. Composes
/// the Phase 1b/1c/1d.2/1d.3a/1d.3b pieces and emits a generated
/// continuation to stdout.
///
/// Pipeline (in order — order matters; see ADR-0023):
///   1. Resolve & build the engine.
///   2. **Capability check.** Fail fast if the engine doesn't support
///      MM, BEFORE running the encoder (which takes minutes).
///   3. Load LM weights + tokenizer (from the vindex path).
///   4. Load SigLIP + projector weights (from `--mm-weights` dir).
///   5. Parse SigLIP config from `mm_weights/config.json`.
///   6. Tokenize the prompt.
///   7. `prepare_multimodal_input` → `EmbeddingPlan`.
///   8. `embed_plan` → initial hidden state.
///   9. `generate_with_engine_from_hidden` → emit tokens.
pub fn run_with_images(
    vindex_path: &Path,
    prompt: &str,
    args: &RunArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    let mm_weights_dir = args.mm_weights.as_deref().ok_or(
        "--image requires --mm-weights <DIR> pointing at the original safetensors snapshot \
         (vindex carries LM weights only; SigLIP + projector live alongside config.json)",
    )?;

    // ── 1. Engine ──
    use larql_kv::EngineKind;
    let engine_spec = args
        .engine
        .clone()
        .or_else(|| std::env::var("LARQL_KV_ENGINE").ok());
    let kind = match engine_spec {
        Some(spec) => {
            EngineKind::from_name(&spec).unwrap_or(EngineKind::Standard { window_size: None })
        }
        None => EngineKind::Standard { window_size: None },
    };
    let backend = larql_inference::default_engine_backend();
    let mut engine = kind.build(backend);

    // ── 2. Capability check (BEFORE the encoder runs) ──
    ensure_engine_supports_multimodal(&engine)?;

    // ── 3. LM weights + tokenizer ──
    // Dispatch on the vindex's quant format. Phase 1d.4 supports both
    // f32 and Q4K. For Q4K we use the same strategy as
    // `run_cmd::experts::load_runtime`: load kquant weights, load the
    // VectorIndex with kquant attn + interleaved tensors, then
    // dequantise attention into `weights.tensors` so the engine seam
    // (which is attention-pass + FFN-via-supplied-backend) sees f32 on
    // the attention side and a Q4K-aware `WalkFfn` on the FFN side.
    // The engine itself (StandardEngine::prefill_from_hidden) doesn't
    // need to know about Q4K — Phase 1d.3a's `index: None` hardcode
    // works because attention's already-dequantised by this point.
    let mut cb = larql_vindex::SilentLoadCallbacks;
    let cfg = larql_vindex::load_vindex_config(vindex_path)?;
    let is_quant = !matches!(cfg.quant, larql_vindex::QuantFormat::None);
    let mut weights = if is_quant {
        larql_vindex::load_model_weights_kquant(vindex_path, &mut cb)?
    } else {
        larql_vindex::load_model_weights_with_opts(
            vindex_path,
            &mut cb,
            larql_vindex::LoadWeightsOptions::default(),
        )?
    };
    let q_index = if is_quant {
        let mut idx = larql_vindex::VectorIndex::load_vindex(vindex_path, &mut cb)?;
        idx.load_attn_kquant(vindex_path)?;
        idx.load_interleaved_kquant(vindex_path)?;
        let _ = idx.load_lm_head_kquant(vindex_path);
        // Materialise f32 attention tensors into `weights.tensors` so the
        // engine's attention dispatch reads f32 even though the on-disk
        // tensors are Q4K.
        larql_inference::vindex::ensure_attn_tensors_dequantised_resident(&mut weights, &idx);
        Some(idx)
    } else {
        None
    };
    let tokenizer = load_vindex_tokenizer(vindex_path)?;

    // ── 4. + 5. SigLIP config + weights, projector weights ──
    let siglip_config = load_siglip_config_from_dir(mm_weights_dir)?;
    if args.verbose {
        eprintln!(
            "loading SigLIP encoder from {} ({}×{} image, {} layers, hidden={})",
            mm_weights_dir.display(),
            siglip_config.image_size,
            siglip_config.image_size,
            siglip_config.num_hidden_layers,
            siglip_config.hidden_size,
        );
    }
    let siglip = load_vision_tower_from_safetensors(mm_weights_dir, siglip_config.clone())?;
    let projector = load_projector_from_safetensors(mm_weights_dir)?;

    // ── 6. Tokenize the prompt ──
    // Phase 1d cheap path: bare tokenization, no chat template. Phase 3
    // (Gemma 3 native interleaving) takes ownership of MM-aware
    // placeholder emission via ChatTemplate. See TODO in
    // `prepare_multimodal_input`.
    let encoding = tokenizer
        .encode(prompt, false)
        .map_err(|e| format!("tokenize prompt: {e}"))?;
    let text_token_ids: Vec<u32> = encoding.get_ids().to_vec();

    // ── 7. + 8. Build plan + embed ──
    let encoder = VisionEncoder::new(&siglip);
    let mm_tokens_per_image = match weights.arch.multimodal().unwrap().image_token_budget() {
        larql_models::TokenBudget::Fixed(n) => n,
        _ => 0,
    };
    let connector = VisionProjector::new(&projector, &siglip_config, mm_tokens_per_image.max(1))?;
    let plan = prepare_multimodal_input(
        &*weights.arch,
        &encoder,
        &connector,
        siglip_config.image_size,
        &args.image,
        &text_token_ids,
    )?;
    if args.verbose {
        eprintln!(
            "embedding plan: {} chunks, {} total rows ({} text tokens, {} images)",
            plan.chunks.len(),
            plan.total_rows(),
            text_token_ids.len(),
            args.image.len(),
        );
    }
    let initial_hidden = embed_plan(&weights, &plan);

    // ── 9. Generate ──
    // FFN backend choice depends on quant format: WalkFfn reads Q4K
    // tensors from the index, WeightFfn reads f32 from weights.tensors.
    // Both implement FfnBackend so the engine seam stays uniform.
    use std::io::Write;
    let mut stdout = std::io::stdout();
    use larql_inference::ffn::{FfnBackend, WeightFfn};
    let walk_ffn_storage;
    let ffn: &dyn FfnBackend = if let Some(ref idx) = q_index {
        walk_ffn_storage = larql_inference::vindex::WalkFfn::new_unlimited(&weights, idx);
        &walk_ffn_storage
    } else {
        let _ = q_index; // suppress unused warning on f32 path
        &WeightFfn { weights: &weights }
    };
    let generated = larql_kv::generation::generate_with_engine_from_hidden(
        &mut engine,
        &weights,
        &tokenizer,
        ffn,
        &initial_hidden,
        args.max_tokens,
        |_id, tok| {
            print!("{tok}");
            let _ = stdout.flush();
        },
    );
    println!();
    if args.verbose {
        eprintln!(
            "  Generated {} tokens (engine={}, mm-capable={})",
            generated.len(),
            engine.name(),
            engine.supports_multimodal(),
        );
    }
    Ok(())
}

/// Parse SigLIP config from `<dir>/config.json`'s `vision_config` field.
/// Matches the HF Gemma 3 layout.
fn load_siglip_config_from_dir(dir: &Path) -> Result<VisionConfig, Box<dyn std::error::Error>> {
    let config_path = dir.join("config.json");
    let raw = std::fs::read_to_string(&config_path)
        .map_err(|e| format!("read {}: {e}", config_path.display()))?;
    let value: serde_json::Value = serde_json::from_str(&raw)?;
    let vision_config = value
        .get("vision_config")
        .ok_or_else(|| format!("{} has no vision_config field", config_path.display()))?;
    Ok(VisionConfig::from_json(vision_config)?)
}

/// Tokenizer loader — mirrors the helper used elsewhere in run_cmd /
/// walk_cmd. Vindex directories ship `tokenizer.json` alongside the
/// FFN payload.
fn load_vindex_tokenizer(
    vindex_path: &Path,
) -> Result<larql_inference::tokenizers::Tokenizer, Box<dyn std::error::Error>> {
    let tok_path = vindex_path.join("tokenizer.json");
    larql_inference::tokenizers::Tokenizer::from_file(&tok_path)
        .map_err(|e| format!("load tokenizer from {}: {e}", tok_path.display()).into())
}

#[cfg(test)]
mod tests;
