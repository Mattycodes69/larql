use super::*;
use image::{ImageBuffer, Rgb};
use larql_models::connectors::projector::ProjectorWeights;
use larql_models::encoders::vision_tower::{
    LayerNormWeights, ProjWithBias, VisionLayerWeights, VisionWeights,
};
use larql_models::{MultiModalProtocol, PlaceholderProtocol, PrecomputedScaling, TokenBudget};
use ndarray::{Array2, Array4};

/// Standalone test arch — minimal `ModelArchitecture` that returns
/// a known multi-modal protocol. Borrows the test-fixtures-built
/// weights' arch config (so we don't have to manually construct a
/// 30-field `ModelConfig` here just to satisfy the trait).
struct TestMmArch {
    config: larql_models::ModelConfig,
    mm: TestMm,
}
struct TestMm;
impl MultiModalProtocol for TestMm {
    fn vision_encoder(&self) -> Option<&str> {
        Some("siglip")
    }
    fn image_placeholder(&self) -> Option<PlaceholderProtocol> {
        Some(PlaceholderProtocol {
            start: Some(900),
            fill: 901,
            end: Some(902),
        })
    }
    fn image_token_budget(&self) -> TokenBudget {
        TokenBudget::Fixed(4) // 2x2 spatial pool over 4x4 patches
    }
    fn precomputed_scaling(&self) -> PrecomputedScaling {
        PrecomputedScaling::None
    }
}
use larql_models::architecture_prelude::*;

impl ArchitectureCore for TestMmArch {
    fn family(&self) -> &str {
        "test-mm"
    }

    fn config(&self) -> &larql_models::ModelConfig {
        &self.config
    }

    fn multimodal(&self) -> Option<&dyn MultiModalProtocol> {
        Some(&self.mm)
    }
}

impl TensorKeys for TestMmArch {}
impl Norms for TestMmArch {}
impl Position for TestMmArch {}
impl Attention for TestMmArch {}
impl FeedForward for TestMmArch {}
impl LatentAttention for TestMmArch {}
impl Embeddings for TestMmArch {}
impl ModelArchitecture for TestMmArch {}

fn synth_arch() -> TestMmArch {
    // Borrow the config from larql_models's test-utils-built weights.
    // Cheaper than enumerating every ModelConfig field by hand, and
    // any future field additions don't break this test.
    let w = larql_models::test_fixtures::make_test_weights();
    TestMmArch {
        config: w.arch.config().clone(),
        mm: TestMm,
    }
}

fn synth_siglip_config_4x4_patch2() -> VisionConfig {
    // 4×4 image, 2×2 patches → 2×2 = 4 patches per image. With
    // Fixed(4) budget, AvgPool kernel = 2/2 = 1 (identity pool).
    VisionConfig {
        hidden_size: 8,
        intermediate_size: 16,
        num_attention_heads: 2,
        num_hidden_layers: 1,
        patch_size: 2,
        image_size: 4,
        num_channels: 3,
        layer_norm_eps: 1e-6,
        hidden_act: "gelu_pytorch_tanh".to_string(),
        norm_type: "layer_norm".to_string(),
    }
}

fn asymmetric(i: usize, j: usize) -> f32 {
    let h = (i.wrapping_mul(2654435761) ^ j.wrapping_mul(40503)).wrapping_mul(2654435761);
    ((h & 0xff) as i32 - 128) as f32 / 800.0
}

fn synth_siglip(config: &VisionConfig) -> VisionWeights {
    let hidden = config.hidden_size;
    let inter = config.intermediate_size;
    let num_patches = config.num_patches();
    let channels = config.num_channels;
    let p = config.patch_size;
    let proj = |out: usize, in_: usize| ProjWithBias {
        weight: Array2::<f32>::from_shape_fn((out, in_), |(i, j)| asymmetric(i, j)),
        bias: vec![0.0; out],
    };
    let lnorm = |n: usize| LayerNormWeights {
        weight: vec![1.0; n],
        bias: vec![0.0; n],
    };
    VisionWeights {
        config: config.clone(),
        patch_embed: Array4::<f32>::from_shape_fn((hidden, channels, p, p), |(h, c, dy, dx)| {
            asymmetric(h * 13, c * 7 + dy * 3 + dx)
        }),
        patch_embed_bias: vec![0.0; hidden],
        position_embed: Array2::<f32>::zeros((num_patches, hidden)),
        layers: vec![VisionLayerWeights {
            layer_norm1: lnorm(hidden),
            q_proj: proj(hidden, hidden),
            k_proj: proj(hidden, hidden),
            v_proj: proj(hidden, hidden),
            out_proj: proj(hidden, hidden),
            layer_norm2: lnorm(hidden),
            fc1: proj(inter, hidden),
            fc2: proj(hidden, inter),
        }],
        post_layernorm: lnorm(hidden),
    }
}

fn synth_projector(vision_hidden: usize, text_hidden: usize) -> ProjectorWeights {
    ProjectorWeights {
        input_projection: Array2::<f32>::from_shape_fn((vision_hidden, text_hidden), |(i, j)| {
            asymmetric(i, j)
        }),
        soft_emb_norm: (0..vision_hidden).map(|i| asymmetric(i, 0)).collect(),
    }
}

fn write_synth_png(dir: &Path, name: &str, side: u32) -> std::path::PathBuf {
    let path = dir.join(name);
    let mut img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::new(side, side);
    // Salt the pixel pattern with the filename's bytes so two PNGs
    // written with different names have genuinely different content.
    // Without this, the two-image test silently exercises identical
    // inputs and the "different projected rows" assertion fails
    // for the wrong reason.
    let salt: u32 = name.bytes().map(|b| b as u32).sum::<u32>().max(1);
    for y in 0..side {
        for x in 0..side {
            img.put_pixel(
                x,
                y,
                Rgb([
                    ((x.wrapping_mul(11).wrapping_add(salt)) & 0xff) as u8,
                    ((y.wrapping_mul(17).wrapping_add(salt.wrapping_mul(3))) & 0xff) as u8,
                    ((x.wrapping_add(y)
                        .wrapping_mul(23)
                        .wrapping_add(salt.wrapping_mul(7)))
                        & 0xff) as u8,
                ]),
            );
        }
    }
    img.save(&path).unwrap();
    path
}

#[test]
fn single_image_plan_has_expected_chunk_shape() {
    let arch = synth_arch();
    let siglip_cfg = synth_siglip_config_4x4_patch2();
    let siglip = synth_siglip(&siglip_cfg);
    let lm_hidden: usize = 12;
    let projector = synth_projector(siglip_cfg.hidden_size, lm_hidden);

    let tmp = tempfile::tempdir().unwrap();
    let img = write_synth_png(tmp.path(), "one.png", 4);
    let text = [1u32, 2, 3, 4, 5];

    let encoder = VisionEncoder::new(&siglip);
    let connector = VisionProjector::new(&projector, &siglip_cfg, 4).unwrap();
    let plan = prepare_multimodal_input(
        &arch,
        &encoder,
        &connector,
        siglip_cfg.image_size,
        std::slice::from_ref(&img),
        &text,
    )
    .expect("prepare");

    // For 1 image with TestMm placeholders (start + end both Some):
    //   [Tokens(start), Precomputed, Tokens(end), Tokens(text)]
    // = 4 chunks total.
    assert_eq!(plan.chunks.len(), 4);
    assert!(matches!(plan.positions, PositionScheme::Sequential));
    assert!(
        !plan.is_text_only(),
        "MM plan must force the mixed embed_plan path"
    );

    // Chunk 0: start_of_image marker
    match &plan.chunks[0] {
        EmbeddingChunk::Tokens(toks) => assert_eq!(toks, &vec![900]),
        other => panic!("chunk 0 should be start marker tokens, got {other:?}"),
    }
    // Chunk 1: 4 rows of projected vision embeddings at lm_hidden
    match &plan.chunks[1] {
        EmbeddingChunk::Precomputed { rows, modality } => {
            assert_eq!(rows.shape(), &[4, lm_hidden]);
            assert!(rows.iter().all(|v| v.is_finite()));
            assert_eq!(*modality, Modality::Image);
        }
        other => panic!("chunk 1 should be precomputed vision rows, got {other:?}"),
    }
    // Chunk 2: end_of_image marker
    match &plan.chunks[2] {
        EmbeddingChunk::Tokens(toks) => assert_eq!(toks, &vec![902]),
        _ => panic!("chunk 2 should be end marker tokens"),
    }
    // Chunk 3: text
    match &plan.chunks[3] {
        EmbeddingChunk::Tokens(toks) => assert_eq!(toks, &text.to_vec()),
        _ => panic!("chunk 3 should be text tokens"),
    }
}

#[test]
fn two_image_plan_chunks_are_per_image_then_text() {
    // Two images → 2 × 3 (start/precomputed/end) + 1 text = 7 chunks.
    let arch = synth_arch();
    let siglip_cfg = synth_siglip_config_4x4_patch2();
    let siglip = synth_siglip(&siglip_cfg);
    let projector = synth_projector(siglip_cfg.hidden_size, 12);

    let tmp = tempfile::tempdir().unwrap();
    let imgs = vec![
        write_synth_png(tmp.path(), "a.png", 4),
        write_synth_png(tmp.path(), "b.png", 4),
    ];
    let text = [9u32, 10];

    let encoder = VisionEncoder::new(&siglip);
    let connector = VisionProjector::new(&projector, &siglip_cfg, 4).unwrap();
    let plan = prepare_multimodal_input(
        &arch,
        &encoder,
        &connector,
        siglip_cfg.image_size,
        &imgs,
        &text,
    )
    .expect("prepare 2 images");

    assert_eq!(plan.chunks.len(), 7);
    // Both image fragments produce identical chunk SHAPES but
    // (because the images differ) distinct precomputed values.
    // Spot-check that the two Precomputed chunks differ — this is
    // the "the pipeline actually conditions on image content"
    // sanity check that the Phase 1d.4 caption test will harden.
    let row_a = match &plan.chunks[1] {
        EmbeddingChunk::Precomputed { rows, .. } => rows.row(0).to_owned(),
        _ => panic!(),
    };
    let row_b = match &plan.chunks[4] {
        EmbeddingChunk::Precomputed { rows, .. } => rows.row(0).to_owned(),
        _ => panic!(),
    };
    let differ = row_a
        .iter()
        .zip(row_b.iter())
        .any(|(a, b)| (a - b).abs() > 1e-6);
    assert!(
        differ,
        "two different images should yield two different projected rows"
    );
}

// ─── Phase 1d.3c: capability check fires before encoder work ────────
//
// The contract per ADR-0023 is "fail fast on engine incompatibility
// BEFORE the encoder runs." `ensure_engine_supports_multimodal` is
// the extracted helper; this test pins its semantics:
//   - Standard (the MM-capable engine) → Ok
//   - NoCache (default-false debt) → Err, naming both the
//     incapable engine AND the recommended fix (`--engine standard`).
//
// Ordering — that the helper is called in `run_with_images` BEFORE
// load_vision_tower_from_safetensors or any encode — is structural and
// covered by code review of `run_with_images`. The full-integration
// ordering test lives in Phase 1d.4.

#[test]
fn ensure_engine_supports_multimodal_accepts_standard() {
    use larql_inference::kv_engine::AnyEngine;
    use larql_kv::engines::standard::StandardEngine;
    // Wrap in AnyEngine::Kv — post kv-engine-retrieval-trait-split,
    // the helper takes AnyEngine, not a raw KvEngine trait object.
    let engine = AnyEngine::Kv(Box::new(StandardEngine::new(None)));
    ensure_engine_supports_multimodal(&engine).expect("Standard supports MM");
}

#[test]
fn ensure_engine_supports_multimodal_rejects_no_cache_with_actionable_message() {
    use larql_inference::kv_engine::AnyEngine;
    use larql_kv::engines::no_cache::NoCacheEngine;
    let engine = AnyEngine::Kv(Box::new(NoCacheEngine::new()));
    let err =
        ensure_engine_supports_multimodal(&engine).expect_err("NoCache should be rejected for MM");
    // Error must name BOTH the incapable engine (so the user knows
    // what failed) AND the fix (so they know what to do).
    assert!(
        err.contains("no-cache") || err.contains("nocache") || err.contains("NoCache"),
        "error must name the incapable engine: {err}"
    );
    assert!(
        err.contains("--engine standard") || err.contains("`--engine standard`"),
        "error must suggest --engine standard as the fix: {err}"
    );
}

#[test]
fn text_only_call_with_zero_images_still_works() {
    // Edge case: --image flag empty → no image chunks, just a
    // single Tokens chunk for text. This is the "MM-flag-aware
    // path on a text-only run" case.
    let arch = synth_arch();
    let siglip_cfg = synth_siglip_config_4x4_patch2();
    let siglip = synth_siglip(&siglip_cfg);
    let projector = synth_projector(siglip_cfg.hidden_size, 12);

    let imgs: Vec<std::path::PathBuf> = vec![];
    let text = [42u32, 43];
    let encoder = VisionEncoder::new(&siglip);
    let connector = VisionProjector::new(&projector, &siglip_cfg, 4).unwrap();
    let plan = prepare_multimodal_input(
        &arch,
        &encoder,
        &connector,
        siglip_cfg.image_size,
        &imgs,
        &text,
    )
    .expect("zero-images call");
    assert_eq!(plan.chunks.len(), 1);
    assert!(plan.is_text_only());
}

// ─── Phase 2: PerTile path (Granite Vision) ──────────────────────────

struct TestPerTileMm;
impl MultiModalProtocol for TestPerTileMm {
    fn vision_encoder(&self) -> Option<&str> {
        Some("siglip2")
    }
    fn image_placeholder(&self) -> Option<PlaceholderProtocol> {
        Some(PlaceholderProtocol {
            start: None,
            fill: 49152,
            end: None,
        })
    }
    fn image_token_budget(&self) -> TokenBudget {
        TokenBudget::PerTile { tokens_per_tile: 4 }
    }
    fn precomputed_scaling(&self) -> PrecomputedScaling {
        PrecomputedScaling::None
    }
    fn valid_tile_counts(&self) -> &[usize] {
        &[1, 2, 3, 4]
    }
}

struct TestPerTileArch {
    config: larql_models::ModelConfig,
    mm: TestPerTileMm,
}

impl ArchitectureCore for TestPerTileArch {
    fn family(&self) -> &str {
        "test-per-tile"
    }

    fn config(&self) -> &larql_models::ModelConfig {
        &self.config
    }

    fn multimodal(&self) -> Option<&dyn MultiModalProtocol> {
        Some(&self.mm)
    }
}

impl TensorKeys for TestPerTileArch {}
impl Norms for TestPerTileArch {}
impl Position for TestPerTileArch {}
impl Attention for TestPerTileArch {}
impl FeedForward for TestPerTileArch {}
impl LatentAttention for TestPerTileArch {}
impl Embeddings for TestPerTileArch {}
impl ModelArchitecture for TestPerTileArch {}

fn synth_per_tile_arch() -> TestPerTileArch {
    let w = larql_models::test_fixtures::make_test_weights();
    TestPerTileArch {
        config: w.arch.config().clone(),
        mm: TestPerTileMm,
    }
}

#[test]
fn per_tile_plan_has_multiple_splice_points_per_image() {
    let arch = synth_per_tile_arch();
    let siglip_cfg = synth_siglip_config_4x4_patch2();
    let siglip = synth_siglip(&siglip_cfg);
    let encoder = VisionEncoder::new(&siglip);
    use larql_compute::connectors::mlp_connector::MlpGelu;
    use larql_models::connectors::mlp_connector::MlpConnectorWeights;
    let mlp_weights = MlpConnectorWeights {
        fc1_weight: Array2::from_shape_fn((16, 8), |(i, j)| asymmetric(i, j)),
        fc1_bias: vec![0.01; 16],
        fc2_weight: Array2::from_shape_fn((12, 16), |(i, j)| asymmetric(i + 5, j)),
        fc2_bias: vec![-0.01; 12],
    };
    let connector = MlpGelu::new(&mlp_weights);

    let tmp = tempfile::tempdir().unwrap();
    let img = write_synth_png(tmp.path(), "tile_test.png", 8);
    let text = [1u32, 2, 3];

    let plan = prepare_multimodal_input(
        &arch,
        &encoder,
        &connector,
        siglip_cfg.image_size,
        std::slice::from_ref(&img),
        &text,
    )
    .expect("per-tile plan");

    // PerTile path: each tile generates (fill_token, Precomputed).
    // For an 8x8 image with tile_size=4 and valid_tile_counts=[1..4],
    // the grid should be >= 1 tile. Base + detail tiles = N+1 total.
    // Each tile = 2 chunks (fill token + precomputed).
    // Plus 1 trailing text chunk.
    assert!(
        plan.chunks.len() >= 3,
        "need at least 1 tile (2 chunks) + text = 3; got {}",
        plan.chunks.len()
    );
    assert!(!plan.is_text_only());

    // Verify the pattern: alternating (Tokens([fill]), Precomputed)
    let tile_chunks = plan.chunks.len() - 1; // exclude trailing text
    assert_eq!(
        tile_chunks % 2,
        0,
        "tile chunks should come in pairs (fill, precomputed); got {tile_chunks}"
    );
    let num_tiles = tile_chunks / 2;
    assert!(
        num_tiles >= 2,
        "AnyRes should produce base + at least 1 detail tile; got {num_tiles}"
    );

    for i in 0..num_tiles {
        match &plan.chunks[i * 2] {
            EmbeddingChunk::Tokens(toks) => {
                assert_eq!(toks, &vec![49152], "tile {i} fill token");
            }
            other => panic!("tile {i} chunk 0 should be fill token, got {other:?}"),
        }
        match &plan.chunks[i * 2 + 1] {
            EmbeddingChunk::Precomputed { rows, modality } => {
                assert_eq!(rows.ncols(), 12, "projected to connector output dim");
                assert!(rows.iter().all(|v| v.is_finite()));
                assert_eq!(*modality, Modality::Image);
            }
            other => panic!("tile {i} chunk 1 should be precomputed, got {other:?}"),
        }
    }

    // Trailing text chunk
    match plan.chunks.last().unwrap() {
        EmbeddingChunk::Tokens(toks) => assert_eq!(toks, &vec![1, 2, 3]),
        other => panic!("last chunk should be text, got {other:?}"),
    }
}

#[test]
fn per_tile_two_images_produce_independent_tile_sets() {
    let arch = synth_per_tile_arch();
    let siglip_cfg = synth_siglip_config_4x4_patch2();
    let siglip = synth_siglip(&siglip_cfg);
    let encoder = VisionEncoder::new(&siglip);
    use larql_compute::connectors::mlp_connector::MlpGelu;
    use larql_models::connectors::mlp_connector::MlpConnectorWeights;
    let mlp_weights = MlpConnectorWeights {
        fc1_weight: Array2::from_shape_fn((16, 8), |(i, j)| asymmetric(i, j)),
        fc1_bias: vec![0.01; 16],
        fc2_weight: Array2::from_shape_fn((12, 16), |(i, j)| asymmetric(i + 5, j)),
        fc2_bias: vec![-0.01; 12],
    };
    let connector = MlpGelu::new(&mlp_weights);

    let tmp = tempfile::tempdir().unwrap();
    let imgs = vec![
        write_synth_png(tmp.path(), "a.png", 8),
        write_synth_png(tmp.path(), "b.png", 8),
    ];
    let text = [99u32];

    let plan = prepare_multimodal_input(
        &arch,
        &encoder,
        &connector,
        siglip_cfg.image_size,
        &imgs,
        &text,
    )
    .expect("per-tile 2 images");

    // Two images should produce more tile chunks than one.
    let tile_chunks = plan.chunks.len() - 1;
    assert!(tile_chunks % 2 == 0);
    let num_tiles = tile_chunks / 2;
    assert!(
        num_tiles >= 4,
        "two images should yield at least 2*(base+1 detail) = 4 tiles; got {num_tiles}"
    );
}
