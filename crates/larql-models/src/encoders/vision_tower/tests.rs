use super::*;

/// Env var naming a local `google/gemma-3-4b-it` snapshot directory, for
/// the `#[ignore]`d real-checkpoint tests below.
const GEMMA3_4B_IT_SNAPSHOT_ENV: &str = "LARQL_GEMMA3_4B_IT_SNAPSHOT";

fn gemma3_4b_it_vision_config() -> VisionConfig {
    VisionConfig {
        hidden_size: 1152,
        intermediate_size: 4304,
        num_attention_heads: 16,
        num_hidden_layers: 27,
        patch_size: 14,
        image_size: 896,
        num_channels: 3,
        layer_norm_eps: 1e-6,
        hidden_act: "gelu_pytorch_tanh".to_string(),
        norm_type: "layer_norm".to_string(),
    }
}

#[test]
fn parses_gemma3_4b_it_vision_config_from_json() {
    // This is exactly the `vision_config` object from
    // google/gemma-3-4b-it/config.json (model_type stripped — we only
    // parse the encoder-relevant fields).
    let json = serde_json::json!({
        "hidden_size": 1152,
        "image_size": 896,
        "intermediate_size": 4304,
        "num_attention_heads": 16,
        "num_hidden_layers": 27,
        "patch_size": 14
    });
    let cfg = VisionConfig::from_json(&json).expect("parse");
    assert_eq!(cfg.hidden_size, 1152);
    assert_eq!(cfg.intermediate_size, 4304);
    assert_eq!(cfg.num_attention_heads, 16);
    assert_eq!(cfg.num_hidden_layers, 27);
    assert_eq!(cfg.patch_size, 14);
    assert_eq!(cfg.image_size, 896);
    assert_eq!(cfg.num_channels, 3, "default num_channels");
    assert!((cfg.layer_norm_eps - 1e-6).abs() < 1e-12, "default eps");
}

#[test]
fn config_geometry_helpers() {
    let cfg = gemma3_4b_it_vision_config();
    assert_eq!(cfg.patches_per_side(), 64, "896 / 14");
    assert_eq!(cfg.num_patches(), 4096, "64 * 64");
    assert_eq!(cfg.head_dim(), 72, "1152 / 16");
}

#[test]
fn config_honours_explicit_num_channels_and_eps() {
    let json = serde_json::json!({
        "hidden_size": 768,
        "image_size": 224,
        "intermediate_size": 3072,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "patch_size": 16,
        "num_channels": 1,
        "layer_norm_eps": 1e-5
    });
    let cfg = VisionConfig::from_json(&json).expect("parse");
    assert_eq!(cfg.num_channels, 1);
    assert!((cfg.layer_norm_eps - 1e-5).abs() < 1e-12);
}

#[test]
fn config_rejects_missing_required_field() {
    let json = serde_json::json!({
        "hidden_size": 1152,
        // missing image_size, intermediate_size, etc.
    });
    let err = VisionConfig::from_json(&json).expect_err("should fail");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("missing") || msg.contains("image_size"),
        "expected missing-field error, got: {msg}"
    );
}

#[test]
fn load_vision_tower_from_safetensors_errors_on_missing_dir() {
    let cfg = gemma3_4b_it_vision_config();
    let err = load_vision_tower_from_safetensors("/nonexistent/path/xyz", cfg)
        .expect_err("should fail on missing dir");
    assert!(!format!("{err:?}").is_empty());
}

#[test]
fn load_siglip_errors_on_directory_with_no_vision_tower_tensors() {
    // Empty tempdir → no safetensors files at all → "no vision_tower
    // tensors found" error.
    let tmp = tempfile::tempdir().expect("tempdir");
    let cfg = gemma3_4b_it_vision_config();
    let err = load_vision_tower_from_safetensors(tmp.path(), cfg)
        .expect_err("empty dir should fail to load SigLIP");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("vision_tower") || msg.contains("multimodal"),
        "error should mention vision_tower / multimodal: {msg}"
    );
}

// ── Synthetic-fixture loader tests ─────────────────────────────────────
//
// Round-trips a tiny SigLIP through `safetensors::serialize` →
// tempfile → `load_vision_tower_from_safetensors`. Covers the entire loader
// body (mmap open, per-layer assembly, take_tensor / take_vec error
// paths) without needing a real ~9 GB Gemma 3 checkpoint. Runs in CI.

fn tiny_siglip_config() -> VisionConfig {
    // 4×4 image, 2×2 patches → 4 positions; hidden=8, 1 layer,
    // intermediate=16. Just enough for shape-correct safetensors
    // round-trip.
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

fn f32_bytes(values: Vec<f32>) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Write a synthetic Gemma 3 vision_tower safetensors file into
/// `dir`. Returns the path. Tensors are filled with zeros — shape
/// fidelity is what the loader validates.
fn write_synth_siglip_safetensors(dir: &std::path::Path, cfg: &VisionConfig) {
    use safetensors::tensor::{serialize_to_file, TensorView};
    use safetensors::Dtype;

    let h = cfg.hidden_size;
    let p = cfg.patch_size;
    let c = cfg.num_channels;
    let np = cfg.num_patches();
    let inter = cfg.intermediate_size;

    // Build all the byte buffers first so TensorView's borrows are
    // valid for the lifetime of the serialize call.
    let mut bufs: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
    let prefix = "vision_tower.vision_model.";

    bufs.push((
        format!("{prefix}embeddings.patch_embedding.weight"),
        vec![h, c, p, p],
        f32_bytes(vec![0.0; h * c * p * p]),
    ));
    bufs.push((
        format!("{prefix}embeddings.patch_embedding.bias"),
        vec![h],
        f32_bytes(vec![0.0; h]),
    ));
    bufs.push((
        format!("{prefix}embeddings.position_embedding.weight"),
        vec![np, h],
        f32_bytes(vec![0.0; np * h]),
    ));
    bufs.push((
        format!("{prefix}post_layernorm.weight"),
        vec![h],
        f32_bytes(vec![1.0; h]),
    ));
    bufs.push((
        format!("{prefix}post_layernorm.bias"),
        vec![h],
        f32_bytes(vec![0.0; h]),
    ));

    for l in 0..cfg.num_hidden_layers {
        let lp = format!("{prefix}encoder.layers.{l}.");
        // layer norms (weight + bias each)
        for which in ["layer_norm1", "layer_norm2"] {
            bufs.push((
                format!("{lp}{which}.weight"),
                vec![h],
                f32_bytes(vec![1.0; h]),
            ));
            bufs.push((
                format!("{lp}{which}.bias"),
                vec![h],
                f32_bytes(vec![0.0; h]),
            ));
        }
        // attention projections — all (h, h) with (h,) bias
        for proj in [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.out_proj",
        ] {
            bufs.push((
                format!("{lp}{proj}.weight"),
                vec![h, h],
                f32_bytes(vec![0.0; h * h]),
            ));
            bufs.push((format!("{lp}{proj}.bias"), vec![h], f32_bytes(vec![0.0; h])));
        }
        // MLP fc1 (inter, h) + bias (inter,), fc2 (h, inter) + bias (h,)
        bufs.push((
            format!("{lp}mlp.fc1.weight"),
            vec![inter, h],
            f32_bytes(vec![0.0; inter * h]),
        ));
        bufs.push((
            format!("{lp}mlp.fc1.bias"),
            vec![inter],
            f32_bytes(vec![0.0; inter]),
        ));
        bufs.push((
            format!("{lp}mlp.fc2.weight"),
            vec![h, inter],
            f32_bytes(vec![0.0; h * inter]),
        ));
        bufs.push((
            format!("{lp}mlp.fc2.bias"),
            vec![h],
            f32_bytes(vec![0.0; h]),
        ));
    }

    // Wrap each as TensorView<'_> against the borrowed byte buffers.
    let views: Vec<(String, TensorView<'_>)> = bufs
        .iter()
        .map(|(name, shape, bytes)| {
            (
                name.clone(),
                TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap(),
            )
        })
        .collect();
    let view_refs: Vec<(&str, &TensorView<'_>)> =
        views.iter().map(|(n, v)| (n.as_str(), v)).collect();
    let path = dir.join("model.safetensors");
    serialize_to_file(view_refs, None, &path).expect("write synth siglip safetensors");
}

#[test]
fn load_siglip_round_trip_against_synthetic_safetensors() {
    let tmp = tempfile::tempdir().unwrap();
    let cfg = tiny_siglip_config();
    write_synth_siglip_safetensors(tmp.path(), &cfg);

    let w = load_vision_tower_from_safetensors(tmp.path(), cfg.clone())
        .expect("synthetic siglip should load cleanly");
    assert_eq!(w.config.num_hidden_layers, cfg.num_hidden_layers);
    assert_eq!(w.layers.len(), cfg.num_hidden_layers);
    assert_eq!(
        w.patch_embed.shape(),
        &[
            cfg.hidden_size,
            cfg.num_channels,
            cfg.patch_size,
            cfg.patch_size
        ]
    );
    assert_eq!(w.patch_embed_bias.len(), cfg.hidden_size);
    assert_eq!(
        w.position_embed.shape(),
        &[cfg.num_patches(), cfg.hidden_size]
    );
    assert_eq!(w.post_layernorm.weight.len(), cfg.hidden_size);
    assert_eq!(w.post_layernorm.bias.len(), cfg.hidden_size);
    let l0 = &w.layers[0];
    assert_eq!(
        l0.q_proj.weight.shape(),
        &[cfg.hidden_size, cfg.hidden_size]
    );
    assert_eq!(l0.q_proj.bias.len(), cfg.hidden_size);
    assert_eq!(
        l0.fc1.weight.shape(),
        &[cfg.intermediate_size, cfg.hidden_size]
    );
    assert_eq!(l0.fc1.bias.len(), cfg.intermediate_size);
    assert_eq!(
        l0.fc2.weight.shape(),
        &[cfg.hidden_size, cfg.intermediate_size]
    );
    assert_eq!(l0.fc2.bias.len(), cfg.hidden_size);
}

#[test]
fn load_siglip_errors_on_missing_required_tensor() {
    // Write a partial fixture — drop the post_layernorm tensors.
    use safetensors::tensor::{serialize_to_file, TensorView};
    use safetensors::Dtype;
    let tmp = tempfile::tempdir().unwrap();
    let cfg = tiny_siglip_config();
    let h = cfg.hidden_size;
    let p = cfg.patch_size;
    let c = cfg.num_channels;
    let bufs: [(String, Vec<usize>, Vec<u8>); 2] = [
        (
            "vision_tower.vision_model.embeddings.patch_embedding.weight".to_string(),
            vec![h, c, p, p],
            f32_bytes(vec![0.0; h * c * p * p]),
        ),
        (
            "vision_tower.vision_model.embeddings.patch_embedding.bias".to_string(),
            vec![h],
            f32_bytes(vec![0.0; h]),
        ),
    ];
    let views: Vec<(String, TensorView<'_>)> = bufs
        .iter()
        .map(|(n, s, b)| {
            (
                n.clone(),
                TensorView::new(Dtype::F32, s.clone(), b).unwrap(),
            )
        })
        .collect();
    let view_refs: Vec<(&str, &TensorView<'_>)> =
        views.iter().map(|(n, v)| (n.as_str(), v)).collect();
    let path = tmp.path().join("model.safetensors");
    serialize_to_file(view_refs, None, &path).expect("write partial siglip");

    let err = load_vision_tower_from_safetensors(tmp.path(), cfg)
        .expect_err("missing tensor should error");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("missing") || msg.contains("post_layernorm") || msg.contains("position"),
        "error must name what's missing: {msg}"
    );
}

#[test]
fn loader_ignores_non_vision_tower_tensors_and_unknown_ranks() {
    // Exercises two uncovered branches in load_one_file:
    //   1. strip_prefix returns None → tensor is silently skipped
    //   2. shape.len() matches the catch-all `_ => {}` arm (rank 3)
    // Both are edge-case tolerance for real checkpoints that carry
    // extra tensors (e.g. language_model.* alongside vision_tower.*).
    use safetensors::tensor::{serialize_to_file, TensorView};
    use safetensors::Dtype;
    let tmp = tempfile::tempdir().unwrap();
    let cfg = tiny_siglip_config();
    // Write the full valid fixture first.
    write_synth_siglip_safetensors(tmp.path(), &cfg);
    // Append a second safetensors file with a non-vision tensor and
    // a rank-3 tensor that should both be ignored.
    let non_vision_bytes = f32_bytes(vec![0.0; 16]);
    let rank3_bytes = f32_bytes(vec![0.0; 2 * 2 * 2]);
    let nv = TensorView::new(Dtype::F32, vec![4, 4], &non_vision_bytes).unwrap();
    let r3 = TensorView::new(Dtype::F32, vec![2, 2, 2], &rank3_bytes).unwrap();
    // A vision-prefixed rank-3 tensor exercises the `_ => {}` arm.
    let pairs: Vec<(&str, &TensorView<'_>)> = vec![
        ("language_model.embed.weight", &nv),
        ("vision_tower.vision_model.extra_3d_tensor", &r3),
    ];
    serialize_to_file(pairs, None, &tmp.path().join("extra.safetensors"))
        .expect("write extra safetensors");
    // Load should succeed — the extra tensors are silently ignored.
    let w = load_vision_tower_from_safetensors(tmp.path(), cfg)
        .expect("loader should ignore non-vision and unknown-rank tensors");
    assert_eq!(w.layers.len(), 1, "only the valid layer loaded");
}

// ── End-to-end: real Gemma 3 4B-it checkpoint ─────────────────────────
//
// Loads vision_tower tensors from the locally-cached Gemma 3 4B-it
// snapshot. Ignored by default — the checkpoint is ~9 GB total and
// not present on CI. Run locally via:
//
//   cargo test -p larql-models --lib encoders::siglip::tests::load_real \
//       -- --ignored --nocapture
//
// Point LARQL_GEMMA3_4B_IT_SNAPSHOT at the snapshot directory first.

#[test]
#[ignore = "requires google/gemma-3-4b-it in the local HF cache"]
fn load_real_gemma3_4b_it_vision_tower() {
    let snap = std::env::var(GEMMA3_4B_IT_SNAPSHOT_ENV).unwrap_or_else(|_| {
        panic!("set {GEMMA3_4B_IT_SNAPSHOT_ENV} to a local google/gemma-3-4b-it snapshot dir")
    });
    let snap = snap.as_str();
    let cfg = gemma3_4b_it_vision_config();
    let w = load_vision_tower_from_safetensors(snap, cfg).expect("load");
    // Geometry checks against the real checkpoint.
    assert_eq!(w.config.num_hidden_layers, 27);
    assert_eq!(w.layers.len(), 27);
    assert_eq!(
        w.patch_embed.shape(),
        &[1152, 3, 14, 14],
        "Conv2D patch projection shape"
    );
    assert_eq!(w.patch_embed_bias.len(), 1152);
    assert_eq!(w.position_embed.shape(), &[4096, 1152]);
    assert_eq!(w.post_layernorm.weight.len(), 1152);
    assert_eq!(w.post_layernorm.bias.len(), 1152);
    // Spot-check layer 0 shapes.
    let l0 = &w.layers[0];
    assert_eq!(l0.q_proj.weight.shape(), &[1152, 1152]);
    assert_eq!(l0.q_proj.bias.len(), 1152);
    assert_eq!(l0.fc1.weight.shape(), &[4304, 1152]);
    assert_eq!(l0.fc1.bias.len(), 4304);
    assert_eq!(l0.fc2.weight.shape(), &[1152, 4304]);
    assert_eq!(l0.fc2.bias.len(), 1152);
    assert_eq!(l0.layer_norm1.weight.len(), 1152);
    // Spot-check finiteness on a single tensor.
    assert!(l0.q_proj.weight.iter().all(|v| v.is_finite()));
}

// ── SigLIP2 config extensions ────────────────────────────────────────

#[test]
fn siglip_config_is_not_siglip2() {
    let cfg = gemma3_4b_it_vision_config();
    assert!(!cfg.is_siglip2());
}

#[test]
fn siglip2_gelu_config_is_siglip2() {
    let mut cfg = gemma3_4b_it_vision_config();
    cfg.hidden_act = "gelu".to_string();
    assert!(cfg.is_siglip2());
}

#[test]
fn siglip2_silu_config_is_siglip2() {
    let mut cfg = gemma3_4b_it_vision_config();
    cfg.hidden_act = "silu".to_string();
    assert!(cfg.is_siglip2());
}

#[test]
fn siglip2_rmsnorm_config_is_siglip2() {
    let mut cfg = gemma3_4b_it_vision_config();
    cfg.norm_type = "rms_norm".to_string();
    assert!(cfg.is_siglip2());
}

#[test]
fn hidden_act_defaults_to_gelu_pytorch_tanh() {
    let json = serde_json::json!({
        "hidden_size": 768,
        "image_size": 224,
        "intermediate_size": 3072,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "patch_size": 16
    });
    let cfg = VisionConfig::from_json(&json).unwrap();
    assert_eq!(cfg.hidden_act, "gelu_pytorch_tanh");
    assert_eq!(cfg.norm_type, "layer_norm");
    assert!(!cfg.is_siglip2());
}

#[test]
fn explicit_hidden_act_and_norm_type_parse() {
    let json = serde_json::json!({
        "hidden_size": 768,
        "image_size": 224,
        "intermediate_size": 3072,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "patch_size": 16,
        "hidden_act": "silu",
        "norm_type": "rms_norm"
    });
    let cfg = VisionConfig::from_json(&json).unwrap();
    assert_eq!(cfg.hidden_act, "silu");
    assert_eq!(cfg.norm_type, "rms_norm");
    assert!(cfg.is_siglip2());
}
