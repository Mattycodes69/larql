//! Tests for [`super::write_synthetic_bitnet_model_dir`]: the fixture is
//! only useful if the real BitNet loader accepts it.

use super::*;

#[test]
fn synthetic_bitnet_container_loads_through_load_bitnet_model() {
    let dir = tempfile::tempdir().expect("tempdir");
    write_synthetic_bitnet_model_dir(dir.path()).expect("write fixture");

    let config = larql_vindex::load_vindex_config(dir.path()).expect("index.json");
    let layout = config
        .bitnet_layout
        .as_ref()
        .expect("bitnet_layout stamped");
    assert!(
        config.layers.is_empty(),
        "a --dense-only build carries no gate layers"
    );
    const BITLINEARS_PER_LAYER: usize = 7;
    assert_eq!(
        layout.tensors.len(),
        BITLINEARS_PER_LAYER * BITNET_TEST_NUM_LAYERS
    );

    let model = crate::ternary::load_bitnet_model(dir.path()).expect("load_bitnet_model");
    assert_eq!(model.layers.len(), BITNET_TEST_NUM_LAYERS);
    assert_eq!(model.head_dim, BITNET_TEST_HEAD_DIM);
    assert_eq!(model.n_q_heads, BITNET_TEST_NUM_Q_HEADS);
    assert_eq!(model.n_kv_heads, BITNET_TEST_NUM_KV_HEADS);
    assert_eq!(model.output_norm.len(), BITNET_TEST_HIDDEN);
    assert_eq!(
        model.embed.shape(),
        &[BITNET_TEST_VOCAB, BITNET_TEST_HIDDEN]
    );
    let q = &model.layers[0].attn_q;
    assert_eq!(
        (q.rows, q.cols),
        (
            BITNET_TEST_NUM_Q_HEADS * BITNET_TEST_HEAD_DIM,
            BITNET_TEST_HIDDEN
        )
    );
    assert!(
        q.channel_scales.iter().all(|&s| s == BITNET_TEST_I2S_SCALE),
        "the per-tensor I2_S scale is broadcast to every row"
    );
    assert_eq!(model.layers[0].ffn.ffn_sub_norm.len(), BITNET_TEST_INTER);
}

#[test]
fn synthetic_bitnet_container_refuses_a_non_directory() {
    let dir = tempfile::tempdir().expect("tempdir");
    let file = dir.path().join("occupied");
    std::fs::write(&file, b"not a directory").expect("write");
    let err = write_synthetic_bitnet_model_dir(&file).expect_err("a file is not a dir");
    assert!(err.contains("build_vindex_dense_only"), "{err}");
}
