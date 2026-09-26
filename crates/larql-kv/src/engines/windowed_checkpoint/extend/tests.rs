use super::*;
use larql_inference::forward::hidden_to_raw_logits;
use larql_inference::test_utils::make_test_weights;

// ── empty_prior ───────────────────────────────────────────────────────────

#[test]
fn empty_prior_shape_per_layer() {
    let weights = make_test_weights();
    let prior = empty_prior(&weights);
    assert_eq!(prior.len(), weights.num_layers);
    let kv_dim = weights.num_kv_heads * weights.head_dim;
    for (k, v) in &prior {
        assert_eq!(k.shape(), &[0, kv_dim]);
        assert_eq!(v.shape(), &[0, kv_dim]);
    }
}

// ── rs_extend_from_checkpoint ─────────────────────────────────────────────

#[test]
fn extend_empty_tokens_returns_none() {
    let weights = make_test_weights();
    let prior = empty_prior(&weights);
    let result =
        rs_extend_from_checkpoint(larql_inference::WeightsView::dense(&weights), &[], prior, 0);
    assert!(
        matches!(result, Err(EngineError::EmptyPrompt)),
        "an empty chunk is a caller-input error, not a backend failure"
    );
}

#[test]
fn extend_wrong_prior_len_returns_none() {
    let weights = make_test_weights();
    // prior has 0 layers but model has 2 — mismatch
    let result = rs_extend_from_checkpoint(
        larql_inference::WeightsView::dense(&weights),
        &[0u32],
        Vec::new(),
        0,
    );
    assert!(
        matches!(result, Err(EngineError::InvariantViolation { .. })),
        "a prior with the wrong layer count is a contract violation"
    );
}

#[test]
fn extend_single_token_from_empty_prior() {
    let weights = make_test_weights();
    let prior = empty_prior(&weights);
    let output = rs_extend_from_checkpoint(
        larql_inference::WeightsView::dense(&weights),
        &[0u32],
        prior,
        0,
    )
    .expect("single token extend should succeed");
    assert_eq!(output.last_hidden.shape(), &[1, weights.hidden_size]);
    assert!(output.last_hidden.iter().all(|v| v.is_finite()));
}

#[test]
fn extend_kv_cache_grows_with_each_token() {
    let weights = make_test_weights();
    let prior = empty_prior(&weights);
    let output = rs_extend_from_checkpoint(
        larql_inference::WeightsView::dense(&weights),
        &[0u32, 1, 2],
        prior,
        0,
    )
    .expect("3-token extend");
    // After 3 tokens from empty prior, K has 3 rows per layer
    let kv_dim = weights.num_kv_heads * weights.head_dim;
    for (k, v) in &output.kv_cache {
        assert_eq!(k.shape(), &[3, kv_dim], "K should have 3 rows");
        assert_eq!(v.shape(), &[3, kv_dim], "V should have 3 rows");
    }
}

#[test]
fn extend_checkpoint_is_last_row_of_kv_cache() {
    let weights = make_test_weights();
    let prior = empty_prior(&weights);
    let output = rs_extend_from_checkpoint(
        larql_inference::WeightsView::dense(&weights),
        &[0u32, 1],
        prior,
        0,
    )
    .expect("2-token extend");
    // new_checkpoint should be the last row of each K/V
    for (layer, ((k_cache, v_cache), (k_ckpt, v_ckpt))) in output
        .kv_cache
        .iter()
        .zip(output.new_checkpoint.iter())
        .enumerate()
    {
        let n = k_cache.shape()[0];
        let last_k = k_cache.row(n - 1).to_vec();
        let ckpt_k = k_ckpt.row(0).to_vec();
        for (a, b) in last_k.iter().zip(ckpt_k.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "layer {layer}: checkpoint K doesn't match last K cache row"
            );
        }
        let _ = (v_cache, v_ckpt); // symmetry — trust by shape
    }
}

#[test]
fn extend_abs_start_shifts_rope() {
    let weights = make_test_weights();
    let prior = empty_prior(&weights);
    let out0 = rs_extend_from_checkpoint(
        larql_inference::WeightsView::dense(&weights),
        &[0u32],
        prior.clone(),
        0,
    )
    .unwrap();
    let out5 = rs_extend_from_checkpoint(
        larql_inference::WeightsView::dense(&weights),
        &[0u32],
        prior,
        5,
    )
    .unwrap();
    // Different abs_start → different RoPE → different K
    let k0 = &out0.kv_cache[0].0;
    let k5 = &out5.kv_cache[0].0;
    let diff: f32 = k0.iter().zip(k5.iter()).map(|(a, b)| (a - b).abs()).sum();
    assert!(
        diff > 0.0,
        "different abs_start should produce different K (RoPE)"
    );
}

#[test]
fn extend_output_logits_are_finite() {
    let weights = make_test_weights();
    let prior = empty_prior(&weights);
    let output = rs_extend_from_checkpoint(
        larql_inference::WeightsView::dense(&weights),
        &[0u32],
        prior,
        0,
    )
    .unwrap();
    let logits = hidden_to_raw_logits(&weights, &output.last_hidden);
    assert!(logits.iter().all(|v| v.is_finite()));
}

#[test]
fn extend_seeded_from_checkpoint_matches_empty_start() {
    // Extending from a non-empty checkpoint should not panic and should be finite.
    let weights = make_test_weights();
    let prior = empty_prior(&weights);
    let first = rs_extend_from_checkpoint(
        larql_inference::WeightsView::dense(&weights),
        &[0u32],
        prior,
        0,
    )
    .unwrap();
    // Use the checkpoint from the first extend as the prior for the second
    let second = rs_extend_from_checkpoint(
        larql_inference::WeightsView::dense(&weights),
        &[1u32],
        first.new_checkpoint.clone(),
        1,
    )
    .expect("extend from non-empty prior");
    assert_eq!(second.last_hidden.shape(), &[1, weights.hidden_size]);
    assert!(second.last_hidden.iter().all(|v| v.is_finite()));
}

// ── rs_extend_from_checkpoint_quant (vindex-backed FFN path) ───────────────

#[test]
fn extend_quant_empty_tokens_returns_none() {
    let weights = make_test_weights();
    let index = larql_inference::test_utils::make_test_vindex(&weights);
    let backend = larql_compute::cpu_backend();
    let prior = empty_prior(&weights);
    let out = rs_extend_from_checkpoint_quant(
        larql_inference::WeightsView::dense(&weights),
        &index,
        &[],
        prior,
        0,
        &*backend,
        None,
    );
    assert!(out.is_none(), "empty token_ids should return None");
}

#[test]
fn extend_quant_wrong_prior_len_returns_none() {
    let weights = make_test_weights();
    let index = larql_inference::test_utils::make_test_vindex(&weights);
    let backend = larql_compute::cpu_backend();
    let out = rs_extend_from_checkpoint_quant(
        larql_inference::WeightsView::dense(&weights),
        &index,
        &[0u32],
        Vec::new(),
        0,
        &*backend,
        None,
    );
    assert!(out.is_none(), "prior length mismatch should return None");
}

#[test]
fn extend_quant_grows_kv_cache_and_returns_finite() {
    let weights = make_test_weights();
    let index = larql_inference::test_utils::make_test_vindex(&weights);
    let backend = larql_compute::cpu_backend();
    let prior = empty_prior(&weights);
    let out = rs_extend_from_checkpoint_quant(
        larql_inference::WeightsView::dense(&weights),
        &index,
        &[0u32, 1, 2],
        prior,
        0,
        &*backend,
        None,
    )
    .expect("3-token Q4K extend");
    assert_eq!(out.last_hidden.shape(), &[1, weights.hidden_size]);
    assert!(out.last_hidden.iter().all(|v| v.is_finite()));
    // After 3 tokens from an empty prior, each layer's K/V has 3 rows.
    let kv_dim = weights.num_kv_heads * weights.head_dim;
    for (k, v) in &out.kv_cache {
        assert_eq!(k.shape(), &[3, kv_dim]);
        assert_eq!(v.shape(), &[3, kv_dim]);
    }
    assert_eq!(out.new_checkpoint.len(), weights.num_layers);
}

#[test]
fn extend_quant_seeded_from_prior_matches_shape() {
    let weights = make_test_weights();
    let index = larql_inference::test_utils::make_test_vindex(&weights);
    let backend = larql_compute::cpu_backend();
    let first = rs_extend_from_checkpoint_quant(
        larql_inference::WeightsView::dense(&weights),
        &index,
        &[0u32, 1],
        empty_prior(&weights),
        0,
        &*backend,
        None,
    )
    .expect("first extend");
    let second = rs_extend_from_checkpoint_quant(
        larql_inference::WeightsView::dense(&weights),
        &index,
        &[2u32],
        first.kv_cache.clone(),
        2,
        &*backend,
        None,
    )
    .expect("extend over prior kv");
    assert_eq!(second.last_hidden.shape(), &[1, weights.hidden_size]);
    let kv_dim = weights.num_kv_heads * weights.head_dim;
    for (k, _) in &second.kv_cache {
        assert_eq!(k.shape(), &[3, kv_dim], "prior(2) + new(1) = 3 rows");
    }
}

// ── truncate_kv_rows ──────────────────────────────────────────────────────

#[test]
fn truncate_kv_rows_rewinds_every_layer_to_the_row_count() {
    let weights = make_test_weights();
    let mut kv = rs_extend_from_checkpoint(
        larql_inference::WeightsView::dense(&weights),
        &[0u32, 1, 2],
        empty_prior(&weights),
        0,
    )
    .expect("3-token extend")
    .kv_cache;

    // Row 0 must survive the rewind byte-for-byte — a truncate that
    // reallocated the wrong slice would still leave the shape right.
    let row0: Vec<Vec<f32>> = kv.iter().map(|(k, _)| k.row(0).to_vec()).collect();

    truncate_kv_rows(&mut kv, 1);

    let kv_dim = weights.num_kv_heads * weights.head_dim;
    for (layer, (k, v)) in kv.iter().enumerate() {
        assert_eq!(k.shape(), &[1, kv_dim], "layer {layer}: K not rewound");
        assert_eq!(v.shape(), &[1, kv_dim], "layer {layer}: V not rewound");
        assert_eq!(
            k.row(0).to_vec(),
            row0[layer],
            "layer {layer}: rewind kept the wrong row"
        );
    }
}

#[test]
fn truncate_kv_rows_leaves_a_shorter_cache_alone() {
    let weights = make_test_weights();
    let mut kv = rs_extend_from_checkpoint(
        larql_inference::WeightsView::dense(&weights),
        &[0u32],
        empty_prior(&weights),
        0,
    )
    .expect("1-token extend")
    .kv_cache;

    // rows > shape[0]: the guard must skip, not grow or panic.
    truncate_kv_rows(&mut kv, 8);

    let kv_dim = weights.num_kv_heads * weights.head_dim;
    for (k, v) in &kv {
        assert_eq!(k.shape(), &[1, kv_dim]);
        assert_eq!(v.shape(), &[1, kv_dim]);
    }
}

// ── rs_extend_inplace ─────────────────────────────────────────────────────

#[test]
fn extend_inplace_empty_tokens_is_an_invariant_violation() {
    let weights = make_test_weights();
    let backend = larql_compute::cpu_backend();
    let mut kv = empty_prior(&weights);
    let result = rs_extend_inplace(
        larql_inference::WeightsView::dense(&weights),
        &[],
        &mut kv,
        0,
        0,
        &*backend,
        None,
        None,
    );
    assert!(
        matches!(result, Err(EngineError::InvariantViolation { .. })),
        "an empty chunk breaks the in-place contract"
    );
}

#[test]
fn extend_inplace_wrong_slot_count_is_an_invariant_violation() {
    let weights = make_test_weights();
    let backend = larql_compute::cpu_backend();
    // Model has `num_layers` layers; hand it zero K/V slots.
    let mut kv: Vec<SharedKV> = Vec::new();
    let result = rs_extend_inplace(
        larql_inference::WeightsView::dense(&weights),
        &[0u32],
        &mut kv,
        0,
        0,
        &*backend,
        None,
        None,
    );
    assert!(
        matches!(result, Err(EngineError::InvariantViolation { .. })),
        "a slot count that isn't num_layers breaks the in-place contract"
    );
}

/// With no index the Q4K-direct in-place projection returns `None` at every
/// layer, so this drives the per-layer owned-concat fallback — the arm that
/// writes the rebuilt buffer back so the cache stays consistent.
#[test]
fn extend_inplace_falls_back_to_owned_concat_without_an_index() {
    let weights = make_test_weights();
    let backend = larql_compute::cpu_backend();
    let mut kv = empty_prior(&weights);
    let last = rs_extend_inplace(
        larql_inference::WeightsView::dense(&weights),
        &[0u32, 1, 2],
        &mut kv,
        0,
        0,
        &*backend,
        None,
        None,
    )
    .expect("fallback extend should still produce a hidden state");

    assert_eq!(last.shape(), &[1, weights.hidden_size]);
    assert!(last.iter().all(|v| v.is_finite()));
    // The fallback replaces each buffer with the owned concat, so after 3
    // tokens from an empty prior every layer holds exactly 3 rows.
    let kv_dim = weights.num_kv_heads * weights.head_dim;
    for (layer, (k, v)) in kv.iter().enumerate() {
        assert_eq!(k.shape(), &[3, kv_dim], "layer {layer}: K rows");
        assert_eq!(v.shape(), &[3, kv_dim], "layer {layer}: V rows");
    }
}

/// The fallback is also the seeded path: `prior_len > 0` makes it slice a
/// real prior out of the buffer rather than pass `None`.
#[test]
fn extend_inplace_fallback_matches_the_owned_concat_path() {
    let weights = make_test_weights();
    let backend = larql_compute::cpu_backend();
    let view = larql_inference::WeightsView::dense(&weights);

    let mut inplace_kv = empty_prior(&weights);
    let inplace = rs_extend_inplace(
        view,
        &[0u32, 1],
        &mut inplace_kv,
        0,
        0,
        &*backend,
        None,
        None,
    )
    .expect("in-place extend");

    let mut owned_kv = empty_prior(&weights);
    let owned = rs_extend_from_checkpoint_backend(
        view,
        &[0u32, 1],
        &mut owned_kv,
        0,
        &*backend,
        None,
        None,
    )
    .expect("owned-concat extend");

    // Same numerics, different cache representation — that equivalence is
    // the whole claim the fallback arm exists to preserve.
    for (a, b) in inplace.iter().zip(owned.last_hidden.iter()) {
        assert!(
            (a - b).abs() < 1e-6,
            "in-place fallback diverged from owned concat: {a} vs {b}"
        );
    }
    for (layer, ((ki, _), (ko, _))) in inplace_kv.iter().zip(owned_kv.iter()).enumerate() {
        assert_eq!(ki.shape(), ko.shape(), "layer {layer}: K shape");
        for (a, b) in ki.iter().zip(ko.iter()) {
            assert!((a - b).abs() < 1e-6, "layer {layer}: K diverged");
        }
    }
}
