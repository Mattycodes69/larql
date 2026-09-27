//! tensors.rs: the pure-MoE arm
//! tensors.rs
//! hooks.rs

use super::*;

#[test]
fn pure_moe_layer_inserts_attention_only() {
    let (weights, index) = attn_only_fixture();
    assert!(weights.arch.is_moe() && !weights.arch.is_hybrid_moe());
    let mut scratch = larql_models::DequantScratch::new();
    let keys = insert_q4k_layer_tensors(&mut scratch, &weights, &index, 0)
        .expect("pure MoE must not require a dense FFN slab");
    assert_eq!(keys.len(), 4, "attention only: Q/K/V/O");
    assert!(keys.iter().all(|k| k.contains("self_attn")));
    for k in &keys {
        assert!(scratch.contains_key(k), "{k} not inserted");
    }
}

#[test]
fn dense_arch_with_missing_ffn_slices_still_errors() {
    let (mut weights, index) = attn_only_fixture();
    // Same index, but a DENSE architecture: the missing slab is now a
    // defect, not a topology.
    weights.arch = larql_models::detect_from_json(&serde_json::json!({
        "model_type": "llama",
        "hidden_size": 16,
        "intermediate_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "head_dim": 4,
    }));
    let mut scratch = larql_models::DequantScratch::new();
    let err = insert_q4k_layer_tensors(&mut scratch, &weights, &index, 0)
        .expect_err("a dense arch without FFN slices is a broken vindex");
    assert!(err.contains("ffn Q4K slices missing"), "{err}");
}

#[test]
fn walk_ffn_kquant_layer_runs_gelu_tanh_path() {
    // Gemma-3 weights → GeluTanh activation branch.
    let weights = make_test_q4k_weights();
    let idx = make_q4k_fixture_index(&weights);
    let x =
        Array2::<f32>::from_shape_vec((1, weights.hidden_size), vec![0.01; weights.hidden_size])
            .unwrap();
    let out = kquant_ffn_forward_layer(&*weights.arch, &idx, 0, &x);
    assert_eq!(out.shape(), &[1, weights.hidden_size]);
}

#[test]
fn walk_ffn_kquant_layer_runs_silu_path() {
    // SiLU-activation sibling weights → silu_gate_up branch.
    let weights = make_test_q4k_weights_silu();
    let idx = make_q4k_fixture_index(&weights);
    let x =
        Array2::<f32>::from_shape_vec((1, weights.hidden_size), vec![0.01; weights.hidden_size])
            .unwrap();
    let out = kquant_ffn_forward_layer(&*weights.arch, &idx, 0, &x);
    assert_eq!(out.shape(), &[1, weights.hidden_size]);
}

/// `kquant_ffn_forward_layer` non-aligned-intermediate branch:
/// when the index reports `num_features` that's not a multiple of
/// `K_QUANT_BLOCK_ELEMS`, the down-projection path pads up to the
/// next multiple and slices back down. Covers walk_ffn.rs:65-66.
#[test]
fn walk_ffn_kquant_layer_handles_non_aligned_intermediate() {
    struct NonAlignedIntermediate<'a> {
        inner: &'a crate::test_fixtures::Q4kFixtureIndex,
        claimed_intermediate: usize,
    }
    impl crate::KvIndex for NonAlignedIntermediate<'_> {
        fn num_features(&self, _l: usize) -> usize {
            self.claimed_intermediate
        }
        fn attn_kquant_layer_data(&self, l: usize) -> Option<[(&[u8], &str); 4]> {
            self.inner.attn_kquant_layer_data(l)
        }
        fn interleaved_kquant_layer_data(
            &self,
            l: usize,
        ) -> Option<[(&[u8], &str); crate::FFN_COMPONENTS_PER_LAYER]> {
            self.inner.interleaved_kquant_layer_data(l)
        }
        fn interleaved_kquant_mmap_ref(&self) -> Option<&[u8]> {
            self.inner.interleaved_kquant_mmap_ref()
        }
        // No `kquant_ffn_layer_once` — forces dequant path which
        // pads to the next K_QUANT_BLOCK_ELEMS multiple.
    }
    let weights = make_test_q4k_weights();
    let inner = make_q4k_fixture_index(&weights);
    let idx = NonAlignedIntermediate {
        inner: &inner,
        // Real intermediate is 256; claim 200 → padded to 256, branch
        // fires.
        claimed_intermediate: 200,
    };
    let x = ndarray::Array2::<f32>::from_shape_vec(
        (1, weights.hidden_size),
        vec![0.01; weights.hidden_size],
    )
    .unwrap();
    let result = kquant_ffn_forward_layer(&*weights.arch, &idx, 0, &x);
    // The non-aligned branch slices the down-projection output;
    // shape should be `(1, hidden_size)`.
    assert_eq!(result.shape(), &[1, weights.hidden_size]);
}

#[test]
fn walk_ffn_kquant_layer_runs_dequant_fallback_when_cache_disabled() {
    // `disable_ffn_cache` forces `kquant_ffn_layer_once` → None, so
    // walk_ffn takes the `dequantize_matrix` branch on every
    // gate/up/down.
    let weights = make_test_q4k_weights();
    let idx = make_q4k_fixture_index(&weights).without_ffn_cache();
    let x =
        Array2::<f32>::from_shape_vec((1, weights.hidden_size), vec![0.01; weights.hidden_size])
            .unwrap();
    let out = kquant_ffn_forward_layer(&*weights.arch, &idx, 0, &x);
    assert_eq!(out.shape(), &[1, weights.hidden_size]);
}

#[test]
fn walk_ffn_kquant_layer_q8k_runs_gelu_path() {
    use crate::cpu::ops::q4k_q8k_dot::quantize_x_to_q8k;
    let weights = make_test_q4k_weights();
    let idx = make_q4k_fixture_index(&weights);
    let h_in: Vec<f32> = vec![0.01; weights.hidden_size];
    let h_q8k = quantize_x_to_q8k(&h_in);
    let out = kquant_ffn_forward_layer_q8k(&*weights.arch, &idx, 0, &h_q8k);
    assert_eq!(out.shape(), &[1, weights.hidden_size]);
}

#[test]
fn walk_ffn_kquant_layer_q8k_runs_silu_fallback_path() {
    // SiLU activation + cache disabled exercises the fallback
    // (OnceLock cache None) path on the down-projection.
    use crate::cpu::ops::q4k_q8k_dot::quantize_x_to_q8k;
    let weights = make_test_q4k_weights_silu();
    let idx = make_q4k_fixture_index(&weights).without_ffn_cache();
    let h_in: Vec<f32> = vec![0.01; weights.hidden_size];
    let h_q8k = quantize_x_to_q8k(&h_in);
    let out = kquant_ffn_forward_layer_q8k(&*weights.arch, &idx, 0, &h_q8k);
    assert_eq!(out.shape(), &[1, weights.hidden_size]);
}

/// Regression for docs/audits/dec-readiness-review-2026-07-22.md §1a:
/// pins the numerical equivalence the server's batched-GEMM Q8K
/// handler (`larql-server/routes/walk_ffn/q8k.rs`) depends on. Same-
/// layer entries dequantised to f32 and run through
/// `kquant_ffn_forward_layer` as ONE multi-row GEMM must reproduce
/// (within f32 rounding) what the single-row `q4k_q8k_matvec_into`
/// kernel produces per entry — otherwise batching same-layer rows to
/// fix the ~linear-in-B batch curve would silently change the numbers.
#[test]
fn walk_ffn_kquant_layer_q8k_batched_gemm_matches_per_row_single_kernel() {
    use crate::cpu::ops::q4k_q8k_dot::quantize_x_to_q8k;
    let weights = make_test_q4k_weights();
    let idx = make_q4k_fixture_index(&weights);
    let hidden = weights.hidden_size;

    let rows: Vec<Vec<f32>> = (0..3)
        .map(|i| {
            (0..hidden)
                .map(|j| ((i * hidden + j) as f32 * 0.001).sin() * 0.05)
                .collect()
        })
        .collect();

    // Reference: each row through the single-row Q8K fast kernel,
    // exactly as a batch-1 request (or the pre-fix code) would.
    let single_row_outputs: Vec<Vec<f32>> = rows
        .iter()
        .map(|r| {
            let h_q8k = quantize_x_to_q8k(r);
            kquant_ffn_forward_layer_q8k(&*weights.arch, &idx, 0, &h_q8k)
                .into_raw_vec_and_offset()
                .0
        })
        .collect();

    // Batched: dequantise each row's Q8K activation back to f32 and
    // run all 3 rows through ONE GEMM, mirroring the server's grouped
    // handler.
    let mut dequantised: Vec<f32> = Vec::with_capacity(3 * hidden);
    for r in &rows {
        let h_q8k = quantize_x_to_q8k(r);
        for b in 0..h_q8k.n_blocks() {
            let d = h_q8k.d[b];
            for i in 0..256 {
                dequantised.push(d * (h_q8k.qs[b * 256 + i] as f32));
            }
        }
    }
    let x = ndarray::Array2::from_shape_vec((3, hidden), dequantised).unwrap();
    let batched = kquant_ffn_forward_layer(&*weights.arch, &idx, 0, &x);

    for (row_idx, single) in single_row_outputs.iter().enumerate() {
        let batched_row = batched.row(row_idx);
        for (a, &b) in single.iter().zip(batched_row.iter()) {
            assert!(
                (a - b).abs() < 1e-3,
                "row {row_idx}: single-kernel {a} vs batched-GEMM {b} diverge"
            );
        }
    }
}

/// Regression for docs/audits/dec-readiness-review-2026-07-22.md §1d:
/// the down-projection fast path must consult the down slab's own
/// format tag (`ffn[2].1`), not just the block-alignment guard.
/// Before the fix, any down slab — regardless of its declared format —
/// went straight into the Q4_K-only `q4k_q8k_matvec_into` kernel
/// whenever `intermediate` was 256-aligned, silently misreading any
/// non-Q4_K byte layout. Here the down component is (still
/// Q4_K-encoded bytes, but) tagged with an unsupported format string;
/// with the fix, that tag mismatch routes to the format-aware
/// `dequantize_matrix` fallback, which loudly rejects an unknown tag
/// instead of the fast path silently "succeeding" on the wrong
/// assumption.
#[test]
#[should_panic(expected = "unsupported quant format")]
fn walk_ffn_kquant_layer_q8k_rejects_down_slab_with_non_q4k_format_tag() {
    struct BogusDownFormat<'a> {
        inner: &'a crate::test_fixtures::Q4kFixtureIndex,
    }
    impl crate::KvIndex for BogusDownFormat<'_> {
        fn num_features(&self, l: usize) -> usize {
            self.inner.num_features(l)
        }
        fn attn_kquant_layer_data(&self, l: usize) -> Option<[(&[u8], &str); 4]> {
            self.inner.attn_kquant_layer_data(l)
        }
        fn interleaved_kquant_layer_data(
            &self,
            l: usize,
        ) -> Option<[(&[u8], &str); crate::FFN_COMPONENTS_PER_LAYER]> {
            let mut ffn = self.inner.interleaved_kquant_layer_data(l)?;
            ffn[2].1 = "BOGUS_FORMAT";
            Some(ffn)
        }
        fn interleaved_kquant_mmap_ref(&self) -> Option<&[u8]> {
            self.inner.interleaved_kquant_mmap_ref()
        }
        // No `kquant_ffn_layer_once` — forces the dequant fallback,
        // which is the only branch that consults the format tag.
    }
    use crate::cpu::ops::q4k_q8k_dot::quantize_x_to_q8k;
    let weights = make_test_q4k_weights();
    let inner = make_q4k_fixture_index(&weights);
    let idx = BogusDownFormat { inner: &inner };
    let h_in: Vec<f32> = vec![0.01; weights.hidden_size];
    let h_q8k = quantize_x_to_q8k(&h_in);
    let _ = kquant_ffn_forward_layer_q8k(&*weights.arch, &idx, 0, &h_q8k);
}

#[test]
fn tensors_insert_q4k_layer_populates_dense_f32_keys() {
    let weights = make_test_q4k_weights();
    let mut scratch = larql_models::DequantScratch::new();
    let idx = make_q4k_fixture_index(&weights);
    let keys = insert_q4k_layer_tensors(&mut scratch, &weights, &idx, 0)
        .expect("insert_q4k_layer_tensors must succeed on Q4K fixture");
    // Q/K/V/O + gate/up/down = 7 keys per layer.
    assert_eq!(keys.len(), 7);
    for key in &keys {
        assert!(scratch.contains_key(key));
    }
    remove_layer_tensors(&mut scratch, keys.clone());
    for key in &keys {
        assert!(!scratch.contains_key(key));
    }
}

#[test]
fn tensors_insert_q4k_layer_errors_on_missing_attn_data() {
    // An EmptyKvIndex returns None from every accessor — the
    // `ok_or_else` branch in `insert_q4k_layer_tensors` fires.
    struct EmptyIdx;
    impl crate::KvIndex for EmptyIdx {}
    let weights = make_test_q4k_weights();
    let mut scratch = larql_models::DequantScratch::new();
    let result = insert_q4k_layer_tensors(&mut scratch, &weights, &EmptyIdx, 0);
    let err = result.expect_err("missing attn data must fail");
    assert!(err.contains("attn"));
}

#[test]
fn tensors_insert_q4k_layer_errors_on_missing_ffn_data() {
    // Provide attn but not ffn — the second `ok_or_else` fires.
    struct AttnOnlyIdx {
        attn_bytes: Vec<u8>,
    }
    impl crate::KvIndex for AttnOnlyIdx {
        fn num_features(&self, _l: usize) -> usize {
            256
        }
        fn attn_kquant_layer_data(&self, _l: usize) -> Option<[(&[u8], &str); 4]> {
            Some([
                (self.attn_bytes.as_slice(), "Q4_K"),
                (self.attn_bytes.as_slice(), "Q4_K"),
                (self.attn_bytes.as_slice(), "Q4_K"),
                (self.attn_bytes.as_slice(), "Q4_K"),
            ])
        }
    }
    // Reuse a real Q4K-quant slice — the test should hit the ffn
    // check before dequant runs, so the actual content is fine.
    let weights = make_test_q4k_weights();
    let real_idx = make_q4k_fixture_index(&weights);
    let attn_bytes = {
        let dyn_idx: &dyn crate::KvIndex = &real_idx;
        dyn_idx.attn_kquant_layer_data(0).unwrap()[0].0.to_vec()
    };
    let idx = AttnOnlyIdx { attn_bytes };
    let weights = make_test_q4k_weights();
    let mut scratch = larql_models::DequantScratch::new();
    let result = insert_q4k_layer_tensors(&mut scratch, &weights, &idx, 0);
    let err = result.expect_err("missing ffn data must fail");
    assert!(err.contains("ffn"));
}

/// `kquant_ffn_forward_layer` panics when the layer has no
/// interleaved Q4K data. Server-side bug if you reach this path
/// without preloading; the panic message is the contract.
#[test]
#[should_panic(expected = "interleaved_kquant layer data missing")]
fn walk_ffn_panics_when_layer_data_missing() {
    struct AttnOnlyNoFfn;
    impl crate::KvIndex for AttnOnlyNoFfn {
        fn num_features(&self, _l: usize) -> usize {
            256
        }
        // interleaved_kquant_layer_data inherits default None → panic.
    }
    let weights = make_test_q4k_weights();
    let idx = AttnOnlyNoFfn;
    let x = Array2::<f32>::zeros((1, weights.hidden_size));
    let _ = kquant_ffn_forward_layer(&*weights.arch, &idx, 0, &x);
}

/// Same panic path on the Q8K-fused variant.
#[test]
#[should_panic(expected = "interleaved_kquant layer data missing")]
fn walk_ffn_q8k_panics_when_layer_data_missing() {
    use crate::cpu::ops::q4k_q8k_dot::quantize_x_to_q8k;
    struct AttnOnlyNoFfn;
    impl crate::KvIndex for AttnOnlyNoFfn {
        fn num_features(&self, _l: usize) -> usize {
            256
        }
    }
    let weights = make_test_q4k_weights();
    let idx = AttnOnlyNoFfn;
    let h_q8k = quantize_x_to_q8k(&vec![0.0; weights.hidden_size]);
    let _ = kquant_ffn_forward_layer_q8k(&*weights.arch, &idx, 0, &h_q8k);
}
