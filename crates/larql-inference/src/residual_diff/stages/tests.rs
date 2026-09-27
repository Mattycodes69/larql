use super::*;

fn cap(stages: &[(&str, Vec<f32>)], layer: usize, backend: &'static str) -> StageCapture {
    StageCapture {
        stages: stages
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect(),
        layer,
        seq_len: 1,
        backend,
    }
}

fn cap_with_seq(
    stages: &[(&str, Vec<f32>)],
    layer: usize,
    seq_len: usize,
    backend: &'static str,
) -> StageCapture {
    StageCapture {
        stages: stages
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect(),
        layer,
        seq_len,
        backend,
    }
}

#[test]
fn project_to_last_position_slices_per_stride() {
    // [seq=3, hidden=2] for s0; [seq=3, qdim=4] for s1.
    let s0 = vec![1.0, 2.0, 10.0, 20.0, 100.0, 200.0];
    let s1 = vec![0.1, 0.2, 0.3, 0.4, 1.1, 1.2, 1.3, 1.4, 9.1, 9.2, 9.3, 9.4];
    let cap = cap_with_seq(&[("s0", s0), ("s1", s1)], 0, 3, "cpu");
    let proj = cap.project_to_last_position();
    assert_eq!(proj.seq_len, 1);
    assert_eq!(proj.get("s0").unwrap(), &[100.0, 200.0]);
    assert_eq!(proj.get("s1").unwrap(), &[9.1, 9.2, 9.3, 9.4]);
}

#[test]
fn project_to_last_position_keeps_unaligned_stages_unchanged() {
    // seq_len=3 but stage has 7 floats (not a multiple of 3) —
    // unexpected shape. Don't truncate; let the comparison
    // surface it as a length mismatch.
    let cap = cap_with_seq(&[("weird", vec![1.0; 7])], 0, 3, "cpu");
    let proj = cap.project_to_last_position();
    assert_eq!(proj.get("weird").unwrap().len(), 7);
}

#[test]
fn compare_stages_clean_when_all_match() {
    let a = cap(
        &[("norm_out", vec![1.0, 2.0]), ("q_out", vec![3.0, 4.0])],
        0,
        "a",
    );
    let b = cap(
        &[("norm_out", vec![1.0, 2.0]), ("q_out", vec![3.0, 4.0])],
        0,
        "b",
    );
    let r = compare_stages(
        &a,
        &b,
        &[("norm_out", "norm_out"), ("q_out", "q_out")],
        ParityThreshold::tight(),
    );
    assert!(r.is_clean(), "{}", r.summary());
}

#[test]
fn compare_stages_first_bad_is_first_diverging() {
    // Stage 0 matches, stage 1 diverges — first_bad must be 1.
    let a = cap(&[("s0", vec![1.0; 4]), ("s1", vec![1.0; 4])], 0, "a");
    let mut b1 = vec![1.0; 4];
    b1[0] = 100.0;
    let b = cap(&[("s0", vec![1.0; 4]), ("s1", b1)], 0, "b");
    let r = compare_stages(
        &a,
        &b,
        &[("s0", "s0"), ("s1", "s1")],
        ParityThreshold::tight(),
    );
    assert_eq!(r.first_bad, Some(1));
    assert!(!r.is_clean());
    assert!(r.summary().contains("s1"));
}

#[test]
fn compare_stages_missing_stage_flags_first_bad() {
    let a = cap(&[("s0", vec![1.0])], 0, "a");
    let b = cap(&[("s0", vec![1.0])], 0, "b");
    // Asking for "s1" which neither side has.
    let r = compare_stages(
        &a,
        &b,
        &[("s0", "s0"), ("s1", "s1")],
        ParityThreshold::tight(),
    );
    assert_eq!(r.first_bad, Some(1));
    assert!(r.pairs[1].missing);
}

#[test]
fn compare_stages_supports_asymmetric_names() {
    // CPU's "q_out_after_rope" pairs with Metal's "q_out".
    let a = cap(&[("q_out_after_rope", vec![1.0, 2.0])], 0, "cpu");
    let b = cap(&[("q_out", vec![1.0, 2.0])], 0, "metal");
    let r = compare_stages(
        &a,
        &b,
        &[("q_out_after_rope", "q_out")],
        ParityThreshold::tight(),
    );
    assert!(r.is_clean());
}

// ── stage_stat pure-function tests ────────────────────────────────

#[test]
fn stage_stat_mismatched_lengths_marks_infinite_max_abs() {
    // L429-437: length mismatch returns a sentinel `max_abs=inf` so
    // any threshold-based comparison treats the pair as bad.
    let s = stage_stat(7, &[1.0, 2.0, 3.0], &[1.0, 2.0]);
    assert_eq!(s.layer, 7);
    assert_eq!(s.cos, 0.0);
    assert!(s.max_abs.is_infinite());
    assert_eq!(s.a_norm, 0.0);
    assert_eq!(s.b_norm, 0.0);
}

#[test]
fn stage_stat_identical_vectors_have_cosine_one() {
    let v: Vec<f32> = vec![3.0, 4.0, 0.0];
    let s = stage_stat(0, &v, &v);
    assert!((s.cos - 1.0).abs() < 1e-6, "cos={}", s.cos);
    assert_eq!(s.max_abs, 0.0);
    // ‖v‖ = 5.
    assert!((s.a_norm - 5.0).abs() < 1e-6);
    assert!((s.b_norm - 5.0).abs() < 1e-6);
}

#[test]
fn stage_stat_zero_norm_vectors_have_zero_cosine() {
    // a_sq or b_sq == 0 → cosine = 0 (no division by zero).
    let zero = vec![0.0f32; 4];
    let nonzero = vec![1.0f32, 2.0, 3.0, 4.0];
    let s = stage_stat(0, &zero, &nonzero);
    assert_eq!(s.cos, 0.0);
    assert_eq!(s.a_norm, 0.0);
    assert!(s.b_norm > 0.0);
    let s2 = stage_stat(0, &nonzero, &zero);
    assert_eq!(s2.cos, 0.0);
}

#[test]
fn stage_stat_tracks_pointwise_max_abs_diff() {
    let a = vec![1.0f32, 2.0, 3.0];
    let b = vec![1.0f32, 2.0, 0.5];
    // The pointwise diffs are 0, 0, 2.5 → max_abs = 2.5.
    let s = stage_stat(0, &a, &b);
    assert!((s.max_abs - 2.5).abs() < 1e-6, "max_abs={}", s.max_abs);
}

// ── read_stage_dir / read_f32_vec filesystem tests ────────────────

fn write_f32_file(path: &std::path::Path, vals: &[f32]) {
    let mut bytes = Vec::with_capacity(vals.len() * 4);
    for v in vals {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, &bytes).unwrap();
}

#[test]
fn read_f32_vec_round_trips_little_endian_floats() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("data.f32");
    let vals: Vec<f32> = vec![1.0, -2.5, 3.75, 0.0];
    write_f32_file(&path, &vals);
    let got = read_f32_vec(&path).expect("read");
    assert_eq!(got, vals);
}

#[test]
fn read_f32_vec_returns_none_on_byte_count_not_multiple_of_four() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("data.f32");
    std::fs::write(&path, [0u8, 1, 2]).unwrap();
    assert!(read_f32_vec(&path).is_none());
}

#[test]
fn read_f32_vec_returns_none_on_missing_file() {
    let dir = tempfile::tempdir().unwrap();
    assert!(read_f32_vec(&dir.path().join("nope.f32")).is_none());
}

#[test]
fn read_stage_dir_picks_up_prefixed_f32_files() {
    let dir = tempfile::tempdir().unwrap();
    // Files we want picked up.
    write_f32_file(&dir.path().join("cpu_L03_q_out.f32"), &[1.0, 2.0]);
    write_f32_file(&dir.path().join("cpu_L03_k_out.f32"), &[3.0]);
    // Files that should be skipped: wrong prefix, missing .f32 suffix.
    write_f32_file(&dir.path().join("metal_L03_q_out.f32"), &[9.0]);
    std::fs::write(dir.path().join("cpu_L03_readme.txt"), b"skip").unwrap();
    let got = read_stage_dir(dir.path(), "cpu_L03_").unwrap();
    assert_eq!(got.len(), 2);
    assert_eq!(got.get("q_out"), Some(&vec![1.0, 2.0]));
    assert_eq!(got.get("k_out"), Some(&vec![3.0]));
}

#[test]
fn read_stage_dir_empty_dir_returns_empty_map() {
    let dir = tempfile::tempdir().unwrap();
    let got = read_stage_dir(dir.path(), "anything_").unwrap();
    assert!(got.is_empty());
}

#[test]
fn read_stage_dir_errors_when_dir_missing() {
    let dir = tempfile::tempdir().unwrap();
    let missing = dir.path().join("no_such_subdir");
    let err = read_stage_dir(&missing, "p_").unwrap_err();
    assert!(
        err.contains("read_dir"),
        "expected read_dir error, got: {err}"
    );
}

#[test]
fn read_stage_dir_errors_when_truncated_f32_file_present() {
    // A correctly-named file with a non-multiple-of-4 byte count
    // hits the L514 `return Err(...)` branch.
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("p_truncated.f32"), [1u8, 2, 3]).unwrap();
    let err = read_stage_dir(dir.path(), "p_").unwrap_err();
    assert!(err.contains("could not read f32 file"), "got: {err}");
}

// ── run_with_two_env_vars test ────────────────────────────────────

#[test]
fn run_with_two_env_vars_sets_and_restores_both_flags() {
    use larql_compute::options::{env_value, ScopedEnvOverride};
    const D: &str = "LARQL_TEST_DIR_VAR_RW2EV";
    const L: &str = "LARQL_TEST_LAYER_VAR_RW2EV";
    let _prior_layer = ScopedEnvOverride::set(L, Some("preexisting"));

    let observed_dir = std::cell::Cell::new(String::new());
    let observed_layer = std::cell::Cell::new(String::new());
    let dir = run_with_two_env_vars(D, L, "42", || {
        observed_dir.set(env_value(D).unwrap_or_default());
        observed_layer.set(env_value(L).unwrap_or_default());
        assert!(std::env::var(D).is_err(), "process env must stay untouched");
    })
    .expect("tempdir + run ok");
    // While the closure ran the flags named the tempdir + layer string.
    assert_eq!(observed_dir.into_inner(), dir.path().to_string_lossy());
    assert_eq!(observed_layer.into_inner(), "42");
    // Afterwards: D had no value → none; L had "preexisting" → back.
    assert_eq!(env_value(D), None);
    assert_eq!(env_value(L).as_deref(), Some("preexisting"));
}

// ── StageCapture::len / is_empty / num_layers accessors ──────────────

#[test]
fn stage_capture_len_and_is_empty() {
    let empty = cap(&[], 0, "x");
    assert_eq!(empty.len(), 0);
    assert!(empty.is_empty());
    let one = cap(&[("a", vec![1.0])], 0, "x");
    assert_eq!(one.len(), 1);
    assert!(!one.is_empty());
}

// ── cpu_prefill (full path against Q4K fixture) ──────────────────────

#[test]
fn cpu_prefill_runs_end_to_end_against_q4k_fixture() {
    use crate::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
    let mut weights = make_test_q4k_weights();
    let index = make_test_q4k_vindex(&weights);
    // cpu_prefill drives `predict_kquant_hidden` with the env vars
    // set, then reads back every `cpu_L0_<stage>.f32` file written
    // into the temp dir. Note: the dump config in
    // `crate::forward::dump_config` uses a `OnceLock`-cached env-var
    // read, so other tests may have observed the unset state first
    // — in that case the dump never fires and the `stages` map is
    // empty. We assert the capture *shape* (layer/seq_len/backend
    // labels) without depending on the cache miss.
    let cap = StageCapture::cpu_prefill(&mut weights, &[0u32, 1, 2], &index, 0)
        .expect("cpu_prefill against Q4K fixture must succeed");
    assert_eq!(cap.layer, 0);
    assert_eq!(cap.seq_len, 3);
    assert_eq!(cap.backend, "cpu_prefill");
}

/// Round-trip on `project_to_last_position`: the prefill capture
/// returns `seq_len > 1`, projection slices each stage down to the
/// last position and reports `seq_len=1`.
#[test]
fn cpu_prefill_then_project_to_last_position_returns_single_row_capture() {
    use crate::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
    let mut weights = make_test_q4k_weights();
    let index = make_test_q4k_vindex(&weights);
    let cap = StageCapture::cpu_prefill(&mut weights, &[0u32, 1, 2], &index, 0).unwrap();
    let projected = cap.project_to_last_position();
    assert_eq!(projected.seq_len, 1);
    // Projection produces a fresh map with the same keys (whether
    // or not the dump fired — empty in, empty out).
    assert_eq!(projected.stages.len(), cap.stages.len());
}

/// `metal_prefill` drives the GPU prefill body — needs a Q4-supporting
/// backend. With `MockGpuBackend` the env-var-gated dump never produces
/// per-stage files (Metal-only) but the capture wrapper still runs
/// end-to-end and reports the right `backend` label.
#[test]
fn metal_prefill_runs_end_to_end_with_mock_gpu_backend() {
    use crate::test_utils::{make_test_q4k_vindex, make_test_q4k_weights, MockGpuBackend};
    let mut weights = make_test_q4k_weights();
    let index = make_test_q4k_vindex(&weights);
    let backend = MockGpuBackend::new();
    let cap = StageCapture::metal_prefill(&mut weights, &[0u32, 1], &index, &backend, 0)
        .expect("metal_prefill against mock backend should succeed");
    assert_eq!(cap.layer, 0);
    assert_eq!(cap.seq_len, 2);
    assert_eq!(cap.backend, "metal_prefill");
}

/// `metal_decode` runs prefill + a single decode_token via the
/// mock backend. The mock returns shape-correct zero vectors from
/// both calls so the function reaches the env-var-gated stage dump.
#[test]
fn metal_decode_runs_end_to_end_with_mock_gpu_backend() {
    use crate::test_utils::{make_test_q4k_vindex, make_test_q4k_weights, MockGpuBackend};
    let mut weights = make_test_q4k_weights();
    let index = make_test_q4k_vindex(&weights);
    let backend = MockGpuBackend::new();
    let cap = StageCapture::metal_decode(&mut weights, &[0u32, 1, 2], 3u32, &index, &backend, 0)
        .expect("metal_decode against mock backend should succeed");
    assert_eq!(cap.layer, 0);
    assert_eq!(cap.seq_len, 1);
    assert_eq!(cap.backend, "metal_decode");
}

/// `metal_decode` rejects vindexes with no Q4 FFN mmap — the
/// `q4_ffn.ok_or` guard fires before any backend dispatch.
#[test]
fn metal_decode_errors_when_no_q4_ffn_data() {
    use crate::test_utils::MockGpuBackend;
    let mut weights = crate::test_utils::make_test_q4k_weights();
    let empty_index = larql_vindex::VectorIndex::new(
        vec![None; weights.num_layers],
        vec![None; weights.num_layers],
        weights.num_layers,
        weights.hidden_size,
    );
    let backend = MockGpuBackend::new();
    let result = StageCapture::metal_decode(&mut weights, &[0u32], 1u32, &empty_index, &backend, 0);
    let err = match result {
        Ok(_) => panic!("missing Q4 FFN mmap must error"),
        Err(e) => e,
    };
    assert!(err.contains("Q4"), "error must mention Q4: {err}");
}
