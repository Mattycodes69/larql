use super::*;
use crate::test_utils::make_test_weights;
use ndarray::arr1;
use ndarray::arr2;
use ndarray::Array2;

fn assert_opt_err_contains(tokens: &[u32], target_id: u32, needle: &str) {
    let weights = make_test_weights();
    let err = optimise_target_delta(
        &weights,
        tokens,
        target_id,
        weights.num_layers - 1,
        TargetDeltaOpts {
            steps: 0,
            ..TargetDeltaOpts::default()
        },
    )
    .expect_err("invalid input should return Err");
    assert!(
        err.contains(needle),
        "expected error to contain {needle:?}, got {err:?}"
    );
}

#[test]
fn optimise_target_delta_rejects_empty_prompt() {
    assert_opt_err_contains(&[], 0, "prompt tokens may not be empty");
}

#[test]
fn optimise_target_delta_rejects_target_outside_vocab() {
    let weights = make_test_weights();
    assert_opt_err_contains(&[0], weights.vocab_size as u32, "target_id");
}

#[test]
fn optimise_target_delta_rejects_token_outside_vocab() {
    let weights = make_test_weights();
    assert_opt_err_contains(&[weights.vocab_size as u32], 0, "token id");
}

#[test]
fn optimise_target_delta_rejects_bad_lm_head_shape() {
    let mut weights = make_test_weights();
    weights.lm_head =
        Array2::<f32>::zeros((weights.vocab_size, weights.hidden_size + 1)).into_shared();

    let err = optimise_target_delta(
        &weights,
        &[0],
        0,
        weights.num_layers - 1,
        TargetDeltaOpts::default(),
    )
    .expect_err("bad lm_head shape should return Err");
    assert!(err.contains("lm_head shape"), "{err}");
}

#[test]
fn optimise_target_delta_rejects_bad_final_norm_shape() {
    let mut weights = make_test_weights();
    let key = weights.arch.final_norm_key().to_string();
    weights
        .vectors
        .insert(key, vec![1.0; weights.hidden_size + 1]);

    let err = optimise_target_delta(
        &weights,
        &[0],
        0,
        weights.num_layers - 1,
        TargetDeltaOpts::default(),
    )
    .expect_err("bad final norm shape should return Err");
    assert!(err.contains("final norm weight len"), "{err}");
}

#[test]
fn cross_entropy_and_grad_matches_numerical() {
    // Reference: with logits [1.0, 2.0, 0.5], target=1
    // softmax(logits) ≈ [0.2312, 0.6285, 0.1402]
    // loss = -log(0.6285) ≈ 0.4644
    // dlogits = softmax - onehot(1) = [0.2312, -0.3715, 0.1402]
    let logits = arr1(&[1.0_f32, 2.0, 0.5]);
    let (loss, dlogits) = cross_entropy_and_grad(logits.view(), 1);
    assert!((loss - 0.4644).abs() < 1e-3, "loss {loss}");
    assert!((dlogits[0] - 0.2312).abs() < 1e-3);
    assert!((dlogits[1] - (-0.3715)).abs() < 1e-3);
    assert!((dlogits[2] - 0.1402).abs() < 1e-3);
}

#[test]
fn lm_head_backward_shape_and_values() {
    // embed shape (vocab=3, hidden=4), dlogits (3,) → dh (4,)
    let embed = arr2(&[
        [1.0_f32, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
    ]);
    let dlogits = arr1(&[0.5_f32, -0.3, 0.2]);
    let dh = lm_head_backward(embed.view(), dlogits.view());
    // dh[i] = Σ_v dlogits[v] * embed[v,i]
    //  dh[0] = 0.5*1 + -0.3*0 + 0.2*1 = 0.7
    //  dh[1] = 0.5*0 + -0.3*1 + 0.2*1 = -0.1
    //  dh[2] = 0.2
    //  dh[3] = 0.2
    assert!((dh[0] - 0.7).abs() < 1e-5);
    assert!((dh[1] - (-0.1)).abs() < 1e-5);
    assert!((dh[2] - 0.2).abs() < 1e-5);
    assert!((dh[3] - 0.2).abs() < 1e-5);
}

#[test]
fn gated_ffn_backward_finite_difference() {
    // Small hand-sized case: hidden=3, ffn_dim=4
    let x = arr1(&[0.5_f32, -0.3, 1.0]);
    let gate_w = arr2(&[
        [0.1_f32, -0.2, 0.3],
        [0.4, 0.5, -0.1],
        [-0.3, 0.2, 0.4],
        [0.1, 0.1, -0.2],
    ]);
    let up_w = arr2(&[
        [0.2_f32, 0.1, -0.3],
        [-0.1, 0.4, 0.2],
        [0.3, -0.2, 0.1],
        [0.0, 0.3, 0.2],
    ]);
    let down_w = arr2(&[
        [0.1_f32, 0.2, -0.1, 0.3],
        [0.4, -0.2, 0.1, 0.1],
        [-0.3, 0.1, 0.2, -0.1],
    ]);
    // Forward helper
    let fwd = |xi: &Array1<f32>| -> Array1<f32> {
        let g_pre: Array1<f32> = (0..gate_w.nrows())
            .map(|i| (0..xi.len()).map(|j| gate_w[[i, j]] * xi[j]).sum())
            .collect();
        let u: Array1<f32> = (0..up_w.nrows())
            .map(|i| (0..xi.len()).map(|j| up_w[[i, j]] * xi[j]).sum())
            .collect();
        let g: Array1<f32> = g_pre.map(|&z| z / (1.0 + (-z).exp()));
        let act: Array1<f32> = g.iter().zip(u.iter()).map(|(&a, &b)| a * b).collect();
        (0..down_w.nrows())
            .map(|k| (0..down_w.ncols()).map(|i| down_w[[k, i]] * act[i]).sum())
            .collect()
    };
    // Loss = sum(out) so d_out = ones
    let d_out = Array1::from_elem(3, 1.0_f32);
    let dx_analytical = gated_ffn_backward(
        x.view(),
        gate_w.view(),
        up_w.view(),
        down_w.view(),
        d_out.view(),
    );
    let h = 1e-4_f32;
    for i in 0..x.len() {
        let mut xp = x.clone();
        xp[i] += h;
        let mut xm = x.clone();
        xm[i] -= h;
        let lp: f32 = fwd(&xp).iter().sum();
        let lm: f32 = fwd(&xm).iter().sum();
        let num = (lp - lm) / (2.0 * h);
        let err = (dx_analytical[i] - num).abs();
        assert!(
            err < 1e-2,
            "dx[{i}]: analytical {} vs numerical {num}",
            dx_analytical[i]
        );
    }
}

#[test]
fn target_delta_opts_default_matches_python_reference() {
    let d = TargetDeltaOpts::default();
    assert_eq!(d.steps, 60);
    assert!((d.lr - 0.5).abs() < 1e-6);
    assert!((d.kl_weight - 0.0625).abs() < 1e-6);
    assert!(!d.normalise);
}

#[test]
fn optimise_target_delta_happy_path_returns_finite_delta() {
    // Drive the actual Adam loop with tiny steps so we touch every
    // line of the optimisation body without burning CPU.
    let weights = make_test_weights();
    let opts = TargetDeltaOpts {
        steps: 2,
        lr: 0.1,
        kl_weight: 0.0,
        normalise: false,
    };
    let install_layer = weights.num_layers - 1;
    let result = optimise_target_delta(&weights, &[0u32], 1, install_layer, opts)
        .expect("happy path must succeed");
    assert_eq!(result.layer, install_layer);
    assert_eq!(result.delta.len(), weights.hidden_size);
    assert!(result.delta.iter().all(|v| v.is_finite()));
    assert!(result.final_loss.is_finite());
    assert!(result.baseline_loss.is_finite());
}

#[test]
fn optimise_target_delta_with_kl_weight_runs_kl_branch() {
    // kl_weight != 0.0 → triggers the `dlogits += kl * (cur - base)`
    // and `kl_val` accumulation branches.
    let weights = make_test_weights();
    let opts = TargetDeltaOpts {
        steps: 1,
        lr: 0.1,
        kl_weight: 0.05,
        normalise: false,
    };
    let install_layer = weights.num_layers - 1;
    let result = optimise_target_delta(&weights, &[0u32], 1, install_layer, opts)
        .expect("kl-weighted opt must succeed");
    assert!(result.final_loss.is_finite());
}

#[test]
fn optimise_target_delta_with_normalise_returns_unit_norm_delta() {
    let weights = make_test_weights();
    let opts = TargetDeltaOpts {
        steps: 2,
        lr: 0.5,
        kl_weight: 0.0,
        normalise: true,
    };
    let install_layer = weights.num_layers - 1;
    let result = optimise_target_delta(&weights, &[0u32, 1], 2, install_layer, opts)
        .expect("normalise must succeed");
    let norm: f32 = result.delta.iter().map(|x| x * x).sum::<f32>().sqrt();
    // Either zero (degenerate optimization) or unit norm.
    assert!(
        norm == 0.0 || (norm - 1.0).abs() < 1e-4,
        "normalised delta should have unit norm or be zero, got {norm}"
    );
}

#[test]
fn optimise_target_delta_rejects_mid_layer() {
    let weights = make_test_weights();
    // num_layers=2 in test fixture, so install_layer=0 is mid-layer.
    let err = optimise_target_delta(&weights, &[0u32], 0, 0, TargetDeltaOpts::default())
        .expect_err("mid-layer install must error");
    assert!(
        err.contains("only install_layer = n_layers-1"),
        "expected mid-layer rejection, got: {err}"
    );
}

#[test]
fn optimise_target_delta_rejects_layer_out_of_range() {
    let weights = make_test_weights();
    let err = optimise_target_delta(
        &weights,
        &[0u32],
        0,
        weights.num_layers + 5,
        TargetDeltaOpts::default(),
    )
    .expect_err("install_layer >= n_layers must error");
    assert!(err.contains("≥ n_layers"), "got: {err}");
}

#[test]
fn softmax_1d_sums_to_one_and_handles_extreme_logits() {
    // Tiny + huge logits exercise the numerical-stability shift.
    let logits = arr1(&[100.0f32, 100.5, 99.0, -1000.0]);
    let probs = softmax_1d(&logits);
    let sum: f32 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "softmax sum: {sum}");
    // The smallest logit should produce ~0 probability.
    assert!(probs[3] < 1e-30 || probs[3] == 0.0);
    // Max-logit index should hold the largest probability.
    assert!(probs[1] > probs[0]);
    assert!(probs[1] > probs[2]);
}

#[test]
fn rmsnorm_backward_finite_difference() {
    // Analytical gradient should match numerical at a random point.
    let x = arr1(&[0.5_f32, 1.0, -0.5, 2.0]);
    let w = arr1(&[1.0_f32, 0.5, 2.0, 1.5]);
    let eps = 1e-5_f32;

    // Forward helper
    let fwd = |xi: &Array1<f32>| -> Array1<f32> {
        let d = xi.len() as f32;
        let ms = xi.iter().map(|v| v * v).sum::<f32>() / d;
        let rms = (ms + eps).sqrt();
        xi.iter()
            .zip(w.iter())
            .map(|(xv, wv)| (xv / rms) * wv)
            .collect()
    };

    // Loss = sum of y (so dy = ones)
    let dy = Array1::from_elem(x.len(), 1.0_f32);
    let dx_analytical = rmsnorm_backward_pos(x.view(), w.view(), dy.view(), eps);

    // Numerical dx via finite difference
    let h = 1e-4_f32;
    for i in 0..x.len() {
        let mut xp = x.clone();
        xp[i] += h;
        let mut xm = x.clone();
        xm[i] -= h;
        let loss_p: f32 = fwd(&xp).iter().sum();
        let loss_m: f32 = fwd(&xm).iter().sum();
        let num = (loss_p - loss_m) / (2.0 * h);
        let err = (dx_analytical[i] - num).abs();
        assert!(
            err < 1e-2,
            "dx[{i}]: analytical {} vs numerical {num} (err {err})",
            dx_analytical[i]
        );
    }
}
