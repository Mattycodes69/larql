//! generate_cached_hooked

use super::*;

// The unhooked and hooked decode paths are mathematically equivalent
// under NoopHook, but BLAS reduction order can drift call-to-call on
// Windows OpenBLAS — observed argmax flipping after the first decode
// step. Linux/macOS BLAS implementations are bit-stable enough for
// this assertion to hold, so we keep the coverage there.
#[cfg(not(windows))]
#[test]
fn generate_cached_hooked_with_noop_matches_baseline() {
    let weights = make_test_weights();
    let tokenizer = make_test_tokenizer(weights.vocab_size);
    let ffn = WeightFfn { weights: &weights };

    let baseline = generate_cached(&weights, &tokenizer, &ffn, &[0u32, 1, 2], 4, |_, _| {});

    let hooked = generate_cached_hooked(
        &weights,
        &tokenizer,
        &ffn,
        &[0u32, 1, 2],
        4,
        None,
        None,
        &mut NoopHook,
        |_, _| {},
    );

    assert_eq!(baseline, hooked, "noop hook must not change generated ids");
}

#[test]
fn generate_cached_hooked_record_fires_during_prefill_and_decode() {
    struct CountHook {
        calls: std::collections::HashMap<usize, usize>,
    }
    impl LayerHook for CountHook {
        fn on_post_layer(&mut self, layer: usize, _h: &mut Array2<f32>) {
            *self.calls.entry(layer).or_insert(0) += 1;
        }
    }

    let weights = make_test_weights();
    let tokenizer = make_test_tokenizer(weights.vocab_size);
    let ffn = WeightFfn { weights: &weights };
    let max_new = 3usize;
    let mut hook = CountHook {
        calls: std::collections::HashMap::new(),
    };

    let _ = generate_cached_hooked(
        &weights,
        &tokenizer,
        &ffn,
        &[0u32, 1],
        max_new,
        None,
        None,
        &mut hook,
        |_, _| {},
    );

    for layer in 0..weights.num_layers {
        let count = *hook.calls.get(&layer).unwrap_or(&0);
        assert!(
            count >= 1,
            "hook should fire at least once per layer (got {count} for layer {layer})"
        );
        assert!(
            count <= max_new,
            "hook fires at most max_new times per layer (got {count} for layer {layer})"
        );
    }
}

#[test]
fn generate_cached_hooked_steer_changes_output() {
    use larql_inference::forward::SteerHook;
    use ndarray::Array1;

    let weights = make_test_weights();
    let tokenizer = make_test_tokenizer(weights.vocab_size);
    let ffn = WeightFfn { weights: &weights };
    let prompt = vec![1u32, 2, 3];

    let baseline = generate_cached(&weights, &tokenizer, &ffn, &prompt, 4, |_, _| {});

    let v = Array1::from_vec(
        (0..weights.hidden_size)
            .map(|i| (i as f32 + 1.0) * 0.1)
            .collect(),
    );
    let mut steer = SteerHook::new().add(0, v, 5.0);

    let steered = generate_cached_hooked(
        &weights,
        &tokenizer,
        &ffn,
        &prompt,
        4,
        None,
        None,
        &mut steer,
        |_, _| {},
    );

    if !baseline.is_empty() && !steered.is_empty() {
        assert_ne!(
            baseline, steered,
            "steering with α=5 must change generated tokens"
        );
    }
}
