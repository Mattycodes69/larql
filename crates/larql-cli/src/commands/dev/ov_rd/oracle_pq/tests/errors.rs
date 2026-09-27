//! Every flag refusal, as `(flags, fragment of the error)`. Validation runs
//! right after the vindex loads, so these cases are cheap; the Mode D
//! prerequisite cases also run the base fit.

use super::run_with;

/// A group index the fixture's two-group config does not have.
const BAD_GROUP: &str = "5";
/// A code a 4-bit group cannot express.
const BAD_CODE: &str = "99";

#[rustfmt::skip]
const CASES: &[(&[&str], &str)] = &[
    (&["--heads", ""], "no heads selected"),
    (&["--configs", ""], "no PQ configs selected"),
    // key group
    (&["--address-key-group-probe", "--address-key-groups", ""], "requires at least one --address-key-groups"),
    (&["--address-key-group-probe", "--address-key-groups", BAD_GROUP], "--address-key-groups includes group 5"),
    // majority
    (&["--address-majority-group-probe", "--address-majority-groups", ""], "requires at least one --address-majority-groups"),
    (&["--address-majority-group-probe", "--address-majority-groups", BAD_GROUP], "--address-majority-groups includes group 5"),
    // code substitution
    (&["--address-code-substitution-group-probe", "--address-code-substitution-groups", ""], "requires at least one --address-code-substitution-groups"),
    (&["--address-code-substitution-group-probe", "--address-code-substitution-to-codes", ""], "requires at least one --address-code-substitution-to-codes"),
    (&["--address-code-substitution-group-probe", "--address-code-substitution-groups", BAD_GROUP], "--address-code-substitution-groups includes group 5"),
    (&["--address-code-substitution-group-probe", "--address-code-substitution-from-codes", BAD_CODE], "--address-code-substitution-from-codes includes code 99"),
    (&["--address-code-substitution-group-probe", "--address-code-substitution-to-codes", BAD_CODE], "--address-code-substitution-to-codes includes code 99"),
    // class collapse
    (&["--address-code-class-collapse-group-probe", "--address-code-class-collapse-groups", "", "--address-code-class-collapse-specs", "a=1:2"], "requires at least one --address-code-class-collapse-groups"),
    (&["--address-code-class-collapse-group-probe"], "--address-code-class-collapse-specs must include at least one spec"),
    (&["--address-code-class-collapse-group-probe", "--address-code-class-collapse-groups", BAD_GROUP, "--address-code-class-collapse-specs", "a=1:2"], "--address-code-class-collapse-groups includes group 5"),
    (&["--address-code-class-collapse-group-probe", "--address-code-class-collapse-specs", "a=1:99"], "class-collapse spec \"a\" targets code 99"),
    (&["--address-code-class-collapse-group-probe", "--address-code-class-collapse-specs", "a=99:1"], "class-collapse spec \"a\" includes source code 99"),
    // position interaction
    (&["--address-code-position-interaction-probe"], "requires --address-code-position-prompt-id"),
    (&["--address-code-position-interaction-probe", "--address-code-position-prompt-id", "p0", "--address-code-position-primary-codes", ""], "--address-code-position-primary-codes must include"),
    (&["--address-code-position-interaction-probe", "--address-code-position-prompt-id", "p0", "--address-code-position-secondary-codes", ""], "--address-code-position-secondary-codes must include"),
    (&["--address-code-position-interaction-probe", "--address-code-position-prompt-id", "p0", "--address-code-position-group", BAD_GROUP], "--address-code-position-group is 5"),
    (&["--address-code-position-interaction-probe", "--address-code-position-prompt-id", "p0", "--address-code-position-target-code", BAD_CODE], "--address-code-position-target-code is 99"),
    (&["--address-code-position-interaction-probe", "--address-code-position-prompt-id", "p0", "--address-code-position-primary-codes", BAD_CODE], "primary/secondary code 99 exceeds"),
    // conditional quotient
    (&["--address-code-conditional-quotient-group-probe", "--address-code-conditional-quotient-primary-codes", ""], "--address-code-conditional-quotient-primary-codes must include"),
    (&["--address-code-conditional-quotient-group-probe", "--address-code-conditional-quotient-secondary-codes", ""], "--address-code-conditional-quotient-secondary-codes must include"),
    (&["--address-code-conditional-quotient-group-probe", "--address-code-conditional-quotient-guards", ""], "--address-code-conditional-quotient-guards must include"),
    (&["--address-code-conditional-quotient-group-probe", "--address-code-conditional-quotient-group", BAD_GROUP], "--address-code-conditional-quotient-group is 5"),
    (&["--address-code-conditional-quotient-group-probe", "--address-code-conditional-quotient-target-code", BAD_CODE], "--address-code-conditional-quotient-target-code is 99"),
    (&["--address-code-conditional-quotient-group-probe", "--address-code-conditional-quotient-primary-codes", BAD_CODE], "conditional-quotient primary/secondary code 99"),
    (&["--address-code-conditional-quotient-group-probe", "--address-code-conditional-quotient-extra-specs", "x=1:99"], "conditional quotient extra spec \"x\" targets code 99"),
    // code occurrences
    (&["--address-code-occurrences", "--address-code-occurrence-groups", ""], "requires at least one --address-code-occurrence-groups"),
    (&["--address-code-occurrences", "--address-code-occurrence-split", "test"], "must be train, eval, or all"),
    (&["--address-code-occurrences", "--address-code-occurrence-groups", BAD_GROUP], "--address-code-occurrence-groups includes group 5"),
    (&["--address-code-occurrences", "--address-code-occurrence-codes", BAD_CODE], "--address-code-occurrence-codes includes code 99"),
    // code7 BOS rule
    (&["--address-code7-bos-rule-group-probe", "--address-code7-bos-rule-groups", ""], "requires at least one --address-code7-bos-rule-groups"),
    (&["--address-code7-bos-rule-group-probe", "--address-code7-bos-rule-groups", BAD_GROUP], "--address-code7-bos-rule-groups includes group 5"),
    (&["--address-code7-bos-rule-group-probe", "--address-code7-bos-rule-code", BAD_CODE], "--address-code7-bos-rule-code is 99"),
    // code7 oracle binary
    (&["--address-code7-oracle-binary-group-probe", "--address-code7-oracle-binary-groups", ""], "requires at least one --address-code7-oracle-binary-groups"),
    (&["--address-code7-oracle-binary-group-probe", "--address-code7-oracle-binary-filters", ""], "--address-code7-oracle-binary-filters must include"),
    (&["--address-code7-oracle-binary-group-probe", "--address-code7-oracle-binary-filters", "bos"], "unsupported --address-code7-oracle-binary-filters value \"bos\""),
    (&["--address-code7-oracle-binary-group-probe", "--address-code7-oracle-binary-groups", BAD_GROUP], "--address-code7-oracle-binary-groups includes group 5"),
    (&["--address-code7-oracle-binary-group-probe", "--address-code7-oracle-binary-code", BAD_CODE], "--address-code7-oracle-binary-code is 99"),
    // LSH
    (&["--address-lsh-group-probe", "--address-lsh-groups", ""], "requires at least one --address-lsh-groups"),
    (&["--address-lsh-group-probe", "--address-lsh-bits", "0"], "--address-lsh-bits must be greater than zero"),
    (&["--address-lsh-group-probe", "--address-lsh-bits", "17"], "--address-lsh-bits is capped at 16"),
    (&["--address-lsh-group-probe", "--address-lsh-seeds", "0"], "--address-lsh-seeds must be greater than zero"),
    (&["--address-lsh-group-probe", "--address-lsh-groups", BAD_GROUP], "--address-lsh-groups includes group 5"),
    // supervised
    (&["--address-supervised-group-probe", "--address-supervised-groups", ""], "requires at least one --address-supervised-groups"),
    (&["--address-supervised-group-probe", "--address-supervised-epochs", "0"], "--address-supervised-epochs must be greater than zero"),
    (&["--address-supervised-group-probe", "--address-supervised-lr", "0"], "--address-supervised-lr must be greater than zero"),
    (&["--address-supervised-group-probe", "--address-supervised-l2=-1"], "--address-supervised-l2 must be non-negative"),
    (&["--address-supervised-group-probe", "--address-supervised-groups", BAD_GROUP], "--address-supervised-groups includes group 5"),
    // gamma projected
    (&["--address-gamma-projected-group-probe", "--address-gamma-projected-groups", ""], "requires at least one --address-gamma-projected-groups"),
    (&["--address-gamma-projected-group-probe", "--address-gamma-projected-layers", ""], "must include at least one value"),
    (&["--address-gamma-projected-group-probe", "--address-gamma-projected-layers", "", "--address-gamma-learned-ranks", "4"], "--address-gamma-learned-ranks requires at least one --address-gamma-projected-layers"),
    (&["--address-gamma-projected-group-probe", "--address-gamma-projected-layers", "9"], "includes layer 9, but the model has only 2 layers"),
    (&["--address-gamma-projected-group-probe", "--address-gamma-projected-layers", "0"], "includes post-L0, before target L1H0"),
    (&["--address-gamma-projected-group-probe", "--address-gamma-projected-layers", "1", "--address-gamma-random-ranks", "0"], "--address-gamma-random-ranks includes rank 0"),
    (&["--address-gamma-projected-group-probe", "--address-gamma-projected-layers", "1", "--address-gamma-random-ranks", "4", "--address-gamma-random-seeds", ""], "--address-gamma-random-seeds must include at least one seed"),
    (&["--address-gamma-projected-group-probe", "--address-gamma-projected-layers", "1", "--address-gamma-learned-ranks", "999"], "--address-gamma-learned-ranks includes rank 999"),
    (&["--address-gamma-projected-group-probe", "--address-gamma-projected-layers", "1", "--address-gamma-learned-epochs", "0"], "--address-gamma-learned-epochs must be greater than zero"),
    (&["--address-gamma-projected-group-probe", "--address-gamma-projected-layers", "1", "--address-gamma-learned-lr", "0"], "--address-gamma-learned-lr must be greater than zero"),
    (&["--address-gamma-projected-group-probe", "--address-gamma-projected-layers", "1", "--address-gamma-learned-l2=-1"], "--address-gamma-learned-l2 must be non-negative"),
    (&["--address-gamma-projected-group-probe", "--address-gamma-projected-layers", "1", "--address-gamma-learned-pca-iters", "0"], "--address-gamma-learned-pca-iters must be greater than zero"),
    (&["--address-gamma-projected-group-probe", "--address-gamma-projected-layers", "1", "--address-gamma-projected-groups", BAD_GROUP], "--address-gamma-projected-groups includes group 5"),
    // code stability
    (&["--address-code-stability", "--address-code-stability-groups", ""], "requires at least one --address-code-stability-groups"),
    (&["--address-code-stability", "--address-code-stability-groups", BAD_GROUP], "--address-code-stability-groups includes group 5"),
    // previous-FFN features
    (&["--address-prev-ffn-feature-group-probe", "--address-prev-ffn-feature-groups", ""], "requires at least one --address-prev-ffn-feature-groups"),
    (&["--address-prev-ffn-feature-group-probe", "--address-prev-ffn-feature-top-k", "0"], "--address-prev-ffn-feature-top-k must be greater than zero"),
    (&["--address-prev-ffn-feature-group-probe", "--address-prev-ffn-feature-groups", BAD_GROUP], "--address-prev-ffn-feature-groups includes group 5"),
    // FFN-first features
    (&["--address-ffn-first-feature-group-probe", "--address-ffn-first-feature-groups", ""], "requires at least one --address-ffn-first-feature-groups"),
    (&["--address-ffn-first-feature-group-probe", "--address-ffn-first-feature-top-k", "0"], "--address-ffn-first-feature-top-k must be greater than zero"),
    (&["--address-ffn-first-feature-group-probe", "--address-ffn-first-feature-groups", BAD_GROUP], "--address-ffn-first-feature-groups includes group 5"),
    // attention relation
    (&["--address-attention-relation-group-probe", "--address-attention-relation-groups", ""], "requires at least one --address-attention-relation-groups"),
    (&["--address-attention-relation-group-probe", "--address-attention-relation-groups", BAD_GROUP], "--address-attention-relation-groups includes group 5"),
    // attention clusters
    (&["--address-attention-cluster-group-probe", "--address-attention-cluster-groups", ""], "requires at least one --address-attention-cluster-groups"),
    (&["--address-attention-cluster-group-probe", "--address-attention-cluster-ks", ""], "--address-attention-cluster-ks must include at least one k"),
    (&["--address-attention-cluster-group-probe", "--address-attention-cluster-ks", "1"], "--address-attention-cluster-ks values must be between 2 and 128"),
    (&["--address-attention-cluster-group-probe", "--address-attention-cluster-groups", BAD_GROUP], "--address-attention-cluster-groups includes group 5"),
    // reduced-QK clusters
    (&["--address-reduced-qk-cluster-group-probe", "--address-reduced-qk-cluster-groups", ""], "requires at least one --address-reduced-qk-cluster-groups"),
    (&["--address-reduced-qk-cluster-group-probe", "--address-reduced-qk-ranks", ""], "--address-reduced-qk-ranks must include at least one rank"),
    (&["--address-reduced-qk-cluster-group-probe", "--address-reduced-qk-cluster-ks", ""], "--address-reduced-qk-cluster-ks must include at least one k"),
    (&["--address-reduced-qk-cluster-group-probe", "--address-reduced-qk-cluster-ks", "200"], "--address-reduced-qk-cluster-ks values must be between 2 and 128"),
    (&["--address-reduced-qk-cluster-group-probe", "--address-reduced-qk-cluster-groups", BAD_GROUP], "--address-reduced-qk-cluster-groups includes group 5"),
    // stratum-conditioned codebooks
    (&["--stratum-conditioned-pq-groups", BAD_GROUP], "--stratum-conditioned-pq-groups includes group 5"),
];

/// Mode D prerequisites surface after the base fit: a fitted family
/// refuses before its fit, the unfitted families in their historical order
/// (corruption before majority).
#[rustfmt::skip]
const MODE_D_CASES: &[(&[&str], &str)] = &[
    (&["--address-probes"], "--address-probes/--address-mixed-key-probe requires --mode-d-check"),
    (&["--address-majority-group-probe", "--address-corruption-sweep"], "--address-corruption-sweep requires --mode-d-check"),
];

fn assert_refused(cases: &[(&[&str], &str)]) {
    let out = tempfile::tempdir().unwrap();
    let mut failures = Vec::new();
    for (flags, fragment) in cases {
        let mut argv = vec!["--pq-iters", "2", "--eval-mod", "4"];
        argv.extend_from_slice(flags);
        match run_with(out.path(), &argv) {
            Ok(()) => failures.push(format!("{flags:?}: accepted, expected {fragment:?}")),
            Err(err) if !err.to_string().contains(fragment) => {
                failures.push(format!("{flags:?}: {err}, expected {fragment:?}"))
            }
            Err(_) => {}
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn every_invalid_probe_setting_is_refused() {
    assert_refused(CASES);
}

#[test]
fn mode_d_prerequisites_are_refused_in_order() {
    assert_refused(MODE_D_CASES);
}
