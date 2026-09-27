//! End-to-end `oracle-pq` runs on a synthetic Q4K vindex: every probe
//! family switched on at once, and every flag-validation refusal.

mod errors;
mod fixture;
mod lookups;
mod specs;

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use clap::Parser;

use super::{run_oracle_pq, OraclePqArgs};

#[derive(Parser)]
struct Harness {
    #[command(flatten)]
    args: OraclePqArgs,
}

/// One fixture per test binary: `(root, vindex, prompts)`. Tests only read
/// it; each run writes under its own output directory.
fn shared_fixture() -> &'static (tempfile::TempDir, PathBuf, PathBuf) {
    static FIXTURE: OnceLock<(tempfile::TempDir, PathBuf, PathBuf)> = OnceLock::new();
    FIXTURE.get_or_init(|| {
        let root = tempfile::tempdir().unwrap();
        let (vindex, prompts) = fixture::build(root.path());
        (root, vindex, prompts)
    })
}

/// Parse `oracle-pq` flags against the shared fixture and run it,
/// writing into `out`. Unless `extra` sets them, the fixture's prompts, two
/// heads (one per layer) and one 4-bit two-group config are selected.
fn run_with(out: &Path, extra: &[&str]) -> Result<(), Box<dyn std::error::Error>> {
    let (_, vindex, prompts) = shared_fixture();
    let prompts = prompts.display().to_string();
    let mut argv = vec![
        "oracle-pq".to_string(),
        "--index".to_string(),
        vindex.display().to_string(),
        "--out".to_string(),
        out.display().to_string(),
    ];
    for (flag, default) in [
        ("--prompts", prompts.as_str()),
        ("--heads", "0:1,1:0"),
        ("--configs", "8:2:4"),
    ] {
        if !extra.contains(&flag) {
            argv.extend([flag.to_string(), default.to_string()]);
        }
    }
    argv.extend(extra.iter().map(|s| s.to_string()));
    let harness = Harness::try_parse_from(argv)?;
    run_oracle_pq(harness.args)
}

/// Every probe family enabled, with values sized to the fixture; the
/// held-out split puts one prompt from each stratum in eval.
const ALL_PROBES: &[&str] = &[
    "--pq-iters",
    "6",
    "--eval-mod",
    "4",
    "--mode-d-check",
    "--address-probes",
    "--address-mixed-key-probe",
    "--address-key-group-probe",
    "--address-majority-group-probe",
    "--address-code-substitution-group-probe",
    "--address-code-substitution-from-codes",
    "0,6,7,10",
    "--address-code-substitution-to-codes",
    "majority,3",
    "--address-code-class-collapse-group-probe",
    "--address-code-class-collapse-specs",
    "c1=6+10:13|7:10;c2=0+1:2",
    "--address-code-position-interaction-probe",
    "--address-code-position-prompt-id",
    "p0",
    "--address-code-position-primary-codes",
    "0,1,2,3,4,5,6,7",
    "--address-code-position-secondary-codes",
    "8,9,10,11,12,14,15",
    "--address-code-conditional-quotient-group-probe",
    "--address-code-conditional-quotient-extra-specs",
    "x=4:13",
    "--address-code-occurrences",
    "--address-code-occurrence-groups",
    "0,1",
    "--address-code7-bos-rule-group-probe",
    "--address-code7-oracle-binary-group-probe",
    "--address-corruption-sweep",
    "--address-group-importance",
    "--address-lsh-group-probe",
    "--address-lsh-seeds",
    "4",
    "--address-supervised-group-probe",
    "--address-supervised-epochs",
    "4",
    "--address-gamma-projected-group-probe",
    "--address-gamma-projected-layers",
    "1",
    "--address-gamma-random-ranks",
    "4",
    "--address-gamma-learned-ranks",
    "4",
    "--address-gamma-learned-epochs",
    "2",
    "--address-gamma-learned-pca-iters",
    "2",
    "--address-code-stability",
    "--address-code-stability-groups",
    "0,1",
    "--address-prev-ffn-feature-group-probe",
    "--address-ffn-first-feature-group-probe",
    "--address-attention-relation-group-probe",
    "--address-attention-cluster-group-probe",
    "--address-attention-cluster-ks",
    "2,4",
    "--address-reduced-qk-cluster-group-probe",
    "--address-reduced-qk-ranks",
    "0,8",
    "--address-reduced-qk-cluster-ks",
    "2",
    "--stratum-conditioned-pq-groups",
    "1",
];

/// Probe-name prefixes, one per family the registry evaluates. A family
/// that drops out of the registry leaves its prefix unmatched.
const FAMILY_PREFIXES: &[&str] = &[
    "token_id",                      // --address-probes
    "mixed_best_simple_key",         // --address-mixed-key-probe
    "token_id_groups_[0]_",          // --address-key-group-probe
    "majority_groups_",              // majority
    "code_subst_g0_",                // code substitution
    "code_class_collapse_c1_",       // class collapse
    "pos_interaction_g0_",           // position interaction
    "code_conditional_quotient_g0_", // conditional quotient
    "code7_bos_non_arithmetic_",     // code7 BOS rule
    "oracle_binary_all_code7_",      // code7 oracle binary
    "lsh_groups_",                   // LSH
    "supervised_hyperplane_groups_", // supervised
    "gamma_raw_groups_",             // gamma projected
    "random_rank4_seed0_groups_",    // gamma random bridge
    "gamma_learned_post_l1_rank4_",  // gamma learned bridge
    "prev_ffn_top1_groups_",         // previous-FFN features
    "ffn_first_top1_groups_",        // FFN-first features
    "attn_argmax_groups_",           // attention relation
    "attn_cluster_2_groups_",        // attention clusters
    "qk_rank8_attn_cluster_2_",      // reduced-QK clusters
];

#[test]
fn every_probe_family_reports_on_a_synthetic_q4k_vindex() {
    let out = tempfile::tempdir().unwrap();
    run_with(out.path(), ALL_PROBES).unwrap();
    let text = std::fs::read_to_string(out.path().join("oracle_pq.json")).unwrap();
    let report: serde_json::Value = serde_json::from_str(&text).unwrap();

    assert_eq!(report["prompts_seen"], 12);
    assert_eq!(report["eval_prompts_seen"], 3);
    assert_eq!(report["address_lsh_groups"], serde_json::json!([0]));
    assert_eq!(
        report["stratum_conditioned_pq_groups"],
        serde_json::json!([1])
    );
    let heads = report["heads"].as_array().unwrap();
    assert_eq!(heads.len(), 2);
    for head in heads {
        let point = &head["points"][0];
        assert_eq!(point["per_prompt"].as_array().unwrap().len(), 3);
        assert_eq!(point["code_stability"].as_array().unwrap().len(), 2);
        assert_eq!(
            point["address_group_importance"].as_array().unwrap().len(),
            2
        );
        assert!(!point["address_corruption_sweep"]
            .as_array()
            .unwrap()
            .is_empty());
        assert!(point["mode_d_mean_kl"].is_number());
        let names: Vec<&str> = point["address_probes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|probe| probe["name"].as_str().unwrap())
            .collect();
        for prefix in FAMILY_PREFIXES {
            assert!(
                names.iter().any(|name| name.starts_with(prefix)),
                "L{}H{}: no probe named {prefix}*",
                head["layer"],
                head["head"]
            );
        }
    }
    let occurrences = std::fs::read_to_string(out.path().join("code_occurrences.json")).unwrap();
    let occurrences: serde_json::Value = serde_json::from_str(&occurrences).unwrap();
    assert!(!occurrences.as_array().unwrap().is_empty());
}

#[test]
fn disabled_probes_leave_their_settings_at_zero_in_the_report() {
    let out = tempfile::tempdir().unwrap();
    run_with(out.path(), &["--pq-iters", "2", "--eval-mod", "4"]).unwrap();
    let text = std::fs::read_to_string(out.path().join("oracle_pq.json")).unwrap();
    let report: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(report["mode_d_check"], false);
    assert_eq!(report["address_key_groups"], serde_json::json!([]));
    assert_eq!(report["address_gamma_learned_epochs"], 0);
    // Always-recorded settings keep their flag defaults.
    assert_eq!(report["address_lsh_bits"], 4);
    let point = &report["heads"][0]["points"][0];
    assert!(point["mode_d_mean_kl"].is_null());
    assert!(point["address_probes"].is_null());
    assert!(!out.path().join("code_occurrences.json").exists());
}
