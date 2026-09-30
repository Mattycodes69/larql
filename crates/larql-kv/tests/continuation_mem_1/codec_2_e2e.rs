//! CONTINUATION-CODEC-2 W2 erratum: the run module EXECUTED end to end on
//! a fixture ladder — every arm, the checkpoint names, a resume, the pooled
//! record — and the assembly shown to reproduce the run's own record from
//! W2's as-run layout. W1 controlled the statistics but never executed
//! this glue; W2 lost its record to it (notes, W2).

use std::collections::BTreeMap;
use std::path::Path;

use sha2::{Digest, Sha256};

use super::codec_2_arms::{record, rescored_c, run_arms, strata, Ladder};
use super::codec_2_run::verify_inputs;
use super::codec_2_store::{logits_base, scores_name, w2_as_run_logits_base, Checkpoints};
use super::*;

/// Three rungs of one bootstrap block each, over in-vocabulary ids.
fn ladder() -> Ladder {
    Ladder {
        rungs: vec![40, 48, 64],
        resume: 4,
        decode: 32,
    }
}

fn ids() -> Vec<u32> {
    (0..80).map(|i| ((i * 7 + 3) % 29) as u32).collect()
}

/// Run every arm on fixtures into `dir`; returns the store and its stamp.
fn run_fixture(dir: &Path) -> (Checkpoints, Value) {
    let exact = subjects::fixture(miniature_glimmer, "codec2-e2e-exact");
    let yard = subjects::fixture(miniature_glimmer, "codec2-e2e-yard");
    let backend = ReferenceBackend::new();
    let (exact_ops, yard_ops) = (exact.prepare(&backend), yard.prepare(&backend));
    let stamp = json!({"implementation_git_sha": "fixture", "binary_sha256": "fixture"});
    let store = Checkpoints::open(dir, &stamp);
    run_arms(
        (&exact, &exact_ops),
        (&yard, &yard_ops),
        &backend,
        &ids(),
        &ladder(),
        &store,
    );
    (store, stamp)
}

fn snapshot(store: &Checkpoints) -> BTreeMap<String, String> {
    store
        .files()
        .into_iter()
        .map(|n| {
            let hash = format!(
                "{:x}",
                Sha256::digest(std::fs::read(store.path(&n)).unwrap())
            );
            (n, hash)
        })
        .collect()
}

#[test]
fn the_run_module_runs_resumes_and_pools_end_to_end() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let (store, stamp) = run_fixture(dir.path());

    let mut expected = vec!["authorities.json".to_string()];
    for n in ladder().rungs {
        for arm in ["null", "c", "ch", "h"] {
            expected.push(scores_name(n, arm));
        }
        for arm in ["r", "c"] {
            expected.push(format!("{}.f32", logits_base(n, arm)));
            expected.push(format!("{}.json", logits_base(n, arm)));
        }
    }
    expected.sort();
    assert_eq!(
        store.files(),
        expected,
        "one file per (rung, arm, kind), none shared"
    );

    let c = strata(&store, &ladder(), "c", "positions");
    assert!(
        c.iter().all(|s| s.len() == ladder().decode),
        "C scored at every rung"
    );
    let rec = record("fixture", &stamp, &store, &ladder(), &c);
    assert!(
        rec["verdict_ch"]["verdict"].is_string(),
        "{}",
        rec["verdict_ch"]
    );
    assert!(rec["reported"]["c_ch_kl"].is_object());

    let before = snapshot(&store);
    let (resumed, _) = run_fixture(dir.path());
    assert_eq!(
        snapshot(&resumed),
        before,
        "a complete store resumes without rerunning an arm"
    );
}

/// Rewrite a run's store into W2's as-run layout, including the collision:
/// C's logits index takes the name of C's scores.
fn as_w2_layout(from: &Checkpoints, to: &Path) -> Checkpoints {
    let copy = |a: &str, b: &str| std::fs::copy(from.path(a), to.join(b)).unwrap();
    copy("authorities.json", "authorities.json");
    for n in ladder().rungs {
        for arm in ["null", "ch", "h"] {
            copy(&scores_name(n, arm), &scores_name(n, arm));
        }
        for arm in ["r", "c"] {
            for ext in ["f32", "json"] {
                copy(
                    &format!("{}.{ext}", logits_base(n, arm)),
                    &format!("{}.{ext}", w2_as_run_logits_base(n, arm)),
                );
            }
        }
    }
    Checkpoints::open_as_run(to).0
}

#[test]
fn the_assembly_from_w2s_layout_reproduces_the_runs_own_record() {
    let _serial = serial();
    let (a_dir, b_dir) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let (a, stamp) = run_fixture(a_dir.path());
    let b = as_w2_layout(&a, b_dir.path());
    for n in ladder().rungs {
        assert_eq!(
            scores_name(n, "c"),
            format!("{}.json", w2_as_run_logits_base(n, "c")),
            "the as-run layout reproduces W2's collision"
        );
    }

    let c_run = strata(&a, &ladder(), "c", "positions");
    let c_assembled = rescored_c(&b, &ladder(), &ids(), w2_as_run_logits_base);
    assert_eq!(
        c_assembled, c_run,
        "re-scoring the kept logits is exactly the run's scoring"
    );
    let (ra, rb) = (
        record("fixture", &stamp, &a, &ladder(), &c_run),
        record("fixture", &stamp, &b, &ladder(), &c_assembled),
    );
    assert_eq!(ra["verdict_ch"], rb["verdict_ch"]);
    assert_eq!(ra["reported"], rb["reported"]);

    let list: String = snapshot(&b)
        .iter()
        .map(|(n, h)| format!("{h}  checkpoints/{n}\n"))
        .collect();
    verify_inputs(&b, &list);
    std::fs::write(b.path("authorities.json"), "{}").unwrap();
    let tampered = std::panic::catch_unwind(|| verify_inputs(&b, &list));
    assert!(tampered.is_err(), "a changed input is refused");
}
