//! RESIDUAL-BUS-2 I1 and I2: the execution identity binds the effective
//! model, every value-changing setting has a fate, and an unanchored
//! identity is refused by name.

use std::path::{Path, PathBuf};

use crate::format::vindex3::opplan::exec::cpu::physical::{
    ArithmeticArm, KQuantExecution, KQUANT_EXEC_WIDEN,
};
use crate::format::vindex3::opplan::exec::identity::{
    ExecutionIdentity, ModelAuthority, ProcessArithmetic, SettingFate, SETTING_FATES,
};
use crate::format::vindex3::opplan::exec::operands::{
    OperandEdit, OperandOverrides, OperandSource,
};
use crate::format::vindex3::opplan::exec::prepared::{ExecutionSlice, PreparedOperands};
use crate::format::vindex3::opplan::exec::production::ProductionBackend;
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;
use crate::format::vindex3::opplan::LayerFfn;

use super::draft_slice::hybrid;

/// Two distinct, well-formed model authorities.
const MODEL_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const MODEL_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

fn authority(hex: &str) -> Option<ModelAuthority> {
    Some(ModelAuthority::new(hex).unwrap())
}

#[test]
fn the_same_image_and_model_give_the_same_digest() {
    let (_h, plan, store) = hybrid();
    let backend = ReferenceBackend::new();
    let ops = PreparedOperands::load(&plan, &store, &backend, ExecutionSlice::Full).unwrap();
    let a = ExecutionIdentity::of(&ops, authority(MODEL_A)).unwrap();
    let b = ExecutionIdentity::of(&ops, authority(MODEL_A)).unwrap();
    assert_eq!(a.digest(), b.digest());
    assert!(a.is_anchored());
    a.ensure_anchored().unwrap();
}

/// D1: the same executable realization is not the same identity when the
/// model differs.
#[test]
fn a_different_model_is_a_different_identity() {
    let (_h, plan, store) = hybrid();
    let backend = ReferenceBackend::new();
    let ops = PreparedOperands::load(&plan, &store, &backend, ExecutionSlice::Full).unwrap();
    let a = ExecutionIdentity::of(&ops, authority(MODEL_A)).unwrap();
    let b = ExecutionIdentity::of(&ops, authority(MODEL_B)).unwrap();
    assert_ne!(a.digest(), b.digest());
}

/// The lowering and every pinned realization are part of the identity:
/// the same model under Production and Reference is two computations
/// (BUS-1's Granite result).
#[test]
fn a_different_lowering_is_a_different_identity() {
    let (_h, plan, store) = hybrid();
    let reference = PreparedOperands::load(
        &plan,
        &store,
        &ReferenceBackend::new(),
        ExecutionSlice::Full,
    )
    .unwrap();
    let production = PreparedOperands::load(
        &plan,
        &store,
        &ProductionBackend::new(),
        ExecutionSlice::Full,
    )
    .unwrap();
    let a = ExecutionIdentity::of(&reference, authority(MODEL_A)).unwrap();
    let b = ExecutionIdentity::of(&production, authority(MODEL_A)).unwrap();
    assert_ne!(a.lowering, b.lowering);
    assert_ne!(a.digest(), b.digest());
}

/// The slice is scope inside the identity, not a substitute for it.
#[test]
fn a_different_slice_is_a_different_identity() {
    let (_h, plan, store) = hybrid();
    let backend = ReferenceBackend::new();
    let full = PreparedOperands::load(&plan, &store, &backend, ExecutionSlice::Full).unwrap();
    let draft =
        PreparedOperands::load(&plan, &store, &backend, ExecutionSlice::Draft { end: 1 }).unwrap();
    let a = ExecutionIdentity::of(&full, authority(MODEL_A)).unwrap();
    let b = ExecutionIdentity::of(&draft, authority(MODEL_A)).unwrap();
    assert_ne!(a.digest(), b.digest());
}

/// F1: every recorded process setting moves the digest. The settings are
/// process-global, so each is varied on a copy of the resolved record.
#[test]
fn every_recorded_process_setting_moves_the_digest() {
    let (_h, plan, store) = hybrid();
    let backend = ReferenceBackend::new();
    let ops = PreparedOperands::load(&plan, &store, &backend, ExecutionSlice::Full).unwrap();
    let base = ProcessArithmetic::current().unwrap();
    let digest = |p: ProcessArithmetic| {
        ExecutionIdentity::with_process(&ops, authority(MODEL_A), p).digest()
    };
    let reference = digest(base.clone());
    let variants: Vec<(&str, ProcessArithmetic)> = vec![
        ("arithmetic_arm", {
            let mut p = base.clone();
            p.arithmetic_arm = if p.arithmetic_arm == ArithmeticArm::Q8TimesQ8 {
                ArithmeticArm::FloatActivation
            } else {
                ArithmeticArm::Q8TimesQ8
            };
            p
        }),
        ("kquant_execution", {
            let mut p = base.clone();
            p.kquant_execution = if p.kquant_execution == KQuantExecution::Widen {
                KQuantExecution::Direct
            } else {
                KQuantExecution::Widen
            };
            p
        }),
        ("activation_scaling", {
            let mut p = base.clone();
            p.activation_scaling.push('!');
            p
        }),
        ("activation_block", {
            let mut p = base.clone();
            p.activation_block += 1;
            p
        }),
        ("activation_code", {
            let mut p = base.clone();
            p.activation_code.push('!');
            p
        }),
        ("bit_identical_only", {
            let mut p = base.clone();
            p.bit_identical_only = !p.bit_identical_only;
            p
        }),
        ("weight_index", {
            let mut p = base.clone();
            p.weight_index = !p.weight_index;
            p
        }),
        ("cpu_workers", {
            let mut p = base.clone();
            p.cpu_workers += 1;
            p
        }),
    ];
    for (name, variant) in variants {
        assert_ne!(digest(variant), reference, "{name} did not move the digest");
    }
}

/// D8: no model authority, and an overlaid source, are both unanchored,
/// and each refusal says which.
#[test]
fn an_unanchored_identity_is_refused_by_name() {
    let (_h, plan, store) = hybrid();
    let backend = ReferenceBackend::new();
    let ops = PreparedOperands::load(&plan, &store, &backend, ExecutionSlice::Full).unwrap();
    let no_model = ExecutionIdentity::of(&ops, None).unwrap();
    assert!(!no_model.is_anchored());
    let err = no_model.ensure_anchored().unwrap_err().to_string();
    assert!(err.contains("no model authority"), "{err}");

    // A real edit: an EMPTY overlay collapses to the bare store
    // (`OperandSource::overlaid`), which is the base model and anchored.
    let gate = plan
        .layers
        .iter()
        .find_map(|layer| match &layer.ffn {
            Some(LayerFfn::Dense(op)) => op.gate.clone(),
            _ => None,
        })
        .expect("the hybrid fixture has a gated dense FFN");
    let mut overrides = OperandOverrides::new();
    overrides.push(
        &gate,
        OperandEdit::Row {
            index: 0,
            values: vec![1.0; gate.shape[1]],
        },
    );
    let overlaid = PreparedOperands::load(
        &plan,
        OperandSource::overlaid(&store, &overrides),
        &backend,
        ExecutionSlice::Full,
    )
    .unwrap();
    let with_overlay = ExecutionIdentity::of(&overlaid, authority(MODEL_A)).unwrap();
    assert!(with_overlay.overlaid);
    let err = with_overlay.ensure_anchored().unwrap_err().to_string();
    assert!(err.contains("overlaid"), "{err}");
}

#[test]
fn a_model_authority_must_be_a_digest() {
    assert!(ModelAuthority::new("not-a-digest").is_err());
    assert!(ModelAuthority::new(&MODEL_A[1..]).is_err());
    assert_eq!(
        ModelAuthority::new(MODEL_A.to_ascii_uppercase()).unwrap(),
        ModelAuthority::new(MODEL_A).unwrap()
    );
}

/// The executor's production source, as `(relative path, contents)`.
/// Test modules are skipped: a fixture variable is not a setting the
/// executor reads.
fn production_sources() -> Vec<(String, String)> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/format/vindex3/opplan/exec");
    let mut out = Vec::new();
    let mut stack: Vec<PathBuf> = vec![root.clone()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            let name = path.file_name().unwrap().to_string_lossy().to_string();
            if path.is_dir() {
                if name != "tests" {
                    stack.push(path);
                }
            } else if name.ends_with(".rs") && !name.ends_with("_tests.rs") && name != "tests.rs" {
                let rel = path.strip_prefix(&root).unwrap().display().to_string();
                out.push((rel, std::fs::read_to_string(&path).unwrap()));
            }
        }
    }
    out
}

/// Every `"LARQL_…"` string literal in `source`.
fn settings_named_in(source: &str) -> Vec<String> {
    const PREFIX: &str = "\"LARQL_";
    let mut found = Vec::new();
    let mut rest = source;
    while let Some(start) = rest.find(PREFIX) {
        let tail = &rest[start + 1..];
        let end = tail.find('"').unwrap_or(tail.len());
        found.push(tail[..end].to_string());
        rest = &tail[end..];
    }
    found
}

/// I2: every environment setting the executor's production code names
/// has a fate, and every fate names a setting that is still read. A new
/// setting with no fate fails here, which is the point.
#[test]
fn every_setting_the_executor_reads_has_a_fate() {
    let mut read: Vec<(String, String)> = Vec::new();
    for (file, source) in production_sources() {
        for setting in settings_named_in(&source) {
            read.push((setting, file.clone()));
        }
    }
    let fated: Vec<&str> = SETTING_FATES.iter().map(|(name, _)| *name).collect();
    let unfated: Vec<&(String, String)> = read
        .iter()
        .filter(|(setting, _)| !fated.contains(&setting.as_str()))
        .collect();
    assert!(
        unfated.is_empty(),
        "settings read with no fate in identity::SETTING_FATES: {unfated:?}"
    );
    for (name, fate) in SETTING_FATES {
        assert!(
            read.iter().any(|(setting, _)| setting == name),
            "{name} has a fate but no production code reads it"
        );
        if let SettingFate::Excluded(reason) = fate {
            assert!(!reason.is_empty(), "{name} is excluded with no reason");
        }
    }
}

/// Set by the parent of [`every_value_changing_setting_moves_the_digest_across_processes`]
/// so the child below prints instead of skipping.
const CHILD_ENV: &str = "LARQL_IDENTITY_CHILD";
/// How the child reports its digest on stdout.
const DIGEST_MARK: &str = "IDENTITY_DIGEST=";
/// How the child reports the CPU pool size it resolved, so the parent can
/// ask for a DIFFERENT one on whatever machine it runs on (a fixed count
/// can coincide with a runner's default, and did: macOS CI resolves 3).
const WORKERS_MARK: &str = "IDENTITY_WORKERS=";
/// This module's path inside the test binary, for `--exact`.
const CHILD_TEST: &str = "format::vindex3::opplan::exec::tests::execution_identity::identity_child";

/// The child half of the cross-process witness: prints the digest of one
/// fixed image under whatever settings this process was started with.
#[test]
#[ignore = "run by every_value_changing_setting_moves_the_digest_across_processes"]
fn identity_child() {
    if std::env::var(CHILD_ENV).is_err() {
        return;
    }
    let (_h, plan, store) = hybrid();
    let backend = ReferenceBackend::new();
    let ops = PreparedOperands::load(&plan, &store, &backend, ExecutionSlice::Full).unwrap();
    let identity = ExecutionIdentity::of(&ops, authority(MODEL_A)).unwrap();
    println!("{WORKERS_MARK}{}", identity.process.cpu_workers);
    println!("{DIGEST_MARK}{}", identity.digest());
}

/// Run the child with every identity setting cleared, then `set` applied,
/// and return the value it printed after `mark`.
fn child_value(set: &[(&str, &str)], mark: &str) -> String {
    let mut cmd = std::process::Command::new(std::env::current_exe().unwrap());
    cmd.args([
        CHILD_TEST,
        "--exact",
        "--ignored",
        "--nocapture",
        "--test-threads=1",
    ])
    .env(CHILD_ENV, "1");
    for (name, fate) in SETTING_FATES {
        if *fate == SettingFate::InIdentity {
            cmd.env_remove(name);
        }
    }
    for (name, value) in set {
        cmd.env(name, value);
    }
    let out = cmd.output().unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    stdout
        .lines()
        // libtest prints `test <name> ... ` on the same line first.
        .find_map(|l| {
            l.find(mark)
                .map(|at| l[at + mark.len()..].trim().to_string())
        })
        .unwrap_or_else(|| {
            panic!(
                "the child printed no {mark} under {set:?}: {stdout}\n{}",
                String::from_utf8_lossy(&out.stderr)
            )
        })
}

fn child_digest(set: &[(&str, &str)]) -> String {
    child_value(set, DIGEST_MARK)
}

/// RESIDUAL-BUS-2 I3's witness: two processes that differ in exactly one
/// value-changing setting present different digests, so their bindings
/// cannot agree. `q8xq8` against `q8xq8b` is the case the reconnaissance
/// found the old fingerprint blind to: the SAME arithmetic arm, differing
/// only in activation scale span.
#[test]
fn every_value_changing_setting_moves_the_digest_across_processes() {
    let baseline = child_digest(&[]);
    assert_eq!(
        baseline,
        child_digest(&[]),
        "the child is not deterministic"
    );
    /// One comparison: the child under `before` against the child under
    /// `after`, which differ in exactly one setting.
    struct Case {
        what: &'static str,
        before: &'static [(&'static str, &'static str)],
        after: &'static [(&'static str, &'static str)],
    }
    let cases = [
        Case {
            what: "arithmetic arm",
            before: &[],
            after: &[("LARQL_CPU_ARITHMETIC", "q8xq8")],
        },
        Case {
            what: "scale span under one arm",
            before: &[("LARQL_CPU_ARITHMETIC", "q8xq8")],
            after: &[("LARQL_CPU_ARITHMETIC", "q8xq8b")],
        },
        Case {
            what: "activation block",
            before: &[],
            after: &[("LARQL_CPU_ACT_BLOCK", "32")],
        },
        Case {
            what: "activation code",
            before: &[],
            after: &[("LARQL_CPU_ACT_CODE", "asymmetric")],
        },
        Case {
            what: "K2-only kernels",
            before: &[],
            after: &[("LARQL_CPU_BIT_IDENTICAL", "1")],
        },
        Case {
            what: "weight index",
            before: &[],
            after: &[("LARQL_CPU_WEIGHT_INDEX", "1")],
        },
        Case {
            what: "kquant execution",
            before: &[],
            after: &[("LARQL_KQUANT_EXEC", KQUANT_EXEC_WIDEN)],
        },
    ];
    for case in cases {
        let a = if case.before.is_empty() {
            baseline.clone()
        } else {
            child_digest(case.before)
        };
        assert_ne!(
            a,
            child_digest(case.after),
            "{} did not move the digest across processes",
            case.what
        );
    }
    // The pool size is machine-dependent, so ask for one more worker than
    // this machine resolves by default rather than a fixed count.
    let resolved: usize = child_value(&[], WORKERS_MARK).parse().unwrap();
    let other = (resolved + 1).to_string();
    assert_ne!(
        baseline,
        child_digest(&[("LARQL_CPU_WORKERS", other.as_str())]),
        "CPU pool size ({resolved} -> {other}) did not move the digest across processes"
    );
}
