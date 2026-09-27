//! RESIDUAL-BUS-2 I1 and I2: the execution identity binds the effective
//! model, every value-changing setting has a fate, and an unanchored
//! identity is refused by name.

use std::path::{Path, PathBuf};

use crate::format::vindex3::opplan::exec::cpu::physical::{ArithmeticArm, KQuantExecution};
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
