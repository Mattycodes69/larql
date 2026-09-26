//! `load_weight`'s refusals and its stored-bytes arms, over a real
//! container.
//!
//! Each compact format needs something of the operand before it reads a
//! byte — an `[out, in]` matrix to block along, an even input axis to
//! pack, a declared scale grid — and each refuses by name when that is
//! missing. The bf16 and f16 arms bind STORED bytes, so they are judged
//! against a container whose projections are stored in those forms.

use super::bf16_zlib_execution::{transcode, Transcode};
use crate::format::vindex3::fixtures::{dense_f32_model, encode_fixture_container};
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::backend::WeightFormat;
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::exec::weights::{load_weight, LoadedWeight};
use crate::format::vindex3::opplan::{plan_component_ops, LayerAttention, OperandRef};

/// Which projections a transcode rewrites.
const PROJECTION_SUFFIX: &str = "_proj.weight";
/// An input axis q4 cannot pack two codes to the byte.
const ODD_IN: usize = 3;
/// Bytes per stored bf16 element.
const BF16_WIDTH: usize = std::mem::size_of::<u16>();

struct Dense {
    _src: tempfile::TempDir,
    container: tempfile::TempDir,
}

impl Dense {
    fn build(into: Option<Transcode>) -> Self {
        let src = tempfile::tempdir().unwrap();
        let container = tempfile::tempdir().unwrap();
        encode_fixture_container(dense_f32_model, src.path(), container.path(), "dense");
        if let Some(into) = into {
            let rewritten = transcode(
                container.path(),
                |name, shape| shape.len() == 2 && name.ends_with(PROJECTION_SUFFIX),
                into,
            );
            assert!(!rewritten.is_empty(), "the dense fixture has projections");
        }
        Self {
            _src: src,
            container,
        }
    }

    /// The store, and layer 0's query projection.
    fn open(&self) -> (OperandStore, OperandRef) {
        let inspection = inspect_container(self.container.path(), false).unwrap();
        let plan = plan_component_ops(&inspection, self.container.path(), "target")
            .unwrap()
            .plan
            .expect("the dense fixture plans");
        let LayerAttention::Softmax(op) = &plan.layers[0].attention else {
            panic!("the dense fixture's layer 0 is softmax attention");
        };
        let q = op.q.clone();
        let store = OperandStore::open(self.container.path(), &inspection).unwrap();
        (store, q)
    }
}

fn refusal(store: &OperandStore, operand: &OperandRef, format: WeightFormat) -> String {
    load_weight(store.into(), operand, format)
        .expect_err("the load must be refused")
        .to_string()
}

/// q8 and q4 block along the INPUT axis, so an operand that is not an
/// `[out, in]` matrix has no blocks and is refused before any read.
#[test]
fn block_quantised_formats_refuse_an_operand_with_no_input_axis() {
    let dense = Dense::build(None);
    let (store, q) = dense.open();
    let flat = OperandRef {
        shape: vec![q.shape.iter().product()],
        ..q
    };
    for format in [WeightFormat::Q8, WeightFormat::Q4] {
        let err = refusal(&store, &flat, format);
        assert!(
            err.contains("blocks along the INPUT axis") && err.contains(&flat.tensor),
            "{format:?}: {err}"
        );
    }
}

/// q4 packs two codes a byte; an odd input axis would silently drop a
/// weight per row, so it is refused.
#[test]
fn q4_refuses_an_odd_input_axis() {
    let dense = Dense::build(None);
    let (store, q) = dense.open();
    let odd = OperandRef {
        shape: vec![q.shape[0], ODD_IN],
        ..q
    };
    let err = refusal(&store, &odd, WeightFormat::Q4);
    assert!(err.contains("odd input axis"), "{err}");
}

/// Fine-grained FP8 codes mean nothing without the grid the container
/// declares for them; an operand with no such dependency is refused.
#[test]
fn fp8_block_refuses_an_operand_with_no_declared_scale_grid() {
    let dense = Dense::build(None);
    let (store, q) = dense.open();
    let err = refusal(&store, &q, WeightFormat::Fp8Block);
    assert!(
        err.contains("declares no") && err.contains("dependency"),
        "{err}"
    );
}

/// bf16 residency over a bf16-stored projection keeps the stored bytes:
/// one resident bf16 image, the operand's element count wide.
#[test]
fn bf16_residency_binds_a_bf16_stored_projection_as_stored() {
    let dense = Dense::build(Some(Transcode::Bf16));
    let (store, q) = dense.open();
    let loaded = load_weight((&store).into(), &q, WeightFormat::Bf16).unwrap();
    let LoadedWeight::Bf16(bytes) = &loaded else {
        panic!("bf16 residency must bind the bf16 form");
    };
    let elements: usize = q.shape.iter().product();
    assert_eq!(bytes.logical_len(), elements * BF16_WIDTH);
    assert!(!loaded.is_widened_f32());
}

/// The f16 arm narrows bf16 and f32 only; an f16-STORED operand is a
/// dtype it has no judged narrowing for, and it says so by name.
#[test]
fn f16_residency_refuses_a_dtype_it_has_no_judged_narrowing_for() {
    let dense = Dense::build(Some(Transcode::F16));
    let (store, q) = dense.open();
    let err = refusal(&store, &q, WeightFormat::F16);
    assert!(
        err.contains("no judged f16 narrowing") && err.contains(&q.tensor),
        "{err}"
    );
}
