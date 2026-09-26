//! The operand store's own refusals: a byte window over an operand it
//! does not hold, and a load that would manufacture a representation
//! under a source that forbids it.

use crate::format::vindex3::fixtures::{dense_f32_model, encode_fixture_container};
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::backend::WeightFormat;
use crate::format::vindex3::opplan::exec::operands::{OperandStore, RepresentationSource};
use crate::format::vindex3::opplan::exec::weights::load_weight;
use crate::format::vindex3::opplan::{plan_component_ops, LayerAttention, OperandRef};

/// A name no fixture here stores.
const ABSENT: &str = "no.such.name";

struct Dense {
    _src: tempfile::TempDir,
    container: tempfile::TempDir,
}

impl Dense {
    fn build() -> Self {
        let src = tempfile::tempdir().unwrap();
        let container = tempfile::tempdir().unwrap();
        encode_fixture_container(dense_f32_model, src.path(), container.path(), "dense");
        Self {
            _src: src,
            container,
        }
    }

    /// The store under `source`, and layer 0's query projection.
    fn open(&self, source: RepresentationSource) -> (OperandStore, OperandRef) {
        let inspection = inspect_container(self.container.path(), false).unwrap();
        let plan = plan_component_ops(&inspection, self.container.path(), "target")
            .unwrap()
            .plan
            .expect("the dense fixture plans");
        let LayerAttention::Softmax(op) = &plan.layers[0].attention else {
            panic!("the dense fixture's layer 0 is softmax attention");
        };
        let store =
            OperandStore::open_for(self.container.path(), &inspection, None, source).unwrap();
        (store, op.q.clone())
    }
}

/// A byte window names its object and tensor; either one absent is
/// refused by name before any seek.
#[test]
fn a_byte_window_over_an_absent_object_or_tensor_is_refused_by_name() {
    let dense = Dense::build();
    let (store, q) = dense.open(RepresentationSource::Auto);
    let len = store
        .stored_len(&q)
        .expect("the query projection is stored");

    let no_object = OperandRef {
        object: ABSENT.to_string(),
        ..q.clone()
    };
    let err = store
        .load_raw_range(&no_object, len, 0, len)
        .err()
        .expect("an absent object is refused")
        .to_string();
    assert!(
        err.contains("missing object") && err.contains(ABSENT),
        "{err}"
    );

    let no_tensor = OperandRef {
        tensor: ABSENT.to_string(),
        ..q
    };
    let err = store
        .load_raw_range(&no_tensor, len, 0, len)
        .err()
        .expect("an absent tensor is refused")
        .to_string();
    assert!(
        err.contains("missing tensor") && err.contains(ABSENT),
        "{err}"
    );
}

/// Under `stored`, a format the container holds no compiled bytes for is
/// refused at load rather than quantised, and nothing is counted as
/// manufactured; under `auto` the same load quantises and is counted.
#[test]
fn a_stored_source_refuses_to_quantise_at_load_and_auto_counts_it() {
    let dense = Dense::build();
    let (stored, q) = dense.open(RepresentationSource::Stored);
    assert_eq!(stored.representation_source(), RepresentationSource::Stored);
    let err = load_weight((&stored).into(), &q, WeightFormat::Mxfp4)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("forbids manufacturing") && err.contains(&q.tensor),
        "{err}"
    );
    assert_eq!(stored.runtime_quantised(), 0);

    let (auto, q) = dense.open(RepresentationSource::Auto);
    assert_eq!(auto.representation_source(), RepresentationSource::Auto);
    load_weight((&auto).into(), &q, WeightFormat::Mxfp4).expect("auto quantises at load");
    assert_eq!(auto.runtime_quantised(), 1);
}
