//! A matrix operand's geometry comes from its declared shape, and only
//! an `[out, in]` shape is a matrix.

use super::super::mla::two_dims;
use crate::format::vindex3::opplan::OperandRef;

fn operand(shape: &[usize]) -> OperandRef {
    OperandRef {
        object: "layers".to_string(),
        tensor: "self_attn.kv_b_proj.weight".to_string(),
        dtype: String::new(),
        shape: shape.to_vec(),
    }
}

#[test]
fn a_two_dimensional_operand_is_out_by_in() {
    const OUT: usize = 6;
    const IN: usize = 4;
    assert_eq!(two_dims(&operand(&[OUT, IN])).unwrap(), (OUT, IN));
}

/// Any other rank is refused by name rather than read as a matrix whose
/// row count a caller would then have to infer from a padded slice.
#[test]
fn an_operand_of_any_other_rank_is_refused_by_name() {
    for shape in [vec![], vec![4], vec![2, 3, 4]] {
        let err = two_dims(&operand(&shape)).unwrap_err().to_string();
        assert!(
            err.contains("kv_b_proj") && err.contains("`[out, in]`"),
            "{shape:?}: {err}"
        );
    }
}
