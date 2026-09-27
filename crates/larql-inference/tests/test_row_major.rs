use larql_inference::row_major::row_major;
use ndarray::Array2;
use std::borrow::Cow;

#[test]
fn contiguous_arrays_are_borrowed() {
    let a = Array2::from_shape_vec((2, 3), vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let rows = row_major(&a);
    assert!(matches!(rows, Cow::Borrowed(_)));
    assert_eq!(&*rows, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn a_non_standard_layout_is_copied_in_logical_order_not_emptied() {
    let a = Array2::from_shape_vec((2, 3), vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let t = a.t();
    assert!(t.as_slice().is_none(), "the transpose is not row-major");
    assert_eq!(&*row_major(&t), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}
