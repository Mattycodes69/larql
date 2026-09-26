//! Contiguous views of ndarray data, never silently empty.

use std::borrow::Cow;

use ndarray::{ArrayBase, Data, Dimension};

/// The array's elements in row-major (logical) order: borrowed when the
/// storage already is, copied when it is not. `as_slice()` returns `None`
/// for a non-standard layout, and substituting an empty slice there would
/// feed a zero-length embedding or K/V into decode without complaint.
pub fn row_major<S, D>(array: &ArrayBase<S, D>) -> Cow<'_, [f32]>
where
    S: Data<Elem = f32>,
    D: Dimension,
{
    match array.as_slice() {
        Some(slice) => Cow::Borrowed(slice),
        None => Cow::Owned(array.iter().copied().collect()),
    }
}
