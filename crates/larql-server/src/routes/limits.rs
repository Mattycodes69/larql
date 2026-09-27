//! Bounds on client-supplied request sizes.
//!
//! Every route that takes a generation length, a result count or a KNN
//! width reads it through here. A value over its bound is rejected with a
//! 400, never silently clamped: a client asking for more than the server
//! will do should learn that, not receive a shorter answer.
//!
//! The bounds exist because these values size allocations and hold the
//! model lock. An unbounded `max_tokens` reaches `Vec::with_capacity` in
//! the decode loop, where a huge value aborts the whole process.

/// Most tokens one generation request may ask for.
pub const MAX_GENERATION_TOKENS: usize = 32_768;

/// Most rows a listing route (`limit`, `top`) may return.
pub const MAX_RESULT_ROWS: usize = 10_000;

/// Widest gate-KNN selection a request may ask for. Covers the largest
/// FFN widths in use (tens of thousands of features per layer).
pub const MAX_TOP_K: usize = 1 << 17;

/// `value` when it is within `max`, else a message naming the field.
pub fn within(field: &str, value: usize, max: usize) -> Result<usize, String> {
    if value > max {
        Err(format!(
            "{field} = {value} exceeds the server limit of {max}"
        ))
    } else {
        Ok(value)
    }
}

/// A proto3 count, where `0` means unset: `default` then, otherwise the
/// value within `max`.
pub fn proto_count(field: &str, value: u32, default: usize, max: usize) -> Result<usize, String> {
    match value {
        0 => Ok(default),
        v => within(field, usize::try_from(v).unwrap_or(usize::MAX), max),
    }
}

/// The generation length: the request's value (or `default`) within
/// [`MAX_GENERATION_TOKENS`].
pub fn generation_tokens(
    field: &str,
    requested: Option<usize>,
    default: usize,
) -> Result<usize, String> {
    within(field, requested.unwrap_or(default), MAX_GENERATION_TOKENS)
}
