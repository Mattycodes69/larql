//! Byte-budgeted reading for the GGUF header.
//!
//! Every count and length in a GGUF header is an untrusted `u64`. Before one
//! sizes an allocation it is checked against the bytes the file can still
//! yield: a declared count whose smallest possible encoding exceeds what
//! remains is refused instead of reserved.

use std::io::{Cursor, Read};

use crate::detect::ModelError;

/// A reader that knows how many bytes it can still yield.
pub(super) trait BoundedRead: Read {
    fn remaining(&self) -> u64;
}

impl<T: AsRef<[u8]>> BoundedRead for Cursor<T> {
    fn remaining(&self) -> u64 {
        (self.get_ref().as_ref().len() as u64).saturating_sub(self.position())
    }
}

/// Wraps a reader of known total length, tracking what has been consumed.
pub(super) struct Budgeted<R> {
    inner: R,
    consumed: u64,
    total: u64,
}

impl<R: Read> Budgeted<R> {
    pub(super) fn new(inner: R, total: u64) -> Self {
        Self {
            inner,
            consumed: 0,
            total,
        }
    }

    /// Bytes consumed from the start of the stream.
    pub(super) fn position(&self) -> u64 {
        self.consumed
    }
}

impl<R: Read> Read for Budgeted<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.consumed += n as u64;
        Ok(n)
    }
}

impl<R: Read> BoundedRead for Budgeted<R> {
    fn remaining(&self) -> u64 {
        self.total.saturating_sub(self.consumed)
    }
}

/// Accept `count` items of at least `min_item_bytes` each only if they can
/// fit in what `r` has left, and only if the count fits in `usize`.
pub(super) fn checked_count(
    r: &impl BoundedRead,
    count: u64,
    min_item_bytes: u64,
    what: &str,
) -> Result<usize, ModelError> {
    let remaining = r.remaining();
    let fits = count
        .checked_mul(min_item_bytes)
        .is_some_and(|needed| needed <= remaining);
    if !fits {
        return Err(ModelError::Parse(format!(
            "GGUF {what}: declares {count} items of at least {min_item_bytes} bytes, \
             but only {remaining} bytes remain"
        )));
    }
    usize::try_from(count)
        .map_err(|_| ModelError::Parse(format!("GGUF {what}: count {count} exceeds usize")))
}
