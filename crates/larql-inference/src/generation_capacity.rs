//! Preallocation for per-token generation buffers.
//!
//! `max_tokens` is a caller's upper bound, not a forecast: most generations
//! stop at EOS long before it. Sizing a `Vec` by it directly lets a huge
//! request abort the process in the allocator before a token is produced.
//! Buffers start at most this large and grow as tokens arrive.

/// Largest up-front reservation for a per-token buffer.
pub const GENERATION_PREALLOC_CAP: usize = 4096;

/// A `Vec` capacity for up to `max_tokens` generated entries.
pub fn generation_capacity(max_tokens: usize) -> usize {
    max_tokens.min(GENERATION_PREALLOC_CAP)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_requests_reserve_exactly_and_huge_ones_are_capped() {
        assert_eq!(generation_capacity(16), 16);
        assert_eq!(generation_capacity(usize::MAX), GENERATION_PREALLOC_CAP);
    }
}
