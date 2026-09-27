//! Format-aware matrix-operand loading for the plan executor.
//!
//! The interpreter asks the backend which [`WeightFormat`] it computes
//! in and loads every matrix operand through [`load_weight`]; backends
//! receive slices, never operand references, exactly as before. The f16
//! path exists for device residency: a device buffer cache keyed by
//! `(pointer, length)` sees the same allocation on every call and keeps
//! the weight resident instead of re-uploading it per forward.
//!
//! **The bf16 → f16 conversion is exact for every normal-range value.**
//! bf16 carries 7 mantissa bits and f16 carries 10, so any bf16 value
//! whose magnitude lies in f16's normal range converts without rounding.
//! Overflow (|x| ≥ 65520, unrepresentable in f16) fails closed naming
//! the tensor — it would silently become infinity. Values below f16's
//! normal range land on subnormals and may round in the last bits; that
//! tail is a bounded realisation choice, and the parity gates against
//! the f32 backends and the upstream trace are its judge.

use super::backend::{KQuantActivation, Nvfp4Activation, WeightFormat};
use crate::format::vindex3::represent::kquant::KQuant;
use crate::format::vindex3::represent::physical::WeightRegion;
// MXFP4 group geometry — per row, `k/32` groups of 16 packed bytes (lo
// nibble first) plus one e8m0 scale byte each — is the models crate's,
// so the kernel's layout contract and the codec's have one definition.
use staged::StagedF32;

/// Alignment (and length granularity) of f16 weight allocations:
/// the Apple-GPU page size. A page-aligned, page-multiple allocation
/// lets a Metal device wrap the memory zero-copy instead of copying it
/// into a private buffer; any other device simply sees ordinary bytes.
pub const DEVICE_PAGE_ALIGN: usize = 16384;

/// Safetensors dtypes this loader can narrow to f16. bf16 converts
/// exactly (normal range); f32 rounds to nearest-even.
const DTYPE_BF16: &str = "BF16";
const DTYPE_F32: &str = "F32";

/// A page-aligned, page-multiple, zero-padded byte buffer.
///
/// [`AlignedBytes::as_slice`] returns the *padded* slice on purpose:
/// callers hand the whole allocation to a device so the buffer length
/// stays page-multiple; matrix geometry always travels separately.
#[derive(Debug)]
pub struct AlignedBytes {
    ptr: std::ptr::NonNull<u8>,
    /// Allocation length — `logical` rounded up to the page.
    padded: usize,
    /// Meaningful bytes at the front of the allocation.
    logical: usize,
}

// The buffer is plain owned bytes; nothing about the raw pointer ties
// it to a thread.
unsafe impl Send for AlignedBytes {}
unsafe impl Sync for AlignedBytes {}

impl AlignedBytes {
    /// Allocate a zeroed, page-aligned buffer holding `logical` bytes.
    pub fn zeroed(logical: usize) -> Self {
        let padded = logical.div_ceil(DEVICE_PAGE_ALIGN).max(1) * DEVICE_PAGE_ALIGN;
        let layout = std::alloc::Layout::from_size_align(padded, DEVICE_PAGE_ALIGN)
            .expect("page-aligned layout is always valid");
        // SAFETY: layout has non-zero size (padded >= one page).
        let raw = unsafe { std::alloc::alloc_zeroed(layout) };
        let ptr = std::ptr::NonNull::new(raw).unwrap_or_else(|| {
            std::alloc::handle_alloc_error(layout);
        });
        Self {
            ptr,
            padded,
            logical,
        }
    }

    /// A page-aligned copy of `bytes` — how a natively stored quantised
    /// operand (an MXFP4 expert's blocks or scales) is bound without a
    /// numeric transform: the bytes are the checkpoint's, only the
    /// alignment is ours.
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let mut aligned = Self::zeroed(bytes.len());
        aligned.as_mut_slice()[..bytes.len()].copy_from_slice(bytes);
        aligned
    }

    /// The full padded allocation — page-aligned pointer, page-multiple
    /// length, zero beyond `logical_len`.
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: the allocation is `padded` bytes, initialised (zeroed
        // at alloc, fronts overwritten by the converter).
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.padded) }
    }

    /// Writable, for the converters that FILL a buffer — narrowing to
    /// f16, quantising to Q8. Not public beyond the executor: a caller
    /// that could rewrite a resident weight in place would be editing the
    /// model behind the operand store.
    pub(super) fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: as above, and `&mut self` guarantees uniqueness.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.padded) }
    }

    /// Meaningful bytes at the front of the allocation.
    pub fn logical_len(&self) -> usize {
        self.logical
    }
}

impl Drop for AlignedBytes {
    fn drop(&mut self) {
        let layout = std::alloc::Layout::from_size_align(self.padded, DEVICE_PAGE_ALIGN)
            .expect("layout validated at allocation");
        // SAFETY: allocated with exactly this layout in `zeroed`.
        unsafe { std::alloc::dealloc(self.ptr.as_ptr(), layout) };
    }
}

/// The stored forms the CPU executor runs in place over a whole matrix,
/// and therefore the only forms a mapped binding can take: bf16 through
/// the fused bf16 matvec, f32 through BLAS. A stored form outside this
/// enum needs a decode, which a mapping by definition does not do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MappedForm {
    Bf16,
    F32,
}

impl MappedForm {
    pub fn format(self) -> WeightFormat {
        match self {
            Self::Bf16 => WeightFormat::Bf16,
            Self::F32 => WeightFormat::F32,
        }
    }

    /// Bytes per element of the stored form.
    pub fn width(self) -> usize {
        match self {
            Self::Bf16 => std::mem::size_of::<u16>(),
            Self::F32 => std::mem::size_of::<f32>(),
        }
    }

    /// The form a resident representation maps as, when it maps at all.
    pub fn of(format: WeightFormat) -> Option<Self> {
        match format {
            WeightFormat::Bf16 => Some(Self::Bf16),
            WeightFormat::F32 => Some(Self::F32),
            _ => None,
        }
    }
}

/// One loaded matrix operand, owning its bytes in the format the
/// backend declared.
#[derive(Debug)]
pub enum LoadedWeight {
    F32(StagedF32),
    /// Symmetric int8 codes plus one f32 scale per [`Q8_BLOCK`]
    /// elements. The only LOSSY residency format on this path: the values
    /// resident are not the values stored.
    Q8 {
        codes: Vec<i8>,
        scales: Vec<f32>,
        /// Per-[`SUM_BLOCK`] sums of the codes — a materialised execution
        /// index for the asymmetric-activation path, EMPTY where no arm
        /// consumes it. Not model semantics: it is derivable from
        /// `codes`, and is stored only because recomputing it every token
        /// costs a second integer reduction per block.
        sums: Vec<i16>,
    },
    /// Symmetric int4 codes packed two per byte, plus one f32 scale per
    /// [`Q4_BLOCK`] elements. 4.5 bits/weight — half of Q8's bytes and
    /// 18.1x its quantisation step.
    Q4 {
        packed: Vec<u8>,
        scales: Vec<f32>,
    },
    /// Stored bf16 code units, byte-for-byte as the checkpoint holds
    /// them. The cheapest possible load: no conversion at all.
    Bf16(AlignedBytes),
    /// The stored bytes themselves, bound as a region of the container's
    /// mapped segment: nothing copied, nothing converted, paged in as
    /// touched. Only the two forms the CPU executes in place over a
    /// whole matrix can be mapped, and the form is part of the variant so
    /// no other can be constructed.
    Mapped {
        region: WeightRegion,
        form: MappedForm,
    },
    F16(AlignedBytes),
    Mxfp4 {
        packed: AlignedBytes,
        scales: AlignedBytes,
    },
    Nvfp4 {
        packed: AlignedBytes,
        scales: AlignedBytes,
        tensor_scale: f32,
        /// The activation this pack is bound to run against, fixed at load.
        activation: Nvfp4Activation,
    },
    /// Fine-grained FP8 as the CHECKPOINT stores it: E4M3 codes and the
    /// f32 scale grid, both byte-for-byte, neither widened.
    ///
    /// The only variant here bound from TWO container tensors. Every
    /// other compact form keeps its scales in a stream this build
    /// produced (`Q8`, `Q4`) or inside the blocks (`KQuant`); this one's
    /// scales are a separate tensor the checkpoint shipped, reached
    /// through [`OperandSource::companion`].
    Fp8Block {
        codes: AlignedBytes,
        scales: Vec<f32>,
        /// The tile, DERIVED from the two tensors' shapes at load and
        /// carried thereafter — never re-read from
        /// `quantization_config.weight_block_size`, which one checkpoint
        /// may contradict per tensor.
        block_rows: usize,
        block_cols: usize,
        scale_cols: usize,
    },
    /// A compiled ggml K-quant pack, byte-for-byte as the container holds
    /// it, with the codec that names its layout.
    ///
    /// The cheapest load after [`Self::Bf16`]: the bytes are read and
    /// kept. No decode, no requantise — PARETO-1's v3 arm, whose whole
    /// claim is that the bytes the kernel reads are the artifact's own.
    /// Plain owned bytes rather than [`AlignedBytes`]: no device wraps a
    /// K-quant, so page alignment would buy nothing here.
    KQuant {
        blocks: Vec<u8>,
        codec: KQuant,
        /// The activation form this binding runs against, fixed by the
        /// pinned realization — see [`WeightFormat::KQuantQ8k`].
        activation: KQuantActivation,
    },
    /// The stored bytes for an operand whose representation this loader
    /// does not know how to widen, quantise, or otherwise interpret —
    /// read and kept exactly as [`WeightFormat::CodecOwned`] promises,
    /// with a copy of the operand's own stored representation name.
    CodecOwned {
        bytes: Vec<u8>,
        label: String,
    },
}

pub mod staged;
pub use staged::StagedF32 as F32Image;

mod load;
// Names the test module reaches through `use super::*`.
#[cfg(test)]
use super::backend::WeightSlice;
#[cfg(test)]
use super::quantise::{Q4_BLOCK, Q8_BLOCK};
#[cfg(test)]
use crate::format::vindex3::represent::kquant;
#[cfg(test)]
use larql_models::quant::mxfp4::e8m0_to_f32;

mod loaded;
mod quantize;
pub use load::*;
use loaded::*;
pub use quantize::*;

#[cfg(test)]
mod staged_tests;
#[cfg(test)]
mod tests;
