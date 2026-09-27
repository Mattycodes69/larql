//! GGML block quantization — encode/decode Q4_0, Q4_1, Q5_0, Q5_1,
//! Q8_0, Q4_K, Q6_K.
//!
//! Data format operations only:
//! - **Dequantize**: packed bytes → f32 (GGUF loading)
//! - **Quantize**: f32 → packed bytes (Q4_0, Q8_0 for vindex)
//! - **Metadata**: tensor_data_size, type_name
//!
//! Compute operations (matvec, vecmat, GPU shaders) are in
//! `larql-compute`. Used by GGUF model files. Each format stores
//! blocks of 32 (legacy) or 256 (K-quants) elements with shared scale
//! factors.
//!
//! Module split (post 2026-04-25 audit):
//! - `legacy`   — Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q8_0 (32-element blocks)
//! - `q4_k`     — Q4_K row-dot / row-scaled-add / dequantize (256)
//! - `q6_k`     — Q6_K row-dot / row-scaled-add / dequantize (256)
//! - `quantize` — encode-side helpers for the legacy formats
//!
//! `mod.rs` carries the type-id constants, the generic `dequantize`
//! dispatch, the shared `check_block_input` validator, and the test
//! mod.

use super::half::{decode_bf16, decode_f16};
use crate::detect::ModelError;

pub mod legacy;
pub mod q3_k;
pub mod q4_k;
pub mod q5_k;
pub mod q6_k;
pub mod quantize;
pub mod tq;

pub use legacy::{dequantize_q4_0, dequantize_q5_0, dequantize_q5_1, dequantize_q8_0};
pub use q3_k::dequantize_q3_k;
pub use q4_k::{dequantize_q4_k, q4k_row_dot, q4k_row_scaled_add};
pub use q5_k::dequantize_q5_k;
pub use q6_k::{dequantize_q6_k, q6k_row_dot, q6k_row_scaled_add};

#[cfg(test)]
#[path = "type_id_conformance_tests.rs"]
mod type_id_conformance_tests;
pub use quantize::{quantize_q4_0, quantize_q8_0};

// ── Tensor-type IDs (match GGML wire format) ────────────────────────────
pub const TYPE_F32: u32 = 0;
pub const TYPE_F16: u32 = 1;
pub const TYPE_Q4_0: u32 = 2;
pub const TYPE_Q4_1: u32 = 3;
// Upstream ggml's ids, verified against `ggml_get_type_traits` itself —
// see `larql-vindex`'s `ggml_kquant_golden` fixture, which carries the
// numbers ggml reports for the types it encodes.
//
// These four were TRANSPOSED: `TYPE_Q8_0` was 6 (upstream's Q5_0) and
// `TYPE_Q5_0` was 8 (upstream's Q8_0), so a GGUF containing either was
// decoded as the other — wrong values AND a wrong block stride. It
// stayed invisible because every internal caller passes these same
// constants both ways round, and `loading/gguf/parser.rs` is the only
// place an id arrives from outside. Q8_0 is a common GGUF type; this
// was a live data-corruption bug on GGUF ingest, not a latent one.
pub const TYPE_Q5_0: u32 = 6;
pub const TYPE_Q5_1: u32 = 7;
pub const TYPE_Q8_0: u32 = 8;
/// Upstream's Q8_1. No decoder yet — named so the id cannot be
/// reassigned to something else by accident.
pub const TYPE_Q8_1: u32 = 9;
pub const TYPE_Q2_K: u32 = 10;
pub const TYPE_Q3_K: u32 = 11;
pub const TYPE_Q4_K: u32 = 12;
pub const TYPE_Q5_K: u32 = 13;
pub const TYPE_Q6_K: u32 = 14;
pub const TYPE_BF16: u32 = 30;
/// BitNet 1.58 ternary, 5-trits-per-byte base-3 packing (1.6875 bpw).
pub const TYPE_TQ1_0: u32 = 34;
/// BitNet 1.58 ternary, 4-trits-per-byte 2-bit packing (2.0625 bpw).
pub const TYPE_TQ2_0: u32 = 35;
/// Microsoft bitnet.cpp's 2-bit signed packing (2 bpw, no per-block
/// scale).  Used by `microsoft/bitnet-b1.58-2B-4T-gguf`.  Per-channel
/// scale lives in adjacent `*_sub_norm.weight` F32 tensors.
pub const TYPE_I2_S: u32 = 36;

/// `GGML_TYPE_NVFP4`. 64-element blocks of four UE4M3 scales and
/// thirty-two E2M1 code bytes. The per-tensor scale GGML has no room
/// for travels as a sibling `.scale` tensor — see `quant::nvfp4_ggml`.
pub const TYPE_NVFP4: u32 = 40;

// ── Block geometry (canonical GGML wire format) ─────────────────────────
//
// Legacy quants (Q4_0/Q4_1/Q5_0/Q5_1/Q8_0) pack 32 elements per block.
// K-quants (Q4_K/Q6_K) pack 256 elements per super-block.
//
// Block byte sizes are exact and must never be rederived inline — they
// are part of the on-disk wire format. Q4_K and Q4_0 happen to share the
// same effective rate (0.5625 B/elem), which is exactly why we silently
// shipped a Q4_K file that the reader dispatched as Q4_0 once. Constants
// remove that footgun: callers compare to `Q4_K_BLOCK_BYTES` directly.

/// Elements per block for legacy quants (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0).
pub const LEGACY_BLOCK_ELEMS: usize = 32;

/// Elements per super-block for K-quants (Q4_K, Q6_K).
pub const K_QUANT_BLOCK_ELEMS: usize = 256;

/// Row stride, in elements, of a k-quant row-padded matrix with `cols`
/// logical columns. The k-quant writers pad each row to the next
/// super-block boundary (`pad_rows_to_block`) so every row starts on a
/// block boundary; readers must index rows at THIS stride — decoding a
/// row-padded slab at the unpadded width silently shifts every row
/// after the first whenever `cols % 256 != 0` (e.g. GPT-OSS-20B
/// hidden=2880, Gemma 3 1B hidden=1152).
pub fn k_quant_padded_cols(cols: usize) -> usize {
    cols.div_ceil(K_QUANT_BLOCK_ELEMS) * K_QUANT_BLOCK_ELEMS
}

/// Bytes per Q4_0 block (32 elements + f16 scale): 2 + 16.
pub const Q4_0_BLOCK_BYTES: usize = 18;
/// Elements per Q4_0 block.
pub const Q4_0_BLOCK_ELEMS: usize = LEGACY_BLOCK_ELEMS;

/// Bytes per Q4_1 block (32 elements + f16 scale + f16 min): 2 + 2 + 16.
pub const Q4_1_BLOCK_BYTES: usize = 20;

/// Bytes per Q5_0 block (32 elements + f16 scale + 4-byte high-bits + 16 nibbles).
pub const Q5_0_BLOCK_BYTES: usize = 22;

/// Bytes per Q5_1 block (32 elements + f16 scale + f16 min + 4-byte high-bits + 16 nibbles).
pub const Q5_1_BLOCK_BYTES: usize = 24;

/// Bytes per Q8_0 block (32 elements + f16 scale): 2 + 32.
pub const Q8_0_BLOCK_BYTES: usize = 34;

/// Bytes per Q4_K super-block (256 elements): 2 + 2 + 12 + 128.
///
/// Layout: f16 d (2) + f16 dmin (2) + 12 packed (scale, min) bytes + 128 nibble bytes.
pub const Q4_K_BLOCK_BYTES: usize = 144;
/// Elements per Q4_K super-block.
pub const Q4_K_BLOCK_ELEMS: usize = K_QUANT_BLOCK_ELEMS;

/// Bytes per Q3_K super-block (256 elements): 32 hmask + 64 qs + 12 scales + 2 d.
pub const Q3_K_BLOCK_BYTES: usize = 110;
/// Elements per Q3_K super-block.
pub const Q3_K_BLOCK_ELEMS: usize = K_QUANT_BLOCK_ELEMS;

/// Bytes per Q5_K super-block (256 elements): 2 d + 2 dmin + 12 scales + 32 qh + 128 qs.
pub const Q5_K_BLOCK_BYTES: usize = 176;
/// Elements per Q5_K super-block.
pub const Q5_K_BLOCK_ELEMS: usize = K_QUANT_BLOCK_ELEMS;

/// Bytes per Q6_K super-block (256 elements): 128 + 64 + 16 + 2.
pub const Q6_K_BLOCK_BYTES: usize = 210;
/// Elements per Q6_K super-block.
pub const Q6_K_BLOCK_ELEMS: usize = K_QUANT_BLOCK_ELEMS;

/// Bytes per TQ1_0 super-block (256 elements): 48 (qs) + 4 (qh) + 2 (d) = 54.
/// 1.6875 bits per weight — BitNet 1.58 5-trits-per-byte base-3 layout.
pub const TQ1_0_BLOCK_BYTES: usize = 54;
/// Elements per TQ1_0 super-block.
pub const TQ1_0_BLOCK_ELEMS: usize = K_QUANT_BLOCK_ELEMS;

/// Bytes per TQ2_0 super-block (256 elements): 64 (qs) + 2 (d) = 66.
/// 2.0625 bits per weight — BitNet 1.58 4-trits-per-byte 2-bit layout.
pub const TQ2_0_BLOCK_BYTES: usize = 66;
/// Elements per TQ2_0 super-block.
pub const TQ2_0_BLOCK_ELEMS: usize = K_QUANT_BLOCK_ELEMS;

/// I2_S has no super-block: it is a flat 2-bit-per-weight packing
/// where every 4 weights occupy one byte.  We treat the unit as 4
/// elements per "block" so size and dispatch helpers stay symmetric
/// with the K-quants above.
pub const I2_S_BLOCK_BYTES: usize = 1;
pub const I2_S_BLOCK_ELEMS: usize = 4;

/// Validate that `data` holds at least `n_blocks` blocks of
/// `block_size` bytes for `n_elements` total elements (which must be a
/// multiple of `block_elems`). Returns the block count.
///
/// Checks `data.len() >= need` (not `==`) so callers can pass
/// over-sized buffers — the safetensors loader hands us slices that
/// sometimes carry trailing padding from the next tensor.
pub(crate) fn check_block_input(
    name: &'static str,
    data: &[u8],
    n_elements: usize,
    block_elems: usize,
    block_size: usize,
) -> Result<usize, ModelError> {
    if !n_elements.is_multiple_of(block_elems) {
        return Err(ModelError::Parse(format!(
            "{name}: n_elements {n_elements} not a multiple of {block_elems}"
        )));
    }
    let n_blocks = n_elements / block_elems;
    let need = n_blocks.checked_mul(block_size).ok_or_else(|| {
        ModelError::Parse(format!(
            "{name}: byte-size overflow ({n_blocks} blocks × {block_size} bytes)"
        ))
    })?;
    if data.len() < need {
        return Err(ModelError::Parse(format!(
            "{name}: data too short: {} bytes < expected {} ({} blocks × {} bytes)",
            data.len(),
            need,
            n_blocks,
            block_size
        )));
    }
    Ok(n_blocks)
}

/// Bytes occupied by `n_elements` quantised at `tensor_type`.
///
/// Refuses a count that is not a whole number of blocks, and any size that
/// overflows `usize`: both arise only from a malformed header.
pub fn tensor_data_size(tensor_type: u32, n_elements: usize) -> Result<usize, ModelError> {
    let blocks = |block_elems: usize, block_bytes: usize| -> Result<usize, ModelError> {
        if !n_elements.is_multiple_of(block_elems) {
            return Err(ModelError::Parse(format!(
                "tensor type {tensor_type}: {n_elements} elements is not a multiple of \
                 its {block_elems}-element block"
            )));
        }
        (n_elements / block_elems)
            .checked_mul(block_bytes)
            .ok_or_else(|| {
                ModelError::Parse(format!(
                    "tensor type {tensor_type}: byte size of {n_elements} elements overflows usize"
                ))
            })
    };
    match tensor_type {
        TYPE_F32 => blocks(1, std::mem::size_of::<f32>()),
        TYPE_F16 | TYPE_BF16 => blocks(1, std::mem::size_of::<u16>()),
        TYPE_Q4_0 => blocks(LEGACY_BLOCK_ELEMS, Q4_0_BLOCK_BYTES),
        TYPE_Q4_1 => blocks(LEGACY_BLOCK_ELEMS, Q4_1_BLOCK_BYTES),
        TYPE_Q5_0 => blocks(LEGACY_BLOCK_ELEMS, Q5_0_BLOCK_BYTES),
        TYPE_Q5_1 => blocks(LEGACY_BLOCK_ELEMS, Q5_1_BLOCK_BYTES),
        TYPE_Q8_0 => blocks(LEGACY_BLOCK_ELEMS, Q8_0_BLOCK_BYTES),
        TYPE_Q3_K => blocks(K_QUANT_BLOCK_ELEMS, Q3_K_BLOCK_BYTES),
        TYPE_Q4_K => blocks(K_QUANT_BLOCK_ELEMS, Q4_K_BLOCK_BYTES),
        TYPE_Q5_K => blocks(K_QUANT_BLOCK_ELEMS, Q5_K_BLOCK_BYTES),
        TYPE_Q6_K => blocks(K_QUANT_BLOCK_ELEMS, Q6_K_BLOCK_BYTES),
        TYPE_TQ1_0 => blocks(K_QUANT_BLOCK_ELEMS, TQ1_0_BLOCK_BYTES),
        TYPE_TQ2_0 => blocks(K_QUANT_BLOCK_ELEMS, TQ2_0_BLOCK_BYTES),
        TYPE_NVFP4 => crate::quant::nvfp4_ggml::ggml_nvfp4_bytes(n_elements),
        TYPE_I2_S => {
            if !n_elements.is_multiple_of(I2_S_BLOCK_ELEMS) {
                return Err(ModelError::Parse(format!(
                    "I2_S: n_elements {n_elements} not a multiple of 4"
                )));
            }
            Ok(n_elements / I2_S_BLOCK_ELEMS)
        }
        _ => Err(ModelError::Parse(format!(
            "tensor_data_size: unsupported type id {tensor_type}"
        ))),
    }
}

/// Human-readable name for a GGML tensor type. Returns `"unknown"`
/// (lowercase) for unrecognised ids — tests pin this casing.
pub fn type_name(tensor_type: u32) -> &'static str {
    match tensor_type {
        TYPE_F32 => "F32",
        TYPE_F16 => "F16",
        TYPE_Q4_0 => "Q4_0",
        TYPE_Q4_1 => "Q4_1",
        TYPE_Q8_0 => "Q8_0",
        TYPE_Q5_0 => "Q5_0",
        TYPE_Q5_1 => "Q5_1",
        TYPE_Q2_K => "Q2_K",
        TYPE_Q3_K => "Q3_K",
        TYPE_Q4_K => "Q4_K",
        TYPE_Q5_K => "Q5_K",
        TYPE_Q6_K => "Q6_K",
        TYPE_TQ1_0 => "TQ1_0",
        TYPE_TQ2_0 => "TQ2_0",
        TYPE_I2_S => "I2_S",
        TYPE_BF16 => "BF16",
        _ => "unknown",
    }
}

/// Dequantize raw bytes to f32 based on GGML tensor type.
///
/// Returns `ModelError::Parse` if `data` is too short for the
/// requested number of elements rather than panicking on a slice OOB.
pub fn dequantize(
    data: &[u8],
    tensor_type: u32,
    n_elements: usize,
) -> Result<Vec<f32>, ModelError> {
    match tensor_type {
        TYPE_F32 => {
            let need = n_elements
                .checked_mul(4)
                .ok_or_else(|| ModelError::Parse(format!("F32: size overflow ({n_elements}×4)")))?;
            if data.len() < need {
                return Err(ModelError::Parse(format!(
                    "F32: data too short: {} bytes < expected {need} ({n_elements} elements)",
                    data.len()
                )));
            }
            Ok(data[..need]
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect())
        }
        TYPE_F16 => decode_passthrough(data, n_elements, "F16", decode_f16),
        TYPE_BF16 => decode_passthrough(data, n_elements, "BF16", decode_bf16),
        TYPE_Q4_0 => dequantize_q4_0(data, n_elements),
        TYPE_Q4_1 => legacy::dequantize_q4_1(data, n_elements),
        TYPE_Q8_0 => legacy::dequantize_q8_0(data, n_elements),
        TYPE_Q5_0 => dequantize_q5_0(data, n_elements),
        TYPE_Q5_1 => dequantize_q5_1(data, n_elements),
        TYPE_Q3_K => dequantize_q3_k(data, n_elements),
        TYPE_Q4_K => dequantize_q4_k(data, n_elements),
        TYPE_Q5_K => dequantize_q5_k(data, n_elements),
        TYPE_Q6_K => dequantize_q6_k(data, n_elements),
        TYPE_TQ1_0 => tq::dequantize_tq1_0(data, n_elements),
        TYPE_TQ2_0 => tq::dequantize_tq2_0(data, n_elements),
        TYPE_I2_S => tq::dequantize_i2_s(data, n_elements),
        other => Err(ModelError::UnsupportedDtype(format!("GGML type {other}"))),
    }
}

/// Bounds-checked decode of an f16 / bf16 byte slice via the supplied
/// half-precision decoder.
#[inline]
fn decode_passthrough(
    data: &[u8],
    n_elements: usize,
    name: &'static str,
    decoder: fn(&[u8]) -> Vec<f32>,
) -> Result<Vec<f32>, ModelError> {
    let need = n_elements
        .checked_mul(2)
        .ok_or_else(|| ModelError::Parse(format!("{name}: size overflow ({n_elements}×2)")))?;
    if data.len() < need {
        return Err(ModelError::Parse(format!(
            "{name}: data too short: {} bytes < expected {need} ({n_elements} elements)",
            data.len()
        )));
    }
    Ok(decoder(&data[..need]))
}

#[cfg(test)]
mod tests;
