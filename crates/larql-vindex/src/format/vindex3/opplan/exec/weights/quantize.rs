//! NVFP4/MXFP4 quantisation of f32 weights.

use super::super::backend::Nvfp4Activation;
use crate::error::VindexError;
use larql_models::quant::mxfp4::MXFP4_TABLE;

#[allow(unused_imports)]
use super::*;

/// Quantise one `[rows, k]` f32 matrix to NVFP4 into page-aligned
/// buffers, delegating the numerics to `larql_models::quant::nvfp4` so
/// the format has exactly one definition — the CPU reference, this
/// loader, and the Metal kernel all read that module's contract.
///
/// The only thing added here is residency: the same page-aligned
/// allocation MXFP4 uses, so a device can wrap the buffers zero-copy.
pub fn quantize_nvfp4(
    values: &[f32],
    rows: usize,
    k: usize,
    name: &str,
) -> Result<LoadedWeight, VindexError> {
    use larql_models::quant::nvfp4::{
        quantize_row_into, tensor_scale_for, NVFP4_GROUP_BYTES, NVFP4_GROUP_ELEMS,
    };
    if !k.is_multiple_of(NVFP4_GROUP_ELEMS) {
        return Err(VindexError::Parse(format!(
            "tensor `{name}`: k={k} is not a multiple of the NVFP4 \
             {NVFP4_GROUP_ELEMS}-element group"
        )));
    }
    if values.len() != rows * k {
        return Err(VindexError::Parse(format!(
            "tensor `{name}`: {} values do not fill [{rows}, {k}]",
            values.len()
        )));
    }
    let groups = k / NVFP4_GROUP_ELEMS;
    // The tensor scale is a property of the whole matrix, so it is chosen
    // once before any row is encoded — rows cannot each pick their own
    // and still decode under one shared scale.
    let tensor_scale = tensor_scale_for(values);
    let mut packed = AlignedBytes::zeroed(rows * groups * NVFP4_GROUP_BYTES);
    let mut scales = AlignedBytes::zeroed(rows * groups);
    {
        use rayon::prelude::*;
        let packed_dst = packed.as_mut_slice();
        let scales_dst = scales.as_mut_slice();
        // Rows are independent given the tensor scale, so the parallelism
        // lives here while the numerics stay in one place
        // (`quant::nvfp4::quantize_row_into`), shared with the CPU
        // reference the kernel is judged against.
        packed_dst[..rows * groups * NVFP4_GROUP_BYTES]
            .par_chunks_mut(groups * NVFP4_GROUP_BYTES)
            .zip(scales_dst[..rows * groups].par_chunks_mut(groups))
            .zip(values.par_chunks(k))
            .for_each(|((row_packed, row_scales), row_values)| {
                quantize_row_into(row_values, tensor_scale, row_packed, row_scales);
            });
    }
    Ok(LoadedWeight::Nvfp4 {
        packed,
        scales,
        tensor_scale,
        activation: Nvfp4Activation::F32,
    })
}

/// The e2m1 code nearest to `v` (ties to the even code index),
/// saturating at ±6.
pub(super) fn nearest_mxfp4_code(v: f32) -> u8 {
    let sign = if v.is_sign_negative() { 8u8 } else { 0 };
    let mag = v.abs().min(MXFP4_MAX_MAG);
    let mut best = 0u8;
    let mut best_err = f32::INFINITY;
    for (code, value) in MXFP4_TABLE.iter().enumerate().take(8) {
        let err = (mag - value).abs();
        if err < best_err || (err == best_err && code.is_multiple_of(2)) {
            best = code as u8;
            best_err = err;
        }
    }
    if best == 0 {
        0 // ±0 collapse to +0: the table's -0.0 encodes nothing extra
    } else {
        sign | best
    }
}
