//! Batched (multi-row) matmul encoders.

use crate::MetalBackend;
use metal::{Buffer, ComputeCommandEncoderRef};

#[allow(unused_imports)]
use super::*;

/// VERIFY-N: the multi-RHS arm each dispatch width uses, as an index into
/// [`MATMUL_ARMS`](crate::shaders::nvfp4_matvec::MATMUL_ARMS), widest
/// first. x2 through 4 rows; x4 at 8, where sharing each X load across
/// four weight rows measured 4.7x vs x2_r8's 6.3x over one Gemma 3 layer
/// (`examples/nvfp4_verify_widths.rs`).
pub(super) const MATMUL_ROWS_ARMS: [(usize, usize); 4] = [(8, 5), (4, 2), (2, 1), (1, 0)];

/// Positions at which a verify block switches to the tiled simdgroup-matrix
/// kernels. The tiled kernels always compute 8 positions (zero-padded), so
/// their cost is nearly flat in R; the multi-RHS arms grow with R. Measured
/// crossover on Gemma 3 4B shapes: multi-RHS wins at R<=4 (1.8-2.1x vs
/// 2.2-2.5x), tiled at R=8 (2.5-2.9x vs 4.1-4.8x).
pub(super) const TILED_MIN_ROWS: usize = 5;
/// Output rows below which the f16 tiled kernel is not used: it walks K
/// serially per simdgroup, so it only wins when the matrix is wide enough
/// to fill the device with threadgroups (the LM head: 1.22x vs 1.57x at
/// R=8; on a 10K-row projection it loses 2.8x vs 1.7x).
pub(super) const F16_TILED_MIN_OUT_ROWS: usize = 65536;

/// The f16 arms by width, as indices into
/// [`f16_gemv::MATMUL_ARMS`](crate::shaders::f16_gemv::MATMUL_ARMS).
pub(super) const F16_MATMUL_ROWS_ARMS: [(usize, usize); 3] = [(8, 2), (4, 1), (2, 0)];

impl MetalBackend {
    /// VERIFY-N: `out[r] = W · x[r]` for `at.rows` positions, the weight
    /// stream read once per dispatch rather than once per position. The
    /// rows are covered by the widest arms that fit (7 = 4 + 2 + 1).
    ///
    /// Blocks of [`TILED_MIN_ROWS`]..=8 positions run the tiled
    /// simdgroup-matrix kernels (one dispatch); narrower blocks run the
    /// multi-RHS arms. NVFP4 and f16 have both; MXFP4 is refused by name.
    /// A matrix run one GEMV per row would present the block's cost as
    /// amortised when it is not, so no such fallback exists.
    pub fn encode_matmul_rows(
        &self,
        enc: &ComputeCommandEncoderRef,
        w: &LoweredMatrix<'_>,
        at: &MatmulRowsTarget<'_>,
    ) -> Result<(), String> {
        let f = std::mem::size_of::<f32>() as u64;
        let tiled =
            (TILED_MIN_ROWS..=crate::shaders::nvfp4_matvec::MATMUL_SGK_MAX_ROWS).contains(&at.rows);
        match w {
            LoweredMatrix::Nvfp4 {
                packed,
                packed_offset,
                scales,
                scales_offset,
                tensor_scale,
            } if tiled => {
                self.encode_nvfp4_matmul_tiled(
                    enc,
                    packed,
                    *packed_offset,
                    scales,
                    *scales_offset,
                    *tensor_scale,
                    at,
                );
                return Ok(());
            }
            LoweredMatrix::F16 { bytes }
                if tiled
                    && at.n >= F16_TILED_MIN_OUT_ROWS
                    && at.n.is_multiple_of(8)
                    && at.k.is_multiple_of(crate::shaders::f16_gemv::MATMUL_SG_KC) =>
            {
                self.encode_f16_matmul_sg(enc, bytes, at);
                return Ok(());
            }
            _ => {}
        }
        let (packed, packed_offset, scales, scales_offset, tensor_scale) = match w {
            LoweredMatrix::Nvfp4 {
                packed,
                packed_offset,
                scales,
                scales_offset,
                tensor_scale,
            } => (packed, packed_offset, scales, scales_offset, tensor_scale),
            // No multi-RHS kernel for these yet: one GEMV per row, so the
            // block pays this matrix's weight stream `rows` times.
            // Correct, and visible — `unamortised_rows` counts it.
            LoweredMatrix::F16 { bytes } => {
                let mut done = 0usize;
                while done < at.rows {
                    let left = at.rows - done;
                    // Widest multi-RHS arm that fits; a lone row runs the
                    // production GEMV.
                    let (width, kernel) =
                        match F16_MATMUL_ROWS_ARMS.iter().find(|(w, _)| *w <= left) {
                            Some(&(w, arm)) => (w, &self.f16_matmul_pipelines[arm]),
                            None => (1, &self.f16_gemv_pipeline),
                        };
                    enc.set_compute_pipeline_state(&kernel.state);
                    enc.set_buffer(0, Some(*bytes), 0);
                    enc.set_buffer(1, Some(at.x), at.x_offset + (done * at.k) as u64 * f);
                    enc.set_buffer(2, Some(at.out), at.out_offset + (done * at.n) as u64 * f);
                    set_u32(enc, 3, at.n as u32);
                    set_u32(enc, 4, at.k as u32);
                    enc.dispatch_thread_groups(
                        metal::MTLSize::new((at.n as u64).div_ceil(kernel.rows_per_tg), 1, 1),
                        metal::MTLSize::new(kernel.threads_per_tg, 1, 1),
                    );
                    done += width;
                }
                return Ok(());
            }
            LoweredMatrix::Mxfp4 { .. } => {
                return Err("multi-position matmul has no MXFP4 path yet".into());
            }
        };
        let mut done = 0usize;
        while done < at.rows {
            let left = at.rows - done;
            let &(width, arm) = MATMUL_ROWS_ARMS
                .iter()
                .find(|(w, _)| *w <= left)
                .expect("width 1 always fits");
            let kernel = &self.quant.nvfp4_matmul_pipelines[arm];
            enc.set_compute_pipeline_state(&kernel.state);
            enc.set_buffer(0, Some(*packed), *packed_offset);
            enc.set_buffer(1, Some(*scales), *scales_offset);
            enc.set_buffer(2, Some(at.x), at.x_offset + (done * at.k) as u64 * f);
            enc.set_buffer(3, Some(at.out), at.out_offset + (done * at.n) as u64 * f);
            set_u32(enc, 4, at.n as u32);
            set_u32(enc, 5, at.k as u32);
            set_f32(enc, 6, *tensor_scale);
            enc.dispatch_thread_groups(
                metal::MTLSize::new((at.n as u64).div_ceil(kernel.rows_per_tg), 1, 1),
                metal::MTLSize::new(kernel.threads_per_tg, 1, 1),
            );
            done += width;
        }
        Ok(())
    }
}

impl MetalBackend {
    /// VERIFY-N: the production tiled NVFP4 matmul
    /// ([`SGF_PRODUCTION_ARM`](crate::shaders::nvfp4_matvec::SGF_PRODUCTION_ARM))
    /// over up to 8 positions, bound at the matrix's slice offsets.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_nvfp4_matmul_tiled(
        &self,
        enc: &ComputeCommandEncoderRef,
        packed: &Buffer,
        packed_offset: u64,
        scales: &Buffer,
        scales_offset: u64,
        tensor_scale: f32,
        at: &MatmulRowsTarget<'_>,
    ) {
        let kernel = &self.quant.nvfp4_matmul_tiled_pipeline;
        enc.set_compute_pipeline_state(&kernel.state);
        enc.set_buffer(0, Some(packed), packed_offset);
        enc.set_buffer(1, Some(scales), scales_offset);
        enc.set_buffer(2, Some(at.x), at.x_offset);
        enc.set_buffer(3, Some(at.out), at.out_offset);
        set_u32(enc, 4, at.n as u32);
        set_u32(enc, 5, at.k as u32);
        set_f32(enc, 6, tensor_scale);
        set_u32(enc, 7, at.rows as u32);
        enc.dispatch_thread_groups(
            metal::MTLSize::new((at.n as u64).div_ceil(kernel.rows_per_tg), 1, 1),
            metal::MTLSize::new(kernel.threads_per_tg, 1, 1),
        );
    }
}

impl MetalBackend {
    /// VERIFY-N: the split-K simdgroup-matrix NVFP4 matmul over up to
    /// [`MATMUL_SGK_MAX_ROWS`](crate::shaders::nvfp4_matvec::MATMUL_SGK_MAX_ROWS)
    /// positions, bound at the matrix's slice offsets. Requires
    /// `at.k % 16 == 0`.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_nvfp4_matmul_sgk(
        &self,
        enc: &ComputeCommandEncoderRef,
        packed: &Buffer,
        packed_offset: u64,
        scales: &Buffer,
        scales_offset: u64,
        tensor_scale: f32,
        at: &MatmulRowsTarget<'_>,
        sgf_arm: Option<usize>,
    ) {
        let kernel = match sgf_arm {
            Some(i) => &self.quant.nvfp4_matmul_sgf_pipelines[i],
            None => &self.quant.nvfp4_matmul_sgk_pipeline,
        };
        enc.set_compute_pipeline_state(&kernel.state);
        enc.set_buffer(0, Some(packed), packed_offset);
        enc.set_buffer(1, Some(scales), scales_offset);
        enc.set_buffer(2, Some(at.x), at.x_offset);
        enc.set_buffer(3, Some(at.out), at.out_offset);
        set_u32(enc, 4, at.n as u32);
        set_u32(enc, 5, at.k as u32);
        set_f32(enc, 6, tensor_scale);
        set_u32(enc, 7, at.rows as u32);
        enc.dispatch_thread_groups(
            metal::MTLSize::new((at.n as u64).div_ceil(kernel.rows_per_tg), 1, 1),
            metal::MTLSize::new(kernel.threads_per_tg, 1, 1),
        );
    }
}

impl MetalBackend {
    /// VERIFY-N: the tiled simdgroup-matrix f16 matmul over up to
    /// [`MATMUL_SG_MAX_ROWS`](crate::shaders::f16_gemv::MATMUL_SG_MAX_ROWS)
    /// positions. Requires `at.n % 8 == 0` and
    /// `at.k % `[`MATMUL_SG_KC`](crate::shaders::f16_gemv::MATMUL_SG_KC)` == 0`
    /// — lowering geometry, checked by the caller that picks the arm.
    pub fn encode_f16_matmul_sg(
        &self,
        enc: &ComputeCommandEncoderRef,
        w: &Buffer,
        at: &MatmulRowsTarget<'_>,
    ) {
        let kernel = &self.f16_matmul_sg_pipeline;
        enc.set_compute_pipeline_state(&kernel.state);
        enc.set_buffer(0, Some(w), 0);
        enc.set_buffer(1, Some(at.x), at.x_offset);
        enc.set_buffer(2, Some(at.out), at.out_offset);
        set_u32(enc, 3, at.n as u32);
        set_u32(enc, 4, at.k as u32);
        set_u32(enc, 5, at.rows as u32);
        enc.dispatch_thread_groups(
            metal::MTLSize::new((at.n as u64).div_ceil(kernel.rows_per_tg), 1, 1),
            metal::MTLSize::new(kernel.threads_per_tg, 1, 1),
        );
    }
}
