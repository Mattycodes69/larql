//! Norms, scales, residual adds and the linear dispatch helper.

use crate::MetalBackend;
use metal::{Buffer, ComputeCommandEncoderRef};

#[allow(unused_imports)]
use super::*;

impl MetalBackend {
    /// Encode weightless per-head RMS over `x` **in place**, one
    /// threadgroup per head.
    ///
    /// In place because the interpreter's `qk_norm_in_place` is, and a
    /// lowering that quietly introduced a copy would diverge the moment
    /// a caller relied on aliasing.
    pub fn encode_parameter_free_qk_norm(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &Buffer,
        x_offset: u64,
        num_heads: usize,
        head_dim: usize,
        eps: f32,
    ) {
        let pipeline = &self.norms.qk_norm_parameter_free_pipeline;
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(x), x_offset);
        set_u32(enc, 1, head_dim as u32);
        set_f32(enc, 2, eps);
        // One threadgroup per head; threads cooperate over `head_dim`.
        // Capped at the pipeline's own limit, and at 1024 so the
        // shader's 32-slot simdgroup-partial array cannot overflow.
        let threads = (head_dim as u64)
            .next_power_of_two()
            .clamp(32, pipeline.max_total_threads_per_threadgroup().min(1024));
        enc.dispatch_thread_groups(
            metal::MTLSize::new(num_heads as u64, 1, 1),
            metal::MTLSize::new(threads, 1, 1),
        );
    }

    /// Encode a WEIGHTED per-head RMS norm in place — Gemma's `q_norm` /
    /// `k_norm` (`[head_dim]` weight, `1 + w` when `weight_offset` is 1),
    /// through the served `qk_norm` kernel, one threadgroup per head.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_weighted_qk_norm(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &Buffer,
        x_offset: u64,
        weight: &Buffer,
        num_heads: usize,
        head_dim: usize,
        eps: f32,
        weight_offset: f32,
    ) {
        let pipeline = &self.norms.qk_norm_pipeline;
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(x), x_offset);
        enc.set_buffer(1, Some(x), x_offset);
        enc.set_buffer(2, Some(weight), 0);
        set_u32(enc, 3, head_dim as u32);
        set_u32(enc, 4, num_heads as u32);
        set_f32(enc, 5, eps);
        set_f32(enc, 6, weight_offset);
        // The served stage's geometry: one threadgroup per head, threads a
        // power of two up to 512 covering `head_dim`.
        let threads = (head_dim as u64)
            .next_power_of_two()
            .clamp(1, crate::stages::qk_norm::MAX_TG_WIDTH);
        enc.dispatch_thread_groups(
            metal::MTLSize::new(num_heads as u64, 1, 1),
            metal::MTLSize::new(threads, 1, 1),
        );
    }

    /// Encode `out = a * sigmoid(g)` — the judged attention output gate.
    pub fn encode_sigmoid_gate(
        &self,
        enc: &ComputeCommandEncoderRef,
        a: &Buffer,
        g: &Buffer,
        out: &Buffer,
        len: usize,
    ) {
        let pipeline = &self.norms.sigmoid_gate_multiply_pipeline;
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(a), 0);
        enc.set_buffer(1, Some(g), 0);
        enc.set_buffer(2, Some(out), 0);
        set_u32(enc, 3, len as u32);
        let tg = pipeline
            .max_total_threads_per_threadgroup()
            .clamp(1, crate::kernels::DISPATCH_TG_MAX_THREADS);
        enc.dispatch_thread_groups(
            metal::MTLSize::new((len as u64).div_ceil(tg), 1, 1),
            metal::MTLSize::new(tg, 1, 1),
        );
    }
}

impl MetalBackend {
    /// Encode `x *= scalar` over `len` floats, in place.
    pub fn encode_scale_vector(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &Buffer,
        len: usize,
        scalar: f32,
    ) {
        let pipeline = &self.norms.scale_vector_pipeline;
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(x), 0);
        enc.set_buffer(1, Some(x), 0);
        set_u32(enc, 2, len as u32);
        set_f32(enc, 3, scalar);
        dispatch_linear(enc, pipeline, len);
    }

    /// Encode RoPE over `num_heads` heads at `position`, in place.
    ///
    /// `inv_freq` is host-computed as `theta^(-2i/head_dim)` to match the
    /// interpreter's `rope_rotate`; both use the half-split convention
    /// (`x[i]`, `x[i + head_dim/2]` are the real/imaginary pair), which
    /// is the detail an interleaved-convention kernel would get silently
    /// wrong.
    ///
    /// One position is a one-row [`Self::encode_rope_rows`]: decode and a
    /// VERIFY-N block must rotate through the SAME kernel. `rope_rows` and
    /// `rope_at_pos_batched` share their arithmetic in source, but under
    /// fast math the compiler may lower a uniform `pos` and a per-thread
    /// `pos` differently — the macos-14 runner's GPU does, and the two
    /// disagreed in the last bits. One kernel makes verify == greedy a
    /// property of the code, not of the GPU's compiler.
    ///
    /// The cos/sin amplitude — 1.0 for plain rope, YaRN's
    /// `attention_amplitude` for a scaled layer — comes from the plan's
    /// position policy, never invented here (A-9.4).
    #[allow(clippy::too_many_arguments)]
    pub fn encode_rope(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &Buffer,
        x_offset: u64,
        num_heads: usize,
        head_dim: usize,
        inv_freq: &Buffer,
        position: usize,
        amplitude: f32,
    ) {
        self.encode_rope_rows(
            enc, x, x_offset, num_heads, head_dim, inv_freq, position, amplitude, 1,
        );
    }

    /// Encode `x[off..][i] += bias[i]` over `len` elements — a projection
    /// bias joining its output in place (the same `bias_add` kernel the
    /// decode path uses).
    pub fn encode_bias_add(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &Buffer,
        x_offset: u64,
        bias: &Buffer,
        len: usize,
    ) {
        crate::stages::bias_add::encode(
            enc,
            &self.attention.bias_add_pipeline,
            x,
            x_offset,
            bias,
            len,
        );
    }

    /// Encode `out = a + b_scale * b`.
    pub fn encode_residual_add(
        &self,
        enc: &ComputeCommandEncoderRef,
        a: &Buffer,
        b: &Buffer,
        out: &Buffer,
        len: usize,
        b_scale: f32,
    ) {
        let pipeline = &self.norms.residual_add_pipeline;
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(a), 0);
        enc.set_buffer(1, Some(b), 0);
        enc.set_buffer(2, Some(out), 0);
        set_u32(enc, 3, len as u32);
        set_f32(enc, 4, b_scale);
        dispatch_linear(enc, pipeline, len);
    }
}

/// One thread per element, threadgroups sized from the pipeline's own
/// limit rather than a shader constant.
///
/// Public so an out-of-crate parity test can dispatch a bound kernel with
/// the SAME geometry production uses: a test that computed its own
/// threadgroup size would be checking a dispatch this crate never issues.
pub fn dispatch_linear(
    enc: &ComputeCommandEncoderRef,
    pipeline: &metal::ComputePipelineState,
    len: usize,
) {
    let tg = pipeline
        .max_total_threads_per_threadgroup()
        .clamp(1, crate::kernels::DISPATCH_TG_MAX_THREADS);
    enc.dispatch_thread_groups(
        metal::MTLSize::new((len as u64).div_ceil(tg), 1, 1),
        metal::MTLSize::new(tg, 1, 1),
    );
}

impl MetalBackend {
    /// Encode `out = branch, normalised if the plan carries a post-norm`,
    /// then `h_out = h_in + residual_scale * out` (`residual_scale` is 1.0
    /// when the plan carries no residual-scale op).
    ///
    /// The order is load-bearing and the reason this is one function
    /// rather than two calls at each site: the interpreter normalises the
    /// **branch output** and then adds it to the residual stream. Adding
    /// first and normalising the sum is a different model, and
    /// "post-attention norm" is an ambiguous enough name that a lowering
    /// could plausibly do either.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_branch_norm_then_residual(
        &self,
        enc: &ComputeCommandEncoderRef,
        h_in: &Buffer,
        branch: &Buffer,
        h_out: &Buffer,
        post: Option<&PostNorm<'_>>,
        hidden: usize,
        residual_scale: f32,
    ) {
        let addend = match post {
            Some(p) => {
                crate::stages::input_norm::encode_f32(
                    enc,
                    &self.norms.rms_norm_pipeline,
                    branch,
                    0,
                    p.weight,
                    p.scratch,
                    0,
                    hidden,
                    p.eps,
                    p.weight_offset,
                );
                p.scratch
            }
            None => branch,
        };
        self.encode_residual_add(enc, h_in, addend, h_out, hidden, residual_scale);
    }
}

impl MetalBackend {
    /// VERIFY-N: [`Self::encode_branch_norm_then_residual`] over `rows`
    /// positions — the post-norm (a per-row reduction, one threadgroup per
    /// row) into `post_scratch` (`[rows, hidden]`), then ONE residual add
    /// over the whole block, which is elementwise.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_branch_norm_then_residual_rows(
        &self,
        enc: &ComputeCommandEncoderRef,
        h_in: &Buffer,
        branch: &Buffer,
        h_out: &Buffer,
        post: Option<&PostNorm<'_>>,
        post_scratch: &Buffer,
        hidden: usize,
        rows: usize,
        residual_scale: f32,
    ) {
        let addend = match post {
            Some(p) => {
                self.encode_rms_norm_rows(
                    enc,
                    branch,
                    0,
                    p.weight,
                    post_scratch,
                    0,
                    hidden,
                    rows,
                    p.eps,
                    p.weight_offset,
                );
                post_scratch
            }
            None => branch,
        };
        self.encode_residual_add(enc, h_in, addend, h_out, rows * hidden, residual_scale);
    }
}

impl MetalBackend {
    /// VERIFY-N: RMS norm of `rows` vectors of `len` floats, `[rows, len]`
    /// from `x_offset` into `out` from `out_offset`, in ONE dispatch — one
    /// threadgroup per row at `input_norm::encode_f32`'s width, so each row
    /// is bit-identical to that single-row dispatch.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_rms_norm_rows(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &Buffer,
        x_offset: u64,
        weight: &Buffer,
        out: &Buffer,
        out_offset: u64,
        len: usize,
        rows: usize,
        eps: f32,
        weight_offset: f32,
    ) {
        let pipeline = &self.norms.rms_norm_rows_pipeline;
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(x), x_offset);
        enc.set_buffer(1, Some(weight), 0);
        enc.set_buffer(2, Some(out), out_offset);
        set_u32(enc, 3, len as u32);
        set_f32(enc, 4, eps);
        set_f32(enc, 5, weight_offset);
        enc.dispatch_thread_groups(
            metal::MTLSize::new(rows as u64, 1, 1),
            metal::MTLSize::new(
                crate::kernels::DISPATCH_TG_MAX_THREADS.min(len as u64),
                1,
                1,
            ),
        );
    }

    /// VERIFY-N: [`Self::encode_rope`] over `rows` consecutive positions
    /// from `position` (`[rows, num_heads, head_dim]` from `x_offset`), in
    /// one dispatch.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_rope_rows(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &Buffer,
        x_offset: u64,
        num_heads: usize,
        head_dim: usize,
        inv_freq: &Buffer,
        position: usize,
        amplitude: f32,
        rows: usize,
    ) {
        let pipeline = &self.attention.rope_rows_pipeline;
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(x), x_offset);
        set_u32(enc, 1, head_dim as u32);
        enc.set_buffer(2, Some(inv_freq), 0);
        set_u32(enc, 3, position as u32);
        // rotary_dim 0 = rotate the whole head, matching `rope_rotate`.
        set_u32(enc, 4, 0);
        set_u32(enc, 5, num_heads as u32);
        set_f32(enc, 6, amplitude);
        set_u32(enc, 7, (rows * num_heads) as u32);
        enc.dispatch_thread_groups(
            metal::MTLSize::new((head_dim / 2) as u64, (rows * num_heads) as u64, 1),
            metal::MTLSize::new(1, 1, 1),
        );
    }
}
