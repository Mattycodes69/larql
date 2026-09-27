//! `MatMul` impl + private encoder helpers shared by `f32_gemv` and
//! `f16_gemv` (threshold-gated and force variants).

use ndarray::{Array2, ArrayView2};
use std::sync::atomic::Ordering;

use crate::MetalBackend;
use larql_compute::backend::{MatMul, MatMulOp};

mod dispatch;

impl MatMul for MetalBackend {
    fn matmul(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
        self.f32_ops.matmul(
            &self.queue,
            &self.bufs,
            a,
            b,
            self.flop_threshold.load(Ordering::Relaxed),
        )
    }

    fn matmul_transb(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
        self.f32_ops.matmul_transb(
            &self.queue,
            &self.bufs,
            a,
            b,
            self.flop_threshold.load(Ordering::Relaxed),
        )
    }

    fn f32_gemv(&self, w: ArrayView2<f32>, x: &[f32]) -> Option<Vec<f32>> {
        let (n, k) = (w.shape()[0], w.shape()[1]);
        if x.len() != k {
            return None;
        }
        // Fall back below the GPU threshold — small gemvs are dominated by
        // dispatch overhead.
        if 2 * n * k < self.flop_threshold.load(Ordering::Relaxed) {
            return None;
        }
        self.encode_f32_gemv(w, x)
    }

    fn f32_gemv_force(&self, w: ArrayView2<f32>, x: &[f32]) -> Option<Vec<f32>> {
        let (_n, k) = (w.shape()[0], w.shape()[1]);
        if x.len() != k {
            return None;
        }
        self.encode_f32_gemv(w, x)
    }

    fn f16_gemv(&self, w_f16: &[u8], x: &[f32], n: usize, k: usize) -> Option<Vec<f32>> {
        if w_f16.len() < n * k * 2 || x.len() != k {
            return None;
        }
        if 2 * n * k < self.flop_threshold.load(Ordering::Relaxed) {
            return None;
        }
        self.encode_f16_gemv(w_f16, x, n, k)
    }

    fn f16_gemv_force(&self, w_f16: &[u8], x: &[f32], n: usize, k: usize) -> Option<Vec<f32>> {
        if w_f16.len() < n * k * 2 || x.len() != k {
            return None;
        }
        self.encode_f16_gemv(w_f16, x, n, k)
    }

    /// bf16 weights (the top 16 bits of each f32), threshold-gated.
    /// Encoder body and shape rule live in [`super::bf16_gemv`].
    fn bf16_gemv(&self, w_bf16: &[u8], x: &[f32], n: usize, k: usize) -> Option<Vec<f32>> {
        if !Self::bf16_gemv_shape_ok(w_bf16, x, n, k) {
            return None;
        }
        if 2 * n * k < self.flop_threshold.load(Ordering::Relaxed) {
            return None;
        }
        self.encode_bf16_gemv(w_bf16, x, n, k)
    }

    fn bf16_gemv_force(&self, w_bf16: &[u8], x: &[f32], n: usize, k: usize) -> Option<Vec<f32>> {
        if !Self::bf16_gemv_shape_ok(w_bf16, x, n, k) {
            return None;
        }
        self.encode_bf16_gemv(w_bf16, x, n, k)
    }

    /// N bf16 gemvs over one shared input as one submission. Bit-identical
    /// to N sequential `bf16_gemv_force` calls; see [`super::bf16_gemv`].
    fn bf16_gemv_multi(
        &self,
        weights: &[(&[u8], usize, usize)],
        x: &[f32],
    ) -> Option<Vec<Vec<f32>>> {
        for &(w, n, k) in weights {
            if !Self::bf16_gemv_shape_ok(w, x, n, k) {
                return None;
            }
        }
        if weights.is_empty() {
            return Some(Vec::new());
        }
        self.encode_bf16_gemv_multi(weights, x)
    }

    /// One command buffer, one encoder, one input upload, N dispatches,
    /// one wait — same kernel and same per-dispatch arguments as the
    /// sequential path, so the results are bit-identical to N separate
    /// [`Self::f16_gemv_force`] calls. The dispatches are mutually
    /// independent (distinct output buffers), so encoding them without
    /// barriers reorders nothing observable.
    fn f16_gemv_multi(
        &self,
        weights: &[(&[u8], usize, usize)],
        x: &[f32],
    ) -> Option<Vec<Vec<f32>>> {
        for &(w, n, k) in weights {
            if w.len() < n * k * 2 || x.len() != k {
                return None;
            }
        }
        if weights.is_empty() {
            return Some(Vec::new());
        }

        let x_buf = self.bufs.output((x.len() * 4) as u64);
        let x_ptr = x_buf.contents() as *mut f32;
        if x_ptr.is_null() {
            return None;
        }
        // SAFETY: pooled buffer is at least x.len()*4 bytes and the GPU
        // is not reading it — it has not been bound to any encoder yet.
        unsafe { std::ptr::copy_nonoverlapping(x.as_ptr(), x_ptr, x.len()) };

        let kernel = &self.f16_gemv_pipeline;
        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&kernel.state);
        // Weight buffers must outlive the wait; the cache clone in this
        // vec guarantees it independent of encoder retention semantics.
        let mut w_bufs = Vec::with_capacity(weights.len());
        let mut out_bufs = Vec::with_capacity(weights.len());
        for &(w, n, k) in weights {
            let w_buf = self.bufs.get_bytes(w);
            let out_buf = self.bufs.output((n * 4) as u64);
            let n_u32 = n as u32;
            let k_u32 = k as u32;
            enc.set_buffer(0, Some(&w_buf), 0);
            enc.set_buffer(1, Some(&x_buf), 0);
            enc.set_buffer(2, Some(&out_buf), 0);
            enc.set_bytes(3, 4, &n_u32 as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(4, 4, &k_u32 as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                metal::MTLSize::new((n as u64).div_ceil(kernel.rows_per_tg), 1, 1),
                metal::MTLSize::new(kernel.threads_per_tg, 1, 1),
            );
            w_bufs.push(w_buf);
            out_bufs.push((out_buf, n));
        }
        enc.end_encoding();
        self.commit_and_wait(
            cmd,
            "crates/larql-compute-metal/src/trait_impl/matmul.rs:125",
        );

        let results: Option<Vec<Vec<f32>>> = out_bufs
            .iter()
            .map(|(buf, n)| crate::buffers::try_read_buffer_f32(buf, *n))
            .collect();
        // Recycle after the wait and the copy-out, per the pool contract.
        for (buf, _) in out_bufs {
            self.bufs.recycle(buf);
        }
        self.bufs.recycle(x_buf);
        results
    }

    /// One command buffer that references every buffer so the driver
    /// wires them all in one pass (see the trait doc). The single 1×1
    /// gemv gives the encoder real work — an encoder with no dispatch
    /// may not trigger residency — and reads two bytes of the first
    /// buffer without writing anywhere a caller can see.
    fn submission_clock(&self) -> Option<larql_compute::SubmissionClock> {
        Some(self.submission_counters.snapshot())
    }

    fn wire_resident(&self, buffers: &[&[u8]]) {
        let Some(first) = buffers.first() else {
            return;
        };
        if first.len() < 2 {
            return;
        }
        let wired: Vec<metal::Buffer> = buffers.iter().map(|b| self.bufs.get_bytes(b)).collect();
        let x_buf = self.bufs.output(4);
        let x_ptr = x_buf.contents() as *mut f32;
        if x_ptr.is_null() {
            return;
        }
        // SAFETY: pooled buffer holds at least one f32; not yet bound.
        unsafe { *x_ptr = 0.0 };
        let out_buf = self.bufs.output(4);

        let kernel = &self.f16_gemv_pipeline;
        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&kernel.state);
        for buf in &wired {
            enc.use_resource(buf, metal::MTLResourceUsage::Read);
        }
        let n_u32: u32 = 1;
        let k_u32: u32 = 1;
        enc.set_buffer(0, Some(&wired[0]), 0);
        enc.set_buffer(1, Some(&x_buf), 0);
        enc.set_buffer(2, Some(&out_buf), 0);
        enc.set_bytes(3, 4, &n_u32 as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(4, 4, &k_u32 as *const u32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(
            metal::MTLSize::new(1, 1, 1),
            metal::MTLSize::new(kernel.threads_per_tg, 1, 1),
        );
        enc.end_encoding();
        self.commit_and_wait(
            cmd,
            "crates/larql-compute-metal/src/trait_impl/matmul.rs:181",
        );
        self.bufs.recycle(out_buf);
        self.bufs.recycle(x_buf);
    }

    /// Routed through [`MatMul::mxfp4_gemv_multi`] as a one-element batch
    /// rather than through `mxfp4_matmul_small_block`.
    ///
    /// Same kernel and same arguments either way, but the small-block
    /// helper builds a *transient* input buffer per call and never
    /// recycles its output back to the pool, so a decode that dispatches
    /// it a few times per layer across 52 layers churns pool allocations
    /// on every token. Measured on the all-MXFP4 preset: 298 ms/token
    /// through the helper against ~105 ms predicted by the byte model
    /// the other presets obey — a ~2.8x penalty that reads as a property
    /// of the *format* if the two paths are compared without noticing.
    /// The helper stays for its `t > 1` block callers.
    fn mxfp4_gemv(
        &self,
        packed: &[u8],
        scales: &[u8],
        x: &[f32],
        n: usize,
        k: usize,
    ) -> Option<Vec<f32>> {
        let results = self.mxfp4_gemv_multi(&[(packed, scales, n, k)], x)?;
        results.into_iter().next()
    }

    /// One command buffer, one encoder, one input upload, N dispatches,
    /// one wait — the MXFP4 mirror of [`MatMul::f16_gemv_multi`], with
    /// two weight planes bound per dispatch.
    fn mxfp4_gemv_multi(
        &self,
        weights: &[(&[u8], &[u8], usize, usize)],
        x: &[f32],
    ) -> Option<Vec<Vec<f32>>> {
        const GROUP_ELEMS: usize = 32;
        const GROUP_BYTES: usize = 16;
        for &(packed, scales, n, k) in weights {
            if !k.is_multiple_of(GROUP_ELEMS) || x.len() < k {
                return None;
            }
            let groups = k / GROUP_ELEMS;
            if packed.len() < n * groups * GROUP_BYTES || scales.len() < n * groups {
                return None;
            }
        }
        if weights.is_empty() {
            return Some(Vec::new());
        }

        let x_buf = self.bufs.output((x.len() * 4) as u64);
        let x_ptr = x_buf.contents() as *mut f32;
        if x_ptr.is_null() {
            return None;
        }
        // SAFETY: pooled buffer is at least x.len()*4 bytes; not bound yet.
        unsafe { std::ptr::copy_nonoverlapping(x.as_ptr(), x_ptr, x.len()) };

        let kernel = &self.quant.mxfp4_matvec_pipeline;
        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&kernel.state);
        let mut held = Vec::with_capacity(weights.len() * 2);
        let mut out_bufs = Vec::with_capacity(weights.len());
        for &(packed, scales, n, k) in weights {
            let p_buf = self.bufs.get_bytes(packed);
            let s_buf = self.bufs.get_bytes(scales);
            let out_buf = self.bufs.output((n * 4) as u64);
            let m_u32 = n as u32;
            let k_u32 = k as u32;
            enc.set_buffer(0, Some(&p_buf), 0);
            enc.set_buffer(1, Some(&s_buf), 0);
            enc.set_buffer(2, Some(&x_buf), 0);
            enc.set_buffer(3, Some(&out_buf), 0);
            enc.set_bytes(4, 4, &m_u32 as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(5, 4, &k_u32 as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                metal::MTLSize::new((n as u64).div_ceil(kernel.rows_per_tg), 1, 1),
                metal::MTLSize::new(kernel.threads_per_tg, 1, 1),
            );
            held.push(p_buf);
            held.push(s_buf);
            out_bufs.push((out_buf, n));
        }
        enc.end_encoding();
        self.commit_and_wait(
            cmd,
            "crates/larql-compute-metal/src/trait_impl/matmul.rs:269",
        );

        let results: Option<Vec<Vec<f32>>> = out_bufs
            .iter()
            .map(|(buf, n)| crate::buffers::try_read_buffer_f32(buf, *n))
            .collect();
        for (buf, _) in out_bufs {
            self.bufs.recycle(buf);
        }
        self.bufs.recycle(x_buf);
        results
    }

    fn nvfp4_gemv(
        &self,
        packed: &[u8],
        scales: &[u8],
        tensor_scale: f32,
        x: &[f32],
        n: usize,
        k: usize,
    ) -> Option<Vec<f32>> {
        let results = self.nvfp4_gemv_multi(&[(packed, scales, tensor_scale, n, k)], x)?;
        results.into_iter().next()
    }

    /// One command buffer, one encoder, one input upload, N dispatches,
    /// one wait — the NVFP4 mirror of [`MatMul::mxfp4_gemv_multi`], with
    /// the per-matrix tensor scale bound as a fourth argument.
    fn nvfp4_gemv_multi(
        &self,
        weights: &[larql_compute::backend::matmul::Nvfp4Operand<'_>],
        x: &[f32],
    ) -> Option<Vec<Vec<f32>>> {
        use larql_models::quant::nvfp4::{NVFP4_GROUP_BYTES, NVFP4_GROUP_ELEMS};
        for &(packed, scales, _, n, k) in weights {
            if !k.is_multiple_of(NVFP4_GROUP_ELEMS) || x.len() < k {
                return None;
            }
            let groups = k / NVFP4_GROUP_ELEMS;
            if packed.len() < n * groups * NVFP4_GROUP_BYTES || scales.len() < n * groups {
                return None;
            }
        }
        if weights.is_empty() {
            return Some(Vec::new());
        }

        let x_buf = self.bufs.output((x.len() * 4) as u64);
        let x_ptr = x_buf.contents() as *mut f32;
        if x_ptr.is_null() {
            return None;
        }
        // SAFETY: pooled buffer is at least x.len()*4 bytes; not bound yet.
        unsafe { std::ptr::copy_nonoverlapping(x.as_ptr(), x_ptr, x.len()) };

        let kernel = &self.quant.nvfp4_matvec_pipeline;
        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&kernel.state);
        let mut held = Vec::with_capacity(weights.len() * 2);
        let mut out_bufs = Vec::with_capacity(weights.len());
        for &(packed, scales, tensor_scale, n, k) in weights {
            let p_buf = self.bufs.get_bytes(packed);
            let s_buf = self.bufs.get_bytes(scales);
            let out_buf = self.bufs.output((n * 4) as u64);
            let m_u32 = n as u32;
            let k_u32 = k as u32;
            enc.set_buffer(0, Some(&p_buf), 0);
            enc.set_buffer(1, Some(&s_buf), 0);
            enc.set_buffer(2, Some(&x_buf), 0);
            enc.set_buffer(3, Some(&out_buf), 0);
            enc.set_bytes(4, 4, &m_u32 as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(5, 4, &k_u32 as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(6, 4, &tensor_scale as *const f32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                metal::MTLSize::new((n as u64).div_ceil(kernel.rows_per_tg), 1, 1),
                metal::MTLSize::new(kernel.threads_per_tg, 1, 1),
            );
            held.push(p_buf);
            held.push(s_buf);
            out_bufs.push((out_buf, n));
        }
        enc.end_encoding();
        self.commit_and_wait(
            cmd,
            "crates/larql-compute-metal/src/trait_impl/matmul.rs:354",
        );

        let results: Option<Vec<Vec<f32>>> = out_bufs
            .iter()
            .map(|(buf, n)| crate::buffers::try_read_buffer_f32(buf, *n))
            .collect();
        for (buf, _) in out_bufs {
            self.bufs.recycle(buf);
        }
        self.bufs.recycle(x_buf);
        results
    }

    fn f32_gemv_topk1(&self, w: ArrayView2<f32>, x: &[f32]) -> Option<(u32, f32)> {
        MetalBackend::f32_gemv_topk1(self, w, x)
    }

    fn f16_gemv_topk1(&self, w_f16: &[u8], x: &[f32], n: usize, k: usize) -> Option<(u32, f32)> {
        MetalBackend::f16_gemv_topk1(self, w_f16, x, n, k)
    }

    fn f16_gemv_topk(
        &self,
        w_f16: &[u8],
        x: &[f32],
        n: usize,
        k: usize,
        top_k: usize,
    ) -> Option<Vec<(u32, f32)>> {
        MetalBackend::f16_gemv_topk(self, w_f16, x, n, k, top_k)
    }

    fn matmul_batch(&self, ops: &[MatMulOp]) -> Vec<Array2<f32>> {
        ops.iter()
            .map(|op| {
                if op.transpose_b {
                    self.matmul_transb(op.a.view(), op.b.view())
                } else {
                    self.matmul(op.a.view(), op.b.view())
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests;
