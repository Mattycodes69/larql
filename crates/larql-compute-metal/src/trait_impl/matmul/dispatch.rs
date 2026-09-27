//! Inherent MetalBackend matmul dispatch helpers behind the `MatMul` impl.

use crate::MetalBackend;
use ndarray::ArrayView2;

#[allow(unused_imports)]
use super::*;

impl MetalBackend {
    /// Shared GPU dispatch body for `f32_gemv` (threshold-gated) and
    /// `f32_gemv_force` (direct). Kept inherent so the 30+ lines of
    /// Metal plumbing aren't duplicated.
    pub(super) fn encode_f32_gemv(&self, w: ArrayView2<f32>, x: &[f32]) -> Option<Vec<f32>> {
        let (n, k) = (w.shape()[0], w.shape()[1]);
        if x.len() != k {
            return None;
        }
        let w_buf = match w.as_slice() {
            Some(s) => self.bufs.get_f32(s),
            None => {
                let owned = w.as_standard_layout().into_owned();
                self.bufs.transient_from_f32(owned.as_slice().unwrap())
            }
        };
        let x_buf = self.bufs.transient_from_f32(x);
        let out_buf = self.bufs.output((n * 4) as u64);

        // Geometry travels with the f32_gemv KernelHandle.
        let kernel = &self.f32_gemv_pipeline;
        let n_u32 = n as u32;
        let k_u32 = k as u32;
        let num_tgs = (n as u64).div_ceil(kernel.rows_per_tg);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&kernel.state);
        enc.set_buffer(0, Some(&w_buf), 0);
        enc.set_buffer(1, Some(&x_buf), 0);
        enc.set_buffer(2, Some(&out_buf), 0);
        enc.set_bytes(3, 4, &n_u32 as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(4, 4, &k_u32 as *const u32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(
            metal::MTLSize::new(num_tgs, 1, 1),
            metal::MTLSize::new(kernel.threads_per_tg, 1, 1),
        );
        enc.end_encoding();
        self.commit_and_wait(
            cmd,
            "crates/larql-compute-metal/src/trait_impl/matmul.rs:140",
        );

        // `try_read_buffer_f32` returns `None` if the GPU ran out of
        // memory and `contents()` is null — callers fall back to CPU
        // rather than crash. Hit by the lm-head 2.5 GB bench shape on
        // memory-constrained CI runners.
        crate::buffers::try_read_buffer_f32(&out_buf, n)
    }

    /// GPU gemv → GPU argmax, returning (token_id, score) without a 1MB readback.
    ///
    /// Replaces the three-step `f32_gemv` + read 262K floats + CPU argmax with:
    /// 1. f32_gemv kernel → scores buffer (stays on GPU)
    /// 2. f32_argmax_partial → 1024 (val, idx) partial results (8 KB)
    /// 3. Read back 8 KB, CPU reduces 1024 candidates (~1 µs)
    ///
    /// Saves ~0.33ms (1MB readback eliminated). Used by lm_head top-1 path.
    pub fn f32_gemv_topk1(&self, w: ArrayView2<f32>, x: &[f32]) -> Option<(u32, f32)> {
        let (n, k) = (w.shape()[0], w.shape()[1]);
        if x.len() != k || n == 0 {
            return None;
        }

        let w_buf = match w.as_slice() {
            Some(s) => self.bufs.get_f32(s),
            None => {
                let owned = w.as_standard_layout().into_owned();
                self.bufs.transient_from_f32(owned.as_slice().unwrap())
            }
        };
        let x_buf = self.bufs.transient_from_f32(x);
        let scores = self.bufs.output((n * 4) as u64);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();

        let kh = &self.f32_gemv_pipeline;
        let n_u32 = n as u32;
        let k_u32 = k as u32;
        let gemv_tgs = (n as u64).div_ceil(kh.rows_per_tg);
        enc.set_compute_pipeline_state(&kh.state);
        enc.set_buffer(0, Some(&w_buf), 0);
        enc.set_buffer(1, Some(&x_buf), 0);
        enc.set_buffer(2, Some(&scores), 0);
        enc.set_bytes(3, 4, &n_u32 as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(4, 4, &k_u32 as *const u32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(
            metal::MTLSize::new(gemv_tgs, 1, 1),
            metal::MTLSize::new(kh.threads_per_tg, 1, 1),
        );

        let (partial_vals, partial_idxs, n_partials) = self.encode_argmax_partial(enc, &scores, n);
        enc.end_encoding();
        self.commit_and_wait(
            cmd,
            "crates/larql-compute-metal/src/trait_impl/matmul.rs:194",
        );
        Self::reduce_argmax_partial(&partial_vals, &partial_idxs, n_partials)
    }

    /// f16 gemv + GPU argmax. Mirrors `f32_gemv_topk1` for the tied-embed
    /// lm_head path on Gemma 3/4 (mmap'd `embeddings.bin` reused as f16
    /// lm_head). Saves the 1MB readback + 262K-element CPU sort that
    /// `f16_gemv` + `top_k_sorted` would otherwise spend on each greedy
    /// decode step.
    pub fn f16_gemv_topk1(
        &self,
        w_f16: &[u8],
        x: &[f32],
        n: usize,
        k: usize,
    ) -> Option<(u32, f32)> {
        if w_f16.len() < n * k * 2 || x.len() != k || n == 0 {
            return None;
        }
        let w_buf = self.bufs.get_bytes(w_f16);
        let x_buf = self.bufs.transient_from_f32(x);
        let scores = self.bufs.output((n * 4) as u64);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();

        let kh = &self.f16_gemv_pipeline;
        let n_u32 = n as u32;
        let k_u32 = k as u32;
        let gemv_tgs = (n as u64).div_ceil(kh.rows_per_tg);
        enc.set_compute_pipeline_state(&kh.state);
        enc.set_buffer(0, Some(&w_buf), 0);
        enc.set_buffer(1, Some(&x_buf), 0);
        enc.set_buffer(2, Some(&scores), 0);
        enc.set_bytes(3, 4, &n_u32 as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(4, 4, &k_u32 as *const u32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(
            metal::MTLSize::new(gemv_tgs, 1, 1),
            metal::MTLSize::new(kh.threads_per_tg, 1, 1),
        );

        let (partial_vals, partial_idxs, n_partials) = self.encode_argmax_partial(enc, &scores, n);
        enc.end_encoding();
        self.commit_and_wait(
            cmd,
            "crates/larql-compute-metal/src/trait_impl/matmul.rs:238",
        );
        Self::reduce_argmax_partial(&partial_vals, &partial_idxs, n_partials)
    }

    /// Encode `f32_argmax_partial` over `scores[..n]` into `enc`. Returns
    /// the (vals_buf, idxs_buf, n_partials) needed for `reduce_argmax_partial`
    /// once the command buffer commits. The encoder is left active for any
    /// downstream dispatches the caller wants to add (none today).
    pub(crate) fn encode_argmax_partial(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        scores: &metal::Buffer,
        n: usize,
    ) -> (metal::Buffer, metal::Buffer, usize) {
        // Same TG width as `encode_topk_partial` — flows from the Rust
        // constant the templated MSL is built from.
        let tg_sz = crate::shaders::f32_gemv::PARTIAL_TG_SZ;
        let argmax_tgs = (n as u64).div_ceil(tg_sz);
        let partial_vals = self.bufs.output(argmax_tgs * 4);
        let partial_idxs = self.bufs.output(argmax_tgs * 4);
        let n_u32 = n as u32;
        enc.set_compute_pipeline_state(&self.f32_argmax_partial_pipeline);
        enc.set_buffer(0, Some(scores), 0);
        enc.set_buffer(1, Some(&partial_vals), 0);
        enc.set_buffer(2, Some(&partial_idxs), 0);
        enc.set_bytes(3, 4, &n_u32 as *const u32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(
            metal::MTLSize::new(argmax_tgs, 1, 1),
            metal::MTLSize::new(tg_sz, 1, 1),
        );
        (partial_vals, partial_idxs, argmax_tgs as usize)
    }

    /// CPU side of the argmax_partial pipeline: read back ≤1024 partial
    /// (val, idx) pairs (≤8 KB) and pick the global maximum. The caller
    /// must have committed and waited on the command buffer that wrote
    /// `partial_vals` / `partial_idxs`.
    pub(crate) fn reduce_argmax_partial(
        partial_vals: &metal::Buffer,
        partial_idxs: &metal::Buffer,
        n_partials: usize,
    ) -> Option<(u32, f32)> {
        let vals = crate::buffers::read_buffer_f32(partial_vals, n_partials);
        let idxs_raw = unsafe {
            let ptr = partial_idxs.contents() as *const u32;
            std::slice::from_raw_parts(ptr, n_partials)
        };
        let (best_idx, best_val) = vals
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, v)| v.is_finite())
            .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, v)| {
                if v > bv {
                    (i, v)
                } else {
                    (bi, bv)
                }
            });
        if best_val == f32::NEG_INFINITY {
            return None;
        }
        Some((idxs_raw[best_idx], best_val))
    }

    /// Encode `f32_topk_partial` over `scores[..n]`. Each TG of 256 threads
    /// emits `K_TOPK` (val, idx) pairs sorted descending; the caller merges
    /// `num_tgs × K_TOPK` candidates on CPU. Returns
    /// `(partial_vals, partial_idxs, num_tgs)`.
    pub(crate) fn encode_topk_partial(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        scores: &metal::Buffer,
        n: usize,
    ) -> (metal::Buffer, metal::Buffer, usize) {
        // TG width and per-TG K both flow from the same Rust constants the
        // MSL source is templated from; can't drift.
        let tg_sz = crate::shaders::f32_gemv::PARTIAL_TG_SZ;
        let k_topk = crate::shaders::f32_gemv::K_TOPK as u64;
        let topk_tgs = (n as u64).div_ceil(tg_sz);
        let partial_vals = self.bufs.output(topk_tgs * k_topk * 4);
        let partial_idxs = self.bufs.output(topk_tgs * k_topk * 4);
        let n_u32 = n as u32;
        enc.set_compute_pipeline_state(&self.f32_topk_partial_pipeline);
        enc.set_buffer(0, Some(scores), 0);
        enc.set_buffer(1, Some(&partial_vals), 0);
        enc.set_buffer(2, Some(&partial_idxs), 0);
        enc.set_bytes(3, 4, &n_u32 as *const u32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(
            metal::MTLSize::new(topk_tgs, 1, 1),
            metal::MTLSize::new(tg_sz, 1, 1),
        );
        (partial_vals, partial_idxs, topk_tgs as usize)
    }

    /// CPU final reduction of `num_tgs × K_TOPK` partial top-K candidates
    /// into the caller's requested `top_k`. Uses a size-`top_k` min-heap.
    /// Returns sorted descending `(token_id, score)` pairs.
    pub(crate) fn reduce_topk_partial(
        partial_vals: &metal::Buffer,
        partial_idxs: &metal::Buffer,
        num_tgs: usize,
        top_k: usize,
    ) -> Vec<(u32, f32)> {
        let k_topk = crate::shaders::f32_gemv::K_TOPK;
        let total = num_tgs * k_topk;
        let vals = crate::buffers::read_buffer_f32(partial_vals, total);
        let idxs = unsafe {
            let ptr = partial_idxs.contents() as *const u32;
            std::slice::from_raw_parts(ptr, total)
        };

        let k = top_k.min(total);
        if k == 0 {
            return Vec::new();
        }

        let mut heap: Vec<(f32, u32)> = Vec::with_capacity(k + 1);

        fn sift_down(h: &mut [(f32, u32)], mut i: usize) {
            let n = h.len();
            loop {
                let mut smallest = i;
                let l = 2 * i + 1;
                let r = 2 * i + 2;
                if l < n && h[l].0 < h[smallest].0 {
                    smallest = l;
                }
                if r < n && h[r].0 < h[smallest].0 {
                    smallest = r;
                }
                if smallest == i {
                    break;
                }
                h.swap(i, smallest);
                i = smallest;
            }
        }

        for (i, &v) in vals.iter().enumerate() {
            if !v.is_finite() {
                continue;
            }
            // Skip the sentinel-index slots emitted by trailing TGs that
            // had nothing to rank (idx = ~0u from the masked-out lanes).
            if idxs[i] == u32::MAX {
                continue;
            }
            if heap.len() < k {
                heap.push((v, idxs[i]));
                if heap.len() == k {
                    for j in (0..k / 2).rev() {
                        sift_down(&mut heap, j);
                    }
                }
            } else if v > heap[0].0 {
                heap[0] = (v, idxs[i]);
                sift_down(&mut heap, 0);
            }
        }
        if heap.len() < k && heap.len() > 1 {
            for j in (0..heap.len() / 2).rev() {
                sift_down(&mut heap, j);
            }
        }
        heap.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        heap.into_iter().map(|(s, i)| (i, s)).collect()
    }

    /// f16 gemv + GPU partial top-K. Mirrors `f16_gemv_topk1` but produces
    /// the top `top_k` scores in one round-trip (top_k ≤ K_TOPK = 8).
    /// Returns `None` if `top_k` exceeds the per-TG capacity — the caller
    /// then falls back to `f16_gemv` + CPU sort.
    pub fn f16_gemv_topk(
        &self,
        w_f16: &[u8],
        x: &[f32],
        n: usize,
        k: usize,
        top_k: usize,
    ) -> Option<Vec<(u32, f32)>> {
        if top_k == 0 || top_k > crate::shaders::f32_gemv::K_TOPK {
            return None;
        }
        if w_f16.len() < n * k * 2 || x.len() != k || n == 0 {
            return None;
        }
        let w_buf = self.bufs.get_bytes(w_f16);
        let x_buf = self.bufs.transient_from_f32(x);
        let scores = self.bufs.output((n * 4) as u64);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();

        let kh = &self.f16_gemv_pipeline;
        let n_u32 = n as u32;
        let k_u32 = k as u32;
        let gemv_tgs = (n as u64).div_ceil(kh.rows_per_tg);
        enc.set_compute_pipeline_state(&kh.state);
        enc.set_buffer(0, Some(&w_buf), 0);
        enc.set_buffer(1, Some(&x_buf), 0);
        enc.set_buffer(2, Some(&scores), 0);
        enc.set_bytes(3, 4, &n_u32 as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(4, 4, &k_u32 as *const u32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(
            metal::MTLSize::new(gemv_tgs, 1, 1),
            metal::MTLSize::new(kh.threads_per_tg, 1, 1),
        );

        let (partial_vals, partial_idxs, num_tgs) = self.encode_topk_partial(enc, &scores, n);
        enc.end_encoding();
        self.commit_and_wait(
            cmd,
            "crates/larql-compute-metal/src/trait_impl/matmul.rs:450",
        );
        Some(Self::reduce_topk_partial(
            &partial_vals,
            &partial_idxs,
            num_tgs,
            top_k,
        ))
    }

    /// Q4_K stride-32 matvec → full f32 scores. Same Q4_K input format
    /// as `q4k_matvec`, but uses the shader at
    /// `shaders::q4k_matvec_stride32` whose 32-lane reduction matches
    /// `f16_gemv`'s tree (lane k accumulates stride-32 elements then
    /// `simd_sum`). Required for the LM head when the production
    /// `q4k_matvec`'s block-aware lane split drifts enough vs CPU to
    /// flip top-1 on close-call tokens.
    pub fn q4k_matvec_stride32(
        &self,
        q4k_data: &[u8],
        x: &[f32],
        num_rows: usize,
        hidden: usize,
    ) -> Option<Vec<f32>> {
        if hidden == 0 || !hidden.is_multiple_of(256) {
            return None;
        }
        let kh = &self.quant.q4k_matvec_stride32_pipeline;
        let buf_w = self.bufs.get_bytes(q4k_data);
        let buf_x = self.bufs.transient_from_f32(x);
        let buf_out = self.bufs.output((num_rows * 4) as u64);
        let n = num_rows as u32;
        let k = hidden as u32;
        let num_tgs = (num_rows as u64).div_ceil(kh.rows_per_tg);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&kh.state);
        enc.set_buffer(0, Some(&buf_w), 0);
        enc.set_buffer(1, Some(&buf_x), 0);
        enc.set_buffer(2, Some(&buf_out), 0);
        enc.set_bytes(3, 4, &n as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(4, 4, &k as *const u32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(
            metal::MTLSize::new(num_tgs, 1, 1),
            metal::MTLSize::new(kh.threads_per_tg, 1, 1),
        );
        enc.end_encoding();
        self.commit_and_wait(
            cmd,
            "crates/larql-compute-metal/src/trait_impl/matmul.rs:498",
        );

        Some(crate::buffers::read_buffer_f32(&buf_out, num_rows))
    }

    /// Shared dispatch body for f16-weight gemv (behind both trait
    /// variants: threshold-gated `f16_gemv` and direct `f16_gemv_force`).
    pub(super) fn encode_f16_gemv(
        &self,
        w_f16: &[u8],
        x: &[f32],
        n: usize,
        k: usize,
    ) -> Option<Vec<f32>> {
        let w_buf = self.bufs.get_bytes(w_f16);
        // Pooled x and out: a decode step issues hundreds of gemvs, and
        // per-call device allocations were a measurable share of its
        // cost. Recycled after the wait, per the pool contract.
        let x_buf = self.bufs.output((x.len() * 4) as u64);
        let x_ptr = x_buf.contents() as *mut f32;
        if x_ptr.is_null() {
            return None;
        }
        // SAFETY: pooled buffer is at least x.len()*4 bytes, and it is
        // not bound to any encoder yet, so the GPU is not reading it.
        unsafe { std::ptr::copy_nonoverlapping(x.as_ptr(), x_ptr, x.len()) };
        let out_buf = self.bufs.output((n * 4) as u64);

        // Geometry travels with the f16_gemv KernelHandle.
        let kernel = &self.f16_gemv_pipeline;
        let n_u32 = n as u32;
        let k_u32 = k as u32;
        let num_tgs = (n as u64).div_ceil(kernel.rows_per_tg);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&kernel.state);
        enc.set_buffer(0, Some(&w_buf), 0);
        enc.set_buffer(1, Some(&x_buf), 0);
        enc.set_buffer(2, Some(&out_buf), 0);
        enc.set_bytes(3, 4, &n_u32 as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(4, 4, &k_u32 as *const u32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(
            metal::MTLSize::new(num_tgs, 1, 1),
            metal::MTLSize::new(kernel.threads_per_tg, 1, 1),
        );
        enc.end_encoding();
        self.commit_and_wait(
            cmd,
            "crates/larql-compute-metal/src/trait_impl/matmul.rs:530",
        );

        let result = crate::buffers::try_read_buffer_f32(&out_buf, n);
        self.bufs.recycle(out_buf);
        self.bufs.recycle(x_buf);
        result
    }
}
