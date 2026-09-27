//! Step 5a: O projection, then the optional O-projection bias.

use metal::{ComputeCommandEncoderRef, MTLSize};

use super::context::AttnCtx;
use crate::MetalBackend;
use larql_models::quant::ggml::LEGACY_BLOCK_ELEMS;

impl MetalBackend {
    pub(super) fn encode_o_projection(
        &self,
        enc: &ComputeCommandEncoderRef,
        c: &AttnCtx<'_, '_, '_>,
    ) {
        let layer = c.layer;
        let bufs = &c.bufs;
        let hidden = c.hidden;
        let layer_q_dim = c.layer_q_dim;
        // Dispatched on `wo`'s own format, not `wq`'s. Selecting an
        // operand's ABI from a *different* operand's representation is
        // what produced the defect this replaced: `uses_kquant` reads
        // `wq`, so a Q4_0 `wo` fell to the Q8-only branch below and was
        // handed to `q8_matvec` as though its 18-byte packed blocks were
        // int8 rows, alongside a weight-scale buffer Q4_0 has no source
        // for. That was survivable only because every caller fabricated
        // an empty scale buffer; removing the fabrication exposed it.
        match layer.wo.format() {
            // `o_proj::encode` understands all four packed formats — its
            // own doc covers "Q4_0 / Q8_0: quantise attn_in -> Q8 int8 +
            // per-32 f16 scale" as well as the k-quant families.
            larql_compute::QuantFormat::Q4_0
            | larql_compute::QuantFormat::Q4_K
            | larql_compute::QuantFormat::Q4_KF
            | larql_compute::QuantFormat::Q6_K => {
                use crate::stages::quant_matvec::Pipelines;
                let pipes = Pipelines {
                    q4kf_proj: Some(&self.attention.q4kf_proj_pipeline.state),
                    q4k_matvec_fallback: &self.quant.q4k_matvec_pipeline,
                    q6k_matvec: &self.quant.q6k_matvec_pipeline,
                    q4_matvec: &self.q4.matvec,
                    q4k_matmul: None,
                };
                // K from the store's own row width, not the logical
                // `layer_q_dim` — O rows may be padded to the quant block
                // (same audit-F17 hazard as the QKV kernels; the padded
                // columns dequantise to zero, so the stored width is
                // exact given the input buffer's zeroed slack).
                let o_k_store = layer.wo.stored_cols(hidden, layer_q_dim);
                crate::stages::o_proj::encode(
                    enc,
                    &pipes,
                    &self.quant.q8_quant_pipeline,
                    layer.wo.format(),
                    bufs.wo,
                    bufs.attn_out_buf,
                    0,
                    bufs.o_q8_scratch,
                    0,
                    bufs.o_q8s_scratch,
                    0,
                    bufs.o_out_buf,
                    0,
                    o_k_store,
                    hidden,
                );
            }
            // Q8 legacy path: decode-specific `q8_matvec` shader (not in
            // stages::quant_matvec, which uses `q4_matvec` with a
            // different buffer layout). Q8_0 only — it is the one format
            // that supplies the external weight scales this shader reads
            // at buffer index 2.
            larql_compute::QuantFormat::Q8_0 => {
                let dim_val = layer_q_dim as u32;
                let blocks = layer_q_dim.div_ceil(LEGACY_BLOCK_ELEMS) as u32;
                enc.set_compute_pipeline_state(&self.quant.q8_quant_pipeline);
                enc.set_buffer(0, Some(bufs.attn_out_buf), 0);
                enc.set_buffer(1, Some(bufs.o_q8_scratch), 0);
                enc.set_buffer(2, Some(bufs.o_q8s_scratch), 0);
                enc.set_bytes(3, 4, &dim_val as *const u32 as *const std::ffi::c_void);
                enc.dispatch_threads(
                    MTLSize::new(blocks as u64, 1, 1),
                    MTLSize::new(
                        crate::kernels::DISPATCH_TG_MAX_THREADS.min(blocks as u64),
                        1,
                        1,
                    ),
                );

                let o_rows = hidden as u32;
                let o_k = layer_q_dim as u32;
                assert!(
                    layer_q_dim <= crate::shaders::q8_matvec::MAX_K,
                    "q8_matvec stages its Q8 input in threadgroup memory capped at \
                     K = {}; layer_q_dim {layer_q_dim} would corrupt it (audit F13)",
                    crate::shaders::q8_matvec::MAX_K,
                );
                enc.set_compute_pipeline_state(&self.quant.q8_matvec_pipeline.state);
                enc.set_buffer(0, Some(bufs.wo), 0);
                enc.set_buffer(1, Some(bufs.o_q8_scratch), 0);
                enc.set_buffer(
                    2,
                    Some(
                        bufs.wo_scales
                            .expect("legacy scale path requires an external-scale format"),
                    ),
                    0,
                );
                enc.set_buffer(3, Some(bufs.o_q8s_scratch), 0);
                enc.set_buffer(4, Some(bufs.o_out_buf), 0);
                enc.set_bytes(5, 4, &o_rows as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(6, 4, &o_k as *const u32 as *const std::ffi::c_void);
                enc.dispatch_thread_groups(
                    MTLSize::new((hidden as u64).div_ceil(8), 1, 1),
                    MTLSize::new(256, 1, 1),
                );
            }
            // Loud rather than silent. The previous predicate routed
            // anything non-k-quant here, so an unsupported format was
            // reinterpreted as Q8 and produced plausible numbers instead
            // of an error.
            format => panic!(
                "O projection has no dispatch for {format:?}; supported: \
                 Q4_0, Q4_K, Q4_KF, Q6_K (o_proj::encode) and Q8_0 (legacy q8_matvec)"
            ),
        }

        // O-projection bias (GPT-OSS) joins before the residual add —
        // the same point the CPU reference (`forward::add_bias`) applies
        // it. Dispatched only when the layer carries the bias.
        if let Some(b) = layer.attn_o_bias {
            assert_eq!(
                b.len(),
                hidden,
                "O projection bias has {} entries but hidden is {hidden} — \
                 the extracted tensor does not match this model",
                b.len()
            );
            let b_buf = self.bufs.get_f32(b);
            crate::stages::bias_add::encode(
                enc,
                &self.attention.bias_add_pipeline,
                bufs.o_out_buf,
                0,
                &b_buf,
                hidden,
            );
        }
    }
}
