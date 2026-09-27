//! The Kimi source model.

use super::super::stack_metal::{DeviceAttn, DeviceLayer, DeviceState, HybridHead};
use crate::error::VindexError;
use crate::format::vindex3::represent::compile::CandidatePlacement;
use crate::format::vindex3::represent::compiler::CandidateIndex;
use crate::format::vindex3::represent::physical::{
    EncodedRegion, ExpertBankBinding, ExpertEncoding, ExtentPolicy, PhysicalStore,
    ProjectionAddressing, RoutedProjection, SharedExpertBinding,
};
use crate::format::vindex3::represent::source_bank::source_expert_bank;
use larql_compute::backend::ComputeBackend;
use larql_compute_metal::trait_impl::grouped_experts::ExpertOffset;
use larql_compute_metal::trait_impl::kda::KdaDeviceState;
use larql_compute_metal::trait_impl::kimi_layer::ExpertEncoding as MetalEncoding;
use larql_compute_metal::trait_impl::mla::MlaDeviceState;
use larql_compute_metal::MetalBackend;
use std::path::Path;
use std::sync::Arc;

#[allow(unused_imports)]
use super::*;

impl KimiSourceModel {
    pub fn open(dir: &Path) -> Result<Self, VindexError> {
        let geometry = geometry_from_graph(&graph_value(dir)?)?;
        let seg = |name: &str| dir.join("segments").join(format!("{name}.bin"));
        Ok(Self {
            geometry,
            dir: dir.to_path_buf(),
            decoder: SegmentTensors::open(
                "kimi-source-decoder-stack",
                &seg("target.decoder_stack"),
            )?,
            experts: SegmentTensors::open("kimi-source-expert-bank", &seg("target.expert_bank"))?,
        })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The shared expert's binding for one layer — always from the
    /// decoder stack, never from any expert bank, whichever arm asked.
    pub(super) fn shared_binding(&self, layer: usize) -> Result<SharedExpertBinding, VindexError> {
        let region = |proj: &str| -> Result<EncodedRegion, VindexError> {
            Ok(EncodedRegion {
                region: self.decoder.region(&format!(
                    "{layer}.block_sparse_moe.shared_experts.{proj}.weight"
                ))?,
                encoding: ExpertEncoding::Bf16,
            })
        };
        Ok(SharedExpertBinding {
            gate: region("gate_proj")?,
            up: region("up_proj")?,
            down: region("down_proj")?,
        })
    }

    /// The eleven small f32 operands a KDA layer carries beside its
    /// wide projections. `A_log` is stored `[1,1,H,1]` and flattens to
    /// `[H]`; the conv weights' middle `1` dimension is inert under
    /// row-major flattening — the only two transforms the proven
    /// exporter ever applied. Always from the SOURCE stack: no
    /// candidate kind stores them.
    pub(super) fn kda_f32s(&self, layer: usize) -> Result<Vec<Vec<f32>>, VindexError> {
        let t = |suffix: &str| format!("{layer}.self_attn.{suffix}");
        [
            "q_conv1d.weight",
            "k_conv1d.weight",
            "v_conv1d.weight",
            "f_a_proj.weight",
            "f_b_proj.weight",
            "g_a_proj.weight",
            "g_b_proj.weight",
            "b_proj.weight",
            "A_log",
            "dt_bias",
            "o_norm.weight",
        ]
        .iter()
        .map(|suffix| self.decoder.f32s(&t(suffix)))
        .collect()
    }

    /// One layer's attention operands, whichever operator the graph
    /// declares for it.
    pub(super) fn attention(
        &self,
        metal: &MetalBackend,
        layer: usize,
        kda: Option<&KdaOverlay>,
    ) -> Result<(DeviceAttn, DeviceState), VindexError> {
        let g = &self.geometry;
        let t = |suffix: &str| format!("{layer}.self_attn.{suffix}");
        // Refused BY NAME, before any tensor is bound (K3-REP-GATE-1,
        // freeze D6). The decision is a pure function of declared facts,
        // so a test without a device can witness it.
        if let Some(refusal) = super::super::device_refusal::declared_gate_refusal(
            layer,
            g.mla_layer[layer],
            g.kda_full_rank_gate,
            g.mla_output_gate,
            g.mla_q_lora_rank,
        ) {
            return Err(VindexError::Parse(refusal));
        }
        if g.mla_layer[layer] {
            return Ok((
                DeviceAttn::Mla {
                    q: self.decoder.bytes(&t("q_proj.weight"))?,
                    kv_a: self.decoder.bytes(&t("kv_a_proj_with_mqa.weight"))?,
                    kv_b: self.decoder.bytes(&t("kv_b_proj.weight"))?,
                    o: self.decoder.bytes(&t("o_proj.weight"))?,
                    kv_a_norm: self.decoder.f32s(&t("kv_a_layernorm.weight"))?,
                    encoding: MetalEncoding::Bf16,
                },
                DeviceState::Mla(MlaDeviceState::with_capacity(
                    metal,
                    g.mla,
                    MLA_CACHE_POSITIONS,
                )),
            ));
        }
        let gate_form = larql_compute_metal::trait_impl::kda::declared_gate_form(g.kda_gate_form)
            .map_err(|e| VindexError::Parse(format!("layer {layer}: {e:?}")))?;
        // A compiled KDA candidate substitutes the four wide projections
        // and NOTHING else — the f32 operands below still come from the
        // source stack, exactly as the behavioural evidence was earned.
        if let Some(binding) = kda.and_then(|o| o.binding(layer as u32)) {
            let b = binding?;
            let f32s = self.kda_f32s(layer)?;
            return Ok((
                DeviceAttn::Kda {
                    qkv_bank: b.qkv_bank,
                    qkv_offsets: b.qkv_offsets,
                    o_proj: b.o_proj,
                    encoding: b.encoding,
                    gate_form,
                    f32s,
                },
                DeviceState::Kda(KdaDeviceState::zeros(metal, g.kda)),
            ));
        }
        let (q, k, v) = (
            self.decoder.bytes(&t("q_proj.weight"))?,
            self.decoder.bytes(&t("k_proj.weight"))?,
            self.decoder.bytes(&t("v_proj.weight"))?,
        );
        let per = q.len();
        if k.len() != per || v.len() != per {
            return Err(VindexError::Parse(format!(
                "layer {layer}: KDA q/k/v projections differ in size ({per}/{}/{})",
                k.len(),
                v.len()
            )));
        }
        let mut qkv_bank = Vec::with_capacity(3 * per);
        for b in [&q, &k, &v] {
            qkv_bank.extend_from_slice(b);
        }
        let f32s = self.kda_f32s(layer)?;
        Ok((
            DeviceAttn::Kda {
                qkv_bank,
                qkv_offsets: [
                    ExpertOffset(0),
                    ExpertOffset(per as u32),
                    ExpertOffset((2 * per) as u32),
                ],
                o_proj: self.decoder.bytes(&t("o_proj.weight"))?,
                // The loader binds the container's own bf16 bytes; a
                // compiled KDA-projection candidate is a later rung's
                // overlay arm, not a silent default.
                encoding: MetalEncoding::Bf16,
                gate_form,
                f32s,
            },
            DeviceState::Kda(KdaDeviceState::zeros(metal, g.kda)),
        ))
    }

    /// Build one layer, routed experts from the source bank or — when
    /// the overlay compiled this layer — from the candidate's own store.
    pub fn device_layer(
        &self,
        metal: &MetalBackend,
        layer: usize,
        overlay: Option<&CandidateOverlay>,
    ) -> Result<DeviceLayer, VindexError> {
        self.device_layer_with_kda(metal, layer, overlay, None)
    }

    /// [`Self::device_layer`], with a KDA candidate riding beside the
    /// expert candidate — the two substitute DISJOINT operand families,
    /// so their composition needs no arbitration.
    pub fn device_layer_with_kda(
        &self,
        metal: &MetalBackend,
        layer: usize,
        overlay: Option<&CandidateOverlay>,
        kda: Option<&KdaOverlay>,
    ) -> Result<DeviceLayer, VindexError> {
        let g = &self.geometry;
        let (attn, state) = self.attention(metal, layer, kda)?;
        let common = |bank, router_weight, router_bias, inter, top_k, dense| {
            DeviceLayer {
                attn,
                state,
                bank,
                input_norm: Vec::new(), // filled below
                post_norm: Vec::new(),
                router_weight,
                router_bias,
                inter,
                top_k,
                dense,
                renormalize: g.renormalize,
                branch_scale: g.branch_scale,
                norm_eps: g.rms_eps,
                kda_shape: g.kda,
                mla_shape: g.mla,
                mla_norm_eps: g.mla_norm_eps,
            }
        };
        let mut d = if layer < g.dense_prefix_layers {
            // The dense MLP: three whole tensors of the decoder stack,
            // bound as one-expert regions.
            let region = |proj: &str| -> Result<EncodedRegion, VindexError> {
                Ok(EncodedRegion {
                    region: self.decoder.region(&format!("{layer}.mlp.{proj}.weight"))?,
                    encoding: ExpertEncoding::Bf16,
                })
            };
            // One "expert" at offset zero, per projection. The dense
            // MLP is three whole tensors of the decoder stack, and each
            // is exactly its own region — so `Exact` is a real claim
            // here and the surplus-byte check does bite.
            let dense = |proj: &str| -> Result<RoutedProjection, VindexError> {
                Ok(RoutedProjection {
                    region: region(proj)?,
                    addressing: ProjectionAddressing::Table(vec![0]),
                    extent: ExtentPolicy::Exact,
                })
            };
            common(
                ExpertBankBinding {
                    gate: dense("gate_proj")?,
                    up: dense("up_proj")?,
                    down: dense("down_proj")?,
                    shared: None,
                },
                // A dense layer carries no router at all.
                Vec::new(),
                Vec::new(),
                g.dense_intermediate,
                0,
                true,
            )
        } else {
            let shared = self.shared_binding(layer)?;
            let router_weight = self
                .decoder
                .f32s(&format!("{layer}.block_sparse_moe.gate.weight"))?;
            let router_bias = self.decoder.f32s(&format!(
                "{layer}.block_sparse_moe.gate.e_score_correction_bias"
            ))?;
            // **The source binding is always built**, and the overlay
            // then replaces only the projections it actually compiled.
            //
            // Per projection, not per bank: a projection-scoped
            // candidate holds `w1` alone, and `w3`/`w2` must still
            // resolve from the source segment — a different store, a
            // different encoding and a different addressing mode, in
            // the same layer. Composing in this direction also makes
            // the fallback the source rather than a hole, so an overlay
            // that compiled nothing yields the baseline arm exactly.
            let source = source_expert_bank(
                &self.experts.store,
                &self.experts.offsets,
                layer as u32,
                g.experts,
                g.source_projection_bytes(),
            )?;
            let mut bank = source.binding;
            bank.shared = Some(shared);
            if let Some(o) = overlay {
                for (name, slot) in [
                    ("w1", &mut bank.gate),
                    ("w3", &mut bank.up),
                    ("w2", &mut bank.down),
                ] {
                    if let Some(compiled) = o.projection_binding(layer as u32, name) {
                        *slot = compiled?;
                    }
                }
            }
            common(
                bank,
                router_weight,
                router_bias,
                g.moe_intermediate,
                g.top_k,
                false,
            )
        };
        d.input_norm = self
            .decoder
            .f32s(&format!("{layer}.input_layernorm.weight"))?;
        d.post_norm = self
            .decoder
            .f32s(&format!("{layer}.post_attention_layernorm.weight"))?;
        d.validate_banks(g.hidden)?;
        Ok(d)
    }

    /// The final norm and vocabulary projection, for the head to ride in
    /// the last device epoch.
    pub fn head(&self) -> Result<HybridHead, VindexError> {
        let seg = |name: &str| self.dir.join("segments").join(format!("{name}.bin"));
        let final_norm = SegmentTensors::open("kimi-source-final-norm", &seg("target.final_norm"))?;
        let head = SegmentTensors::open("kimi-source-output-head", &seg("target.output_head"))?;
        Ok(HybridHead {
            norm_weight: final_norm.f32s("weight")?,
            weight: head.bytes("weight")?,
            vocab: self.geometry.vocab,
            norm_eps: self.geometry.rms_eps,
            encoding: MetalEncoding::Bf16,
        })
    }

    /// Register the mmap-backed stores with the backend, page-aligned.
    ///
    /// Region bases must be page-aligned for zero-copy registration, and
    /// a payload span almost never is — so registration cuts from each
    /// store's BACKING allocation (whose mmap base is aligned by
    /// construction) at page boundaries. Without this, the first
    /// `weights()` resolution misses and stages a multi-gigabyte copy,
    /// silently.
    ///
    /// The expert bank is registered as one span per MoE layer (~3.6 GB
    /// each) rather than one 94 GB buffer; `moe_layers` names which.
    pub fn register_stores(
        &self,
        metal: &MetalBackend,
        moe_layers: &[u32],
    ) -> Result<usize, VindexError> {
        const PAGE: usize = larql_compute_metal::buffers::PAGE_SIZE;
        let mut registered = 0usize;
        metal.register_weight_region(self.decoder.store.backing_bytes());
        registered += 1;
        let backing = self.experts.store.backing_bytes();
        let payload_start = self.experts.store.payload_start();
        for &layer in moe_layers {
            let bank = source_expert_bank(
                &self.experts.store,
                &self.experts.offsets,
                layer,
                self.geometry.experts,
                self.geometry.source_projection_bytes(),
            )?;
            let start = (payload_start + bank.layer_base) as usize / PAGE * PAGE;
            let end = (payload_start + bank.layer_base + bank.layer_len) as usize;
            metal.register_weight_region(&backing[start..end]);
            registered += 1;
        }
        Ok(registered)
    }
}

/// A compiled candidate bank, opened against the source it depends on.
pub struct CandidateOverlay {
    pub index: CandidateIndex,
    pub(super) store: Arc<PhysicalStore>,
    /// Layers whose routed bank the candidate holds, from the ledger's
    /// own seals — never from the map alone, which states intent rather
    /// than bytes.
    pub(super) layers: Vec<u32>,
    /// Projections the candidate holds, in the checkpoint's own
    /// spelling (`w1` gate, `w3` up, `w2` down). Fewer than three is a
    /// projection-scoped candidate, and the rest stay source-backed.
    /// Derived from the SEALS, for the same reason as `layers`.
    pub(super) projections: Vec<String>,
    /// Physical representation per compiled LAYER — a composed map's
    /// layers need not share one (Q8_0 band beside a Q6_K layer).
    pub(super) encodings: std::collections::BTreeMap<u32, ExpertEncoding>,
    /// Where each layer's bank sits in the segment — the same
    /// definition the compiler wrote and `verify_complete` proved.
    /// Geometry lives in the placement's layouts; the overlay keeps
    /// only the expert count, which the layouts do not carry.
    pub(super) placement: CandidatePlacement,
    pub(super) experts: u32,
}
