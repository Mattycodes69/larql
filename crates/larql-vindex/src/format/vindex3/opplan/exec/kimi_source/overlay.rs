//! Candidate and KDA overlays, and completeness verification.

use crate::error::VindexError;
use crate::format::vindex3::represent::compile::CandidatePlacement;
use crate::format::vindex3::represent::compiler::{
    bank_base, read_source_identity, CandidateIndex,
};
use crate::format::vindex3::represent::kda_candidate::{KdaPlacement, KDA_PROJECTIONS};
use crate::format::vindex3::represent::physical::{
    EncodedRegion, ExpertBankBinding, ExpertEncoding, ExtentPolicy, PhysicalStore,
    ProjectionAddressing, RoutedProjection,
};
use crate::format::vindex3::represent::policy::{layer_of, projection_of, Role};
use larql_compute::backend::ComputeBackend;
use larql_compute_metal::trait_impl::grouped_experts::ExpertOffset;
use larql_compute_metal::trait_impl::kimi_layer::ExpertEncoding as MetalEncoding;
use larql_compute_metal::MetalBackend;
use std::path::Path;
use std::sync::Arc;

#[allow(unused_imports)]
use super::*;

impl CandidateOverlay {
    /// Open the overlay, verify its source dependency against the
    /// container actually present, and prove the ledger COMPLETE for
    /// every layer it claims: all `experts x 3` operands sealed at
    /// exactly the layout's offsets, no two seals overlapping.
    ///
    /// Completeness at load is what makes "remove one compiled operand"
    /// a refusal instead of a silent fallback — an Identity-addressed
    /// bank has no table entry to mark absent, so the ledger is the only
    /// witness that every slot's bytes were actually compiled.
    pub fn open(
        dir: &Path,
        source_dir: &Path,
        geometry: &KimiGeometry,
    ) -> Result<Self, VindexError> {
        let index: CandidateIndex = serde_json::from_slice(&std::fs::read(dir.join("index.json"))?)
            .map_err(|e| VindexError::Parse(format!("candidate index: {e}")))?;
        index.source.verify(&read_source_identity(source_dir)?)?;
        let (layers, projections, placement) = verify_complete(&index, geometry)?;
        // Per LAYER, from the placement the completeness proof ran on —
        // a composed map's layers need not share one encoding.
        let mut encodings = std::collections::BTreeMap::new();
        for &layer in &layers {
            let name = &placement.layout(layer)?.encoding;
            let enc = ExpertEncoding::parse(name).ok_or_else(|| {
                VindexError::Parse(format!(
                    "candidate encodes layer {layer} as `{name}`, which no grouped \
                     kernel reads"
                ))
            })?;
            encodings.insert(layer, enc);
        }
        let segment = dir.join("segments").join(format!("{}.bin", index.object));
        let store = Arc::new(PhysicalStore::map_compiled(
            // Encoding-agnostic on purpose: the id lands in evidence
            // attributions, and a Q8_0 candidate labelled "q6" would
            // misstate the representation under test.
            "kimi-candidate-bank",
            &segment,
            &index.ledger,
        )?);
        Ok(Self {
            index,
            store,
            layers,
            projections,
            encodings,
            placement,
            experts: geometry.experts,
        })
    }

    pub fn compiled_layers(&self) -> &[u32] {
        &self.layers
    }

    /// Which projections this candidate compiled. Fewer than three
    /// means the rest execute from the source.
    /// The physical representation this LAYER's sealed operands carry —
    /// from the map the compiler executed, verified parseable at open.
    pub fn encoding_of(&self, layer: u32) -> Result<ExpertEncoding, VindexError> {
        self.encodings.get(&layer).copied().ok_or_else(|| {
            VindexError::Parse(format!("layer {layer} is not compiled in this candidate"))
        })
    }

    pub fn compiled_projections(&self) -> &[String] {
        &self.projections
    }

    /// How the arm reads in a report: the scope, not the file name.
    pub fn scope(&self) -> String {
        format!(
            "layers {:?} / {} / {}",
            self.layers,
            if self.projections.len() == 3 {
                "all projections".to_string()
            } else {
                self.projections.join("+")
            },
            self.layers
                .iter()
                .map(|l| {
                    let enc = self.encodings.get(l).map(|e| e.name()).unwrap_or("?");
                    format!("L{l}:{enc}")
                })
                .collect::<Vec<_>>()
                .join(" ")
        )
    }

    /// **The bytes this overlay READS are the bytes the compiler
    /// WROTE**, proven per operand against the ledger's own
    /// `target_hash`.
    ///
    /// Region extent and seal offsets agreeing (`verify_complete`) shows
    /// the layout is self-consistent; it cannot show that the view is
    /// positioned where the writer put the payload. A whole-bank shift —
    /// a projection base off by one stride, a header the reader skips
    /// and the writer did not — satisfies every offset check and serves
    /// a neighbouring expert's weights, which decode to plausible
    /// numbers and would read as "quantisation is catastrophic".
    ///
    /// `sample` operands per projection, evenly spread across the
    /// expert range so a shift anywhere in the bank is caught.
    pub fn verify_reads_match_seals(
        &self,
        layer: u32,
        sample: usize,
    ) -> Result<usize, VindexError> {
        // Only the projections this candidate actually compiled: a
        // scoped one has nothing to say about the other two, and
        // demanding seals for them would refuse a complete overlay.
        let compiled: Vec<(String, RoutedProjection)> = self
            .projections
            .iter()
            .map(|proj| {
                let binding = self.projection_binding(layer, proj).ok_or_else(|| {
                    VindexError::Parse(format!("layer {layer} / {proj} is not compiled here"))
                })??;
                Ok((proj.clone(), binding))
            })
            .collect::<Result<_, VindexError>>()?;
        let mut checked = 0usize;
        let step = (self.experts as usize / sample.max(1)).max(1);
        for expert in (0..self.experts as usize).step_by(step) {
            for (proj, projection) in &compiled {
                let stride = match projection.addressing {
                    ProjectionAddressing::Identity { stride, .. } => stride,
                    ProjectionAddressing::Table(_) => {
                        return Err(VindexError::Parse(
                            "a compiled projection is identity-addressed by construction".into(),
                        ))
                    }
                };
                let tensor = format!("{layer}.block_sparse_moe.experts.{expert}.{proj}.weight");
                let seal = self
                    .index
                    .ledger
                    .get(&self.index.object, &tensor)
                    .ok_or_else(|| VindexError::Parse(format!("no seal for `{tensor}`")))?;
                let base = expert * stride as usize;
                let bytes = &projection.region.region.bytes()[base..base + stride as usize];
                let got = super::super::super::super::represent::compile::hash_bytes(bytes);
                if got != seal.target_hash {
                    return Err(VindexError::Parse(format!(
                        "`{tensor}`: the loader reads {} at slot {expert} but the compiler                          sealed {} — the view is not positioned where the payload was                          written, so this bank serves the wrong expert's weights",
                        &got[..12],
                        &seal.target_hash[..12]
                    )));
                }
                checked += 1;
            }
        }
        Ok(checked)
    }

    /// **One projection's compiled binding**, when this overlay holds
    /// that projection of that layer — `None` when it does not, which
    /// tells the caller to keep the source's.
    ///
    /// Per projection because a precision map may scope one: `w1` alone
    /// is a candidate, and the sweep asking WHICH projection initiates
    /// the routing cascade needs exactly that. `None` is a real answer
    /// rather than a failure, which is what makes composition additive
    /// over the source binding.
    ///
    /// No shared branch: the overlay compiles ROUTED experts only, and
    /// the caller attaches the shared binding from the source — the
    /// composition D0 exists to make expressible.
    pub fn projection_binding(
        &self,
        layer: u32,
        projection: &str,
    ) -> Option<Result<RoutedProjection, VindexError>> {
        if !self.layers.contains(&layer) || !self.projections.iter().any(|p| p == projection) {
            return None;
        }
        Some(self.binding_inner(layer, projection))
    }

    /// The whole routed bank, for an overlay that compiled all three
    /// projections. `None` if it compiled fewer.
    pub fn routed_binding(&self, layer: u32) -> Option<Result<ExpertBankBinding, VindexError>> {
        if !self.layers.contains(&layer) || self.projections.len() != 3 {
            return None;
        }
        Some((|| {
            Ok(ExpertBankBinding {
                gate: self.binding_inner(layer, "w1")?,
                up: self.binding_inner(layer, "w3")?,
                down: self.binding_inner(layer, "w2")?,
                shared: None,
            })
        })())
    }

    pub(super) fn binding_inner(
        &self,
        layer: u32,
        projection: &str,
    ) -> Result<RoutedProjection, VindexError> {
        // The SAME placement `verify_complete` proved every seal
        // against — one definition of where an expert's bytes are.
        let layout = self.placement.layout(layer)?.clone();
        let layer_base = self.placement.layer_base(layer)?;
        if layout.gate_up_stride != layout.down_stride {
            return Err(VindexError::Parse(format!(
                "layer {layer}: gate/up stride {} != down stride {} — identity addressing \
                 carries ONE stride for all three projections",
                layout.gate_up_stride, layout.down_stride
            )));
        }
        let stride = u32::try_from(layout.gate_up_stride).map_err(|_| {
            VindexError::Parse("a compiled expert stride does not fit 32 bits".to_string())
        })?;
        let bank_bytes = layout.bank_bytes("w1")?;
        let encoding = self.encoding_of(layer)?;
        let region = |proj: &str| -> Result<EncodedRegion, VindexError> {
            let base = layer_base + bank_base(&layout, proj)?;
            Ok(EncodedRegion {
                region: self.store.span(base, bank_bytes).ok_or_else(|| {
                    VindexError::Parse(format!(
                        "candidate segment is too short for the {proj} bank at {base}"
                    ))
                })?,
                encoding,
            })
        };
        // Every compiled projection is a full execution-shaped bank:
        // addressed by identity at its own stride, and exactly its own
        // region. The stride now travels WITH the projection rather
        // than beside the binding, so a caller cannot pair one
        // projection's bytes with another's stride.
        Ok(RoutedProjection {
            region: region(projection)?,
            addressing: ProjectionAddressing::Identity {
                experts: self.experts,
                stride,
            },
            // A compiled bank IS its region, exactly.
            extent: ExtentPolicy::Exact,
        })
    }

    pub fn store_id(&self) -> &str {
        self.store.id()
    }

    /// Register the compiled segment for zero-copy binding.
    pub fn register_store(&self, metal: &MetalBackend) {
        metal.register_weight_region(self.store.backing_bytes());
    }
}

/// Every operand the map compiled is sealed, at the layout's exact
/// offset and length. Returns the layers the ledger covers.
pub fn verify_complete(
    index: &CandidateIndex,
    geometry: &KimiGeometry,
) -> Result<(Vec<u32>, Vec<String>, CandidatePlacement), VindexError> {
    let overlaps = index.ledger.overlaps();
    if !overlaps.is_empty() {
        return Err(VindexError::Parse(format!(
            "candidate ledger has overlapping seals: {overlaps:?}"
        )));
    }
    let mut layers: Vec<u32> = index
        .ledger
        .sealed
        .values()
        .filter_map(|s| layer_of(&s.tensor))
        .collect();
    layers.sort_unstable();
    layers.dedup();
    if layers.is_empty() {
        return Err(VindexError::Parse(
            "candidate ledger seals nothing — an overlay with no overlay".to_string(),
        ));
    }
    // Which projections the candidate actually SEALED. A
    // projection-scoped map compiles one, and completeness is then a
    // claim about that one — demanding all three would refuse a
    // candidate that is complete for what it set out to hold.
    let mut projections: Vec<String> = index
        .ledger
        .sealed
        .values()
        .filter_map(|s| projection_of(&s.tensor).map(str::to_string))
        .collect();
    projections.sort();
    projections.dedup();
    // Whatever the map INTENDED, the seals are what exist. A map naming
    // a projection the ledger does not hold is an incomplete compile,
    // caught by the per-operand check below.
    if let Some(scoped) = index
        .map
        .exceptions
        .iter()
        .find(|e| e.encoding.is_some())
        .and_then(|e| e.projection.clone())
    {
        if !projections.contains(&scoped) {
            return Err(VindexError::Parse(format!(
                "the map scopes projection `{scoped}` but the ledger seals none of it"
            )));
        }
    }
    // ONE placement, the same definition the compiler wrote under: each
    // layer's base is the sum of the preceding compiled layers' extents,
    // each at its OWN encoding — which is what lets a composed map hold
    // a Q8_0 band beside a Q6_K layer in one candidate.
    let placement = CandidatePlacement::resolve(
        &index.map,
        Role::ExpertWeight,
        &layers,
        geometry.experts,
        geometry.hidden,
        geometry.moe_intermediate,
    )?;
    for &layer in &layers {
        let layout = placement.layout(layer)?;
        let layer_base = placement.layer_base(layer)?;
        for expert in 0..geometry.experts {
            for proj in projections.iter().map(String::as_str) {
                let tensor = format!("{layer}.block_sparse_moe.experts.{expert}.{proj}.weight");
                let seal = index.ledger.get(&index.object, &tensor).ok_or_else(|| {
                    VindexError::Parse(format!(
                        "candidate ledger has no seal for `{tensor}` — the compiled bank is \
                         INCOMPLETE and an identity-addressed route to expert {expert} would \
                         read unsealed bytes; refusing rather than falling back to source"
                    ))
                })?;
                let slot = layout.slot(proj, expert)?;
                let want_offset = layer_base + bank_base(layout, proj)? + slot.offset;
                if seal.target_offset != want_offset || seal.target_len != slot.len {
                    return Err(VindexError::Parse(format!(
                        "`{tensor}` is sealed at {}+{} but the layout places it at \
                         {want_offset}+{} — the ledger and the layout disagree",
                        seal.target_offset, seal.target_len, slot.len
                    )));
                }
                if seal.encoding != layout.encoding {
                    return Err(VindexError::Parse(format!(
                        "`{tensor}` is sealed as {} but the map resolves layer {layer} \
                         to {}",
                        seal.encoding, layout.encoding
                    )));
                }
            }
        }
    }
    Ok((layers, projections, placement))
}

/// One compiled KDA layer's binding, copied out of the candidate bank.
pub struct KdaBinding {
    pub qkv_bank: Vec<u8>,
    pub qkv_offsets: [ExpertOffset; 3],
    pub o_proj: Vec<u8>,
    pub encoding: MetalEncoding,
}

/// A compiled KDA candidate beside the source container — the second
/// overlay kind. Opening PROVES completeness: every compiled layer's
/// four projections sealed at exactly the placement's offsets, before
/// any byte is served.
pub struct KdaOverlay {
    pub index: CandidateIndex,
    pub(super) bank: Vec<u8>,
    pub(super) placement: KdaPlacement,
}

impl KdaOverlay {
    pub fn open(
        dir: &Path,
        source_dir: &Path,
        geometry: &KimiGeometry,
    ) -> Result<Self, VindexError> {
        let index: CandidateIndex = serde_json::from_slice(&std::fs::read(dir.join("index.json"))?)
            .map_err(|e| VindexError::Parse(format!("kda candidate index: {e}")))?;
        index.source.verify(&read_source_identity(source_dir)?)?;
        let object = "target.kda_bank";
        let mut layers: Vec<u32> = index
            .ledger
            .sealed
            .values()
            .filter(|seal| seal.object == object)
            .filter_map(|seal| layer_of(&seal.tensor))
            .collect();
        layers.sort_unstable();
        layers.dedup();
        let width = geometry.kda.num_heads * geometry.kda.head_dim;
        let placement = KdaPlacement::resolve(
            &index.map,
            Role::DecoderLinear,
            &layers,
            width,
            geometry.hidden,
        )?;
        // Completeness: all four projections per layer, at the
        // placement's own offsets — a missing or relocated seal refuses
        // here, not as garbage bytes mid-decode.
        for &layer in &layers {
            let layout = placement.layout(layer)?;
            let base = placement.layer_base(layer)?;
            for proj in KDA_PROJECTIONS {
                let tensor = format!("{layer}.self_attn.{proj}.weight");
                let seal = index.ledger.get(object, &tensor).ok_or_else(|| {
                    VindexError::Parse(format!(
                        "KDA candidate has no seal for `{tensor}` — the bank is incomplete"
                    ))
                })?;
                let (off, len) = layout.slot(proj)?;
                if seal.target_offset != base + off || seal.target_len != len {
                    return Err(VindexError::Parse(format!(
                        "`{tensor}` sealed at {}+{} but the placement puts it at {}+{len}",
                        seal.target_offset,
                        seal.target_len,
                        base + off
                    )));
                }
            }
        }
        let bank = std::fs::read(dir.join("segments").join(format!("{object}.bin")))?;
        Ok(Self {
            index,
            bank,
            placement,
        })
    }

    pub fn compiled_layers(&self) -> Vec<u32> {
        self.placement.layers().collect()
    }

    /// The compiled binding for `layer`, or `None` when this candidate
    /// does not hold it. Bytes are verified against the seals' hashes
    /// on every call — 200 MB hashes in milliseconds, and a bank whose
    /// file was truncated or edited must refuse, not decode noise.
    pub fn binding(&self, layer: u32) -> Option<Result<KdaBinding, VindexError>> {
        if !self.placement.layers().any(|l| l == layer) {
            return None;
        }
        Some(self.binding_inner(layer))
    }

    pub(super) fn binding_inner(&self, layer: u32) -> Result<KdaBinding, VindexError> {
        let layout = self.placement.layout(layer)?;
        let base = self.placement.layer_base(layer)?;
        let object = "target.kda_bank";
        let slice = |proj: &str| -> Result<Vec<u8>, VindexError> {
            let (off, len) = layout.slot(proj)?;
            let start = (base + off) as usize;
            let end = start + len as usize;
            let bytes = self.bank.get(start..end).ok_or_else(|| {
                VindexError::Parse(format!(
                    "KDA bank file ends before layer {layer} `{proj}` ({end} > {})",
                    self.bank.len()
                ))
            })?;
            let tensor = format!("{layer}.self_attn.{proj}.weight");
            let seal = self
                .index
                .ledger
                .get(object, &tensor)
                .expect("open() proved completeness");
            let hash = crate::format::vindex3::represent::compile::hash_bytes(bytes);
            if hash != seal.target_hash {
                return Err(VindexError::Parse(format!(
                    "`{tensor}`: bank bytes hash {hash} but the seal says {} — the bank \
                     and its ledger disagree",
                    seal.target_hash
                )));
            }
            Ok(bytes.to_vec())
        };
        let q = slice("q_proj")?;
        let stride = q.len() as u32;
        let mut qkv_bank = q;
        qkv_bank.extend_from_slice(&slice("k_proj")?);
        qkv_bank.extend_from_slice(&slice("v_proj")?);
        let encoding = match layout.encoding.as_str() {
            "BF16" => MetalEncoding::Bf16,
            "Q8_0" => MetalEncoding::Q80,
            "Q6_K" => MetalEncoding::Q6K,
            "Q4_K" => MetalEncoding::Q4K,
            other => {
                return Err(VindexError::Parse(format!(
                    "KDA candidate encodes `{other}`, which no grouped kernel reads"
                )))
            }
        };
        Ok(KdaBinding {
            qkv_bank,
            qkv_offsets: [
                ExpertOffset(0),
                ExpertOffset(stride),
                ExpertOffset(2 * stride),
            ],
            o_proj: slice("o_proj")?,
            encoding,
        })
    }
}
