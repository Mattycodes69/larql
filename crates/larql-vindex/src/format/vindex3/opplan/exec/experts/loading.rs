//! Loading packed and f32 expert banks.

use super::super::backend::WeightFormat;
use super::super::narrow::f32_bytes_to_f16;
use super::super::operands::OperandSource;
use super::super::realization::RepresentationFacts;
use super::super::weights::{quantize_mxfp4, quantize_nvfp4, AlignedBytes, LoadedWeight};
use crate::error::VindexError;
use crate::format::vindex3::opplan::planned::declared_bank_representation;
use crate::format::vindex3::opplan::{OperandRef, PackedProjection, RoutedFfnOp};
use crate::format::vindex3::represent::codec::codecs::lyrw2::bind_region;
use crate::format::vindex3::represent::codec::codecs::mxfp4::DTYPE_MXFP4;
use crate::format::vindex3::represent::codec::streams::{GROUP_SCALES, VALUES};
use larql_models::quant::mxfp4::{MXFP4_GROUP_BYTES, MXFP4_GROUP_ELEMS};

#[allow(unused_imports)]
use super::*;

/// Load one packed projection as `experts` matrices of `[rows, k]` in
/// `format`: the bank bound once through its codec, each expert decoded
/// as a row range of it.
pub(super) fn load_packed(
    store: OperandSource<'_>,
    projection: &PackedProjection,
    op: &RoutedFfnOp,
    rows: usize,
    k: usize,
    format: WeightFormat,
) -> Result<Vec<LoadedWeight>, VindexError> {
    let name = projection.weights.tensor.as_str();
    let facts = bank_facts(store, op, &projection.weights)?;
    let Some(registered) = facts.registered.as_ref() else {
        return Err(VindexError::Parse(format!(
            "`{name}`: `{}` names no registered codec, so the bank cannot be decoded",
            facts.label
        )));
    };
    // Declared geometry before bytes: a `k` the codec's row alignment
    // cannot tile is a fact about the op, and refusing it costs no read.
    if !registered.capabilities.admits_k(k) {
        return Err(VindexError::Parse(format!(
            "`{name}`: k={k} is not a whole number of `{}` row alignments of {} elements",
            facts.label, registered.capabilities.row_align_elems
        )));
    }
    // The same admission the selector made before this loader was
    // reached — one derivation, so a direct caller gets the same refusal
    // rather than a read followed by a decode-time one.
    facts.admit_row_slicing()?;
    let codec = store.registry().resolve(&facts.label, name)?;
    let raw = store.load_raw(&projection.weights)?;
    let scales = projection
        .scales
        .as_ref()
        .map(|scales_ref| store.load_raw(scales_ref))
        .transpose()?;
    // The whole bank is one region of `experts × rows` rows: the codec
    // validates every stream's length against that geometry, and a codec
    // that declares no scales stream refuses a partner it cannot consume.
    let bank_shape = [op.experts * rows, k];
    let operands = bind_region(
        codec,
        &bank_shape,
        &raw.bytes,
        scales.as_ref().map(|s| s.bytes.as_slice()),
        name,
    )?;
    (0..op.experts)
        .map(|e| {
            let expert_rows = e * rows..(e + 1) * rows;
            match format {
                // Native: the bank IS MXFP4 — the codec bound it as
                // MXFP4's two streams — so each expert's rows are one
                // contiguous slab of each stream, copied as they are.
                WeightFormat::Mxfp4 if facts.label == DTYPE_MXFP4 => {
                    let groups = k / MXFP4_GROUP_ELEMS;
                    let code_row = groups * MXFP4_GROUP_BYTES;
                    let codes = operands.stream_of_len(
                        VALUES,
                        expert_rows.end * code_row,
                        DTYPE_MXFP4,
                        name,
                    )?;
                    let scales = operands.stream_of_len(
                        GROUP_SCALES,
                        expert_rows.end * groups,
                        DTYPE_MXFP4,
                        name,
                    )?;
                    Ok(LoadedWeight::Mxfp4 {
                        packed: AlignedBytes::from_bytes(
                            &codes[expert_rows.start * code_row..expert_rows.end * code_row],
                        ),
                        scales: AlignedBytes::from_bytes(
                            &scales[expert_rows.start * groups..expert_rows.end * groups],
                        ),
                    })
                }
                // Everything else decodes through the codec and converts
                // exactly as a dense matrix would.
                other => {
                    let mut values = vec![0.0f32; rows * k];
                    // The bank was bound whole, so it decodes whole: the
                    // codec's deepest extent, not depth 0.
                    codec.decode_rows(
                        &operands,
                        &bank_shape,
                        expert_rows,
                        codec.terminal_extent(),
                        &mut values,
                        name,
                    )?;
                    from_f32(values, rows, k, other, name)
                }
            }
        })
        .collect()
}

pub(super) fn from_f32(
    values: Vec<f32>,
    rows: usize,
    k: usize,
    format: WeightFormat,
    name: &str,
) -> Result<LoadedWeight, VindexError> {
    match format {
        WeightFormat::F32 => Ok(LoadedWeight::F32(
            super::super::weights::staged::StagedF32::stage(values)?,
        )),
        WeightFormat::F16 => {
            let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            Ok(LoadedWeight::F16(f32_bytes_to_f16(&bytes, name)?))
        }
        // A packed expert bank arrives already widened to f32, so the
        // CPU compact formats have no stored bytes to keep here — the
        // same reason `Bf16` is refused below. Naming them explicitly
        // rather than falling through keeps the refusal a decision.
        // Same refusal as Q4 and Bf16 below, and for a sharper reason:
        // fine-grained FP8's stored form is the CHECKPOINT's, so a bank
        // that has already been widened to f32 has irreversibly left it.
        // Re-quantising here would manufacture codes and scales nothing
        // declared — a different tensor wearing the format's name.
        WeightFormat::Fp8Block => Err(VindexError::Parse(format!(
            "expert bank `{name}` cannot be made FP8-resident: the bank is widened to f32 on \
             the way in, and fine-grained FP8 is a stored form this build never manufactures"
        ))),
        WeightFormat::Q4 => Err(VindexError::Parse(format!(
            "expert bank `{name}` cannot be made q4-resident: the bank is widened to f32 on \
             the way in, so there is nothing compact left to keep"
        ))),
        WeightFormat::Nvfp4Q8 => Err(VindexError::Parse(format!(
            "expert bank `{name}` cannot bind a stored NVFP4 pack for a Q8 activation: the bank \
             is widened to f32 on the way in, so there is no stored pack left to bind"
        ))),
        WeightFormat::KQuant | WeightFormat::KQuantQ8k => Err(VindexError::Parse(format!(
            "expert bank `{name}` cannot bind a stored K-quant: the bank is widened to f32 on \
             the way in, so the stored blocks are no longer what is being bound"
        ))),
        WeightFormat::Mxfp4 => quantize_mxfp4(&values, rows, k, name),
        WeightFormat::Nvfp4 => quantize_nvfp4(&values, rows, k, name),
        // This path has already widened to f32 (packed expert banks
        // arrive that way), and narrowing back would ROUND — bf16
        // residency means the stored bytes are the resident bytes, and
        // there are no stored bytes left here to keep. Refuse rather
        // than quietly return something the format does not promise.
        WeightFormat::Bf16 | WeightFormat::Q8 => Err(VindexError::Parse(format!(
            "tensor `{name}`: compact residency needs the stored bytes, and this expert path \
             has already widened to f32"
        ))),
        // Same reasoning as Bf16/Q8/KQuant above, and sharper: this
        // format's bytes are whatever a codec's own encoder produced, and
        // this path has neither a codec to ask nor the stored bytes left
        // to keep — only an already-widened f32 image.
        WeightFormat::CodecOwned => Err(VindexError::Parse(format!(
            "expert bank `{name}` cannot bind codec-owned bytes: the bank is widened to f32 on \
             the way in, so there are no stored bytes left to hand back, and this loader has no \
             codec to re-encode them through"
        ))),
    }
}

/// What a packed bank IS: the same resolution the selector made before
/// this loader was reached — the container's label when it names a codec,
/// else the codec the plan's declared layout carries — so a direct caller
/// is refused exactly as the selector would have refused it.
pub(super) fn bank_facts(
    store: OperandSource<'_>,
    op: &RoutedFfnOp,
    operand: &OperandRef,
) -> Result<RepresentationFacts, VindexError> {
    let registry = store.registry();
    let declared = declared_bank_representation(op.expert_format);
    match (store.stored_dtype(operand), declared) {
        (Some(stored), declared) => Ok(RepresentationFacts::resolve_declared(
            registry, stored, declared,
        )),
        (None, Some(declared)) => Ok(RepresentationFacts::resolve_in(registry, declared)),
        (None, None) => Err(VindexError::Parse(format!(
            "`{}`: the bank is neither stored under a label nor declared by the plan",
            operand.tensor
        ))),
    }
}

/// The scalar a gated shared branch is multiplied by, for one token:
/// `activation(weight · x)` under the gate's judged semantics. Every
/// variant is matched, so a gate semantic this build has not judged fails
/// to compile here rather than running the branch at the wrong weight.
pub(in super::super) fn shared_branch_scale(
    spec: &larql_models::config::SharedExpertGateSpec,
    weight: &[f32],
    x: &[f32],
) -> Result<f32, VindexError> {
    use larql_models::config::{GateActivation, GateCombine, SharedExpertGateSource};
    if weight.len() != x.len() {
        return Err(VindexError::Parse(format!(
            "shared-expert branch gate has {} weights for a {}-wide input; the gate is \
             one logit per token over the block input",
            weight.len(),
            x.len()
        )));
    }
    let SharedExpertGateSource::HiddenStateToScalar = spec.source;
    let logit: f32 = weight.iter().zip(x).map(|(w, v)| w * v).sum();
    let gate = match spec.activation {
        GateActivation::Sigmoid => 1.0 / (1.0 + (-logit).exp()),
    };
    match spec.combine {
        GateCombine::ElementwiseMultiply => Ok(gate),
    }
}
