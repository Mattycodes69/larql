//! Selecting a realization per operand.

use super::super::backend::{MatrixClass, WeightFormat};
use super::super::cpu::physical::PhysicalProjectionPlan;
use crate::format::vindex3::opplan::planned::{Operation, PlannedOperand};
use crate::format::vindex3::represent::codec::ResidencyProfile;

#[allow(unused_imports)]
use super::*;

/// The selection every backend makes for the operations that are not a
/// projection, given the candidate it offers for them; `None` for the
/// shared expert, which no backend binds through the prepared plan.
pub fn common_selection(
    operand: &PlannedOperand,
    facts: &RepresentationFacts,
    bank_convert: WeightFormat,
) -> Option<Result<Selection, Box<SelectionRefusal>>> {
    let refuse = |kind, considered: Vec<(RealizationId, String)>| {
        Box::new(SelectionRefusal {
            operand: operand.operand.clone(),
            operation: operand.operation,
            representation: facts.label.clone(),
            requested: operand.access,
            kind,
            considered,
        })
    };
    match operand.operation {
        Operation::Embed => {
            let id = RealizationId::cpu(RealizationForm::DecodedGather);
            Some(if facts.registered.is_some() {
                Ok(Selection {
                    realization: id,
                    residency: ResidencyProfile::DECODED_F32,
                    reason: SelectionReason::EmbeddingGather,
                    candidates: vec![id],
                })
            } else {
                Err(refuse(RefusalKind::UnregisteredRepresentation, vec![]))
            })
        }
        Operation::ExpertBankSlice => {
            let id = RealizationId::cpu(RealizationForm::SliceStored {
                convert: bank_convert,
            });
            if facts.registered.is_none() {
                return Some(Err(refuse(RefusalKind::UnregisteredRepresentation, vec![])));
            }
            Some(match facts.admit_row_slicing() {
                Ok(()) => Ok(Selection {
                    realization: id,
                    residency: resident_profile(bank_convert),
                    reason: SelectionReason::BankSlicedAtLoad,
                    candidates: vec![id],
                }),
                Err(e) => Err(refuse(
                    RefusalKind::AccessRefused,
                    vec![(id, e.to_string())],
                )),
            })
        }
        // The scalar gate on a shared branch (Qwen MoE): one row, read
        // whole as f32 glue and applied by the routed-FFN composer the
        // same way on every backend, so there is one candidate.
        Operation::SharedExpertBranchGate => {
            let id = RealizationId::cpu(RealizationForm::Decode(PhysicalProjectionPlan::ScalarF32));
            Some(if facts.registered.is_some() {
                Ok(Selection {
                    realization: id,
                    residency: ResidencyProfile::DECODED_F32,
                    reason: SelectionReason::ScalarBranchGate,
                    candidates: vec![id],
                })
            } else {
                Err(refuse(RefusalKind::UnregisteredRepresentation, vec![]))
            })
        }
        // A per-expert bank: the stored bytes are bound as a mapping and
        // executed in their stored form when the executor has a kernel
        // for that form over a whole matrix; nothing is copied or
        // converted, so there is exactly one candidate.
        Operation::ExpertProject { .. } => {
            let Some(registered) = &facts.registered else {
                return Some(Err(refuse(RefusalKind::UnregisteredRepresentation, vec![])));
            };
            let format = mapped_format(&facts.label);
            let id = RealizationId::cpu(RealizationForm::MappedStored {
                format: format.unwrap_or(WeightFormat::F32),
                access: MappedAccess::Demand,
            });
            Some(
                match (
                    format,
                    registered
                        .capabilities
                        .require(operand.access, &facts.label),
                ) {
                    (Some(format), Ok(())) => Ok(Selection {
                        realization: id,
                        residency: resident_profile(format),
                        reason: SelectionReason::BankMappedAsStored,
                        candidates: vec![id],
                    }),
                    (Some(_), Err(e)) => Err(refuse(
                        RefusalKind::AccessRefused,
                        vec![(id, e.to_string())],
                    )),
                    (None, _) => Err(refuse(
                        RefusalKind::MissingRealization,
                        vec![(
                            id,
                            format!(
                            "`{}` has no in-place kernel over a whole matrix; a per-expert bank \
                             is never decoded or copied",
                            facts.label
                        ),
                        )],
                    )),
                },
            )
        }
        // A shared expert's projections are whole matrices: the same
        // candidates as any dense FFN projection, chosen by the backend.
        Operation::Project(_) | Operation::OutputHead | Operation::SharedExpertProject => None,
    }
}

/// What `id` would make resident for an operand with these facts — the
/// ONE pricing every selector and the budget's re-selection read, so a
/// candidate costs the same wherever it is weighed: a direct kernel the
/// codec's own declared profile, a decode the codec's decode residency,
/// every executor-owned form the executor's own geometry.
pub fn realization_residency(facts: &RepresentationFacts, id: RealizationId) -> ResidencyProfile {
    match id.form {
        RealizationForm::Direct(plan) => facts.direct_residency(plan).unwrap_or_else(|| {
            facts
                .registered
                .as_ref()
                .map(|r| r.decode_residency)
                .unwrap_or(ResidencyProfile::DECODED_F32)
        }),
        RealizationForm::Decode(_) => facts
            .registered
            .as_ref()
            .map(|r| r.decode_residency)
            .unwrap_or(ResidencyProfile::DECODED_F32),
        RealizationForm::DecodedGather => ResidencyProfile::DECODED_F32,
        RealizationForm::Requantise(_)
        | RealizationForm::SliceStored { .. }
        | RealizationForm::MappedStored { .. }
        | RealizationForm::DeviceResident(_) => resident_profile(id.format()),
    }
}

/// The resident form a stored label executes in WITHOUT conversion, when
/// the CPU executor has a whole-matrix kernel for it: bf16 through the
/// fused bf16 matvec, f32 through BLAS. Every other stored form would need
/// a decode, which a mapped binding by definition does not do.
pub(super) fn mapped_format(label: &str) -> Option<WeightFormat> {
    use crate::format::vindex3::represent::codec::codecs::float::FloatDtype;
    if label == FloatDtype::Bf16.label() {
        Some(WeightFormat::Bf16)
    } else if label == FloatDtype::F32.label() {
        Some(WeightFormat::F32)
    } else {
        None
    }
}

/// The reference backend's answer: the literal f32 transcription, always,
/// for every projection of a registered representation.
pub fn reference_selection(
    operand: &PlannedOperand,
    facts: &RepresentationFacts,
) -> Result<Selection, Box<SelectionRefusal>> {
    if let Some(common) = common_selection(operand, facts, WeightFormat::F32) {
        return common;
    }
    let id = RealizationId::cpu(RealizationForm::Decode(PhysicalProjectionPlan::ScalarF32));
    match &facts.registered {
        Some(r) => Ok(Selection {
            realization: id,
            residency: r.decode_residency,
            reason: SelectionReason::ReferenceOracle,
            candidates: vec![id],
        }),
        None => Err(Box::new(SelectionRefusal {
            operand: operand.operand.clone(),
            operation: operand.operation,
            representation: facts.label.clone(),
            requested: operand.access,
            kind: RefusalKind::UnregisteredRepresentation,
            considered: vec![],
        })),
    }
}

/// The class a projection-class operation names; the two non-projection
/// operations never reach a class table.
pub fn class_of(operation: Operation) -> Option<MatrixClass> {
    match operation {
        Operation::Project(class) => Some(class),
        Operation::OutputHead => Some(MatrixClass::OutputHead),
        Operation::SharedExpertProject => Some(MatrixClass::FfnProjection),
        Operation::Embed
        | Operation::ExpertBankSlice
        | Operation::SharedExpertBranchGate
        | Operation::ExpertProject { .. } => None,
    }
}
