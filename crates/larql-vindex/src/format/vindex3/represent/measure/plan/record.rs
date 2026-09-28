//! The run's record on disk: the report, the per-position evidence, the
//! optional sketch, and the digests that bind the evidence to the receipt.
//!
//! The evidence files are serialised once, in memory, and the bytes that are
//! hashed are the bytes that are written. The receipt carries the digests, so
//! a `positions.jsonl` or `sketch.f32` replaced after the run no longer
//! matches an admissible receipt ([`verify_record`]).

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::metrics::PositionMetrics;
use super::sketch::{sketch_bytes, SketchSpec, SKETCH_FILE, SKETCH_GENERATOR};
use super::{
    identity, PlanExecutionFailure, PlanInadmissible, PlanMeasureRequest, PlanReceipt, PlanRefusal,
    ARMS, P99_BIASED_BELOW_SEQUENCES, P99_QUARTER_PRECISION_SEQUENCES, POSITIONS_FILE, PROCEDURE,
    RECEIPT_FILE, REPORT_FILE,
};
use crate::format::vindex3::represent::token_bank::{sha256_hex, TokenBank};

/// The sketch a receipt binds: which generator, which projection, how many
/// rows, and the digest of the file as written.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SketchRecord {
    pub generator: String,
    pub spec: SketchSpec,
    pub positions: usize,
    pub sha256: String,
}

/// The evidence files' bytes, before they are written.
pub(super) struct Evidence {
    positions: Vec<u8>,
    sketch: Option<(SketchRecord, Vec<u8>)>,
}

impl Evidence {
    /// Serialise the positions (one JSON object per line) and, if a sketch
    /// was asked for, its rows.
    pub(super) fn serialise(
        positions: &[PositionMetrics],
        sketch: Option<(&SketchSpec, &[Vec<f32>])>,
    ) -> Result<Self, serde_json::Error> {
        let mut lines = Vec::new();
        for position in positions {
            serde_json::to_writer(&mut lines, position)?;
            lines.push(b'\n');
        }
        let sketch = sketch.map(|(spec, rows)| {
            let bytes = sketch_bytes(rows);
            let record = SketchRecord {
                generator: SKETCH_GENERATOR.to_string(),
                spec: *spec,
                positions: rows.len(),
                sha256: sha256_hex(&bytes),
            };
            (record, bytes)
        });
        Ok(Self {
            positions: lines,
            sketch,
        })
    }

    pub(super) fn positions_sha256(&self) -> String {
        sha256_hex(&self.positions)
    }

    pub(super) fn sketch_record(&self) -> Option<SketchRecord> {
        self.sketch.as_ref().map(|(record, _)| record.clone())
    }
}

pub(super) fn write_json(path: &Path, value: &impl Serialize) -> std::io::Result<()> {
    let bytes = serde_json::to_vec_pretty(value).map_err(std::io::Error::other)?;
    std::fs::write(path, bytes)
}

/// A container's bound representations and its declared program, for the
/// report.
pub(super) type ContainerRecord = (
    BTreeMap<String, identity::RecordedRepresentation>,
    Option<crate::format::vindex3::represent::map::PrecisionMap>,
);

/// The sketch if one was asked for, the report, then the positions. The
/// receipt is `run`'s.
pub(super) fn write_record(
    request: &PlanMeasureRequest,
    bank: &TokenBank,
    receipt: &PlanReceipt,
    evidence: &Evidence,
    containers: [ContainerRecord; ARMS],
) -> std::io::Result<()> {
    std::fs::create_dir_all(&request.output)?;
    let [(reference_digests, reference_program), (candidate_digests, candidate_program)] =
        containers;
    let mut report = serde_json::json!({
        "procedure": PROCEDURE,
        "label": request.label,
        "bank": {
            "dir": request.bank,
            "id": bank.manifest().bank_id,
            "prompts": bank.manifest().prompts,
            "tokenizer_sha256": bank.manifest().tokenizer_sha256,
            "sequences": request.sequences,
            "of": bank.sample_count(),
        },
        "provenance": request.provenance,
        "reference": {
            "arm": receipt.reference,
            "representations": reference_digests,
            "precision_map": reference_program,
        },
        "candidate": {
            "arm": receipt.candidate,
            "representations": candidate_digests,
            "precision_map": candidate_program,
        },
        "facts": receipt.facts,
        "summary": receipt.summary,
        "sample_size": {
            "sequences": request.sequences,
            "p99_biased_low_below": P99_BIASED_BELOW_SEQUENCES,
            "p99_quarter_precision_from": P99_QUARTER_PRECISION_SEQUENCES,
            "adequate_for_p99": request.sequences >= P99_QUARTER_PRECISION_SEQUENCES,
        },
        "units": "nats, full vocabulary",
    });
    if let Some((record, bytes)) = &evidence.sketch {
        let mut described = record.spec.describe(record.positions);
        described["sha256"] = record.sha256.clone().into();
        report["sketch"] = described;
        std::fs::write(request.output.join(SKETCH_FILE), bytes)?;
    }
    write_json(&request.output.join(REPORT_FILE), &report)?;
    std::fs::write(request.output.join(POSITIONS_FILE), &evidence.positions)
}

/// Re-read a finished run's evidence against its receipt. Returns the
/// receipt when `positions.jsonl`, and `sketch.f32` if the receipt binds
/// one, are the bytes the run wrote.
pub fn verify_record(dir: &Path) -> Result<PlanReceipt, PlanRefusal> {
    let unreadable = |detail: String| {
        PlanRefusal::Execution(PlanExecutionFailure::ArtifactUnreadable { detail })
    };
    let refused = |detail: &str| {
        PlanRefusal::Execution(PlanExecutionFailure::RequestRefused {
            detail: detail.into(),
        })
    };
    let read =
        |file: &str| std::fs::read(dir.join(file)).map_err(|e| unreadable(format!("{file}: {e}")));
    let written: serde_json::Value = serde_json::from_slice(&read(RECEIPT_FILE)?)
        .map_err(|e| unreadable(format!("{RECEIPT_FILE}: {e}")))?;
    if written["admissible"] != serde_json::Value::Bool(true) {
        return Err(refused(
            "the receipt is not admissible; there is no evidence to verify",
        ));
    }
    let receipt: PlanReceipt = serde_json::from_value(written["receipt"].clone())
        .map_err(|e| unreadable(format!("{RECEIPT_FILE}: {e}")))?;
    if receipt.positions_sha256.is_empty() {
        return Err(refused("the receipt predates evidence digests"));
    }
    let check = |file: &str, expected: &str| {
        let found = sha256_hex(&read(file)?);
        if found == expected {
            Ok(())
        } else {
            Err(PlanRefusal::Inadmissible(PlanInadmissible::SealMismatch {
                what: file.to_string(),
                expected: expected.to_string(),
                found,
            }))
        }
    };
    check(POSITIONS_FILE, &receipt.positions_sha256)?;
    if let Some(sketch) = &receipt.sketch {
        check(SKETCH_FILE, &sketch.sha256)?;
    }
    Ok(receipt)
}
