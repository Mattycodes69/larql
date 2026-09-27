//! The run recorder, its errors and the record header.

use larql_vindex::format::vindex3::opplan::exec::intervene::InterventionPlan;
use larql_vindex::format::vindex3::opplan::exec::intervene_heads::{
    HeadInterventionKind, HeadInterventionPlan,
};
use larql_vindex::format::vindex3::opplan::exec::observe::{
    CarrierWriteRecord, StepEvent, StepObserver,
};
use larql_vindex::format::vindex3::opplan::exec::observe_heads::{HeadReader, HEAD_SUM_METHOD};
use larql_vindex::format::vindex3::opplan::exec::observe_lens::{LensReader, LENS_METHOD};
use larql_vindex::format::vindex3::opplan::exec::observe_stats::StatsObserver;
use larql_vindex::format::vindex3::opplan::exec::prepared::PreparedOperands;
use larql_vindex::format::vindex3::opplan::exec::provenance::{ExecutionProvenance, RunProvenance};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::mpsc::SyncSender;
use std::time::Instant;

#[allow(unused_imports)]
use super::*;

impl<'a> RunRecorder<'a> {
    /// Arm a recorder with an explicit provenance.
    pub fn arm(
        identity: RunIdentity,
        provenance: RunProvenance,
        stats: Option<StatsObserver>,
    ) -> Self {
        Self {
            identity,
            provenance,
            stats,
            lens: None,
            heads: None,
            uncovered: Vec::new(),
            armed: Instant::now(),
            position: 0,
            events: Vec::new(),
            live: None,
            complete: false,
            interventions: None,
            applied: 0,
            positions_executed: 0,
            head_interventions: None,
            head_applied: 0,
            pending_head_zero: Vec::new(),
        }
    }

    /// Declare the interventions this run executes under (V3-INTERVENE-1).
    /// The declaration's hash joins the run identity; the receipt counts
    /// firings against declarations and names any address never reached.
    pub fn with_interventions(mut self, interventions: &InterventionPlan) -> Self {
        self.identity.intervention_sha256 = interventions.declaration_sha256();
        self.interventions = Some(interventions.clone());
        self
    }

    /// Declare the head interventions this run executes under
    /// (V3-INTERVENE-2). The declaration's hash joins the run identity;
    /// the receipt counts firings against declarations and names any
    /// address never reached. A `zero` firing's shortcut gap (J5) is
    /// computed only when the head reader is ALSO armed and retaining
    /// children ([`HeadReader`]/[`super::super::HeadStats::retaining_children`]).
    pub fn with_head_interventions(mut self, head_interventions: &HeadInterventionPlan) -> Self {
        self.identity.head_intervention_sha256 = head_interventions.declaration_sha256();
        self.head_interventions = Some(head_interventions.clone());
        self
    }

    /// Arm a recorder over a prepared image: the provenance is read off
    /// the image and the stats observer, exactly as the executor states
    /// them.
    pub fn for_image(
        identity: RunIdentity,
        image: &PreparedOperands,
        stats: Option<StatsObserver>,
    ) -> Self {
        let provenance = RunProvenance::new(ExecutionProvenance::of(image), stats.as_ref());
        Self::arm(identity, provenance, stats)
    }

    /// Arm a logit lens (V3-LENS-1): its readouts become events in the
    /// same sequence, and its head passes go on the receipt.
    pub fn with_lens(mut self, lens: Box<dyn LensReader + 'a>) -> Self {
        self.lens = Some(lens);
        self
    }

    /// Arm a head reader (V3-HEAD-OBS-1): the recorder then asks the
    /// executor for per-head records, hands them to the reader, and at
    /// every attention write records the reader's stats-level rows and
    /// the head-sum residual as events in the same sequence; the count,
    /// the uncovered layers and any failure go on the receipt.
    pub fn with_heads(mut self, heads: Box<dyn HeadReader + 'a>) -> Self {
        self.heads = Some(heads);
        self
    }

    /// Attach the live tap. Events already recorded are not replayed
    /// into it.
    pub fn with_live_tap(mut self, sender: SyncSender<RecordedEvent>) -> Self {
        self.live = Some(LiveTap::new(sender));
        self
    }

    pub fn events(&self) -> &[RecordedEvent] {
        &self.events
    }

    pub fn drop_ledger(&self) -> Option<&DropLedger> {
        self.live.as_ref().map(LiveTap::ledger)
    }

    pub fn provenance(&self) -> &RunProvenance {
        &self.provenance
    }

    /// Mark the run as having reached its end.
    pub fn complete(&mut self) {
        self.complete = true;
    }

    pub(super) fn push(&mut self, event: EventKind) {
        let recorded = RecordedEvent {
            sequence: u64::try_from(self.events.len()).expect("sequence fits"),
            timestamp_ns: u64::try_from(self.armed.elapsed().as_nanos()).unwrap_or(u64::MAX),
            position: self.position,
            event,
        };
        if let Some(tap) = &mut self.live {
            tap.offer(&recorded);
        }
        self.events.push(recorded);
    }

    /// Seal the record: the receipt hashes the event lines exactly as
    /// [`RunRecord::write_jsonl`] writes them.
    pub fn finish(self) -> RunRecord {
        let lines = event_lines(&self.events);
        let live_dropped = self.live.as_ref().map(|t| t.ledger.dropped).unwrap_or(0);
        let head_passes = self
            .lens
            .as_ref()
            .map(|l| u64::try_from(l.head_passes()).expect("count fits"))
            .unwrap_or(0);
        let lens_failure = self
            .lens
            .as_ref()
            .and_then(|l| l.failure().map(ToString::to_string));
        let head_records = self
            .heads
            .as_ref()
            .map(|h| u64::try_from(h.records()).expect("count fits"))
            .unwrap_or(0);
        let head_failure = self
            .heads
            .as_ref()
            .and_then(|h| h.failure().map(ToString::to_string));
        let head_layers_uncovered = self.uncovered.clone();
        let fingerprint = self.provenance.fingerprint();
        let interventions_declared = self
            .interventions
            .as_ref()
            .map(|p| u64::try_from(p.declared()).expect("count fits"))
            .unwrap_or(0);
        // A declared address the run never reached is named on the
        // receipt, but only once the run is complete: an incomplete
        // record is a valid prefix, and a prefix has not failed to reach
        // anything yet.
        let intervention_refusal = match (&self.interventions, self.complete) {
            (Some(plan), true) => {
                let unreached = plan.unreached(self.positions_executed);
                (!unreached.is_empty()).then(|| {
                    let named: Vec<String> = unreached
                        .iter()
                        .map(|u| format!("layer {} {:?} position {}", u.layer, u.site, u.position))
                        .collect();
                    format!(
                        "declared intervention never fired: the run executed {} position(s) and \
                         did not reach {}",
                        self.positions_executed,
                        named.join(", ")
                    )
                })
            }
            _ => None,
        };
        let head_interventions_declared = self
            .head_interventions
            .as_ref()
            .map(|p| u64::try_from(p.declared()).expect("count fits"))
            .unwrap_or(0);
        let head_intervention_refusal = match (&self.head_interventions, self.complete) {
            (Some(plan), true) => {
                let unreached = plan.unreached(self.positions_executed);
                (!unreached.is_empty()).then(|| {
                    let named: Vec<String> = unreached
                        .iter()
                        .map(|u| {
                            format!("layer {} head {} position {}", u.layer, u.head, u.position)
                        })
                        .collect();
                    format!(
                        "declared head intervention never fired: the run executed {} \
                         position(s) and did not reach {}",
                        self.positions_executed,
                        named.join(", ")
                    )
                })
            }
            _ => None,
        };
        RunRecord {
            identity: self.identity,
            provenance: serde_json::to_value(&self.provenance).expect("provenance serialises"),
            provenance_fingerprint: fingerprint.clone(),
            receipt: Receipt {
                events: u64::try_from(self.events.len()).expect("count fits"),
                last_sequence: self.events.last().map(|e| e.sequence),
                log_sha256: hash_lines(&lines),
                provenance_fingerprint: fingerprint,
                complete: self.complete,
                live_dropped,
                head_passes,
                lens_failure,
                head_records,
                head_layers_uncovered,
                head_failure,
                interventions_declared,
                interventions_applied: self.applied,
                intervention_refusal,
                head_interventions_declared,
                head_interventions_applied: self.head_applied,
                head_intervention_refusal,
            },
            events: self.events,
        }
    }
}

impl StepObserver for RunRecorder<'_> {
    fn event(&mut self, event: StepEvent) {
        let kind = match event {
            StepEvent::Embedded { position } => {
                self.position = position;
                self.positions_executed = self.positions_executed.max(position + 1);
                EventKind::Embedded
            }
            StepEvent::AttentionDone { layer } => EventKind::AttentionDone { layer },
            StepEvent::FfnDone { layer } => EventKind::FfnDone { layer },
            StepEvent::Logits { vocab } => EventKind::Logits { vocab },
            StepEvent::HeadsObserved { layer, heads } => EventKind::HeadsObserved { layer, heads },
            StepEvent::HeadsUncovered { layer } => {
                if let Err(at) = self.uncovered.binary_search(&layer) {
                    self.uncovered.insert(at, layer);
                }
                EventKind::HeadsUncovered { layer }
            }
            StepEvent::Intervened { layer, site, kind } => {
                self.applied += 1;
                EventKind::Intervened {
                    layer,
                    site: site.into(),
                    intervention: kind.into(),
                }
            }
            StepEvent::HeadIntervened { layer, head, kind } => {
                self.head_applied += 1;
                if kind == HeadInterventionKind::Zero {
                    self.pending_head_zero.push((layer, head, self.position));
                }
                EventKind::HeadIntervened {
                    layer,
                    head,
                    intervention: kind.into(),
                }
            }
            StepEvent::CarrierWrite {
                layer,
                site,
                carrier,
            } => EventKind::CarrierWrite {
                layer,
                site: site.into(),
                carrier: carrier.into(),
            },
            other => EventKind::Unknown {
                debug: format!("{other:?}"),
            },
        };
        self.push(kind);
    }

    fn entering_carrier(&mut self, position: usize, values: &[f32]) {
        self.position = position;
        self.push(EventKind::EnteringCarrier {
            hidden: values.len(),
        });
    }

    fn wants_attention_heads(&self) -> bool {
        self.heads.is_some()
    }

    fn attention_head(
        &mut self,
        layer: usize,
        record: larql_vindex::format::vindex3::opplan::exec::observe::AttentionHeadRecord<'_>,
    ) {
        if let Some(heads) = &mut self.heads {
            heads.attention_head(layer, record);
        }
    }

    fn carrier_write(&mut self, record: CarrierWriteRecord<'_>) {
        // V3-HEAD-OBS-1: the write's head decomposition, keyed beneath the
        // write and recorded before its stats and structural event.
        let head_write = match &mut self.heads {
            Some(heads) => heads.finish_write(&record),
            None => None,
        };
        if let Some(write) = head_write {
            // V3-INTERVENE-2, J5: a `zero` firing at THIS write's
            // (layer, position) gets its shortcut gap now — the one
            // moment `write.sum` (delta_base) and `write.children`
            // (c′_h, from the SAME run's pre-intervention records) are
            // both in hand, beside the write's own `delta` (delta_real).
            let (matched, remaining): (Vec<_>, Vec<_>) = self
                .pending_head_zero
                .drain(..)
                .partition(|&(layer, _, position)| {
                    layer == write.layer && position == record.position
                });
            self.pending_head_zero = remaining;
            for (layer, head, _position) in matched {
                if let Some(children) = &write.children {
                    if let Some(i) = write.rows.iter().position(|row| row.head == head) {
                        let base_norm: f64 = write
                            .sum
                            .iter()
                            .map(|v| f64::from(*v).powi(2))
                            .sum::<f64>()
                            .sqrt();
                        let shortcut: Vec<f64> = write
                            .sum
                            .iter()
                            .zip(&children[i])
                            .map(|(s, c)| f64::from(*s) - f64::from(*c))
                            .collect();
                        let err: f64 = shortcut
                            .iter()
                            .zip(record.delta)
                            .map(|(s, d)| (s - f64::from(*d)).powi(2))
                            .sum::<f64>()
                            .sqrt();
                        let gap = if base_norm > 0.0 {
                            err / base_norm
                        } else {
                            err
                        };
                        self.push(EventKind::HeadInterventionGap { layer, head, gap });
                    }
                }
            }
            self.push(EventKind::HeadSum {
                layer: write.layer,
                method: HEAD_SUM_METHOD.to_string(),
                heads: write.rows.len(),
                residual: write.residual,
            });
            for row in write.rows {
                self.push(EventKind::HeadWrite {
                    layer: write.layer,
                    head: row.head,
                    kv_head: row.kv_head,
                    norm: row.norm,
                    projection: row.projection,
                    sources: row
                        .sources
                        .into_iter()
                        .map(|(position, weight)| SourceStanding { position, weight })
                        .collect(),
                    sink: row.sink,
                });
            }
        }
        if let Some(stats) = &mut self.stats {
            stats.carrier_write(record);
            let row = stats
                .rows
                .pop()
                .expect("the stats observer appends one row per write");
            self.push(EventKind::CarrierStats {
                layer: row.layer,
                site: row.site.into(),
                norm: row.norm,
                delta_norm: row.delta_norm,
                layer_scale: row.layer_scale,
                projection: row.projection,
                probe: row.probe,
            });
        }
        let readout = match &mut self.lens {
            Some(lens) => lens.read(record),
            None => None,
        };
        if let Some(readout) = readout {
            self.push(EventKind::Readout {
                layer: readout.layer,
                site: readout.site.into(),
                method: LENS_METHOD.to_string(),
                tokens: readout
                    .tokens
                    .into_iter()
                    .map(|t| TokenStanding {
                        id: t.id,
                        logprob: t.logprob,
                        rank: t.rank,
                    })
                    .collect(),
                top: readout
                    .top
                    .into_iter()
                    .map(|(id, logprob)| TopStanding { id, logprob })
                    .collect(),
            });
        }
    }
}

/// Why a record could not be read back.
#[derive(Debug, thiserror::Error)]
pub enum RecordError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("record line {line}: {source}")]
    Parse {
        line: usize,
        #[source]
        source: serde_json::Error,
    },
    #[error("record has no header line")]
    MissingHeader,
    #[error("record has no receipt line; it is a prefix, not a run")]
    MissingReceipt,
    #[error("the receipt's log hash {expected} does not match the event lines ({actual})")]
    HashMismatch { expected: String, actual: String },
    #[error("the receipt counts {expected} events but {actual} lines were read")]
    CountMismatch { expected: u64, actual: u64 },
}

#[derive(Serialize, Deserialize)]
pub(super) struct Header {
    pub(super) identity: RunIdentity,
    pub(super) provenance: serde_json::Value,
    pub(super) provenance_fingerprint: String,
}

#[derive(Serialize, Deserialize)]
#[serde(untagged)]
pub(super) enum Line {
    Header { header: Header },
    Receipt { receipt: Receipt },
    Event(RecordedEvent),
}

/// The event lines exactly as written and hashed.
pub fn event_lines(events: &[RecordedEvent]) -> Vec<String> {
    events
        .iter()
        .map(|e| serde_json::to_string(e).expect("an event serialises"))
        .collect()
}

/// SHA-256 over each line followed by a newline, as lowercase hex.
pub fn hash_lines(lines: &[String]) -> String {
    let mut hasher = Sha256::new();
    for line in lines {
        hasher.update(line.as_bytes());
        hasher.update(b"\n");
    }
    hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

impl RunRecord {
    /// One JSON object per line: the header, every event, the receipt.
    pub fn write_jsonl(&self, path: &Path) -> Result<(), RecordError> {
        let mut out = std::io::BufWriter::new(std::fs::File::create(path)?);
        let header = Line::Header {
            header: Header {
                identity: self.identity.clone(),
                provenance: self.provenance.clone(),
                provenance_fingerprint: self.provenance_fingerprint.clone(),
            },
        };
        writeln!(
            out,
            "{}",
            serde_json::to_string(&header).expect("header serialises")
        )?;
        for line in event_lines(&self.events) {
            writeln!(out, "{line}")?;
        }
        let receipt = Line::Receipt {
            receipt: self.receipt.clone(),
        };
        writeln!(
            out,
            "{}",
            serde_json::to_string(&receipt).expect("receipt serialises")
        )?;
        out.flush()?;
        Ok(())
    }

    /// Read a record back and VERIFY it: the receipt's hash must match
    /// the event lines and its count must match the lines read. A
    /// record that fails either is refused, not returned as a prefix.
    pub fn read_jsonl(path: &Path) -> Result<Self, RecordError> {
        let reader = BufReader::new(std::fs::File::open(path)?);
        let mut header: Option<Header> = None;
        let mut receipt: Option<Receipt> = None;
        let mut events = Vec::new();
        let mut lines = Vec::new();
        for (index, line) in reader.lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let parsed: Line =
                serde_json::from_str(&line).map_err(|source| RecordError::Parse {
                    line: index + 1,
                    source,
                })?;
            match parsed {
                Line::Header { header: h } => header = Some(h),
                Line::Receipt { receipt: r } => receipt = Some(r),
                Line::Event(event) => {
                    lines.push(line);
                    events.push(event);
                }
            }
        }
        let header = header.ok_or(RecordError::MissingHeader)?;
        let receipt = receipt.ok_or(RecordError::MissingReceipt)?;
        let actual = hash_lines(&lines);
        if actual != receipt.log_sha256 {
            return Err(RecordError::HashMismatch {
                expected: receipt.log_sha256,
                actual,
            });
        }
        let count = u64::try_from(events.len()).expect("count fits");
        if count != receipt.events {
            return Err(RecordError::CountMismatch {
                expected: receipt.events,
                actual: count,
            });
        }
        Ok(Self {
            identity: header.identity,
            provenance: header.provenance,
            provenance_fingerprint: header.provenance_fingerprint,
            events,
            receipt,
        })
    }
}
