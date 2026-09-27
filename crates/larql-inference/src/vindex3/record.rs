//! V3-STREAM-1: the run record (`docs/v3-stream-1-run-record.md`).
//!
//! One runner-side consumer turns a tokenwise VINDEX3 run into a
//! LOSSLESS, sequenced, provenance-bearing record that replays to the
//! same events; a separate LIVE TAP may lose events into a bounded
//! channel without ever stalling the executor or touching the record.
//!
//! The executor allocates no identity: `run_id`, the model name, the
//! component and the prompt come from the caller, the provenance from
//! the session's own prepared image, and every event's sequence number
//! and run-relative timestamp from this recorder — assigned BEFORE the
//! live fan-out, so a dropped live event still has an identity.
//!
//! The record's event vocabulary is the runner's, mirrored from the
//! executor's rather than borrowed: the executor's enum is
//! non-exhaustive and carries no serialisation, and a wire schema must
//! not move whenever the executor learns a new event. A variant this
//! module does not know is recorded as [`EventKind::Unknown`] with the
//! executor's own debug spelling — recorded, never skipped.

use std::io::Write as _;
use std::sync::mpsc::{SyncSender, TrySendError};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use larql_vindex::format::vindex3::opplan::exec::intervene::{InterventionKind, InterventionPlan};
use larql_vindex::format::vindex3::opplan::exec::intervene_heads::{
    HeadInterventionKind, HeadInterventionPlan,
};
use larql_vindex::format::vindex3::opplan::exec::observe::{CarrierForm, SublayerSite};
use larql_vindex::format::vindex3::opplan::exec::observe_heads::HeadReader;
use larql_vindex::format::vindex3::opplan::exec::observe_lens::LensReader;
use larql_vindex::format::vindex3::opplan::exec::observe_stats::StatsObserver;
use larql_vindex::format::vindex3::opplan::exec::provenance::RunProvenance;
use serde::{Deserialize, Serialize};

mod recorder;
pub use recorder::*;

/// The schema every record written by this module names.
pub const RECORD_SCHEMA: &str = "larql.run-record.v1";

/// Who ran what: caller-supplied, never executor-allocated.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunIdentity {
    pub run_id: String,
    /// The container's self-declared model name.
    pub model: String,
    pub component: String,
    /// The prompt as token ids, in order.
    pub tokens: Vec<u32>,
    pub schema: String,
    /// Wall-clock start, metadata only; never an ordering key.
    pub started_unix_ms: u64,
    /// V3-INTERVENE-1: the hash of the intervention declaration this run
    /// executed under; `None` for an unintervened run. Part of the
    /// identity, so a record from an intervened run is never read as a
    /// baseline of the same prompt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intervention_sha256: Option<String>,
    /// V3-INTERVENE-2: the hash of the head-intervention declaration this
    /// run executed under; `None` for a run with none. Independent of
    /// [`Self::intervention_sha256`] — a declaration may carry both
    /// carrier and head entries, and each gets its own hash.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub head_intervention_sha256: Option<String>,
}

impl RunIdentity {
    pub fn new(run_id: &str, model: &str, component: &str, tokens: &[u32]) -> Self {
        Self {
            run_id: run_id.to_string(),
            model: model.to_string(),
            component: component.to_string(),
            tokens: tokens.to_vec(),
            schema: RECORD_SCHEMA.to_string(),
            started_unix_ms: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
                .unwrap_or(0),
            intervention_sha256: None,
            head_intervention_sha256: None,
        }
    }
}

/// The runner's spelling of a sublayer site.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Site {
    Attention,
    Ffn,
}

impl From<SublayerSite> for Site {
    fn from(site: SublayerSite) -> Self {
        match site {
            SublayerSite::Attention => Self::Attention,
            SublayerSite::Ffn => Self::Ffn,
        }
    }
}

/// The runner's spelling of a carrier form.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Carrier {
    Single,
    Bundle,
    History,
}

impl From<CarrierForm> for Carrier {
    fn from(form: CarrierForm) -> Self {
        match form {
            CarrierForm::Single => Self::Single,
            CarrierForm::Bundle => Self::Bundle,
            CarrierForm::History => Self::History,
        }
    }
}

/// The runner's spelling of an intervention kind (V3-INTERVENE-1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    Zero,
    Add,
    Replace,
}

impl From<InterventionKind> for Kind {
    fn from(kind: InterventionKind) -> Self {
        match kind {
            InterventionKind::Zero => Self::Zero,
            InterventionKind::Add => Self::Add,
            InterventionKind::Replace => Self::Replace,
        }
    }
}

/// The runner's spelling of a head intervention kind (V3-INTERVENE-2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HeadKind {
    Zero,
    Scale,
    Replace,
}

impl From<HeadInterventionKind> for HeadKind {
    fn from(kind: HeadInterventionKind) -> Self {
        match kind {
            HeadInterventionKind::Zero => Self::Zero,
            HeadInterventionKind::Scale => Self::Scale,
            HeadInterventionKind::Replace => Self::Replace,
        }
    }
}

/// What one recorded event says.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EventKind {
    /// The token was embedded; the position is the envelope's.
    Embedded,
    /// The carrier entering layer 0: its width, not its values.
    EnteringCarrier {
        hidden: usize,
    },
    /// The stats the observer computed AT the write, while the vectors
    /// were still borrowed. Precedes the structural write event, as the
    /// executor fires them.
    CarrierStats {
        layer: usize,
        site: Site,
        norm: f64,
        delta_norm: f64,
        layer_scale: Option<f32>,
        projection: Vec<f32>,
        probe: Vec<f32>,
    },
    CarrierWrite {
        layer: usize,
        site: Site,
        carrier: Carrier,
    },
    AttentionDone {
        layer: usize,
    },
    FfnDone {
        layer: usize,
    },
    Logits {
        vocab: usize,
    },
    /// What the model's own head says at an armed site (V3-LENS-1):
    /// the image's final norm and output head applied to the layer
    /// output, then a full log-softmax. One head pass per readout.
    Readout {
        layer: usize,
        site: Site,
        method: String,
        tokens: Vec<TokenStanding>,
        top: Vec<TopStanding>,
    },
    /// V3-HEAD-OBS-1: this layer's attention emitted one head record per
    /// query head on this step, before its write (structural).
    HeadsObserved {
        layer: usize,
        heads: usize,
    },
    /// V3-HEAD-OBS-1: this layer's attention has no softmax heads, so the
    /// write above it has no head decomposition; the receipt names the
    /// layer as uncovered.
    HeadsUncovered {
        layer: usize,
    },
    /// V3-HEAD-OBS-1, stats level: one query head's child of the
    /// attention write — `‖c′_h‖`, its projection on the run's basis, the
    /// KV head, the top source positions with their weights, and the
    /// sink mass. Keyed beneath the site's own write, which stays
    /// authoritative.
    HeadWrite {
        layer: usize,
        head: usize,
        kv_head: usize,
        norm: f64,
        projection: Vec<f32>,
        sources: Vec<SourceStanding>,
        sink: f32,
    },
    /// V3-HEAD-OBS-1: the head-sum law's measured residual at one
    /// attention write, `‖Σ_h c′_h + bias′ − delta‖ / ‖delta‖`, by the
    /// named method.
    HeadSum {
        layer: usize,
        method: String,
        heads: usize,
        residual: f64,
    },
    /// V3-INTERVENE-1: an intervention fired on this site's write at the
    /// envelope's position. Precedes the write's stats and structural
    /// event, as the executor fires it.
    Intervened {
        layer: usize,
        site: Site,
        intervention: Kind,
    },
    /// V3-INTERVENE-2: a head intervention fired on this head's `ctx_h`,
    /// inside the attention kernel, at the envelope's position. Precedes
    /// the layer's `HeadsObserved`/`HeadWrite`/`HeadSum` and its
    /// attention `CarrierWrite`.
    HeadIntervened {
        layer: usize,
        head: usize,
        intervention: HeadKind,
    },
    /// V3-INTERVENE-2, J5: the shortcut gap for one `zero` head firing —
    /// `‖delta_real − (delta_base − c′_h)‖ / ‖delta_base‖`, where
    /// `delta_base` and `c′_h` are reconstructed from this SAME run's
    /// pre-intervention head records (the head-sum law's own terms, not
    /// a second run). Requires the head reader armed and retaining
    /// children; recorded once per `zero` firing, after that write's
    /// `HeadSum`.
    HeadInterventionGap {
        layer: usize,
        head: usize,
        gap: f64,
    },
    /// An executor event this schema has no spelling for yet. Recorded
    /// with the executor's debug form so nothing is lost.
    Unknown {
        debug: String,
    },
}

/// One source position's attention weight in a [`EventKind::HeadWrite`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceStanding {
    pub position: usize,
    pub weight: f32,
}

/// One declared token's standing in a readout.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TokenStanding {
    pub id: u32,
    pub logprob: f64,
    pub rank: usize,
}

/// One of the top ids in a readout.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TopStanding {
    pub id: u32,
    pub logprob: f64,
}

/// One event with its run-scoped identity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecordedEvent {
    /// Strictly increasing by one from zero within a run.
    pub sequence: u64,
    /// Nanoseconds since the recorder was armed, from a monotonic clock.
    pub timestamp_ns: u64,
    /// The absolute position the event belongs to (the last `Embedded`).
    pub position: usize,
    #[serde(flatten)]
    pub event: EventKind,
}

/// What the live tap could not deliver — kept OUTSIDE the channel that
/// overflowed, so a full channel cannot lose the count of its own loss.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DropLedger {
    pub dropped: u64,
    pub first_dropped_sequence: Option<u64>,
    pub last_dropped_sequence: Option<u64>,
}

impl DropLedger {
    fn note(&mut self, sequence: u64) {
        self.dropped += 1;
        self.first_dropped_sequence.get_or_insert(sequence);
        self.last_dropped_sequence = Some(sequence);
    }
}

/// The lossy consumer: a bounded, non-blocking channel and its ledger.
pub struct LiveTap {
    sender: SyncSender<RecordedEvent>,
    ledger: DropLedger,
}

impl LiveTap {
    pub fn new(sender: SyncSender<RecordedEvent>) -> Self {
        Self {
            sender,
            ledger: DropLedger::default(),
        }
    }

    /// Never blocks: a full or disconnected channel drops the event and
    /// the ledger records which one.
    fn offer(&mut self, event: &RecordedEvent) {
        match self.sender.try_send(event.clone()) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => {
                self.ledger.note(event.sequence);
            }
        }
    }

    pub fn ledger(&self) -> &DropLedger {
        &self.ledger
    }
}

/// The receipt: describes the event log without being part of it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Receipt {
    pub events: u64,
    pub last_sequence: Option<u64>,
    /// SHA-256 over the exact bytes of the event lines as written, each
    /// followed by a newline; the receipt line is not covered.
    pub log_sha256: String,
    pub provenance_fingerprint: String,
    /// Whether the run reached its end; a record without this is a valid
    /// prefix, never a complete run.
    pub complete: bool,
    /// The live tap's loss, if a tap was attached; zero otherwise.
    pub live_dropped: u64,
    /// Head passes the lens performed — its price — zero without a lens.
    #[serde(default)]
    pub head_passes: u64,
    /// The lens's first failure, if it had one; the readouts stop there.
    #[serde(default)]
    pub lens_failure: Option<String>,
    /// V3-HEAD-OBS-1: how many head records the executor handed the head
    /// reader; zero when none was armed.
    #[serde(default)]
    pub head_records: u64,
    /// V3-HEAD-OBS-1: the executed layers whose attention has no softmax
    /// heads, named once each, ascending; empty when heads were not
    /// armed or every layer was covered.
    #[serde(default)]
    pub head_layers_uncovered: Vec<usize>,
    /// V3-HEAD-OBS-1: why the head reader stopped, if it did; the record
    /// stays complete without its rows from that point.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub head_failure: Option<String>,
    /// V3-INTERVENE-1: how many interventions the run declared.
    #[serde(default)]
    pub interventions_declared: u64,
    /// V3-INTERVENE-1: how many firings the run recorded.
    #[serde(default)]
    pub interventions_applied: u64,
    /// V3-INTERVENE-1: a declared address the run never reached, named
    /// at the end of the run. A declared intervention that never fired
    /// is a refusal on the receipt, never a silent no-op.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intervention_refusal: Option<String>,
    /// V3-INTERVENE-2: how many head interventions the run declared.
    #[serde(default)]
    pub head_interventions_declared: u64,
    /// V3-INTERVENE-2: how many head firings the run recorded.
    #[serde(default)]
    pub head_interventions_applied: u64,
    /// V3-INTERVENE-2: a declared head address the run never reached.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub head_intervention_refusal: Option<String>,
}

/// A finished record: header, events, receipt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunRecord {
    pub identity: RunIdentity,
    /// The `RunProvenance` as the executor serialised it.
    pub provenance: serde_json::Value,
    pub provenance_fingerprint: String,
    pub events: Vec<RecordedEvent>,
    pub receipt: Receipt,
}

/// The lossless consumer.
pub struct RunRecorder<'a> {
    identity: RunIdentity,
    provenance: RunProvenance,
    stats: Option<StatsObserver>,
    lens: Option<Box<dyn LensReader + 'a>>,
    /// V3-HEAD-OBS-1: the head reader, when armed; the recorder asks the
    /// executor for heads exactly when one is attached.
    heads: Option<Box<dyn HeadReader + 'a>>,
    /// Layers the executor named as uncovered, ascending, once each.
    uncovered: Vec<usize>,
    armed: Instant,
    position: usize,
    events: Vec<RecordedEvent>,
    live: Option<LiveTap>,
    complete: bool,
    /// V3-INTERVENE-1: the declaration this run executes under, if any.
    interventions: Option<InterventionPlan>,
    /// Firings recorded so far.
    applied: u64,
    /// One past the highest position an event carried, so the receipt
    /// can name declared addresses the run never reached.
    positions_executed: usize,
    /// V3-INTERVENE-2: the head declaration this run executes under.
    head_interventions: Option<HeadInterventionPlan>,
    /// Head firings recorded so far.
    head_applied: u64,
    /// `zero` head firings not yet matched to their write's decomposition
    /// (J5): `(layer, head, position)`, cleared as `carrier_write`
    /// consumes them.
    pending_head_zero: Vec<(usize, usize, usize)>,
}
