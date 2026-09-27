//! Whether a carrier hand-off is the next legal one (RESIDUAL-BUS-2, S1–S3).
//!
//! An address says where a transition belongs and an identity says which
//! computation produced it. Neither says whether this transition is the
//! one a receiver should accept NEXT. A stream that crosses a process can
//! arrive duplicated, with a transition missing, reordered, carrying a
//! position nobody asked for, or cut short, and every one of those is a
//! plausible stream of the wrong computation.
//!
//! The sequence is two-dimensional (D4): a declared domain of absolute
//! positions per operation, and an ordinal within each position. Batch
//! emits layer-major and decode position-major, so a single global counter
//! would differ between them; per-position ordinals do not, because
//! BUS-1's T1 already makes each position's sequence identical. The
//! receiver never requires positions to arrive in order, only that each
//! position's own ordinals do.
//!
//! [`SequenceGuard`] fails closed. The first refusal ends the stream, as on
//! the VFF1 WebSocket stream, and unlike VFF1 a gap is refused. BUS-2 has
//! no production transport; BUS-3 adopts the guard (D7). Interventions
//! (decode-only, BUS-1 T9) add transitions the declared count does not
//! include, so an intervened stream is not guarded here.

use std::collections::BTreeMap;
use std::fmt;
use std::ops::Range;

use super::super::ComponentOpPlan;
use super::address::CarrierAddress;
use super::attention_residual::is_block_boundary;
use super::observe::{CarrierForm, CarrierTransition};
use super::prepared::PreparedOperands;

/// Which stream a sequenced transition belongs to: the execution identity
/// digest of the computation that produced it, and a run id the producer
/// chooses. Two runs of the same computation are different streams.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct StreamId {
    pub identity: String,
    pub run: u64,
}

/// One transition as it crosses a boundary: its stream, its address, and
/// its ordinal within its position.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sequenced {
    pub stream: StreamId,
    pub address: CarrierAddress,
    pub ordinal: usize,
    pub transition: CarrierTransition,
}

/// Stamps transitions with their per-position ordinal, in emission order.
/// The same stamping over batch, decode or a chunked prefill yields the
/// same ordinals, because each position's own order is fixed.
#[derive(Debug)]
pub struct Sequencer {
    stream: StreamId,
    next: BTreeMap<usize, usize>,
}

impl Sequencer {
    pub fn new(stream: StreamId) -> Self {
        Self {
            stream,
            next: BTreeMap::new(),
        }
    }

    pub fn stamp(&mut self, address: CarrierAddress, transition: CarrierTransition) -> Sequenced {
        let next = self.next.entry(address.position).or_default();
        let ordinal = *next;
        *next += 1;
        Sequenced {
            stream: self.stream.clone(),
            address,
            ordinal,
            transition,
        }
    }
}

/// How many transitions every position performs on `ops`, derived from
/// the plan and the prepared image, the way BUS-1's F1 derives writes.
///
/// One `Enter`; then, per executed layer, one attention-site transition,
/// one FFN-site transition when the layer runs an FFN, one `Scale` when it
/// declares a layer scalar, and a snapshot and a reset at each
/// attention-residual block boundary. Interventions are not counted.
pub fn declared_transitions(plan: &ComponentOpPlan, ops: &PreparedOperands) -> usize {
    let block_size = ops.attention_residual_block_size();
    let first = ops.first_layer();
    let per_layer: usize = ops
        .layers()
        .iter()
        .enumerate()
        .map(|(offset, prepared)| {
            let index = first + offset;
            let ffn = prepared.ffn.is_some() && plan.layers[index].ffn.is_some();
            let boundary = block_size.is_some_and(|size| is_block_boundary(index, size));
            1 + usize::from(ffn)
                + usize::from(prepared.layer_scale.is_some())
                + if boundary { 2 } else { 0 }
        })
        .sum();
    1 + per_layer
}

/// Why a guard refused. The first refusal ends the stream.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SequenceRefusal {
    StaleIdentity {
        bound: String,
        got: String,
    },
    LayerOutOfRange {
        layer: usize,
        bound: Range<usize>,
    },
    WrongForm {
        bound: CarrierForm,
        got: CarrierForm,
    },
    PositionOutsideDomain {
        position: usize,
        domain: Range<usize>,
    },
    Duplicate {
        position: usize,
        ordinal: usize,
    },
    Gap {
        position: usize,
        expected: usize,
        got: usize,
    },
    BeyondDeclared {
        position: usize,
        ordinal: usize,
        declared: usize,
    },
    MissingPosition {
        position: usize,
    },
    Truncated {
        position: usize,
        received: usize,
        declared: usize,
    },
    StreamEnded,
}

impl fmt::Display for SequenceRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StaleIdentity { bound, got } => write!(
                f,
                "a transition from execution identity {got} arrived on a stream bound to {bound}"
            ),
            Self::LayerOutOfRange { layer, bound } => {
                write!(
                    f,
                    "layer {layer} is outside the bound layer range {bound:?}"
                )
            }
            Self::WrongForm { bound, got } => {
                write!(
                    f,
                    "a {got:?} carrier arrived on a stream bound to {bound:?}"
                )
            }
            Self::PositionOutsideDomain { position, domain } => write!(
                f,
                "position {position} is outside this operation's declared domain {domain:?}"
            ),
            Self::Duplicate { position, ordinal } => {
                write!(f, "ordinal {ordinal} at position {position} arrived twice")
            }
            Self::Gap {
                position,
                expected,
                got,
            } => write!(
                f,
                "position {position} expected ordinal {expected} and got {got}: a transition is \
                 missing or out of order"
            ),
            Self::BeyondDeclared {
                position,
                ordinal,
                declared,
            } => write!(
                f,
                "ordinal {ordinal} at position {position} exceeds the {declared} transitions the \
                 plan declares"
            ),
            Self::MissingPosition { position } => {
                write!(f, "declared position {position} never arrived")
            }
            Self::Truncated {
                position,
                received,
                declared,
            } => write!(
                f,
                "position {position} stopped after {received} of its {declared} declared \
                 transitions"
            ),
            Self::StreamEnded => write!(f, "the stream already ended on an earlier refusal"),
        }
    }
}

impl std::error::Error for SequenceRefusal {}

/// The receiver's check on one operation's stream (S2).
#[derive(Debug)]
pub struct SequenceGuard {
    stream: StreamId,
    layers: Range<usize>,
    form: CarrierForm,
    domain: Range<usize>,
    declared: usize,
    received: BTreeMap<usize, usize>,
    ended: bool,
}

impl SequenceGuard {
    /// Open a guard for one operation over `domain`, bound to `stream`,
    /// the layer range and carrier form this receiver serves, and the
    /// per-position transition count the plan declares.
    pub fn open(
        stream: StreamId,
        layers: Range<usize>,
        form: CarrierForm,
        domain: Range<usize>,
        declared: usize,
    ) -> Self {
        Self {
            stream,
            layers,
            form,
            domain,
            declared,
            received: BTreeMap::new(),
            ended: false,
        }
    }

    /// Accept the next transition, or refuse it and end the stream.
    pub fn accept(&mut self, next: &Sequenced) -> Result<(), SequenceRefusal> {
        if self.ended {
            return Err(SequenceRefusal::StreamEnded);
        }
        let verdict = self.check(next);
        if verdict.is_err() {
            self.ended = true;
        }
        verdict
    }

    fn check(&mut self, next: &Sequenced) -> Result<(), SequenceRefusal> {
        let address = next.address;
        if next.stream.identity != self.stream.identity {
            return Err(SequenceRefusal::StaleIdentity {
                bound: self.stream.identity.clone(),
                got: next.stream.identity.clone(),
            });
        }
        if !self.layers.contains(&address.layer) {
            return Err(SequenceRefusal::LayerOutOfRange {
                layer: address.layer,
                bound: self.layers.clone(),
            });
        }
        if address.form != self.form {
            return Err(SequenceRefusal::WrongForm {
                bound: self.form,
                got: address.form,
            });
        }
        if !self.domain.contains(&address.position) {
            return Err(SequenceRefusal::PositionOutsideDomain {
                position: address.position,
                domain: self.domain.clone(),
            });
        }
        let expected = self.received.entry(address.position).or_default();
        if next.ordinal < *expected {
            return Err(SequenceRefusal::Duplicate {
                position: address.position,
                ordinal: next.ordinal,
            });
        }
        if next.ordinal > *expected {
            return Err(SequenceRefusal::Gap {
                position: address.position,
                expected: *expected,
                got: next.ordinal,
            });
        }
        if next.ordinal >= self.declared {
            return Err(SequenceRefusal::BeyondDeclared {
                position: address.position,
                ordinal: next.ordinal,
                declared: self.declared,
            });
        }
        *expected += 1;
        Ok(())
    }

    /// Close the operation: every declared position must have arrived,
    /// complete.
    pub fn close(self) -> Result<(), SequenceRefusal> {
        if self.ended {
            return Err(SequenceRefusal::StreamEnded);
        }
        for position in self.domain.clone() {
            match self.received.get(&position) {
                None => return Err(SequenceRefusal::MissingPosition { position }),
                Some(&received) if received < self.declared => {
                    return Err(SequenceRefusal::Truncated {
                        position,
                        received,
                        declared: self.declared,
                    })
                }
                Some(_) => {}
            }
        }
        Ok(())
    }
}
