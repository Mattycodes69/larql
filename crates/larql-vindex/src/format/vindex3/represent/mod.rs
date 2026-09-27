//! Representation compilation: add a compiled physical encoding of an
//! object to a container, without changing what the model *is*.
//!
//! ```text
//! source tensors → representation compiler → persisted representation pack
//!                                                      ↓
//!                                          a profile selects the pack
//! ```
//!
//! This is a third verb alongside the two that already exist, and the
//! distinction is the point:
//!
//! - **COMPILE** materialises overlay *meaning* into rewritten segments.
//! - **COMPACT** reorganises bytes while preserving meaning exactly
//!   (`SemanticDiff(input, output) == ∅`).
//! - **REPRESENT** adds a *lossy alternative encoding* beside the
//!   canonical bytes. It preserves neither byte-equality (the pack is new
//!   bytes) nor exact semantics (4-bit is an approximation), so it cannot
//!   hide behind either gate. Nearest packs must equal the transient nearest
//!   quantizer byte-for-byte. Calibrated recipes instead bind completed bytes
//!   to their derivation; consuming those bytes requires only the codec ABI.
//!
//! The default path still loads through [`OperandSource::load`] and calls the
//! transient nearest quantizer. [`compile_representation_recipe`] explicitly
//! selects nearest or calibrated GPTQ without changing that default.
//!
//! ## What it is for
//!
//! A 30B BF16 source is tens of gigabytes on disk and pays the
//! quantisation cost on every cold load. Neither is a property of the
//! model; both are properties of having only one stored representation.
//! Compiling one changes the artifact, not the semantics — and at K3
//! scale, where sparse expert fetches make bytes-per-expert an input to
//! the inference algorithm rather than a storage detail, it stops being a
//! convenience.
//!
//! ## What it does not do
//!
//! It does not replace the canonical representation. The source bytes stay
//! in the container and stay canonical; the compiled pack is added beside
//! them with [`Fidelity::Approximate`]. A profile then selects between
//! representations that *exist* — the rule
//! [`super::variants`] already states, and the reason a compiler is needed
//! at all: a profile cannot turn one encoding's bytes into another's.

pub mod activation;
pub mod actuate;
pub mod arena;
pub mod assessment;
pub mod auto_rep;
pub mod bank;
pub mod byte_ledger;
pub mod calibration;
pub mod candidate_authority;
pub mod codec;
pub mod compile;
pub mod compiler;
pub mod constraint;
pub mod decision;
pub mod derivation;
pub mod diagnostic;
pub mod execution_cost;
pub mod experiment;
pub mod experiment_identity;
pub mod gptq;
pub mod ingest;
pub mod kda_candidate;
pub mod kquant;
pub mod map;
pub mod map_check;
pub mod measure;
#[cfg(test)]
mod measure_tests;
pub mod measurement;
pub mod nvfp4_pack;
pub mod observation_stream;
pub mod participation;
pub mod physical;
#[cfg(test)]
mod plan_loop_tests;
pub mod plan_roles;
#[cfg(test)]
mod plan_roles_tests;
pub mod policy;
pub mod produce;
pub mod promotion;
pub mod quality;
pub mod reading;
pub mod recipe;
pub use recipe::{compile_representation_recipe, Nvfp4Recipe};
#[cfg(feature = "reference-encoder")]
pub mod reference_encoder;
pub mod resampling;
pub mod search_evidence;
pub mod selection;
pub mod source_bank;
pub mod source_identity;
pub mod state;
pub mod statistic;
pub mod stream_replay;
pub mod token_bank;
pub mod view;

#[cfg(test)]
use super::opplan::exec::weights::LoadedWeight;
use codec::EncoderRegistry;
use map::PrecisionMap;

// Names the test modules reach through `use super::*`.
#[cfg(test)]
use super::encode::segment::read_segment_header;
#[cfg(test)]
use super::graph::object::Fidelity;
#[cfg(test)]
use super::index::{RepresentationEntry, Vindex3Index};
#[cfg(test)]
use super::inspect::inspect_container;
#[cfg(test)]
use super::opplan::exec::operands::OperandStore;
#[cfg(test)]
use super::opplan::OperandRef;
#[cfg(test)]
use crate::error::VindexError;
#[cfg(test)]
use crate::format::filenames::INDEX_JSON;
#[cfg(test)]
use codec::{CodecError, RepresentationEncoder};
#[cfg(test)]
use nvfp4_pack::{CodecIdentity, EncoderRecipe, PackLayout, DTYPE_NVFP4};
#[cfg(test)]
use std::collections::{BTreeMap, BTreeSet};
#[cfg(test)]
use std::io::{Read, Seek, SeekFrom};

mod pipeline;
mod report_types;
use pipeline::*;
pub use report_types::*;

#[cfg(test)]
#[path = "compile_real_tests.rs"]
mod compile_real_tests;

#[cfg(test)]
#[path = "compat_tests.rs"]
mod compat_tests;

#[cfg(test)]
#[path = "pareto_tests.rs"]
mod pareto_tests;

#[cfg(test)]
#[path = "frontier_scale_tests.rs"]
mod frontier_scale_tests;

#[cfg(test)]
#[path = "frontier_explore_tests.rs"]
mod frontier_explore_tests;

#[cfg(test)]
#[path = "frontier_spend_tests.rs"]
mod frontier_spend_tests;

#[cfg(test)]
#[path = "depth_invariance_tests.rs"]
mod depth_invariance_tests;

#[cfg(test)]
#[path = "terminal_tests.rs"]
mod terminal_tests;

#[cfg(test)]
#[path = "tests/mod.rs"]
mod tests;
