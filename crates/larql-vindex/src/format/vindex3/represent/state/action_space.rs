//! **The vocabulary of moves, declared rather than inferred.**
//!
//! A search over precision maps needs a set of named, reusable edits —
//! `E24`, `K25`, `M26` — and the question of *which* edits exist is not
//! one this module may answer by guessing. R5-F6 is why: neighbourhood 1
//! drew its in-moves from `{H, M23, K24}`, the candidates left
//! unpromoted at iteration 4, and missed `{E20,E22,E23,E24,E25}`
//! entirely. **In an exchange frame every never-admitted action is an
//! in-move, whenever it was last considered.** Two moves worth ~430 MB
//! each were invisible because the vocabulary had been mistaken for the
//! last round's leftovers.
//!
//! So the vocabulary is an input. This module holds it, fixes its order,
//! and turns an applied SET of edits into a map — nothing more.
//!
//! # Why a set, and how it becomes an ordered map
//!
//! [`super::super::map::PrecisionMap`] resolves exceptions in
//! declaration order, first match deciding, so a map is not a set. But a
//! *search state* is: `{E26, M26, K25}` names one map however the round
//! that built it happened to order its edits. The vocabulary's own
//! declaration order supplies the map order, so an applied set has
//! exactly one map and two rounds that reach the same set reach the same
//! bytes — which is what makes 1a's identity contract meaningful over
//! search states rather than only over hand-written maps.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use super::super::map::{Exception, PrecisionMap};
use crate::error::VindexError;

/// One named, reusable change to a precision map: one or more exceptions
/// applied together. A declared search group (MEASURE-PLAN-3's `attn-qkv`
/// is three projections across ten layers) is one edit of several rules.
///
/// Serialised as `exception` when it holds one, which is every edit written
/// before groups could hold several, so those records keep their bytes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "EditForm", into = "EditForm")]
pub struct MapEdit {
    /// The programme's own name for it — `"E24"`, `"K25"`.
    pub name: String,
    /// What it puts into the map, in order. Never empty.
    exceptions: Vec<Exception>,
}

impl MapEdit {
    pub fn new(name: impl Into<String>, exception: Exception) -> Self {
        Self {
            name: name.into(),
            exceptions: vec![exception],
        }
    }

    /// An edit of several exceptions, refused when it has none: an edit
    /// that changes nothing names a state identical to not applying it.
    pub fn group(name: impl Into<String>, exceptions: Vec<Exception>) -> Result<Self, VindexError> {
        let name = name.into();
        if exceptions.is_empty() {
            return Err(VindexError::Parse(format!(
                "action `{name}` declares no exception, so applying it changes nothing"
            )));
        }
        Ok(Self { name, exceptions })
    }

    /// What it puts into the map, in order.
    pub fn exceptions(&self) -> &[Exception] {
        &self.exceptions
    }
}

/// `MapEdit`'s stored form: `exception` for one, `exceptions` for several.
#[derive(Serialize, Deserialize)]
struct EditForm {
    name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    exception: Option<Exception>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    exceptions: Vec<Exception>,
}

impl TryFrom<EditForm> for MapEdit {
    type Error = String;

    fn try_from(form: EditForm) -> Result<Self, String> {
        match (form.exception, form.exceptions.is_empty()) {
            (Some(one), true) => Ok(MapEdit::new(form.name, one)),
            (None, false) => MapEdit::group(form.name, form.exceptions).map_err(|e| e.to_string()),
            (Some(_), false) => Err(format!(
                "action `{}` declares both `exception` and `exceptions`",
                form.name
            )),
            (None, true) => Err(format!("action `{}` declares no exception", form.name)),
        }
    }
}

impl From<MapEdit> for EditForm {
    fn from(edit: MapEdit) -> Self {
        let MapEdit {
            name,
            mut exceptions,
        } = edit;
        if exceptions.len() == 1 {
            Self {
                name,
                exception: exceptions.pop(),
                exceptions: Vec::new(),
            }
        } else {
            Self {
                name,
                exception: None,
                exceptions,
            }
        }
    }
}

/// **Every move the search may make**, in the order that fixes map
/// order.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ActionVocabulary {
    edits: Vec<MapEdit>,
}

impl ActionVocabulary {
    /// Build a vocabulary, refusing a repeated name.
    ///
    /// Two edits under one name would make an applied set ambiguous and
    /// the map it produces dependent on which was found first.
    pub fn new(edits: impl IntoIterator<Item = MapEdit>) -> Result<Self, VindexError> {
        let edits: Vec<MapEdit> = edits.into_iter().collect();
        let mut seen = BTreeSet::new();
        for edit in &edits {
            if !seen.insert(edit.name.as_str()) {
                return Err(VindexError::Parse(format!(
                    "action `{}` is declared twice — an applied set naming it would resolve to \
                     whichever declaration was found first",
                    edit.name
                )));
            }
        }
        Ok(Self { edits })
    }

    pub fn edits(&self) -> &[MapEdit] {
        &self.edits
    }

    pub fn len(&self) -> usize {
        self.edits.len()
    }

    pub fn is_empty(&self) -> bool {
        self.edits.is_empty()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.edits.iter().any(|e| e.name == name)
    }

    /// Every name, in declaration order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.edits.iter().map(|e| e.name.as_str())
    }

    /// **The map an applied set produces.**
    ///
    /// Exceptions are emitted in vocabulary order and prepended to the
    /// base map's own, so an applied edit always outranks a default the
    /// base map declared. Unknown names are refused rather than skipped:
    /// silently dropping one would hand back a map for a state that does
    /// not exist.
    pub fn map_for(
        &self,
        base: &PrecisionMap,
        applied: &BTreeSet<String>,
    ) -> Result<PrecisionMap, VindexError> {
        if let Some(unknown) = applied.iter().find(|n| !self.contains(n)) {
            return Err(VindexError::Parse(format!(
                "action `{unknown}` is not in this vocabulary — a state cannot apply a move the \
                 search does not have"
            )));
        }
        let mut exceptions: Vec<Exception> = self
            .edits
            .iter()
            .filter(|e| applied.contains(&e.name))
            .flat_map(|e| e.exceptions.iter().cloned())
            .collect();
        exceptions.extend(base.exceptions.iter().cloned());
        Ok(PrecisionMap {
            name: base.name.clone(),
            encoding: base.encoding.clone(),
            roles: base.roles.clone(),
            exceptions,
        })
    }
}

#[cfg(test)]
mod tests;
