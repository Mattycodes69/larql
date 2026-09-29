//! **Which variables a proposal problem has.** Either 1a's original grouping,
//! one variable per (projection, layer), or a declared vocabulary: one
//! variable per edit, each a named set of (projection, layer range) rules
//! (MEASURE-PLAN-3's `attn-qkv` × quarter).
//!
//! A declared vocabulary must partition the eligible tensors: a tensor two
//! edits both match would be protected by either, so the state a proposal
//! names would depend on which the solver chose; a tensor no edit matches
//! could never be protected, a search space narrowed without saying so.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::super::action_space::ActionVocabulary;
use crate::error::VindexError;

/// One variable. `Tensor` serialises exactly as 1a's original key did, so
/// every record written before declared vocabularies keeps its bytes.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(untagged)]
pub enum GroupKey {
    /// One projection at one layer.
    Tensor { projection: String, layer: u32 },
    /// One edit of a declared vocabulary, by its name.
    Declared { name: String },
}

impl GroupKey {
    pub fn new(projection: impl Into<String>, layer: u32) -> Self {
        Self::Tensor {
            projection: projection.into(),
            layer,
        }
    }

    pub fn declared(name: impl Into<String>) -> Self {
        Self::Declared { name: name.into() }
    }

    /// The (projection, layer) of a `Tensor` key.
    pub fn tensor(&self) -> Option<(&str, u32)> {
        match self {
            Self::Tensor { projection, layer } => Some((projection, *layer)),
            Self::Declared { .. } => None,
        }
    }

    pub fn describe(&self) -> String {
        match self {
            Self::Tensor { projection, layer } => format!("{projection}@{layer}"),
            Self::Declared { name } => name.clone(),
        }
    }
}

/// One source-precision rule: a projection over an inclusive layer range.
pub(super) type Rule = (String, u32, u32);

/// How a problem's tensors become variables.
#[derive(Debug, Clone, Copy)]
pub enum Grouping<'a> {
    /// One variable per (projection, layer).
    PerProjectionLayer,
    /// One variable per edit of this vocabulary.
    Declared(&'a ActionVocabulary),
}

fn refused(message: impl Into<String>) -> VindexError {
    VindexError::Parse(format!("AUTO-REP group vocabulary: {}", message.into()))
}

/// A declared vocabulary's edits as rules, refused unless every exception
/// holds a projection at source over a stated layer range: the only move
/// 1a's two-valued domain (compile or source) can make.
pub(super) fn declared_rules(
    vocabulary: &ActionVocabulary,
) -> Result<BTreeMap<GroupKey, Vec<Rule>>, VindexError> {
    let mut groups = BTreeMap::new();
    for edit in vocabulary.edits() {
        let mut rules = Vec::with_capacity(edit.exceptions().len());
        for exception in edit.exceptions() {
            match (&exception.projection, exception.layers, &exception.encoding) {
                (Some(projection), Some((lo, hi)), None) if lo <= hi => {
                    rules.push((projection.clone(), lo, hi))
                }
                _ => {
                    return Err(refused(format!(
                        "`{}` holds {exception:?}; a group rule names one projection over an \
                         ordered layer range and holds it at source",
                        edit.name
                    )))
                }
            }
        }
        groups.insert(GroupKey::declared(&edit.name), rules);
    }
    Ok(groups)
}

impl Grouping<'_> {
    /// The variable an eligible (projection, layer) tensor belongs to.
    pub(super) fn assign(
        &self,
        declared: &BTreeMap<GroupKey, Vec<Rule>>,
        projection: &str,
        layer: u32,
    ) -> Result<GroupKey, VindexError> {
        if let Grouping::PerProjectionLayer = self {
            return Ok(GroupKey::new(projection, layer));
        }
        let mut matches = declared.iter().filter(|(_, rules)| {
            rules
                .iter()
                .any(|(p, lo, hi)| p == projection && (*lo..=*hi).contains(&layer))
        });
        match (matches.next(), matches.next()) {
            (None, _) => Err(refused(format!(
                "no group holds `{projection}` at layer {layer}; the vocabulary must cover every \
                 eligible tensor"
            ))),
            (Some((key, _)), None) => Ok(key.clone()),
            (Some((a, _)), Some((b, _))) => Err(refused(format!(
                "`{projection}` at layer {layer} is in both `{}` and `{}`; groups must not overlap",
                a.describe(),
                b.describe()
            ))),
        }
    }
}

/// The source-precision rules a set of protected groups amounts to, merged
/// into maximal runs per projection. For `Tensor` keys this is exactly 1a's
/// original run-merging.
pub(super) fn protected_rules(
    protected: &[GroupKey],
    declared: &BTreeMap<GroupKey, Vec<Rule>>,
) -> Vec<Rule> {
    let mut rules: Vec<Rule> = protected
        .iter()
        .flat_map(|g| match g {
            GroupKey::Tensor { projection, layer } => vec![(projection.clone(), *layer, *layer)],
            GroupKey::Declared { .. } => declared.get(g).cloned().unwrap_or_default(),
        })
        .collect();
    rules.sort();
    let mut merged: Vec<Rule> = Vec::with_capacity(rules.len());
    for (projection, lo, hi) in rules {
        match merged.last_mut() {
            Some((p, _, last_hi)) if *p == projection && lo <= last_hi.saturating_add(1) => {
                *last_hi = (*last_hi).max(hi);
            }
            _ => merged.push((projection, lo, hi)),
        }
    }
    merged
}

/// **Whether `vocabulary` is a declared group vocabulary of this surface**:
/// source-precision rules, every eligible (projection, layer) tensor in
/// exactly one group, and no group that matches nothing. The same checks a
/// proposal problem makes, available before any price table exists, so a
/// record is refused when it is produced rather than at its first search.
pub fn check_declared(
    surface: &super::super::surface::TensorSurface,
    base: &super::super::super::map::PrecisionMap,
    vocabulary: &ActionVocabulary,
) -> Result<(), VindexError> {
    use super::super::super::policy::{layer_of, projection_of};
    let declared = declared_rules(vocabulary)?;
    let grouping = Grouping::Declared(vocabulary);
    let mut hit = std::collections::BTreeSet::new();
    for t in surface.entries() {
        if !base.roles.iter().any(|r| r == t.role.name()) {
            continue;
        }
        if let (Some(projection), Some(layer)) = (projection_of(&t.tensor), layer_of(&t.tensor)) {
            hit.insert(grouping.assign(&declared, projection, layer)?);
        }
    }
    match declared.keys().find(|k| !hit.contains(*k)) {
        Some(dead) => Err(refused(format!(
            "declared group `{}` matches no eligible tensor; a move that changes nothing is a \
             dead rule",
            dead.describe()
        ))),
        None => Ok(()),
    }
}
