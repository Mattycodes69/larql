//! The address-probe contract and the contexts the orchestrator hands a
//! probe: [`FitContext`] once during setup, [`EvalContext`] once per
//! `(eval prompt, head, config)`.

use std::collections::HashMap;
use std::fmt::Debug;

use larql_inference::ModelWeights;
use larql_vindex::VectorIndex;

use crate::commands::dev::ov_rd::basis::{WoRoundtripBasis, ZPcaBasis};
use crate::commands::dev::ov_rd::oracle_pq_eval::evaluate_predicted_address;
use crate::commands::dev::ov_rd::oracle_pq_reports::OraclePqPointAccumulator;
use crate::commands::dev::ov_rd::pq::{ModeDTable, PqCodebook};
use crate::commands::dev::ov_rd::reports::{AddressProbePromptReport, OraclePqReport};
use crate::commands::dev::ov_rd::stats::StaticHeadMeans;
use crate::commands::dev::ov_rd::types::{HeadId, PqConfig, PromptRecord};

pub(super) type ProbeResult<T = ()> = Result<T, Box<dyn std::error::Error>>;
pub(super) type HeadConfigMap<T> = HashMap<(HeadId, PqConfig), T>;

/// Label for a PQ group the probe leaves at its oracle code.
pub(super) const ORACLE_GROUP_KEY: &str = "oracle";

/// One family of address probes (`--address-*-probe`). Every family is
/// registered whether or not its flag is set, because the report records
/// each family's settings either way; only enabled families fit and run.
pub(super) trait AddressProbe {
    fn enabled(&self) -> bool;

    /// Fit any per-`(head, config)` models from the training split. Families
    /// with a fit also own their `--mode-d-check` prerequisite, checked here
    /// so the error surfaces before the (slow) fit, as it always has.
    fn fit(&mut self, _ctx: &mut FitContext<'_>) -> ProbeResult {
        Ok(())
    }

    /// Whether this family reads the train-split per-group majority codes.
    fn needs_majority_codes(&self) -> bool {
        self.enabled()
    }

    /// Evaluate one `(prompt, head, config)` and record the reports.
    fn evaluate(&self, ctx: &mut EvalContext<'_>) -> ProbeResult;

    /// Write this family's settings into the run report.
    fn write_report(&self, report: &mut OraclePqReport);
}

/// What the probe settings are validated against.
pub(super) struct ProbeTargets<'a> {
    pub(super) configs: &'a [PqConfig],
    pub(super) heads: &'a [HeadId],
    pub(super) num_layers: usize,
    pub(super) hidden_size: usize,
}

pub(super) struct FitContext<'a> {
    pub(super) weights: &'a mut ModelWeights,
    pub(super) index: &'a VectorIndex,
    pub(super) tokenizer: &'a tokenizers::Tokenizer,
    pub(super) fit_prompts: &'a [PromptRecord],
    pub(super) heads: &'a [HeadId],
    pub(super) bases: &'a HashMap<HeadId, WoRoundtripBasis>,
    pub(super) means: &'a HashMap<HeadId, StaticHeadMeans>,
    pub(super) pca_bases: &'a HashMap<HeadId, ZPcaBasis>,
    pub(super) codebooks: &'a HeadConfigMap<PqCodebook>,
    pub(super) mode_d_check: bool,
}

impl FitContext<'_> {
    pub(super) fn require_mode_d(&self, message: &str) -> ProbeResult {
        if self.mode_d_check {
            Ok(())
        } else {
            Err(message.into())
        }
    }
}

/// The oracle-address Mode D outcome for the current prompt, reported for a
/// substitution whose source code never occurs (nothing to substitute).
#[derive(Clone, Copy)]
pub(super) struct OracleModeD {
    pub(super) kl: f64,
    pub(super) top1_agree: bool,
    pub(super) baseline_top1_in_top5: bool,
}

pub(super) struct EvalContext<'a> {
    pub(super) weights: &'a mut ModelWeights,
    pub(super) index: &'a VectorIndex,
    pub(super) token_ids: &'a [u32],
    pub(super) stratum: &'a str,
    pub(super) label: &'a str,
    pub(super) head: HeadId,
    pub(super) config: PqConfig,
    pub(super) baseline_logp: &'a [f64],
    pub(super) baseline_top1: u32,
    pub(super) oracle_codes_by_position: &'a [Vec<usize>],
    pub(super) oracle_mode_d: OracleModeD,
    pub(super) mode_d_tables: &'a HeadConfigMap<ModeDTable>,
    pub(super) majority_codes: &'a HeadConfigMap<Vec<usize>>,
    pub(super) accumulator: &'a mut OraclePqPointAccumulator,
}

impl<'a> EvalContext<'a> {
    pub(super) fn mode_d_table(&self, probe: &str) -> ProbeResult<&'a ModeDTable> {
        let what = format!("Mode D table for {probe}");
        lookup(self.mode_d_tables, self.head, self.config, &what)
    }

    pub(super) fn majority(&self, probe: &str) -> ProbeResult<&'a Vec<usize>> {
        let what = format!("majority codes for {probe}");
        lookup(self.majority_codes, self.head, self.config, &what)
    }

    /// This `(head, config)`'s entry in a fitted-model map; `what` names the
    /// model in the error.
    pub(super) fn model<'m, T>(
        &self,
        models: &'m HeadConfigMap<T>,
        what: &str,
    ) -> ProbeResult<&'m T> {
        lookup(models, self.head, self.config, &format!("{what} for"))
    }

    pub(super) fn evaluate(
        &mut self,
        mode_d_table: &ModeDTable,
        predicted_codes_by_position: &[Vec<usize>],
    ) -> ProbeResult<AddressProbePromptReport> {
        evaluate_predicted_address(
            self.weights,
            self.token_ids,
            self.index,
            self.head,
            mode_d_table,
            predicted_codes_by_position,
            self.stratum,
            self.label,
            self.baseline_logp,
            self.baseline_top1,
            self.oracle_codes_by_position,
        )
    }

    /// Evaluate a predicted address and record it under `name`.
    pub(super) fn evaluate_probe(
        &mut self,
        mode_d_table: &ModeDTable,
        predicted_codes_by_position: &[Vec<usize>],
        name: &str,
        selected_group_keys: &[String],
    ) -> ProbeResult {
        let report = self.evaluate(mode_d_table, predicted_codes_by_position)?;
        self.accumulator
            .add_address_probe(name, selected_group_keys, report);
        Ok(())
    }

    /// One label per PQ group: `selected(group)` for a group the probe
    /// replaces, [`ORACLE_GROUP_KEY`] for the rest.
    pub(super) fn group_keys(&self, selected: impl Fn(usize) -> Option<String>) -> Vec<String> {
        (0..self.config.groups)
            .map(|group| selected(group).unwrap_or_else(|| ORACLE_GROUP_KEY.to_string()))
            .collect()
    }
}

/// `L{layer} H{head}`, as the missing-entry errors name a head.
pub(super) fn head_label(head: HeadId) -> String {
    format!("L{} H{}", head.layer, head.head)
}

/// A per-head entry, or `missing {what} L{layer} H{head}`.
pub(super) fn lookup_head<'m, T>(
    map: &'m HashMap<HeadId, T>,
    head: HeadId,
    what: &str,
) -> ProbeResult<&'m T> {
    map.get(&head)
        .ok_or_else(|| format!("missing {what} {}", head_label(head)).into())
}

/// A per-`(head, config)` entry, or `missing {what} L{layer} H{head} {config:?}`.
pub(super) fn lookup<'m, T>(
    map: &'m HeadConfigMap<T>,
    head: HeadId,
    config: PqConfig,
    what: &str,
) -> ProbeResult<&'m T> {
    map.get(&(head, config))
        .ok_or_else(|| format!("missing {what} {} {config:?}", head_label(head)).into())
}

/// The two variants every predicted-group probe reports: the unselected
/// groups held at their oracle codes (`true`), and at the train majority
/// (`false`).
pub(super) fn rest_variants(name: &str, groups: &(impl Debug + ?Sized)) -> [(String, bool); 2] {
    [
        (format!("{name}_groups_{groups:?}_oracle_rest"), true),
        (format!("{name}_groups_{groups:?}_majority_rest"), false),
    ]
}

/// The codes the unselected groups start from for one variant.
pub(super) fn rest_codes<'c>(
    use_oracle_rest: bool,
    oracle_codes: &'c [usize],
    majority: &'c [usize],
) -> &'c [usize] {
    if use_oracle_rest {
        oracle_codes
    } else {
        majority
    }
}

/// A slice per position of a per-position table, empty past its end.
pub(super) fn row_at<T>(rows: &[Vec<T>], pos: usize) -> &[T] {
    rows.get(pos).map(Vec::as_slice).unwrap_or(&[])
}
