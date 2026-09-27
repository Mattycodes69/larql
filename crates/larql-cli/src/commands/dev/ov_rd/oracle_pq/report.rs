//! Assemble and write `oracle_pq.json`.

use std::collections::HashMap;

use crate::commands::dev::ov_rd::oracle_pq_reports::OraclePqPointAccumulator;
use crate::commands::dev::ov_rd::reports::{
    CodeStabilityReport, OraclePqHeadReport, OraclePqReport,
};
use crate::commands::dev::ov_rd::types::{HeadId, PqConfig};

use super::args::OraclePqArgs;
use super::probe::{lookup_head, HeadConfigMap, ProbeResult};
use super::registry::ProbeRegistry;
use super::setup::{BaseFit, PromptSplit};

/// File the run report is written to under `--out`.
const REPORT_FILE: &str = "oracle_pq.json";
/// The static base every head's residual is measured against.
const STATIC_BASE: &str = "position_mean";

pub(super) struct ReportInputs<'a> {
    pub(super) args: &'a OraclePqArgs,
    pub(super) heads: Vec<HeadId>,
    pub(super) configs: Vec<PqConfig>,
    pub(super) hidden_size: usize,
    pub(super) prompts: &'a PromptSplit,
    pub(super) base: &'a BaseFit,
    pub(super) registry: &'a ProbeRegistry,
    pub(super) code_stability: &'a HeadConfigMap<Vec<CodeStabilityReport>>,
}

fn head_reports(
    inputs: &ReportInputs<'_>,
    mut accumulators: HashMap<(HeadId, PqConfig), OraclePqPointAccumulator>,
) -> ProbeResult<Vec<OraclePqHeadReport>> {
    let mut reports = Vec::new();
    for head in &inputs.heads {
        let basis = lookup_head(&inputs.base.bases, *head, "basis for")?;
        let pca_basis = lookup_head(&inputs.base.pca_bases, *head, "PCA basis for")?;
        let static_train_samples = inputs.base.means.get(head).map(|m| m.count).unwrap_or(0);
        let mut points = Vec::new();
        for &config in &inputs.configs {
            let acc = accumulators
                .remove(&(*head, config))
                .expect("oracle PQ accumulator missing at finish");
            let stability = inputs
                .code_stability
                .get(&(*head, config))
                .cloned()
                .unwrap_or_default();
            points.push(acc.finish(config, inputs.hidden_size, stability));
        }
        reports.push(OraclePqHeadReport {
            layer: head.layer,
            head: head.head,
            head_dim: basis.head_dim,
            rank_retained: basis.rank_retained(),
            empirical_rank: pca_basis.rank(),
            sigma_max: basis.sigma_max,
            sigma_min_retained: basis.sigma_min_retained,
            static_train_samples,
            points,
        });
    }
    Ok(reports)
}

pub(super) fn write_report(
    inputs: ReportInputs<'_>,
    accumulators: HashMap<(HeadId, PqConfig), OraclePqPointAccumulator>,
) -> ProbeResult {
    let heads = head_reports(&inputs, accumulators)?;
    let args = inputs.args;
    let mut report = OraclePqReport {
        index: args.index.display().to_string(),
        prompt_file: args.prompts.display().to_string(),
        prompts_seen: inputs.prompts.all.len(),
        train_prompts_seen: inputs.prompts.fit.len(),
        eval_prompts_seen: inputs.prompts.eval.len(),
        max_per_stratum: args.max_per_stratum,
        eval_mod: args.eval_mod,
        eval_offset: args.eval_offset,
        static_base: STATIC_BASE.to_string(),
        configs: inputs.configs,
        sigma_rel_cutoff: args.sigma_rel_cutoff,
        pq_iters: args.pq_iters,
        mode_d_check: args.mode_d_check,
        selected_heads: inputs.heads,
        heads,
        ..OraclePqReport::default()
    };
    inputs.registry.write_report(&mut report);

    let out_path = args.out.join(REPORT_FILE);
    let file = std::fs::File::create(&out_path)?;
    serde_json::to_writer_pretty(file, &report)?;
    eprintln!("Wrote {}", out_path.display());
    Ok(())
}
