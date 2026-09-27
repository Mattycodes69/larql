//! The two setup-time code diagnostics that are not per-prompt probes:
//! `--address-code-stability` (train vs eval code distributions) and
//! `--address-code-occurrences` (a per-position export).

use crate::commands::dev::ov_rd::oracle_pq_address::collect_code_occurrences;
use crate::commands::dev::ov_rd::oracle_pq_stability::measure_code_stability;
use crate::commands::dev::ov_rd::reports::{CodeStabilityReport, OraclePqReport};
use crate::commands::dev::ov_rd::types::PromptRecord;

use super::args::OraclePqArgs;
use super::probe::{FitContext, HeadConfigMap, ProbeResult, ProbeTargets};
use super::validate::{check_codes, check_groups, check_groups_all, parse_sorted, when};

/// File the occurrence export writes under `--out`.
const OCCURRENCES_FILE: &str = "code_occurrences.json";
const SPLIT_TRAIN: &str = "train";
const SPLIT_EVAL: &str = "eval";
const SPLIT_ALL: &str = "all";

pub(super) struct CodeStability {
    enabled: bool,
    groups: Vec<usize>,
}

impl CodeStability {
    pub(super) fn parse(args: &OraclePqArgs, targets: &ProbeTargets<'_>) -> ProbeResult<Self> {
        let groups = parse_sorted(&args.address_code_stability_groups)?;
        if args.address_code_stability {
            if groups.is_empty() {
                return Err(
                    "--address-code-stability requires at least one --address-code-stability-groups value"
                        .into(),
                );
            }
            check_groups_all("--address-code-stability-groups", &groups, targets.configs)?;
        }
        Ok(Self {
            enabled: args.address_code_stability,
            groups,
        })
    }

    pub(super) fn measure(
        &self,
        ctx: &mut FitContext<'_>,
        eval_prompts: &[PromptRecord],
    ) -> ProbeResult<HeadConfigMap<Vec<CodeStabilityReport>>> {
        if !self.enabled {
            return Ok(HeadConfigMap::new());
        }
        eprintln!("Measuring PQ code stability for groups {:?}", self.groups);
        measure_code_stability(
            ctx.weights,
            ctx.index,
            ctx.tokenizer,
            ctx.fit_prompts,
            eval_prompts,
            ctx.heads,
            ctx.bases,
            ctx.means,
            ctx.pca_bases,
            ctx.codebooks,
            &self.groups,
        )
    }

    pub(super) fn write_report(&self, report: &mut OraclePqReport) {
        report.address_code_stability = self.enabled;
        report.address_code_stability_groups = when(self.enabled, self.groups.clone());
    }
}

pub(super) struct CodeOccurrences {
    enabled: bool,
    groups: Vec<usize>,
    codes: Vec<usize>,
    split: String,
}

impl CodeOccurrences {
    pub(super) fn parse(args: &OraclePqArgs, targets: &ProbeTargets<'_>) -> ProbeResult<Self> {
        let groups = parse_sorted(&args.address_code_occurrence_groups)?;
        let codes = parse_sorted(&args.address_code_occurrence_codes)?;
        let split = args
            .address_code_occurrence_split
            .trim()
            .to_ascii_lowercase();
        if args.address_code_occurrences {
            if groups.is_empty() {
                return Err(
                    "--address-code-occurrences requires at least one --address-code-occurrence-groups value"
                        .into(),
                );
            }
            if ![SPLIT_TRAIN, SPLIT_EVAL, SPLIT_ALL].contains(&split.as_str()) {
                return Err("--address-code-occurrence-split must be train, eval, or all".into());
            }
            for config in targets.configs {
                check_groups("--address-code-occurrence-groups", &groups, config)?;
                check_codes("--address-code-occurrence-codes", &codes, config)?;
            }
        }
        Ok(Self {
            enabled: args.address_code_occurrences,
            groups,
            codes,
            split,
        })
    }

    /// Export the occurrences of the selected split to `out`.
    pub(super) fn export(
        &self,
        ctx: &mut FitContext<'_>,
        splits: (&[PromptRecord], &[PromptRecord], &[PromptRecord]),
        out: &std::path::Path,
    ) -> ProbeResult {
        if !self.enabled {
            return Ok(());
        }
        let (fit_prompts, eval_prompts, all_prompts) = splits;
        let occurrence_prompts = match self.split.as_str() {
            SPLIT_TRAIN => fit_prompts.to_vec(),
            SPLIT_EVAL => eval_prompts.to_vec(),
            SPLIT_ALL => all_prompts.to_vec(),
            _ => unreachable!("validated code occurrence split"),
        };
        eprintln!(
            "Exporting code occurrences for groups {:?}, codes {:?}, split {}",
            self.groups, self.codes, self.split
        );
        let occurrences = collect_code_occurrences(
            ctx.weights,
            ctx.index,
            ctx.tokenizer,
            &occurrence_prompts,
            ctx.heads,
            ctx.bases,
            ctx.means,
            ctx.pca_bases,
            ctx.codebooks,
            &self.groups,
            &self.codes,
        )?;
        let occurrence_path = out.join(OCCURRENCES_FILE);
        let file = std::fs::File::create(&occurrence_path)?;
        serde_json::to_writer_pretty(file, &occurrences)?;
        eprintln!("Wrote {}", occurrence_path.display());
        Ok(())
    }
}
