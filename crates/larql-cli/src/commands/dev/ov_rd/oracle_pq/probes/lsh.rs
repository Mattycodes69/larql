//! `--address-lsh-group-probe`: selected groups predicted from fixed
//! random-hyperplane hashes of the residual entering the target layer.

use crate::commands::dev::ov_rd::address::AddressLshGroupModel;
use crate::commands::dev::ov_rd::oracle_pq_address::fit_address_lsh_group_models;
use crate::commands::dev::ov_rd::oracle_pq_forward::capture_layer_input_hidden;
use crate::commands::dev::ov_rd::reports::OraclePqReport;

use super::super::args::OraclePqArgs;
use super::super::probe::{
    rest_codes, rest_variants, AddressProbe, EvalContext, FitContext, HeadConfigMap, ProbeResult,
    ProbeTargets,
};
use super::super::validate::{check_groups_all, parse_sorted, when};

/// Hash width cap that keeps the bucket table bounded for a diagnostic.
const MAX_LSH_BITS: usize = 16;

pub(in super::super) struct LshProbe {
    enabled: bool,
    groups: Vec<usize>,
    bits: usize,
    seeds: usize,
    models: HeadConfigMap<AddressLshGroupModel>,
}

impl LshProbe {
    pub(in super::super) fn parse(
        args: &OraclePqArgs,
        targets: &ProbeTargets<'_>,
    ) -> ProbeResult<Self> {
        let groups = parse_sorted(&args.address_lsh_groups)?;
        if args.address_lsh_group_probe {
            if groups.is_empty() {
                return Err(
                    "--address-lsh-group-probe requires at least one --address-lsh-groups value"
                        .into(),
                );
            }
            if args.address_lsh_bits == 0 {
                return Err("--address-lsh-bits must be greater than zero".into());
            }
            if args.address_lsh_bits > MAX_LSH_BITS {
                return Err("--address-lsh-bits is capped at 16 for bounded diagnostics".into());
            }
            if args.address_lsh_seeds == 0 {
                return Err("--address-lsh-seeds must be greater than zero".into());
            }
            check_groups_all("--address-lsh-groups", &groups, targets.configs)?;
        }
        Ok(Self {
            enabled: args.address_lsh_group_probe,
            groups,
            bits: args.address_lsh_bits,
            seeds: args.address_lsh_seeds,
            models: HeadConfigMap::new(),
        })
    }
}

impl AddressProbe for LshProbe {
    fn enabled(&self) -> bool {
        self.enabled
    }

    fn fit(&mut self, ctx: &mut FitContext<'_>) -> ProbeResult {
        if !self.enabled {
            return Ok(());
        }
        ctx.require_mode_d("--address-lsh-group-probe requires --mode-d-check")?;
        eprintln!(
            "Fitting LSH group address probes for groups {:?} (bits={}, seeds={})",
            self.groups, self.bits, self.seeds
        );
        self.models = fit_address_lsh_group_models(
            ctx.weights,
            ctx.index,
            ctx.tokenizer,
            ctx.fit_prompts,
            ctx.heads,
            ctx.bases,
            ctx.means,
            ctx.pca_bases,
            ctx.codebooks,
            &self.groups,
            self.bits,
            self.seeds,
        )?;
        Ok(())
    }

    fn evaluate(&self, ctx: &mut EvalContext<'_>) -> ProbeResult {
        let mode_d_table = ctx.mode_d_table("LSH group probe")?;
        let lsh_model = ctx.model(&self.models, "LSH group probe model")?;
        let group_majority = ctx.majority("LSH group probe")?;
        let layer_input =
            capture_layer_input_hidden(ctx.weights, ctx.token_ids, ctx.index, ctx.head.layer)?;
        let selected_group_keys = lsh_model.selected_group_keys();
        for (probe_name, use_oracle_rest) in rest_variants("lsh", &lsh_model.groups) {
            let predicted_codes_by_position = ctx
                .oracle_codes_by_position
                .iter()
                .enumerate()
                .map(|(pos, oracle_codes)| {
                    let base_codes = rest_codes(use_oracle_rest, oracle_codes, group_majority);
                    lsh_model.predict_selected_groups(&layer_input, pos, base_codes)
                })
                .collect::<Vec<_>>();
            ctx.evaluate_probe(
                mode_d_table,
                &predicted_codes_by_position,
                &probe_name,
                &selected_group_keys,
            )?;
        }
        Ok(())
    }

    fn write_report(&self, report: &mut OraclePqReport) {
        report.address_lsh_group_probe = self.enabled;
        report.address_lsh_groups = when(self.enabled, self.groups.clone());
        report.address_lsh_bits = self.bits;
        report.address_lsh_seeds = self.seeds;
    }
}
