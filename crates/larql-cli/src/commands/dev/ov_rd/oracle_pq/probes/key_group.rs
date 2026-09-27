//! Graph-native discrete-key address probes (`--address-probes`,
//! `--address-mixed-key-probe`) and their selected-group variant
//! (`--address-key-group-probe`).

use crate::commands::dev::ov_rd::address::AddressProbeModel;
use crate::commands::dev::ov_rd::oracle_pq_address::fit_address_probe_models;
use crate::commands::dev::ov_rd::reports::OraclePqReport;

use super::super::args::OraclePqArgs;
use super::super::probe::{
    rest_variants, AddressProbe, EvalContext, FitContext, HeadConfigMap, ProbeResult, ProbeTargets,
};
use super::super::specs::parse_string_list;
use super::super::validate::{check_groups_all, parse_sorted};

/// The mixed probe's model name; it runs under `--address-mixed-key-probe`
/// even when the full `--address-probes` set is off.
const MIXED_PROBE_NAME: &str = "mixed_best_simple_key";

pub(in super::super) struct KeyGroupProbe {
    address_probes: bool,
    mixed_key_probe: bool,
    key_group_probe: bool,
    groups: Vec<usize>,
    probe_names: Vec<String>,
    models: HeadConfigMap<Vec<AddressProbeModel>>,
}

impl KeyGroupProbe {
    pub(in super::super) fn parse(
        args: &OraclePqArgs,
        targets: &ProbeTargets<'_>,
    ) -> ProbeResult<Self> {
        let groups = parse_sorted(&args.address_key_groups)?;
        let probe_names = parse_string_list(&args.address_key_group_probe_names);
        if args.address_key_group_probe {
            if groups.is_empty() {
                return Err(
                    "--address-key-group-probe requires at least one --address-key-groups value"
                        .into(),
                );
            }
            check_groups_all("--address-key-groups", &groups, targets.configs)?;
        }
        Ok(Self {
            address_probes: args.address_probes,
            mixed_key_probe: args.address_mixed_key_probe,
            key_group_probe: args.address_key_group_probe,
            groups,
            probe_names,
            models: HeadConfigMap::new(),
        })
    }

    fn evaluate_key_groups(
        &self,
        ctx: &mut EvalContext<'_>,
        probe_model: &AddressProbeModel,
    ) -> ProbeResult {
        if !self.probe_names.is_empty() && !self.probe_names.contains(&probe_model.name) {
            return Ok(());
        }
        let group_majority = ctx.majority("key group probe")?;
        let mode_d_table = ctx.mode_d_table("address probes")?;
        for (probe_name, use_oracle_rest) in rest_variants(&probe_model.name, &self.groups) {
            let predicted_codes_by_position = ctx
                .oracle_codes_by_position
                .iter()
                .enumerate()
                .map(|(pos, oracle_codes)| {
                    let mut codes = if use_oracle_rest {
                        oracle_codes.clone()
                    } else {
                        group_majority.clone()
                    };
                    let probe_codes = probe_model.predict_codes(ctx.token_ids, ctx.stratum, pos);
                    for &group in &self.groups {
                        codes[group] = probe_codes[group];
                    }
                    codes
                })
                .collect::<Vec<_>>();
            ctx.evaluate_probe(
                mode_d_table,
                &predicted_codes_by_position,
                &probe_name,
                &probe_model.selected_group_keys,
            )?;
        }
        Ok(())
    }
}

impl AddressProbe for KeyGroupProbe {
    fn enabled(&self) -> bool {
        self.address_probes || self.mixed_key_probe || self.key_group_probe
    }

    fn fit(&mut self, ctx: &mut FitContext<'_>) -> ProbeResult {
        if !self.enabled() {
            return Ok(());
        }
        ctx.require_mode_d("--address-probes/--address-mixed-key-probe requires --mode-d-check")?;
        eprintln!("Fitting graph-native address probes");
        self.models = fit_address_probe_models(
            ctx.weights,
            ctx.index,
            ctx.tokenizer,
            ctx.fit_prompts,
            ctx.heads,
            ctx.bases,
            ctx.means,
            ctx.pca_bases,
            ctx.codebooks,
            self.mixed_key_probe,
        )?;
        Ok(())
    }

    fn needs_majority_codes(&self) -> bool {
        self.key_group_probe
    }

    fn evaluate(&self, ctx: &mut EvalContext<'_>) -> ProbeResult {
        let mode_d_table = ctx.mode_d_table("address probes")?;
        let probe_models = ctx.model(&self.models, "address probe models")?;
        for probe_model in probe_models {
            if self.address_probes || probe_model.name == MIXED_PROBE_NAME {
                let predicted_codes_by_position = (0..ctx.token_ids.len())
                    .map(|pos| probe_model.predict_codes(ctx.token_ids, ctx.stratum, pos))
                    .collect::<Vec<_>>();
                ctx.evaluate_probe(
                    mode_d_table,
                    &predicted_codes_by_position,
                    &probe_model.name,
                    &probe_model.selected_group_keys,
                )?;
            }
            if self.key_group_probe {
                self.evaluate_key_groups(ctx, probe_model)?;
            }
        }
        Ok(())
    }

    fn write_report(&self, report: &mut OraclePqReport) {
        report.address_probes = self.address_probes;
        report.address_mixed_key_probe = self.mixed_key_probe;
        report.address_key_group_probe = self.key_group_probe;
        if self.key_group_probe {
            report.address_key_groups = self.groups.clone();
            report.address_key_group_probe_names = self.probe_names.clone();
        }
    }
}
