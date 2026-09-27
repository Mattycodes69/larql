//! `--address-supervised-group-probe`: selected groups predicted by
//! SGD-trained binary hyperplanes over the residual entering the target
//! layer.

use crate::commands::dev::ov_rd::address::AddressSupervisedGroupModel;
use crate::commands::dev::ov_rd::oracle_pq_address::fit_address_supervised_group_models;
use crate::commands::dev::ov_rd::oracle_pq_forward::capture_layer_input_hidden;
use crate::commands::dev::ov_rd::reports::OraclePqReport;

use super::super::args::OraclePqArgs;
use super::super::probe::{
    rest_codes, rest_variants, AddressProbe, EvalContext, FitContext, HeadConfigMap, ProbeResult,
    ProbeTargets,
};
use super::super::validate::{check_groups_all, parse_sorted, when};

/// The SGD settings shared with the gamma-projected probe, which trains
/// the same group-bit classifiers on a projected input.
#[derive(Clone, Copy)]
pub(in super::super) struct SgdSettings {
    pub(in super::super) epochs: usize,
    pub(in super::super) lr: f32,
    pub(in super::super) l2: f32,
}

pub(in super::super) struct SupervisedProbe {
    enabled: bool,
    groups: Vec<usize>,
    sgd: SgdSettings,
    models: HeadConfigMap<AddressSupervisedGroupModel>,
}

impl SupervisedProbe {
    pub(in super::super) fn parse(
        args: &OraclePqArgs,
        targets: &ProbeTargets<'_>,
    ) -> ProbeResult<Self> {
        let groups = parse_sorted(&args.address_supervised_groups)?;
        if args.address_supervised_group_probe {
            if groups.is_empty() {
                return Err(
                    "--address-supervised-group-probe requires at least one --address-supervised-groups value".into(),
                );
            }
            if args.address_supervised_epochs == 0 {
                return Err("--address-supervised-epochs must be greater than zero".into());
            }
            if args.address_supervised_lr <= 0.0 {
                return Err("--address-supervised-lr must be greater than zero".into());
            }
            if args.address_supervised_l2 < 0.0 {
                return Err("--address-supervised-l2 must be non-negative".into());
            }
            check_groups_all("--address-supervised-groups", &groups, targets.configs)?;
        }
        Ok(Self {
            enabled: args.address_supervised_group_probe,
            groups,
            sgd: SgdSettings::from_args(args),
            models: HeadConfigMap::new(),
        })
    }
}

impl SgdSettings {
    pub(in super::super) fn from_args(args: &OraclePqArgs) -> Self {
        Self {
            epochs: args.address_supervised_epochs,
            lr: args.address_supervised_lr,
            l2: args.address_supervised_l2,
        }
    }
}

impl AddressProbe for SupervisedProbe {
    fn enabled(&self) -> bool {
        self.enabled
    }

    fn fit(&mut self, ctx: &mut FitContext<'_>) -> ProbeResult {
        if !self.enabled {
            return Ok(());
        }
        ctx.require_mode_d("--address-supervised-group-probe requires --mode-d-check")?;
        eprintln!(
            "Fitting supervised group address probes for groups {:?} (epochs={}, lr={}, l2={})",
            self.groups, self.sgd.epochs, self.sgd.lr, self.sgd.l2
        );
        self.models = fit_address_supervised_group_models(
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
            self.sgd.epochs,
            self.sgd.lr,
            self.sgd.l2,
        )?;
        Ok(())
    }

    fn evaluate(&self, ctx: &mut EvalContext<'_>) -> ProbeResult {
        let mode_d_table = ctx.mode_d_table("supervised group probe")?;
        let supervised_model = ctx.model(&self.models, "supervised group probe model")?;
        let group_majority = ctx.majority("supervised group probe")?;
        let layer_input =
            capture_layer_input_hidden(ctx.weights, ctx.token_ids, ctx.index, ctx.head.layer)?;
        let selected_group_keys = supervised_model.selected_group_keys();
        for (probe_name, use_oracle_rest) in
            rest_variants("supervised_hyperplane", &supervised_model.groups)
        {
            let predicted_codes_by_position = ctx
                .oracle_codes_by_position
                .iter()
                .enumerate()
                .map(|(pos, oracle_codes)| {
                    let base_codes = rest_codes(use_oracle_rest, oracle_codes, group_majority);
                    supervised_model.predict_selected_groups(&layer_input, pos, base_codes)
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
        report.address_supervised_group_probe = self.enabled;
        report.address_supervised_groups = when(self.enabled, self.groups.clone());
        report.address_supervised_epochs = self.sgd.epochs;
        report.address_supervised_lr = self.sgd.lr;
        report.address_supervised_l2 = self.sgd.l2;
    }
}
