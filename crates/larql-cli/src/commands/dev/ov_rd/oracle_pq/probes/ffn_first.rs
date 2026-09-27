//! `--address-ffn-first-feature-group-probe`: selected groups predicted
//! from the target layer's FFN run on the pre-attention residual (a
//! diagnostic FFN-first reordering; the real forward is unchanged).

use crate::commands::dev::ov_rd::address::{ffn_first_feature_key, AddressProbeModel};
use crate::commands::dev::ov_rd::oracle_pq_address::fit_address_ffn_first_feature_group_models;
use crate::commands::dev::ov_rd::oracle_pq_forward::capture_ffn_first_feature_keys;
use crate::commands::dev::ov_rd::reports::OraclePqReport;

use super::super::args::OraclePqArgs;
use super::super::probe::{
    row_at, AddressProbe, EvalContext, FitContext, HeadConfigMap, ProbeResult, ProbeTargets,
};
use super::super::validate::{check_groups_all, parse_sorted, when};
use super::keyed::evaluate_keyed_models;

pub(in super::super) struct FfnFirstFeatureProbe {
    enabled: bool,
    groups: Vec<usize>,
    top_k: usize,
    models: HeadConfigMap<Vec<AddressProbeModel>>,
}

impl FfnFirstFeatureProbe {
    pub(in super::super) fn parse(
        args: &OraclePqArgs,
        targets: &ProbeTargets<'_>,
    ) -> ProbeResult<Self> {
        let groups = parse_sorted(&args.address_ffn_first_feature_groups)?;
        if args.address_ffn_first_feature_group_probe {
            if groups.is_empty() {
                return Err("--address-ffn-first-feature-group-probe requires at least one --address-ffn-first-feature-groups value".into());
            }
            if args.address_ffn_first_feature_top_k == 0 {
                return Err("--address-ffn-first-feature-top-k must be greater than zero".into());
            }
            check_groups_all(
                "--address-ffn-first-feature-groups",
                &groups,
                targets.configs,
            )?;
        }
        Ok(Self {
            enabled: args.address_ffn_first_feature_group_probe,
            groups,
            top_k: args.address_ffn_first_feature_top_k,
            models: HeadConfigMap::new(),
        })
    }
}

impl AddressProbe for FfnFirstFeatureProbe {
    fn enabled(&self) -> bool {
        self.enabled
    }

    fn fit(&mut self, ctx: &mut FitContext<'_>) -> ProbeResult {
        if !self.enabled {
            return Ok(());
        }
        ctx.require_mode_d("--address-ffn-first-feature-group-probe requires --mode-d-check")?;
        eprintln!(
            "Fitting FFN-first feature group address probes for groups {:?} (top_k={})",
            self.groups, self.top_k
        );
        self.models = fit_address_ffn_first_feature_group_models(
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
            self.top_k,
        )?;
        Ok(())
    }

    fn evaluate(&self, ctx: &mut EvalContext<'_>) -> ProbeResult {
        let mode_d_table = ctx.mode_d_table("FFN-first feature group probe")?;
        let models = ctx.model(&self.models, "FFN-first feature group probe model")?;
        let group_majority = ctx.majority("FFN-first feature group probe")?;
        let features = capture_ffn_first_feature_keys(
            ctx.weights,
            ctx.token_ids,
            ctx.index,
            ctx.head.layer,
            self.top_k,
        )?;
        evaluate_keyed_models(
            ctx,
            mode_d_table,
            group_majority,
            models,
            &self.groups,
            |model, ctx, pos| {
                ffn_first_feature_key(
                    &model.name,
                    ctx.token_ids,
                    ctx.stratum,
                    pos,
                    row_at(&features, pos),
                )
            },
        )
    }

    fn write_report(&self, report: &mut OraclePqReport) {
        report.address_ffn_first_feature_group_probe = self.enabled;
        report.address_ffn_first_feature_groups = when(self.enabled, self.groups.clone());
        report.address_ffn_first_feature_top_k = self.top_k;
    }
}
