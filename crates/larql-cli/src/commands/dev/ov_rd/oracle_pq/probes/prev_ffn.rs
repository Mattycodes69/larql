//! `--address-prev-ffn-feature-group-probe`: selected groups predicted from
//! the previous layer's top FFN-activation features.

use crate::commands::dev::ov_rd::address::{prev_ffn_feature_key, AddressProbeModel};
use crate::commands::dev::ov_rd::oracle_pq_address::fit_address_prev_ffn_feature_group_models;
use crate::commands::dev::ov_rd::oracle_pq_forward::capture_prev_ffn_feature_keys;
use crate::commands::dev::ov_rd::reports::OraclePqReport;

use super::super::args::OraclePqArgs;
use super::super::probe::{
    row_at, AddressProbe, EvalContext, FitContext, HeadConfigMap, ProbeResult, ProbeTargets,
};
use super::super::validate::{check_groups_all, parse_sorted, when};
use super::keyed::evaluate_keyed_models;

pub(in super::super) struct PrevFfnFeatureProbe {
    enabled: bool,
    groups: Vec<usize>,
    top_k: usize,
    models: HeadConfigMap<Vec<AddressProbeModel>>,
}

impl PrevFfnFeatureProbe {
    pub(in super::super) fn parse(
        args: &OraclePqArgs,
        targets: &ProbeTargets<'_>,
    ) -> ProbeResult<Self> {
        let groups = parse_sorted(&args.address_prev_ffn_feature_groups)?;
        if args.address_prev_ffn_feature_group_probe {
            if groups.is_empty() {
                return Err("--address-prev-ffn-feature-group-probe requires at least one --address-prev-ffn-feature-groups value".into());
            }
            if args.address_prev_ffn_feature_top_k == 0 {
                return Err("--address-prev-ffn-feature-top-k must be greater than zero".into());
            }
            for head in targets.heads {
                if head.layer == 0 {
                    eprintln!(
                        "warning: L{}H{} has no previous layer; previous-FFN feature keys will be 'none'",
                        head.layer, head.head
                    );
                }
            }
            check_groups_all(
                "--address-prev-ffn-feature-groups",
                &groups,
                targets.configs,
            )?;
        }
        Ok(Self {
            enabled: args.address_prev_ffn_feature_group_probe,
            groups,
            top_k: args.address_prev_ffn_feature_top_k,
            models: HeadConfigMap::new(),
        })
    }
}

impl AddressProbe for PrevFfnFeatureProbe {
    fn enabled(&self) -> bool {
        self.enabled
    }

    fn fit(&mut self, ctx: &mut FitContext<'_>) -> ProbeResult {
        if !self.enabled {
            return Ok(());
        }
        ctx.require_mode_d("--address-prev-ffn-feature-group-probe requires --mode-d-check")?;
        eprintln!(
            "Fitting previous-FFN feature group address probes for groups {:?} (top_k={})",
            self.groups, self.top_k
        );
        self.models = fit_address_prev_ffn_feature_group_models(
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
        let mode_d_table = ctx.mode_d_table("previous-FFN feature group probe")?;
        let models = ctx.model(&self.models, "previous-FFN feature group probe model")?;
        let group_majority = ctx.majority("previous-FFN feature group probe")?;
        let features = capture_prev_ffn_feature_keys(
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
                prev_ffn_feature_key(
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
        report.address_prev_ffn_feature_group_probe = self.enabled;
        report.address_prev_ffn_feature_groups = when(self.enabled, self.groups.clone());
        report.address_prev_ffn_feature_top_k = self.top_k;
    }
}
