//! `--address-attention-relation-group-probe`: selected groups predicted
//! from discrete keys over the head's attention distribution (QK routing
//! structure rather than token or FFN-feature state).

use crate::commands::dev::ov_rd::address::{attention_relation_key, AddressProbeModel};
use crate::commands::dev::ov_rd::oracle_pq_address::fit_address_attention_relation_group_models;
use crate::commands::dev::ov_rd::oracle_pq_forward::capture_attention_relation_rows;
use crate::commands::dev::ov_rd::reports::OraclePqReport;

use super::super::args::OraclePqArgs;
use super::super::probe::{
    row_at, AddressProbe, EvalContext, FitContext, HeadConfigMap, ProbeResult, ProbeTargets,
};
use super::super::validate::{check_groups_all, parse_sorted, when};
use super::keyed::evaluate_keyed_models;

pub(in super::super) struct AttentionRelationProbe {
    enabled: bool,
    groups: Vec<usize>,
    models: HeadConfigMap<Vec<AddressProbeModel>>,
}

impl AttentionRelationProbe {
    pub(in super::super) fn parse(
        args: &OraclePqArgs,
        targets: &ProbeTargets<'_>,
    ) -> ProbeResult<Self> {
        let groups = parse_sorted(&args.address_attention_relation_groups)?;
        if args.address_attention_relation_group_probe {
            if groups.is_empty() {
                return Err("--address-attention-relation-group-probe requires at least one --address-attention-relation-groups value".into());
            }
            check_groups_all(
                "--address-attention-relation-groups",
                &groups,
                targets.configs,
            )?;
        }
        Ok(Self {
            enabled: args.address_attention_relation_group_probe,
            groups,
            models: HeadConfigMap::new(),
        })
    }
}

impl AddressProbe for AttentionRelationProbe {
    fn enabled(&self) -> bool {
        self.enabled
    }

    fn fit(&mut self, ctx: &mut FitContext<'_>) -> ProbeResult {
        if !self.enabled {
            return Ok(());
        }
        ctx.require_mode_d("--address-attention-relation-group-probe requires --mode-d-check")?;
        eprintln!(
            "Fitting attention-relation group address probes for groups {:?}",
            self.groups
        );
        self.models = fit_address_attention_relation_group_models(
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
        )?;
        Ok(())
    }

    fn evaluate(&self, ctx: &mut EvalContext<'_>) -> ProbeResult {
        let mode_d_table = ctx.mode_d_table("attention-relation group probe")?;
        let models = ctx.model(&self.models, "attention-relation group probe model")?;
        let group_majority = ctx.majority("attention-relation group probe")?;
        let attention_rows =
            capture_attention_relation_rows(ctx.weights, ctx.token_ids, ctx.index, ctx.head)?;
        evaluate_keyed_models(
            ctx,
            mode_d_table,
            group_majority,
            models,
            &self.groups,
            |model, ctx, pos| {
                attention_relation_key(
                    &model.name,
                    ctx.token_ids,
                    ctx.stratum,
                    pos,
                    row_at(&attention_rows, pos),
                )
            },
        )
    }

    fn write_report(&self, report: &mut OraclePqReport) {
        report.address_attention_relation_group_probe = self.enabled;
        report.address_attention_relation_groups = when(self.enabled, self.groups.clone());
    }
}
