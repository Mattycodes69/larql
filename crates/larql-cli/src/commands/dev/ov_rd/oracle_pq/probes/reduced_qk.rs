//! `--address-reduced-qk-cluster-group-probe`: the attention-cluster probe
//! with the attention distribution recomputed from only the first `r` Q/K
//! dimensions (rank 0 is the full-QK control).

use std::collections::HashMap;

use crate::commands::dev::ov_rd::address::AddressAttentionClusterGroupModel;
use crate::commands::dev::ov_rd::oracle_pq_address::fit_address_reduced_qk_cluster_group_models;
use crate::commands::dev::ov_rd::oracle_pq_forward::{
    capture_attention_relation_rows, capture_reduced_qk_attention_rows,
};
use crate::commands::dev::ov_rd::reports::OraclePqReport;

use super::super::args::OraclePqArgs;
use super::super::probe::{
    AddressProbe, EvalContext, FitContext, HeadConfigMap, ProbeResult, ProbeTargets,
};
use super::super::specs::parse_string_list;
use super::super::validate::{check_groups_all, parse_sorted, when};
use super::attn_cluster::CLUSTER_COUNT_RANGE;
use super::keyed::{evaluate_cluster_model, name_selected};

pub(in super::super) struct ReducedQkClusterProbe {
    enabled: bool,
    groups: Vec<usize>,
    ranks: Vec<usize>,
    cluster_ks: Vec<usize>,
    probe_names: Vec<String>,
    models: HeadConfigMap<Vec<AddressAttentionClusterGroupModel>>,
}

impl ReducedQkClusterProbe {
    pub(in super::super) fn parse(
        args: &OraclePqArgs,
        targets: &ProbeTargets<'_>,
    ) -> ProbeResult<Self> {
        let groups = parse_sorted(&args.address_reduced_qk_cluster_groups)?;
        let ranks = parse_sorted(&args.address_reduced_qk_ranks)?;
        let cluster_ks = parse_sorted(&args.address_reduced_qk_cluster_ks)?;
        let probe_names = parse_string_list(&args.address_reduced_qk_cluster_probe_names);
        if args.address_reduced_qk_cluster_group_probe {
            if groups.is_empty() {
                return Err("--address-reduced-qk-cluster-group-probe requires at least one --address-reduced-qk-cluster-groups value".into());
            }
            if ranks.is_empty() {
                return Err("--address-reduced-qk-ranks must include at least one rank".into());
            }
            if cluster_ks.is_empty() {
                return Err("--address-reduced-qk-cluster-ks must include at least one k".into());
            }
            if cluster_ks.iter().any(|k| !CLUSTER_COUNT_RANGE.contains(k)) {
                return Err(
                    "--address-reduced-qk-cluster-ks values must be between 2 and 128".into(),
                );
            }
            check_groups_all(
                "--address-reduced-qk-cluster-groups",
                &groups,
                targets.configs,
            )?;
        }
        Ok(Self {
            enabled: args.address_reduced_qk_cluster_group_probe,
            groups,
            ranks,
            cluster_ks,
            probe_names,
            models: HeadConfigMap::new(),
        })
    }
}

impl AddressProbe for ReducedQkClusterProbe {
    fn enabled(&self) -> bool {
        self.enabled
    }

    fn fit(&mut self, ctx: &mut FitContext<'_>) -> ProbeResult {
        if !self.enabled {
            return Ok(());
        }
        ctx.require_mode_d("--address-reduced-qk-cluster-group-probe requires --mode-d-check")?;
        eprintln!(
            "Fitting reduced-QK cluster group address probes for groups {:?} (ranks={:?}, k={:?})",
            self.groups, self.ranks, self.cluster_ks
        );
        self.models = fit_address_reduced_qk_cluster_group_models(
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
            &self.ranks,
            &self.cluster_ks,
        )?;
        Ok(())
    }

    fn evaluate(&self, ctx: &mut EvalContext<'_>) -> ProbeResult {
        let mode_d_table = ctx.mode_d_table("reduced-QK cluster group probe")?;
        let cluster_models = ctx.model(&self.models, "reduced-QK cluster group probe model")?;
        let group_majority = ctx.majority("reduced-QK cluster group probe")?;
        let mut rows_by_rank: HashMap<Option<usize>, Vec<Vec<f32>>> = HashMap::new();
        for cluster_model in cluster_models {
            if !name_selected(&self.probe_names, &cluster_model.name) {
                continue;
            }
            if !rows_by_rank.contains_key(&cluster_model.qk_rank) {
                let rows = match cluster_model.qk_rank {
                    Some(qk_rank) => capture_reduced_qk_attention_rows(
                        ctx.weights,
                        ctx.token_ids,
                        ctx.index,
                        ctx.head,
                        qk_rank,
                    )?,
                    None => capture_attention_relation_rows(
                        ctx.weights,
                        ctx.token_ids,
                        ctx.index,
                        ctx.head,
                    )?,
                };
                rows_by_rank.insert(cluster_model.qk_rank, rows);
            }
            let attention_rows = rows_by_rank
                .get(&cluster_model.qk_rank)
                .expect("reduced-QK rows were just inserted");
            evaluate_cluster_model(
                ctx,
                mode_d_table,
                group_majority,
                cluster_model,
                &self.groups,
                attention_rows,
            )?;
        }
        Ok(())
    }

    fn write_report(&self, report: &mut OraclePqReport) {
        let enabled = self.enabled;
        report.address_reduced_qk_cluster_group_probe = enabled;
        report.address_reduced_qk_cluster_groups = when(enabled, self.groups.clone());
        report.address_reduced_qk_ranks = when(enabled, self.ranks.clone());
        report.address_reduced_qk_cluster_ks = when(enabled, self.cluster_ks.clone());
        report.address_reduced_qk_cluster_probe_names = when(enabled, self.probe_names.clone());
    }
}
