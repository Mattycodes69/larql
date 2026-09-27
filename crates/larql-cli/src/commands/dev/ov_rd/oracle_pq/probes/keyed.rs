//! The shared evaluation of discrete-key group probes: a per-position key
//! looks up predicted codes for the selected groups, under both the
//! oracle-rest and majority-rest variants.

use crate::commands::dev::ov_rd::address::{AddressAttentionClusterGroupModel, AddressProbeModel};
use crate::commands::dev::ov_rd::pq::ModeDTable;

use super::super::probe::{rest_codes, rest_variants, row_at, EvalContext, ProbeResult};

pub(super) fn evaluate_keyed_models(
    ctx: &mut EvalContext<'_>,
    mode_d_table: &ModeDTable,
    group_majority: &[usize],
    models: &[AddressProbeModel],
    groups: &[usize],
    key_at: impl Fn(&AddressProbeModel, &EvalContext<'_>, usize) -> String,
) -> ProbeResult {
    for probe_model in models {
        let selected_group_keys = probe_model.selected_group_keys.clone();
        for (probe_name, use_oracle_rest) in rest_variants(&probe_model.name, groups) {
            let predicted_codes_by_position = ctx
                .oracle_codes_by_position
                .iter()
                .enumerate()
                .map(|(pos, oracle_codes)| {
                    let mut codes = if use_oracle_rest {
                        oracle_codes.clone()
                    } else {
                        group_majority.to_vec()
                    };
                    let key = key_at(probe_model, ctx, pos);
                    let probe_codes = probe_model.predict_codes_from_key(&key);
                    for &group in groups {
                        codes[group] = probe_codes[group];
                    }
                    codes
                })
                .collect::<Vec<_>>();
            ctx.evaluate_probe(
                mode_d_table,
                &predicted_codes_by_position,
                &probe_name,
                &selected_group_keys,
            )?;
        }
    }
    Ok(())
}

/// One attention-pattern cluster model over precomputed attention rows.
pub(super) fn evaluate_cluster_model(
    ctx: &mut EvalContext<'_>,
    mode_d_table: &ModeDTable,
    group_majority: &[usize],
    cluster_model: &AddressAttentionClusterGroupModel,
    groups: &[usize],
    attention_rows: &[Vec<f32>],
) -> ProbeResult {
    let selected_group_keys = cluster_model.selected_group_keys.clone();
    for (probe_name, use_oracle_rest) in rest_variants(&cluster_model.name, groups) {
        let predicted_codes_by_position = ctx
            .oracle_codes_by_position
            .iter()
            .enumerate()
            .map(|(pos, oracle_codes)| {
                let base_codes = rest_codes(use_oracle_rest, oracle_codes, group_majority);
                cluster_model.predict_selected_groups(
                    ctx.token_ids,
                    ctx.stratum,
                    pos,
                    row_at(attention_rows, pos),
                    base_codes,
                )
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

/// The name filter of the cluster probes: empty keeps every model.
pub(super) fn name_selected(names: &[String], name: &String) -> bool {
    names.is_empty() || names.contains(name)
}
