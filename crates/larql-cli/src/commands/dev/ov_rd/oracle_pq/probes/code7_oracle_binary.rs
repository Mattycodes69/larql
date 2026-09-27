//! `--address-code7-oracle-binary-group-probe`: the oracle upper bound of a
//! binary special-code-vs-majority address under a structural filter.

use crate::commands::dev::ov_rd::address::attention_argmax;
use crate::commands::dev::ov_rd::oracle_pq_forward::capture_attention_relation_rows;
use crate::commands::dev::ov_rd::reports::OraclePqReport;

use super::super::args::OraclePqArgs;
use super::super::probe::{row_at, AddressProbe, EvalContext, ProbeResult, ProbeTargets};
use super::super::specs::parse_string_list;
use super::super::validate::{check_groups, levels, parse_sorted, when};
use super::code7_bos::BOS_POSITION;

const FILTER_ALL: &str = "all";
const FILTER_PROSE_BOS: &str = "natural_prose_bos";
const FILTER_PROSE_BOS_OR_PREV: &str = "natural_prose_bos_or_prev";
const NATURAL_PROSE_STRATUM: &str = "natural_prose";

pub(in super::super) struct Code7OracleBinaryProbe {
    enabled: bool,
    groups: Vec<usize>,
    code: usize,
    filters: Vec<String>,
}

impl Code7OracleBinaryProbe {
    pub(in super::super) fn parse(
        args: &OraclePqArgs,
        targets: &ProbeTargets<'_>,
    ) -> ProbeResult<Self> {
        let groups = parse_sorted(&args.address_code7_oracle_binary_groups)?;
        let filters = parse_string_list(&args.address_code7_oracle_binary_filters);
        let code = args.address_code7_oracle_binary_code;
        if args.address_code7_oracle_binary_group_probe {
            if groups.is_empty() {
                return Err("--address-code7-oracle-binary-group-probe requires at least one --address-code7-oracle-binary-groups value".into());
            }
            if filters.is_empty() {
                return Err(
                    "--address-code7-oracle-binary-filters must include at least one filter".into(),
                );
            }
            for filter in &filters {
                if ![FILTER_ALL, FILTER_PROSE_BOS, FILTER_PROSE_BOS_OR_PREV]
                    .contains(&filter.as_str())
                {
                    return Err(format!(
                        "unsupported --address-code7-oracle-binary-filters value {filter:?}; expected all, natural_prose_bos, or natural_prose_bos_or_prev"
                    )
                    .into());
                }
            }
            for config in targets.configs {
                check_groups("--address-code7-oracle-binary-groups", &groups, config)?;
                let levels = levels(config);
                if code >= levels {
                    return Err(format!(
                        "--address-code7-oracle-binary-code is {code}, but config {:?} has only {levels} levels",
                        config
                    )
                    .into());
                }
            }
        }
        Ok(Self {
            enabled: args.address_code7_oracle_binary_group_probe,
            groups,
            code,
            filters,
        })
    }
}

/// Whether `filter` admits position `pos` (filters are validated at parse).
fn relation_matches(filter: &str, stratum: &str, pos: usize, attention_weights: &[f32]) -> bool {
    match filter {
        FILTER_ALL => true,
        FILTER_PROSE_BOS => {
            stratum == NATURAL_PROSE_STRATUM
                && !attention_weights.is_empty()
                && attention_argmax(attention_weights, pos) == BOS_POSITION
        }
        FILTER_PROSE_BOS_OR_PREV => {
            stratum == NATURAL_PROSE_STRATUM
                && (!attention_weights.is_empty()
                    && (attention_argmax(attention_weights, pos) == BOS_POSITION
                        || attention_argmax(attention_weights, pos) == pos.saturating_sub(1)))
        }
        _ => unreachable!("validated oracle binary filter"),
    }
}

impl AddressProbe for Code7OracleBinaryProbe {
    fn enabled(&self) -> bool {
        self.enabled
    }

    fn evaluate(&self, ctx: &mut EvalContext<'_>) -> ProbeResult {
        let mode_d_table = ctx.mode_d_table("code7 oracle binary probe")?;
        let group_majority = ctx.majority("code7 oracle binary probe")?;
        let attention_rows =
            capture_attention_relation_rows(ctx.weights, ctx.token_ids, ctx.index, ctx.head)?;
        for filter in &self.filters {
            let predicted_codes_by_position = ctx
                .oracle_codes_by_position
                .iter()
                .enumerate()
                .map(|(pos, oracle_codes)| {
                    let mut codes = oracle_codes.clone();
                    let matches =
                        relation_matches(filter, ctx.stratum, pos, row_at(&attention_rows, pos));
                    for &group in &self.groups {
                        codes[group] = if matches && oracle_codes[group] == self.code {
                            self.code
                        } else {
                            group_majority[group]
                        };
                    }
                    codes
                })
                .collect::<Vec<_>>();
            let selected_group_keys = ctx.group_keys(|group| {
                self.groups.contains(&group).then(|| {
                    format!(
                        "oracle_{}_code{}_else_majority{}",
                        filter, self.code, group_majority[group]
                    )
                })
            });
            ctx.evaluate_probe(
                mode_d_table,
                &predicted_codes_by_position,
                &format!(
                    "oracle_binary_{}_code{}_groups_{:?}_oracle_rest",
                    filter, self.code, self.groups
                ),
                &selected_group_keys,
            )?;
        }
        Ok(())
    }

    fn write_report(&self, report: &mut OraclePqReport) {
        report.address_code7_oracle_binary_group_probe = self.enabled;
        report.address_code7_oracle_binary_groups = when(self.enabled, self.groups.clone());
        report.address_code7_oracle_binary_code = when(self.enabled, self.code);
        report.address_code7_oracle_binary_filters = when(self.enabled, self.filters.clone());
    }
}
