//! `--address-code7-bos-rule-group-probe`: a hard-coded fallback rule —
//! predict the special code where attention's argmax is BOS outside the
//! arithmetic stratum, else the train majority.

use crate::commands::dev::ov_rd::address::attention_argmax;
use crate::commands::dev::ov_rd::oracle_pq_forward::capture_attention_relation_rows;
use crate::commands::dev::ov_rd::reports::OraclePqReport;

use super::super::args::OraclePqArgs;
use super::super::probe::{row_at, AddressProbe, EvalContext, ProbeResult, ProbeTargets};
use super::super::validate::{check_groups, levels, parse_sorted, when};

/// Stratum the rule never predicts the special code in.
const ARITHMETIC_STRATUM: &str = "arithmetic";
/// Position of the BOS token an attention argmax points at.
pub(super) const BOS_POSITION: usize = 0;

pub(in super::super) struct Code7BosRuleProbe {
    enabled: bool,
    groups: Vec<usize>,
    code: usize,
}

impl Code7BosRuleProbe {
    pub(in super::super) fn parse(
        args: &OraclePqArgs,
        targets: &ProbeTargets<'_>,
    ) -> ProbeResult<Self> {
        let groups = parse_sorted(&args.address_code7_bos_rule_groups)?;
        let code = args.address_code7_bos_rule_code;
        if args.address_code7_bos_rule_group_probe {
            if groups.is_empty() {
                return Err("--address-code7-bos-rule-group-probe requires at least one --address-code7-bos-rule-groups value".into());
            }
            for config in targets.configs {
                check_groups("--address-code7-bos-rule-groups", &groups, config)?;
                let levels = levels(config);
                if code >= levels {
                    return Err(format!(
                        "--address-code7-bos-rule-code is {code}, but config {:?} has only {levels} levels",
                        config
                    )
                    .into());
                }
            }
        }
        Ok(Self {
            enabled: args.address_code7_bos_rule_group_probe,
            groups,
            code,
        })
    }
}

impl AddressProbe for Code7BosRuleProbe {
    fn enabled(&self) -> bool {
        self.enabled
    }

    fn evaluate(&self, ctx: &mut EvalContext<'_>) -> ProbeResult {
        let mode_d_table = ctx.mode_d_table("code7 BOS rule probe")?;
        let group_majority = ctx.majority("code7 BOS rule probe")?;
        let attention_rows =
            capture_attention_relation_rows(ctx.weights, ctx.token_ids, ctx.index, ctx.head)?;
        let use_special_code = ctx.stratum != ARITHMETIC_STRATUM;
        let predicted_codes_by_position = ctx
            .oracle_codes_by_position
            .iter()
            .enumerate()
            .map(|(pos, oracle_codes)| {
                let mut codes = oracle_codes.clone();
                let attention_weights = row_at(&attention_rows, pos);
                let predicts_special = use_special_code
                    && !attention_weights.is_empty()
                    && attention_argmax(attention_weights, pos) == BOS_POSITION;
                for &group in &self.groups {
                    codes[group] = if predicts_special {
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
                    "bos_non_arithmetic_to_code{}_else_majority{}",
                    self.code, group_majority[group]
                )
            })
        });
        ctx.evaluate_probe(
            mode_d_table,
            &predicted_codes_by_position,
            &format!(
                "code{}_bos_non_arithmetic_groups_{:?}_oracle_rest",
                self.code, self.groups
            ),
            &selected_group_keys,
        )
    }

    fn write_report(&self, report: &mut OraclePqReport) {
        report.address_code7_bos_rule_group_probe = self.enabled;
        report.address_code7_bos_rule_groups = when(self.enabled, self.groups.clone());
        report.address_code7_bos_rule_code = when(self.enabled, self.code);
    }
}
