//! `--address-majority-group-probe`: selected groups replaced with their
//! train-split majority code, the rest oracle-correct.

use crate::commands::dev::ov_rd::reports::OraclePqReport;

use super::super::args::OraclePqArgs;
use super::super::probe::{AddressProbe, EvalContext, ProbeResult, ProbeTargets};
use super::super::validate::{check_groups_all, parse_sorted, when};

/// Group label for a group replaced by its majority code.
const MAJORITY_GROUP_KEY: &str = "majority";

pub(in super::super) struct MajorityGroupProbe {
    enabled: bool,
    groups: Vec<usize>,
}

impl MajorityGroupProbe {
    pub(in super::super) fn parse(
        args: &OraclePqArgs,
        targets: &ProbeTargets<'_>,
    ) -> ProbeResult<Self> {
        let groups = parse_sorted(&args.address_majority_groups)?;
        if args.address_majority_group_probe {
            if groups.is_empty() {
                return Err("--address-majority-group-probe requires at least one --address-majority-groups value".into());
            }
            check_groups_all("--address-majority-groups", &groups, targets.configs)?;
        }
        Ok(Self {
            enabled: args.address_majority_group_probe,
            groups,
        })
    }
}

impl AddressProbe for MajorityGroupProbe {
    fn enabled(&self) -> bool {
        self.enabled
    }

    fn evaluate(&self, ctx: &mut EvalContext<'_>) -> ProbeResult {
        let mode_d_table = ctx.mode_d_table("majority group probe")?;
        let group_majority = ctx.majority("majority group probe")?;
        let predicted_codes_by_position = ctx
            .oracle_codes_by_position
            .iter()
            .map(|oracle_codes| {
                let mut codes = oracle_codes.clone();
                for &group in &self.groups {
                    codes[group] = group_majority[group];
                }
                codes
            })
            .collect::<Vec<_>>();
        let selected_group_keys = ctx.group_keys(|group| {
            self.groups
                .contains(&group)
                .then(|| MAJORITY_GROUP_KEY.to_string())
        });
        ctx.evaluate_probe(
            mode_d_table,
            &predicted_codes_by_position,
            &format!("majority_groups_{:?}_oracle_rest", self.groups),
            &selected_group_keys,
        )
    }

    fn write_report(&self, report: &mut OraclePqReport) {
        report.address_majority_group_probe = self.enabled;
        report.address_majority_groups = when(self.enabled, self.groups.clone());
    }
}
