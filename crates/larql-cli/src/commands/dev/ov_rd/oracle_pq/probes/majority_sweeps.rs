//! The two majority-replacement sweeps over an oracle address:
//! `--address-group-importance` (one group at a time) and
//! `--address-corruption-sweep` (all but a kept prefix of groups).

use crate::commands::dev::ov_rd::oracle_pq_mode_d::corruption_keep_values;
use crate::commands::dev::ov_rd::reports::OraclePqReport;

use super::super::args::OraclePqArgs;
use super::super::probe::{AddressProbe, EvalContext, ProbeResult};

/// Replace `replace(group)` groups of every oracle address with the
/// group's majority code.
fn replace_with_majority(
    ctx: &EvalContext<'_>,
    majority: &[usize],
    replace: impl Fn(usize) -> bool,
) -> Vec<Vec<usize>> {
    ctx.oracle_codes_by_position
        .iter()
        .map(|codes| {
            codes
                .iter()
                .enumerate()
                .map(|(group, &code)| {
                    if replace(group) {
                        majority[group]
                    } else {
                        code
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

pub(in super::super) struct GroupImportanceSweep {
    enabled: bool,
}

impl GroupImportanceSweep {
    pub(in super::super) fn parse(args: &OraclePqArgs) -> Self {
        Self {
            enabled: args.address_group_importance,
        }
    }
}

impl AddressProbe for GroupImportanceSweep {
    fn enabled(&self) -> bool {
        self.enabled
    }

    fn evaluate(&self, ctx: &mut EvalContext<'_>) -> ProbeResult {
        let mode_d_table = ctx.mode_d_table("address group importance")?;
        let group_majority = ctx.majority("address group importance")?;
        for replaced_group in 0..ctx.config.groups {
            let predicted =
                replace_with_majority(ctx, group_majority, |group| group == replaced_group);
            let prompt_report = ctx.evaluate(mode_d_table, &predicted)?;
            ctx.accumulator
                .add_address_group_importance(replaced_group, prompt_report);
        }
        Ok(())
    }

    fn write_report(&self, report: &mut OraclePqReport) {
        report.address_group_importance = self.enabled;
    }
}

pub(in super::super) struct CorruptionSweep {
    enabled: bool,
}

impl CorruptionSweep {
    pub(in super::super) fn parse(args: &OraclePqArgs) -> Self {
        Self {
            enabled: args.address_corruption_sweep,
        }
    }
}

impl AddressProbe for CorruptionSweep {
    fn enabled(&self) -> bool {
        self.enabled
    }

    fn evaluate(&self, ctx: &mut EvalContext<'_>) -> ProbeResult {
        let mode_d_table = ctx.mode_d_table("address corruption")?;
        let group_majority = ctx.majority("address corruption")?;
        for oracle_groups_kept in corruption_keep_values(ctx.config.groups) {
            let predicted =
                replace_with_majority(ctx, group_majority, |group| group >= oracle_groups_kept);
            let prompt_report = ctx.evaluate(mode_d_table, &predicted)?;
            ctx.accumulator
                .add_address_corruption(oracle_groups_kept, prompt_report);
        }
        Ok(())
    }

    fn write_report(&self, report: &mut OraclePqReport) {
        report.address_corruption_sweep = self.enabled;
    }
}
