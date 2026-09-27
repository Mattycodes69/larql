//! `--address-code-position-interaction-probe`: within one named prompt,
//! rewrite chosen subsets of the positions holding primary / secondary
//! source codes to one target code, to localise quotient failures.

use crate::commands::dev::ov_rd::reports::OraclePqReport;

use super::super::args::OraclePqArgs;
use super::super::probe::{AddressProbe, EvalContext, ProbeResult, ProbeTargets};
use super::super::validate::{levels, parse_sorted, when};

/// `A0`..`A2` are the three whole-set variants; the per-secondary-position
/// variants are numbered from here.
const FIRST_PER_POSITION_VARIANT: usize = 3;

pub(in super::super) struct CodePositionProbe {
    enabled: bool,
    prompt_id: String,
    group: usize,
    primary_codes: Vec<usize>,
    secondary_codes: Vec<usize>,
    target_code: usize,
}

impl CodePositionProbe {
    pub(in super::super) fn parse(
        args: &OraclePqArgs,
        targets: &ProbeTargets<'_>,
    ) -> ProbeResult<Self> {
        let primary_codes = parse_sorted(&args.address_code_position_primary_codes)?;
        let secondary_codes = parse_sorted(&args.address_code_position_secondary_codes)?;
        let prompt_id = args.address_code_position_prompt_id.trim().to_string();
        let probe = Self {
            enabled: args.address_code_position_interaction_probe,
            prompt_id,
            group: args.address_code_position_group,
            primary_codes,
            secondary_codes,
            target_code: args.address_code_position_target_code,
        };
        if probe.enabled {
            probe.validate(targets)?;
        }
        Ok(probe)
    }

    fn validate(&self, targets: &ProbeTargets<'_>) -> ProbeResult {
        if self.prompt_id.is_empty() {
            return Err("--address-code-position-interaction-probe requires --address-code-position-prompt-id".into());
        }
        if self.primary_codes.is_empty() {
            return Err(
                "--address-code-position-primary-codes must include at least one code".into(),
            );
        }
        if self.secondary_codes.is_empty() {
            return Err(
                "--address-code-position-secondary-codes must include at least one code".into(),
            );
        }
        for config in targets.configs {
            let levels = levels(config);
            if self.group >= config.groups {
                return Err(format!(
                    "--address-code-position-group is {}, but config {:?} has only {} groups",
                    self.group, config, config.groups
                )
                .into());
            }
            if self.target_code >= levels {
                return Err(format!(
                    "--address-code-position-target-code is {}, but config {:?} has only {levels} levels",
                    self.target_code, config
                )
                .into());
            }
            for &code in self.primary_codes.iter().chain(self.secondary_codes.iter()) {
                if code >= levels {
                    return Err(format!(
                        "--address-code-position primary/secondary code {code} exceeds config {:?} with {levels} levels",
                        config
                    )
                    .into());
                }
            }
        }
        Ok(())
    }

    fn positions_with(&self, ctx: &EvalContext<'_>, codes: &[usize]) -> Vec<usize> {
        ctx.oracle_codes_by_position
            .iter()
            .enumerate()
            .filter_map(|(pos, oracle)| codes.contains(&oracle[self.group]).then_some(pos))
            .collect()
    }

    /// Rewrite `changed_positions` to the target code and record the result;
    /// an empty set is skipped.
    fn emit_variant(
        &self,
        ctx: &mut EvalContext<'_>,
        variant_name: String,
        mut changed_positions: Vec<usize>,
    ) -> ProbeResult {
        changed_positions.sort_unstable();
        changed_positions.dedup();
        if changed_positions.is_empty() {
            return Ok(());
        }
        let mode_d_table = ctx.mode_d_table("code position-interaction probe")?;
        let (group, target_code) = (self.group, self.target_code);
        let predicted_codes_by_position = ctx
            .oracle_codes_by_position
            .iter()
            .enumerate()
            .map(|(pos, oracle_codes)| {
                let mut codes = oracle_codes.clone();
                if changed_positions.binary_search(&pos).is_ok() {
                    codes[group] = target_code;
                }
                codes
            })
            .collect::<Vec<_>>();
        let positions_label = changed_positions
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("+");
        let selected_group_keys = ctx.group_keys(|candidate_group| {
            (candidate_group == group)
                .then(|| format!("{variant_name}_positions_{positions_label}"))
        });
        ctx.evaluate_probe(
            mode_d_table,
            &predicted_codes_by_position,
            &format!("pos_interaction_g{group}_{variant_name}_to{target_code}_oracle_rest"),
            &selected_group_keys,
        )
    }
}

impl AddressProbe for CodePositionProbe {
    fn enabled(&self) -> bool {
        self.enabled
    }

    fn evaluate(&self, ctx: &mut EvalContext<'_>) -> ProbeResult {
        if ctx.label != self.prompt_id.as_str() {
            return Ok(());
        }
        ctx.mode_d_table("code position-interaction probe")?;
        let primary = self.positions_with(ctx, &self.primary_codes);
        let secondary = self.positions_with(ctx, &self.secondary_codes);

        self.emit_variant(ctx, "A0_all_primary".to_string(), primary.clone())?;
        self.emit_variant(ctx, "A1_all_secondary".to_string(), secondary.clone())?;
        let mut all_primary_secondary = primary.clone();
        all_primary_secondary.extend(secondary.iter().copied());
        self.emit_variant(
            ctx,
            "A2_all_primary_all_secondary".to_string(),
            all_primary_secondary,
        )?;
        for (idx, &secondary_pos) in secondary.iter().enumerate() {
            let mut changed = primary.clone();
            changed.push(secondary_pos);
            self.emit_variant(
                ctx,
                format!(
                    "A{}_all_primary_secondary_pos{secondary_pos}",
                    idx + FIRST_PER_POSITION_VARIANT
                ),
                changed,
            )?;
        }
        let leave_one_offset = FIRST_PER_POSITION_VARIANT + secondary.len();
        for (idx, &secondary_pos) in secondary.iter().enumerate() {
            let mut changed = primary.clone();
            changed.extend(
                secondary
                    .iter()
                    .copied()
                    .filter(|pos| *pos != secondary_pos),
            );
            self.emit_variant(
                ctx,
                format!(
                    "A{}_all_primary_all_secondary_except_pos{secondary_pos}",
                    leave_one_offset + idx
                ),
                changed,
            )?;
        }
        for &primary_pos in &primary {
            let mut changed = secondary.clone();
            changed.push(primary_pos);
            self.emit_variant(
                ctx,
                format!("all_secondary_primary_pos{primary_pos}"),
                changed,
            )?;
        }
        for &primary_pos in &primary {
            let mut changed = secondary.clone();
            changed.extend(primary.iter().copied().filter(|pos| *pos != primary_pos));
            self.emit_variant(
                ctx,
                format!("all_primary_except_pos{primary_pos}_all_secondary"),
                changed,
            )?;
        }
        Ok(())
    }

    fn write_report(&self, report: &mut OraclePqReport) {
        let enabled = self.enabled;
        report.address_code_position_interaction_probe = enabled;
        report.address_code_position_prompt_id = when(enabled, self.prompt_id.clone());
        report.address_code_position_group = when(enabled, self.group);
        report.address_code_position_primary_codes = when(enabled, self.primary_codes.clone());
        report.address_code_position_secondary_codes = when(enabled, self.secondary_codes.clone());
        report.address_code_position_target_code = when(enabled, self.target_code);
    }
}
