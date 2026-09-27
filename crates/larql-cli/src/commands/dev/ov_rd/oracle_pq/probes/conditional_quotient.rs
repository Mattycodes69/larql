//! `--address-code-conditional-quotient-group-probe`: primary codes map to
//! the target everywhere, secondary codes except where a guard keeps the
//! oracle code, optionally with extra mappings layered on top.

use crate::commands::dev::ov_rd::oracle_pq_forward::capture_attention_relation_rows;
use crate::commands::dev::ov_rd::reports::OraclePqReport;

use super::super::args::OraclePqArgs;
use super::super::probe::{row_at, AddressProbe, EvalContext, ProbeResult, ProbeTargets};
use super::super::specs::{
    parse_code_class_collapse_specs, parse_conditional_quotient_guards, CodeClassCollapseSpec,
    ConditionalQuotientGuard,
};
use super::super::validate::{levels, parse_sorted, when};
use super::code_class_collapse::check_collapse_specs;

/// Name of the implicit extra spec that adds no mappings.
const BASE_SPEC_NAME: &str = "base";

pub(in super::super) struct ConditionalQuotientProbe {
    enabled: bool,
    group: usize,
    primary_codes: Vec<usize>,
    secondary_codes: Vec<usize>,
    target_code: usize,
    early_position_max: usize,
    guards: Vec<ConditionalQuotientGuard>,
    extra_specs: Vec<CodeClassCollapseSpec>,
}

fn join_codes(codes: &[usize]) -> String {
    codes
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("+")
}

impl ConditionalQuotientProbe {
    pub(in super::super) fn parse(
        args: &OraclePqArgs,
        targets: &ProbeTargets<'_>,
    ) -> ProbeResult<Self> {
        let primary_codes = parse_sorted(&args.address_code_conditional_quotient_primary_codes)?;
        let secondary_codes =
            parse_sorted(&args.address_code_conditional_quotient_secondary_codes)?;
        let guards =
            parse_conditional_quotient_guards(&args.address_code_conditional_quotient_guards)?;
        let mut extra_specs =
            parse_code_class_collapse_specs(&args.address_code_conditional_quotient_extra_specs)?;
        extra_specs.insert(
            0,
            CodeClassCollapseSpec {
                name: BASE_SPEC_NAME.to_string(),
                mappings: Vec::new(),
            },
        );
        let probe = Self {
            enabled: args.address_code_conditional_quotient_group_probe,
            group: args.address_code_conditional_quotient_group,
            primary_codes,
            secondary_codes,
            target_code: args.address_code_conditional_quotient_target_code,
            early_position_max: args.address_code_conditional_quotient_early_position_max,
            guards,
            extra_specs,
        };
        if probe.enabled {
            probe.validate(targets)?;
        }
        Ok(probe)
    }

    fn validate(&self, targets: &ProbeTargets<'_>) -> ProbeResult {
        if self.primary_codes.is_empty() {
            return Err(
                "--address-code-conditional-quotient-primary-codes must include at least one code"
                    .into(),
            );
        }
        if self.secondary_codes.is_empty() {
            return Err("--address-code-conditional-quotient-secondary-codes must include at least one code".into());
        }
        if self.guards.is_empty() {
            return Err(
                "--address-code-conditional-quotient-guards must include at least one guard".into(),
            );
        }
        for config in targets.configs {
            let levels = levels(config);
            if self.group >= config.groups {
                return Err(format!(
                    "--address-code-conditional-quotient-group is {}, but config {:?} has only {} groups",
                    self.group, config, config.groups
                )
                .into());
            }
            if self.target_code >= levels {
                return Err(format!(
                    "--address-code-conditional-quotient-target-code is {}, but config {:?} has only {levels} levels",
                    self.target_code, config
                )
                .into());
            }
            for &code in self.primary_codes.iter().chain(self.secondary_codes.iter()) {
                if code >= levels {
                    return Err(format!(
                        "--address-code-conditional-quotient primary/secondary code {code} exceeds config {:?} with {levels} levels",
                        config
                    )
                    .into());
                }
            }
            check_collapse_specs("conditional quotient extra spec", &self.extra_specs, config)?;
        }
        Ok(())
    }

    fn predict(
        &self,
        ctx: &EvalContext<'_>,
        guard: ConditionalQuotientGuard,
        extra_spec: &CodeClassCollapseSpec,
        attention_rows: &[Vec<f32>],
    ) -> Vec<Vec<usize>> {
        let (group, target_code) = (self.group, self.target_code);
        ctx.oracle_codes_by_position
            .iter()
            .enumerate()
            .map(|(pos, oracle_codes)| {
                let mut codes = oracle_codes.clone();
                let group_code = oracle_codes[group];
                if self.primary_codes.contains(&group_code) {
                    codes[group] = target_code;
                } else if self.secondary_codes.contains(&group_code)
                    && !guard.keeps_secondary_oracle(
                        ctx.stratum,
                        pos,
                        self.early_position_max,
                        row_at(attention_rows, pos),
                    )
                {
                    codes[group] = target_code;
                }
                for mapping in &extra_spec.mappings {
                    if mapping.sources.contains(&group_code) {
                        codes[group] = mapping.target;
                        break;
                    }
                }
                codes
            })
            .collect()
    }
}

impl AddressProbe for ConditionalQuotientProbe {
    fn enabled(&self) -> bool {
        self.enabled
    }

    fn evaluate(&self, ctx: &mut EvalContext<'_>) -> ProbeResult {
        let mode_d_table = ctx.mode_d_table("code conditional-quotient probe")?;
        let (group, target_code) = (self.group, self.target_code);
        let attention_rows =
            capture_attention_relation_rows(ctx.weights, ctx.token_ids, ctx.index, ctx.head)?;
        for &guard in &self.guards {
            for extra_spec in &self.extra_specs {
                let predicted_codes_by_position =
                    self.predict(ctx, guard, extra_spec, &attention_rows);
                let selected_group_keys = ctx.group_keys(|candidate_group| {
                    (candidate_group == group).then(|| {
                        format!(
                            "{}_primary{}_secondary{}_to{}_extra{}",
                            guard.label(),
                            join_codes(&self.primary_codes),
                            join_codes(&self.secondary_codes),
                            target_code,
                            extra_spec.mapping_label_or_base()
                        )
                    })
                });
                ctx.evaluate_probe(
                    mode_d_table,
                    &predicted_codes_by_position,
                    &format!(
                        "code_conditional_quotient_g{group}_{}_extra{}_to{target_code}_oracle_rest",
                        guard.label(),
                        extra_spec.name
                    ),
                    &selected_group_keys,
                )?;
            }
        }
        Ok(())
    }

    fn write_report(&self, report: &mut OraclePqReport) {
        let enabled = self.enabled;
        report.address_code_conditional_quotient_group_probe = enabled;
        report.address_code_conditional_quotient_group = when(enabled, self.group);
        report.address_code_conditional_quotient_primary_codes =
            when(enabled, self.primary_codes.clone());
        report.address_code_conditional_quotient_secondary_codes =
            when(enabled, self.secondary_codes.clone());
        report.address_code_conditional_quotient_target_code = when(enabled, self.target_code);
        report.address_code_conditional_quotient_early_position_max =
            when(enabled, self.early_position_max);
        report.address_code_conditional_quotient_guards = when(
            enabled,
            self.guards
                .iter()
                .map(|guard| guard.label().to_string())
                .collect(),
        );
        report.address_code_conditional_quotient_extra_specs = when(
            enabled,
            self.extra_specs
                .iter()
                .map(CodeClassCollapseSpec::label)
                .collect(),
        );
    }
}
