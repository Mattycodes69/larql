//! `--address-code-substitution-group-probe`: positions whose oracle code
//! in a selected group is a from-code are rewritten to each to-code, with
//! every other group and position left oracle-correct.

use crate::commands::dev::ov_rd::reports::OraclePqReport;

use super::super::args::OraclePqArgs;
use super::super::probe::{AddressProbe, EvalContext, ProbeResult, ProbeTargets};
use super::super::specs::{
    oracle_mode_d_address_report, parse_code_substitution_to_specs, CodeSubstitutionToSpec,
};
use super::super::validate::{check_codes, check_groups, levels, parse_sorted, when};

/// Report label of the `majority` to-spec.
const MAJORITY_SPEC_LABEL: &str = "majority";

pub(in super::super) struct CodeSubstitutionProbe {
    enabled: bool,
    groups: Vec<usize>,
    from_codes: Vec<usize>,
    to_specs: Vec<CodeSubstitutionToSpec>,
}

impl CodeSubstitutionProbe {
    pub(in super::super) fn parse(
        args: &OraclePqArgs,
        targets: &ProbeTargets<'_>,
    ) -> ProbeResult<Self> {
        let groups = parse_sorted(&args.address_code_substitution_groups)?;
        let from_codes = parse_sorted(&args.address_code_substitution_from_codes)?;
        let to_specs = parse_code_substitution_to_specs(&args.address_code_substitution_to_codes)?;
        if args.address_code_substitution_group_probe {
            if groups.is_empty() {
                return Err("--address-code-substitution-group-probe requires at least one --address-code-substitution-groups value".into());
            }
            if to_specs.is_empty() {
                return Err("--address-code-substitution-group-probe requires at least one --address-code-substitution-to-codes value".into());
            }
            let to_codes = to_specs
                .iter()
                .filter_map(|spec| match spec {
                    CodeSubstitutionToSpec::Code(code) => Some(*code),
                    CodeSubstitutionToSpec::Majority => None,
                })
                .collect::<Vec<_>>();
            for config in targets.configs {
                check_groups("--address-code-substitution-groups", &groups, config)?;
                check_codes(
                    "--address-code-substitution-from-codes",
                    &from_codes,
                    config,
                )?;
                check_codes("--address-code-substitution-to-codes", &to_codes, config)?;
            }
        }
        Ok(Self {
            enabled: args.address_code_substitution_group_probe,
            groups,
            from_codes,
            to_specs,
        })
    }
}

impl AddressProbe for CodeSubstitutionProbe {
    fn enabled(&self) -> bool {
        self.enabled
    }

    fn evaluate(&self, ctx: &mut EvalContext<'_>) -> ProbeResult {
        let mode_d_table = ctx.mode_d_table("code substitution probe")?;
        let group_majority = ctx.majority("code substitution probe")?;
        let from_codes = if self.from_codes.is_empty() {
            (0..levels(&ctx.config)).collect::<Vec<_>>()
        } else {
            self.from_codes.clone()
        };
        for &group in &self.groups {
            for &from_code in &from_codes {
                let source_code_present = ctx
                    .oracle_codes_by_position
                    .iter()
                    .any(|codes| codes[group] == from_code);
                for to_spec in &self.to_specs {
                    let to_code = match *to_spec {
                        CodeSubstitutionToSpec::Majority => group_majority[group],
                        CodeSubstitutionToSpec::Code(code) => code,
                    };
                    if to_code == from_code {
                        continue;
                    }
                    let predicted_codes_by_position = ctx
                        .oracle_codes_by_position
                        .iter()
                        .map(|oracle_codes| {
                            let mut codes = oracle_codes.clone();
                            if codes[group] == from_code {
                                codes[group] = to_code;
                            }
                            codes
                        })
                        .collect::<Vec<_>>();
                    let prompt_report = if source_code_present {
                        ctx.evaluate(mode_d_table, &predicted_codes_by_position)?
                    } else {
                        oracle_mode_d_address_report(
                            ctx.label,
                            ctx.stratum,
                            ctx.token_ids.len(),
                            ctx.config.groups,
                            ctx.oracle_mode_d.kl,
                            ctx.oracle_mode_d.top1_agree,
                            ctx.oracle_mode_d.baseline_top1_in_top5,
                        )
                    };
                    let to_label = match *to_spec {
                        CodeSubstitutionToSpec::Majority => {
                            format!("{MAJORITY_SPEC_LABEL}{}", group_majority[group])
                        }
                        CodeSubstitutionToSpec::Code(code) => code.to_string(),
                    };
                    let selected_group_keys = ctx.group_keys(|candidate_group| {
                        (candidate_group == group).then(|| format!("from{from_code}_to{to_label}"))
                    });
                    ctx.accumulator.add_address_probe(
                        &format!("code_subst_g{group}_from{from_code}_to{to_label}_oracle_rest"),
                        &selected_group_keys,
                        prompt_report,
                    );
                }
            }
        }
        Ok(())
    }

    fn write_report(&self, report: &mut OraclePqReport) {
        report.address_code_substitution_group_probe = self.enabled;
        report.address_code_substitution_groups = when(self.enabled, self.groups.clone());
        report.address_code_substitution_from_codes = when(self.enabled, self.from_codes.clone());
        report.address_code_substitution_to_codes = when(
            self.enabled,
            self.to_specs
                .iter()
                .map(|spec| match spec {
                    CodeSubstitutionToSpec::Majority => MAJORITY_SPEC_LABEL.to_string(),
                    CodeSubstitutionToSpec::Code(code) => code.to_string(),
                })
                .collect(),
        );
    }
}
