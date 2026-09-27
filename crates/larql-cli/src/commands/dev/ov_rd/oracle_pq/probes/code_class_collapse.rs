//! `--address-code-class-collapse-group-probe`: every selected group has
//! its oracle code mapped through a class-collapse spec at once.

use crate::commands::dev::ov_rd::reports::OraclePqReport;
use crate::commands::dev::ov_rd::types::PqConfig;

use super::super::args::OraclePqArgs;
use super::super::probe::{AddressProbe, EvalContext, ProbeResult, ProbeTargets};
use super::super::specs::{parse_code_class_collapse_specs, CodeClassCollapseSpec};
use super::super::validate::{check_groups, levels, parse_sorted, when};

/// Refuse a spec whose source or target code the config cannot express.
/// `kind` names the spec family in the error.
pub(super) fn check_collapse_specs(
    kind: &str,
    specs: &[CodeClassCollapseSpec],
    config: &PqConfig,
) -> ProbeResult {
    let levels = levels(config);
    for spec in specs {
        for mapping in &spec.mappings {
            if mapping.target >= levels {
                return Err(format!(
                    "{kind} {:?} targets code {}, but config {:?} has only {levels} levels",
                    spec.name, mapping.target, config
                )
                .into());
            }
            for &source in &mapping.sources {
                if source >= levels {
                    return Err(format!(
                        "{kind} {:?} includes source code {source}, but config {:?} has only {levels} levels",
                        spec.name, config
                    )
                    .into());
                }
            }
        }
    }
    Ok(())
}

pub(in super::super) struct CodeClassCollapseProbe {
    enabled: bool,
    groups: Vec<usize>,
    specs: Vec<CodeClassCollapseSpec>,
}

impl CodeClassCollapseProbe {
    pub(in super::super) fn parse(
        args: &OraclePqArgs,
        targets: &ProbeTargets<'_>,
    ) -> ProbeResult<Self> {
        let groups = parse_sorted(&args.address_code_class_collapse_groups)?;
        let specs = parse_code_class_collapse_specs(&args.address_code_class_collapse_specs)?;
        if args.address_code_class_collapse_group_probe {
            if groups.is_empty() {
                return Err("--address-code-class-collapse-group-probe requires at least one --address-code-class-collapse-groups value".into());
            }
            if specs.is_empty() {
                return Err(
                    "--address-code-class-collapse-specs must include at least one spec".into(),
                );
            }
            for config in targets.configs {
                check_groups("--address-code-class-collapse-groups", &groups, config)?;
                check_collapse_specs("class-collapse spec", &specs, config)?;
            }
        }
        Ok(Self {
            enabled: args.address_code_class_collapse_group_probe,
            groups,
            specs,
        })
    }
}

impl AddressProbe for CodeClassCollapseProbe {
    fn enabled(&self) -> bool {
        self.enabled
    }

    fn evaluate(&self, ctx: &mut EvalContext<'_>) -> ProbeResult {
        let mode_d_table = ctx.mode_d_table("code class-collapse probe")?;
        for collapse_spec in &self.specs {
            let predicted_codes_by_position = ctx
                .oracle_codes_by_position
                .iter()
                .map(|oracle_codes| {
                    let mut codes = oracle_codes.clone();
                    for &group in &self.groups {
                        for mapping in &collapse_spec.mappings {
                            if mapping.sources.contains(&oracle_codes[group]) {
                                codes[group] = mapping.target;
                                break;
                            }
                        }
                    }
                    codes
                })
                .collect::<Vec<_>>();
            let selected_group_keys = ctx.group_keys(|group| {
                self.groups
                    .contains(&group)
                    .then(|| collapse_spec.mapping_label())
            });
            ctx.evaluate_probe(
                mode_d_table,
                &predicted_codes_by_position,
                &format!(
                    "code_class_collapse_{}_groups_{:?}_oracle_rest",
                    collapse_spec.name, self.groups
                ),
                &selected_group_keys,
            )?;
        }
        Ok(())
    }

    fn write_report(&self, report: &mut OraclePqReport) {
        report.address_code_class_collapse_group_probe = self.enabled;
        report.address_code_class_collapse_groups = when(self.enabled, self.groups.clone());
        report.address_code_class_collapse_specs = when(
            self.enabled,
            self.specs
                .iter()
                .map(CodeClassCollapseSpec::label)
                .collect(),
        );
    }
}
