//! Every probe family, parsed and validated from the flags in the order
//! the flags have always been checked, then held in evaluation order.

use crate::commands::dev::ov_rd::reports::OraclePqReport;

use super::args::OraclePqArgs;
use super::diagnostics::{CodeOccurrences, CodeStability};
use super::probe::{AddressProbe, ProbeResult, ProbeTargets};
use super::probes::{
    AttentionClusterProbe, AttentionRelationProbe, Code7BosRuleProbe, Code7OracleBinaryProbe,
    CodeClassCollapseProbe, CodePositionProbe, CodeSubstitutionProbe, ConditionalQuotientProbe,
    CorruptionSweep, FfnFirstFeatureProbe, GammaProjectedProbe, GroupImportanceSweep,
    KeyGroupProbe, LshProbe, MajorityGroupProbe, PrevFfnFeatureProbe, ReducedQkClusterProbe,
    SupervisedProbe,
};
use super::validate::{check_groups_all, parse_sorted};

pub(super) struct ProbeRegistry {
    /// Every family, in per-prompt evaluation order.
    pub(super) probes: Vec<Box<dyn AddressProbe>>,
    pub(super) stability: CodeStability,
    pub(super) occurrences: CodeOccurrences,
    pub(super) stratum_conditioned_groups: Vec<usize>,
}

impl ProbeRegistry {
    pub(super) fn parse(args: &OraclePqArgs, targets: &ProbeTargets<'_>) -> ProbeResult<Self> {
        let key_group = KeyGroupProbe::parse(args, targets)?;
        let majority = MajorityGroupProbe::parse(args, targets)?;
        let code_substitution = CodeSubstitutionProbe::parse(args, targets)?;
        let class_collapse = CodeClassCollapseProbe::parse(args, targets)?;
        let code_position = CodePositionProbe::parse(args, targets)?;
        let conditional_quotient = ConditionalQuotientProbe::parse(args, targets)?;
        let occurrences = CodeOccurrences::parse(args, targets)?;
        let code7_bos = Code7BosRuleProbe::parse(args, targets)?;
        let code7_oracle = Code7OracleBinaryProbe::parse(args, targets)?;
        let lsh = LshProbe::parse(args, targets)?;
        let supervised = SupervisedProbe::parse(args, targets)?;
        let gamma = GammaProjectedProbe::parse(args, targets)?;
        let stability = CodeStability::parse(args, targets)?;
        let prev_ffn = PrevFfnFeatureProbe::parse(args, targets)?;
        let ffn_first = FfnFirstFeatureProbe::parse(args, targets)?;
        let attention_relation = AttentionRelationProbe::parse(args, targets)?;
        let attention_cluster = AttentionClusterProbe::parse(args, targets)?;
        let reduced_qk = ReducedQkClusterProbe::parse(args, targets)?;
        let stratum_conditioned_groups = parse_sorted(&args.stratum_conditioned_pq_groups)?;
        check_groups_all(
            "--stratum-conditioned-pq-groups",
            &stratum_conditioned_groups,
            targets.configs,
        )?;

        let probes: Vec<Box<dyn AddressProbe>> = vec![
            Box::new(key_group),
            Box::new(majority),
            Box::new(code_substitution),
            Box::new(class_collapse),
            Box::new(code_position),
            Box::new(conditional_quotient),
            Box::new(code7_bos),
            Box::new(code7_oracle),
            Box::new(GroupImportanceSweep::parse(args)),
            Box::new(lsh),
            Box::new(supervised),
            Box::new(gamma),
            Box::new(prev_ffn),
            Box::new(ffn_first),
            Box::new(attention_relation),
            Box::new(attention_cluster),
            Box::new(reduced_qk),
            Box::new(CorruptionSweep::parse(args)),
        ];
        Ok(Self {
            probes,
            stability,
            occurrences,
            stratum_conditioned_groups,
        })
    }

    pub(super) fn needs_majority_codes(&self) -> bool {
        self.probes.iter().any(|probe| probe.needs_majority_codes())
    }

    pub(super) fn write_report(&self, report: &mut OraclePqReport) {
        for probe in &self.probes {
            probe.write_report(report);
        }
        self.stability.write_report(report);
        report.stratum_conditioned_pq_groups = self.stratum_conditioned_groups.clone();
    }
}

/// The `--mode-d-check` prerequisite of the families without a fit (those
/// with one check it before fitting), in the order it has always been
/// reported.
pub(super) fn check_unfitted_mode_d_prerequisites(args: &OraclePqArgs) -> ProbeResult {
    if args.mode_d_check {
        return Ok(());
    }
    let prerequisites = [
        (args.address_corruption_sweep, "--address-corruption-sweep"),
        (args.address_group_importance, "--address-group-importance"),
        (
            args.address_majority_group_probe,
            "--address-majority-group-probe",
        ),
        (
            args.address_code_substitution_group_probe,
            "--address-code-substitution-group-probe",
        ),
        (
            args.address_code_class_collapse_group_probe,
            "--address-code-class-collapse-group-probe",
        ),
        (
            args.address_code_position_interaction_probe,
            "--address-code-position-interaction-probe",
        ),
        (
            args.address_code_conditional_quotient_group_probe,
            "--address-code-conditional-quotient-group-probe",
        ),
        (
            args.address_code7_bos_rule_group_probe,
            "--address-code7-bos-rule-group-probe",
        ),
        (
            args.address_code7_oracle_binary_group_probe,
            "--address-code7-oracle-binary-group-probe",
        ),
    ];
    match prerequisites.iter().find(|(enabled, _)| *enabled) {
        Some((_, flag)) => Err(format!("{flag} requires --mode-d-check").into()),
        None => Ok(()),
    }
}
