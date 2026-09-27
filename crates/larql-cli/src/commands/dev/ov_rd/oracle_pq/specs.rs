//! Probe-spec parsing: code-class collapses, quotient guards and substitutions.

use super::super::address::attention_argmax;

use crate::commands::dev::ov_rd::reports::AddressProbePromptReport;

pub(super) fn parse_string_list(spec: &str) -> Vec<String> {
    spec.split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(ToString::to_string)
        .collect()
}

pub(super) fn oracle_mode_d_address_report(
    label: &str,
    stratum: &str,
    positions: usize,
    groups: usize,
    kl: f64,
    top1_agree: bool,
    baseline_top1_in_predicted_top5: bool,
) -> AddressProbePromptReport {
    AddressProbePromptReport {
        id: label.to_string(),
        stratum: stratum.to_string(),
        kl,
        positions,
        groups_correct: positions * groups,
        groups_total: positions * groups,
        exact_address_match: true,
        top1_agree,
        baseline_top1_in_predicted_top5,
    }
}

#[derive(Debug, Clone)]
pub(super) struct CodeClassCollapseSpec {
    pub(super) name: String,
    pub(super) mappings: Vec<CodeClassCollapseMapping>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ConditionalQuotientGuard {
    EarlyProsePosition,
    EarlyProseBosPrev,
    ProseBosPrev,
}

impl ConditionalQuotientGuard {
    pub(super) fn parse(raw: &str) -> Option<Self> {
        match raw.trim() {
            "early_prose_position" | "E_early_prose_position_guard" => {
                Some(ConditionalQuotientGuard::EarlyProsePosition)
            }
            "early_prose_bos_prev" | "F_early_prose_bos_prev_guard" => {
                Some(ConditionalQuotientGuard::EarlyProseBosPrev)
            }
            "prose_bos_prev" | "G_prose_bos_prev_guard" => {
                Some(ConditionalQuotientGuard::ProseBosPrev)
            }
            _ => None,
        }
    }

    pub(super) fn label(self) -> &'static str {
        match self {
            ConditionalQuotientGuard::EarlyProsePosition => "E_early_prose_position_guard",
            ConditionalQuotientGuard::EarlyProseBosPrev => "F_early_prose_bos_prev_guard",
            ConditionalQuotientGuard::ProseBosPrev => "G_prose_bos_prev_guard",
        }
    }

    pub(super) fn keeps_secondary_oracle(
        self,
        stratum: &str,
        pos: usize,
        early_position_max: usize,
        attention_weights: &[f32],
    ) -> bool {
        if stratum != "natural_prose" {
            return false;
        }
        let is_early = pos <= early_position_max;
        match self {
            ConditionalQuotientGuard::EarlyProsePosition => is_early,
            ConditionalQuotientGuard::EarlyProseBosPrev => {
                is_early && is_bos_or_previous_attention(pos, attention_weights)
            }
            ConditionalQuotientGuard::ProseBosPrev => {
                is_bos_or_previous_attention(pos, attention_weights)
            }
        }
    }
}

pub(super) fn is_bos_or_previous_attention(pos: usize, attention_weights: &[f32]) -> bool {
    if attention_weights.is_empty() {
        return false;
    }
    let source = attention_argmax(attention_weights, pos);
    source == 0 || (pos > 0 && source + 1 == pos)
}

impl CodeClassCollapseSpec {
    pub(super) fn label(&self) -> String {
        format!("{}={}", self.name, self.mapping_label())
    }

    pub(super) fn mapping_label(&self) -> String {
        self.mappings
            .iter()
            .map(|mapping| {
                let sources = mapping
                    .sources
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join("+");
                format!("{sources}:{}", mapping.target)
            })
            .collect::<Vec<_>>()
            .join("|")
    }

    pub(super) fn mapping_label_or_base(&self) -> String {
        if self.mappings.is_empty() {
            "base".to_string()
        } else {
            self.mapping_label()
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct CodeClassCollapseMapping {
    pub(super) sources: Vec<usize>,
    pub(super) target: usize,
}

pub(super) fn parse_code_class_collapse_specs(
    spec: &str,
) -> Result<Vec<CodeClassCollapseSpec>, Box<dyn std::error::Error>> {
    let mut out = Vec::new();
    for (idx, raw_spec) in spec
        .split(';')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .enumerate()
    {
        let (raw_name, raw_mappings) = raw_spec
            .split_once('=')
            .map(|(name, mappings)| (name.trim(), mappings.trim()))
            .unwrap_or(("", raw_spec));
        let mappings = parse_code_class_collapse_mappings(raw_mappings)?;
        let fallback_name = sanitize_probe_name(
            &mappings
                .iter()
                .map(|mapping| {
                    let sources = mapping
                        .sources
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join("+");
                    format!("{sources}_to_{}", mapping.target)
                })
                .collect::<Vec<_>>()
                .join("_and_"),
        );
        let name = if raw_name.is_empty() {
            format!("collapse{idx}_{fallback_name}")
        } else {
            sanitize_probe_name(raw_name)
        };
        if name.is_empty() {
            return Err(format!("invalid empty class-collapse name in spec {raw_spec:?}").into());
        }
        out.push(CodeClassCollapseSpec { name, mappings });
    }
    Ok(out)
}

pub(super) fn parse_conditional_quotient_guards(
    spec: &str,
) -> Result<Vec<ConditionalQuotientGuard>, Box<dyn std::error::Error>> {
    let mut out = Vec::new();
    for raw in spec
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
    {
        let guard = ConditionalQuotientGuard::parse(raw).ok_or_else(|| {
            format!(
                "unsupported conditional quotient guard {raw:?}; expected early_prose_position, early_prose_bos_prev, or prose_bos_prev"
            )
        })?;
        if !out.contains(&guard) {
            out.push(guard);
        }
    }
    Ok(out)
}

pub(super) fn parse_code_class_collapse_mappings(
    spec: &str,
) -> Result<Vec<CodeClassCollapseMapping>, Box<dyn std::error::Error>> {
    let mut mappings = Vec::new();
    let mut seen_sources = Vec::new();
    for raw_mapping in spec
        .split('|')
        .map(str::trim)
        .filter(|part| !part.is_empty())
    {
        let (raw_sources, raw_target) = raw_mapping.split_once(':').ok_or_else(|| {
            format!("invalid class-collapse mapping {raw_mapping:?}; expected sources:target")
        })?;
        let mut sources = Vec::new();
        for part in raw_sources
            .split('+')
            .map(str::trim)
            .filter(|part| !part.is_empty())
        {
            sources
                .push(part.parse::<usize>().map_err(|err| {
                    format!("invalid class-collapse source code {part:?}: {err}")
                })?);
        }
        sources.sort_unstable();
        sources.dedup();
        if sources.is_empty() {
            return Err(format!("class-collapse mapping {raw_mapping:?} has no sources").into());
        }
        for &source in &sources {
            if seen_sources.contains(&source) {
                return Err(format!(
                    "class-collapse source code {source} appears in more than one mapping"
                )
                .into());
            }
            seen_sources.push(source);
        }
        let target = raw_target.trim().parse::<usize>().map_err(|err| {
            format!(
                "invalid class-collapse target code {:?}: {err}",
                raw_target.trim()
            )
        })?;
        mappings.push(CodeClassCollapseMapping { sources, target });
    }
    if mappings.is_empty() {
        return Err(format!("class-collapse spec {spec:?} has no mappings").into());
    }
    Ok(mappings)
}

pub(super) fn sanitize_probe_name(name: &str) -> String {
    name.chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

#[derive(Debug, Clone, Copy)]
pub(super) enum CodeSubstitutionToSpec {
    Majority,
    Code(usize),
}

pub(super) fn parse_code_substitution_to_specs(
    spec: &str,
) -> Result<Vec<CodeSubstitutionToSpec>, Box<dyn std::error::Error>> {
    let mut out = Vec::new();
    for part in spec
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
    {
        if part.eq_ignore_ascii_case("majority") {
            out.push(CodeSubstitutionToSpec::Majority);
        } else {
            out.push(CodeSubstitutionToSpec::Code(
                part.parse::<usize>()
                    .map_err(|err| format!("invalid code substitution target {part:?}: {err}"))?,
            ));
        }
    }
    Ok(out)
}
