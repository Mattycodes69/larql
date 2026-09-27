//! Execution-surface and unresolved-interface findings, and resolved plans.

use super::super::graph::BuiltGraph;
use larql_models::inventory::ArchitectureInventory;

#[allow(unused_imports)]
use super::*;

/// Execution-surface verdict per component this artifact sourced: a
/// representable finding when the surface is complete, a blocking one
/// itemising the missing source facts when it is not (V3-G5a). An
/// executor with a partial surface would have to default, which G5
/// forbids — so incompleteness refuses conversion up front.
pub(super) fn execution_surface_findings(artifact: &str, built: &BuiltGraph) -> Vec<Finding> {
    let mut findings: Vec<Finding> = built
        .graph
        .components
        .iter()
        .filter(|c| c.source_artifact == artifact && c.execution.is_some())
        .map(|component| {
            // Name the groups that are actually present — presence follows
            // the program (schema 6), so the sentence must too. Listing
            // absent groups as complete was the fabrication, restated.
            let surface = component.execution.as_ref().expect("filtered above");
            let mut groups: Vec<&str> = Vec::new();
            if surface.attention.is_some() {
                groups.push("attention");
            }
            if surface.mamba2.is_some() {
                groups.push("mamba2 mixer");
            }
            if surface.ffn.is_some() {
                groups.push("ffn");
            }
            groups.push("norm");
            if surface.head.is_some() {
                groups.push("head");
            }
            // A complete surface is not an executable one, and the report
            // must SAY what cannot run — otherwise a row reads as "every
            // declaration has a home" while nothing executes, the
            // looks-supported failure the whole instrument exists to
            // catch. Three facts can refuse here, asked most specific
            // first, and each is read from the SAME authority the
            // executor's preparation step refuses on, so a whole-stack
            // image the report calls executable is one the executor
            // prepares.
            //
            // First: a component declaring the attention-residual
            // topology and shipping no exit object. The exit reduction is
            // part of what the declaration means — the stack's last layer
            // leaves a prefix sum and a history, and something has to
            // collapse them before the final norm — so its absence is a
            // sharper fact than the missing traversal below, and is
            // reported instead of it. The analogue of the head's case for
            // the bundle.
            let attention_residual_exit_missing = matches!(
                surface.residual_topology,
                larql_models::config::ResidualTopology::AttentionResidual { .. }
            ) && !built.graph.objects.iter().any(|o| {
                o.component == component.id
                    && matches!(
                        o.kind,
                        super::super::graph::ObjectKind::AttentionResidualExit
                    )
            });
            // **There is no longer a "cannot traverse this topology"
            // refusal here, and its absence is deliberate.** It read
            // `ResidualTopology::unimplemented_reason`, which is deleted:
            // single-stream always lowered, hyper-connections joined it in
            // wave 19, and attention residuals join it in K3-ATTNRES-1
            // now that the decode (2a) and batch (2b) traversals have each
            // been witnessed against a Torch oracle. A topology that
            // cannot be traversed again brings both the authority and this
            // reader back together, which is the contract the deleted
            // function stated for itself twice.
            //
            // The two refusals that remain are NOT about traversal, and
            // retiring the third must not quietly retire them: they are
            // missing DECLARED objects, and a component that ships no exit
            // pair or no head is blocked whatever this build can traverse.
            //
            // Second: a hyper-connected component with no head object
            // (GLM-5.3-Flash: `mhc` unexplained, no `hc_head_*` shipped)
            // runs layer by layer but has no declared reduction from the
            // bundle to the one vector the final norm reads, and this
            // build does not invent one. A count that dropped here when
            // the topology lifted would be capability granted past that
            // boundary.
            let headless = matches!(
                surface.residual_topology,
                larql_models::config::ResidualTopology::HyperConnection(_)
            ) && !built.graph.objects.iter().any(|o| {
                o.component == component.id
                    && matches!(o.kind, super::super::graph::ObjectKind::HyperConnectionHead)
            });
            let refusal = if attention_residual_exit_missing {
                Some(format!(
                    "{:?} is represented and its per-layer operands are addressed, but the \
                     component declares no attention_residual_exit object: the topology's own \
                     reduction from the snapshot history to the one vector the final norm \
                     reads is required by the declaration, and this build does not invent one",
                    surface.residual_topology
                ))
            } else if headless {
                Some(format!(
                    "{:?} is executable layer by layer, but the component declares no \
                     hyper_connection_head object: a whole-stack execution has no declared \
                     reduction from the bundle to the vector the final norm reads, and this \
                     build does not invent one (a layer-range image runs without a head)",
                    surface.residual_topology
                ))
            } else {
                None
            };
            match refusal {
                Some(what) => Finding {
                    category: FindingCategory::Unrepresented,
                    class: SemanticClass::UnsupportedComponent,
                    component: component.id.clone(),
                    subject: format!("{}.execution_surface", component.id),
                    declared: None,
                    resolved: None,
                    carriage: None,
                    detail: format!(
                        "execution surface complete ({}), and {what}",
                        groups.join(", ")
                    ),
                },
                None => Finding {
                    category: FindingCategory::Representable,
                    class: SemanticClass::ExecutionSemantic,
                    component: component.id.clone(),
                    subject: format!("{}.execution_surface", component.id),
                    declared: None,
                    resolved: None,
                    carriage: None,
                    detail: format!("execution surface complete ({})", groups.join(", ")),
                },
            }
        })
        .collect();
    findings.extend(
        built
            .incomplete_surfaces
            .iter()
            .filter(|s| s.artifact == artifact)
            .map(|s| Finding {
                category: FindingCategory::Unrepresented,
                class: SemanticClass::ExecutionSemantic,
                component: s.component.clone(),
                subject: format!("{}.execution_surface", s.component),
                declared: None,
                resolved: None,
                carriage: None,
                detail: format!(
                    "execution surface incomplete — missing: {}",
                    s.missing.join(", ")
                ),
            }),
    );
    findings
}

/// Blocking finding per interface the builder could not resolve.
pub(super) fn unresolved_interface_findings(artifact: &str, built: &BuiltGraph) -> Vec<Finding> {
    built
        .unresolved_interfaces
        .iter()
        .filter(|u| u.artifact == artifact)
        .map(|u| Finding {
            category: FindingCategory::Interface,
            class: SemanticClass::InterfaceSemantic,
            component: String::new(),
            subject: "hidden_state_interface".to_string(),
            declared: None,
            resolved: None,
            carriage: None,
            detail: u.reason.clone(),
        })
        .collect()
}

// ── The plan-by-source sequence, in one place ────────────────────────

/// Plan artifacts the caller has already resolved.
///
/// `specs[i]` is the argument the caller was given for `resolved[i]` —
/// the string the user typed, which is what the verdict should name, not
/// the local directory an `hf://` reference happened to stage into.
///
/// This exists because three front doors were carrying their own copy of
/// the same fifteen lines (`vindex plan`, `larql vindex3 plan`, and
/// `POST /v1/plan`): resolve, name each artifact's source with the commit
/// its facts were read at, then plan. A verdict that differs by which
/// door asked for it is not a verdict, so the sequence is one function
/// and the doors differ only in how they report it.
///
/// Resolution stays with the caller: it is the step that can be slow and
/// can fail, and each door reports staging differently — the CLIs to
/// stderr as it happens, the server as a field in its response.
pub fn plan_resolved(
    specs: &[std::path::PathBuf],
    resolved: Vec<super::super::artifact::ResolvedArtifact>,
) -> Result<SystemPlan, crate::error::VindexError> {
    if specs.len() != resolved.len() {
        return Err(crate::error::VindexError::Parse(format!(
            "{} spec(s) but {} resolved artifact(s): every artifact needs exactly one spec",
            specs.len(),
            resolved.len()
        )));
    }
    // The verdict names its subject: the argument as given and, for a
    // repo, the commit the facts were read at.
    let sources: Vec<ArtifactSource> = specs
        .iter()
        .zip(&resolved)
        .map(|(spec, a)| ArtifactSource {
            path: spec.display().to_string(),
            revision: a.commit().map(str::to_string),
            unpinned_revision: a.unpinned_revision().map(str::to_string),
        })
        .collect();
    let named: Vec<(String, ArchitectureInventory)> = resolved
        .into_iter()
        .map(|a| (a.name, a.inventory))
        .collect();
    plan_system_with_sources(&named, &sources)
}
