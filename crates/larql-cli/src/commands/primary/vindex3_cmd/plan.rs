//! `vindex3 plan`: the G1 semantic representability gate.

use larql_vindex::format::vindex3::plan::plan_resolved;

#[allow(unused_imports)]
use super::*;

pub(super) fn run_plan(args: PlanArgs) -> Result<(), Box<dyn std::error::Error>> {
    let resolved = artifact::resolve_all(&args.artifacts)?;
    for entry in &resolved {
        report_staging(entry);
    }
    let plan = plan_resolved(&args.artifacts, resolved)?;
    let json = serde_json::to_string_pretty(&plan)?;
    match &args.output {
        Some(path) => {
            std::fs::write(path, &json)?;
            eprintln!("plan written to {}", path.display());
        }
        None => println!("{json}"),
    }
    let summary = &plan.summary;
    eprintln!(
        "plan: {} representable, {} mismatched, {} unrepresented, {} interfaces — {} blocking",
        summary.representable,
        summary.mismatched,
        summary.unrepresented,
        summary.interfaces,
        summary.blocking,
    );
    if plan.admissible {
        Ok(())
    } else {
        Err(format!(
            "plan not admissible: {} blocking finding(s); \
             every one is a schema or resolution gap to close before conversion",
            summary.blocking
        )
        .into())
    }
}
