//! `ov-rd oracle-pq`: the orchestrator. Setup (load, parse, fit the base
//! compression), fit the probe registry, evaluate every held-out prompt,
//! then write the report.

use std::collections::HashMap;

use crate::commands::dev::ov_rd::oracle_pq_address::fit_majority_codes_for_codebooks;

use super::args::OraclePqArgs;
use super::eval_loop::{evaluate_prompts, EvalInputs};
use super::probe::{FitContext, HeadConfigMap, ProbeResult, ProbeTargets};
use super::registry::{check_unfitted_mode_d_prerequisites, ProbeRegistry};
use super::report::{write_report, ReportInputs};
use super::setup::{fit_base, fit_context, load_model, load_prompt_split, parse_targets};

pub(in super::super) fn run_oracle_pq(args: OraclePqArgs) -> ProbeResult {
    std::fs::create_dir_all(&args.out)?;
    let mut model = load_model(&args)?;
    let (heads, configs) = parse_targets(&args)?;
    let mut registry = ProbeRegistry::parse(
        &args,
        &ProbeTargets {
            configs: &configs,
            heads: &heads,
            num_layers: model.weights.num_layers,
            hidden_size: model.weights.hidden_size,
        },
    )?;
    let prompts = load_prompt_split(&args, &heads, &configs)?;
    let base = fit_base(
        &mut model,
        &args,
        &heads,
        &configs,
        &prompts.fit,
        &registry.stratum_conditioned_groups,
    )?;

    let mut fit = fit_context(&mut model, &base, &heads, &prompts.fit, args.mode_d_check);
    for probe in &mut registry.probes {
        probe.fit(&mut fit)?;
    }
    check_unfitted_mode_d_prerequisites(&args)?;
    let majority_codes = fit_majority_codes(&mut fit, registry.needs_majority_codes())?;
    let code_stability = registry.stability.measure(&mut fit, &prompts.eval)?;
    registry.occurrences.export(
        &mut fit,
        (&prompts.fit, &prompts.eval, &prompts.all),
        &args.out,
    )?;

    let accumulators = evaluate_prompts(
        &mut model,
        &EvalInputs {
            heads: &heads,
            configs: &configs,
            base: &base,
            registry: &registry,
            majority_codes: &majority_codes,
            mode_d_check: args.mode_d_check,
        },
        &prompts.eval,
    )?;
    write_report(
        ReportInputs {
            args: &args,
            heads,
            configs,
            hidden_size: model.weights.hidden_size,
            prompts: &prompts,
            base: &base,
            registry: &registry,
            code_stability: &code_stability,
        },
        accumulators,
    )
}

fn fit_majority_codes(
    ctx: &mut FitContext<'_>,
    needed: bool,
) -> ProbeResult<HeadConfigMap<Vec<usize>>> {
    if !needed {
        return Ok(HashMap::new());
    }
    eprintln!("Fitting per-group majority codes for address diagnostics");
    fit_majority_codes_for_codebooks(
        ctx.weights,
        ctx.index,
        ctx.tokenizer,
        ctx.fit_prompts,
        ctx.heads,
        ctx.bases,
        ctx.means,
        ctx.pca_bases,
        ctx.codebooks,
    )
}
