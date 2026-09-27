//! Markdown and JSON renderings of the audit.

use std::path::Path;

#[allow(unused_imports)]
use super::*;

pub(super) fn render_markdown(model: &str, vindex: &Path, runs: &[PathRun]) -> String {
    let mut s = String::new();
    s.push_str("# walk_path_audit\n\n");
    s.push_str(&format!("**Model:** `{}`  \n", model));
    s.push_str(&format!("**Vindex:** `{}`  \n", vindex.display()));
    s.push_str(&format!("**Prompts:** {}\n\n", PROMPTS.len()));
    s.push_str(
        "**Metrics.** Assertion: `min cos`, `max rel L2 = L2 / ‖primary‖` — both \
         magnitude-invariant. Diagnostic: `max abs L2`, `max|Δ|` — vary with residual \
         magnitude, included for triage of outlier observations (e.g. residual-norm \
         spikes at specific (layer, token) pairs).\n\n",
    );

    // Summary table
    s.push_str("## Summary\n\n");
    s.push_str(
        "| path | bound | min cos (assert) | max rel L2 (assert) | top-1 ok | Paris ΔP | max abs L2 (diag) | worst rel-L2 layer | worst rel-L2 prompt | verdict |\n",
    );
    s.push_str("|---|---|---|---|---|---|---|---|---|---|\n");
    for r in runs {
        let top1_ok = if r.per_prompt.values().all(|p| p.top1_match) {
            "✓".to_string()
        } else {
            let bad: Vec<_> = r
                .per_prompt
                .iter()
                .filter(|(_, p)| !p.top1_match)
                .map(|(k, _)| k.as_str())
                .collect();
            format!("✗ ({})", bad.join(","))
        };
        let paris_delta = r
            .per_prompt
            .get(PARIS_KEY)
            .map(|p| format!("{:.3e}", p.prob_delta))
            .unwrap_or_else(|| "—".to_string());
        let verdict = if r.pass { "PASS" } else { "FAIL" };
        let bound = r.bound.expect("bound populated for all real runs");
        s.push_str(&format!(
            "| `{}` | {} (cos≥{:.5}, rel_L2≤{:.0e}) | {:.6} | {:.3e} | {} | {} | {:.3e} | {} | {} | **{}** |\n",
            r.name,
            bound.kind,
            bound.min_cos,
            bound.rel_l2,
            r.path_min_cos,
            r.path_max_rel_l2,
            top1_ok,
            paris_delta,
            r.path_max_l2,
            r.path_worst_rel_l2_layer,
            r.path_worst_rel_l2_prompt,
            verdict,
        ));
    }
    s.push('\n');

    // Per-path detail
    for r in runs {
        let bound = r.bound.expect("bound populated for all real runs");
        s.push_str(&format!("## `{}`\n\n", r.name));
        s.push_str(&format!(
            "**Mask:** fp4={} q4={} interleaved={} full_mmap={} q4k={} down_features={}  \n",
            r.mask.hide_fp4,
            r.mask.hide_q4,
            r.mask.hide_interleaved,
            r.mask.hide_full_mmap,
            r.mask.hide_q4k,
            r.mask.hide_down_features,
        ));
        s.push_str(&format!(
            "**Sparse K:** {}  \n",
            r.sparse_k
                .map(|k| if k == usize::MAX {
                    "MAX".to_string()
                } else {
                    k.to_string()
                })
                .unwrap_or_else(|| "—".to_string())
        ));
        s.push_str(&format!(
            "**Bound ({}):** cos ≥ {:.5}, rel_L2 ≤ {:.0e}  \n",
            bound.kind, bound.min_cos, bound.rel_l2,
        ));
        s.push_str(&format!(
            "**Assertion aggregate:** min cos = {:.6}, max rel_L2 = {:.3e} (layer {}, prompt {}, pos {})  \n",
            r.path_min_cos,
            r.path_max_rel_l2,
            r.path_worst_rel_l2_layer,
            r.path_worst_rel_l2_prompt,
            r.path_worst_rel_l2_pos,
        ));
        s.push_str(&format!(
            "**Diagnostic aggregate:** max abs_L2 = {:.3e} (layer {}, prompt {}, pos {}), max|Δ| = {:.3e}, n_obs = {}  \n",
            r.path_max_l2,
            r.path_worst_layer,
            r.path_worst_prompt,
            r.path_worst_pos,
            r.path_max_abs,
            r.n_total_obs,
        ));
        if !r.dispatch_counts.is_empty() {
            s.push_str("**Dispatch counts:** ");
            let parts: Vec<String> = r
                .dispatch_counts
                .iter()
                .map(|(k, v)| format!("`{}`={}", k, v))
                .collect();
            s.push_str(&parts.join(", "));
            s.push_str("  \n");
        }
        if !r.fallthrough_layers.is_empty() {
            s.push_str(&format!(
                "**⚠ exact→full_mmap fallthrough at layers:** {:?}  \n",
                r.fallthrough_layers
            ));
        }
        if !r.fail_reasons.is_empty() {
            s.push_str("**Fail reasons:**\n");
            for reason in &r.fail_reasons {
                s.push_str(&format!("- {}\n", reason));
            }
        }
        s.push('\n');

        // Per-prompt block
        s.push_str("### Per-prompt\n\n");
        s.push_str("| prompt | walk top-1 | dense top-1 | match | walk P | dense P | ΔP |\n");
        s.push_str("|---|---|---|---|---|---|---|\n");
        for (key, p) in &r.per_prompt {
            s.push_str(&format!(
                "| `{}` | `{}` | `{}` | {} | {:.6} | {:.6} | {:.3e} |\n",
                key,
                p.walk_top1_token,
                p.dense_top1_token,
                if p.top1_match { "✓" } else { "✗" },
                p.walk_top1_prob,
                p.dense_top1_prob,
                p.prob_delta,
            ));
        }
        s.push('\n');

        // Per-layer block. Assertion columns first, then diagnostic.
        s.push_str("### Per-layer\n\n");
        s.push_str(
            "| layer | dispatch | min cos (assert) | max rel L2 (assert) | rel L2 worst (prompt/pos) | max abs L2 (diag) | max\\|Δ\\| (diag) | abs L2 worst (prompt/pos) | n |\n",
        );
        s.push_str("|---|---|---|---|---|---|---|---|---|\n");
        for (i, ls) in r.layers.iter().enumerate() {
            s.push_str(&format!(
                "| {} | `{}`{} | {:.6} | {:.3e} | {}/{} | {:.3e} | {:.3e} | {}/{} | {} |\n",
                i,
                ls.dispatch_label,
                if ls.fallthrough { " ⚠" } else { "" },
                ls.min_cos,
                ls.max_rel_l2,
                ls.worst_rel_l2_prompt,
                ls.worst_rel_l2_pos,
                ls.max_l2,
                ls.max_abs,
                ls.worst_prompt,
                ls.worst_pos,
                ls.n_obs,
            ));
        }
        s.push('\n');
    }

    s
}

pub(super) fn render_json(model: &str, vindex: &Path, runs: &[PathRun]) -> String {
    use serde_json::{json, Value};
    let paths: Vec<Value> = runs
        .iter()
        .map(|r| {
            json!({
                "name": r.name,
                "mask": {
                    "hide_fp4": r.mask.hide_fp4,
                    "hide_q4": r.mask.hide_q4,
                    "hide_interleaved": r.mask.hide_interleaved,
                    "hide_full_mmap": r.mask.hide_full_mmap,
                    "hide_q4k": r.mask.hide_q4k,
                    "hide_down_features": r.mask.hide_down_features,
                },
                "sparse_k": r.sparse_k.map(|k| if k == usize::MAX { -1i64 } else { k as i64 }),
                "bound": r.bound.map(|b| json!({
                    "kind": b.kind,
                    "min_cos": b.min_cos,
                    "rel_l2": b.rel_l2,
                })),
                "aggregate": {
                    "assertion": {
                        "min_cos": r.path_min_cos,
                        "max_rel_l2": r.path_max_rel_l2,
                        "worst_rel_l2_layer": r.path_worst_rel_l2_layer,
                        "worst_rel_l2_prompt": r.path_worst_rel_l2_prompt,
                        "worst_rel_l2_pos": r.path_worst_rel_l2_pos,
                    },
                    "diagnostic": {
                        "max_abs_l2": r.path_max_l2,
                        "mean_abs_l2": r.path_mean_l2,
                        "max_abs": r.path_max_abs,
                        "worst_layer": r.path_worst_layer,
                        "worst_prompt": r.path_worst_prompt,
                        "worst_pos": r.path_worst_pos,
                    },
                    "n_obs": r.n_total_obs,
                },
                "dispatch_counts": r.dispatch_counts,
                "fallthrough_layers": r.fallthrough_layers,
                "per_prompt": r.per_prompt.iter().map(|(k, p)| (k.clone(), json!({
                    "walk_top1_token": p.walk_top1_token,
                    "walk_top1_prob": p.walk_top1_prob,
                    "dense_top1_token": p.dense_top1_token,
                    "dense_top1_prob": p.dense_top1_prob,
                    "top1_match": p.top1_match,
                    "prob_delta": p.prob_delta,
                }))).collect::<serde_json::Map<_, _>>(),
                "per_layer": r.layers.iter().enumerate().map(|(i, ls)| json!({
                    "layer": i,
                    "dispatch": ls.dispatch_label,
                    "fallthrough": ls.fallthrough,
                    "assertion": {
                        "min_cos": ls.min_cos,
                        "max_rel_l2": ls.max_rel_l2,
                        "worst_rel_l2_prompt": ls.worst_rel_l2_prompt,
                        "worst_rel_l2_pos": ls.worst_rel_l2_pos,
                    },
                    "diagnostic": {
                        "max_abs_l2": ls.max_l2,
                        "max_abs": ls.max_abs,
                        "worst_prompt": ls.worst_prompt,
                        "worst_pos": ls.worst_pos,
                    },
                    "n_obs": ls.n_obs,
                })).collect::<Vec<_>>(),
                "verdict": if r.pass { "pass" } else { "fail" },
                "fail_reasons": r.fail_reasons,
            })
        })
        .collect();

    let root = json!({
        "model": model,
        "vindex": vindex.display().to_string(),
        "prompts": PROMPTS.iter().map(|(k, p)| json!({"key": k, "text": p})).collect::<Vec<_>>(),
        "paths": paths,
    });
    serde_json::to_string_pretty(&root).unwrap()
}
