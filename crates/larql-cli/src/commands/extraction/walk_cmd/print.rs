//! Prediction, summary and walk-trace printing.

#[allow(unused_imports)]
use super::*;

pub(super) fn print_predictions(label: &str, predictions: &[(String, f64)], verbose: bool) {
    if verbose {
        println!("\nTop predictions ({label}):");
        for (i, (token, prob)) in predictions.iter().enumerate() {
            println!("  {:2}. {:20} ({:.2}%)", i + 1, token, prob * 100.0);
        }
    } else {
        // Ollama-style clean output — just the top-1 token on stdout,
        // no framing, no probabilities. `-v` for the full table.
        if let Some((token, _)) = predictions.first() {
            println!("{}", token.trim());
        }
    }
}

pub(super) fn print_summary_row(
    label: &str,
    predictions: &[(String, f64)],
    elapsed: std::time::Duration,
) {
    let (top1, prob1) = predictions
        .first()
        .map(|(t, p)| (t.as_str(), *p))
        .unwrap_or(("?", 0.0));
    println!(
        "{:<40} {:<15} {:>7.2}% {:>6.0}ms",
        label,
        top1,
        prob1 * 100.0,
        elapsed.as_secs_f64() * 1000.0,
    );
}

pub(super) fn print_walk_trace(trace: &larql_vindex::WalkTrace, down_top_k: usize) {
    for (layer, hits) in &trace.layers {
        if hits.is_empty() {
            continue;
        }

        println!("Layer {layer}:");
        for (i, hit) in hits.iter().enumerate() {
            let down_tokens: String = hit
                .meta
                .top_k
                .iter()
                .take(down_top_k)
                .map(|t| format!("{} ({:.2})", t.token, t.logit))
                .collect::<Vec<_>>()
                .join(", ");

            // Executed-path values (2026-07-30 review, item 17): hits
            // now come from the runtime trace, so the activation the
            // walk actually computed is available. Absent on post-hoc
            // KNN views — print nothing rather than a fake number.
            let act = hit
                .activation
                .map(|a| format!("  act={a:+.3}"))
                .unwrap_or_default();
            println!(
                "  {:2}. F{:<5} gate={:+.3}{}  hears={:15}  c={:.2}  down=[{}]",
                i + 1,
                hit.feature,
                hit.gate_score,
                act,
                format!("{:?}", hit.meta.top_token),
                hit.meta.c_score,
                down_tokens,
            );
        }
    }
}
