//! RESIDUAL-BUS-1 baseline: what does batch prefill pay for the layer
//! trace nobody reads?
//!
//! `execute_layer` copies the carrier plane twice per layer into its
//! `LayerTrace` (`post_attention`, `post_layer`), even on the production
//! prefill, whose sink discards every event. Those copies are timed as
//! their own leaf, `OpClass::PlaneTrace`. This runs the server's prefill
//! (`PreparedVindex3::prefill_into`) over one prepared image and reports
//! that leaf against the prefill wall.
//!
//! Both copies run on the calling thread outside any parallel region, so
//! the leaf is critical-path wall time and its share of the prefill wall
//! is meaningful. The other leaves are not reported: in the batched driver
//! they sum across threads.
//!
//! Usage:
//!   cargo run --release -p larql-server --example bus1_prefill_trace_cost \
//!       -- <container> [prompt_tokens] [trials] [warmup]

use std::path::PathBuf;
use std::time::Instant;

use larql_inference::vindex3::Vindex3Runtime;
use larql_kv::CanonicalKvState;
use larql_vindex::format::vindex3::opplan::exec::production::ProductionBackend;
use larql_vindex::format::vindex3::opplan::exec::timing::{self, OpClass};

const COMPONENT: &str = "target";
const DEFAULT_PROMPT: usize = 128;
const DEFAULT_TRIALS: usize = 10;
const DEFAULT_WARMUP: usize = 2;
/// Token ids cycle through a fixed band, as `v3_request_phase_profile` does.
const TOKEN_BASE: u32 = 100;
const TOKEN_BAND: u32 = 2000;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let container = PathBuf::from(
        args.next()
            .expect("usage: <container> [prompt_tokens] [trials] [warmup]"),
    );
    let prompt_len: usize = args.next().map_or(DEFAULT_PROMPT, |a| a.parse().unwrap());
    let trials: usize = args.next().map_or(DEFAULT_TRIALS, |a| a.parse().unwrap());
    let warmup: usize = args.next().map_or(DEFAULT_WARMUP, |a| a.parse().unwrap());
    let prompt: Vec<u32> = (0..prompt_len as u32)
        .map(|i| i % TOKEN_BAND + TOKEN_BASE)
        .collect();

    let t = Instant::now();
    let runtime =
        Vindex3Runtime::open(&container, COMPONENT, ProductionBackend::new())?.prepare()?;
    println!("open + prepare {:.3} s", t.elapsed().as_secs_f64());
    println!(
        "{:>5}  {:>12}  {:>14}  {:>8}  {:>8}",
        "trial", "prefill ms", "PlaneTrace ms", "calls", "share %"
    );

    let (mut walls, mut traces, mut shares) = (Vec::new(), Vec::new(), Vec::new());
    for trial in 0..warmup + trials {
        let mut kv = CanonicalKvState::new();
        timing::ledger().reset();
        let t = Instant::now();
        runtime.prefill_into(&prompt, &mut kv)?;
        let wall = t.elapsed().as_secs_f64() * 1e3;
        let tally = timing::ledger().get(OpClass::PlaneTrace);
        let trace = tally.nanos as f64 / 1e6;
        let share = 100.0 * trace / wall;
        let tag = if trial < warmup { " (warm-up)" } else { "" };
        println!(
            "{:>5}  {wall:>12.3}  {trace:>14.3}  {:>8}  {share:>8.3}{tag}",
            trial, tally.calls
        );
        if trial >= warmup {
            walls.push(wall);
            traces.push(trace);
            shares.push(share);
        }
    }
    println!(
        "median over {trials} trials, {prompt_len} prompt tokens: prefill {:.3} ms, PlaneTrace {:.3} ms, share {:.3}%",
        median(walls),
        median(traces),
        median(shares)
    );
    Ok(())
}
