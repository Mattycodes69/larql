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
//! `subscriber` mode measures F3's with-subscriber arm instead: each trial
//! runs three interleaved prefills over the same operands, in rotated
//! order. They are the server's prefill (T8's arm), the streaming entry
//! with a sink that discards every event, and the streaming entry with a
//! bare transition subscriber. That subscriber counts every `Transition`
//! and `CarrierWrite` and reads one value from each borrowed row, so it
//! pays for the emission and the borrow, not for any analysis. The
//! subscriber's added cost is its wall minus the discarding sink's, which
//! share an entry point.
//!
//! Usage:
//!   cargo run --release -p larql-server --example bus1_prefill_trace_cost \
//!       -- <container> [prompt_tokens] [trials] [warmup] [trace|subscriber]

use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use larql_inference::vindex3::Vindex3Runtime;
use larql_kv::CanonicalKvState;
use larql_vindex::format::vindex3::opplan::exec::production::ProductionBackend;
use larql_vindex::format::vindex3::opplan::exec::timing::{self, OpClass};
use larql_vindex::format::vindex3::opplan::exec::{execute_prepared_streaming_in, PlaneEvent};

const COMPONENT: &str = "target";
const DEFAULT_PROMPT: usize = 128;
const DEFAULT_TRIALS: usize = 10;
const DEFAULT_WARMUP: usize = 2;
const MODE_TRACE: &str = "trace";
const MODE_SUBSCRIBER: &str = "subscriber";
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
    let mode = args.next().unwrap_or_else(|| MODE_TRACE.to_string());
    let prompt: Vec<u32> = (0..prompt_len as u32)
        .map(|i| i % TOKEN_BAND + TOKEN_BASE)
        .collect();

    let t = Instant::now();
    let runtime =
        Vindex3Runtime::open(&container, COMPONENT, ProductionBackend::new())?.prepare()?;
    println!("open + prepare {:.3} s", t.elapsed().as_secs_f64());
    match mode.as_str() {
        MODE_TRACE => {}
        MODE_SUBSCRIBER => return subscriber_cost(&runtime, &prompt, trials, warmup),
        other => panic!("unknown mode `{other}`: expected `{MODE_TRACE}` or `{MODE_SUBSCRIBER}`"),
    }
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

/// What the bare subscriber saw, so its reads cannot be optimised away
/// and every trial can be checked to see the same stream.
#[derive(Default, Clone, Copy, PartialEq, Debug)]
struct Seen {
    transitions: usize,
    writes: usize,
    rows: usize,
    fold: u32,
}

type Runtime = larql_inference::vindex3::PreparedVindex3<ProductionBackend>;

/// One prefill through the streaming entry, with or without the bare
/// subscriber. Returns the wall in ms and what the sink saw.
fn streamed(
    runtime: &Runtime,
    prompt: &[u32],
    subscribe: bool,
) -> Result<(f64, Seen), Box<dyn std::error::Error>> {
    let mut kv = CanonicalKvState::new();
    let mut seen = Seen::default();
    let mut sink = |event: PlaneEvent| {
        if subscribe {
            match event {
                PlaneEvent::Transition { transition, .. } => {
                    black_box(transition);
                    seen.transitions += 1;
                }
                PlaneEvent::CarrierWrite(write) => {
                    seen.writes += 1;
                    for (delta, after) in write.deltas.iter().zip(write.after) {
                        seen.rows += 1;
                        seen.fold ^= delta[0].to_bits() ^ after[0].to_bits();
                    }
                }
                _ => {}
            }
        }
        Ok(())
    };
    let t = Instant::now();
    execute_prepared_streaming_in(
        runtime.plan(),
        runtime.operands(),
        prompt,
        runtime.backend(),
        None,
        &mut sink,
        &mut kv,
    )?;
    Ok((t.elapsed().as_secs_f64() * 1e3, black_box(seen)))
}

/// F3's with-subscriber batch arm: three prefills per trial, in an order
/// rotated each trial so none always runs first.
fn subscriber_cost(
    runtime: &Runtime,
    prompt: &[u32],
    trials: usize,
    warmup: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    const ARMS: usize = 3;
    const NAMES: [&str; ARMS] = ["prefill_into", "stream discard", "stream subscriber"];
    println!(
        "{:>5}  {:>14}  {:>16}  {:>18}  {:>10}  {:>6}  {:>7}",
        "trial", "prefill_into ms", "stream discard ms", "stream subscriber ms", "transitions", "writes", "rows"
    );
    let mut walls: [Vec<f64>; ARMS] = Default::default();
    let mut diffs = Vec::new();
    let mut expected: Option<Seen> = None;
    for trial in 0..warmup + trials {
        let mut wall = [0.0; ARMS];
        let mut seen = Seen::default();
        for k in 0..ARMS {
            let arm = (trial + k) % ARMS;
            wall[arm] = match arm {
                0 => {
                    let mut kv = CanonicalKvState::new();
                    let t = Instant::now();
                    runtime.prefill_into(prompt, &mut kv)?;
                    t.elapsed().as_secs_f64() * 1e3
                }
                1 => streamed(runtime, prompt, false)?.0,
                _ => {
                    let (ms, s) = streamed(runtime, prompt, true)?;
                    seen = s;
                    ms
                }
            };
        }
        match expected {
            None => expected = Some(seen),
            Some(e) => assert_eq!(e, seen, "the subscriber saw a different stream at trial {trial}"),
        }
        let tag = if trial < warmup { " (warm-up)" } else { "" };
        println!(
            "{:>5}  {:>14.3}  {:>16.3}  {:>18.3}  {:>10}  {:>6}  {:>7}{tag}",
            trial, wall[0], wall[1], wall[2], seen.transitions, seen.writes, seen.rows
        );
        if trial >= warmup {
            for arm in 0..ARMS {
                walls[arm].push(wall[arm]);
            }
            diffs.push(wall[2] - wall[1]);
        }
    }
    let medians: Vec<f64> = walls.iter().map(|w| median(w.clone())).collect();
    for (name, m) in NAMES.iter().zip(&medians) {
        println!("median over {trials} trials, {} prompt tokens: {name} {m:.3} ms", prompt.len());
    }
    let paired = median(diffs);
    println!(
        "subscriber - discard: paired median {paired:+.3} ms ({:+.3}% of discard), difference of medians {:+.3} ms",
        100.0 * paired / medians[1],
        medians[2] - medians[1]
    );
    Ok(())
}
