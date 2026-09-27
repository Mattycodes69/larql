# RESIDUAL-BUS-1 results: one carrier-transition language for batch and decode

**Class: RECORD.** Dated 2026-09-27, against the [frozen protocol](residual-bus-1.md)
(`e8c6d141`). The implementation is `df38e7a5` through `8948bdda` on
`residual-bus-lcp-1`, over main `b4e029a3`.

**Correctness: ACCEPTED.** T1–T7 and T9 hold, and F1 and F2 hold:

- on every synthetic subject, on both backends;
- on a real Granite 4.2 3B container, on both backends.

**Cost: MEASURED on 2026-09-27 (§4). F3 HOLDS. T8 is PARTIAL.** T8's
substantive claim holds: there is no new copy, the `PlaneTrace` call count is
unchanged, and its share of prefill is inside the baseline range. One literal
subclaim lost: the `PlaneTrace` median is 4.68 ms, **below** the frozen
4.8–5.4 ms band. That is scored as a loss, not a deviation. The
with-subscriber batch cost is reported: it is not resolved from zero, and
every estimator lies within ±26 ms (0.5%) of a 5.2 s prefill.

**BUS-1 is CLOSED.** Correctness is accepted, F3 decode holds, T8 is PARTIAL
with its loss recorded, and no frozen measurement is outstanding.

Two earlier quiet-gated attempts on 2026-09-26/27 found no quiet minute:
another session's llvm-cov run and two agent worktrees' compiles pushed the
1-minute load to 75.9. Nothing was measured under load.

## The claim, and its scope

> One `CarrierTransition` vocabulary describes batch and decode execution
> across Rows, scaled Rows, Bundle, History (with its boundaries), KDA and
> MLA. **Within each backend**, batch and decode emit the same transition
> sequence at every position, and every carrier write is bit-identical
> between them. That includes a real 40-layer Granite 4.2 3B container.

**Not claimed: agreement across backends.** On Granite, Production and
Reference have identical structure but different values.

- **Same structure:** 1,296 events, 640 writes, the same sequence, and a
  bit-identical entering carrier.
- **Different values:** 0 of 640 writes are bit-identical, with drift from
  layer 0. The relative RMS difference is 0.46% at layer 0 and 2.4% at layer
  39, and the final logits differ by 3.4%.
- **Different execution fingerprints:** `5bc6d15a…` (Production) against
  `336860d9…` (Reference).

That cross-backend difference is reported, as the V3-OBS-1 harness designs
it. BUS-1 asserts nothing about it.

## 1. What the implementation found

| Subject | Before BUS-1 | What changed |
|---|---|---|
| Single / Rows | Decode had a per-write record; batch had one post-scale `LayerTrace` per layer | Batch emits `PlaneEvent::CarrierWrite`: per-position `delta`, pre-scale `after` and `layer_scale`, borrowed at the add (`df38e7a5`) |
| Bundle | Already converged (`HcSitePlane` ↔ `HcSiteRecord`), but A7 compared with derived `PartialEq` | A7 compares by `to_bits`; a control shows a signed zero passes `==` and fails A7 (`7d2d7f4c`) |
| History + boundaries | Already converged, with boundary events recorded, but A7, the exit and the logits compared with `==` | The same bitwise upgrade, with controls on both a site event and a boundary event (`f81964ef`) |
| Mixed KDA / MLA | Compared on **logits only** | The first per-state batch/decode witness: 48 writes per side, bit-identical (`8e2e49c2`) |
| All forms | Two transitions had **no record on either path**: a boundary's prefix **reset**, and whether a history write **added or replaced** | `CarrierTransition` names every change on both paths, and both now fire (`8948bdda`) |

The pattern: Rows needed emission work. Bundles and history already had
converged semantics, but their proofs were weaker than the freeze's
bit-for-bit claim. KDA/MLA was the one subject where a hidden divergence was
possible, and none was found.

## 2. Properties

| | Result | Evidence |
|---|---|---|
| T1 one vocabulary | PASS | `Enter`, `Add`, `HcUpdate`, `HistoryWrite{Add\|Replace}`, `HistorySnapshot`, `HistoryReset`, `Intervene`, `Scale`. Decode emits through a defaulted `StepObserver::transition`, batch through `PlaneEvent::Transition`. Per-position sequences are equal on every subject (`carrier_transition.rs` `t1_*`) |
| T2 record, not authority | PASS | No arithmetic moved. Emission sits beside the existing writes, and P1 and T7 hold |
| T3 borrowed, `before` chained | PASS | `CarrierWritePlane` borrows `deltas` and `after`; no `before` is owned |
| T4 the scale is its own transition | PASS | One `Scale` per layer per position on Gemma 4, on both paths; the FFN `Add` record carries the pre-scale `after` and `layer_scale` |
| T5 topology not collapsed | PASS | Bundles fire `HcUpdate`, never `Add`. The bundle and history witnesses panic on a single-stream write event |
| T6 equality | PASS | Rows, the Gemma 4 scale and KDA/MLA via `CarrierWritePlane` ↔ `CarrierWriteRecord`; bundles and history via bitwise A7; Granite, 640 of 640 writes on each backend |
| T7 observation changes nothing | PASS | Batch logits are bit-identical subscribed and unsubscribed. Decode P1 holds on Granite, both backends |
| T8 no subscriber, no new copy | **PARTIAL** | No new copy: `PlaneTrace` makes 80 calls per trial, as at baseline, and its share is 0.095%, inside the baseline's 0.086–0.107%. The literal subclaim lost: the `PlaneTrace` median is 4.68 ms, below the frozen 4.8–5.4 ms band (§4) |
| T9 interventions decode-only | PASS | Batch has no intervention parameter, and `Intervene` fires only in decode |

**Negative controls.** Each fails at exactly the rows or positions it
alters:

- one ulp in a batch `delta`;
- one ulp in a batch `after`;
- a swapped position pair (a simulated swap in the witness, plus the
  existing `SwapPositionsBeforeUpdate` caught by A2 and A7);
- a dropped transition;
- a signed zero in a bundle record, and in both a history site event and a
  boundary event.

## 3. Forecasts

| | Result |
|---|---|
| F1 structure | **HOLDS.** Granite 4.2 3B: 81 transitions per position (80 `Add` + `Enter`), sequences identical at all 8 positions, Production and Reference ([`granite-acceptance.txt`](../bench/residual-bus-1/results/20260927/granite-acceptance.txt)). The synthetic subjects match their declared counts |
| F2 equality | **HOLDS** on all five subjects, both backends. KDA/MLA was the subject that could fail |
| F3 cost | **Decode HOLDS**: the noop median is 57.00 ms/token, inside 56.5–58.5. **Batch: see T8** (PARTIAL). **With a subscriber, reported:** not resolved from zero; every estimator is within ±26 ms (0.5%) of prefill (§4) |

## 4. Cost

Measured at `483f36f4` (the implementation plus the Granite harness, over
main `b4e029a3`), under the §3 protocol of the freeze, with the baselines'
arguments and release binaries. Each run started after a full quiet minute:
no `rustc`, `cargo`, `sccache` or `llvm-cov` process, no `mds_stores`
activity, and a 1-minute load under 3.0. No compile was running at either
run's end. Exclusivity is not claimed: other interactive sessions were open
but idle. Results are in
[`bench/residual-bus-1/results/20260927/`](../bench/residual-bus-1/results/20260927/).

**T8: batch prefill with no subscriber.** `bus1_prefill_trace_cost`, 128
prompt tokens, 10 trials after 2 warm-ups
([`prefill-trace.txt`](../bench/residual-bus-1/results/20260927/prefill-trace.txt)).
Load was 2.41 before and 5.84 after; the baseline's was 2.10 and 6.18, the
rise from prefill's own threads.

| Median | Baseline (`da8ec6a0`) | BUS-1 (`483f36f4`) | Frozen band | |
|---|---:|---:|---|---|
| prefill wall | 5,652 ms (4,808–5,912) | 5,049 ms (4,542–5,188) | within the spread | inside the baseline's trial range |
| `PlaneTrace` | 5.11 ms (4.79–5.42) | **4.68 ms** (4.30–5.27) | 4.8–5.4 ms | **below the band** |
| share | 0.091% (0.086–0.107%) | 0.095% (0.090–0.102%) | — | inside |
| calls per trial | 80 | 80 | — | unchanged |

- **What holds:** there is no new copy. The call count is identical, and the
  copies' share of prefill is inside the baseline's range.
- **What lost:** "`PlaneTrace` unchanged within 4.8–5.4 ms". The median is
  0.12 ms under the band's floor. It missed on the side where cost can't
  have been added, but a band that misses low is still a miss.
- **The likely cause, not measured:** machine state. Prefill wall fell by
  the same order (−10.7%) as `PlaneTrace` (−8.4%), so the ratio barely
  moved. The baseline started at a 5-minute load of 5.69, and this run at
  2.03. Only a same-window A/B against `da8ec6a0` could separate drift from
  a real change, and one was not run.

**F3: decode.** V3-OBS-1 P5, release test binary, Production, 15
interleaved pairs after one discarded warm-up
([`p5.txt`](../bench/residual-bus-1/results/20260927/p5.txt)). Load was 2.27
before and 3.16 after. The same run passed P1, P2, P3, batch/decode, and
RESIDUAL-BUS-1 T1/F1/T6.

| Per token | Baseline noop | BUS-1 noop | BUS-1 stats | stats − noop |
|---|---:|---:|---:|---:|
| median | 57.14 ms | **57.00 ms** | 58.25 ms | +1.25 ms (1.022) |
| mean | 57.16 ms | 56.92 ms | 58.33 ms | +1.41 ms (1.025) |
| min | 56.53 ms | 56.32 ms | 58.00 ms | +1.68 ms (1.030) |
| max | 58.50 ms | 57.51 ms | 58.95 ms | |

The noop median, 57.00 ms, is inside the frozen 56.5–58.5 ms. **F3 decode
HOLDS.**

**F3: batch with a subscriber (reported, not forecast).** The same example
in `subscriber` mode (`c893e05f`), 128 prompt tokens, 10 trials after 2
warm-ups ([`prefill-subscriber.txt`](../bench/residual-bus-1/results/20260927/prefill-subscriber.txt)).
Each trial runs three prefills over the same operands, in an order rotated
per trial:

1. the server's `prefill_into` (T8's arm);
2. the streaming entry with a sink that discards every event;
3. the streaming entry with a bare subscriber. It counts every `Transition`
   and `CarrierWrite` and reads one value from each borrowed row, so it pays
   for emission and the borrow, not for any analysis.

The run started after a 625 s wait for a quiet minute, at a load of 2.03. No
compile was running at the end. Every trial's subscriber saw the same
stream: 10,368 transitions (81 × 128), 80 writes and 10,240 rows.

| Median | prefill_into | stream, discard | stream, subscriber |
|---|---:|---:|---:|
| wall | 5,216 ms | 5,207 ms | 5,209 ms |

| subscriber − discard | ms | % of discard |
|---|---:|---:|
| difference of medians | +1.8 | +0.04% |
| paired median (true, n = 10) | +10.2 | +0.20% |
| paired median (the example's upper median) | +25.8 | +0.49% |
| paired mean | −13.3 | −0.26% |
| paired range | −213 to +45 | |

**Reading.** The subscriber's cost is **not resolved from zero** at this
sample size. The estimators disagree in sign, and the trial-to-trial paired
spread (−213 to +45 ms) is wider than any of them. What the data supports is
a bound: every estimator lies within ±26 ms, about 0.5% of prefill. A bare
transition subscriber is therefore not a material batch cost. No point value
is quotable. The two entry points agree within 9 ms, so the discarding
stream is a fair control for the server's prefill.

**Follow-up, optional, not a closure condition.** A same-window A/B of T8
against `da8ec6a0` would show whether the `PlaneTrace` miss was
machine-state drift. It could explain the loss, but it could not change it.

## 5. Found on the way, for their owners

- **`represent/actuate/plan.rs` line coverage is 89.23%, below the 90%
  floor.** It is pre-existing, and was exposed once #591 made coverage run.
- **`kv_view::a_read_below_base_is_named_not_an_index_panic`** asserts a
  `debug_assert!` message, so it fails in release builds. It should be
  `#[cfg(debug_assertions)]`.
- **The CLI's streaming `vindex3 exec` resets its per-layer stopwatch on
  hyper-connection and attention-residual site events**, so layer times on
  those models are understated. BUS-1's own new events return early and do
  not add to this.
- **Kimi's hand-composed stack adds the shared expert in a different order
  on CPU and Metal.** Each order is checked against Python, and the two are
  never checked against each other.
- **Lowered Metal has no automated test comparing it with the
  interpreter.**
