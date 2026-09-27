# RESIDUAL-BUS-1 results: one carrier-transition language for batch and decode

**Class: RECORD.** Dated 2026-09-27, against the [frozen protocol](residual-bus-1.md)
(`e8c6d141`). The implementation is `df38e7a5` through `8948bdda` on
`residual-bus-lcp-1`, over main `b4e029a3`.

**Correctness: ACCEPTED.** T1–T7 and T9 hold, and F1 and F2 hold:

- on every synthetic subject, on both backends;
- on a real Granite 4.2 3B container, on both backends.

**Cost: PENDING.** T8 and F3 are unmeasured. Two quiet-gated attempts on
2026-09-26/27 found no quiet minute: another session's llvm-cov run and two
agent worktrees' compiles pushed the 1-minute load to 75.9. Nothing was
measured under load. BUS-1 closes when §4 is filled.

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
| T8 no subscriber, no new copy | **PENDING (cost)** | By construction, the new events borrow or carry no payload. The measurement is §4 |
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
| F3 cost | **PENDING** (§4) |

## 4. Cost (pending)

This will be measured under the §3 protocol of the freeze, with the same
arguments as the baselines:

- **T8:** `bus1_prefill_trace_cost`, 128 tokens, 10 trials, 2 warm-ups.
  Prefill wall and `PlaneTrace` must fall within the baseline spread
  (baseline: `PlaneTrace` 5.11 ms (4.8–5.4), share 0.091%).
- **F3:** P5, 15 pairs. The noop median must fall within 56.5–58.5 ms/token.

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
