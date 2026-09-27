# RESIDUAL-BUS-1: one carrier-transition record for the batch and decode traversals

**Class: FREEZE.** Frozen 2026-09-26, in the commit that records both
baselines (§3), before any BUS-1 implementation. The question, properties,
witnesses and forecasts below change only by a new, dated section.

Programme: RESIDUAL-BUS. The reconnaissance is in
[`residual-bus-1-reconnaissance.md`](residual-bus-1-reconnaissance.md). The
programme order is in [`residual-bus-lcp-1.md`](residual-bus-lcp-1.md) §7.

## The question

> Can the batch traversal emit, for every carrier transition it performs, the
> same transition record the decode traversal emits, **bit-identical** per
> (position, layer, site, kind), without changing arithmetic, execution order
> or the unobserved result of either traversal?

The reconnaissance established that the write **authority** is already single
on every production path: decode's `leave_site` and batch's `leave_batch_site`
run the same `PlanBackend` arithmetic, and Kimi runs through both. What differs
is the **record**:

- decode emits a per-write `CarrierWriteRecord` plus a `StepEvent::CarrierWrite`;
- batch emits one `LayerTrace` per layer, post-scale, with no deltas.

This rung converges the record. It does not converge the write paths.

## 1. Frozen properties

**T1: one vocabulary.** Both traversals emit transitions from one enum. Its
kinds are taken from the reconnaissance census (§3 there), not from
`residual_add`:

| Kind | Carrier forms | Meaning |
|---|---|---|
| `Enter` | all | the carrier entering layer 0 (V3-OBS-1 C6), or an external carrier at a `LayerRange` / `ResumePoint` ingress |
| `Add` | Single / Rows | `after = before + delta` |
| `HcUpdate` | Bundle | `comb` / `post` hyper-connection update, not an add |
| `HistoryWrite { Add \| Replace }` | History | an add to the prefix, or a replace after a boundary reset |
| `HistoryBoundary { Snapshot \| Reset }` | History | the boundary state transitions outside `leave_site` |
| `Intervene { kind }` | Single | decode only (T9) |
| `Scale { s }` | Single / Rows | the Gemma 4 `LayerScalar` applied to the whole carrier |

Negative controls (`Mutation`) are **tagged** on the transition they alter.
That covers the HC bypass add, the position swaps, and the history write offset.
A control is never a silent second authority.

**T2: record, not authority.** No arithmetic moves. `residual_add`,
`hyper_connection::update`, `History::write`, `scale_row` and the boundary
functions stay where they are. The record is emitted at the site that performs
the transition.

**T3: values are borrowed; `before` is chained.** A record borrows `delta` /
`after` from where they landed (V3-OBS-1 C1). `before` is never owned. A
consumer reconstructs it from the previous transition (C6), exactly as decode
consumers do today.

**T4: the layer scale is its own transition.** On a component declaring a
`LayerScalar`, the FFN-site `Add` carries the pre-scale `after`, and a
following `Scale { s }` carries the post-scale carrier. Both paths emit both.

The existing `CarrierWriteRecord.layer_scale` field (C5) is kept. T4 is
additive, and no existing consumer loses a field.

**T5: topology is not collapsed.** Bundle and History transitions carry their
existing site values (`HcSiteRecord` / `AttnResSiteRecord` and their plane
forms). No record reduces a bundle or history to one vector (V3-OBS-1 C3).

**T6: the equality.** For every (position, layer, site, kind), batch's record
equals decode's record, with values compared by `to_bits`. This covers both
the structural sequence and the values.

**T7: observation changes nothing.** For both traversals, observed and
unobserved logits and carrier states are bit-identical.

**T8: no subscriber, no new copy.** When no sink subscribes to transition
records, batch performs no per-write copy that it does not perform today.
The existing `LayerTrace` copies (`PlaneTrace`, §3) are not in scope to
remove: `ExecutionTrace` consumers read them.

**T9: interventions stay decode-only.** Batch neither applies nor accepts an
`InterventionPlan`. This is safe today: the one intervening runner steps
prompts through decode, and V3-INTERVENE-1 I8 turns an unfired declaration into
a receipt refusal. BUS-1 adds a test pinning that the batch API has no
intervention parameter, so that adding one is a visible contract change.

## 2. Witnesses

T6 and T7 are witnessed on the Reference and Production backends, over these
subjects:

| Subject | Carrier form | Fixture |
|---|---|---|
| plain stack | Single / Rows | golden synthetic plan (`tests/carrier_write.rs`) |
| Gemma 4 layer scale | Single / Rows + `Scale` | `tests/carrier_write.rs:653`'s stack |
| hyper-connected | Bundle | wave 19 HC substrate (`tests/wave19_hc_batch.rs`) |
| attention residual | History + boundaries | `tests/attn_res_2b_batch.rs` |
| mixed KDA / MLA | Single / Rows | `opplan/tests/kda_mla_exec.rs`'s miniature Kimi. **New coverage:** it is witnessed only on logits today. |

There is also a real-container run on Granite 4.2 3B (`.s6`), in the manner of
`carrier_write_real.rs`, over the same prompt.

**Negative controls.** Each is required to fail the witness at exactly the
transition it alters:

- one ulp in a batch `delta`;
- one ulp in a batch `after`;
- a swapped position pair (the existing `SwapPositionsBeforeUpdate`);
- a dropped `HistoryBoundary`.

## 3. Baselines (measured, not forecast)

**Batch: the existing `LayerTrace` copies.** Instrument `OpClass::PlaneTrace`
(`da8ec6a0`), example `bus1_prefill_trace_cost`, results in
[`bench/residual-bus-1/results/20260926/`](../bench/residual-bus-1/results/20260926/prefill-trace.txt).
Granite 4.2 3B `.s6`, production CPU, 128 prompt tokens, 10 trials after 2
warm-ups:

| | Median |
|---|---:|
| prefill wall | 5,652 ms |
| `PlaneTrace` | 5.11 ms (4.8–5.4) |
| share | **0.091%** (0.086–0.107%) |
| calls per trial | 80 = 2 × 40 layers, every trial |

The 1-minute load averages were 2.10 before and 6.18 after, the rise
consistent with prefill's own threads. No other compile was running at the end.
Exclusivity is not claimed.

**Decode: V3-OBS-1 P5 on the same container.** Harness
`carrier_write_real.rs` at `da8ec6a0`, release test binary, Production
backend, 15 interleaved pairs after one discarded warm-up pair
([`p5.txt`](../bench/residual-bus-1/results/20260926/p5.txt)). The run started
after a full quiet minute (no `rustc`, `cargo`, `sccache` or `mds_stores`
activity, and a 1-minute load under 3.0). Load was 1.64 before and 2.77 after,
with no compile at the end. Two earlier attempts found no quiet minute and
measured nothing.

| Per token | noop | stats observer | difference |
|---|---:|---:|---:|
| median | 57.14 ms | 58.93 ms | **+1.79 ms (ratio 1.031)** |
| mean | 57.16 ms | 58.78 ms | +1.62 ms (1.028) |
| min | 56.53 ms | 58.06 ms | +1.53 ms (1.027) |
| max | 58.50 ms | 60.03 ms | |

The same run passed P1 (bit-identical logits at 8 positions), P2 (640 exact
writes), P3 (640 = forecast) and batch/decode (bit-identical carrier states).

The treatment is V3-OBS-1's **stats observer**: carrier and delta norms, a
3-dimensional fixed projection, and selected-token logits over 80 writes per
token. It is not the cost of a bare record consumer. The three estimators
agree within 0.3 ms, and this **supersedes** V3-OBS-1's +0.15 ms/token median,
which was recorded there as not quotable because its estimators disagreed
about 150-fold across a machine-state shift.

**What the baselines decide.** The unconditional copies cost ~0.1% of
prefill. BUS-1 is therefore justified by **semantics** (one observable record
for prefill and decode, and the first per-state KDA/MLA witness), not by
performance. No speed-up is claimed or forecast.

## 4. Forecasts (about the change only)

- **F1, structure.** Per position, batch emits exactly decode's transition
  sequence:
  - two `Add` per attention+FFN layer, and one per mixer-only layer;
  - one `Scale` per scaled layer;
  - one `HcUpdate` per bundle site;
  - one `HistoryWrite` per history site;
  - one `HistoryBoundary` per boundary decode performs.

  On Granite 4.2 3B, that is 80 transitions per position, excluding `Enter`.
- **F2, equality.** T6 holds on all five subjects, both backends. The
  KDA/MLA subject is the one that could fail: it has never been compared per
  state.
- **F3, cost.**
  - With no subscriber, batch prefill wall is unchanged within the
    measurement's spread, and `PlaneTrace` is unchanged (T8).
  - With a subscriber, the added batch cost is reported and not forecast.
  - Decode's noop per-token wall is unchanged within the P5 baseline's
    noop spread (56.5–58.5 ms on this container), because decode's emission
    sites do not move.

## 5. Declared non-participants

These are unchanged, and named here so the bus claim is honest:

- **Lowered Metal** already refuses `observe` / `--intervene` before
  execution. It has no automated lowered-vs-interpreter parity test; that gap
  belongs to the lowering programme.
- **The hand-composed Kimi stack** (`stack.rs` / `token.rs` / `stack_metal.rs`)
  is not an execution path.
- **Legacy decode** is a migration target.

## 6. Out of scope

- interventions in batch (T9);
- `CarrierAddress`, sequence and identity (BUS-2);
- ROUTE and any transport (BUS-3);
- removing `LayerTrace` (T8);
- the lowered parity gap;
- the Kimi CPU/Metal shared-expert order;
- the pre-existing defects found on the way:
  - `represent/actuate/plan.rs` line coverage 89.23%, below the 90% floor;
  - `kv_view::a_read_below_base_is_named_not_an_index_panic` asserts a
    `debug_assert!` message and fails in release builds.

## 7. Acceptance

BUS-1 is complete when:

- T1–T7 and T9 PASS on both backends over every subject in §2;
- every negative control fails at its transition;
- F1 holds on the real container;
- T8 and F3 are MEASURED under the §3 protocol.

A failed F2 on KDA/MLA is a finding: a batch/decode semantic divergence in
production Kimi. It is not a reason to relax T6.

## Record note: hashes after the rebase onto `b4e029a3`

`residual-bus-lcp-1` was rebased onto main `b4e029a3` (#591). `git range-diff`
shows every commit patch-identical:

| Commit | Before | After |
|---|---|---|
| freeze | `b109e29f` | `61f4c1f5` |
| BUS-0 record | `b5e65a7e` | `fd386668` |
| instrument | `845cd9e9` | `da8ec6a0` |

Earlier records cite the pre-rebase hashes. The trees each commit introduced
are unchanged, and BUS-0's binary was built from the freeze's tree.
