# RESIDUAL-BUS-0 results: the idle floor explains a third of the in-model remainder

**Class: RECORD.** Measured 2026-09-26 against the [frozen protocol](residual-bus-0.md)
at `b109e29f`. The adjudication was applied mechanically from
[`collection-0.json`](../bench/residual-bus-0/results/20260926/collection-0.json);
its output is [`adjudication.txt`](../bench/residual-bus-0/results/20260926/adjudication.txt).

**Outcome: D4 holds on the primary shape, so D4 governs what comes next.**
On an idle-ish host, the entire current wire stack (axum + reqwest + a per-call
client thread) costs **76.8 µs** per GPT-OSS expert round trip. That is below
half of the **232 µs** per-call remainder PROFILE-1 measured inside the model.
About two thirds of the in-model remainder is therefore not intrinsic to the
transport. The next step is a loaded-host measurement, not a protocol.

D1 also holds, narrowly, and the frozen rule that produced it has a defect
(below). Under both the frozen and the corrected arithmetic, framing (D2) does
**not** hold. No evidence here justifies LCP-1.

## Environment

- Source `2756f8f5…0a44` and release binary `d58b56b1…798a` at rev `b109e29f`,
  built with rustc 1.98.0.
- Apple M3 Max, 16 logical CPUs, 128 GiB, macOS 15.7.4
  ([`env.txt`](../bench/residual-bus-0/results/20260926/env.txt)).
- The freeze commit was rewritten message-only (an attribution trailer removed)
  from `0e074e2d` to `b109e29f`. It has the same parent and a byte-identical
  diff, so the tree the binary was built from is unchanged. `env.txt` keeps
  the original hash as recorded.
- 10,000 measured calls after 1,000 warm-up calls per arm×shape, two passes,
  and two Tokio workers.

**Load.** The run waited until no `rustc`/`cargo` process had existed for 60 s
and the 1-minute load was below 3.0. The rebuild from committed source that
followed raised it again. The 1-minute load averages were **6.96 before and
6.53 after**. No other compile was running at the start or end. Peer
exclusivity is not claimed; a browser, two other agent sessions and Codex were
resident. Load can only inflate these round trips, so it biases D4 **against**
holding, not towards it.

## Validity (pass agreement ≤ 10% at p50)

| Shape | T0 | T1 | T2 | T3 | T3s |
|---|---|---|---|---|---|
| `gptoss-experts` (primary) | 1.47% | 2.00% | 7.83% | 1.30% | 2.17% |
| `qwen3-dense` | 0.25% | **23.29% void** | 2.33% | 8.85% | **10.21% void** |
| `gemma3-dense` | 2.23% | 1.82% | 0.27% | 0.73% | 0.24% |

Every primary-shape arm is valid, so the predefined replicate is **not**
triggered. For Qwen, handoff, client_spawn, above_floor and D4 are void,
because each uses a void arm. They are reported as void and not repaired.

## p50 µs per round trip, mean of the two passes

| Shape | T0 | T1 | T2 | T3 | T3s |
|---|---:|---:|---:|---:|---:|
| `gptoss-experts` 11,576 → 46,136 B | 22.7 | 29.1 | 32.4 | 57.6 | 76.8 |
| `qwen3-dense` 4,132 ↔ 4,132 B | 16.5 | void | 30.4 | 51.3 | void |
| `gemma3-dense` 10,276 ↔ 10,276 B | 18.7 | 25.3 | 30.5 | 51.0 | 70.9 |

p99 values are in the collection file. T2–T3s p99 runs 50–125 µs, and T0 p99
runs 24–34 µs.

## Quantities (primary and Gemma; Qwen reported where not void)

| µs | `gptoss-experts` | `gemma3-dense` | `qwen3-dense` |
|---|---:|---:|---:|
| floor (T0) | 22.7 | 18.7 | 16.5 |
| handoff (T1 − T0) | 6.5 | 6.6 | void |
| runtime (T2 − T0) | 9.8 | 11.9 | 13.9 |
| http (T3 − T2) | 25.1 | 20.4 | 20.9 |
| client_spawn (T3s − T3) | 19.2 | 19.9 | void |
| above_floor (T3s − T0) | 54.1 | 52.2 | void |

## Decision

| Rule | `gptoss-experts` | Holds |
|---|---|---|
| D1, as frozen: (handoff + runtime + client_spawn) / above_floor ≥ 50% | 35.4 / 54.1 = **65.5%** | yes |
| D1, corrected: (runtime + client_spawn) / above_floor | 29.0 / 54.1 = **53.5%** | yes, narrowly |
| D2: http / above_floor ≥ 50% | 25.1 / 54.1 = **46.5%** | no |
| D4: T3s < 50% of the 232 µs in-model remainder | 76.8 < 116 | **yes, and it takes precedence** |

Gemma agrees on every row: D1 at 73.5% frozen and 60.9% corrected, D2 at
39.1%, and D4 with 70.9 < 88.5. For Qwen, D4 cannot be evaluated because T3s
is void.

**The defect in the frozen D1.** `runtime = T2 − T0` already contains a thread
handoff, since `spawn_blocking` *is* one. Adding `handoff = T1 − T0` to it counts
that handoff twice. The terms that partition above_floor are
runtime + http + client_spawn, which telescope to T3s − T0. The outcome is the
same under both. But the frozen margin (65.5%) overstates the true one (53.5%),
and the result should be read as **scheduling ≈ HTTP, with scheduling slightly
ahead**. A successor rule must use the partition. The frozen text is not
edited.

## What this means

- **Framing is worth ~0.6 ms per position at most.** Eliminating HTTP
  entirely (25.1 µs × 24 layers) recovers about 0.6 ms of a ~116 ms GPT-OSS
  position, about 0.5%. Even collapsing the whole stack to raw TCP
  (54.1 µs × 24 ≈ 1.3 ms) cannot account for PROFILE-1's 5.3–5.6 ms.
- **The unexplained part is the larger part.** Roughly 155 µs per call
  (232 − 77), or ~3.7 ms per position, arises only when the model runs. The
  frozen suspect is contention: PROFILE-1's processes run eight Rayon threads
  each, and transport threads compete with them for cores and wake-ups. This
  result does not prove that. It establishes that an idle transport does not
  explain the remainder.
- **The client's per-call thread spawn is as large as HTTP** (19–20 µs versus
  20–25 µs). It is a pure client-side cost that a persistent per-shard sender
  removes without any protocol change.

## Next

As the freeze fixed, **RESIDUAL-BUS-1** (carrier-write totality) proceeds.
Separately, D4 selects a **loaded-host** measurement as the next transport rung.
That means the same five arms with a CPU compute pool saturated alongside them,
to test whether contention reproduces the in-model remainder. It needs its own
freeze, and its decision rule must use the corrected partition. LCP-1 is
postponed: no rule selected it.
