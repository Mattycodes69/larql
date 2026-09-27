# RESIDUAL-BUS-0: the transport and scheduling floor, measured before any protocol is built

**Class: FREEZE.** Dated 2026-09-26. The arms, payloads, statistics, validity
rule and decision rule below are fixed before any measured collection. A
smoke run at reduced call counts, to prove the harness executes, is allowed and
is not a result. This page changes only by a new, dated section.

Programme: RESIDUAL-BUS. The reconnaissance is in
[`residual-bus-lcp-1.md`](residual-bus-lcp-1.md).

## Why this is a measurement, not a forecast

Every quantity here describes the **existing** floor: sockets, threads, the
Tokio runtime, axum, reqwest. Those are baseline mechanics, and they can be
measured. So this rung forecasts nothing, and a forecast is not scored. It
fixes what is measured and **which result selects which next step**. A
forecast belongs to the intervention that BUS-0 selects (for example LCP-1),
and is written in that rung's freeze.

## The question

PROFILE-1 measured the critical round trip minus the worker handler at 5.3–5.6
ms per position, ~222–232 µs per call. WIRE-2 measured the per-call remainder
at 139–177 µs on binary HTTP and 85–108 µs on a persistent WebSocket.

> With no model and no compute, how much of a remote carrier round trip is the
> socket itself, how much is the thread handoff, how much is the async
> runtime's blocking-pool hop, how much is HTTP, and how much is the client's
> per-call thread spawn?

## Harness

[`crates/larql-server/examples/bus0_transport_floor.rs`](../crates/larql-server/examples/bus0_transport_floor.rs).
It runs in one process on loopback. The transform is a no-op: the server reads
the whole request and replies with a freshly allocated body of the recorded
response size.

| Arm | Client | Server |
|---|---|---|
| T0 | persistent `TcpStream`, u32-prefixed frames, `TCP_NODELAY` | one dedicated blocking thread per connection |
| T1 | as T0 | T0, plus each frame goes to a compute thread over a channel and back |
| T2 | as T0 | Tokio accept/read, `spawn_blocking` no-op, async write |
| T3 | `reqwest::blocking` with `HttpExpertShards`'s builder settings | axum route: content-type check, `Bytes` body, `spawn_blocking`, per-connection `TCP_NODELAY` |
| T3s | T3's call made from a per-call `std::thread::scope` spawn (the fan-out shape of `Grid::apply`) | as T3 |

Payload shapes (request → response body bytes, from the recorded formulas):

| Shape | Request | Response | Source |
|---|---:|---:|---|
| `gptoss-experts` | 11,576 | 46,136 | PROFILE-1 one-worker VEX1/VEY1 |
| `qwen3-dense` | 4,132 | 4,132 | WIRE-2 VFF1/VFR1 |
| `gemma3-dense` | 10,276 | 10,276 | WIRE-2 VFF1/VFR1 |

**Settings.**

- The release build is
  `cargo run --release -p larql-server --example bus0_transport_floor -- <out.json>`.
- Each arm×shape makes 10,000 measured calls after 1,000 warm-up calls.
- T2, T3 and T3s run on two Tokio workers, matching PROFILE-1's worker
  processes.
- There are two passes, the second in reverse arm order.
- Each arm×shape starts a fresh server and connection.
- The run is the Apple M3 Max used by PROFILE-1 and WIRE-2. `uptime` load
  averages are recorded immediately before and after.

## Quantities

The primary statistic is p50 µs per round trip. p99 is reported alongside it
and never substituted for it. For each shape:

| Name | Definition |
|---|---|
| floor | T0 |
| handoff | T1 − T0 |
| runtime | T2 − T0 |
| http | T3 − T2 |
| client_spawn | T3s − T3 |
| above_floor | T3s − T0 |

## Validity

- An arm×shape is **valid** when its two passes' p50s agree within 10%
  (symmetric relative difference).
- An invalid arm×shape is reported and **excluded from every quantity that uses
  it**. It is never pooled or replaced by a faster pass.
- If any arm on the primary shape is invalid, **one** predefined replicate of
  the whole collection is run. Both collections are reported. The decision uses
  the replicate only if all of its primary-shape arms are valid.
- Peer exclusivity is not claimed. Other sessions may share the host, and the
  load averages are what is on record.

## Decision rule (frozen)

This is evaluated on the primary shape, `gptoss-experts`, using each valid
arm's p50 averaged over its two passes. The dense shapes are reported as
support, and a disagreement between them and the primary shape is stated, not
resolved.

- **D1: scheduling dominates.** If handoff + runtime + client_spawn ≥ 50% of
  above_floor, the next transport work is **scheduling**: dedicated
  per-connection worker threads and persistent per-shard client senders on the
  existing framing. A new wire framing is not justified by BUS-0.
- **D2: framing dominates.** If http ≥ 50% of above_floor, a framing change
  (LCP-1 over persistent TCP) is justified as a candidate intervention.
- **D3: mixed.** If neither holds, any LCP-1 freeze must forecast the scheduling
  and framing components separately.
- **D4: the floor does not explain the model runs.** Evaluated independently.
  It applies if T3s's p50 is below 50% of the in-model per-call remainder for
  the matching shape:
  - PROFILE-1: 232 µs for `gptoss-experts`;
  - WIRE-2 HTTP: 139 µs for `qwen3-dense` and 177 µs for `gemma3-dense`.

  Then most of the in-model remainder is **not intrinsic to transport on an
  idle host**. The leading suspect becomes contention between the transport
  threads and the compute pool (PROFILE-1 runs eight Rayon threads per
  process). D4 takes precedence over D1–D3 for choosing what comes next. The
  next step is a loaded-host measurement, not a protocol.

Whatever the outcome, **RESIDUAL-BUS-1** (every canonical VINDEX3 carrier write
through one semantic authority: `leave_site`, `leave_batch_site`, and the Kimi
KDA/MLA inline adds) proceeds. It does not depend on BUS-0.

## Known limits, fixed now

- Loopback and one host: this is not LAN, and says nothing about NIC latency or
  bandwidth.
- An idle host: real runs have the compute pool busy. D4 exists because of this.
- T3's round trip includes the server's body receipt. PROFILE-1's `handler_ns`
  starts before `Bytes::from_request`, so its remainder does **not** include
  body receipt. WIRE-2's stream remainder does include message assembly. The
  comparison in D4 is therefore approximate in a known direction: T3 includes
  more than the PROFILE-1 remainder does.
- T0–T2 use a four-byte length prefix. This is a floor for framing, not a
  proposed LCP frame.
- No codec runs. Wire encode/decode is measured separately in PROFILE-1
  (0.3–0.8 ms per position summed).

## 2026-09-26 erratum: BUS-1's scope

The decision rule's closing paragraph names "the Kimi KDA/MLA inline adds" as
a write path BUS-1 must bring under `leave_site`. BUS-1 reconnaissance found
that production Kimi already runs through `leave_site` and `leave_batch_site`.
The inline adds are a hand-composed oracle stack
([`residual-bus-1-reconnaissance.md`](residual-bus-1-reconnaissance.md) §2).
BUS-0's arms, measurements and decision are unaffected.
