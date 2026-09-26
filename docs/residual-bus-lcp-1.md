# RESIDUAL-BUS / LCP-1: reconnaissance for a residual carrier bus and a binary carrier protocol

**Class: RECONNAISSANCE.** Dated 2026-09-26, qualified against main `1248287f`
(#590). This page records what the source and the existing records say
**before** any design is frozen or any code is written. It proposes a freeze
(§7) but freezes nothing. The freeze will be a separate, dated commit.

## The proposal being examined

Two ideas, raised together:

1. **A residual bus.** Treat the residual carrier as the message that moves
   between executable components. Its semantic verbs are READ (observe),
   WRITE (commit a delta), ROUTE (execute elsewhere), FORK and JOIN. Local
   execution lowers the bus to borrowed memory; remote execution lowers it to a
   transport. Observation, intervention, distribution and representation
   become subscribers, transforms, placement and codecs of the one bus.
2. **LCP, the LARQL Carrier Protocol.** A small binary protocol over persistent
   TCP with length-prefixed frames. It has a fixed header (identity, sequence,
   position, layer, site, representation, dtype, count), scatter/gather I/O, and
   HELLO/CAPABILITIES negotiation. The protocol is separate from the transport
   (in-process / SHM / TCP / QUIC / RDMA). The event plane is kept separate
   (Kafka-shaped) from the hot data plane.

This page asks what already exists and what the measurements say. It then asks
which parts of the proposal the evidence supports and which it contradicts.

## 1. The bus already exists semantically, on one path

In the VINDEX3 decode executor
(`E = crates/larql-vindex/src/format/vindex3/opplan/exec/`):

| Bus verb | What exists | Where |
|---|---|---|
| WRITE | `leave_site` is the single fold of a branch delta into the carrier. It has three forms: single-stream `residual_add`, bundle `hyper_connection::update`, and history `write` (add **or** replace). | `E/decode.rs:1460`, called at `:987` (attention) and `:1085` (FFN) |
| READ | `StepObserver::carrier_write(CarrierWriteRecord{layer, site, position, delta, after, layer_scale})` and `entering_carrier` | `E/observe.rs:122`, `:279` |
| (mutate) | `InterventionKind {Zero, Add, Replace}` at `Address{layer, site, positions}`, applied *after* the write lands, inside `leave_site`. Bundle/History are refused. | `E/intervene.rs:38`, `:113`, `:350-363`; `E/decode.rs:1514-1532` |
| ROUTE (whole layers) | `ExecutionSlice::LayerRange`, plus `step_from_carrier_intervened` taking an external carrier; `ResumePoint{next_layer, hidden: Plane}` | `E/prepared.rs:89`; `E/decode.rs:484`; `E/mod.rs:386` |
| ROUTE (inside FFN) | `ExecutionSlice::{DenseFfnCoordinator, RoutedExpertCoordinator, DenseFfns, RoutedExperts}` | `E/prepared.rs:89` |
| (record) | `RunIdentity` + `RecordedEvent{sequence, timestamp_ns, position, event}`: a lossless sequenced record and a lossy live tap (V3-STREAM-1) | `crates/larql-inference/src/vindex3/record.rs:46`, `:303` |

The contracts are already written: V3-OBS-1 (C1–C6, "every carrier write on every
topology passes through one function"), V3-INTERVENE-1 (I1–I9) and
V3-STREAM-1 (S1, S2). The "event plane" in the proposal is V3-STREAM-1's record.
Kafka would be one sink for it and has no role on the hot path.

**Where "one bus" is not yet true.** A bus claim has to cover every path that
writes the carrier. Today three do not go through `leave_site`:

- **Batch/prefill** has its own choke point, `leave_batch_site`
  (`E/mod.rs:1959`). It emits `PlaneEvent`/`LayerTrace`, not `StepObserver`
  calls. V3-OBS-1 records batch/decode parity, but these are two
  implementations, not one.
- **The Kimi stack adds inline** (`E/kimi_kda_layer.rs:102,136,209,234`;
  `E/kimi_mla_layer.rs:99,133`) and bypasses `leave_site` entirely.
- **The legacy path** has many add sites, including adds fused into Metal
  kernels (`larql-compute-metal/src/decode/encode_post_ffn.rs:88,123,150`). By
  the programme's rule, the legacy path is a migration target, not a bus
  participant.

**What the address lacks.** There is no `CarrierTopology` type
(`ResidualTopology` lives in `larql-models`). The site is `SublayerSite
{Attention, Ffn}`; there is no finer site. Identity is held by the runner
(`RunIdentity`), not the executor. There is no sequence/stream id inside the
executor.

## 2. The wire today

Routed experts, VINDEX3 (`crates/larql-router-protocol/src/vindex3_experts.rs`):

- Uses `GET` binding, then `POST …/open` (JSON, which returns a 16-byte
  process-incarnation handle), then `POST /v1/vindex3/experts/binary`.
- The 40-byte LE header is `magic, handle[16], sequence u64, layer u32,
  hidden u32, count u32`.
- **Request `VEX1`:** the header, then count × expert id, then **one** f32
  carrier of `hidden` values.
- **Response `VEY1`:** count × (id + an **unweighted** expert output of
  `hidden` f32). The coordinator accumulates the outputs in production
  selection order (`reduce_selected`).
- Telemetry travels in an HTTP header, outside the frame.

Dense FFN has the sibling ABI `VFF1`/`VFR1` (36 bytes) and an experimental
**persistent WebSocket stream** (WIRE-2, `docs/ffn/v3-ffn-wire.md`). The stream
uses one connection per worker, at most one outstanding operation and strictly
increasing sequences. Any fault ends the stream, and there is no reconnection.

The client for routed experts is `HttpExpertShards`: blocking `reqwest`,
`pool_max_idle_per_host(1)`, and one thread per shard per layer. Fan-out joins
every dispatch and fails closed.

The slot a new transport would fill: `ExpertTransport{bindings, forward}`
(`crates/larql-inference/src/vindex3/routed_experts.rs:22`) and `FfnTransport`
(`dense_ffn.rs:201`). There is also legacy scaffolding: UDS, gRPC
(`proto/expert.proto`, which carries routing weights on the wire, unlike V3),
and feature-gated QUIC/H3. None of it is wired into the V3 path.

## 3. Measured baselines

All figures are loopback on the M3 Max, provisional and not peer-exclusive.

**PROFILE-1**, GPT-OSS 20B routed experts, 24 layers, hidden 2880, top-4
(`bench/v3-routed-experts/PROFILE-1-results.md`). The figures are from the
passing block 0, in ms per decode position:

| | One worker | Expert split |
|---|---:|---:|
| Local position (reference) | ~116.3 | ~116.2 |
| Maximum shard compute | 86.391 | 81.000 |
| Fan-out wall | 95.946 | 90.104 |
| **Critical HTTP round trip − handler** | **5.562** | **5.320** |
| per critical call (÷24) | ~0.232 | ~0.222 |
| Request / response bytes per call | 11,576 / 46,136 | — |

**WIRE-2**, dense FFN, persistent WebSocket vs binary HTTP
(`bench/v3-dense-ffn-profile/WIRE-2.md`). Transport remainder per token:

| Model (layers, bytes/call) | HTTP | WS stream | per call HTTP → WS |
|---|---:|---:|---:|
| Qwen3 0.6B (28, 4,132 B) | 3.886 ms | 2.373 ms | 139 → 85 µs |
| Gemma 3 4B (34, 10,276 B) | 6.025 ms | 3.664 ms | 177 → 108 µs |

WIRE-2's own conclusion: the stream **lowers transport overhead by about 40%
but does not establish a consistent end-to-end speedup**. Qwen improved by
1.7–1.9 ms/token. In Gemma's only drift-passing bracket (inner check), the
stream was slower, because useful work rose in both processes for reasons not
established.

## 4. What the evidence says about the proposal

**E1. The protocol is not where the time is.** On loopback, remote expert
*compute* is 81–86 ms per position. The whole transport remainder is 5.3–5.6
ms. That is ~5% of the position. It sets the ceiling on anything a new wire
protocol alone can recover.

**E2. Removing HTTP has already been measured once.** Replacing HTTP with a
persistent socket recovered ~40% of the remainder. About 85–108 µs per call
remains after WS. A raw loopback TCP round trip for ~4–46 KB should be a small
fraction of that (this is a forecast; it is unmeasured on this machine, see §7
step 0). So the residue is most plausibly **scheduling**, not framing:

- the tokio accept/read,
- the `spawn_blocking` hop to the compute pool,
- the wake-up back,
- on the client, one OS thread spawned per shard per layer (`std::thread::scope`).

A raw-TCP LCP that kept those hops would likely reproduce WS's result. If LCP
is to be tested at all, the hypothesis worth testing is **"a dedicated blocking
worker thread per connection, no async runtime on the hot path, and a
persistent client-side sender per shard"**, not "binary TCP instead of HTTP".

**E3. The structural cost is the hop count, and no protocol removes it.**
Decode is layer-sequential. Layer L+1's attention needs layer L's FFN write.
With attention local and FFN remote, every position pays **one synchronous
round trip per layer** (24 for GPT-OSS, 34 for Gemma 3 4B). The remaining
levers are:

- **placement:** a `LayerRange` worker costs one hop per *range*, not per
  layer, but #507 has it recompute the prefix to avoid remote KV;
- **batching positions**, which applies to prefill and verify, not
  single-stream decode;
- **fewer, larger slices.**

A bus design should count hops per position as a first-class quantity, beside
bytes.

**E4. "Return a delta" conflicts with a frozen rule.** V3-FFN-SLICE-1 constraint
1 fixes the wire contract: workers return **unweighted** per-expert outputs and
never see routing weights. The coordinator accumulates them in selection order,
because float addition is not associative. Responses are therefore top-k×
larger than requests (46 KB vs 11.6 KB per call).

A worker-side weighted combine would cut response bytes ~4×. It stays exact
only when one shard owns every selected expert in a layer and evaluates the
same accumulation expression in the same order. That is a new contract with its
own freeze, not a protocol detail. The proposal's other observation still
stands: the carrier *owner* performing the commit is what V3 already does.

**E5. Compact carriers are evidenced, but not for the gate a transport answers
to.** `larql-boundary` has real residual codecs (bf16, int8 clip/absmax, int4)
under `BoundaryContract` levels A–E. They are not used by the V3 executor.

GW-CAR-1a (`docs/gw-car-1a-results.md`, Gemma 3 4B, entering L24 carrier) found
that the following passed both held-out gates:

- f16;
- symmetric q8 (2,564 B/row);
- sparse top-32 (192 B/row, 53× smaller);
- PCA r384.

q4/q2 and PCA ≤ r256 failed at least one gate. Its gate, however, is a
**relational sufficiency gate at one interface**, and the record says so: no
faithful reconstruction, and no construction or latency claim. Every V3
distributed contract (PROFILE-1, WIRE-2, LAN-1) instead gates on **exact token
parity**, which any lossy codec breaks by construction.

A compact carrier on the wire therefore needs a quality gate (for example
MEASURE-PLAN's plan-v1 KL). It is its own rung, not a negotiation option.

There is a further limit on the byte win. Under E4's contract, the response is
the larger stream, and it holds expert outputs, not the carrier. Compressing
the carrier alone touches ~20% of routed-expert bytes.

A `representation` field in the header costs nothing and should exist. LCP-1
would admit only `DenseF32`.

**E6. Kafka.** It is not suitable for the data plane: a per-layer millisecond
broker hop would exceed the entire current transport remainder. As a *sink*
for the V3-STREAM-1 record (`RecordedEvent` / `CarrierStats`), it is plausible
but unrequired; the record is already sequenced and lossless.

## 5. What survives

- **The bus as a semantic interface.** It already exists as `leave_site` +
  `StepObserver` + `InterventionPlan` + `ExecutionSlice`. The engineering work
  is to make it *total*, not to invent it. That means covering
  `leave_batch_site`, the Kimi inline adds, and a sequence/stream id plus
  executor-side identity in the address. A bus that covers two of the three
  write paths is the "right conclusion, wrong authority" failure.
- **The protocol/transport split.** `ExpertTransport`/`FfnTransport` are already
  the seam. A new transport is an implementation beside `HttpExpertShards`, not
  a new execution model.
- **Hop count as a quantity.** This follows from E3 and is the most useful
  sharpening of the proposal.

## 6. What does not survive, or needs its own freeze

- "LCP is the fast protocol LARQL needs": E1 and E2 show the time is compute,
  then scheduling, then framing.
- Delta return as a protocol default (E4).
- Representation negotiation beyond `DenseF32`: this needs a quality gate, not
  a parity gate (E5).
- Kafka on any hot path (E6).

## 7. Proposed next step (NOT frozen)

> **2026-09-26 successor.** Step 0 is frozen as
> [RESIDUAL-BUS-0](residual-bus-0.md), **without** the forecast below. BUS-0's
> arms measure baseline mechanics, which are measured, not forecast. It fixes a
> decision rule instead (D1–D4), and adds a fifth arm, T3s, for the client's
> per-call thread spawn. The programme order is:
>
> 1. BUS-0, the floor;
> 2. BUS-1, carrier-write totality (decode, prefill, Kimi KDA/MLA);
> 3. BUS-2, `CarrierAddress`, identity and sequencing;
> 4. BUS-3, remote ROUTE lowering on the existing HTTP/WS;
> 5. LCP-1, only if BUS-0 selects D2 or D3.
>
> The text below is retained as originally written.

**Step 0: measure the floor before building anything (reconnaissance only).**
A standalone Rust microbenchmark on this M3 Max, loopback, for the PROFILE-1
payload shapes (11,576 B → 46,136 B) and the WIRE-2 shapes (4,132 B and
10,276 B, symmetric). It needs no model and no LARQL execution path. Arms:

| Arm | Meaning |
|---|---|
| T0 | persistent `TcpStream`, blocking echo on a dedicated server thread, `TCP_NODELAY` |
| T1 | T0 + the server hands each frame to a separate compute thread and waits (channel round trip) |
| T2 | T0 + tokio accept/read + `spawn_blocking` hop (the current server's shape) |
| T3 | the existing axum binary-HTTP route shape, against a no-op transform |

Report p50/p99 µs per round trip over ≥10k calls after warm-up. The difference
T3−T0 bounds what *any* protocol change can recover per call. T2−T0 isolates
the async-runtime/blocking-pool hop. T3−T2 isolates HTTP.

**Forecast (frozen with step 0's commit):**

- T0 ≤ 30 µs;
- T2−T0 ≥ 40 µs;
- T3 within ±30% of the WIRE-2 HTTP per-call remainder (139–232 µs).

If T2−T0 is small, E2's scheduling hypothesis is falsified. The remainder would
then lie in the client (thread-per-dispatch, the reqwest pool) or in the payload
path, and LCP-1 would be re-scoped accordingly.

**Then LCP-1 proper.** Its freeze would be written only after step 0:

- an `ExpertTransport` implementation over persistent TCP;
- the unchanged VEX1/VEY1 payloads, so parity is byte-identical by construction;
- whichever server shape step 0 identifies as the floor;
- arms and gates inherited from PROFILE-1: L/C/L brackets, 1% drift, token
  parity against `profile-results/control/ids.json`, no pooling of void blocks;
- a success criterion stated as critical round trip − handler per call, **and**
  the net position delta, reported separately.

**Explicitly out of scope for LCP-1:** representation negotiation, worker-side
combine, QUIC/RDMA, bus totality (the batch and Kimi paths), and LAN hosts.
Each of those is its own rung. Bus totality is the natural *next* one, because
it is correctness work that does not depend on any transport result.
