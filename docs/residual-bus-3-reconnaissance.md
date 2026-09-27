# RESIDUAL-BUS-3 reconnaissance: moving an addressed carrier across a worker boundary

**Class: RECONNAISSANCE.** Dated 2026-09-27 at main `f8b61034` (after
[BUS-2](residual-bus-2-results.md) closed). Read-only. Nothing is frozen here,
and §8 proposes questions for the freeze.

The question handed to this rung was:

> Can a portable, addressed, identity-bound, sequenced carrier cross an existing
> worker boundary and execute an `ExecutionSlice` remotely with exactly the same
> semantics as local execution?

It came with a scope: Rows only, the existing HTTP and WebSocket transport, no
LCP, and hop reduction as the main systems question. Three shapes were proposed:

- **A.** A remote FFN or expert call: proven infrastructure, but one hop per layer.
- **B.** A remote whole layer, attention plus FFN.
- **C.** A remote `LayerRange`: one carrier in, several layers, one carrier out.

**Headline.**

- **Shape C already exists, and it is the only one whose wire carries a
  carrier.** The layer-prefix RPC is a `LayerRange` route. It is stateless by
  ABI: every step re-sends and recomputes every position, as JSON. So each hop
  costs O(n) in both bytes and compute, and it has never been measured (§2).
- **Shape A's wire carries a branch input, not a carrier.** The FFN and expert
  workers receive a normalised row and return branch output. The carrier
  transition happens on the coordinator. BUS-2's address and sequence describe a
  carrier, and they do not fit this wire without redefinition (§2, §5).
- **Shape B does not exist separately.** It is shape C with a one-layer range.
- **Continuation state does not have to be portable for C to be exact. It can
  stay resident on the worker that owns those layers**, so only carriers cross.
  The executor already runs this in-process. A layer-range decode session over a
  provider, fed one carrier per token, runs its layers and advances position.
  Every piece around it is missing in production (§3).
- **No production path checks the bytes of the model it runs, and none checks
  the rows it receives.** The remote results have been compared with local ones
  only within `1e-5`. Nothing asserts a remote carrier is bit-identical (§4).
- **The BUS-2 sequence guard counts transitions, and a hop hands off a plane.**
  The freeze must define the unit a receiver guards (§5).

Paths are at `f8b61034`. Claims were collected by three read-only searches, and
the load-bearing ones were re-read by hand. "Reachable" means a production chain
from `larql` or `larql-server` was traced.

## 1. The three remote paths today

| | Layer-prefix RPC | Dense-FFN worker | Routed-expert worker |
|---|---|---|---|
| Server | `larql-server --layers A-B` | `--ffn-only --layers` | `--ffn-only --layers --experts` |
| Coordinator | `larql run --v3-shards` | `larql run --v3-ffn-shards` | same, on a routed plan |
| Wire | JSON `{binding, rows}` | VFF1/VFR1 binary (HTTP or WebSocket); JSON control | VEX1/VEY1 binary |
| Payload | the **residual carrier**, every position so far | one **normalised FFN input** row, one branch output back | one expert input row; per-expert **unweighted** outputs back |
| Worker state between calls | none, by ABI | none | none |
| Coordinator state | the list of inputs only; no KV | the whole continuation | the whole continuation |
| Identity | full `Binding` (schema 2, digest) on every request and response | full `Binding` at `/open`, then a 16-byte handle | as FFN |
| Order | none | per-connection counter; enforced on the WebSocket stream only | per-shard counter, not enforced |

The layer RPC's ABI states its rules in the code:

- "V3 stateless layer-prefix RPC. Positions always start at zero; no remote
  continuation cache, retries or hidden server affinity are part of this ABI"
  (`crates/larql-router-protocol/src/vindex3.rs:1-2`).
- "Each step recomputes the whole prefix, so remote failure cannot desynchronise
  KV state" (`crates/larql-inference/src/vindex3/distributed.rs:1-3`).

## 2. Hops and cost per shape

| Shape | Hops per decoded token | Bytes per hop | Remote work per token | Measured |
|---|---|---|---|---|
| A: FFN / expert | L, sequential | O(hidden) | one layer's FFN on one row | yes: WIRE-2, PROFILE-1, BUS-0 |
| B: one-layer range | L (S = L) | O(n·hidden) | the whole prefix, one layer | no |
| C today: stateless range | S, sequential | O(n·hidden), JSON, both ways | the whole prefix through the range | **no** |
| C with worker-resident state | S | O(hidden) | one position through the range | not built |

In the table, L is the layer count, S the shard count and n the sequence length.

- **Measurements that exist:**
  - Dense FFN (WIRE-2): 3.9 ms/token of transport on Qwen3 0.6B and 6.0 ms on
    Gemma 3 4B over HTTP, falling to 2.4 and 3.7 ms on the stream.
  - Routed experts (PROFILE-1): about 5.5 ms/position of critical round trip on
    GPT-OSS 20B.
  - BUS-0 puts the idle wire stack at 51 to 77 µs per call.
- **The layer RPC has no measurement.** `docs/vindex3/runtime-followups.md:89`:
  "No distributed throughput claim has been measured."
- **Where C's hop reduction lives.** It runs S hops per token instead of L. Today
  every one of those hops is O(n), because the coordinator re-embeds and re-sends
  every stored input (`crates/larql-inference/src/vindex3/distributed.rs:260-273`)
  and each worker recomputes the whole prefix through its layers. The saving S/L
  is real only if the per-hop cost stops growing with n.

## 3. The state boundary

**What must cross, and what can stay:**

- A carrier crosses: Rows only. Bundles and histories stay refused by BUS-2's
  `ensure_portable`.
- Continuation state (KV, recurrent, latent) **never crosses**. There are two
  ways C can be exact without moving it:
  1. **Stateless recompute (today).** The worker rebuilds its layers' state from
     position 0 on every call, so none persists. It is exact in principle, but
     O(n) per hop, and it is limited to softmax: the stateless forward passes no
     provider, so `ensure_supported` refuses any plan with a non-softmax layer
     (`distributed.rs:45-52`). A one-shot provider consumed by value (#617) could
     let recurrent layers recompute the same way. That is not built.
  2. **Worker-resident continuation.** The worker keeps its own provider for its
     layers across steps. Only one carrier crosses per token.

**The executor already does (2) in-process.** `DecodeSession::over_prepared`
accepts a `LayerRange` image. `step_from_carrier_intervened` enters one carrier,
runs the slice's layers against a resident provider, and advances position
(`crates/larql-vindex/src/format/vindex3/opplan/exec/decode/steps.rs:304-328`). It
is reached only from tests and research demos. Six pieces are missing in
production:

| Missing | Where it stops today |
|---|---|
| A production caller of the carrier entry, and a non-intervened form | callers are tests and `larql-demos` only |
| The step's output carrier, returned | `StepRun::exit` is `#[cfg(test)]` (`exec/decode.rs:183-185`); callers rebuild it from `carrier_write` |
| A layer-range **prefill** into a provider | `prefill_prepared_observed` always passes `resume = None` (`exec/streaming.rs:250-259`), and a range has no embedding table (`exec/traverse.rs:214-221`) |
| Slice-aware provider selection | every provider is prepared against the whole plan. `window/v1` and `codec/v1` refuse a softmax slice of a hybrid plan because of layers outside the slice (`exec/kv.rs:230-252`) |
| A per-stream holder on the server | the only resident state is the Responses-API KV cache: take-once, keyed by response id, 4 entries, 600 s TTL, **no byte budget** (`crates/larql-server/src/response_kv/mod.rs:46,51`), and not wired to the layer route |
| Position on the wire, checked against the worker's provider | the request is `{binding, rows}`; positions are implicit |

**The position hazard is exactness, not bookkeeping.**

- Only softmax ties provider position to held rows: `rows.end() == position`
  (`exec/backend/step_calls.rs:113-123`).
- MLA takes its position from its own cache length.
- Recurrent state carries no position at all.
- Conv checks only coverage.

So a worker-resident route must carry the expected absolute position and refuse
unless it equals the worker provider's position, **before any state moves**.
Otherwise a lost or duplicated step silently desynchronises MLA or recurrent
state, which is the failure the stateless ABI was designed to avoid.

## 4. Exactness authority: bytes

**Level 1: model weights.**

- **What is declared.** The index declares `payload_sha256` and `segment_sha256`
  per representation, meaning one hash per segment file. There are no per-tensor
  hashes.
- **Where bytes are verified.** Only by operator commands: `larql verify`,
  `larql vindex3 inspect --verify`, `vindex verify` and `larql vindex3 verify`.
- **Production never checks them.** `Vindex3Runtime::open` inspects with
  verification off and never reads the inspection's defects. The server's load
  path, `OperandStore::open`, and `artifact_identity` do not verify either. So two
  workers with identical `index.json` and different segment bytes present
  identical identities. Nothing catches finite wrong values.
- **What verification would cost.** Every existing path hashes whole segments;
  none is lazy. An expert worker reading only its byte window has nothing it can
  verify that window against. **No verify timing exists for any real container.**

**Level 2: carriers in transit.**

- **What receivers check.** Width, count and finiteness, plus exact frame length
  on binary frames. There is **no checksum**. TLS and a bearer token exist but
  are operator-configured.
- **Serialisation.** Binary frames are bit-exact by construction and tested.
  JSON uses serde's `float_roundtrip`, but **no test asserts a bit-exact f32 JSON
  round trip**; the one JSON row test uses `==` on values like 0.25, which cannot
  tell `-0.0` from `0.0`.
- **What exists to build on.** `vector_sha256`
  (`crates/larql-vindex/src/format/vindex3/opplan/exec/intervene.rs:97-109`)
  already digests a row, and the capture files use it.
- **The end-to-end tests use tolerances.** Both compare remote with local logits
  within `1e-5`:
  - `crates/larql-server/tests/test_vindex3_serve/eos_backends_and_layer_workers.rs:468`
  - `crates/larql-inference/src/vindex3/tests/inputs.rs:184`

  **Nothing asserts that a remote carrier is bit-identical to local.**

## 5. The sequence guard's unit

BUS-2's `SequenceGuard` checks per-position ordinals of carrier *transitions*,
with `declared_transitions(plan, ops)` giving the count. The remote paths hand
off something else:

- The layer RPC hands off one **plane** per range per call. Its worker's sink
  sees `PlaneEvent::Layer`, and never forwards transitions.
- An FFN or expert worker sees only its layers' FFN branch. The
  attention-site, `Enter` and `Scale` ordinals are emitted on the coordinator. As
  written, the guard would refuse such a stream with `Gap`.
- An expert layer fans out to several workers, and each sees a disjoint part of
  one position's hop.
- `declared_transitions` on a `LayerRange` image is untested. BUS-2's witnesses
  prepared full images only.

So the freeze must choose the unit a receiver guards. There are two options:

- **The worker emits its transitions,** and the guard checks them as BUS-2 built
  it.
- **A hop-level sequence** with one ordinal per carrier hand-off per position,
  guarded with the same fail-closed rules.

## 6. The three shapes against the question

| | Carrier on the wire | Exact without moving continuation state | Fewer hops than L | Built |
|---|---|---|---|---|
| A: FFN / expert | no, a branch input | yes: state stays on the coordinator | no | yes |
| B: one-layer range | yes | stateless: yes (O(n)); resident: as C | no | as C |
| C: layer range, stateless | yes | yes, softmax only, at O(n) per hop | yes (S hops) | yes |
| C: layer range, worker-resident | yes | yes, if position and order are checked before state moves | yes (S hops) | executor only |

Only shape C answers both questions at once: a carrier crosses, and the hop
count falls. Its stateless form is exact today and pays O(n) per hop. Its
resident form is the one whose per-hop cost does not grow with n.

## 7. Found on the way, for their owners

- **`larql pull` accepts corrupt containers.**
  - `validate_downloaded_container`
    (`crates/larql-vindex/src/format/vindex3/verify.rs:55`) calls
    `inspect_container(dir, true)?` and discards the result.
  - The `?` catches only I/O and parse errors. A hash mismatch is recorded as a
    `PayloadCorrupt` defect that nothing reads.
  - So a corrupt graph-container segment passes `larql pull` and registry
    resolution.
  - The fix is to refuse unless `is_coherent()`, with a corrupted-segment test.
- **Verification checks one of the two declared hashes.** `inspect_container(.., true)`
  checks `segment_sha256` only. `payload_sha256` is re-checked only by
  `vindex verify`, which reads each whole segment into memory.
- **Conv-QKV's position check is weaker than softmax's.** It checks only
  coverage, so a provider holding extra rows passes.
- **`window/v1` and `codec/v1` can't serve a softmax-only slice of a hybrid
  plan.** They refuse because of layers the slice never runs.
- **The Responses-API KV cache has no byte budget,** only an entry count and a
  TTL.

## 8. What BUS-3 should decide (NOT frozen)

1. **Which shape first?** The evidence points to C: it is the only shape whose
   wire carries a carrier, and the only one that reduces hops. The choice is
   between two orders:
   - make today's stateless C exact first (bit-equality, binary rows, digest,
     guard), then add worker-resident state;
   - go straight to worker-resident C.

   The staged order separates exactness from state management, as BUS-2
   separated identity, address and order.
2. **What exactly does "exact" assert?** Bit-identical carriers against the same
   identity running locally. The first act is to turn the two `1e-5` tests into
   bit equality, and record whether they already are.
3. **Who verifies weight bytes, and when?** The options are eager at worker start
   (measure the cost first; none is measured), per-object on first touch, or new
   per-tensor hashes in the format. Whichever is chosen, the identity must carry
   a "bytes verified" fact the coordinator can require.
4. **How are carriers checked in transit?** A per-carrier `vector_sha256`,
   binary rows, or both.
5. **What unit does a receiver guard?** The worker's own transitions, or a
   hop-level ordinal (§5).
6. **For worker-resident C:** the stream holder's key, its bounds (bytes, not
   just an entry count) and its failure semantics. When a step fails, does the
   stream die and recover by recompute, as the dense-FFN session does today?
7. **A baseline before any cost claim.** The layer RPC has never been measured.
   Measure it first, as a same-window A/B (BUS-1's T8 lesson).
