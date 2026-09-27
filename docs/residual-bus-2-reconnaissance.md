# RESIDUAL-BUS-2 reconnaissance: address, identity and sequence of a carrier

**Class: RECONNAISSANCE.** Dated 2026-09-27 at main `d298574e` (after
[BUS-1](residual-bus-1-results.md) closed). Read-only. Nothing is frozen here,
and §7 proposes questions for the freeze.

BUS-1 gave every carrier change a name (`CarrierTransition`) and one ordered
stream per traversal. BUS-3 will move execution across processes. Before it
can, a carrier state crossing a boundary needs three things:

- an **address**: where in execution the state belongs;
- an **identity**: which exact realization produced it;
- a **sequence**: whether this hand-off is the next legal one.

This document inventories what already provides each one, and which states can
actually cross a boundary. The three are kept separate on purpose: collapsing
them into one opaque id would hide exactly the gaps found below.

**Headline.**

- **Identity is the weakest of the three.** Only one identity crosses a
  process boundary, the router-protocol `Binding`, and on layer workers it
  names no realization. The one identity that records process arithmetic
  (`ExecutionProvenance`) is written to a file and never sent. Some settings
  that change values are recorded nowhere (§2).
- **Position is not one thing.** Decode positions are absolute continuation
  positions. Batch positions are relative to the chunk. That is harmless today
  only because chunked prefill runs with a no-op sink (§3).
- **One ordering rule is enforced in the repo**, on the VFF1 WebSocket stream.
  It refuses stale handles and non-increasing sequences, and accepts gaps.
  Nothing refuses gaps (§4).
- **Addressable ≠ serialisable ≠ resumable.** Only a single-stream Rows carrier
  crosses a boundary today, and every call recomputes from position 0, so no
  KV, recurrent or latent state ever travels (§5).
- **Two resume refusals are missing.** A Rows resume point on an
  attention-residual component is accepted and runs as a single stream. "`kv`
  and `resume` do not combine" is documentation, not a guard (§6).

Paths are at `d298574e`. Claims were collected by three read-only searches and the
load-bearing ones were re-read by hand. "Reachable" means a production chain
from `larql` or `larql-server` was traced. "Latent" means the code path exists
but nothing in production exercises it.

## 1. What crosses a process boundary today

| Path | Payload | Identity sent | Reachable from |
|---|---|---|---|
| Layer-prefix RPC, `/v1/vindex3/layers` | JSON rows for positions `0..n`, re-sent in full on every extend | `Binding{schema, artifact, backend, lowering, start, end, layers, hidden}` | `larql-server --layers`; `larql run --v3-shards` |
| Dense-FFN, HTTP binary (VFF1/VFR1) and WebSocket stream | one normalized row per call | `Binding` + per-operand realization strings, at `/open` only; then a 16-byte handle | `larql run --v3-ffn-shards`; `larql-server --ffn-only` |
| Routed experts (VEX1/VEY1) | one row + expert list per call | as dense FFN, plus byte-range `regions` | as above, with `--experts` |

Nothing else crosses. `ExecutionProvenance`, `ContinuationAuthority`, KV and
recurrent state, bundles and histories all stay inside one process. The server's
Responses-API KV handoff (`V3KvHandoff`) is in-memory and take-once.

## 2. Identity

### 2.1 Inventory

| Identity | Defined | What it covers | What it leaves out | Crosses a boundary? |
|---|---|---|---|---|
| `ExecutionSlice` | `crates/larql-vindex/src/format/vindex3/opplan/exec/prepared.rs:79-121` | layer/expert ranges and role | provider, realization, configuration (no serde) | no (implied by `Binding` ranges) |
| `LoweringIdentity` | `crates/larql-vindex/src/format/vindex3/opplan/exec/lowering.rs:45-51` | provider `family/vN` | provider configuration, **by design** (`crates/larql-vindex/src/format/vindex3/opplan/exec/lowering.rs:17-20`) | as the `Binding.lowering` string |
| `RealizationId` (exec) | `crates/larql-vindex/src/format/vindex3/opplan/exec/realization.rs:291-295` | backend + realization form per operand | process knobs outside the form (§2.3) | as `{:?}` strings in FFN/expert `OperandIdentity` only |
| `ExecutionProvenance` | `crates/larql-vindex/src/format/vindex3/opplan/exec/provenance.rs:48-61` | lowering, realization classes, arithmetic arm, kquant mode; SHA-256 fingerprint | per-operand identity, slice, artifact, other knobs (§2.3) | **no**: written to the `vindex3 observe` record file only |
| Artifact identity | `crates/larql-inference/src/vindex3/distributed.rs:28-33` | SHA-256 of declared `index`, `graph`, `plan` | the weight bytes themselves (declared hashes, not verified) | as `Binding.artifact` |
| `ContinuationAuthority` | `crates/larql-vindex/src/format/vindex3/opplan/exec/continuation_authority.rs:136-139` | continuation provider identity + config digest | artifact, plan, layer range, position | no (in-process handoff) |
| `RunIdentity` | `crates/larql-inference/src/vindex3/record.rs:44-66` | run id, model name, tokens, intervention digests | the computation; caller-supplied | no (record file) |

### 2.2 Findings

- **`ExecutionSlice` is scope, not identity.** It says what may execute. The
  backend is a separate argument to preparation, so one slice can be prepared
  by different providers. BUS-1's Granite run is the proof. The same
  `ExecutionSlice::Full` under Production and Reference gave different values
  and fingerprints (`5bc6d15a…` vs `336860d9…`).
- **`LoweringIdentity` is incomplete by design.** It excludes provider
  configuration, on the grounds that configuration shows up in the pinned
  realization form. That holds only for the knobs that reach the form.
- **The layer-worker `Binding` names no realization.**
  `crates/larql-inference/src/vindex3/distributed.rs:63` hardcodes
  `lowering: LoweringIdentity::cpu_production()` rather than reading it from
  the prepared image. Two `--layers` workers started with different
  `LARQL_CPU_MAX_FORMAT`, `LARQL_CPU_ARITHMETIC` or `LARQL_CPU_Q4_CLASSES` bind
  byte-identically and compute differently. [V3-FFN-SLICE-1](v3-ffn-slice-1.md)
  already recorded this as an open question.
- **The FFN/expert check compares selection, not arithmetic.** The coordinator
  re-selects each remote slice's realizations **in its own process**
  (`crates/larql-inference/src/vindex3/dense_ffn.rs:279-284`) and compares
  strings. It rejects a worker whose selection differs from the coordinator's,
  not one whose arithmetic differs.
- **Metal is not reachable here yet.** Device providers with different format
  tables share `device-matmul/v1`. Sharding is CPU-only today, so this does not
  reach a boundary.

### 2.3 Settings that change values but reach no identity

These knobs change arithmetic but reach neither the realization form nor the
`ExecutionProvenance` fingerprint:

- **Activation scale span, block vs tensor.** `q8xq8` and `q8xq8b` resolve to
  the same `ArithmeticArm` (`crates/larql-vindex/src/format/vindex3/opplan/exec/cpu/physical.rs:245-256`); only the scale
  geometry differs.
- `LARQL_CPU_ACT_BLOCK`.
- `LARQL_CPU_ACT_CODE`, symmetric vs asymmetric.
- `LARQL_CPU_BIT_IDENTICAL`, which selects K2 vs K3 kernels; K3 reassociates.

For the arm, kquant mode and `BIT_IDENTICAL`, a value change is established.
For scale span, block and code it is inferred from the kernel code and not
measured here. `LARQL_CPU_WORKERS`, `LARQL_CPU_STATIONARY`,
`LARQL_FFN_MULTI_POSITION` and `LARQL_F32_STAGE` are also unrecorded, but are
documented as bit-identical or residency-only.

## 3. Address

- **Layer is plan-absolute.** Both traversals compute `index = first_layer +
  offset`, and a `LayerRange` shard's `first_layer` is its start. `Enter` names
  the layer the carrier enters: `first_layer` from an embedding, and
  `next_layer` from a `ResumePoint`.
- **Site** is `SublayerSite{Attention, Ffn}`, with no finer grain.
- **Position is absolute in decode and chunk-relative in batch.**
  - Decode reads `kv.position()`: an absolute continuation position.
  - Batch always emits `0..n` (`crates/larql-vindex/src/format/vindex3/opplan/exec/batch_site.rs:220-233`), even when attention
    reads a nonzero KV base.
  - Today no transition is ever emitted at a nonzero base.
    `prefill_prepared` traverses with a no-op sink, and the event-emitting
    streaming entry refuses KV state that is not at position 0
    (`crates/larql-vindex/src/format/vindex3/opplan/exec/streaming.rs:104-111`).
  - The server does prefill in chunks: the Responses API resumes a KV handoff
    and prefills only the new suffix (`crates/larql-server/src/vindex3.rs:609-627`).
    Attaching a subscriber there would re-emit positions `0..n` and collide with
    earlier chunks. **Latent.**
- **Topology is missing from the transition.** `CarrierForm` appears only on
  decode's `StepEvent::CarrierWrite`. `CarrierTransition`,
  `PlaneEvent::Transition` and `CarrierWritePlane` do not carry it, and no
  transition names a hyper-connection stream.
- **There is no run or stream id in the executor.** `RunIdentity` lives in the
  runner's record, beside the provenance fingerprint.
- **Nothing on the wire carries a position.** VEX1 and VFF1 carry one row, with
  no position, token or step. The layer RPC's positions are implicit in row
  order and restart at 0 on every call.
- **The closest existing serialized address is the carrier capture file**
  (`crates/larql-cli/src/commands/primary/vindex3_cmd/intervention.rs:115-127`):
  `{run_id, captures: [{layer, site, position, sha256, values}]}`, with
  decode-absolute positions. `run_id` is attached but not checked against the
  current run.

## 4. Sequence

| Protocol | Sequence | Receiver on duplicate / gap / reorder / stale handle |
|---|---|---|
| VFF1 over WebSocket | `u64` per (client, shard), from 1, one in flight | **refuse / accept / refuse / refuse**; any fault ends the stream (`crates/larql-server/src/routes/vindex3_ffn/stream.rs:76-78`) |
| VFF1/VFR1 over HTTP | same counter | accepts all silently; the client checks request/response correlation |
| VEX1/VEY1 experts | one counter per shard, shared across layers and threads | accepts all silently; the client checks correlation |
| JSON FFN, layer-prefix RPC | none | not applicable; stateless or unordered |

- **The VFF1 stream rule is partly reusable.** Its refusal pattern can be
  lifted: stale handle, then non-increasing sequence, then end the stream. Its
  counter cannot. It is a per-connection transport counter, shared across
  layers, and it maps to no `(position, layer, site)`. Its handle names a worker
  process incarnation, not a session, and every coordinator shares it.
- **The only design that detects gaps is `RecordedEvent.sequence`**: +1 from 0
  per run, sealed with a SHA-256 over the event lines. No production reader
  verifies contiguity; `read_jsonl` has only test callers.
- **Batch and decode order events differently.** Batch is layer-major, then
  position. Decode is position-major, then layer. On hyper-connections the two
  paths also emit the site record and the `HcUpdate` transition in opposite
  orders. BUS-1's T1 buckets events by position, so it is blind to cross-position
  order by construction. A sequence defined over the global stream would differ
  between batch and decode; one defined per position would agree.
- **The KV position is not monotonic-checked.** `set_position` is documented as
  monotonic (`crates/larql-vindex/src/format/vindex3/opplan/exec/kv.rs:196-200`) but no provider enforces it.
- **No production code consumes transitions yet.** The CLI dump drops them. The
  only non-test reader is the `bus1_prefill_trace_cost` example.

## 5. What can cross, and what can resume

"Serialisable" means an existing writer and reader outside tests. "Resumable"
means entering mid-stack in another call or process.

| State | Addressable today | Serialisable | Resumable at a layer boundary | Explicitly refused |
|---|---|---|---|---|
| Rows carrier | yes | **yes**: layer RPC JSON, CLI plane files | **yes**: batch via `ResumePoint` (layer RPC, CLI `--resume`); decode entry is test/demo only | wrong width, count or topology (`crates/larql-vindex/src/format/vindex3/opplan/exec/traverse.rs:115-158`) |
| Bundle carrier (HC) | yes, without a stream index | no serde | in-process batch only (tests) | layer RPC, CLI dump, decode carrier entry |
| History carrier (attention residual) | yes | no serde | **no** | `ResumePoint` refuses it (`crates/larql-vindex/src/format/vindex3/opplan/exec/traverse.rs:159-176`), as do the decode entry and the RPC |
| Softmax KV rows | per layer, implicitly | no serde | whole provider, in-process only; no split by layer range | missing rows at a step (`crates/larql-vindex/src/format/vindex3/opplan/exec/backend/step_calls.rs:113-123`) |
| Recurrent (KDA, GDN, Mamba2, conv) | per layer, implicitly | no (`from_cells` is test-only) | whole provider, in-process only | no provider, or a KV-only provider |
| MLA latent cache | per layer, implicitly | no serde | whole provider, in-process only | `LatentUnsupported` on a KV-only provider |

- **A `ResumePoint` carries only `next_layer` and a plane.** It has no position,
  per-layer state, history snapshots or identity. Identity comes only from the
  CLI sidecar or the RPC `Binding`.
- **A layer-range shard cannot hold continuation state.** A shard can't prefill
  a provider (`crates/larql-vindex/src/format/vindex3/opplan/exec/traverse.rs:194-199`). That is why the layer RPC recomputes
  every position on every call.
- **The decode carrier entry `step_from_carrier_intervened` is test and demo
  only.** It accepts a single stream only and takes position from the caller's
  provider. It advances every layer's position after writing only the range's
  layers, and its output carrier is not returned outside tests.
- **`CanonicalKvState::into_cache` keeps only the KV cache.** Recurrent and
  latent state are dropped silently. Adopting a KV-only cache into a plan with
  those layers panics. Both paths are test-only.

So today "addressable" is broad: every transition can be named. "Serialisable"
covers Rows only, and "resumable across a process" covers Rows without
continuation state only.

## 6. Found on the way, for their owners

- **A Rows `ResumePoint` on an attention-residual component is not refused.**
  - The resume check matches only on the hyper-connection topology
    (`crates/larql-vindex/src/format/vindex3/opplan/exec/traverse.rs:133`). `hyper_connection()` returns `None` for attention
    residual (`crates/larql-vindex/src/format/vindex3/opplan/exec/prepared/accessors.rs:481-486`).
  - `enter_batch_site` takes the attention-residual path only for a Histories
    plane, so Rows fall through to the single-stream arm
    (`crates/larql-vindex/src/format/vindex3/opplan/exec/batch_site.rs:171-176`).
  - The boundary event fires only on Histories (`crates/larql-vindex/src/format/vindex3/opplan/exec/layer_exec.rs:72-73`), and
    the Rows exit skips the attention-residual exit reduction.
  - Production is guarded upstream: the layer RPC refuses attention residual,
    and the CLI dump fails at plane 000. A hand-made plane file fed to
    `exec --resume` is not refused on any path found.
  - **Shown by execution** after this reconnaissance was written. A test
    resuming rows into the attention-residual substrate runs and emits a
    single-stream carrier write, which BUS-1's T5 guard in the witness sink
    rejects. A refusal is on branch `fix/bus2-resume-refusals`.
- **"`kv` and `resume` do not combine" is not enforced, and the premise is
  wrong.**
  - The streaming path passes both to `traverse`, checking only
    `position() == 0` (`crates/larql-vindex/src/format/vindex3/opplan/exec/streaming.rs:104-124`). Layers below `next_layer` get
    no rows, and the position is not advanced.
  - **Correction, found while implementing a refusal.** The two MUST combine.
    A resumed plan with recurrent or latent layers cannot run without a
    provider, because `traverse` refuses those layers. The CLI's `exec --resume`
    does exactly that, over a scratch state from `one_shot_state` that it then
    drops. A refusal as documented would break a working command.
  - **The real hazard is continuing from any one-shot provider**, resumed or
    not. The streaming path never advances the position, so such a provider
    holds state at position 0. Softmax layers would fail closed on the next
    step (`rows.end() != position`); recurrent and MLA layers would continue
    silently.
  - Every production caller (the CLI's `exec`, inference's `stream_over`)
    builds a fresh provider, runs one traversal and drops it. None continues.
    The public `&mut dyn KvState` signature is what allows it.
- **`next_layer` is bounded by the plan, not the slice** (`crates/larql-vindex/src/format/vindex3/opplan/exec/traverse.rs:115`).
  Below the shard it silently runs the whole shard; above it runs zero layers.
- **A provider layer-count mismatch panics** (`assert_eq!`) instead of refusing.
- **MLA takes its position from `state.len()`**, and never checks it against the
  provider's position (`crates/larql-vindex/src/format/vindex3/opplan/exec/mla.rs:311-313`).

## 7. What BUS-2 should decide (NOT frozen)

The evidence supports three composed values, not one id:

- **`CarrierAddress`**: `{base + position, layer, site, form}`.
  - Position must be absolute, which means batch needs a chunk base.
  - `form` belongs in the address, because the same `(position, layer, site)`
    means different things on Rows, Bundles and Histories.
- **`ExecutionIdentity`**: artifact, lowering, per-operand realizations, and a
  provenance fingerprint extended to cover §2.3.
  - It must be **read from the prepared image**, never hardcoded, because
    `distributed.rs:63` is exactly how a binding and an image disagree.
- **`CarrierSequence`**: `{run or stream id, ordinal}`, checked by the receiver.

Questions the freeze must answer:

1. **What does "same" mean across workers?** Same identity: bit-identical
   transitions, as in BUS-1. Different backend or lowering: identical structure
   only, with value agreement reported, not asserted. This must be declared
   before the first mixed run.
2. **Which settings enter the fingerprint?** Either extend `ExecutionProvenance`
   to the §2.3 knobs, or refuse to shard under any knob it cannot see.
3. **What is the sequence scope: per run, per position, or per routed segment?**
   A per-position ordinal agrees between batch and decode (§4). A global one
   does not.
4. **Should gaps be refused?** No receiver refuses them today. VFF1 accepts
   them. A strict rule would fail closed on duplicate, gap, reorder, stale
   identity and wrong address.
5. **What is routable?** The freeze should make "valid address + non-resumable
   state ≠ routable" explicit. Bundles, histories and all continuation state
   should be named as non-portable, and refused rather than implied. A
   portable form for any of them is its own later rung.
6. **Decided: BUS-2 owns the §6 resume refusals, and they close BUS-2's
   pre-freeze work.** The freeze is not written over known permissive
   behaviour. The sequence is: reconnaissance → the refusals → re-check the
   assumptions here → freeze address, identity and sequence. The second
   refusal needs its shape decided first; the premise correction in §6 says
   why.

Candidate rules, to be decided, not decided here:

- same address + different identity ≠ same carrier;
- same identity + wrong sequence = refuse;
- valid address + non-resumable topology ≠ routable.
