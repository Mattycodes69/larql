# RESIDUAL-BUS-2: the address, identity and sequence of a carrier

**Class: FREEZE.** Frozen 2026-09-27 against main `8afd0fbd`, before any
BUS-2 implementation. It follows the [reconnaissance](residual-bus-2-reconnaissance.md)
(#615) and its two pre-freeze refusals (#616, #617). The question, properties,
witnesses and forecasts below change only by a new, dated section.

Programme: RESIDUAL-BUS. [BUS-1](residual-bus-1.md) named every carrier change
(`CarrierTransition`) in one ordered stream per traversal. BUS-3 will move
execution across processes. BUS-2 fixes, before any transport exists, what a
carrier state must carry to be accepted anywhere but where it was made.

## The question

> Can every carrier transition carry an **address** that places it absolutely,
> an **identity** derived from the prepared image that produced it, and a
> **sequence** a receiver can check — so that a stale, foreign, duplicated,
> skipped or reordered state is refused rather than executed — without changing
> arithmetic, execution order or any unobserved result?

The three are separate values that compose. They are never one opaque id,
because each fails differently:

- same address + different identity ≠ same carrier;
- same identity + wrong sequence = refuse;
- valid address + non-portable state ≠ routable.

## Decisions taken in this freeze

Each is a choice between options the reconnaissance left open. D1, D4 and D9
were strengthened in review before this commit.

| # | Decision | Rejected alternative, and why |
|---|---|---|
| D1 | **Identity is derived from the prepared image and binds the effective model**, never from CLI syntax, a slice or a hardcoded provider. The same executable realization is not enough: it must be the same model | `ExecutionSlice` is scope (recon §2.2); the layer binding's hardcoded lowering (`distributed.rs:63`) is exactly how a binding and an image disagree. An identity of realizations alone would let two different models with the same layout and arithmetic share a digest |
| D2 | **Record** every value-changing process setting in the identity | Refusing to shard under unrecorded settings: all four (§2.3 of the recon) are process-global values with public accessors, so recording is cheap and keeps sharding usable |
| D3 | **Form is part of the address; portability is validation** | Encoding resumability in the address: it is an executor capability that will grow, and every new capability would change the address format |
| D4 | **Sequence is two-dimensional**: a declared domain of absolute positions per operation, and an ordinal within each position, bound to a stream identity `(identity digest, run id)`. Gaps, missing whole positions and truncated positions are all refused | VFF1's per-connection counter means nothing across a reconnect or a second worker; a global ordinal differs between batch (layer-major) and decode (position-major); independent per-position guards would accept a position that never arrived |
| D5 | **Exact mode refuses on identity mismatch**; it never degrades to a structural comparison | Silent degradation is the plausible-wrong-model shape in another place |
| D6 | **Batch positions become absolute**, and chunked prefill becomes observable through an **observation and testing surface**, not a new product API | Leaving batch chunk-relative: the first subscriber on the server's chunked prefill would get colliding positions (recon §3) |
| D7 | The sequence guard has **no production transport consumer in BUS-2**; BUS-3 adopts it | Building transport here would let BUS-2 invent routing semantics while fixing identity |
| D8 | **An image whose model authority cannot be established has no exact identity**, and exact routing refuses it. Today that covers an overlaid operand source, and any source not opened from a container | Deriving a content digest for overlays now: `OperandOverrides` carries only a process-unique `(id, generation)` (`operands.rs:357-368`), which cannot cross a process. A portable overlay identity is its own rung |
| D9 | **The binding change is versioned.** `Binding.schema` goes from 1 to 2 with the digest, and a schema-1 peer is refused by name | Adding a field to an assumed-stable shape: `deny_unknown_fields` would fail, but with a generic message, and exact mode must say *why* it refused |

## 1. Frozen properties

### Identity

**I1: identity comes from the image, and binds the effective model.**
`ExecutionIdentity` is built in one place, from a prepared image plus the
model authority established by whoever opened the container. It holds:

- **the model authority**: today's artifact identity, SHA-256 over the container
  index (which declares every payload's hash), the graph and the exact plan
  (`crates/larql-inference/src/vindex3/distributed.rs:26-33`);
- **the effective operand source**: the base container, and for an overlaid
  source the fact that it is overlaid (D8);
- **the slice**, as scope inside the identity;
- the lowering provider(s), read from the image's pinned realizations;
- every operand's pinned realization (representation, codec, form), in plan
  order, not aggregated to class counts;
- the process arithmetic: arithmetic arm, kquant mode, activation scale span,
  activation block, activation code, and the K2-only (`BIT_IDENTICAL`) switch.

Its digest is SHA-256 over a canonical, versioned serialisation of all of it.
An identity without a model authority, or over an overlaid source, is
**unanchored**. It is still recorded, but no exact claim or exact route may use
it.

The model authority names **declared** payload hashes. Byte verification
(`vindex verify`) is a separate authority. Whether an exact route must verify
bytes before binding is left to BUS-3 (§6).

**I2: every value-changing setting has a fate.** A conformance test enumerates
every CPU arithmetic environment setting the executor reads and requires each
to be either **in the identity** or on an **exclusion list with a cited reason**
(documented bit-identical, or residency only). Today's exclusions are
`LARQL_CPU_WORKERS`, `LARQL_CPU_STATIONARY`, `LARQL_FFN_MULTI_POSITION` and
`LARQL_F32_STAGE`. A new setting with no fate fails the test.

**I3: identity crosses the boundary and is checked.**

- `Binding.schema` becomes 2 (D9). A schema-1 peer is refused with a message
  naming both schemas.
- The layer `Binding` reads `lowering` from the image (the hardcode is
  removed) and carries the identity digest.
- Worker and coordinator refuse a digest mismatch before any rows move.
- The dense-FFN and expert bindings also carry the digest. A worker whose
  arithmetic differs is refused even when its realization strings match the
  coordinator's re-selection (recon §2.2).

**I4: claims are scoped by identity.** Equal digests permit a bit-identity
claim. Unequal digests permit only a structural comparison, with value
agreement reported, not asserted (as V3-OBS-1 does across backends). A path
that requires bit identity refuses on a mismatch, and on an unanchored
identity (D5, D8).

### Address

**A1: one address.** `CarrierAddress{position, layer, site, form}`:

- `position` is the absolute continuation position;
- `layer` is plan-absolute (already true, recon §3);
- `site` is `SublayerSite`, and absent for `Enter` and `Scale`;
- `form` is `CarrierForm{Single, Bundle, History}`.

Batch's `PlaneEvent::Transition` and decode's `StepObserver::transition` both
carry it, so a transition names its topology (recon §3: today only decode's
`StepEvent::CarrierWrite` does).

**A2: batch positions are absolute.** Batch transitions and carrier-write
records are offset by the provider's base position. A new observed prefill
entry makes chunked prefill observable. It is an observation and testing
surface beside the other observed entry points, documented as such, not a new
product API (D6). BUS-1's T1 and T6 continue to hold,
now at a nonzero base.

**A3: address ≠ routability.** One function decides whether an addressed state
may cross a process boundary. Its initial table:

| State | Portable |
|---|---|
| Single-stream rows, no continuation state | yes |
| Bundles, histories | no, refused by name |
| Any continuation state (KV, recurrent, latent) | no, refused by name |

The layer RPC's support check and the `ResumePoint` refusals consult it, so
the rule has one authority. A portable form for anything refused here is its
own later rung.

### Sequence

**S1: one sequence, in two dimensions.** `CarrierSequence{stream, position,
ordinal}`:

- `stream` is `(identity digest, run id)`;
- `position` is absolute (A1);
- `ordinal` is the transition's index within its position; `Enter` is 0.

Because BUS-1's T1 already makes per-position sequences equal between batch and
decode, their ordinals agree (D4). Batch emits layer-major, so positions
interleave on the wire. The guard never requires position-major arrival, only
that each position's own ordinals arrive in order.

**S2: the receiver fails closed.** A `SequenceGuard` is opened for one
operation, bound to:

- the stream identity;
- the layer range and carrier form the receiver serves;
- the operation's **declared position domain**: `base..base+n` for a prefill
  or a batch call, and exactly the next position for a decode step;
- the **declared transition count per position**, derived from the plan and
  slice (Enter, each `Add`, `Scale`, `HcUpdate`, `HistoryWrite` and boundary
  event the traversal performs), as BUS-1's F1 derives write counts.

It refuses **immediately**:

1. a stream whose identity digest differs from the bound one, or is unanchored
   in exact mode;
2. an address outside the bound layer range, or of the wrong form;
3. a position outside the declared domain;
4. a duplicate ordinal;
5. an ordinal gap or reorder within a position;
6. an ordinal beyond the position's declared count.

It refuses **at operation close**:

7. a declared position that never arrived;
8. a position whose ordinals stopped short of its declared count.

The first refusal ends the stream (VFF1's pattern). Unlike VFF1, gaps are
refused, and so are missing and truncated positions.

**S3: no arithmetic moves.** Address, identity and sequence are records beside
the existing writes. BUS-1's T2 and T7 continue to hold.

## 2. Witnesses

- **Identity.** I1 and I2 on the Reference and Production backends. I3
  through a subprocess witness: two processes that differ only in one setting
  from I1's list must produce different digests and refuse to bind. Every
  setting in the list gets one run.
- **Address and sequence.** The BUS-1 subjects (plain rows, the Gemma 4 layer
  scale, bundles, history with boundaries, mixed KDA/MLA) and the Granite 4.2
  3B container, on both backends:
  - A2: a prefill split into chunks emits, per absolute position, exactly the
    transitions of the unchunked prefill and of decode;
  - S1/S2: the guard accepts every batch and decode stream unaltered.

**Negative controls.** Each must fail at exactly the event it alters:

- one setting changed between two processes (I3);
- a hardcoded lowering reintroduced (I1);
- a batch position without its base (A2);
- a transition relabelled with the wrong form (A1);
- a duplicated, dropped or swapped transition (S2, immediate);
- a whole position withheld, with its neighbours intact (S2, at close);
- a position's last transition withheld (S2, at close);
- a position outside the declared domain (S2, immediate);
- a stream presenting a stale identity digest (S2);
- the same image over an overlay, and over a source with no container: both
  unanchored, and refused by exact mode (I1, D8);
- a schema-1 binding presented to a schema-2 receiver, refused by name (D9).

## 3. Baselines

This rung makes no performance claim. The identity digest is computed once per
prepared image and bind, never per token. If any timing claim is made later, it
is measured as a same-window A/B against the parent commit (BUS-1 T8: a band
drawn from one run's trial range is not a tolerance).

The behavioural baselines are the reconnaissance's code facts, re-verified at
`8afd0fbd`:

- the layer binding hardcodes its lowering and names no realization;
- `ExecutionProvenance` omits the four settings in I1 and crosses no boundary;
- batch positions restart at 0 per chunk;
- no receiver refuses a gap.

## 4. Forecasts (about the change only)

- **F1:** each setting in I1's list changes the digest.
- **F2:** chunked and unchunked prefill agree per absolute position on every
  subject, both backends. KDA/MLA is the subject that could fail, because
  recurrent state crosses the chunk boundary inside the provider.
- **F3:** batch and decode ordinals agree on every subject.

## 5. Declared non-participants

- The lowered Metal path emits no transitions.
- The hand-composed Kimi oracle stack is not a production path.
- V2 engines are not VINDEX3.
- Device providers share `device-matmul/v1` across format tables. Sharding is
  CPU-only, so this cannot reach a boundary; BUS-3 must close it before sharding
  a device.

## 6. Out of scope

- Transport and remote ROUTE lowering (BUS-3).
- A portable form for bundles, histories or continuation state.
- A persisted carrier format.
- A portable, content-addressed identity for operand overlays (D8).
- Whether an exact route must verify payload bytes, not just declared hashes,
  before binding (BUS-3's decision, I1).
- LCP-1 (off the path after BUS-0's D4).

## 7. Acceptance

BUS-2 is complete when:

- I1–I4, A1–A3 and S1–S3 pass on both backends over every subject in §2;
- every negative control fails at its event;
- F1–F3 are recorded, as holds or as losses.

A failed F2 on KDA/MLA is a finding: recurrent state that does not survive a
chunk boundary. It is not a reason to relax A2.
