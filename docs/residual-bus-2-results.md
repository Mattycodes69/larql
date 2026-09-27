# RESIDUAL-BUS-2 results: the address, identity and sequence of a carrier

**Class: RECORD.** Dated 2026-09-27, against the [frozen protocol](residual-bus-2.md)
(#621). The work landed in this order:

- the [reconnaissance](residual-bus-2-reconnaissance.md) (#615);
- two pre-freeze refusals (#616, #617);
- the identity rung (#625);
- the address rung (#627);
- the sequence rung, in the PR that carries this record.

**BUS-2 is CLOSED.** Every frozen property passes on both backends over every
subject in §2 of the freeze, every negative control fails at its rule, and
all three forecasts hold. There is one deviation, recorded in §4 with its
reason. BUS-3 can now move an addressed carrier across a process knowing
which computation produced it and whether it is the next legal hand-off.

## 1. Properties

| | Result | Evidence |
|---|---|---|
| **I1** identity comes from the image and binds the effective model | PASS | `ExecutionIdentity` holds the model authority, overlay, slice, lowering, every pinned realization and the process arithmetic. Different models, lowerings and slices give different digests. With no model authority, or over an overlaid source, the identity is unanchored and refused by name (`exec/identity.rs`, `tests/execution_identity.rs`) |
| **I2** every value-changing setting has a fate | PASS | `SETTING_FATES` gives every `LARQL_*` setting the executor's production code reads a fate. A conformance test scans `exec/` and fails on any setting without one (§5) |
| **I3** identity crosses the boundary and is checked | PASS | `Binding.schema` moves from 1 to 2 and carries the digest. The lowering is read from the pins. Each coordinator derives the digest it expects for every remote slice under its own process and refuses a mismatch. A cross-process witness shows every recorded setting moves the digest |
| **I4** claims are scoped by identity | PASS | An exact route refuses a digest mismatch and an unanchored identity. It never falls back to a structural comparison (D5) |
| **A1** one address | PASS | `CarrierAddress {position, layer, site, form}` is carried by batch's `PlaneEvent::Transition` and decode's `StepObserver::transition`. Every transition names its form |
| **A2** batch positions are absolute | PASS | Batch names every transition and write at `base + row`. `prefill_prepared_observed` makes chunked prefill observable (§3, F2) |
| **A3** address ≠ routability | PASS, with a deviation (§4) | `portability::ensure_portable` is the one authority. Rows are portable; bundles and histories are refused by name |
| **S1** one two-dimensional sequence | PASS | `Sequenced {stream, address, ordinal, transition}`. The stream is (identity digest, run id), and ordinals count within a position |
| **S2** the receiver fails closed | PASS | `SequenceGuard` refuses immediately on each immediate rule, and at close on a missing or truncated position. Controls are listed below |
| **S3** no arithmetic moves | PASS | Address, identity and sequence are records beside the writes. BUS-1's T7 and the Granite P1 witness still hold |

**Negative controls.** Each fails at exactly its own rule:

- a changed setting between two processes, and a shard presenting an altered
  digest (I3);
- a schema-1 peer, refused by name (D9);
- an overlay, and a source with no container, both unanchored (I1, D8);
- a batch without its base (A2), checked both by simulation and by forcing the
  base to zero in the executor;
- a transition relabelled with the wrong form (A1);
- a duplicate, a gap, a reorder, a foreign identity, a wrong form, a layer
  outside the bound range, and a position outside the domain (S2, immediate);
- a withheld whole position and a truncated position (S2, at close);
- an ordinal past the declared count (S2).

## 2. Forecasts

| | Result |
|---|---|
| **F1** each recorded setting changes the digest | **HOLDS.** It holds in-process for every field, and across processes for all eight recorded settings. The cross-process cases include `q8xq8` against `q8xq8b`, the same arm with a different scale span, which the old fingerprint could not see. The witness fails if the scale span is removed from the identity |
| **F2** chunked and unchunked prefill agree per absolute position | **HOLDS** on every subject, both backends: the plain stack, the Gemma 4 layer scale, bundles, histories, **mixed KDA/MLA** (the falsifier: recurrent and latent state cross every chunk boundary), and the **real Granite 4.2 3B container**. The agreement covers addressed transitions and bit-identical writes. Forcing the base to zero makes every miniature subject fail |
| **F3** batch and decode ordinals agree | **HOLDS** on every subject, both backends. Batch interleaves positions layer-major and decode position-major, and they stamp identical ordinals at identical addresses |

## 3. What the rungs changed

| Rung | PR | Change |
|---|---|---|
| Pre-freeze | #616 | A rows `ResumePoint` on an attention-residual component is refused. Before, it ran as a single stream and produced the wrong model; that was shown by execution |
| Pre-freeze | #617 | A one-shot traversal consumes its continuation provider. "`kv` and `resume` do not combine" was a false premise: they must combine. The hazard was continuing from a one-shot provider |
| Identity | #625 | `ExecutionIdentity`, `SETTING_FATES`, and schema-2 bindings with the digest. The hardcoded lowering at `distributed.rs:63` is gone |
| Address | #627 | `CarrierAddress`, absolute batch positions, observable chunked prefill, and `portability` |
| Sequence | this PR | `Sequencer`, `SequenceGuard` and `declared_transitions`. Every subject now runs on both backends |

## 4. Deviations from the freeze

- **A3: the `ResumePoint` refusals do not consult the portability function.**
  - The freeze says they do. They do not, deliberately.
  - `traverse`'s in-process resume checks decide topology correctness. An
    in-process bundle `ResumePoint` is supported and tested, and never leaves
    the process.
  - Portability is a process-boundary rule, not an in-process resume rule. It
    is consulted where a carrier actually crosses: the layer RPC's support
    check, and the CLI's plane-file dump and resume.
  - Forcing `traverse` through it would have refused a supported path.

## 5. Found on the way

- **Six settings no one had classified.** I2's first run found them on the
  lowered Metal path: five `LARQL_ABLATE_*` switches that change the model's
  output by design, and `LARQL_LOWERED_GATHER`. They are excluded only because
  that path cannot be sharded. **A device shard must bring them into the
  identity first.**
- **`LARQL_CPU_WORKERS` is in the identity.** No test establishes that results
  are independent of pool size. Workers with different thread counts therefore
  refuse to bind until a test proves equality and moves the setting to the
  exclusions.
- **A fixed witness value coincided with a runner's default.** The cross-process
  witness first set `LARQL_CPU_WORKERS=3`, and macOS CI resolves 3 by default.
  The identity was right and the witness was machine-bound. It now perturbs
  relative to the resolved value.
- **An empty overlay collapses to the base store** (`OperandSource::overlaid`),
  so it is the base model and stays anchored. Only a real edit is unanchored.
- **`DistributedSession::new` takes the operand source,** and
  `PreparedVindex3::store()` exists, so the layer coordinator can select each
  remote slice's pins.

## 6. Declared non-participants

- **Intervened streams.** Interventions add `Intervene` transitions the
  declared count excludes. They stay decode-only (BUS-1 T9) and never cross a
  boundary, so the sequence guard does not cover them.
- **The lowered Metal path** emits no transitions, and sharding refuses it.
- **Device providers** share `device-matmul/v1` across format tables. That must
  be closed before any device shard.

## 7. What BUS-3 inherits

- **Identity:** a digest derived from the prepared image, carried on the wire
  and checked on both sides.
- **Address:** absolute positions and forms on every transition, and one
  portability authority for what may cross.
- **Order:** a guard that refuses every out-of-order, missing, duplicated or
  foreign transition. BUS-3 is its first production consumer.
- **Open questions left to BUS-3:**
  - whether an exact route must verify payload bytes rather than declared
    hashes;
  - a portable form for bundles, histories and continuation state (each its
    own rung);
  - bringing the Metal ablation settings into the identity before any device
    shard.
