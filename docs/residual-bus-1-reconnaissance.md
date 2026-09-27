# RESIDUAL-BUS-1 reconnaissance: who may change the carrier, and what each change means

**Class: RECONNAISSANCE.** Dated 2026-09-26 at `b109e29f` (main `1248287f` +
the BUS-0 freeze). Read-only. Nothing is frozen here, and §6 proposes a question.

The question handed to this rung was:

> Can every canonical VINDEX3 carrier state transition be expressed through one
> semantic write authority, without changing arithmetic or execution order?

It named three write paths: decode `leave_site`, batch `leave_batch_site`, and
the Kimi KDA/MLA inline adds. **The third premise is false** (§2). The rung
therefore changes shape. The bus's authority on production paths is already
single. What differs between the paths is the **record** of a transition, not
who performs it.

`E` below is `crates/larql-vindex/src/format/vindex3/opplan/exec/`.

## 1. Reachability: every production decode runs the interpreter

| Entry | Path | Carrier writes |
|---|---|---|
| `larql serve` (V3), `--v3-backend cpu` (default) or `metal` | `larql_inference::vindex3` session → `DecodeSession` | `leave_site` |
| `larql run` on a V3 container, `--metal` or not | `run_cmd_vindex3` → `DecodeSession::over_prepared` | `leave_site` |
| `larql vindex3 exec [--generate]` (default `Reference`), `observe` (default `Production`) | `with_plan_backend` → `DecodeSession` | `leave_site` |
| Batch/prefill on all of the above | `execute_layer` (`crates/larql-vindex/src/format/vindex3/opplan/exec/mod.rs:1176`) | `leave_batch_site` |
| `vindex3 exec`/`measure` with `--backend metal-lowered*` | `lowered::run_lowered` | fused into GPU kernels, not the interpreter |

Interpreter-on-Metal (`DevicePlanBackend`) still performs the residual add on
the CPU. `device.rs:762` delegates `residual_add` to its `ProductionBackend`
glue.

## 2. Kimi is on the bus (correcting the premise)

The canonical interpreter executes KDA and MLA itself, with no family dispatch:

- `build.rs:1325` and `:1398` build `LayerAttention::{Kda, Mla}`;
- `prepared.rs:782-783` prepares them;
- decode calls `kda::layer_forward_with` and `mla::mla_forward_with` at
  `crates/larql-vindex/src/format/vindex3/opplan/exec/decode.rs:820-857`, then `leave_site`;
- batch does the same at `crates/larql-vindex/src/format/vindex3/opplan/exec/mod.rs:1372-1429`, then `leave_batch_site`;
- `opplan/tests/kda_mla_exec.rs:310` executes the mixed stack.

The **inline** adds (`kimi_kda_layer.rs:98-104, 131-138, 205-211, 229-236`;
`kimi_mla_layer.rs:95-101, 128-135`) belong to a **hand-composed oracle
stack**: `stack.rs` / `token.rs` / `stack_metal.rs`. Its only non-test caller is
REPRESENT's teacher-forced Metal measurement (`represent/measure/teacher_forced.rs:365,531`),
reached only through `TeacherForcedExecutor`, which no CLI or server command
registers. It is a transcription witnessed against the Python oracle
(`tests/stack_parity.rs:377`, `generate_metal.rs:353`). It is not an execution
path the bus must own.

The earlier documents took "inline adds exist" to mean "production bypasses the
bus" without checking reachability. That is corrected in dated sections of
[`residual-bus-lcp-1.md`](residual-bus-lcp-1.md) and
[`residual-bus-0.md`](residual-bus-0.md).

**A side finding (not BUS-1).** The two hand paths sum Kimi's shared expert in
different orders:

- **CPU:** routed experts first, then the shared expert
  (`kimi_moe_block.rs:214-231`).
- **Metal `kimi_moe_combine`:** the shared expert is folded into the same loop
  as slot `top_k` with weight 1.0 (`shaders/kimi_layer.rs:164-199`).

Both paths are gated against the Python oracle; neither is gated against the
other. This belongs to whoever owns the Kimi Metal ladder.

## 3. The transitions that exist

A carrier transition is anything that changes the carrier the next operation
reads. The census below covers the interpreter only.

| Transition | Carrier forms | Decode | Batch | Semantics |
|---|---|---|---|---|
| **Enter** | all | `embed` (scale inside), weightless norm, then `Bundle::replicate` / `History::new` (`decode.rs:648-677`); `Entry::Single/Hidden` takes an external carrier as-is (the ROUTE ingress for `LayerRange`) | `embed` + norm (`mod.rs:977-1013`); `ResumePoint` replaces `h` (`mod.rs:965`) and refuses Histories (`:947-956`) | a source, not an update |
| **Add** | Single / Rows | `backend.residual_add(h, &delta)` (`decode.rs:1515`) | `par_iter` of the same `residual_add` (`mod.rs:2013-2015`) | `after = before + delta`; `delta` is already post-norm and `residual_scale`-shaped (`scale_residual_delta`, `mod.rs:1156`, inline) |
| **HC update** | Bundle | `hyper_connection::update(x, &delta, &split, mutation)` (`decode.rs:1551`) | per position (`mod.rs:2027-2035`) | `comb` carries streams forward, `post` scatters the delta: **not an add** |
| **HC bypass** (control) | Bundle | `residual_add` into stream 0 (`decode.rs:1574`) | same (`mod.rs:2050-2053`) | a negative control expressed as an add |
| **History write** | History | `history.write(&delta)` (`decode.rs:1475`) | serial per position (`mod.rs:1995-1997`) | an **add to the prefix, or a replace** after a boundary reset (`attention_residual.rs:116-124`) |
| **History boundary** | History | `push_snapshot` / `reset_prefix` (`decode.rs:1440, 1451`) | `mod.rs:1933, 1949` | a state transition **outside** `leave_site` |
| **Intervene** | Single only | after the add, inside `leave_site` (`decode.rs:1514-1532`); admission refuses Bundle/History (`intervene.rs:350-365`) | **absent** (no parameter; `mod.rs` never names `Intervention`) | Zero / Add(v) / Replace(v) |
| **Scale** (Gemma 4 `LayerScalar`) | Single / Rows | `scale_row` **after** `leave_site` returns (`decode.rs:1109-1111`) | after the FFN write (`mod.rs:1664-1667`) | the whole carrier × s; refused on Bundle/History |
| **Positional controls** | Rows / Bundles / Histories | none | inside the write: `SwapPositionsBeforeUpdate`, `AttnResSwapPositionHistories`, `AttnResWriteOffsetByOne` (`mod.rs:1973-2024`) | negative controls, batch-only |
| **Exit** | all | reductions to a new vector (`decode.rs:1137-1205`) | `mod.rs:1052-1143` | a read, not a mutation |

Nothing in this census is an FFN-local combine. Routed and shared expert sums
(`experts.rs:889-895`) happen *inside* the branch and reach the carrier only as
`delta`. V3-FFN-SLICE-1's remote seam sits there, below the bus.

## 4. What is uniform, and what is not

**Already uniform (a property, witnessed):**

- one arithmetic path per form: `PlanBackend::residual_add` for Single/Rows,
  `hyper_connection::update` for bundles, `History::write` for histories;
- the same delta shaping in both paths;
- the same KV and recurrent-state ordering (every state side effect precedes
  the write);
- bit parity between batch and decode, witnessed on every form, at two
  different resolutions:
  - **per carrier state:**
    - `tests/carrier_write.rs:637, 653`: plain stack, and Gemma 4 through the
      layer scale;
    - `tests/wave19_hc_batch.rs:167` (A7): bundles;
    - `tests/attn_res_2b_batch.rs:681`: History.
  - **logits only:**
    - `tests/decode.rs:77, 82`: plain plan;
    - `opplan/tests/kda_mla_exec.rs:310`: mixed KDA/MLA, where a
      state-region reset would still produce finite logits, and the test says
      so.

  No per-carrier-state witness exists for KDA/MLA.

**Not uniform:**

- **R1. The record.**
  - Decode emits a per-write `CarrierWriteRecord{layer, site, position, delta,
    after, layer_scale}` on Single, plus `StepEvent::CarrierWrite` on every
    form.
  - Batch emits one `PlaneEvent::Layer{LayerTrace{post_attention, ffn_input,
    post_layer}}` per layer. There is no delta, and `post_layer` is
    **post-scale** with no pre-scale value retained.
  - The same transition is observable at different resolutions depending on
    whether the position was prefilled or decoded.
- **R2. `before` is not owned.** Decode clones `before` only when an
  intervention is armed (`decode.rs:1514`). Consumers reconstruct it by chaining
  (V3-OBS-1 C6). A `CarrierTransition{before, …}` holding `before` by value
  would add a clone per write on the hot path. `before` must stay borrowed or
  chained (C1).
- **R3. A transition's fate is split across sites.**
  - The layer scale is applied after `leave_site` returns. V3-OBS-1 C5 repairs
    that in the record, not in the authority.
  - History boundaries are transitions outside `leave_site` altogether.
  - A single write authority has to absorb both, or declare them as separate
    transitions.
- **R4. Interventions are decode-only and Single-only.** This is safe today.
  The one intervening runner (`vindex3 observe`) steps prompt tokens through
  decode (`observe.rs:295-310`), and V3-INTERVENE-1's I8 turns a declared,
  unfired intervention into a refusal on the receipt. But the batch path cannot
  express one, and nothing refuses one there, because nothing can reach it.
- **R5. Controls live inside the batch write.** A unified write authority must
  carry `Mutation` controls, or they become a second authority.

## 5. Declared non-participants

A bus claim is honest only if what is outside it says so.

| Path | Status | Refusal today |
|---|---|---|
| Lowered Metal (`lowering::stack`) | adds fused into kernels (`attention.rs:367-426`, `ffn.rs:118-202`, `moe_gpu_route/encode.rs:560`, `stack.rs:518-537`) | `observe`/`--intervene` refused before execution (`prepare.rs:246-254`, `tests/decode.rs:214-217`). **There is no automated lowered-vs-interpreter parity test**; the lowering tests compare against hand-transcribed references |
| Hand-composed Kimi stack / `HybridStack` | inline adds, `*Trace` structs | none needed; not registered as an execution path |
| Legacy (`larql-compute` / `larql-compute-metal` decode) | many add sites, including fused kernels | migration target by programme rule |

The missing lowered-vs-interpreter parity test is a real gap. It is a
correctness gap for the lowering programme, not a bus gap. It is recorded here
because a future BUS-3 ROUTE lowering onto a lowered worker would inherit it.

## 6. What BUS-1 should ask instead (NOT frozen)

The authority is already single. The honest rung is about the **record**:

> Can the batch traversal emit, for every carrier transition it performs, the
> same transition record the decode traversal emits, with records **bit-identical**
> per (position, layer, site) across the two traversals, batch observed ==
> unobserved bit-identical, and the observation cost measured?

What that entails, all inside the interpreter:

1. **A transition vocabulary** derived from §3, not from `residual_add`:
   `Enter`, `Add`, `HcUpdate`, `HistoryWrite{add | replace}`,
   `HistoryBoundary{snapshot | reset}`, `Intervene{kind}`, `Scale{s}`. The
   controls are tagged, not hidden. Architecture differences stay explicit.
   This is the evidence-backed form of the "`CarrierTransition{address, topology,
   before, operation, after}`" proposal. `before` stays chained (R2), and
   `topology` is the carrier form, not a collapsed vector (V3-OBS-1 C3).
2. **Batch value records on Rows**: `delta` and pre-scale `after` per row,
   borrowed from the plane at the moment of the write, plus the scale (C5).
3. **The witness**: decode records and batch records equal to the bit on the
   plain, Gemma 4 and mixed KDA/MLA stacks. The KDA/MLA case is new coverage;
   today it is witnessed only on logits. Bundles and histories get equal
   structural events and equal site records.
4. **Refusals declared** for the non-participants in §5, with the text naming
   BUS-1.

**Out of scope for BUS-1:**

- interventions in batch (R4 is safe, and it is its own rung if a runner ever
  needs prefill interventions);
- `CarrierAddress` / sequence identity (BUS-2);
- any ROUTE or transport;
- the lowered parity gap and the Kimi shared-expert order.

**Baseline mechanics are measured before the freeze, not forecast:** the
current cost of `LayerTrace` capture on batch, and of `carrier_write` on
decode, on one real container.
