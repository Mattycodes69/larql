# AUTO-REP-PRIOR-1: does an ecosystem gradient score earn a place as AUTO-REP's prior?

**Frozen 2026-09-27, before any gradient is computed on Granite 4.1 3B.** The question, the score, its
aggregation, the calibration data, the controls, the bars, the arms and the forecasts below are FROZEN.
There is one scoring pass and no revision afterwards.

Programme: REPRESENT. This follows the SENSITIVITY ladder (`bench/prompts/quality-bank-1/SENSITIVITY-1*.md`)
and AUTO-REP-1 (`docs/auto-rep-1.md`). It depends on MEASURE-PLAN-3 / AUTO-REP-1c (`docs/measure-plan-3.md`)
for its gate and its no-prior arm.

## The question

> MLX-LM's `dynamic_quant` and NVIDIA ModelOpt's AutoQuantize choose mixed precision automatically from a
> per-layer sensitivity score. SENSITIVITY-1 falsified every cheap score LARQL tried. Does the
> ecosystem's score (a) rank Granite's protection candidates the way whole-model measurement does, and
> (b) when used as AUTO-REP's ordering, reach the MEASURE-PLAN-3 gate in fewer measurements than no prior?

This rung tests **MLX's** score only. Its source was read at `mlx-lm 0.29.1`
(`mlx_lm/quant/dynamic_quant.py`, `estimate_sensitivities`). ModelOpt's score is out of scope (see "Not
claimed").

## Why this is not a repeat of SENSITIVITY-1

| rung | signal | what it could not see |
|---|---|---|
| 1A | `‖ΔW‖²/‖W‖²` | anything about use |
| 1B-a, 1B′ | local activation-weighted error | downstream sensitivity: `down_proj` has large error but low consequence |
| 1C | suffix replay | died on feasibility (no batch axis, O(directions)) |
| 1D | Gauss–Newton field | theory only: VINDEX3 has no reverse engine |

MLX's score is downstream-aware and first order, and it evades 1D's stationarity argument. 1D showed that a
first-order score against KL is identically zero **at the reference**. MLX evaluates the gradient at the
**all-low-bit point**, where `q ≠ p`, so the gradient is non-zero:

```text
g_W        = ∂ mean KL(p_ref ‖ q_lowbit) / ∂W          evaluated at W = W_low, every layer low
align(W)   = Σ g_W ⊙ (W_low − W_high)                  ≈ KL(all low) − KL(all low, W restored)
mlx(W)     = align(W) / (W.size / 1e6)                 per million parameters
```

To first order, `align(W)` is the mean-KL benefit of restoring one tensor while every other tensor stays
low. That is exactly the candidate question AUTO-REP asks from the uniform-NVFP4 base. The reverse pass runs
in MLX, outside LARQL. That is legitimate because a proposal is never authority (`docs/auto-rep-1.md`).

## A pre-registration input: the bar is metric-specific

The following is computed from banked truth only (`granite-4.1-3b-sweep.json`, the 1B′ pool), with no score.

MLX's loss is **mean KL**, but the SENSITIVITY-1 bar was written against Q-BANK's **p99** verdicts. Rank the
1B′ pool by each measured quantity's return per extra MiB, and ask whether a *perfect* predictor of that
quantity passes the 1B′ bar:

| candidate | +MiB | p99 return / MiB | mean-KL return / MiB |
|---|---|---|---|
| `late5-ffn` | 431.2 | +0.007745 | +0.0003467 |
| `late10-ffn` | 862.5 | +0.003895 | +0.0001837 |
| `late15-ffn` | 1293.7 | +0.002607 | +0.0001277 |
| `late10-ffn-v` | 934.4 | +0.003720 | +0.0001805 |
| `ffn-protected` | 3450.0 | +0.001147 | +0.0000621 |
| `v-protected` | 71.9 | +0.000272 | +0.0001888 |
| `k-protected` | 71.9 | −0.000343 | +0.0000914 |
| `down-protected` | 1150.0 | −0.000155 | +0.0000170 |

| oracle ranks by | condition 1 | condition 2 (v/k/down in bottom half) | knee |
|---|---|---|---|
| p99 | PASS | PASS (6, 8, 7) | PASS |
| mean KL | PASS | **FAIL: `v-protected` rank 2** | PASS |
| ΔNLL | PASS | FAIL: `k-protected` rank 3 | PASS |
| p95 | PASS | FAIL: `k-protected` rank 4 | PASS |
| flips | FAIL | FAIL | PASS |

**Only a p99 oracle passes.** A perfect mean-KL estimator fails condition 2. So reusing the bar unchanged
would score an objective mismatch as an estimation failure. This rung therefore asks two separate questions.

`v-protected`'s rank 2 under mean KL is a near-tie with `late10-ffn` (1.888e-4 against 1.837e-4). No
condition below depends on it.

## The score, frozen

- **Weights.** `ibm-granite/granite-4.1-3b` at `c0650403e44e78ec0262dab1c90914c65b196c4e` (the registry pin
  and the container's source), loaded by `mlx_lm` 0.29.1 on `mlx` 0.30.1.
- **Low and high.** `W_high` is the source BF16 tensor. `W_low` is **LARQL's** `nvfp4/rev1`
  `nvfp4-nearest-v1` reconstruction of that tensor, exported in f32 from the uniform-NVFP4 pack. It is not
  MLX's affine or NVFP4 `qdq`: the domain being searched is LARQL's codec against source.
- **Harness.** A copy of `estimate_sensitivities`. The only change allowed is that the `qdq(low)` and
  `qdq(high)` calls are replaced by the injected `W_low` and `W_high`. The diff against the upstream function
  is committed alongside the scores. Loss, gradient, accumulation and normalisation are untouched.
- **Calibration.** MLX's own default: `load_data(..., num_samples=-1, sequence_length=512)` over
  `calibration_v5.txt` (the gist URL pinned in `mlx_lm/quant/utils.py`), with seed 123, batch 4 and f32
  gradient accumulation. The file's sha256 is recorded. This is MLX's procedure as shipped; Q-BANK is never
  the calibration set.
- **Population.** The 280 decoder projection tensors of the 1A record. Embedding, norms and any other MLX
  leaf are scored by MLX but ignored, and are listed in the output. `o_proj` has gradients here (1B′ could
  not score it).
- **Sign.** Signed, as MLX computes it. A positive value means restoring the tensor is predicted to lower
  KL. There is no absolute value and no clipping.
- **Region score.** `score(R) = Σ_{W∈R} align(W) / extra MiB(R)`, where extra MiB comes from the footprint
  (`source_bytes − compiled_bytes`). That is the 1B′ return metric, and at tensor grain it orders tensors
  the same way `mlx(W)` does.
- **Output.** One JSON record per tensor, carrying `align`, `mlx`, bytes, the source weight digest, the
  uniform pack's payload digest, the calibration sha256 and the harness diff digest.

## Controls. A failure is a harness failure, not a result

1. **Reconstruction.** For every one of the 280 exported tensors, `‖W_low − W_high‖²/‖W_high‖²` matches
   the banked 1A `rel_error` to 1e-4 relative.
2. **Same function.** MLX's BF16 forward and LARQL's `production` reference agree on the 69 Q-BANK-1
   sequences: top-1 agreement ≥ 0.99 and mean KL ≤ 1e-2 nats. Granite's four multipliers are where two
   engines are most likely to disagree.
3. **Disjoint calibration.** There are zero shared 13-grams (in the container tokenizer's tokens) between
   `calibration_v5.txt` and Q-BANK-1 `prompts.json`.
4. **Name mapping.** MLX's `model.layers.N.<proj>` maps one-to-one onto the 280 LARQL tensor names, and the
   bytes agree with the 1A record.
5. **Region congruence.** The existing `region_congruence.py` check passes for every region scored.

## Part 1: ranking, from existing truth only (no new measurements)

**1a. Estimation: does the score rank candidates like the quantity it estimates?** Truth is the mean-KL
return per MiB (table above), over the 1B′ pool of eight.

- **E1.** `late5-ffn` ranks 1st.
- **E2.** `down-protected` ranks 7th or 8th. This is 1B′'s failure mode: large error that is not
  important error.
- **E3.** The knee: `score(late5-ffn) > score(late10-ffn) > score(late15-ffn)`.

All three must pass. Spearman is reported but cannot rescue a failed condition.

**1b. Decision: does it rank like the p99 verdicts?** The 1B′ Granite conditions 1–3 apply verbatim, over
the same pool.

**Secondary, reported but unable to change a verdict:** the same ranking over all 13 sweep regions,
including the five that contain `o_proj`.

**Evidence class.** 1a decides the prior's `SearchEvidence`. A pass declares it
`OrderingProxy { calibration: <calibration sha256> }`. A fail declares it `Unusable`, and Part 2's arm C is
then refused by construction. 1b does not gate arm C: 1b asks whether mean KL is the right target, and
arm C measures that directly against the joint gate.

## Part 2: arms against the MEASURE-PLAN-3 gate

Part 2 needs `granite-4.1-3b-plan-v1-vs-late10-ffn/v1` registered from the anchor, and it runs **after**
the AUTO-REP-1c campaign concludes, so no reading here can touch that frozen campaign. It uses the same
bank, reference and candidate arms, tail policy and 40-measurement budget as 1c.

- **Arm A: MLX's own rule, one measurement.** Order the 280 tensors by `mlx(W)`, descending. Protect the
  longest prefix whose extra bytes do not exceed the anchor's extra bytes (`late10-ffn`, 862.5 MiB). This
  is `estimate_threshold`'s prefix semantics, written as bytes rather than bits per weight. There is no
  knapsack fill. The prefix is compiled at tensor grain through `Protections` (1a's grain), then measured
  once. It runs whatever Part 1's verdict, because it answers "what would the ecosystem procedure have
  handed you at the hand map's bytes?"
- **Arm B: no prior.** This is AUTO-REP-1c as frozen, and it is **not re-run**. Its `CampaignRecord` is the
  comparator.
- **Arm C: prior-prefix proposer.** This runs only if the evidence class is `OrderingProxy`. Rank the 12
  groups of the 1c vocabulary by `score` and propose the nested prefixes (top 1, top 2, … top 12) in
  increasing bytes. Then fall back to 1a's cheapest-first order over the states not yet proposed. It uses
  ranks only and never adds or compares magnitudes, so it stays inside what `OrderingProxy` permits. It
  starts from a fresh snapshot, with no cuts carried over from B. A state B already measured may reuse B's
  reading by state id, but it still counts against C's budget. Two reused states are re-measured, and their
  report digests must be equal. The stopping rule is 1c's: stop at the first admission.

**Outcomes.** B is exact cheapest-first, so B's first admission `X` is the cheapest admissible state, and C
cannot beat it on bytes.

- If B admitted `X` after `n_B` measurements, C wins when it admits a state of bytes ≤ `bytes(X)` after
  fewer than `n_B` measurements. If C first admits a dearer state, it loses on bytes, and the gap is
  reported.
- If B spent its budget without an admission cheaper than the anchor, C wins when it admits any state
  strictly cheaper than the anchor within 40 measurements.

## Forecasts, made before any gradient

1. **The controls pass.** The riskiest is control 2 (the four Granite multipliers in MLX's `granite.py`).
2. **1a: E1 and E3 pass.** On E2 the forecast is also a pass: the gradient carries what happens downstream
   of `down_proj`, which is exactly what 1B′ lacked. This is the forecast the rung exists to test.
3. **1b is conditional.** If 1a passes, 1b is forecast to **fail** condition 2, with a small attention
   projection (`v` or `k`) ranked in the top half, as the mean-KL oracle does. A 1b pass alongside a 1a pass
   would mean the score tracks something other than its own target.
4. **Arm A, low confidence: fails the gate on KL p99.** A mean-KL prefix spends part of the anchor's bytes
   on small attention tensors, which buy mean KL but no tail.
5. **Arm C: unknown, and that is the point.**

## What must be built first (engineering, no result)

1. A `vindex3` export of one pack's per-tensor f32 reconstruction, with provenance (the pack payload digest
   and the source weight digest), used by control 1.
2. The MLX harness under `bench/prompts/quality-bank-1/`: the copy of `estimate_sensitivities`, its
   committed upstream diff, pinned versions, the calibration sha256 and controls 2–4.
3. Arm C's proposer, as a second `Proposer` next to the branch-and-bound (`docs/auto-rep-1.md` reserves
   that trait for this comparison). It must keep 1a's gates: every proposal passes `check_against`,
   compiles through `Protections` and is a distinct state.

The gradient pass runs on the GPU, so it needs a quiet-window handshake with peer sessions first.

## Not claimed

- **Anything about ModelOpt AutoQuantize.** Its score is different, it runs on CUDA, and its exact form must
  be read from source at a pinned version before a rung of its own is pre-registered.
- **MLX's native configuration** (affine 4/5-bit, group size 64). LARQL cannot compile it, and it is not
  the domain searched here.
- **Other models.** A Glimmer rung, like 1B′'s conditions 4 and 5, is gated on a Granite pass.
- **That `align(W)` measures restoration.** The `≈` above is a first-order reading, not an equality. Three
  layers stay separate in the result: the gradient alignment is the predictor, the banked mean-KL
  restoration is 1a's truth, and Q-BANK p99 is the deployment evidence. A good ranking does not turn the
  gradient into a causal measurement.
- **That a failed arm C falsifies the score.** Nested prefixes are one construction. The best mixed
  state need not be a prefix of any ranking, so 1a passing while C loses points at the construction.
- **Calibration of magnitude.** An `OrderingProxy` pass permits ordering and nothing more.
- **Promotion, or any execution-speed claim.**

## Amendment 1 (2026-09-28): control 2's comparator and MLX's compute dtype

**Written after control 2 failed and before any gradient was computed.** No score exists. Nothing above is
edited; this section overrides it where the two conflict.

### What happened

Control 2 as frozen (MLX BF16 against LARQL `production`) failed: top-1 agreement 0.961 against the
≥ 0.99 bar, with mean KL 3.2e-3 inside its 1e-2 bar. Stopping there was the plan's rule. The diagnosis
used the kept LARQL dumps and one extra LARQL run (`--backend reference`, the naive f32 path that shares no
arithmetic with `larql-compute`), over the same 69 prompts and 1,667 positions:

| comparison | top-1 | mean KL (nats) | max \|Δ log p\| |
|---|---|---|---|
| LARQL `reference` vs MLX f32 | **1.0000** | **8.4e-11** | 5.7e-4 |
| LARQL `reference` vs MLX BF16 | 0.969 | 1.1e-3 | |
| LARQL `reference` vs LARQL `production` | 0.976 | 2.1e-3 | 1.41 |
| LARQL `production` vs MLX BF16 (as frozen) | 0.961 | 3.2e-3 | |

MLX in f32 computes LARQL's Granite function exactly, which is the property control 2 exists to protect.
The frozen comparator was the wrong one: `production` differs from LARQL's own reference by more than the
bar allows. That gap is a separate LARQL finding and is not investigated here.

### What changes

1. **Control 2's comparator is LARQL `reference`**, and MLX runs in **f32**. The bar is unchanged:
   top-1 ≥ 0.99 and mean KL ≤ 1e-2 nats.
2. **The gradient pass runs MLX in f32** (`model.set_dtype(float32)` after load), so the prior is computed
   on the function control 2 verified. MLX in BF16 would still fail control 2 against `reference`
   (0.969). This is a further departure from MLX's procedure as shipped, which computes in the checkpoint's
   BF16. The score formula, its sign and aggregation, the calibration data, seed, sequence length, batch
   size, f32 gradient accumulation and every bar are unchanged.
3. **Control 2 may reuse the diagnostic LARQL `reference` dumps**, provided the bank they were run on is
   byte-identical to the one the control regenerates.

### Implementation clarifications (no change to the protocol)

- **Where `W_low` comes from.** It is `round_trip(source)`, i.e. `dequantize(quantize(source))`: the same
  `quantize` a pack stores through `encode` and the same `dequantize_into` a reader decodes with. The
  export's manifest records the source payload digest, the codec (`nvfp4/rev1`), the encoder
  (`nvfp4-nearest-v1`) and the quantiser's name, not a pack digest. Control 1 checks the export against the
  banked 1A; it cannot check pack equivalence, because 1A came from the same `round_trip`. A provenance
  witness is queued for when 1c compiles the uniform pack: the 7 projections at an early, a middle and a
  late layer (21 tensors), each required to be bit-equal to its export. Until it runs, the equivalence
  holds by construction and is not measured.
- **The low point.** Only the 280 exported decoder projections are lowered. Every other MLX leaf,
  including the tied embedding, stays at source, so the gradient is taken at AUTO-REP's base state. The
  "all-low-bit point" above means that state.
