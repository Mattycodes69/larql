# AUTO-REP-LANDSCAPE-1: the whole precision cube for Granite 4.1 3B, measured

**Frozen 2026-09-28, before any MEASURE-PLAN-3 reading and before any cube reading.** The atoms, arms,
bank, analysis, replay arms, budgets, forecasts, thresholds and the values below are FROZEN.

| frozen value | |
|---|---|
| instrument | commit `31f8e3b5` (PR #635): `teacher-forced-two-arm/plan-v1` + sketch `rademacher-splitmix64/v1` + receipt digests |
| `sketch_seed` | `381943897866997936` |
| sketch K | 256 |
| `cube_order_seed` | `800104581919694298` |
| schedule | `bench/auto-rep-landscape-1/schedule.json`, sha256 `ddfc7dd5306d4b9a2b71d65e0c1d07fc970d1b8f1bdd325a0e8a1ed5697fc437` |

Both seeds were drawn from the OS random source (`secrets.randbits(63)`) when this document was frozen.
The cube runs on a build whose `measure/plan/{mod,record,sketch}.rs` are byte-identical to `31f8e3b5`'s,
checked by `git diff 31f8e3b5 <build> -- crates/larql-vindex/src/format/vindex3/represent/measure/plan`
reading empty. PR #635 had not merged when this was frozen, so the freeze cites the instrument commit
rather than a merge commit.

Programme: REPRESENT. A sibling of AUTO-REP-1c (`docs/measure-plan-3.md`), following AUTO-REP-PRIOR-1
(`docs/auto-rep-prior-1.md`, Part 1 FAIL, #634).

## Amendment 1 (2026-09-28): four measurements at a time

**Committed before any MEASURE-PLAN-3 campaign reading and before any cube reading.** The trigger is the
pilot (procedure step 0), measured on MEASURE-PLAN-3's anchor with the frozen arms and instrument: one
measurement took 690 s of wall time with about 2 cores busy. 266 steps is about 51 h plus about 4 h of
compiles, over the 36 h limit.

- **What changes: wall-clock scheduling only.** The frozen schedule's steps are taken in order, in
  consecutive batches of at most four, and each batch runs as independent processes. Each step still runs
  the full reference and candidate arms, null arm included, and writes its own admissible receipt. No
  logits, arms or intermediate state are shared between steps. The bank, arms, instrument, seeds,
  schedule, analysis and forecasts are unchanged.
- **Why this is sound.** The shuffled order guards against drift lining up with the cube's structure. The
  arms are deterministic (every receipt's null arm checks this, and so do control 1's repeats), and a batch
  holds four consecutive shuffled steps, so the randomisation is kept.
- **Concurrency equivalence control, run before step 0.** Measure ∅ (map 0) once alone, then four copies
  of ∅ concurrently, all with the frozen sketch. All five receipts must carry the same `positions_sha256`
  and the same sketch `sha256`. These are control runs, not schedule steps, and their values are not
  analysed. If any digest differs, the cube runs serially as originally frozen, and that is recorded here.
- **Not adopted:** reusing the reference's logits across maps. It would change the procedure's proof
  boundary, since the reference is re-run and null-checked on every measurement by design.

## Why

PRIOR-1 Part 1 showed that MLX's gradient score is a poor decision rule for AUTO-REP. It did **not** show
why. Three explanations remain, and they call for different engineering:

1. **Wrong proxy, additive landscape.** Effects compose, and the gradient is a bad estimator of them. The
   score is mean-KL where the bar is p99, it is first order, and it is normalised per MiB. PRIOR-1's own
   pre-registration found that even a perfect mean-KL oracle fails the 1B′ bar. The fix is cheap measured
   marginals plus the existing branch-and-bound.
2. **Additive per-position effects, nonlinear aggregate.** Each position's error composes, but p99 is a
   nonlinear function of the per-position distribution, so the aggregate does not. The fix is a
   distributional objective, not a higher-order model of the map.
3. **Real interaction in the representation.** The logit perturbations themselves fail to compose. The fix is
   search over whole maps (pairwise terms, then an active measure over maps).

PRIOR-1 found that the gradient score's total was about right (Σalign 0.315 against a measured R0 mean KL
of 0.278) while individual regions were off 3–4×. That fits world 1 as well as world 3. The banked Q-BANK-1
sweep has two measured pairs, and both are close to additive (measured against uniform NVFP4):

| pair | metric | sum of singletons | measured | interaction |
|---|---|---|---|---|
| `late10-ffn` + `v-protected` | mean KL | 0.1720 | 0.1687 | −2% |
| `late10-ffn` + `v-protected` | p99 | 3.379 | 3.476 | +3% |
| `late10-ffn` + `o-protected` | mean KL | 0.1727 | 0.1756 | +2% |
| `late10-ffn` + `o-protected` | p99 | 3.773 | 3.575 | −5% |

Two pairs decide nothing. They set the prior: **mostly additive** is the expected world. A result in world
3 would be a surprise, and would need to clear the determinism floor (control 1) to count.

The sweep's regions cannot answer this question because they nest (late5 ⊂ late10 ⊂ late15 ⊂
ffn-protected; down ⊂ ffn). Toggling nested regions produces duplicate maps and invented interaction terms.
This rung uses **disjoint atoms**.

## The question

> On a Boolean cube of disjoint precision atoms, which of worlds 1, 2 and 3 holds? And how many
> measurements does each proposal strategy need to find the cube's cheapest gate-admitted map?

## Model, bank and arms

Identical to MEASURE-PLAN-3, so that its gate reads directly on the cube:

- **Model:** `~/chris-models/granite-4.1-3b.vindex3` (`ibm-granite/granite-4.1-3b` @ `c0650403`, the
  registry pin).
- **Bank:** Q-BANK-1, all 69 sequences, `--max-tokens 128`.
- **Arms:** reference `production` on the source's canonical bytes; candidate `production-nvfp4` on the
  stored pack. Interpreter arms, so a heterogeneous map runs without a lowered refusal.
- **Instrument:** `larql vindex3 measure` (MEASURE-PLAN-1). Per map it keeps `report.json`,
  `receipt.json` and `positions.jsonl` (per-position `kl`, `delta_nll`, `top1_agree`,
  `reference_margin`). Only admissible receipts count.
- **Logit sketch** (`measure --sketch-dim K --sketch-seed S`, generator `rademacher-splitmix64/v1`,
  `measure/plan/sketch.rs`). At each position `measure` also writes s_t = R · c(z_cand − z_ref), where c(·)
  subtracts the vector's mean over the vocabulary. Softmax is shift-invariant, so c(Δz) is also the centred
  log-probability difference. R is K × V with entries ±1/√K, drawn from SplitMix64 under the seed, and is
  specified bit for bit in the module header, so the analysis can regenerate it independently. **K = 256.**
  The output is `sketch.f32` (positions × K, little-endian, in `positions.jsonl` row order), about 1.7 MB per
  map. The seed and the instrument commit are fixed in the freeze commit.
  - *Why Rademacher, not Gaussian:* R must be bit-identical on every platform, for replay and for the
    analysis to regenerate it. Gaussian draws need `ln`/`cos`, which are not correctly rounded, and
    Rademacher is integer-only. The detection guarantee is still exact: for any nonzero v, each row gives
    zero with probability at most ½ (Littlewood–Offord), so P(Rv = 0) ≤ 2⁻²⁵⁶.
    The probability is over the seed's draw. Once the seed is frozen, R is deterministic. The seed is fixed
    before any cube reading, so the guarantee holds for the experiment as run.
  - *Evidence binding:* the receipt carries `positions_sha256` and a `sketch` record (generator, K, seed,
    rows, sha256). `measure::plan::record::verify_record` re-hashes both files against the receipt, and the
    analysis runs it on every map before reading any number. A map whose evidence fails it is excluded
    and reported by name.
  - **Two seeds, frozen separately:** `cube_order_seed` (procedure step 1) and `sketch_seed` (R). Neither is
    derived from the other.

## Atoms

Eight atoms: {`attn`, `ffn`} × the four layer quarters (Q1 = 0–9, Q2 = 10–19, Q3 = 20–29, Q4 = 30–39).

| family | projections | extra MiB per quarter (held vs NVFP4) |
|---|---|---|
| `attn` | `q_proj`, `k_proj`, `v_proj`, `o_proj` | 215.6 |
| `ffn` | `gate_proj`, `up_proj`, `down_proj` | 862.5 |

This merges MEASURE-PLAN-3's `attn-qkv` and `attn-o` into one family, so the cube is a coarsening of 1c's
vocabulary, not the same space. Each atom is held at source or compiled NVFP4, giving **2⁸ = 256 maps**.
∅ is uniform NVFP4. {`ffn`×Q4} is MEASURE-PLAN-3's anchor (`late10-ffn`).

**Structural fact:** bytes depend only on (attention atoms held, FFN atoms held), so the 256 maps fall into
**25 byte levels**. Within a level the frontier is decided by quality alone. Four attention atoms (862.5
MiB) tie exactly with one FFN atom. A cheapest-first search meets large equal-cost groups, and 1a's bound
covers the whole ranking tuple for exactly this reason.

## Procedure

0. **Pilot (engineering, no result).** Time one measurement of ∅. The total budget is 256 maps plus
   repeats (control 1). If the projected wall time exceeds 36 h, stop and decide before running.
1. Measure the 266 steps of `schedule.json` in order: the 256 maps shuffled under `cube_order_seed`, so
   drift cannot line up with the cube's structure, and the repeats below at their seeded positions. Map
   ids are bitmasks over the atoms (bit 0 `attn`×Q1 … bit 3 `attn`×Q4, bit 4 `ffn`×Q1 … bit 7 `ffn`×Q4; set
   means held at source). The file is the authority, not the generator that wrote it.
2. Repeats: ∅ (0), the full map (255), {`ffn`×Q4} (128) and two seeded pairs, {`ffn`×Q1, `ffn`×Q2} (48)
   and {`ffn`×Q3, `ffn`×Q4} (192). Each is measured three times in all.
3. Bank each map's `positions.jsonl` digest and aggregates in one results file. Delete compiled
   intermediates once a map's receipt is banked.

## Analysis (fixed before any reading)

For a map S and position t, let f_t(S) be a per-position quantity measured against ∅. Two expansions are
computed exactly from the full cube:

- **Möbius (from ∅):** f(S) = Σ_{T⊆S} m(T). The order-1 truncation is the "predict from 8 singletons"
  model. The order-2 truncation adds the 28 pairs.
- **Walsh–Hadamard:** the orthogonal decomposition, reporting the fraction of variance at each
  interaction order, per position and pooled.

These are applied at four levels. Each level adds exactly one source of nonlinearity to the one before it:

- **L0, logit perturbation.** f_t(S) = s_t(S) − s_t(∅), a K-vector. R is linear, so the sketch composes
  exactly when the centred logit perturbation does, and the order-1 test here is a direct test of whether
  the *model's* perturbations compose. The sketch cannot create a violation, because exact additivity
  gives a zero sketch residual, and it hides a nonzero one with probability at most 2⁻ᴷ. Magnitude is
  secondary: a sketched norm is a noisy estimate of the true one, with relative standard deviation roughly
  1/√(2K) (about 4% at K = 256) and no hard bound. L0 is used to ask whether the order-1 residual clears the
  noise floor below, not to measure its size precisely. **This is the only level whose failure is called model interaction (world 3).**
- **L1, ΔNLL per position.** ΔNLL = gᵀΔz + ½ΔzᵀHΔz + …, so even perfectly additive Δz leaves pair
  terms at finite perturbation size. An L1 error above L0's is **curvature of the metric**, not
  interaction.
- **L2, KL per position.** KL ≈ ½·Var_p(Δz) is second order from its first term, so additive
  perturbations leave pair terms Cov_p(Δz_i, Δz_j). Their sign says whether two atoms' errors cancel or
  compound. An L2 error above L0's is also curvature.
- **L3, aggregates over positions.** Computed three ways for every map: (a) measured; (b) from the order-1
  per-position KL predictions, then aggregated; (c) by summing the singletons' aggregate changes.
  - **KL p99** is the discriminating aggregate. If (b) matches (a) and (c) does not, the per-position
    effects compose and the tail statistic does not (world 2).
  - **KL mean** does not discriminate: the mean of a sum is the sum of the means, so (b) = (c)
    identically. It is reported, never used to classify.
  - **Top-1 disagreement** has no per-position additive prediction (argmax is not a function of the
    kept per-position values), so it is reported as (a) and (c) only.

**Error measure.** For one level and one map, e(S) = ‖f(S) − f̂(S)‖ / ‖f(S)‖, taken over positions (and
over the K sketch coordinates at L0). Report the median and the 90th percentile over the scored maps.

**Noise floor.** Control 1's repeats give, for each level, the norm of the difference between repeat
readings of the same map, σ. For each level:
- a map is **scored** only if ‖f(S)‖ ≥ 3σ, so the denominator is not dominated by noise. The count of
  unscored maps is reported per level;
- an interaction term |m(T)| < 3σ is reported as **unresolved**, never as zero and never as nonzero.

**Classification rule**, over the maps with |S| ≥ 2 that are scored at every level involved:
- **world 3** if L0 fails F1;
- **world 2** if L0 passes F1, and p99 passes F3's (b) clause and fails its (c) clause;
- **world 1** if L0 passes F1 and p99 passes both F3 clauses;
- otherwise **mixed**, with every level's numbers.

F2 (curvature) is reported alongside the world and does not change it.

## Replay (offline, no model runs)

With the cube banked, each proposal strategy runs against it through 1b's scripted executor. The cube is
the oracle. The gate is MEASURE-PLAN-3's `granite-4.1-3b-plan-v1-vs-late10-ffn/v1` once registered. Until
then, the gate is the {`ffn`×Q4} vertex's own values under the same rule ("no worse on every criterion").

| arm | strategy | measurements before it can predict |
|---|---|---|
| R1 | random order (1,000 seeds, report the distribution) | 0 |
| R2 | gradient prior: MLX align per atom per MiB, order frozen below | 0 |
| R3 | 1a branch-and-bound, cheapest first, no prior (1c's solver) | 0 |
| R4 | order-1 per-position model from ∅ + 8 singletons, then measure its predicted-admitted maps cheapest first | 9 |
| R5 | order-2 model: R4 + 28 pairs | 37 |
| R6 | exploratory: an adaptive distribution over maps, updated from each reading | declared later |

R6 is exploratory. It is reported but cannot change a verdict.

**R2's order, frozen now** from `granite-4.1-3b-prior1-scores.json` (Σ align / extra MiB):
`attn`×Q2 (1.84e-4), `attn`×Q1 (1.48e-4), `attn`×Q3 (1.42e-4), `attn`×Q4 (1.37e-4), `ffn`×Q4 (7.71e-5),
`ffn`×Q1 (6.19e-5), `ffn`×Q2 (4.23e-5), `ffn`×Q3 (3.11e-5). The atoms' align sums to 0.3149.

**Metrics** for each arm at budgets N ∈ {9, 16, 24, 40, 64}:
- whether it has measured the cube's cheapest admitted map;
- byte regret, meaning the bytes of the cheapest admitted map it found minus the true minimum (∞ if none);
- the fraction of the true (bytes, KL p99) Pareto set it has measured.

The headline number is **the measurements needed to reach zero byte regret**.

## Controls

1. **Determinism floor.** The repeats in procedure step 2 give σ per level (see Analysis). The measure
   procedure's own null arm must also pass on every map. Each repeat's sketch must be bit-identical to the
   first on the same map, given deterministic arms. If it is not, σ comes from the spread, and that is
   reported.
2. **Byte ledger.** Every map's footprint must equal ∅'s plus the sum of its atoms' extras, exactly, for all
   256. Bytes are additive by construction, so a mismatch is an engineering bug and the run stops.
3. **Full map.** The all-held map runs `production` against `production` on the same bytes. Its KL and its
   sketch must be at the null arm's floor.
4. **Sketch does not perturb.** On ∅, `positions.jsonl` with the sketch enabled must be byte-identical to
   the same run with it disabled.
5. **Sketch is linear.** Unit test: for random rows a, b, sketch(a + b) = sketch(a) + sketch(b) to f32
   rounding, with the frozen seed.
6. **Congruence with the sweep.** The cube vertices {`ffn`×Q4}, all-`attn` and all-`ffn` correspond to
   `late10-ffn`, `attn-protected` and `ffn-protected`. They are reported side by side with the sweep, as
   direction only, because the instrument differs.

## Forecasts (made before any cube reading; thresholds fixed at freeze)

All forecasts are over maps with |S| ≥ 2 that are scored at the level named.

- **F1 (world, L0).** The order-1 sketch error has median ≤ 0.10 and 90th percentile ≤ 0.25. The banked
  pairs interact by 2–5%, so this is forecast to pass, which is world 1 or 2.
- **F2 (curvature).** The order-1 median error at L1 and at L2 each exceeds L0's by more than L0's own
  noise-limited resolution (3σ / ‖f‖, median over the same maps).
- **F3 (aggregate, KL p99).** (b) is within 10% of (a) on ≥ 90% of maps; and (c) is within 10% of (a) on
  fewer maps than (b) is. The first clause is F3(b), the second F3(c) as used by the classification rule.
- **F4 (gate).** No map cheaper than {`ffn`×Q4} is admitted. In the sweep, `attn-protected` has p99 4.26
  against `late10-ffn`'s 1.26, and every cheaper map is attention-only.
- **F5 (replay).** R4 reaches zero byte regret by N = 40, and R2 does not. R4 reaches zero byte regret in
  fewer measurements than R3.

A failed forecast is recorded as failed, next to the evidence (`feedback_rung_order_and_falsified_forecasts`).

## Sequencing

1. MEASURE-PLAN-3 is already frozen (#588).
2. **Instrument PR** (branch `measure/logit-sketch`): the sketch in `measure` and `auto-rep run`, with no
   Granite result. Controls 4 and 5 are its tests
   (`a_sketch_adds_one_row_per_position_and_changes_no_position`, `project_is_linear`).
3. **Freeze PR for this document**, citing the merged instrument commit and fixing both seeds (cube order,
   sketch projection). It merges before any MEASURE-PLAN-3 reading, so the cube's analysis cannot be tuned to
   1c's result.
4. Engineering shared with 1c, which produces no result: declared group vocabularies in 1a/1b, not yet on
   main.
5. MEASURE-PLAN-3's anchor, gate and campaign run and close, so its frozen question is answered without
   knowledge of the cube.
6. This rung runs, starting with its pilot.

## Not claimed

- Anything about 1c's 12-group vocabulary, finer grains, other models, other banks or other codecs. The
  cube is 8 atoms on one model, one bank and one codec.
- Transfer. Whether a structure found here predicts another bank or model is the next rung's question.
- That additivity at quarter × family grain implies it at finer grain.
- Promotion or execution speed. Bytes are the objective.

## Next rungs, depending on the outcome

- **World 1:** make R4 the production proposer, then test whether 9 measurements generalise to 12 groups
  (1c's vocabulary, 13 measurements) and to a second model.
- **World 2:** give the gate a distributional objective (predicted per-position KL → p99 with UNCERTAINTY-2
  bands), keeping order-1 per-position marginals.
- **World 3:** a sparse order-2 model plus active measurement (R6 made a real arm), then measure how
  interactions scale with the number of atoms.
