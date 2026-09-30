# CONTINUATION-CODEC-2 — reconnaissance

**Status: reconnaissance, not a freeze.** Nothing here is a threshold. It
measures the yardstick CODEC-1 could not use, so that CODEC-2's rules are
chosen from evidence instead of forecast. The freeze follows in its own PR.

## Why a successor, not an amendment

CONTINUATION-CODEC-1 closed with **quality UNINFORMATIVE, residency
ESTABLISHED** ([notes](represent/forecasts/continuation-codec-1-notes.json),
`closure`). Its frozen informativeness guard fired: the yardstick C (row/v1 on
the Q4_K-weight container) agreed with the reference R on top-1 at only
**0.796** of scored positions, against a floor of **0.90** that had never been
measured on the bank. Its rule also mixed two authorities: the mean and tail
rules were relative to C, but the confident-band rule was absolute
(≥ 99.5%), a bar C itself does not reach (0.941).

CODEC-1 is preserved as run. CODEC-2 asks a better-decomposed question:

| | question | arms | role |
|---|---|---|---|
| **A** | is C a usable baseline? | R ↔ C | calibration; a narrow guard |
| **B** | what extra damage does KV compression add to the already-quantised execution? | C ↔ H, C ↔ H3 | **the codec verdict** |
| **C** | what does the deployed system look like? | R ↔ H, R ↔ H3 | reported; acceptance only if separately declared |

## Contamination boundary

CODEC-1's pooled H and H3 results on the Pride and Prejudice bank are known,
and its checkpoints hold their per-position records. Therefore:

- this reconnaissance reads **only the R↔C records** (`n{1024,4096,8192}-c.json`,
  digests in [continuation-codec-1-records.json](represent/forecasts/continuation-codec-1-records.json));
  no per-position H or H3 record was opened and no paired H−C statistic was
  computed;
- CODEC-2 **adjudicates on a fresh, held-out bank**, frozen before any arm
  runs on it. Bank 1 is calibration only.

## A — the yardstick on bank 1 (R ↔ C, 1,536 scored positions)

Per rung (each rung scores the last 512 decode positions of its prefix):

| rung | top-1 | KL mean | p50 | p90 | p99 | max |
|---|---|---|---|---|---|---|
| 1,024 | 0.754 | 0.403 | 0.146 | 1.041 | 3.505 | 6.52 |
| 4,096 | 0.816 | 0.189 | 0.086 | 0.394 | 1.715 | 4.48 |
| 8,192 | 0.818 | 0.138 | 0.073 | 0.304 | 1.024 | 2.42 |
| pooled | **0.796** | **0.243** | 0.091 | 0.578 | 2.355 | 6.52 |

Uncertainty (bootstrap, 2,000 resamples): pooled top-1 95% CI
**[0.775, 0.816]**; KL mean **[0.201, 0.289]** with 32-position blocks
([0.218, 0.269] per position). Flip lag-1 autocorrelation is **0.005**:
positions behave as independent, so paired per-position tests (McNemar,
paired bootstrap) are appropriate. Per-position KL sd is 0.504 — heavy-tailed
relative to its mean.

By R's margin (top-1 minus top-2 probability):

| margin | n | top-1 | KL mean | KL p99 |
|---|---|---|---|---|
| [0, 0.05) | 150 | 0.393 | 0.325 | 2.13 |
| [0.05, 0.1) | 101 | 0.505 | 0.327 | 2.42 |
| [0.1, 0.2) | 166 | 0.675 | 0.265 | 1.72 |
| [0.2, 0.3) | 132 | 0.712 | 0.358 | 3.50 |
| [0.3, 0.5) | 205 | 0.834 | 0.308 | 2.30 |
| [0.5, 0.7) | 191 | 0.885 | 0.308 | 3.98 |
| [0.7, 0.9) | 217 | 0.908 | 0.248 | 2.94 |
| [0.9, 0.99) | 220 | 0.991 | 0.081 | 0.92 |
| [0.99, 1] | 154 | 0.987 | 0.044 | 1.05 |

Confident-band agreement, R margin ≥ t:

| t | n | top-1 | per rung (1,024 / 4,096 / 8,192) |
|---|---|---|---|
| 0.5 | 782 | 0.941 | 0.888 / 0.958 / 0.996 |
| 0.7 | 591 | 0.959 | 0.929 / 0.966 / 0.994 |
| 0.9 | 374 | 0.989 | 0.987 / 0.991 / 0.990 |
| 0.99 | 154 | 0.987 | 1.000 / 0.981 / 0.975 |

### What A shows

1. **C is a coarse but coherent baseline.** It flips top-1 on 20% of
   positions, almost all where R itself is uncertain: 16% of positions have
   margin < 0.1 and there C agrees only 39–51%. Above margin 0.9 it agrees
   ~99%.
2. **No absolute top-1 bar near 99.5% is reachable by C on any band.** Even
   at margin ≥ 0.99 C flips 1.3% (2 of 154). An absolute confident-band rule
   would reject a codec for error the yardstick already carries — CODEC-1's
   rule (3) was structurally unmeetable.
3. **Rung is confounded with text.** Each rung scores a different 512-token
   passage (the last 512 of its prefix), so the 1,024 rung's worse numbers
   (0.754, KL p99 3.5) cannot be attributed to context length. A fresh bank
   does not remove this; per-rung results should be reported, never read as
   a length law.

## B — what the codec verdict can resolve

The verdict is a paired, per-position comparison of H against C, both
measured against R on the same positions. For top-1 as non-inferiority
(`agreement(H,R) ≥ agreement(C,R) − δ`, one-sided α = 0.05, power 0.8, true
difference 0), the positions needed depend on the unknown discordance rate
p_d (the fraction of positions where exactly one of H, C agrees with R):

| p_d | δ = 0.01 | δ = 0.02 | δ = 0.03 |
|---|---|---|---|
| 0.10 | 6,180 | 1,545 | 687 |
| 0.20 | 12,360 | 3,090 | 1,373 |
| 0.30 | 18,540 | 4,635 | 2,060 |

C alone flips 20%, so p_d between 0.1 and 0.3 is the planning range. With
CODEC-1's shape (3 rungs × 512 = 1,536 positions), **δ ≈ 0.03 is resolvable;
δ = 0.01 is not** without roughly 10× the positions. Doubling decode to
1,024 per rung (3,072 positions) makes δ = 0.03 robust across the whole p_d
range and δ = 0.02 resolvable at p_d ≤ 0.2, at roughly twice the arm time.

For KL, the incremental quantity is `d_i = KL(R‖H)_i − KL(R‖C)_i`. Its
distribution is unknown before the run (and must stay so); KL(R‖C) has sd
0.50 against a mean of 0.24, so a mean rule needs a block-bootstrap interval,
and a tail rule should be stated on paired quantities rather than on two
separately computed p99s.

## Decisions the freeze must make (options, not choices)

1. **Bank 2.** A public-domain text disjoint from bank 1, chosen by a rule
   written before it is tokenised (source, edition, start, paragraph rule), with
   CODEC-1's P0b checks: tokenizer byte-identical across both containers, 0
   unknown tokens, 8,193 IDs, digest committed. Open: prose again (comparable
   to bank 1) or a different register; a heavily memorised canonical text
   shifts the margin distribution upward.
2. **Guard A.** Its job narrows to "C is not so broken that incremental
   comparison is meaningless". Candidates: pooled R↔C top-1 and KL mean inside
   a band derived from bank 1 (e.g. top-1 ≥ 0.70, KL mean ≤ 0.50 — well
   outside bank 1's CIs), fixed before bank 2 is scored.
3. **Rule B.** Top-1 non-inferiority with δ from the table above (and the
   positions per rung that δ requires); KL mean non-inferiority on the paired
   `d_i` with a stated ε and block bootstrap; a confident-band rule that is
   also relative to C, on a band whose C agreement leaves room (margin ≥ 0.9
   on bank 1: 0.989).
4. **Report C.** R↔H and R↔H3 reported with the same summaries; no acceptance
   unless declared.
5. **Positions.** 512 or 1,024 decode positions per rung — the table ties this
   to δ.

## Instrument deltas the freeze will need

- The C3 harness hard-codes bank 1 (`BANK_PATH`, `BANK_LEN`,
  `BANK_IDS_SHA256` in `crates/larql-kv/tests/continuation_mem_1/codec_1_c3.rs`)
  and CODEC-1's rule constants (`CONFIDENT_TOP1_MIN`, `YARDSTICK_TOP1_MIN`).
  CODEC-2 needs its own run module with bank 2's authority and its own rule,
  leaving CODEC-1's module untouched as the record of what ran.
- The paired statistics (McNemar on top-1, paired bootstrap on `d_i`) are
  computable from the per-position records the harness already writes; no
  new logit capture is needed.
- Residency is not re-adjudicated (CODEC-1 established it); CODEC-2 records
  it as a regression check only.
