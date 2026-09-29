# MEASURE-PLAN-3 and MEASURE-PLAN-3-CLOSURE: results

**Closed 2026-09-29.** Protocols: `docs/measure-plan-3.md` (frozen #588) and
`docs/measure-plan-3-closure.md` (frozen #640, amendments 1–2). Model: Granite 4.1 3B
(`c0650403`). Bank: Q-BANK-1, 69 sequences, 1,667 positions. Gate:
`granite-4.1-3b-plan-v1-vs-late10-ffn/v1` (#637). Every receipt is admissible and its
`positions_sha256` verifies. Readings: `bench/measure-plan-3/{anchor,campaign,closure}/`.

## Headline

1. **MEASURE-PLAN-3: no admitted representation among the 39 cheapest states the frozen search
   visited.** Outcome `BudgetSpent`, 39 of 39 measured, 0 admitted, and every proposal refused on
   all four criteria. The largest map reached held 2,447 MiB against the anchor's 3,040 MiB.
2. **MEASURE-PLAN-3-CLOSURE: 0 of 8 admitted, as forecast.** All eight inclusion-maximal cheaper
   attention maps fail the gate. *Under the additional, unverified assumption of admission
   monotonicity on the eight-group attention lattice, this excludes every cheaper map in this
   vocabulary.*
3. **Attention precision is a poor use of bytes on this model. Late-FFN precision is not.** Holding
   nearly all attention at source (737–773 MiB) lowers mean KL by 15–18% and p99 by about 6%.
   Holding FFN in layers 30–39 (862 MiB) lowers them by 57% and 74%.

## Limitation of the MEASURE-PLAN-3 freeze

The freeze says a fail makes "the anchor … the frontier in this vocabulary". That overclaims. In the
12-group vocabulary, all eight attention groups together cost exactly one FFN quarter (89.8 MiB per
`attn-o` quarter, 125.8 per `attn-qkv` quarter, 862.5 per `ffn` quarter). So 255 attention-only states
are strictly cheaper than the anchor, and a budget of 39 spent cheapest-first cannot cover them. The
closure check exists to address that gap, and its conclusion is only conditional.

## The anchor and the gate

| criterion | anchor = gate limit |
|---|---|
| KL p99 | 0.8823 |
| KL mean | 0.08731 |
| top-1 agreement | 84.28% (disagreement ≤ 0.1572) |
| ΔNLL mean | −0.01149 |
| positions | 1,667 |

## MEASURE-PLAN-3 campaign

Forecasts from the freeze:
- **Forecast 1 held.** Uniform NVFP4 fails on every criterion: KL p99 3.435, mean 0.2047, top-1
  82.54%, ΔNLL +0.0722.
- **Forecast 2** (the anchor admits itself) holds by construction, and is tested in #637.
- **Forecast 3:** fail.

Best reading of the 39 on each criterion, not necessarily from the same map:

| criterion | best of 39 | gate |
|---|---|---|
| KL p99 | 3.213 | ≤ 0.882 |
| KL mean | 0.1894 | ≤ 0.0873 |
| top-1 | 83.92% | ≥ 84.28% |
| ΔNLL | +0.0503 | ≤ −0.0115 |

The search visited the empty map, every combination of the four `attn-o` quarters, and small sets
of `attn-qkv` quarters, cheapest first. It never reached an FFN group.

**Run history.** A first run was stopped after 3 proposals, at about 24 minutes each. Two changes
followed. Parallel sample scoring (#639) is bit-identical: the anchor re-measured to the same
`positions_sha256`. Checkpoint and resume came with #638. The second run reproduced the first run's
readings byte for byte (`positions.jsonl` sha256 equal) and took 6.2 h for 39 proposals. The
campaign's build (`08945482`) is code-identical to main at `17ce2168`.

## MEASURE-PLAN-3-CLOSURE

| omitted group | extra MiB | KL p99 | KL mean | top-1 | ΔNLL | fails |
|---|---|---|---|---|---|---|
| `attn-qkv`×Q1 | 736.7 | 3.211 | 0.1692 | 85.42% | +0.0787 | p99, mean, ΔNLL |
| `attn-o`×Q1 | 772.7 | 3.483 | 0.1715 | 85.66% | +0.0765 | p99, mean, ΔNLL |
| `attn-qkv`×Q2 | 736.7 | 3.252 | 0.1748 | 85.06% | +0.0562 | p99, mean, ΔNLL |
| `attn-o`×Q2 | 772.7 | 3.243 | 0.1674 | 86.02% | +0.0696 | p99, mean, ΔNLL |
| `attn-qkv`×Q3 | 736.7 | 3.254 | 0.1746 | 84.52% | +0.0896 | p99, mean, ΔNLL |
| `attn-o`×Q3 | 772.7 | 3.243 | 0.1686 | 85.12% | +0.0710 | p99, mean, ΔNLL |
| `attn-qkv`×Q4 | 736.7 | 3.244 | 0.1737 | 85.18% | +0.0539 | p99, mean, ΔNLL |
| `attn-o`×Q4 | 772.7 | 3.243 | 0.1670 | 85.60% | +0.0738 | p99, mean, ΔNLL |

Each compiled pack's footprint equals the uniform pack's plus the frozen extra, to 0.1 MiB. All
eight pass the top-1 criterion. That is descriptive: no inference is drawn from a single bank-level
top-1 aggregate. They fail KL p99, KL mean and ΔNLL by wide margins.

## What the readings do and do not show

Established at sequence level. The unit is the sequence (paired over 69 sequences), and headline
intervals come from a paired sequence bootstrap of the pooled statistic:
- Protecting `attn-qkv` quarters lowers mean KL reliably, by about 2–5% each (sequence t ≈ −5 for
  Q3 and Q4). `attn-qkv`×Q3 also lowers ΔNLL (t ≈ −4.6).
- Single `attn-o` quarters have small or indistinguishable effects on mean KL.

Not established, and retired as claims:
- That early `attn-o` repairs the tail. Its p99 change had a 95% interval of [−0.41, +0.02].
- Criterion-dependent importance and error cancellation. The apparent sign reversals were churn
  among positions (MEASURE-PLAN-3-CLOSURE amendment 2).
- Any p99 ranking among the singleton attention maps. p99 is decided by about 17 positions, and its
  intervals overlap.

The rules that came out of this, now binding on AUTO-REP-LANDSCAPE-1 (amendments 2–3):
- A deterministic reading reproduces exactly on its bank, but is not thereby significant.
- The sequence is the unit, not the position.
- Large families of comparisons need a multiplicity correction.

## Next

- **AUTO-REP-LANDSCAPE-1** (#636, amendments 1–4): the concurrency-equivalence control, then the
  266-step cube on main `17ce2168`. It tests whether the precision landscape over {`attn`, `ffn`} ×
  quarter is low-order, and whether a cheap search could exploit that.
- **A finer vocabulary.** The savings, if any, lie below quarter × family grain. In the old Q-BANK-1
  sweep, `late5-ffn` came within about 2% of `late10-ffn` on p99 at half the bytes, and the quarter
  vocabulary cannot express it.
