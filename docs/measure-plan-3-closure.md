# MEASURE-PLAN-3-CLOSURE: a conditional boundary check on the eight maximal cheaper states

**Frozen 2026-09-29, while MEASURE-PLAN-3's campaign was running, before this experiment's first
measurement.** It is post-hoc relative to MEASURE-PLAN-3's freeze (`docs/measure-plan-3.md`, #588) and
pre-registered relative to the campaign's outcome and to AUTO-REP-LANDSCAPE-1 (#636). At freeze the
campaign had recorded 3 of its 39 measurements: uniform NVFP4, {`attn-o`×Q4} and {`attn-o`×Q3}, all
refused. None is one of the eight states below.

Programme: REPRESENT.

## Amendment 2 (2026-09-29, before any of the eight is measured): amendment 1's counterexample is not significant

Amendment 1 called the {`attn-o`×Q2} reading "real, not noise" because the measurement is deterministic.
Determinism means the reading reproduces exactly on this bank. It does not make a difference significant.
Paired against uniform NVFP4 over the same 1,667 positions:

| map | top-1 gained / lost (net, McNemar z) | ΔNLL paired t | KL paired t |
|---|---|---|---|
| `attn-o`×Q4 | 10 / 7 (+3, +0.73) | −1.72 | −2.18 |
| `attn-o`×Q3 | 16 / 13 (+3, +0.56) | −1.12 | −3.95 |
| `attn-o`×Q2 | 12 / 14 (−2, −0.39) | +0.76 | −1.61 |
| `attn-o`×Q1 | 12 / 17 (−5, −0.93) | −2.20 | −4.26 |

Q2's worse top-1 and ΔNLL, and Q1's worse top-1, cannot be told apart from zero. Every change moves
about 20–30 top-1 positions each way, and these net differences are that churn. The corrected statement:
**on this bank the aggregates moved the wrong way, and the paired differences are indistinguishable from
zero.** This is no evidence of error cancellation, and no evidence against per-criterion monotonicity
either. The KL improvements are broad and significant (Q1 and Q3 at t ≈ −4; KL is lower at 55–64% of
positions).

The experiment stays a conditional boundary check. Admission monotonicity is unverified whatever this
datum says, and the gate compares bank-level aggregates, so bank-sampling variation (UNCERTAINTY-2) is
part of what "admitted" means. Amendment 1's point about LANDSCAPE still stands.

## Amendment 1 (2026-09-29, before any of the eight is measured)

Two changes after the freeze, both made before this experiment's first measurement.

1. **Known counterevidence goes into the protocol.** The campaign's fourth reading, {`attn-o`×Q2}, is
   *worse* than uniform NVFP4 on two criteria: top-1 agreement 82.42% against 82.54%, and ΔNLL +0.0736
   against +0.0722. The measurement is deterministic, so this is real, not noise. *(Corrected by amendment
   2: reproducible on this bank, but not significant.)* Holding a group at
   source can make a metric worse, because errors introduced in different places can partly cancel.
   **Per-criterion monotonicity is therefore empirically false.** Only admission monotonicity is assumed,
   and this counterexample is why the whole-space conclusion below is conditional.
2. **LANDSCAPE cannot verify the assumption.** Its cube's attention atoms merge `attn-o` and `attn-qkv`
   within each quarter, four atoms where this lattice has eight. It can make admission monotonicity more
   or less credible at the coarser grain: a violation there counts against it, and a smooth, additive
   surface counts indirectly in its favour. It cannot certify the assumption over this experiment's 255
   states. The assumption stays **unverified**, not "pending LANDSCAPE". This is a **conditional boundary
   check**, not a certification.

## Why

MEASURE-PLAN-3's campaign cannot certify its own negative. In its 12-group vocabulary the bytes held at
source are 89.8 MiB per `attn-o` quarter, 125.8 MiB per `attn-qkv` quarter and 862.5 MiB per `ffn`
quarter. So all eight attention groups together cost exactly one FFN quarter, the anchor. Every
attention-only subset except the full set, **255 states**, is strictly cheaper than the anchor. The
campaign's budget is 39, spent cheapest first. If it ends `BudgetSpent`, it has visited the cheapest 39
and said nothing about the other 216.

**Result language for MEASURE-PLAN-3, corrected here.** Its freeze says a fail makes "the anchor … the
frontier in this vocabulary". A spent budget supports only: **no admitted representation among the 39
cheapest states the frozen search visited.** The campaign's results must use the narrower statement, and
must record this as a limitation of that freeze.

## The assumption, stated exactly

This experiment **uses** one assumption and does not test it. **Admission monotonicity**: for protected
sets S ⊆ T in this vocabulary, if the gate admits S then it admits T. Protecting more groups can never
turn an admitted map into a refused one. This is weaker than every metric improving monotonically, and
it is all the argument needs.

Under it, the eight states that hold all attention except one group are the **inclusion-maximal**
strictly-cheaper states. Every other cheaper attention subset is contained in at least one of them. So
admission anywhere in the cheaper space implies admission of at least one of the eight.

This experiment makes no claim about whether the assumption holds, and no planned experiment verifies it
on this lattice (amendment 1).

## The eight maps

Each map holds every attention group at source except the one named. Everything else is uniform NVFP4.

| omitted group | held at source | extra MiB |
|---|---|---|
| `attn-qkv`×Q1 | `o_proj` 0–39; `q/k/v_proj` 10–39 | 736.7 |
| `attn-qkv`×Q2 | `o_proj` 0–39; `q/k/v_proj` 0–9, 20–39 | 736.7 |
| `attn-qkv`×Q3 | `o_proj` 0–39; `q/k/v_proj` 0–19, 30–39 | 736.7 |
| `attn-qkv`×Q4 | `o_proj` 0–39; `q/k/v_proj` 0–29 | 736.7 |
| `attn-o`×Q1 | `q/k/v_proj` 0–39; `o_proj` 10–39 | 772.7 |
| `attn-o`×Q2 | `q/k/v_proj` 0–39; `o_proj` 0–9, 20–39 | 772.7 |
| `attn-o`×Q3 | `q/k/v_proj` 0–39; `o_proj` 0–19, 30–39 | 772.7 |
| `attn-o`×Q4 | `q/k/v_proj` 0–39; `o_proj` 0–29 | 772.7 |

The anchor's extra is 862.5 MiB, so all eight are strictly cheaper. Each map is compiled with `vindex3
represent --protect <proj>@<lo>-<hi>`, one rule per range above. Its compiled footprint must equal the
uniform pack's plus the table's extra; if it does not, the run stops as an engineering fault.

## Same authority as the anchor

- **Model:** `~/chris-models/granite-4.1-3b.vindex3` (registry pin `c0650403`).
- **Bank:** the Q-BANK-1 token bank the anchor used (`bank_id 6a957421…`), all 69 sequences.
- **Arms:** reference `production` on the source's canonical bytes; candidate `production-nvfp4` on the
  compiled pack (`vindex3 measure`).
- **Instrument:** plan-v1, the build MEASURE-PLAN-3's campaign runs on. Parallel scoring (#639) was shown
  bit-identical to the anchor's serial measurement: equal `positions_sha256`, summary and facts.
- **Gate:** `granite-4.1-3b-plan-v1-vs-late10-ffn/v1` (#637), applied exactly as the loop applies it.
  Admitted when KL p99, KL mean, top-1 disagreement and ΔNLL mean are each ≤ the anchor's value, and
  positions ≥ 1,667.
- **Evidence:** every receipt must be admissible and pass `verify_record` before its numbers are read.

## Procedure

1. Only after MEASURE-PLAN-3's campaign has closed and its record is written.
2. Compile and measure all eight. **No early stopping:** the aim is the maximal boundary, not the first
   admission. They are independent and deterministic, so they may run up to four at a time, as in
   LANDSCAPE amendment 1.
3. Record each map's four criteria, margins and verdict.

## Interpretation, frozen

| outcome | reading |
|---|---|
| **0 / 8 admitted** | "All eight inclusion-maximal cheaper attention maps fail the gate. Under the additional, unverified assumption of admission monotonicity on the eight-group attention lattice, this excludes every cheaper map." |
| **1–8 / 8 admitted** | **A search-depth miss by MEASURE-PLAN-3:** a cheaper admitted map exists in this vocabulary. The admitted states are named, with their bytes. |
| LANDSCAPE later observes a violation of admission monotonicity at its coarser grain | The conditional exclusion is **withdrawn**, not reinterpreted. The eight measurements stay as evidence. A smooth LANDSCAPE surface leaves the assumption unverified. |

Whatever the outcome, the eight are the most attention precision any cheaper map can hold. So they are
the strongest test available before LANDSCAPE of whether attention precision alone can approach the
late-FFN anchor.

## Forecast

**0 / 8 admitted.** In the Q-BANK-1 sweep (another instrument, direction only), holding *all* attention
at source left KL p99 at 4.26 against `late10-ffn`'s 1.26. In the campaign so far, single `o_proj`
quarters move p99 by about 0.3%.

## Not claimed

- That admission monotonicity holds, or that any planned experiment can verify it on this lattice.
- Anything about per-criterion monotonicity. Amendment 1's counterexample is not significant (amendment 2).
- Anything at a finer grain than quarter × family, or on another model or bank.
- That the anchor is optimal among maps of equal or greater cost.
