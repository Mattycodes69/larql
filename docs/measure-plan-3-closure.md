# MEASURE-PLAN-3-CLOSURE: the eight maximal cheaper states

**Frozen 2026-09-29, while MEASURE-PLAN-3's campaign was running, before this experiment's first
measurement.** It is post-hoc relative to MEASURE-PLAN-3's freeze (`docs/measure-plan-3.md`, #588) and
pre-registered relative to the campaign's outcome and to AUTO-REP-LANDSCAPE-1 (#636). At freeze the
campaign had recorded 3 of its 39 measurements: uniform NVFP4, {`attn-o`×Q4} and {`attn-o`×Q3}, all
refused. None is one of the eight states below.

Programme: REPRESENT.

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

AUTO-REP-LANDSCAPE-1's cube measures every subset of {`attn`, `ffn`} × quarter. That is a coarser
grouping, but it can observe violations of monotonicity directly. This experiment makes no claim about
whether the assumption holds.

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
| **0 / 8 admitted** | **Conditional closure:** no strictly cheaper state in this vocabulary is admitted, *assuming admission monotonicity*. |
| **1–8 / 8 admitted** | **A search-depth miss by MEASURE-PLAN-3:** a cheaper admitted map exists in this vocabulary. The admitted states are named, with their bytes. |
| LANDSCAPE later observes a violation of admission monotonicity | The conditional closure is **withdrawn**, not reinterpreted. The eight measurements stay as evidence. |

Whatever the outcome, the eight are the most attention precision any cheaper map can hold. So they are
the strongest test available before LANDSCAPE of whether attention precision alone can approach the
late-FFN anchor.

## Forecast

**0 / 8 admitted.** In the Q-BANK-1 sweep (another instrument, direction only), holding *all* attention
at source left KL p99 at 4.26 against `late10-ffn`'s 1.26. In the campaign so far, single `o_proj`
quarters move p99 by about 0.3%.

## Not claimed

- That admission monotonicity holds.
- Anything at a finer grain than quarter × family, or on another model or bank.
- That the anchor is optimal among maps of equal or greater cost.
