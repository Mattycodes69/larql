#!/usr/bin/env python3
"""AUTO-REP-PRIOR-1 Part 1: rank the pool by MLX's score, apply the frozen bars.

    python3 score_prior1.py <scores.json> <same-function.json> <out.json>
        [--sweep granite-4.1-3b-sweep.json]

Aggregation, frozen in docs/auto-rep-prior-1.md:

    score(R) = sum of align(W) over R's tensors / extra MiB(R)

which is 1B′'s return metric with `align` in place of `num`, so the 1B′
aggregation, rho and Spearman are reused as they are.

1a (estimation; decides the evidence class) against mean-KL return per MiB:
    E1  late5-ffn ranks 1st
    E2  down-protected ranks 7th or 8th
    E3  score(late5-ffn) > score(late10-ffn) > score(late15-ffn)
1b (decision) is 1B′'s Granite conditions 1-3, verbatim, against p99.

Spearman is reported for both and rescues neither. The 13-region table is
secondary and changes no verdict.
"""
import argparse
import json
import sys
from pathlib import Path

import candidates as C
import score_1b_prime as B

HERE = Path(__file__).resolve().parent
MIB = 2 ** 20
SCORES_SCHEMA = "auto-rep-prior-1/scores/v1"
SAME_FUNCTION_SCHEMA = "auto-rep-prior-1/same-function/v1"
OUT_SCHEMA = "auto-rep-prior-1/part1/v1"
E2_MIN_RANK = 7
ALL_LAYERS = None
ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
FFN = ("gate_proj", "up_proj", "down_proj")

# The five sweep regions the 1B′ pool excluded for want of an o_proj site.
# MLX has o_proj gradients, so they are scored here, as secondary only.
SECONDARY = {
    "attn-protected": [(p, ALL_LAYERS) for p in ATTN],
    "o-protected": [("o_proj", ALL_LAYERS)],
    "late10-ffn-o": [(p, (30, 39)) for p in FFN] + [("o_proj", (30, 39))],
    "early10": [(p, (0, 9)) for p in ATTN + FFN],
    "late10": [(p, (30, 39)) for p in ATTN + FFN],
}


def refuse(message):
    raise SystemExit(f"REFUSED: {message}")


def region_scores(records, rules_by_label):
    out = {}
    for label, rules in rules_by_label.items():
        sel = [r for r in records if C.selects(r["tensor"], rules)]
        num = sum(r["align"] for r in sel)
        extra = sum(r["source_bytes"] - r["compiled_bytes"] for r in sel) / MIB
        out[label] = {"num": num, "extra_mib": extra, "score": num / extra, "tensors": len(sel)}
    return out


def truth(sweep, metric):
    arms = {c["label"]: c for c in sweep["candidates"]}
    base = arms[C.BASE_LABEL]
    pick = {"p99": lambda a: a["kl"]["p99"], "kl_mean": lambda a: a["kl"]["mean"]}[metric]
    return {
        label: (pick(base) - pick(arm)) / ((arm["payload_bytes"] - base["payload_bytes"]) / MIB)
        for label, arm in arms.items()
        if label != C.BASE_LABEL and arm["payload_bytes"] > base["payload_bytes"]
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scores", type=Path)
    ap.add_argument("same_function", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--sweep", type=Path, default=HERE / "granite-4.1-3b-sweep.json")
    a = ap.parse_args()
    if a.output.exists():
        refuse(f"{a.output} exists; Part 1 is scored once")

    same = json.loads(a.same_function.read_text())
    if same.get("schema") != SAME_FUNCTION_SCHEMA or same.get("verdict") != "PASS":
        refuse("control 2 has not passed; the prior would score a different function")
    doc = json.loads(a.scores.read_text())
    if doc.get("schema") != SCORES_SCHEMA:
        refuse(f"scores schema {doc.get('schema')}")
    records = doc["records"]
    sweep = json.loads(a.sweep.read_text())

    gaps = B.check_candidate_coverage(records)
    if gaps:
        refuse(f"candidates name tensors with no score: {gaps}")

    agg = region_scores(records, C.POOL)
    order = sorted(C.POOL, key=lambda l: -agg[l]["score"])
    ranks = {l: i for i, l in enumerate(order, 1)}
    t_mean, t_p99 = truth(sweep, "kl_mean"), truth(sweep, "p99")

    print(f"{'rank':>4} {'candidate':16s} {'+MiB':>8s} {'MLX align/MiB':>14s} {'meanKL/MiB':>11s} {'p99/MiB':>10s}")
    for i, l in enumerate(order, 1):
        neg = "  (negative)" if l in C.NEGATIVES else ""
        print(f"{i:4d} {l:16s} {agg[l]['extra_mib']:8.1f} {agg[l]['score']:14.6g} "
              f"{t_mean[l]:11.4g} {t_p99[l]:10.4g}{neg}")

    # ---- 1a: estimation, against the score's own target --------------------
    e1 = ranks["late5-ffn"] == 1
    e2 = ranks["down-protected"] >= E2_MIN_RANK
    e3 = agg["late5-ffn"]["score"] > agg["late10-ffn"]["score"] > agg["late15-ffn"]["score"]
    s_mean = B.spearman([agg[l]["score"] for l in order], [t_mean[l] for l in order])
    part_1a = e1 and e2 and e3
    print("\n1a  ESTIMATION (truth: mean-KL return / MiB)")
    print(f"  E1 late5-ffn ranks 1st              : {'PASS' if e1 else 'FAIL'}  (rank {ranks['late5-ffn']})")
    print(f"  E2 down-protected ranks >= {E2_MIN_RANK}        : {'PASS' if e2 else 'FAIL'}  (rank {ranks['down-protected']})")
    print(f"  E3 late5 > late10 > late15          : {'PASS' if e3 else 'FAIL'}")
    print(f"  Spearman vs mean-KL/MiB {s_mean:+.3f}  (reported; rescues nothing)")

    # ---- 1b: decision, 1B′'s bar verbatim ------------------------------------
    n = len(order)
    rho = B.rho(agg)
    c1 = all(ranks["late5-ffn"] < ranks[g] for g in C.NEGATIVES)
    c2 = all(ranks[g] > n / 2 for g in C.NEGATIVES)
    c3 = (agg["late5-ffn"]["score"] > agg["late10-ffn"]["score"]
          and agg["late5-ffn"]["score"] > agg["late15-ffn"]["score"]
          and rho is not None and rho > 1.0)
    s_p99 = B.spearman([agg[l]["score"] for l in order], [t_p99[l] for l in order])
    part_1b = c1 and c2 and c3
    print("\n1b  DECISION (1B′ Granite bar, truth: p99 return / MiB)")
    print(f"  1. late5-ffn above all negatives    : {'PASS' if c1 else 'FAIL'}")
    print(f"  2. v/k/down all in the bottom half  : {'PASS' if c2 else 'FAIL'}  "
          f"({ {g: ranks[g] for g in C.NEGATIVES} })")
    print(f"  3. knee survives, rho > 1           : {'PASS' if c3 else 'FAIL'}  "
          f"(rho {rho if rho is None else round(rho, 3)}, truth {round(B.truth_rho(sweep), 2)})")
    print(f"  Spearman vs p99/MiB {s_p99:+.3f}  (reported; rescues nothing)")

    # ---- secondary: all 13 sweep regions ------------------------------------
    wide = region_scores(records, {**C.POOL, **SECONDARY})
    wide_order = sorted(wide, key=lambda l: -wide[l]["score"])
    print("\nsecondary (13 regions, o_proj included; changes no verdict):")
    print("  " + ", ".join(f"{i}. {l}" for i, l in enumerate(wide_order, 1)))

    evidence = ({"OrderingProxy": {"calibration": doc["calibration_sha256"]}}
                if part_1a else "Unusable")
    out = {
        "schema": OUT_SCHEMA,
        "plan": "docs/auto-rep-prior-1.md",
        "scores_calibration_sha256": doc["calibration_sha256"],
        "pool_order": order,
        "regions": agg,
        "part_1a": {"E1": e1, "E2": e2, "E3": e3, "spearman_mean_kl": s_mean, "verdict": part_1a},
        "part_1b": {"c1": c1, "c2": c2, "c3": c3, "rho": rho, "spearman_p99": s_p99, "verdict": part_1b},
        "secondary_order": wide_order,
        "evidence": evidence,
    }
    a.output.write_text(json.dumps(out, indent=1))
    print(f"\n=> 1a {'PASS' if part_1a else 'FAIL'}   1b {'PASS' if part_1b else 'FAIL'}   evidence {evidence}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
