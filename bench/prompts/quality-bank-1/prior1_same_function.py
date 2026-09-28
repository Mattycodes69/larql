#!/usr/bin/env python3
"""AUTO-REP-PRIOR-1 control 2: MLX and LARQL compute the same function.

    python3 prior1_same_function.py <hf-snapshot-dir> <container> <work-dir> <out.json>

The gradient prior is computed in MLX and applied to LARQL. If the two
engines disagree about Granite's forward (its embedding, attention, residual
and logit multipliers are where they most plausibly would), the prior scores
a different model and no verdict about it means anything.

Both engines are teacher-forced through identical ids: the 69 Q-BANK-1
prompts, tokenised once here by the container's own tokenizer and truncated
to 128 ids. LARQL runs `vindex3 exec --backend reference --bank` (naive f32,
sharing no arithmetic with `larql-compute`); MLX runs the pinned snapshot in
f32. Per position, KL(p_larql || p_mlx) in nats and top-1 agreement.

Amendment 1 of the plan set the comparator and dtype: the frozen form (MLX
BF16 against `production`) failed, and the diagnosis located the gap inside
LARQL's `production` backend, not in MLX. `--larql-dump DIR` reuses an
existing reference run over a byte-identical bank.

Frozen bar: top-1 agreement >= 0.99 and mean KL <= 1e-2 nats. A FAIL is a
harness failure, not a result: `score_prior1.py` refuses without a PASS.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import mlx_lm
import numpy as np

HERE = Path(__file__).resolve().parent
LARQL = os.environ.get("LARQL", "./target/release/larql")

# ---- frozen by docs/auto-rep-prior-1.md -----------------------------------
TOP1_MIN = 0.99
KL_MEAN_MAX = 1e-2
MAX_TOKENS = 128
LARQL_BACKEND = "reference"
BANK_FILE = "bank.jsonl"
DUMP_DIR = "larql-logits"
RECORD_SCHEMA = "auto-rep-prior-1/same-function/v1"


def refuse(message):
    raise SystemExit(f"REFUSED: {message}")


def bank(container):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(container / "tokenizer.json"))
    prompts = json.loads((HERE / "prompts.json").read_text())["prompts"]
    return [
        {"id": p["id"], "ids": tok.encode(p["text"]).ids[:MAX_TOKENS]}
        for p in prompts
    ]


def bank_text(entries):
    return "".join(json.dumps(e) + "\n" for e in entries)


def reuse_larql(entries, dump):
    bank_path = dump.parent / BANK_FILE
    if not bank_path.exists() or bank_path.read_text() != bank_text(entries):
        refuse(f"{dump} was not run over this bank ({bank_path} differs or is missing)")
    return dump


def run_larql(container, entries, work):
    path = work / BANK_FILE
    path.write_text(bank_text(entries))
    dump = work / DUMP_DIR
    dump.mkdir()
    cmd = [LARQL, "vindex3", "exec", str(container), "--tokens", "1",
           "--backend", LARQL_BACKEND, "--bank", str(path), "--dump-dir", str(dump)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        refuse(f"larql bank run failed:\n{r.stdout}\n{r.stderr}")
    return dump


def log_softmax(x):
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("snapshot", type=Path)
    ap.add_argument("container", type=Path)
    ap.add_argument("work", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--larql-dump", type=Path,
                    help="reuse a `--backend reference` dump dir; its sibling bank.jsonl must match")
    a = ap.parse_args()
    if a.output.exists():
        refuse(f"{a.output} exists")
    a.work.mkdir(parents=True, exist_ok=False)

    entries = bank(a.container)
    dump = reuse_larql(entries, a.larql_dump) if a.larql_dump else run_larql(a.container, entries, a.work)
    model, _ = mlx_lm.load(str(a.snapshot))
    model.set_dtype(mx.float32)

    kls, agree, per_prompt = [], 0, {}
    positions = 0
    for e in entries:
        n = len(e["ids"])
        raw = np.fromfile(dump / f"{e['id']}.f32", dtype="<f4")
        if raw.size % n:
            refuse(f"{e['id']}: dump of {raw.size} floats is not {n} rows")
        p_larql = log_softmax(raw.reshape(n, -1).astype(np.float64))
        logits = model(mx.array(e["ids"])[None])[0].astype(mx.float32)
        p_mlx = log_softmax(np.array(logits).astype(np.float64))
        if p_mlx.shape != p_larql.shape:
            refuse(f"{e['id']}: MLX {p_mlx.shape} vs LARQL {p_larql.shape}")
        kl = (np.exp(p_larql) * (p_larql - p_mlx)).sum(-1)
        top = (p_larql.argmax(-1) == p_mlx.argmax(-1)).sum()
        kls.extend(kl.tolist())
        agree += int(top)
        positions += n
        per_prompt[e["id"]] = {"positions": n, "kl_mean": float(kl.mean()), "top1": int(top)}

    kl_mean = float(np.mean(kls))
    top1 = agree / positions
    verdict = top1 >= TOP1_MIN and kl_mean <= KL_MEAN_MAX
    out = {
        "schema": RECORD_SCHEMA,
        "plan": "docs/auto-rep-prior-1.md",
        "larql_backend": LARQL_BACKEND,
        "mlx_compute_dtype": "float32",
        "larql_dump": str(dump),
        "positions": positions,
        "top1_agreement": top1,
        "kl_mean": kl_mean,
        "kl_max": float(np.max(kls)),
        "bar": {"top1_min": TOP1_MIN, "kl_mean_max": KL_MEAN_MAX},
        "verdict": "PASS" if verdict else "FAIL",
        "per_prompt": per_prompt,
    }
    a.output.write_text(json.dumps(out, indent=1))
    print(f"control 2  {out['verdict']}  top-1 {top1:.4f} (>= {TOP1_MIN})  "
          f"mean KL {kl_mean:.3e} (<= {KL_MEAN_MAX})  over {positions} positions")
    return 0 if verdict else 2


if __name__ == "__main__":
    sys.exit(main())
