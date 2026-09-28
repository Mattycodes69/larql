#!/usr/bin/env python3
"""AUTO-REP-PRIOR-1: MLX's gradient sensitivity, against LARQL's codec.

    python3 prior1_mlx_scores.py <hf-snapshot-dir> <reconstruction-dir> <container> <out.json>
        [--inventory granite-4.1-3b-sensitivity-1a.json] [--grad-checkpoint]

Read `docs/auto-rep-prior-1.md` first. This is the one scoring pass: it
refuses to overwrite an existing output, and every control below is a
harness check whose failure stops the run before any gradient is taken.

    pins       mlx_lm / mlx versions, the upstream function's source hash,
               the model revision, the source payload digest, the codec
    control 4  MLX leaf names map one-to-one onto the 280 LARQL tensors
    control 1  every exported W_low reproduces the banked 1A rel_error
               against MLX's own copy of the source, to 1e-4 relative
    control 3  calibration_v5 shares no 13-gram with Q-BANK-1's prompts
               (in the container tokenizer's ids)

Control 2 (MLX and LARQL compute the same function) is
`prior1_same_function.py`; `score_prior1.py` refuses without its PASS.

`<reconstruction-dir>` comes from
`larql vindex3 sensitivity <container> --output X --reconstruction DIR`.
"""
import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx_lm
import mlx_lm.quant.dynamic_quant as upstream
from mlx.utils import tree_flatten
from mlx_lm.quant.utils import load_data

from prior1_estimate import estimate_sensitivities

HERE = Path(__file__).resolve().parent

# ---- frozen by docs/auto-rep-prior-1.md -----------------------------------
MLX_LM_VERSION = "0.29.1"
MLX_VERSION = "0.30.1"
UPSTREAM_SOURCE_SHA256 = "385dfa6ceed997e870af2fe0dc43548973e3f0b5eb5c4031dfabb616ef6607b5"
MODEL_REVISION = "c0650403e44e78ec0262dab1c90914c65b196c4e"
SOURCE_OBJECT = "target.decoder_stack"
SOURCE_PAYLOAD_SHA256 = "374562b3ff81c81fb72f1fcd4c842912ef75e167f8c0e9568997965a15f2612c"
CODEC = "nvfp4/rev1"
ENCODER = "nvfp4-nearest-v1"
MANIFEST_SCHEMA = "larql.vindex3.reconstruction/v1"
SEED = 123
SEQUENCE_LENGTH = 512
ALL_SAMPLES = -1
BATCH_SIZE = 4
REL_ERROR_TOLERANCE = 1e-4
NGRAM = 13
CALIBRATION_FILE = Path.home() / ".cache/mlx-lm/calibration_v5.txt"
MLX_LAYER_PREFIX = "model.layers."
LARQL_WEIGHT_SUFFIX = ".weight"
EXPECTED_TENSORS = 280
PARAMS_PER_MILLION = 1e6
RECORD_SCHEMA = "auto-rep-prior-1/scores/v1"


def refuse(message):
    raise SystemExit(f"REFUSED: {message}")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_pins(snapshot):
    if mlx_lm.__version__ != MLX_LM_VERSION:
        refuse(f"mlx_lm {mlx_lm.__version__}, plan pins {MLX_LM_VERSION}")
    if mx.__version__ != MLX_VERSION:
        refuse(f"mlx {mx.__version__}, plan pins {MLX_VERSION}")
    up = inspect.getsource(upstream.estimate_sensitivities)
    got = hashlib.sha256(up.encode()).hexdigest()
    if got != UPSTREAM_SOURCE_SHA256:
        refuse(f"upstream estimate_sensitivities changed ({got}); the committed diff no longer describes the copy")
    if snapshot.name != MODEL_REVISION:
        refuse(f"snapshot {snapshot.name} is not the registry pin {MODEL_REVISION}")


def larql_name(mlx_leaf):
    """`model.layers.N.<proj path>` -> `N.<proj path>.weight`, or None."""
    if not mlx_leaf.startswith(MLX_LAYER_PREFIX):
        return None
    return mlx_leaf[len(MLX_LAYER_PREFIX):] + LARQL_WEIGHT_SUFFIX


def load_manifest(recon):
    m = json.loads((recon / "manifest.json").read_text())
    if m["schema"] != MANIFEST_SCHEMA:
        refuse(f"manifest schema {m['schema']}")
    if (m["codec"], m["encoder"]) != (CODEC, ENCODER):
        refuse(f"manifest codec {m['codec']} / {m['encoder']}, plan freezes {CODEC} / {ENCODER}")
    if m["source_payloads"].get(SOURCE_OBJECT) != SOURCE_PAYLOAD_SHA256:
        refuse(f"reconstruction was made from a different {SOURCE_OBJECT} payload")
    return m


def control_4_names(model, manifest, inventory):
    leaves = {
        k: l
        for k, l in tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module)
        if hasattr(l, "to_quantized")
    }
    mapped = {larql_name(k): k for k in leaves if larql_name(k)}
    exported = {t["tensor"] for t in manifest["tensors"]}
    banked = {t["tensor"] for t in inventory}
    if not (len(exported) == len(banked) == EXPECTED_TENSORS and exported == banked):
        refuse(f"export ({len(exported)}) and 1A inventory ({len(banked)}) disagree on the population")
    missing = sorted(exported - mapped.keys())
    if missing:
        refuse(f"{len(missing)} exported tensors have no MLX leaf, e.g. {missing[:3]}")
    ignored = sorted(k for k in leaves if larql_name(k) not in exported)
    print(f"control 4  PASS  {len(exported)} tensors map one-to-one; ignored MLX leaves: {ignored}")
    return {t: (mapped[t], leaves[mapped[t]]) for t in exported}, ignored


def control_1_reconstruction(recon, manifest, inventory, by_tensor):
    banked = {t["tensor"]: t for t in inventory}
    low = {}
    worst = 0.0
    for t in manifest["tensors"]:
        path = recon / t["file"]
        if sha256_file(path) != t["file_sha256"]:
            refuse(f"{t['file']}: bytes differ from the manifest")
        w_low = mx.load(str(path))["weight"]
        leaf_name, leaf = by_tensor[t["tensor"]]
        w_high = leaf.weight.astype(mx.float32)
        if tuple(w_low.shape) != tuple(w_high.shape):
            refuse(f"{t['tensor']}: export {w_low.shape} vs MLX {w_high.shape}")
        rel = (mx.sum((w_low - w_high) ** 2) / mx.sum(w_high ** 2)).item()
        for label, want in (("manifest", t["rel_error"]), ("1A", banked[t["tensor"]]["rel_error"])):
            dev = abs(rel - want) / want
            worst = max(worst, dev)
            if dev > REL_ERROR_TOLERANCE:
                refuse(f"{t['tensor']}: rel_error {rel:.9g} vs {label} {want:.9g} ({dev:.2e} > {REL_ERROR_TOLERANCE})")
        low[leaf_name] = w_low
    print(f"control 1  PASS  {len(low)} reconstructions, worst relative deviation {worst:.2e}")
    return low, worst


def ngrams(ids, n):
    return {tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)}


def control_3_disjoint(container):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(container / "tokenizer.json"))
    cal = ngrams(tok.encode(CALIBRATION_FILE.read_text(), add_special_tokens=False).ids, NGRAM)
    prompts = json.loads((HERE / "prompts.json").read_text())["prompts"]
    shared = {}
    for p in prompts:
        hit = ngrams(tok.encode(p["text"], add_special_tokens=False).ids, NGRAM) & cal
        if hit:
            shared[p["id"]] = len(hit)
    if shared:
        refuse(f"calibration shares {NGRAM}-grams with Q-BANK-1: {shared}")
    print(f"control 3  PASS  0 shared {NGRAM}-grams across {len(prompts)} prompts")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("snapshot", type=Path)
    ap.add_argument("reconstruction", type=Path)
    ap.add_argument("container", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--inventory", type=Path, default=HERE / "granite-4.1-3b-sensitivity-1a.json")
    ap.add_argument("--grad-checkpoint", action="store_true")
    a = ap.parse_args()

    if a.output.exists():
        refuse(f"{a.output} exists; this rung scores once")
    check_pins(a.snapshot)
    manifest = load_manifest(a.reconstruction)
    inventory = json.loads(a.inventory.read_text())

    model, tokenizer = mlx_lm.load(str(a.snapshot))
    # Amendment 1: compute in f32, the dtype control 2 verified against
    # LARQL's reference. BF16 -> f32 is exact, so the weights are unchanged.
    model.set_dtype(mx.float32)
    by_tensor, ignored = control_4_names(model, manifest, inventory)
    low, worst = control_1_reconstruction(a.reconstruction, manifest, inventory, by_tensor)

    # main()'s order upstream: seed, then load_data. load_data also fetches
    # the pinned calibration file on first use.
    mx.random.seed(SEED)
    data = load_data(tokenizer, num_samples=ALL_SAMPLES, sequence_length=SEQUENCE_LENGTH)
    calibration_sha256 = sha256_file(CALIBRATION_FILE)
    control_3_disjoint(a.container)

    scores = dict(estimate_sensitivities(
        model, data, low, batch_size=BATCH_SIZE,
        gradient_accum_dtype=mx.float32, gradient_checkpoint=a.grad_checkpoint,
    ))

    banked = {t["tensor"]: t for t in inventory}
    records = []
    for tensor, (leaf_name, leaf) in sorted(by_tensor.items()):
        size = leaf.weight.size
        mlx_score = scores[leaf_name]
        b = banked[tensor]
        records.append({
            "tensor": tensor,
            "layer": int(tensor.split(".")[0]),
            "projection": tensor.split(".")[-2],
            "mlx": mlx_score,
            "align": mlx_score * size / PARAMS_PER_MILLION,
            "params": size,
            "source_bytes": b["source_bytes"],
            "compiled_bytes": b["compiled_bytes"],
        })

    out = {
        "schema": RECORD_SCHEMA,
        "plan": "docs/auto-rep-prior-1.md",
        "mlx_lm": mlx_lm.__version__,
        "mlx": mx.__version__,
        "upstream_source_sha256": UPSTREAM_SOURCE_SHA256,
        "harness_diff_sha256": sha256_file(HERE / "prior1_estimate.upstream.diff"),
        "model_revision": MODEL_REVISION,
        "source_payload_sha256": SOURCE_PAYLOAD_SHA256,
        "codec": CODEC,
        "encoder": ENCODER,
        "w_low_dtype": "float32",
        "mlx_compute_dtype": "float32",
        "amendment": "docs/auto-rep-prior-1.md#amendment-1-2026-09-28-control-2s-comparator-and-mlxs-compute-dtype",
        "calibration_sha256": calibration_sha256,
        "calibration": {"seed": SEED, "sequence_length": SEQUENCE_LENGTH,
                        "sequences": int(data.shape[0]), "batch_size": BATCH_SIZE},
        "control_1_worst_relative_deviation": worst,
        "ignored_mlx_leaves": ignored,
        "records": records,
    }
    a.output.write_text(json.dumps(out, indent=1))
    print(f"scored {len(records)} tensors over {data.shape[0]} sequences -> {a.output}")


if __name__ == "__main__":
    sys.exit(main())
