"""Synthetic browse-level vindex builder shared by the binding tests.

Writes the minimal file set `larql.load` needs — index.json, gate vectors,
embeddings, down metadata and a character-level tokenizer — so tests run
anywhere without model files.
"""

import json
import os
import struct

import numpy as np

DOWN_META_MAGIC = 0x444D4554
DOWN_META_VERSION = 1
DOWN_META_TOP_K = 3
F32_BYTES = 4
TOKENIZER_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?'-_"


def _write_f32(path, data):
    """Write a flat f32 array to a binary file."""
    np.asarray(data, dtype=np.float32).tofile(str(path))


def _gate_vectors(num_layers, hidden_size, num_features):
    """Each gate vector is a simple pattern so KNN results are predictable."""
    gates = np.zeros((num_layers, num_features, hidden_size), dtype=np.float32)
    for layer in range(num_layers):
        for feat in range(num_features):
            gates[layer, feat, feat % hidden_size] = 1.0 + layer * 0.1
            gates[layer, feat, (feat + 1) % hidden_size] = 0.5
    return gates


def _down_meta(num_layers, num_features, vocab_size, free_per_layer):
    """Binary down_meta: file header, then per layer a feature count and one
    record per feature. The last `free_per_layer` records of each layer are
    empty, leaving free slots for INSERT."""
    record_size = 8 + DOWN_META_TOP_K * 8
    meta = bytearray(
        struct.pack("<IIII", DOWN_META_MAGIC, DOWN_META_VERSION, num_layers, DOWN_META_TOP_K)
    )
    for layer in range(num_layers):
        meta.extend(struct.pack("<I", num_features))
        for feat in range(num_features):
            if feat >= num_features - free_per_layer:
                meta.extend(b"\x00" * record_size)
                continue
            token_id = (layer * num_features + feat) % vocab_size
            c_score = 0.5 + feat * 0.01
            record = struct.pack("<If", token_id, c_score)
            for k in range(DOWN_META_TOP_K):
                record += struct.pack("<If", (token_id + k + 1) % vocab_size, c_score - k * 0.1)
            meta.extend(record)
    return bytes(meta)


def _tokenizer(vocab_size):
    """Character-level BPE tokenizer so any input tokenizes."""
    vocab = {c: i for i, c in enumerate(TOKENIZER_CHARS)}
    for idx in range(len(vocab), vocab_size):
        vocab[f"<t{idx}>"] = idx
    return {
        "version": "1.0",
        "model": {"type": "BPE", "vocab": vocab, "merges": []},
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": None,
        "decoder": None,
    }


def build_synthetic_vindex(
    root,
    *,
    num_layers,
    hidden_size,
    intermediate_size,
    vocab_size,
    num_features,
    embed_scale=1.0,
    free_per_layer=4,
):
    """Write a synthetic browse-level vindex into the existing directory `root`."""
    layer_bytes = num_features * hidden_size * F32_BYTES
    config = {
        "version": 1,
        "model": f"test/synthetic-{num_layers}l",
        "family": "test",
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "vocab_size": vocab_size,
        "embed_scale": embed_scale,
        "extract_level": "browse",
        "dtype": "f32",
        "down_top_k": DOWN_META_TOP_K,
        "has_model_weights": False,
        "layers": [
            {
                "layer": layer,
                "num_features": num_features,
                "offset": layer * layer_bytes,
                "length": layer_bytes,
            }
            for layer in range(num_layers)
        ],
        "layer_bands": {
            "syntax": [0, 1],
            "knowledge": [2, 3],
            "output": [3, 3],
        },
    }
    with open(os.path.join(root, "index.json"), "w") as f:
        json.dump(config, f)

    _write_f32(
        os.path.join(root, "gate_vectors.bin"),
        _gate_vectors(num_layers, hidden_size, num_features),
    )

    embeddings = np.zeros((vocab_size, hidden_size), dtype=np.float32)
    embeddings[np.arange(vocab_size), np.arange(vocab_size) % hidden_size] = 1.0
    _write_f32(os.path.join(root, "embeddings.bin"), embeddings)

    with open(os.path.join(root, "down_meta.bin"), "wb") as f:
        f.write(_down_meta(num_layers, num_features, vocab_size, free_per_layer))

    with open(os.path.join(root, "tokenizer.json"), "w") as f:
        json.dump(_tokenizer(vocab_size), f)
    return root
