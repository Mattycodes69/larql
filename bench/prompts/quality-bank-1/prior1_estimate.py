# Copied from mlx_lm 0.29.1, mlx_lm/quant/dynamic_quant.py,
# `estimate_sensitivities` (Copyright © 2025 Apple Inc., MIT).
#
# AUTO-REP-PRIOR-1 (docs/auto-rep-prior-1.md) freezes one change, and
# `prior1_estimate.upstream.diff` is the whole of it:
#
#   - W_low is injected (LARQL's NVFP4 reconstruction) instead of qdq(low_bits)
#   - W_high is the source weight instead of qdq(high_bits)
#   - only the injected layers are lowered; every other leaf stays at source,
#     so the gradient is taken at AUTO-REP's base state (uniform NVFP4 over
#     the decoder projections, everything else source)
#
# Loss, gradient, accumulation and normalisation are untouched. Do not edit
# this function without regenerating the diff and re-reading the plan.

import copy

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map, tree_unflatten
from tqdm import tqdm

from mlx_lm.tuner.losses import kl_div_loss
from mlx_lm.tuner.trainer import grad_checkpoint


def estimate_sensitivities(
    model,
    data,
    low_weights,
    batch_size: int = 4,
    gradient_accum_dtype: mx.Dtype = mx.float32,
    gradient_checkpoint: bool = False,
):
    layers = tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module)
    layers = {k: l for k, l in layers if hasattr(l, "to_quantized") and k in low_weights}
    q_model = copy.deepcopy(model)
    q_layers = copy.deepcopy(layers)
    for k, l in q_layers.items():
        l.weight = low_weights[k]
        # Freeze everything but the quantizable weight
        l.freeze()
        l.unfreeze(keys=["weight"])
    q_model.freeze()
    q_model.update_modules(tree_unflatten(list(q_layers.items())))

    def loss_fn(batch, targets):
        return kl_div_loss(q_model(batch), targets).mean()

    if gradient_checkpoint:
        grad_checkpoint(q_model.layers[0])

    grad_accum = tree_map(
        lambda x: mx.zeros(x.shape, dtype=gradient_accum_dtype),
        q_model.trainable_parameters(),
    )
    for e, s in tqdm(
        enumerate(range(0, len(data), batch_size)),
        total=len(data) // batch_size,
        desc="Estimating sensitivities",
    ):
        batch = data[s : s + batch_size]
        targets = model(batch)
        mx.eval(targets)
        _, grads = nn.value_and_grad(q_model, loss_fn)(batch, targets)
        grad_accum = tree_map(lambda x, y: x + y, grad_accum, grads)
        del grads
        mx.eval(grad_accum)

    def compute_sensitivity(gradient, low_q_weight, original_weight):
        n_batches = (len(data) + batch_size - 1) // batch_size
        gradient = gradient / n_batches
        high_q_weight = original_weight
        param_size = original_weight.size / 1e6
        alignment = (gradient * (low_q_weight - high_q_weight)).sum()
        return alignment / param_size

    sensitivities = tree_map(
        compute_sensitivity,
        grad_accum,
        q_model.parameters(),
        model.parameters(),
    )
    mx.eval(sensitivities)

    sensitivities = [(k[:-7], s.item()) for k, s in tree_flatten(sensitivities)]

    return sensitivities
