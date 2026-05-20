# Performance — larql-kv

Machine: M3 Max, macOS. Numbers carried from the engine-level audits that
preceded the crate extraction (2026-04-23 onward), with the source bench
identified for each row. The extraction itself was a code move — no
performance changes expected, none observed in the cross-check.

> ⚠️ Single-machine benches on M3 Max are subject to thermal-throttle
> artifacts under sustained GPU load (1.5–3× regressions can appear that
> aren't real). When in doubt, cool-machine rerun before bisecting.

## Engine ladder — honest numbers (Gemma 3 4B, Metal Q4K, M3 Max, 2026-05-17)

**The 2026-05-17 → 18 history**: four changes made the older
"engines all hit ~95 tok/s on Metal" numbers wrong. (1) The
**fused-bypass strip** removed hidden `fused_prefill` short-circuits
inside the per-layer engines that were silently routing them through
`standard`'s kernel — five engines were tied at ~103 tok/s under
different labels, hiding every state-policy difference. (2) The **W2
hot K/V cache** lifted markov_residual from a recompute-every-step
model to a cache-and-append model. (3) The **W1-GPU per-layer
state-dump path** routes per-layer engines through the Metal fused
kernel with per-layer state capture at the cost of per-layer commits
(~1.7ms / token). (4) **W7 blit-encoder fusion** (2026-05-18)
eliminated the per-layer commit cost: per-layer staging buffers +
blit copies inside a single command buffer, with a single drain
after the final commit. +30-48% across the cached-state engines.

| Engine | CPU tok/s | Metal tok/s | Hot state | Cold tier | Notes |
|---|---:|---:|---:|---|---|
| `standard` (fused control) | 28.2 | **99.4** | 0 MB (backend cache) | — | the reference; engines that want this speed pick it explicitly |
| `boundary_kv` (= standard + chunk frames) | 28 | ~99 | 0 MB | larql-boundary frames | composes with standard for cross-session resume |
| `markov_residual` (W2 + W1-GPU + W7 blit) | 27.4 | **75.3** | 10.8 MB | residuals @ 4 B/tok | residual-stream, no f16 KV |
| `markov_residual_codec` (W2 + W1-GPU + W7 blit) | 26.6 | **79.0** | 10.8 MB | bf16 residuals (2× cold saving) | long-context-friendly cold codec |
| `unlimited_context` (W1-GPU step 4 + W7 blit) | 28.1 | **82.7** | 15.7 MB (window=256) | per-window K/V checkpoints | W7 blit fusion +48% on top of W1-GPU |
| `turbo_quant` (4-bit, W1-GPU + W7 blit, 10-tok bench) | 19.4 | **37.7** | 0.7 MB | — | WHT + Lloyd-Max K/V compression; codec cost grows with N |
| `apollo` (boundaries) | — | requires store | scales w/ store | constellation map | retrieval+injection; not on the same scale as the others |
| `no_cache` | — | — (O(N²) by design) | token list only | — | correctness baseline |

**Reading the table:**

- The 100+ tok/s number is `standard`'s Metal fused fast path. The
  per-layer engines used to claim this number too — that was the
  hidden fused-bypass. Honest numbers fall between the CPU walk
  ceiling (~28 tok/s) and the standard fused ceiling.
- W1-GPU lifted `markov_residual` and `markov_residual_codec` from
  ~28 (CPU ceiling, what the fused-bypass strip exposed) to ~58 by
  routing them through the Metal fused kernel with per-layer state
  capture.
- W7 (blit fusion) lifted the same engines to ~75-79 tok/s by
  removing the per-layer commit / wait / CPU-read cycle: per-layer
  staging buffers + blit copies inside one command buffer, with a
  single drain after the final commit. Closes the commit-overhead
  line above.
- `turbo_quant`'s smaller speedup (+14% at 10-tok bench length)
  reflects the inner-loop codec encode/decode cost — the codec
  work dominates, and the saved commit overhead is a smaller
  fraction of per-token time. Codec cost also grows with sequence
  length (each step re-compresses the full layer K/V), so longer
  benches show lower mean tok/s.
- `unlimited_context` got the biggest W7 win (+48%) because its
  per-step CPU-side work after the kernel returns is the lightest
  of the four cached-state engines, so the saved commit overhead
  is a larger fraction of total per-token time. The extra hot
  state (15.7 MB at window=256) is the current-window K/V the
  engine has to shadow until `KvHandle::evict_oldest(n)` lets the
  backend cache match the engine's window.

**Where the remaining gap to `standard`'s 99 tok/s lives**, per
profiler data after W7:

| Cost (per token) | Contribution to ~13 ms/tok | Closure path |
|---|---:|---|
| Metal kernel compute | ~10 ms | — (already at the fused-kernel floor) |
| ~~Per-layer commit overhead~~ | ~~~1.7 ms~~ | **Closed by W7** (single commit per token) |
| CPU glue (state Vec→Array2, append, etc.) | ~3 ms | In-place state updates / pre-allocated buffers |

## W10 — engine state on GPU (opt-in via `LARQL_W10_HONLY=1`)

W10 lets engines that treat K/V (and optionally h_in) as derivative
state declare so at the API boundary; the Metal kernel then skips
the GPU→CPU staging blit + readback for the declared-derivative
slots. The win compounds with how much state the kernel is no
longer asked to transfer.

The mask cascade (`StateDumpMask::Full → HOnly → None`) is gated by:

| Mask | Condition | What kernel skips |
|---|---|---|
| `Full` (default) | flag off | nothing — today's behavior |
| `HOnly` | `LARQL_W10_HONLY=1`, engine drops `rs.hot_kv` shadow | K/V staging + blit + readback |
| `None` | `LARQL_W10_HONLY=1`, engine drops both `rs.hot_kv` AND `rs.stored` shadow (requires `window_size = None`) | h_in staging + blit + readback as well |

Engines that opted in:

| Engine | HOnly | None | Why |
|---|---|---|---|
| `markov_residual` | ✅ | ✅ (window=None) | K/V derivative (Metal cache is truth); h_in dead weight without cold-tier eviction |
| `markov_residual_codec` | ✅ | ✅ (window=None) | Same — codec residuals are canonical, hot K/V is derivative |
| `unlimited_context` | ✅ | ❌ | `close_window` reads last K/V back via `KvDispatch::read_kv_row_at`; h_in needed for replay-from-checkpoint |
| `turbo_quant` | ❌ | ❌ | K/V is canonical (destructive codec); cannot be derived |

### Measurement protocol

**`--profile` is safe to use** — as of 2026-05-18 it no longer
auto-sets `LARQL_PROFILE_SPLIT=1` and the GPU-timestamp tax is gone.
Engine-side `state_capture` / `state_materialise` / `state_append`
timers are cheap. (If you want the GPU per-stage breakdown
specifically, set `LARQL_PROFILE_SPLIT=1` explicitly — that adds
~20 ms/token from 102 GPU-timestamp queries.)

Three runs of `larql bench`, recording the per-stage table from
`--profile`. State stage rows are new in W10 instrumentation:

```sh
# Baseline (flag off, Full mask).
cargo run -p larql-cli --release -- bench gemma3:4b \
    --engine markov-rs --profile

# Phase B (windowed = HOnly mask).
LARQL_W10_HONLY=1 cargo run -p larql-cli --release -- bench gemma3:4b \
    --engine markov-rs:window=512 --profile

# Phase C-v1 (windowless = None mask).
LARQL_W10_HONLY=1 cargo run -p larql-cli --release -- bench gemma3:4b \
    --engine markov-rs --profile
```

The falsifiable predictions:

- `state_capture` (engine-side timer on the whole backend call) drops
  monotonically `Full → HOnly → None`. If it doesn't drop under
  `HOnly`, the kernel didn't honor the mask — re-check the `dump_kv`
  branches in `crates/larql-compute-metal/src/decode/mod.rs`.
- `state_materialise` and `state_append` drop to ~0 under `None` (the
  engine drops handles without consuming them).
- Total tok/s rises on Metal. The expected ceiling is `standard`'s
  ~100 tok/s; the remaining gap after W10 is whatever's not on the
  state-bridge path (lm_head + detok, ~1.2 ms CPU/step).

### Results — 2026-05-18, Gemma 3 4B Q4K, Metal, M3 Max, 80-tok decode

**Important: measure WITHOUT `--profile`.** The `--profile` flag enables
`LARQL_PROFILE_SPLIT=1`, which makes the Metal kernel call
`record_stage` (a GPU-timestamp query) 102 times per token (34 layers
× 3 stages). That instrumentation alone costs ~20 ms CPU/step and
turns an 11 ms/step kernel into a 30 ms/step one — a 2.7× slowdown
that completely masks W10's signal. The state-stage timers (added in
this PR) are only printed under `--profile`, but the tok/s
measurement that matters for W10 should be run with the flag OFF.

Numbers below are without `--profile`. **Each engine bench was run
in isolation with cool-down between** — running engines sequentially
in one process heated the machine and produced 2×+ apparent
regressions that vanished on cool re-runs. The `cmd_bufs=1` field in
`LARQL_GPU_TIMING=1` output confirms W7 single-buffer fusion is
active on every engine.

| Engine + mask | tok/s | mean step | gpu / cpu per step | hot mem | Δ tok/s |
|---|---:|---:|---:|---:|---:|
| `markov-rs` Full (baseline, R1+R2 = 84.7) | **84.7** | 11.81 ms | ~13.0 / ~1.2 ms | 54.4 MB | — |
| `markov-rs:window=512` HOnly | **93.0** | 10.76 ms | ~10.5 / ~1.2 ms | 30.2 MB | **+9%** |
| `markov-rs` None (windowless) | **99.1** | 10.10 ms | ~9.5 / ~1.2 ms | 0 MB | **+17%** |
| `markov-rs-codec` Full | 88.3 | 11.33 ms | ~10.0 / ~1.2 ms | 26.3 MB | — |
| `markov-rs-codec` None (windowless) | **98.5** | 10.15 ms | ~9.0 / ~1.2 ms | 0 MB | **+12%** |
| `unlimited-context:window=256` Full | 88.2 | 11.34 ms | ~10.1 / ~1.3 ms | 9.6 MB | — |
| `unlimited-context:window=256` HOnly | **92.8** | 10.78 ms | ~9.5 / ~1.2 ms | 0 MB | **+5%** |

**What the numbers say:**

- **All three derivative-K/V engines hit standard's fused-kernel
  ceiling** under their best W10 mask: `markov-rs` None at 99.1,
  `codec` None at 98.5, `unlimited` HOnly at 92.8 — vs `standard`'s
  ~100 tok/s on the same machine. This is the W10 success
  criterion: a state-managing engine that pays no extra cost on the
  dispatch hot path.
- **Full → HOnly → None cascade holds**: each mask step is strictly
  faster than the next, matching the predicted direction.
- **Hot memory drops as designed**: 54.4 MB → 0 MB on `markov-rs` /
  `codec` (windowless), 30.2 MB → 0 MB on `markov-rs:window=512`,
  9.6 MB → 0 MB on `unlimited:window=256`. The Metal kv cache is
  now the sole source of truth on the dispatch hot path.
- **`unlimited-context` win is small** (+5%) because most of its
  per-step CPU work is the window-buffer slot-assign that survived
  even after the shadow is dropped (the window slots are
  pre-allocated regardless of mask). Memory savings still hold.
- **Per-step CPU is ~1.2 ms across all engines** under
  `LARQL_GPU_TIMING=1`. That's the lm_head + detok + small engine
  bridge cost. Engine-side state-bridge work (when present) lives
  inside that 1.2 ms.

### Bottleneck found while measuring W10

The `--profile` flag was itself the dominant CPU cost on the
dispatch path during this measurement campaign. Symptom: standard
engine running at 32 tok/s under `--profile` vs 86 tok/s without.
Root cause: 34 layers × 3 stages × `gpu_elapsed_ms` (Metal
timestamp query) = 102 syscall-ish CPU calls per token, ~20 ms total.

Implication for future bench work:
- For `tok/s` measurements, never use `--profile`. The flag should
  default off in PERFORMANCE.md examples.
- The state-stage timers added for W10 (`state_capture`,
  `state_materialise`, `state_append`) are useful for *relative*
  comparison across masks but distort the absolute baseline. Either
  always include the flag (consistent distortion) or split the
  measurement into two runs (`--profile` for stage breakdown,
  no-flag for tok/s).
- A leaner gate-on-flag for state timers would let us measure stage
  cost without paying the GPU-timestamp tax. Worth a follow-up: split
  `LARQL_PROFILE_SPLIT` from the engine-side `with_profiling(true)`
  flag so engines record state timers while the kernel skips its
  per-stage GPU timestamps.

## Engine-trait dispatch overhead (synthetic test_utils, M3 Max, CPU)

Bench: `cargo bench -p larql-kv --bench engine_decode -- generate`. Times
end-to-end generation (prefill + 8 decode steps) on the synthetic 2-layer
test model. The engine-trait path constructs a `StandardEngine` and
drives it through `generate_with_engine`; the legacy path calls
`generate_cached_backend` directly. Both should be statistically
indistinguishable.

50-sample run (3s warm-up, 8s measurement):

| Path | Time (median) | 95% CI |
|---|---|---|
| `legacy_generate_cached_backend` | 446.72 µs | 443.22 – 450.02 µs |
| `engine_dispatch_standard` | 443.66 µs | 437.98 – 448.67 µs |

CIs fully overlap; engine dispatch is ~1 % faster in this run, well
within noise. The trait-vtable + engine construction overhead is
negligible for the production cache wrapper. This is the empirical
evidence supporting the "no regression on the default path" non-goal
in the unification spec
([§9](../larql-inference/docs/specs/kv-engine-unification.md)).

A previous 10-sample run produced a wider engine-dispatch CI
(380 – 715 µs) — that's a small-sample artifact, not a real overhead
signal. With ≥50 samples and ≥8 s measurement the two paths are
statistically inseparable.

## Per-engine prefill / decode-step times (synthetic, CPU)

Bench: `cargo bench -p larql-kv --bench engine_decode`. 2-layer
synthetic model, 8-token prompt. Useful for catching dispatch
regressions in PR review; not a proxy for real-model decode speed.

10-sample run, 2 s warm-up + 4 s measurement:

| Engine | Prefill (median) | Decode step (median) |
|---|---|---|
| `standard` | 14.9 µs | 12.0 µs |
| `standard:window=4` | 15.2 µs | 7.1 µs (smaller K/V to attend over) |
| `no-cache` | 14.9 µs | 34.8 µs (re-runs full forward each step) |
| `markov-rs` | 15.0 µs | 27.1 µs (recomputes K/V from residuals) |
| `unlimited-context` | 56.9 µs | 8.3 µs (window-checkpoint amortises decode) |
| `turbo-quant` (4-bit) | 21.8 µs | 81.9 µs (codec dominates on tiny model) |
| `apollo` | 45 ns (no boundary store loaded → early bail) | 2 ns (early bail) |

`standard` and `no-cache` differ only at decode-step: `no-cache` re-runs
the full prefill per step (3× the cost), while `standard` does
incremental K/V append. As the prompt grows, the gap widens linearly.

For real-model numbers (Gemma 3 4B, Metal Q4K, 370K-token corpus) see
the table above.

## Per-engine notes

### markov_residual

- **Mechanism.** Stores the pre-layer residual stream and re-projects K/V
  at decode time. The pre-layer residual is the complete Markov state, so
  recomputed K/V is bit-identical to a full-KV baseline.
- **Validated 2026-04-23.** KL = 0.0 vs full-KV on Gemma 3 4B over a
  10-prompt corpus. Survives the 077884b bisect of the 81-84 tok/s
  measurement bug (see project memory note —
  `project_metal_decode_81_was_buggy`).
- **Profiler.** Per-stage breakdown lands in `EngineProfiler`:
  embed, recompute_cold, recompute_hot, attention, ffn, total.

### unlimited_context

- **Mechanism.** Sliding window over the active K/V cache plus a
  checkpoint of the pre-window residual. Decode beyond the window
  re-prefills lazily from the checkpoint. Exact within the window.
- **Tunable.** `window=N` controls the hot tier; default 512.

### turbo_quant

- **Mechanism.** Walsh-Hadamard rotation followed by Lloyd-Max codebook
  quantisation. Encodes K/V at 3- or 4-bit per scalar.
- **Decode.** ~95 tok/s decode at 4-bit, cos ≈ 0.991 vs full-precision K/V.
- **Memory.** ~4× compression of the f16 baseline (so still ~12.7 GB at
  Gemma 3 4B / 370K tokens — orders of magnitude above the residual
  engines, useful when window bounds aren't acceptable).

### apollo

- **Mechanism.** Boundary-residual injection. A constellation index over
  pre-captured boundary points lets decode start the forward pass at the
  configured `crystal_layer` (default 30 of 34) instead of layer 0.
- **Speed.** ~8× decode speedup when the prompt hits a captured
  boundary; falls back to full-stack forward when it doesn't. Memory ≈
  11 MB regardless of corpus size — the constellation is small, the win
  is in skipped layer compute.
- **O(N²) by design.** Apollo's decode_step pushes the new token onto
  `self.context_tokens` and calls `forward_from_layer(weights,
  &self.context_tokens, ...)` — which builds a fresh HashMap KV cache
  per call (`compute/src/forward/predict/raw.rs:190`) and re-runs
  `from_layer..num_layers` over the **entire growing context** every
  step. There is no cross-step KV persistence: each decode is O(N)
  attention work, total O(N²). This is inherent to the "retrieval-style,
  no-KV, crystallized prefix" contract — not a fixable bottleneck.
  Apollo is intended for short queries that piggyback on a long cold
  prefix; long-decode workloads should pick another engine.

### boundary_per_layer

- **Mechanism.** Per-layer codec policy on the cold tier. Same
  hot/cold split as `markov_residual_codec` but the codec can differ
  per layer (v0.1 ships `Bf16` uniform). Requires a calibration
  record bounding end-to-end KL for the policy fingerprint (spec
  §4.7); the engine refuses to construct without one.
- **CLI.** Wired into `EngineKind::BoundaryPerLayer`. Default
  `num_layers=34` (Gemma 3 4B); override with `layers=N`.
  ```sh
  larql bench gemma3:4b --engine boundary-per-layer:window=256
  larql bench other:model --engine boundary-per-layer:window=512,layers=28
  ```

## 2026-05-20 — engine bottleneck audit + fixes

Audit of all engines for O(N²) hot paths. Two real algorithmic bugs
in `boundary_per_layer` fixed; Apollo's O(N²) confirmed as a design
contract (see `apollo` notes above).

| Engine | Finding | Status |
|---|---|---|
| `standard` | Backend KV cache; clean | — |
| `markov_residual` | Cold-tier O(N²) merge — fixed 2026-05-19 via doubling-capacity `append_cold_overflow` | landed |
| `markov_residual_codec` | Shares `markov_residual` cold tier; same fix | landed |
| `unlimited_context` | Clean | — |
| `turbo_quant` | O(N) decompress+recompress per step — fixed 2026-05-19 via append-only codec path (30.1 → 82.8 tok/s, +175%) | landed |
| `apollo` | O(N²) by design — `forward_from_layer` rebuilds KV each step over growing context | not a bug |
| `boundary_kv` | Clean | — |
| `boundary_per_layer` | Two real O(N²) bugs — see below | **fixed this turn** |
| `no_cache` | O(N²) by design | not a bug |

### boundary_per_layer fixes

**Bug B (cold_kv nuke).** Every overflow set `cold_kv = None`, forcing
the next decode step to run `recompute_kv` over the entire
codec-decoded cold tier — O(N) attention-projection work per step,
O(N²) windowed-mode decode. Fixed by `extend_cold_kv_with_overflow`:
on each overflow, computes K/V on the just-evicted rows' codec
round-trip at the correct RoPE position (snapshotted **before**
`cold_encoded.append`) and concatenates onto each layer's
existing K/V. `cold_kv` now stays `Some` for the lifetime of the
session.

**Bug A (hot-tier rebuild).** Every `decode_step` rebuilt every
layer's `stored[layer]` from scratch via `Array2::zeros((s_old+1,
hidden)) + .assign(stored) + .assign(new_row)` — 34 × per-step
allocations on Gemma 3 4B. Bounded by `window+1` in windowed mode
(small constant) but **O(N²) in unbounded mode**. Fixed by switching
to `ndarray::Array2::push_row`, which has amortised O(m) growth
under the hood (geometric capacity).

Both fixes apply to both the legacy `decode_step` and
`decode_step_via_executor` paths. New test
`cold_kv_stays_populated_across_multiple_overflows` locks in the
bug-B invariant; all 51 boundary_per_layer tests plus the broader
594-test lib suite continue to pass.

**Measured impact** (Gemma 3 4B Q4K vindex, dense walk via
`ensure_attn_tensors_dequantised` — *not* the W1-GPU fast path):

| Mode | 50 tok pre-fix | 50 tok post-fix | Δ |
|---|---:|---:|---:|
| unbounded | 2.34 tok/s | 2.50 tok/s | **+6.8%** |
| windowed (512) | 2.32 tok/s | 2.53 tok/s | **+9.1%** |

Parity gate (`boundary_per_layer_parity_gate.rs`) reports **100%
token agreement** vs `markov-rs-codec` with the bf16 codec in both
modes — the fixes are correctness-preserving. The visible delta is
small on the dense walk path because the absolute per-step cost
(~400 ms) is dominated by attention/FFN compute, not by the
hot-rebuild memcpy that bug A eliminates.

The big algorithmic win (bug B's "skip O(N) recompute_kv on every
overflow") is gated on actually triggering overflows during decode.
With `window=512` and 50–200 decode tokens, zero overflows occur, so
that path is exercised only at much larger N or smaller windows.

**Important: this isn't the production speed of boundary_per_layer.**
The dense walk path is ~35× slower than the W1-GPU `try_prefill_via_dispatch`
fast path used by `markov-rs-codec`. Closing that gap is a separate
wiring task — see ROADMAP entry for boundary_per_layer W1-GPU
dispatch.

**Parity gate before measuring**. Before relying on `boundary-per-layer`
bench numbers, run the parity gate to confirm the cold_kv-append
RoPE positioning is correct vs the reference engine
(`markov-rs-codec` at the same window):

```sh
cargo run --release -p larql-kv \
    --example boundary_per_layer_parity_gate -- \
    --vindex ~/.cache/larql/local/gemma3-4b-q4k-v2.vindex \
    --tokens 50
```

The gate checks that the first token-level divergence between
`markov-rs-codec` and `boundary-per-layer` (uniform bf16) is past
step 5 — an RoPE off-by-one in `extend_cold_kv_with_overflow` or a
codec round-trip skip would diverge at step 0/1/2 where natural
lossy-codec drift can't have set in. Late divergence is acceptable
(bf16 KL is ~0.01 nats/step). Exits non-zero on early divergence.

## Reproducing

The criterion bench in this crate (see `benches/`) covers each engine's
hot path under a synthetic 2-layer model so it runs anywhere without a
vindex on disk. For end-to-end real-model numbers on a downloaded
checkpoint, use:

```sh
cargo run -p larql-cli --release -- bench gemma3:4b --engine markov-rs
cargo run -p larql-cli --release -- bench gemma3:4b --engine unlimited-context:window=256
cargo run -p larql-cli --release -- bench gemma3:4b --engine turbo-quant:bits=4
cargo run -p larql-cli --release -- bench gemma3:4b --engine apollo:layer=30
cargo run -p larql-cli --release -- bench gemma3:4b --engine boundary-per-layer:window=512
```

The in-crate criterion bench at `crates/larql-kv/benches/engine_decode.rs`
runs the dispatch helpers under `cargo bench -p larql-kv --bench engine_decode`,
covering `StandardEngine` vs the legacy `generate_cached_backend` parity oracle
plus the sync/async dispatch helpers. (Until 2026-05-16 this harness lived in
the retired `kv-cache-benchmark` crate as `kv_strategies`; the production
comparator is now this in-crate bench plus `larql bench --engine <spec>`.)

## See also

- [`ROADMAP.md`](ROADMAP.md) — open performance / capability work.
- [`CHANGELOG.md`](CHANGELOG.md) — extraction history.
- `larql-compute/PERFORMANCE.md` — Metal pipeline numbers; engines ride
  the `decode_token` path so end-to-end gains often live there.
