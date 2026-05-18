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
