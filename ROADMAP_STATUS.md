# Roadmap Status

Canonical rollup for the next execution slice. Keep the detailed design in
`ROADMAP.md` and crate-local roadmaps; use this file to answer "what is active
now?" without rereading every crate document.

Last updated: 2026-05-28

## Recently shipped (delta since last update)

- **CPU remote-MoE decode — closes #146** (2026-05-28): `larql run --moe-shards …` without `--metal` failed with `decode_token_with_moe returned None during prefill` (CPU backend's `decode_token_with_moe` is a GPU-only trait default; CLI always called the GPU path). Routed the CPU branch through the existing `generate_kquant_cpu_remote` + added a clean attn-presence guard in `grid/setup.rs`. Verified end-to-end on the real Gemma-4-26B-A4B vindex (output "Paris"). **Caveat: full-recompute, no KV cache → 0.1–0.4 tok/s.** Follow-up tracked as the new C1 item below.
- **New item — MoE-aware KV engines (C1)** (2026-05-28): the KvEngine layer is dense-only; MoE decode bypasses it (no KV cache). Make engines MoE-aware + add a `RemoteMoeFfn` `FfnBackend` wrapper so CPU `--moe-shards` is KV-cached and engine-selectable. Primary tracking: [larql-kv ROADMAP](crates/larql-kv/ROADMAP.md) §"MoE-aware KV engines (C1)".
- **Strategic priorities + Query/Edit/Interpret track** (2026-05-28): two new framing sections in [`ROADMAP.md`](ROADMAP.md) layered on the achievability analysis. (1) Single gated critical path — **V1–V4 is the only true P0**; Engine↔Backend unification, CPU-path-to-blazing, and best-in-class mech-interp are downgraded to "P0-conditional, unblocked by V1–V4". (2) V3 (disk-resident mmap) pulled forward on information-value grounds. (3) GPU = credibility tax, D-PREFILL-MM2 first. (4) MoE-first functionality. (5) Query/Edit/Interpret (`DESCRIBE`/`INSERT`/`walk`/compile) promoted to a co-equal functionality track — the moat, lower-risk than the 100× compound. This rollup's Active Sequence + P0 boundaries below are updated to match.
- **Whole-codebase review** (2026-05-28): multi-agent deep review (17 crates, ~415K LOC; per-crate reader + adversarial verification). Clippy clean (2 trivial nits). ~7 verified high/medium hardening items tracked in [`docs/audits/codebase-review-2026-05-28.md`](docs/audits/codebase-review-2026-05-28.md), [`ROADMAP.md`](ROADMAP.md) §"Codebase hardening", and per-crate roadmaps. Top two confirmed by hand: infallible `FfnBackend::forward` aborts serving on remote-shard blips; Metal KV append has no `pos<max_seq` clamp (GPU OOB past 4096 rows).
- **Cross-engine forward-pass correctness gate** (2026-05-16): `larql shannon verify` CLI + multi-arch sweep `scripts/diagnose_models.py` + CI workflow `.github/workflows/shannon-verify.yml`. Four config-loading bugs surfaced and fixed in `larql-models` (rms_norm_eps not parsed; Gemma 3 per-layer-type rope_scaling missing; llama3 rope_scaling missing; StarCoder2 norm_epsilon alias). 7/9 archs in the sweep PASS at <0.5% bits/char vs HF F32 with no env-var overrides. See [`docs/diagnoses/shannon-cross-engine-divergence.md`](docs/diagnoses/shannon-cross-engine-divergence.md).

## Active Sequence

V1–V4 aim-validation is the single P0 gate; items below it are P0-conditional
(unblocked once the gate resolves). V3 is pulled forward to run in parallel with
V1 on information-value grounds (it is the riskiest assumption and reshapes the
most plan if it fails — see [`ROADMAP.md`](ROADMAP.md) §"Strategic priorities").

| Order | Item | Status | Owner | Exit criterion |
|---:|---|---|---|---|
| 1 | V0 aim-validation harness | started | `bench/aim-validation`, `scripts/aim_validation.py` | V1-V4 runs share one model/prompt/metric matrix and emit comparable JSON records. |
| 2 | V1 hash routing across all layers | queued | experiments + `larql-inference` | Per-layer top-k table and end-to-end divergence/tok/s report across the cross-arch matrix. |
| 3 | V3 disk-resident mmap spike (pulled forward) | queued | experiments + `larql-vindex` | Throwaway spike: ≥70B-class MoE vindex on NVMe; page-fault rate + tok/s under MoE routing locality on a single decode stream. Result either keeps the >RAM tier alive or shrinks the aim to "fits in RAM" — resolves KU5 before the backend rewrite. |
| 4 | V2 FP4 generality | queued | experiments + `larql-vindex`/`larql-compute` | FP4-friendliness report by architecture/layer with QAT-required thresholds flagged. |
| 5 | C10 CPU baseline bench | started | `larql-cli`, `bench/` | `larql bench --cpu --output json` works and quant-matched llama.cpp CPU baseline is recorded; next exit is KV-cached CPU Q4K decode plus cross-arch repeats. |
| 6 | MI4/T7 trace truthfulness gate | queued | `larql-inference` | TRACE final residual/logit parity pinned for WalkFfn and patched-vindex paths, then Q4K/MoE. (Also the verify/reversible backbone for the Query/Edit/Interpret track.) |
| 7 | R6 depth-fraction probe API | queued | `larql-inference`, `larql-models` | Stable probe API available before MTP3 layer-choice validation. |
| 8 | MTP1-MTP2 | queued | `larql-models`, `larql-vindex`, `larql-inference` | Gemma 4 assistant drafter loads; verify-loop decode exists before activation-feedback work. |

## Current P0/P1 Boundaries

| Area | Decision |
|---|---|
| Single P0 | Only V1–V4 aim-validation is true P0. Engine↔Backend unification, CPU-path-to-blazing, and best-in-class mech-interp are **P0-conditional** — unblocked by V1–V4, not concurrent with it. |
| Highest leverage | Run V1-V4 aim-validation before expanding long-term CPU/MoE engineering. V3 (disk-residency) runs in parallel with V1 — highest information value. |
| GPU credibility | Credibility tax, not parity. **D-PREFILL-MM2 first** — the 14× prefill gap is the only GPU item that invalidates published claims today. D-ATTN-MTG / D-METAL-PLE stay load-bearing but behind it; MTP1–6 is baseline-*matching* (don't innovate there). |
| MoE-first | Functionality emphasis stays MoE (80% tier) over dense (15%): CPU MoE forward + hash-routed FFN + disk-resident expert paging are the crown jewels. Dense-31B substrate-primary (ADR-019) is for velocity, not the destination. |
| Differentiated functionality | Query/Edit/Interpret (`DESCRIBE`/`INSERT`/`walk`/compile) is a co-equal track — the moat, lower-risk than the compound. Funded alongside aim-validation; harden the `experiments/` surface into tested LQL verbs. |
| CPU credibility | C10 comes first because the CPU track cannot enforce the 10% threshold without measurement. |
| Multi-machine MoE | Stays P2 unless a specific experiment or frontier-scale release re-promotes it. |
| Production-engine features | Continuous batching, PagedAttention, broad OpenAI surface, MCP, and thinking toggles stay deferred unless an experiment needs them. |

## Drift Checks

- If a crate roadmap says an item shipped, but this rollup still says queued,
  update this file in the same change.
- If a benchmark number changes in `README.md`, record whether it updates a
  baseline JSON, a roadmap claim, or both.
- If V1, V2, V3, or V4 fails, update the achievability table in `ROADMAP.md`
  before starting dependent engineering.
- Keep this rollup's Active Sequence + P0 boundaries consistent with the
  "Strategic priorities" and "Query / Edit / Interpret" sections in `ROADMAP.md`
  (review 2026-05-28); if one moves, move the other in the same change.
