# Pre-release review — 2026-09-26

A whole-workspace review against the release bar, which is now written down in
[`AGENTS.md` § Code standards](../../AGENTS.md#code-standards):

- decoupled, modular code;
- no file over 800 lines;
- a clean file structure;
- no hardcoding to a module or to an architecture;
- a general code review.

## Method

- **Mechanical sweep first:** crate sizes, oversized files, family-name literals, `/Users/` paths and tests living under `src/`.
- **Seven read-only readers**, one per crate cluster. Each reader's full findings are in
  [`release-review-2026-09-26/`](release-review-2026-09-26/), with `path:line`, a severity (P0/P1/P2) and a one-line fix for each item.
- **Verification:** every P0 below, and the P1 correctness bugs marked **(verified)**, was re-read at source by the coordinator. Everything else is as the readers reported it. No tests or builds were run.

| Detailed report | Scope |
|---|---|
| [vindex3.md](release-review-2026-09-26/vindex3.md) | `larql-vindex/src/format/vindex3/` |
| [vindex_rest.md](release-review-2026-09-26/vindex_rest.md) | rest of `larql-vindex`, `larql-vindex-spec`, `vindex-cli` |
| [models.md](release-review-2026-09-26/models.md) | `larql-models`, `-factory`, `-core`, `-boundary`, `-execution`, `-experts`, `model-compute` |
| [inference.md](release-review-2026-09-26/inference.md) | `larql-inference`, `larql-router`, `larql-router-protocol` |
| [compute.md](release-review-2026-09-26/compute.md) | `larql-compute`, `larql-compute-metal` |
| [kv_lql.md](release-review-2026-09-26/kv_lql.md) | `larql-kv`, `larql-lql`, `larql-python` |
| [cli_server.md](release-review-2026-09-26/cli_server.md) | `larql-cli`, `larql-server`, `larql-demos`, `larql-continuation-fixture` |

## Scale

- About 1.0M lines of Rust in 22 crates. `larql-vindex` alone is 365k, and about 55% of its `format/vindex3/` directory is test code under `src/`.
- **190 files exceed 800 lines**: 137 source files and 53 test files. By crate:

| Crate | Files over 800 lines |
|---|---|
| vindex | 60 |
| compute-metal | 23 |
| cli | 21 |
| inference | 20 |
| kv | 17 |
| models | 13 |
| server | 11 |
| lql | 10 |
| compute | 6 |
| others | 9 |

- **Most oversized source files come under the cap once their inline tests move out.** Roughly 40 need a genuine split. The detailed reports give a concrete split for each.
- **506 model-family string literals appear outside the `architectures/`, `detect/`, test and fixture code.** Most are in doc comments or test-only modules. The real hardcoding is listed under P1-C below.
- **About 650 test files sit under `src/`**, 452 of them in `larql-vindex`.

## Verdict

The numeric core and the older serving path are in good shape:

- the `ComputeBackend` + `backend_select` path never downcasts to Metal;
- kernels are bound by type, not looked up by string;
- the LQL parser has no panicking paths;
- the wire decoders guard against oversized allocations;
- the `ModelArchitecture` trait-with-defaults design is right.

What isn't release-ready:

1. **Input-hardening gaps.** The server, the GGUF reader, the storage layer and the Python bindings can each be crashed, or made to hit undefined behaviour, by crafted input.
2. **The VINDEX3 execution stack grew up inside the storage crate without a backend abstraction.** It currently has a Kimi-specific parallel path and concrete Metal calls.
3. **Architecture facts leak into generic code in about a dozen places.** Three of them are live wrong-answer bugs.

---

## P0 — release blockers

| # | Issue | Where | Fix |
|---|---|---|---|
| 1 | **The server can be crashed with one request (verified).** `max_tokens` is taken from the request body with no cap and flows into `Vec::with_capacity`. A huge value aborts the process on allocation failure. A merely large one keeps the exclusive weights lock after the request times out, because `spawn_blocking` doesn't cancel. | `larql-server/src/routes/openai/completions.rs:216`, `chat/handler.rs:140`, `v3_completions.rs`, `responses/handler.rs` → `larql-inference` generate loops | Clamp to a `MAX_TOKENS` const (and the context length) at the route boundary, and add a cancellation flag to the blocking loop |
| 2 | **Arbitrary local file read (verified).** `POST /v1/patches {"url": …}` treats any string that isn't `hf://` as a local path and loads it. Internal error text is echoed back to the client, so it also reveals which files exist. | `larql-server/src/routes/patches.rs:40-58`, `error.rs:97` | Accept only `hf://` URLs, or paths under a configured patch root after canonicalisation; stop echoing internal errors |
| 3 | **Open by default.** The server binds to `0.0.0.0`, auth is off, gRPC ignores `--api-key`, and CORS is permissive. | `larql-server/src/bootstrap/cli.rs:16`, `bootstrap/mod.rs:567` | Default to `127.0.0.1`; apply auth, rate limiting and concurrency limits to gRPC too |
| 4 | **A crafted GGUF file aborts the process (verified).** `vec![0u8; len]` and `Vec::with_capacity(n)` take their sizes from u64 values in the file with no bound. Reachable from both the server and the CLI. | `larql-models/src/loading/gguf/reader.rs:62-64,86`, `parser.rs:245-263` | Bound every length by the bytes remaining; use `try_reserve` |
| 5 | **Unaligned `&[f32]` from storage bytes (verified).** About 10 sites cast `&[u8]` at offsets taken from `index.json` with no alignment check. Owned, non-mmap storage has no alignment guarantee either. This is undefined behaviour. | `larql-vindex/src/index/storage/ffn_store/interleaved.rs:52`, `gate_store.rs:308,392`, `attn.rs:73`, `gate_knn/scores_batch.rs:97` | One checked helper modelled on `runtime/tensor.rs:145` (`align_to` + refuse), with checked `start+len` arithmetic |
| 6 | **Undefined behaviour in the Python bindings (verified).** (a) `Vec::from_raw_parts` is called on memory-mapped data. (b) The embeddings path reads out of bounds on a truncated file, with no length check. (c) `PyResidualTrace` holds raw pointers into its parent model, so deleting the model makes it a use-after-free. (d) Unaligned `PyBytes` are cast to `f32`. | `larql-python/src/walk.rs:170,212,237,394,444,488`, `trace_py.rs:14-27` | Use `ArrayView` borrowed from an owner kept alive by `Py<…>`, check lengths, and hold `Py<PyWalkModel>` in the trace |
| 7 | **Metal tests pass without running.** 51 `else { return }` sites across 12 test files. `MetalBackend::new()` returns `None` on a *shader compile failure*, not just when there's no device. | `larql-compute-metal/tests/*`, `backend/mod.rs:305-311` | Return `Result<_, MetalInitError>` from construction; tests `expect` on macOS (already an AGENTS.md rule) |
| 8 | **Unaligned mmap read in the trace format.** The trace reader builds an `&[f32]` at an offset taken from the file with no alignment check. It also indexes `critical_layers[n]` with an unchecked byte from the file. | `larql-inference/src/trace/context.rs:201` | Same checked helper as #5, plus a bounds check |

## P1 — should fix before release

### A. Wrong answers (correctness)

- **Stop tokens are probably never matched.** Each token is decoded with `skip_special_tokens=true` and the result compared against a hardcoded list of stop strings. Special EOS tokens come back as `""`, so generation likely runs to the token limit. **Confirm with a real-tokenizer test.** Fix: compare token ids from `EosConfig`. The hardcoded list also appears in `larql-cli` and `larql-inference/generation.rs:21`. (`larql-kv/src/generation.rs:576-590,716`)
- **Wrong prompt format:** gpt-oss (which uses Harmony) and DeepSeek are wrapped in ChatML when the model has no Hugging Face template, and unknown model ids silently fall back to `Plain`. (`larql-inference/src/prompt.rs:76,94`)
- **Compile writes the wrong architecture (verified).** `COMPILE` writes `Gemma3ForCausalLM` / `gemma3_text` / tied embeddings into *every* output `config.json`, and throws away write errors. (`larql-cli/.../compile_cmd/save.rs:148-167`)
- **Patches assume Gemma's FFN width:** `10240` is hardcoded for every model. (`larql-server/src/routes/patches.rs:73`)
- **`head_dim` silently becomes 0 (verified; this file is modified on the current branch).** When `hidden_size` is undeclared it becomes `0`, and `0.is_multiple_of(heads)` is true, so `head_dim` is 0 and `hidden_size` is never flagged as missing. (`larql-vindex/src/format/vindex3/graph/surface.rs:908-918`)
- **MoE offsets can wrap (verified):** the base offset is checked against u32, but `base + half·gate_half_bytes` is then cast `as u32` and can wrap past 4 GiB into another expert's bytes. (`larql-compute-metal/src/moe_zero_copy.rs:214-218,260,316`)
- **Tensors dropped without an error:**
  - Rank ≥3 tensors (every GGUF MoE `*_exps` bank) in both loaders. (`larql-models/src/loading/gguf/loader.rs:180-189`, `safetensors/mod.rs:348`)
  - Writers skip missing tensors.
  - The MLA path skips Q/K/V if any low-rank tensor is missing. (`larql-vindex/src/format/weights/write_f32.rs:372-392`, `load/f32.rs:97-119`, `load/q4k.rs:99-114`)
- **The GGUF and safetensors loaders disagree on untied models:** GGUF ties `lm_head` to the embeddings even when the config says untied; safetensors refuses. GGUF alignment is also hardcoded to 32, ignoring `general.alignment`. (`gguf/loader.rs:506-510`, `parser.rs:275`)
- **Silent substitutions:**
  - 22 sites use `as_slice().unwrap_or(&[])`, feeding an empty embedding or KV into decode.
  - The Q8K→f32 remote wire fallback isn't recorded.
  - `LARQL_METAL_PLE` falls back to CPU silently.
  - `ov_rd/metal_backend.rs:23` falls back to CPU silently.
- **Hard 4096-token context limit:** the `ComputeBackend` decode/prefill path asserts above 4096 positions. (`larql-compute/src/ops/kv_cache.rs:314,379,453`)
- **User input silently wraps or is ignored:** LQL `expect_u32` truncates with `as u32`. `EngineKind::from_name` wraps `bits=260` to 4 and ignores unknown keys. (`larql-lql/src/parser/helpers.rs:411`, `larql-kv/src/lib.rs:514-542`)

### B. Robustness

- **Panics on bad input or an unexpected call order:**
  - `turbo_quant` panics when decode runs before prefill.
  - `plan_component_ops` has 8 `expect`s on container input.
  - `routed_experts/worker.rs` has 6 bare `unreachable!()`s.
  - `token.rs:62` panics on an unknown token id.
  - The JSONL loader unwraps every field.
  - The streaming extractor unwraps file open and mmap.
  - MLA shape mismatches hit `.expect`.
- **NaN panics:** `partial_cmp().unwrap()` in the top-k sorts (`walker/utils.rs`, `router.rs`, `lm_head/knn.rs`, `down_meta.rs`, `larql-kv/moe.rs:152`, `python/vindex.rs:880`) panics on any NaN score. Use `total_cmp`.
- **Process-wide environment writes:** `std::env::set_var` is called from library code, which is unsound when other threads read the environment. (`larql-inference` residual_diff `capture.rs:410`, `stages.rs:483`)
- **Unbounded server resources:** no caps on `top`, `limit` or `top_k`, the session count, or shard downloads (the tar is held in memory twice). The shard hash check is skipped when the hash is empty.
- **Plugin loading can be undefined behaviour:** `vindex3_cmd/plugins.rs` `dlopen`s Rust trait objects in the release binary, and its ABI stamp omits cargo features.
- **The GIL is never released:** long inference in `larql-python` blocks every Python thread.

### C. Architecture hardcoding outside `larql-models`' architecture layer

- **`larql-models`' own generic code:**
  - `detect/parser.rs` sets Gemma's rope 1e6 and head_dim 256 by `starts_with("gemma")`, and defaults missing head counts to 8/4.
  - `gguf/loader.rs:252-367` has a Gemma-4 block that overrides the file's `key_length` with 256.
  - `DEFAULT_GGUF_VOCAB_SIZE` is Gemma's 262144.
- **Two dispatch tables:** `detect/mod.rs:127-224` is a 25-arm order-sensitive match, and `detect/registry/table.rs` is a parallel registry. Adding a family takes about 6 edits. Fix: put a constructor and GGUF aliases on `ArchitectureEntry`, and delete the match.
- **Kimi-specific execution stack in VINDEX3:**
  - `kimi_kda_layer`, `kimi_mla_layer`, `kimi_moe_block`, `kimi_source`, `stack` and `token` form a second path beside the generic executor.
  - `kimi_router` is the generic sigmoid router, which GLM also uses, and is the third copy of it (`production.rs:578`, `reference.rs:598`).
  - `represent/quality.rs:680` hardcodes Kimi gates.
  - `teacher_forced.rs:126` writes `/tmp/kimi_*.json` from library code.
  - Metal mirrors all of this with `kimi_router_select` / `KimiLayerKernels`.
- **Family-based defaults and checks elsewhere:**
  - `LayerBands::for_family` is a table of (family, layer count) bands for about 9 families. (`larql-vindex/src/config/compliance.rs:37-190`)
  - `BitnetArchMeta::default()` is BitNet-2B-4T's geometry, kept silently when metadata is missing.
  - `boundary-per-layer` defaults `num_layers` to 34 (Gemma 3 4B). (`larql-kv/src/lib.rs:242`)
  - Image input is gated on `family() != "gemma3"`. (`run_cmd_vindex3/inputs.rs:164`)
  - `parity.rs:1243` maps every activation that isn't GELU to SiLU.
  - `MoeRouterKind::Gemma4Hybrid` branches decide which operands are required.
  - The GGUF exporter is hard-wired to `qwen35`.
  - `k3_ledger` bakes in one machine's bandwidth figures.
- **Chat templates:** three parallel template systems keyed on family literals.
- **Hardcoded developer paths:** `/Users/christopherhay/.cache/huggingface/…` is hardcoded in the tests of `connectors/projector.rs` and `encoders/vision_tower.rs`, in both `larql-models` and `larql-compute`.

### D. Decoupling and layering

- **VINDEX3 execution lives in the storage crate.** `format/vindex3/opplan/exec/` is about 60 modules, including 6.5k lines of NEON integer kernels that belong in `larql-compute`.
  - Codecs import the executor's types, so the layering is inverted.
  - Module cycles: `opplan ⇄ represent`, `encode ⇄ opplan`, `graph → opplan/represent`.
  - `larql-inference` reaches in through about 133 deep import paths; `vindex-cli` and `larql-cli` do too.
- **There is no backend trait for the lowered/V3 path.** `stack_metal.rs`, `kda_metal.rs`, `kimi_source.rs`, 12 CLI `vindex3_cmd/lowered/*` files and the server expert routes call `larql_compute_metal::{lowering, trait_impl::*}` directly. `DeviceBuffer` is just a re-export of the `metal` type. A second plan-to-Metal executor lives in the CLI. Fix: an encoder-level `LoweringBackend` trait in `larql-compute` with associated buffer types.
- **Metal's trait impls route everything to CPU:** `impl KvDispatch` and `impl AsyncComputeBackend` for `MetalBackend` pass every call to `CpuBackend`.
- **The shared compute trait carries Metal and format concepts:** methods such as `q4k_matvec_stride32`, `wire_resident` and `decode_token_q4k_moe`.
- **`larql-compute` is really the CPU forward runtime.** 37 of its files take `ModelArchitecture`. Consider splitting it into kernels plus a forward crate.
- **Four separate attention/layer encoders in Metal:** `decode/`, `ops/full_pipeline/`, `decode_hybrid.rs` and `lowering/`. The per-layer engine loop is copied about 7 times in `larql-kv`, and there are at least 10 near-duplicate decode loops in `larql-inference`.
- **`larql-router` depends on all of `larql-inference`** for two constants and three transport traits. Move those into `larql-router-protocol`.
- **`larql-vindex` `format` is a god-module:** it depends both ways on `index`, `extract`, `runtime` and `registry`. VINDEX3 is built on V2 types: the patch format, `KnnStore`, and `WeightSource`, which is defined in the legacy f32 writer.
- **Front-ends reimplement library logic:**
  - The Python bindings reimplement weight loading and DESCRIBE, so their answers diverge from LQL's.
  - The server's JSON-schema FSM is 1350 lines.
  - `routes/expert/cpu.rs` does expert math with a hardcoded BF16 stride.
  - `larql serve` re-declares the server's flags, and 18 of them are unreachable.
- **Process-global env-var switches:** there are about 77 `LARQL_*` knobs in Metal, 15+ in VINDEX3 hot paths and 12 selecting the server's expert path, plus a public global setter, `set_multi_position_ffn`.

### E. Crates and release surface

- **`model-compute` is dead:** nothing depends on it, yet it is in `default-members`. Delete it or move it out.
- **`larql-experts` is never built in CI,** and its consumer tests print "skip" and pass when the wasm is missing.
- **Unused dependencies:** optional `larql-compute-metal` in `larql-kv`, `larql-lql` and `larql-vindex` (used only by a bench), and `larql-core` in `larql-lql`. `larql-models` pulls in `larql-vindex-spec` only for `QuantFormat`.
- **Test-only code ships in release:** fixtures, `test_support` and `test_utils` are unconditional `pub mod`s in `larql-vindex` (about 2.4k lines) and `larql-inference`. A `test-utils` feature already exists; use it.
- **Research tooling ships ungated:** `dev` (26 subcommands, about 19k lines of `ov_rd`), plus `dec-bench`, `k3-ledger`, `moe-locality`, `parity` and `shannon`. Put them behind a `research` feature or a separate binary.
- **The public API surface is very wide:**
  - About 55 `pub mod`s in `exec/mod.rs`.
  - About 25 concrete `*Arch` types are re-exported.
  - Measurement modules are `pub` in `larql-kv` and `larql-lql`.
  - `lib.rs` re-exports the CPU-only `default_backend` next to the Metal-aware `default_compute_backend`.
- **CLI defaults only work inside the repo:** `.venv/bin/python`, `scripts/*.py`, `tests/fixtures/…`, and a walk up from the executable to `crates/larql-experts/target`. Shards default to `/tmp`.

## P2 — structure and hygiene (summary)

- **Tests under `src/`:** about 650 files, including `executor/tests.rs` (5054 lines), `parser/tests.rs` (2015) and 87 flat `*_tests.rs` files in VINDEX3. Moving them to `tests/` folders resolves most of the 800-line violations mechanically.
- **Oversized functions:**
  - `oracle_pq.rs`: about 3250 lines, built from about 40 `if args.*_probe` blocks. Replace with a probe registry.
  - `plan_component_ops`: about 1370 lines.
  - `parser.rs`: one function of about 750 lines.
  - `decode_token_with_moe_split_fn`: 846 lines.
  - `write_f32.rs`: one function of about 495 lines.
- **`ModelArchitecture` is one trait of about 200 methods (1,920 lines).** Split it into supertraits: keys, norms, attention, position, ple, moe, mla_dsa, multimodal. `ModelConfig` (about 120 fields) has no `Default`, so every new field breaks every struct literal.
- **The slice-preset list exists three times**, in the factory, `slice_cmd` and `publish_cmd`, and an unknown preset estimates 0 bytes.
- **`unsafe` without `SAFETY:` comments:** 48 blocks in `q4k_neon.rs` have none, and 36 raw `from_raw_parts(buf.contents())` calls skip the checked `try_read_buffer_f32`.
- **Inconsistent naming and layout:**
  - `*_cmd.rs` files alongside `mod.rs` directories.
  - `foo.rs` alongside `foo/` directories.
  - `--metal: bool` in some commands, `--backend` in others.
- **Shaders compile at startup:** all 69 MSL modules, including experimental kernels, are compiled from source at every `MetalBackend::new()`.
- **Uncommitted qkv-bias work:** `QKV_BIAS_FAMILIES` matches exactly while its sibling list matches by prefix, so `qwen2_vl` and `qwen2_5_vl` fall through. (`larql-models/src/architectures/qwen.rs:45`)

## Suggested order of work

1. **Hardening PR (the P0s).** Each is small and local: clamp and cancel for the server; bind, auth and patch-path fixes; GGUF length bounds; one checked f32-view helper used across `larql-vindex`, `larql-inference/trace` and `larql-python`; the Python lifetime fixes; `MetalInitError` plus the test sweep.
2. **Correctness PR (P1-A).** Stop tokens by id, prompt formats, compile config, patch width, `head_dim` of 0, MoE offset wrap, tensor-drop refusals, GGUF tie/alignment, and the `unwrap_or(&[])` sweep. Each fix needs a regression test.
3. **Architecture-hardcoding PR (P1-C).** Make `ArchitectureEntry` the only dispatch table, move the parser and GGUF family facts into architecture files, and remove the Gemma defaults from the kv/cli/server/vindex sites.
4. **Release-surface PR (P1-E).** Delete `model-compute`, add `larql-experts` to CI, feature-gate fixtures and research commands, and trim unused dependencies.
5. **Mechanical 800-line pass.** Move tests into `tests/` folders, then do the roughly 40 genuine splits from the detailed reports, one crate per PR.
6. **Structural work, post-release unless it blocks a backend.** Pull VINDEX3 `opplan/exec` out of the storage crate, add a `LoweringBackend` trait, fold the Kimi path into the generic executor, and split `ModelArchitecture` into supertraits. These need design first. They aren't mechanical moves.

## Hardening status (branch `release-hardening`)

Worked bottom-up through the crate chain. Gates at the end of the pass: `cargo fmt --check` clean; `cargo clippy --workspace --all-targets -D warnings` clean with default features and with `--no-default-features`; tests green in larql-models (1088), compute (1125), vindex (5556), inference (1834), kv (1332), lql (1108), router (259), server (1306), cli (1033), core, boundary, factory, vindex-spec, vindex-cli and the Python bindings (42 pass, 15 model-backed skips). larql-compute-metal compiles and is clippy-clean. Its GPU suite was not run in this pass.

**Fixed**
- Every source, test and example file is ≤ 800 lines except the five listed under *Open*.
- ModelArchitecture is a supertrait stack. The architecture registry is the single family table: config defaults, GGUF translation, layer bands, chat format.
- GGUF/packed/trace/boundary parsing: bounded counts, checked sizes, checked f32 views. No panic on untrusted headers.
- Python: the unsafe mmap loader is replaced by `larql_vindex::load_model_weights`. `ResidualTrace` owns `Arc`s, so a trace no longer outlives its model's memory. `bytes` → `f32` is alignment-safe. DESCRIBE goes through the new `larql_lql::describe` API, so Python and LQL now give the same answer.
- Server:
  - `max_tokens`, `top`, `limit` and `top_k` are bounded in one `routes::limits` module (HTTP and gRPC); an over-limit value gets a 400.
  - Loopback is the default bind. A network bind needs `--api-key`, `--insecure-public` or `--public-explorer`.
  - `/v1/patches` URLs are confined to `--patch-dir`; `hf://` needs `--allow-hf-patches`.
  - gRPC enforces `--api-key` and a concurrency limit.
- Router: walk-ffn wire constants are single-sourced in larql-router-protocol, and `http.rs` is split into handlers.
- CLI:
  - compile no longer stamps `Gemma3ForCausalLM` onto every model;
  - config-copy errors propagate;
  - `larql serve` lives in `serve_cmd`.
- Plugin ABI stamp includes target, larql-vindex features and `debug_assertions`.

**Open**
- Single-function files still over 800 lines. Each needs a real decomposition, not a move:
  - `decode/encode_attn.rs::encode_attention_block` and `decode/token.rs::decode_token_with_moe_split_fn` (Metal; need a GPU-verified refactor);
  - `ov_rd/oracle_pq.rs::run_oracle_pq` (~3250 lines; the probe-registry design in §2 of cli_server.md);
  - `compare_ollama.rs` (one 1074-line `main`).
- **H9 is a protocol bug, not a missing check.** The router sends the donor's `vindex_identity_hash` (a 16-hex `DefaultHasher` of model id and layer count) as `shard_hash`. `shard_loader` compares that against the SHA-256 of the tar, so Mode-B verification cannot pass against a real donor. The protocol needs a content hash before "missing hash is fatal" means anything.
- GIL release in larql-python (needs `Send` state behind the new `Arc`s).
- Router → inference dependency: the transport traits move cleanly to router-protocol, but the vindex exec profiler (`profile::record_provider_call`) the router calls needs its own seam first.
- `research` cargo feature to keep `larql dev`/ov_rd and the research verbs out of release builds.
- Naive reference kernels (`parity/reference.rs`) → `larql-compute::reference`.
- GGUF exporter target trait (`export_qwen35`); `Gemma4Hybrid` rename; Kimi-named items in represent/opplan.
