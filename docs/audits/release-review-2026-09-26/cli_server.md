# Pre-release review — larql-cli, larql-server, larql-demos, larql-continuation-fixture

Read-only review, 2026-09-26, branch `vindex3/qkv-attention-bias` (working tree has uncommitted changes, none in these crates' src besides `larql-server/tests/test_expert_endpoint.rs`).
Severity: **P0** = release blocker (security/correctness reachable by a user or network peer), **P1** = must fix or consciously accept before release, **P2** = cleanup.

---

## 0. Headline findings (read these first)

| # | Sev | Where | Finding | Fix |
|---|-----|-------|---------|-----|
| H1 | P0 | `larql-server/src/routes/openai/completions.rs:216`, `chat/handler.rs:140`, `v3_completions.rs:46`, `responses/handler.rs:93` | `max_tokens` from the HTTP body is never bounded. It flows into `Vec::with_capacity(max_tokens)` (`larql-inference/src/layer_graph/generate/cpu.rs:106`, `gpu/mod.rs:397`, `gpu/decode_loop.rs:75`, `constrained.rs:237`, `vindex/kquant_forward/generation.rs:44,216,394`). `{"max_tokens": 1e15}` makes the allocation fail, which aborts the whole server process (it's an abort, not a panic that can be caught). A smaller large value keeps generating while holding the exclusive `weights` write lock. The inference timeout drops the future, but `spawn_blocking` keeps running. | Clamp in one request validator: `max_tokens = min(req, model.max_context - prompt_len, SERVER_MAX_TOKENS)`, and return 400 when it's over. In inference, drop the `with_capacity(max_tokens)` or cap it. |
| H2 | P0/P1 | `larql-server/src/bootstrap/cli.rs:16` + `bootstrap/mod.rs:545` | The default bind is `0.0.0.0` and auth is off by default (`--api-key` is optional). The default (non-explorer) profile also mounts mutating and filesystem-reading routes (see H3, H4). | Default to `127.0.0.1`. Refuse `--host` set to a non-loopback address unless `--api-key` or an explicit `--insecure-public` is given. |
| H3 | P1 | `larql-server/src/routes/patches.rs:40-58` | `POST /v1/patches {"url": "<any path>"}` passes the string straight to `PathBuf::from(url)` → `VindexPatch::load`. That lets a client read and probe any file on the server's filesystem. The `Internal(...)` error text is echoed back (`error.rs:97`), which tells the client whether a file exists and how it parsed. `hf://` values also start a hub download on the server, which can fill the disk. | Remove the local-path branch from the HTTP API, or restrict it to a configured `--patch-dir` using canonicalise + `starts_with`. Only accept `hf://` behind an allow-list flag. |
| H4 | P1 | `larql-server/src/capabilities.rs:170` + `plan_service.rs` | `/v1/plan` accepts `SourceScheme::Local` on every profile except PublicExplorer. On the default profile that means an unauthenticated network client can make the server open local directories and read their headers/config. | Allow Local only when auth is on or the bind is loopback. Default-deny otherwise. |
| H5 | P1 | `larql-server/src/bootstrap/mod.rs:567-605`, `grpc.rs` | The gRPC server (`--grpc-port`) gets none of the HTTP middleware: no API-key auth, no rate limit, no concurrency limit. `--api-key` therefore leaves infer/walk/walk_ffn/expert/shard open on the gRPC port. | Add a tonic `Interceptor` that reuses `auth::tokens_match`, plus `tower::limit` layers. |
| H6 | P1 | `larql-server/src/grpc.rs:256,314,449,629` and `routes/{walk,describe,select,infer}.rs` (`top`/`limit: usize`) | `top`, `limit` and `top_k` from requests are unbounded. gRPC defaults `top_k` to the literal `8092`, duplicating `routes/walk_ffn/types.rs:94` (a documented probable typo for 8192). | Add a shared `MAX_TOP_K`/`MAX_LIMIT` clamp in one validation module. Use the constant in grpc.rs. |
| H7 | P1 (arch) | `larql-cli/src/commands/extraction/compile_cmd/save.rs:148-167` | `copy_model_config` writes `"architectures": ["Gemma3ForCausalLM"]` into **every** compiled model's config.json. When `text_config` is present it also forces `model_type: gemma3_text` and `tie_word_embeddings: true`. Compiling a Llama/Qwen/Mistral vindex therefore produces a config that downstream loaders will mis-detect. Every I/O error is ignored (`let _ =`). | Keep the source's `architectures`/`model_type`. Only unwrap `text_config` when the arch trait says it is a multimodal wrapper, and take the text-arch name from the trait. Propagate errors. |
| H8 | P1 (sec) | `larql-cli/src/commands/primary/vindex3_cmd/plugins.rs:196-235` | The `--plugin` loader `dlopen`s a Rust cdylib and passes Rust trait objects across the boundary. Its SAFETY argument is that equal stamps imply equal layout. But the stamp records only the compiler and commit, not the cargo feature set (`larql-continuation-fixture/Cargo.toml:21-25` says so), so a `gpu` vs non-`gpu` mismatch is UB. It also ships in the release binary. | Add the enabled-feature set and a `cfg` hash to `plugin::abi()`, or feature-gate `--plugin` out of release builds. |
| H9 | P1 | `larql-server/src/shard_loader.rs:112-143` | Mode-B shard download buffers the whole tar in memory (`resp.bytes()`, then `bytes.clone()` for the blocking task), so a multi-GB shard is held twice. The hash check is skipped when `expected_hash` is empty or all zeros, so any router or MITM that can supply the hash can switch verification off. | Stream to a temp file while hashing. Make a missing hash a hard error unless `--allow-unverified-shards` is set. |

---

## 1. Decoupling (thin front-ends?)

**CLI.** `larql-cli` depends directly on 11 larql crates: router, core, compute, compute-metal, inference, kv, models, lql, vindex, vindex-spec, factory. It also depends on `safetensors`, `tokenizers`, `ndarray`, `memmap2` and `image`, which only make sense if format and numeric work lives in the CLI. It reaches deep into internals, for example `larql_vindex::format::vindex3::opplan::exec::{backend,operands,weights}` and `larql_compute_metal::lowering::*` in `vindex3_cmd/lowered/mod.rs:19-35`.

| Sev | Where | Business logic in the front-end | Move to |
|-----|-------|------|-----|
| P1 | `larql-cli/src/commands/primary/vindex3_cmd/lowered/` (mod.rs 1005, routed.rs 606, step.rs 524, resident.rs 477 incl. `rope_inv_freq_table` :409, `rope_table_key` :321) | A complete ComponentOpPlan→Metal executor (G6d). The module comment says it sits in the CLI because that is the only place where both vindex and compute-metal are in scope. The server has its own, separate V3 Metal path (`larql-server/src/vindex3.rs:100`), so there are two V3 GPU executors and the server can't use this one. | A bridge crate (e.g. `larql-vindex3-metal`) or `larql-inference` behind `gpu`. There are two consumers (CLI and server), so the crate isn't speculative. |
| P1 | `larql-cli/src/commands/diagnostics/parity.rs:887-1250` | Naive reference kernels (`naive_matvec`, `naive_rms_norm`, `naive_softmax`, `naive_gelu_tanh`, `naive_silu`, `reference_moe_block`, `reference_one_expert`). | `larql-compute::reference` (test-utils or a pub reference module), so tests can share them. |
| P1 | `larql-cli/src/commands/extraction/compile_cmd/save.rs:29-167` | Weight merging and safetensors writing for AOT compile. | `larql-vindex` (patch compile) or `larql-models::loading::safetensors` writer. |
| P1 | `larql-cli/src/commands/primary/run_cmd.rs:1373-1830` (`mod experts`) | Decode-strategy selection (`MetalQ4K/CpuQ4K/CpuF32`, `pick_strategy`), per-token generate loop, and chat-template detection that hand-parses `index.json` (`detect_template` :1602). | `larql-inference` generation dispatch, plus `larql-vindex` for the index.json family (there is already a loader). |
| P1 | `larql-cli/src/commands/primary/run_cmd.rs:712-1370` | Remote-MoE orchestration and engine-compatibility policy (`run_with_moe_shards`, `run_with_routed_container`, `run_with_remote_ffn`). The engine policy is duplicated in `walk_cmd.rs:879-912` (`validate_engine_spec`, `engine_unsupported_on_uncached_path`). | `larql-kv::EngineKind::supports_remote_ffn()` as the one authority. Orchestration goes to `larql-inference::ffn::moe_remote`. |
| P1 | `larql-cli/src/commands/extraction/walk_cmd.rs:475-1330` | Q4K predict/generate paths (resident vs uncached vs remote), `arch_needs_per_layer_embeddings`, stop-token logic `is_stop_token` :1324. | `larql-inference` (there is already a `generate_kquant_*` family). |
| P1 | `larql-cli/src/commands/primary/k3_ledger/fetch.rs` (+ `geometry.rs`) | A safetensors header parser over HTTP range requests (`SAFETENSORS_LEN_PREFIX`), built on its own hand-rolled HF URL (`fetch.rs:37`). | `larql-models::loading` (remote header reader). `larql-vindex` already has remote-source code. |
| P1 | `larql-server/src/routes/expert/cpu.rs:31-310` | Expert FFN math inside an HTTP handler: Q8K activation quantisation, packed BF16 stride arithmetic (`gu_stride = 2*inter*hidden*2` :122 hardcodes BF16), rayon fold. `resolve_bytes` returning `None` means an expert gets skipped (check whether the skip is silent). | `larql-inference::experts` (a `run_experts_batch(weights, layer, ids, h)` API). The handler should only decode and encode. |
| P1 | `larql-server/src/routes/openai/schema/{fsm.rs 1350, parser.rs 475}` | JSON-schema → token FSM for constrained decoding lives in the server, while `larql-inference` owns `generate_constrained_*`. The CLI can't reuse it. | New `larql-inference::constrain::schema` module. The server keeps only OpenAI request mapping. |
| P2 | `larql-server/src/routes/patches.rs:76-130` | Gate-vector synthesis from entity embeddings (`enrich_patch_ops`). | `larql-vindex::patch` (`synthesize_gate_vector(embeddings, tokenizer, entity)`). |
| P2 | `larql-cli/src/main.rs:420-880` (`ServeArgs`, `serve_command_args`, `run_serve`) | `larql serve` re-declares about 30 server flags and shells out to the `larql-server` binary. It has drifted: 18 server flags can't be reached through `larql serve` (`hnsw`, `hnsw_ef_search`, `session_ttl_secs`, `lazy_weights`, `max_q4k_cache_layers`, `infer_timeout_secs`, `no_docs`, `public_explorer`, `public_url`, `vindex_store`, `v3_kv_cache_entries`, `v3_kv_ttl_secs`, `memcheck_*`, `available_ram`, `http3_port`, `quic_cert_fingerprint`, `shard_query_tau`, …). It also falls back to a `larql-server` found on `PATH`, which may be a different version (`main.rs:863-874`). | Use `larql_server::bootstrap::cli::Cli` via `#[command(flatten)]` and call `bootstrap::run` in-process (the server is already a lib — demos link it), or pass argv through verbatim with `trailing_var_arg`. |

**Research/dev tooling separation (P1).** None of it is gated behind a feature or a separate binary.
- `larql dev *` (26 subcommands incl. `ov-rd`, about 19k lines in `commands/dev/ov_rd/`) sits under a `Research` help heading but is always compiled.
- Several research tools are **top-level primary verbs**: `dec-bench` (`main.rs:104`), `k3-ledger` (`:109`, a Kimi-K3-specific research ledger), `moe-locality`, `parity`, `accuracy`, `shannon`, `optimizer-mcp`. Most of `vindex3 {measure, observe, intervention, sensitivity, input-moments, teacher-force}` is research too.
- These pull in repo-relative defaults that fail in an installed binary (§5, §6).

Recommendation: add a `research` cargo feature (off by default for release artifacts, on for the dev workflow and CI). Move `DevCommand`, `DecBench`, `K3Ledger`, `MoeLocality`, `Parity`, `Shannon verify`, and `ov_rd` behind it, or into a second binary `larql-research` in the same crate. This also removes about 60k lines from the release build.

---

## 2. Files over 800 lines (source) and proposed splits

| Lines | File | Proposed split |
|-----|------|------|
| 4042 | `cli/commands/dev/ov_rd/oracle_pq.rs` | `run_oracle_pq` alone is one **~3250-line function** (:503-3755) made of ~40 `if args.address_*_probe {…}` blocks (294 `args.` refs). Split it into `oracle_pq/{args.rs (lines 1-500), setup.rs (capture/fit :503-1320), probes/<one file per probe family: key_group, majority, code_substitution, code_class_collapse, code_position, conditional_quotient, code7, lsh, supervised, gamma, prev_ffn, ffn_first, attn_relation, attn_cluster, reduced_qk>.rs, mode_d.rs, eval_loop.rs, parse.rs (:3755-4042)}`. Probes go into a `Vec<Box<dyn AddressProbe>>` registry so the orchestrator is ~200 lines. |
| 2384 | `cli/commands/primary/dec_bench/replay.rs` | `wire.rs` (WireArm/WireSpec/EndpointKind/Endpoint :25-420), `sweep.rs` (SweepPoint, parse_batch_list, expand_sweep :425-486), `frames.rs` (build_*_frame, check_experts_response :487-688), `summary.rs` (RequestSample…summarize :689-1036), `denominators.rs` (movement_ratio…moe_weight_stats :1037-1227), and `tests/` for :1228-2384 (≈1150 lines of in-src tests). |
| 1964 | `cli/commands/primary/run_cmd.rs` | `run_cmd/{args.rs (RunArgs :77-400), mod.rs (run/run_once/run_chat), bitnet.rs, moe_shards.rs (:712-1077), routed_metal.rs (:1078-1254), remote_ffn.rs (:1255-1372), experts.rs (:1373-1830)}` with tests in `tests/`. Better still, move the logic out (§1). |
| 1606 | `cli/commands/primary/k3_ledger/report.rs` | One file per subcommand renderer: `report/{budget,touch,frontier,block,transcode,ceilings,symbol_census,kda_graph,freqmass,formats,homogeneity,actions}.rs` plus a shared `header.rs`. Each is 70-220 lines. |
| 1481 | `cli/commands/extraction/walk_cmd.rs` | `walk_cmd/{args.rs, vindex_walk.rs, predict_q4k.rs (:475-906), predict.rs (:913-1216), stream.rs, print.rs}`. Move the predict logic to inference. |
| 1444 | `cli/commands/dev/ov_rd/oracle_pq_address.rs` | One file per `fit_address_*_group_models` family. |
| 1350 | `server/routes/openai/schema/fsm.rs` | Move to inference (§1). Split `fsm/{frame.rs, atom.rs, object.rs, array.rs, union.rs}`. |
| 1343 | `cli/commands/diagnostics/parity.rs` | `parity/{args.rs, lm_head.rs, moe_expert.rs, moe_block.rs, layer_diff.rs, residual_dump.rs}`. Naive kernels go to compute (§1). |
| 1265 | `cli/commands/dev/ov_rd/pq_exception.rs` | Split by exception kind plus a report module. |
| 1232 | `cli/commands/primary/vindex3_cmd/mod.rs` | `args.rs` (Vindex3Command + 12 Args structs :19-583), `verify.rs`, `encode.rs`, `represent.rs` (:802-986), `inspect.rs` (:1030-1195), `plan.rs`. |
| 1072 | `cli/commands/primary/dec_bench/capture_format.rs` | Reader, writer and schema/validation. |
| 1050 | `cli/src/main.rs` | `cli.rs` (Commands/DevCommand enums), `dispatch.rs`, `serve.rs` (§1), `trampoline.rs` (`rewrite_legacy_argv`) and tests → `tests/`. The `documentation_tests` at :1015 use `include_str!("../../../docs/...")`. |
| 1014 | `server/src/shard_query.rs` | `shard_query/{source.rs, cache.rs, grpc.rs, scoring.rs}`. |
| 1005 | `cli/commands/primary/vindex3_cmd/lowered/mod.rs` | Move out of the CLI (§1). |
| 948 | `server/routes/embed.rs` | `embed/{binary.rs, json.rs, logits.rs}`. |
| 940 | `cli/commands/dev/ov_rd/probe_program_class.rs` | args / probe / report |
| 918 | `cli/commands/primary/bench/grid_lan.rs` | topology / run / report |
| 913 | `cli/commands/primary/vindex3_cmd/generate.rs` | args / loop / output |
| 883 | `cli/commands/primary/run_cmd_image.rs` | preprocessing (anyres already separate) / run / tests |
| 848, 841, 825 | `ov_rd/{edit_catalog,gamma_address,address}.rs` | per concern |
| 846 | `server/src/grpc.rs` | `grpc/{vindex_service.rs, walk.rs, infer.rs, walk_ffn.rs}` |
| 845 | `cli/commands/extraction/trajectory_trace_cmd.rs` | trace / metrics / print |
| 839 | `server/src/state/loaded_model.rs` | weights-cache / patch state / lifecycle |
| 831 | `server/routes/openai/completions.rs` | `completions/{types.rs, buffered.rs, stream.rs, bitnet.rs}` |
| 831 | `server/src/openapi.rs` | split schema registration by route group |
| 815 | `cli/commands/primary/dec_bench/window_union.rs` | model / report |
| 809 | `server/src/bootstrap/mod.rs` | `bootstrap/{router.rs (middleware stack :520-565), grpc.rs (:567-605), announce.rs, run.rs}` |
| 1253, 834, 805 | `demos/examples/inference/{gwread1_replay,observatory_record}.rs`, `demos/examples/vindex/demo_features.rs` | Examples over the limit. gwread1_replay is a research replay, not a demo: move it to `dev` or split it. |

Test files over 800 lines (the rule applies to them too): `server/tests/test_vindex3_serve.rs` 1804, `test_unit_state.rs` 1588 and `test_http_embed.rs` 1381 should each be split by route/feature.

---

## 3. File/folder structure

- **P1 tests in `src/`**: 84 CLI src files and 52 server src files have `#[cfg(test)]`, and there are 31 CLI and 19 server `*tests*.rs` files under `src/` (e.g. `extraction/verify_cmd_tests.rs`, `primary/continuation_tests.rs`, `k3_ledger/*_tests.rs`, `routes/sessions/tests/`, `routes/openai/responses/tests/`). This conflicts with the project's tests-in-`tests/` rule. Examples: `dec_bench/replay.rs:1228-2384` and `run_cmd.rs:1828-1964` are inline test blocks. Fix: move them to `tests/` and expose a `pub(crate)`→`#[doc(hidden)] pub` seam, or accept sibling `tests/` subdirs as policy and write that down.
- **P2 inconsistent command layout**: `commands/primary/` holds both single files (`run_cmd.rs`, `run_cmd_image.rs`, `run_cmd_speak.rs`, and a `run_cmd_vindex3/` dir) and dirs with `mod.rs`. Suffixes are mixed (`*_cmd.rs` vs `cache.rs`, `continuation.rs`, `serve_resolve.rs` vs `shannon_cmd/` vs `shannon_trace/`). Fix: `commands/<verb>/mod.rs` per verb, with `run/{mod,image,speak,vindex3}.rs`.
- **P2 misfiled groups**: `commands/extraction/` holds `predict_cmd`, `walk_cmd`, `qk_*`, `ffn_latency_cmd`, `kg_bench_cmd`. These are dev/research commands dispatched from `DevCommand`, not extraction. `verify_cmd` and `diag_cmd` are listed under the "Build" help heading. `parity` and `moe_locality` are "Build" in help but live under `diagnostics/`. Fix: make the folder follow the help heading.
- **P2 legacy aliases**: `Extract` + `ExtractIndex` (duplicate), `Chat` (alias of `run`), the legacy `Query/Describe/Stats/Validate/Merge/Filter` graph-file group (pre-LQL), and the argv trampoline `rewrite_legacy_argv` (`main.rs:889`). Decide before release whether they get a deprecation window or are removed.
- **P2 server module style**: `routes/vindex3_ffn.rs` sits alongside `routes/vindex3_ffn/stream.rs` (mixed 2018 style), and `vindex3_experts.rs`/`vindex3_layers.rs` are flat while `expert/` and `walk_ffn/` are dirs. Pick one. Route paths are centralised in `paths.rs` and the `Mount` ledger asserts no double-mount (good).
- **P2 `routes/expert/{cpu,metal}.rs`**: handler files split by backend (see §4).
- `larql-continuation-fixture`: 25-line cdylib, `publish = false`, comment is clear. OK.
- `larql-demos`: `publish = false`, examples-only. OK, apart from the size and path issues below.

---

## 4. Hardcoding to a backend/module

| Sev | Where | Finding | Fix |
|-----|-------|---------|-----|
| P1 | 9 arg structs use `--metal: bool` (`run_cmd.rs`, `walk_cmd.rs`, `bench/args.rs`, `bench/local_runtime.rs`, `dec_bench/args.rs`, `shannon_cmd/{args,vindex}.rs`, `ov_rd/{eval,induce}_program/args.rs`), while `vindex3_cmd`/`bench vindex3`/`parity` use `--backend <kind>` | Two selection surfaces. `backend_select.rs` already has a `BackendKind` registry. | Use `--backend {cpu,metal,…}` everywhere, keep `--metal` as a hidden alias, and route every construction through `backend_select::backend_for_kind`. |
| P1 | Direct `MetalBackend::new()` bypassing the registry: `diagnostics/parity.rs:650`, `ov_rd/metal_backend.rs:17`, `ov_rd/induce_program/capture.rs:165`, `vindex3_cmd/prepare.rs:258,287,308`, `vindex3_cmd/measure.rs:290`, `bench/vindex3_runtime.rs:312` | Backend is hardcoded. `ov_rd/metal_backend.rs:23` **silently falls back to CPU** when Metal is missing, even though `--metal` was asked for. | Go through `backend_for_kind` and make an explicit request fail loudly (as `backend_select` already does). |
| P2 | `larql-server/src/routes/expert/metal.rs` + `cpu.rs`, `env_flags.rs` (12 `LARQL_*` env vars: `USE_METAL_EXPERTS`, `DISABLE_Q4K_DIRECT`, `MOE_BATCH_MODE`, `I8_WIRE`…) | Execution-path selection happens in handlers through env vars, not config. `metal.rs:1-10` says Metal experts are opt-in because of a Gemma-4-26B inter=704 accuracy bug. | Move path selection into an `ExpertExecutor` chosen at boot from `Cli`, and report it in `/v1/capabilities`. Before release, either confirm the Metal path stays off by default or fix the bug. |
| P2 | `routes/expert/cpu.rs:122-123` | Packed-expert stride assumes BF16 (`* 2`). | Get the stride from the stored dtype (see §1). |
| P2 | `run_cmd_vindex3/inputs.rs:153` | `--image` on V3 is refused under `--metal`. That's fine, but it's a bool check and not a capability query. | Ask the backend for the capability. |

---

## 5. Hardcoding to an architecture / model

| Sev | Where | Finding | Fix |
|-----|-------|---------|-----|
| P1 | `cli/commands/extraction/compile_cmd/save.rs:151,153,154,165` | `Gemma3ForCausalLM`/`gemma3_text`/tie=true forced on all compiled output (H7). | See H7. |
| P1 | `server/routes/patches.rs:73` `LEGACY_FEATURE_SLOT_SPACE = 10240` | The comment admits this pins Gemma's FFN width and is wrong for other archs. On any other model, INSERT with `feature==0` hashes into a slot space that doesn't match (possibly out of range). | Read `intermediate_size` from `model.config` (already on `LoadedModel`; no async lock needed) and keep 10240 only as a Gemma regression fixture. |
| P1 | `cli/commands/primary/run_cmd_vindex3/inputs.rs:164` | `arch.family() != "gemma3"` gates V3 image input. | Gate on `arch.multimodal()` protocol plus connector kind (SigLIP/avg-pool), which the code queries a few lines later anyway. |
| P2 | `cli/commands/primary/run_cmd_speak.rs:60` | `family() != "moss_tts_realtime"`. | Use an arch trait capability (`speech_protocol()`). |
| P1 | `cli/commands/diagnostics/parity.rs:1243-1248` `activation_for` | Maps every non-`GeluTanh` activation to `Silu`, so GeluExact/ReLU²/SiTU-GLU are silently wrong in the reference. `pre/post_experts_norm_for` (:1188-1204) silently return `&[]` when the key is missing. | Match exhaustively and error on unsupported. Treat a missing norm as a refusal. |
| P1 | `cli/commands/primary/k3_ledger/args.rs:11,18,28,325` | Top-level release verb defaults to `moonshotai/Kimi-K3` and `model-000NN-of-000096.safetensors`. `frontier.rs:16-17` hardcodes `BW_GPU_GB_S = 367.0`, `BW_CPU_GB_S = 127.0` (one machine). | Research gate (§1). Require the repo argument. Bandwidths become flags or measured values. |
| P2 | `server/routes/openai/chat/mod.rs:28`, `server/vindex3.rs:244`, `cli run_cmd.rs:1602-1624` | Chat template falls back to a substring match on the model id (`ChatTemplate::for_model_id`), and then silently to `Plain`. | Take the template from the container's tokenizer_config/chat_template. Log when falling back. |
| P2 | `server/routes/walk_ffn/types.rs:94`, `grpc.rs:632` | `8092` magic default (probable 8192 typo), duplicated as a literal. | One constant. Decide on the typo before release, since changing it later changes served behaviour. |
| OK | remaining `arch_literals.txt` hits in scope | All in tests or doc comments (`cache.rs`, `pull_cmd.rs:523`, `serve_resolve.rs:101`, `bootstrap/load.rs:600-711`, `publish_cmd/collections.rs` tests, `shard_loader.rs` tests). `publish_cmd/collections.rs::default_family` is a display-name heuristic for cards only. | — |
| P2 | `larql-demos/examples/{inference/chat_demo.rs:36, lql/compile_demo.rs:32-33, server/openai_demo.rs:570}` | Defaults to `output/gemma3-4b-*.vindex`. `compile_demo.rs:99` writes `/tmp/larql_compile_demo.vlp`. | Require an argument or env var. Use `tempfile` (already a dep). |

---

## 6. General code review

### Server
- **P0** H1 (`max_tokens`). **P1** H2-H6, H9.
- **P1** `error.rs:97`: `ServerError::Internal(msg)` returns the raw internal message (paths, IO errors, loader errors) to clients. Fix: log the detail with a request id and return a generic message for 5xx.
- **P1** Timeout (`error.rs:41-45`): "drop the in-flight spawn_blocking future" doesn't stop the blocking work. It keeps holding the `weights` write lock (chat/handler.rs:334 notes generation takes an exclusive write guard). One slow request blocks every other generation. Fix: a cooperative cancel flag checked per decode step, plus the H1 clamp.
- **P2** `CorsLayer::permissive()` (`bootstrap/mod.rs:557`) together with API-key auth lets any origin make credentialed-by-header calls. Fix: `--cors-origin <list>`.
- **P2** Swagger mounted before auth (`bootstrap/mod.rs:537-543`, intentional). Document it.
- **P2** Non-test `unwrap/expect` in handlers are guarded invariants, not request-reachable panics: `vindex3_experts.rs:29,41,77,110` (`expert_wire.as_ref().unwrap()` after `worker()` check), `walk.rs:55`/`grpc.rs:325` (after emptiness check), `completions.rs:277`, `embed.rs:422` (length validated at :403). Binary embed validates token ids against vocab (:164-172). The `expect` calls are fine, but replace the bare `.unwrap()`s with `expect("…invariant…")` or typed accessors.
- **P2** `tar::Archive::unpack` in `shard_loader.rs:143` rejects `..` but will create symlink entries. Fix: `set_overwrite(false)` and reject `EntryType::Symlink/Link`.
- OK: `shard_loader.rs:19-60` validates model_id against traversal. `auth.rs` uses constant-time compare. Body limit is 64 MiB on binary routes. Sessions have TTL eviction. Check for a max-session count; none was seen in `session/manager.rs`, so a client can create unlimited sessions within the TTL (**P2**, add a cap).

### CLI
- **P1** Repo-relative defaults that break in an installed binary: `shannon_cmd/args.rs:229-245` (`.venv/bin/python`, `scripts/shannon_score_{mlx,hf}.py` + a `Command::new(python)` at `shannon_cmd/verify.rs:302`), `dec_bench/args.rs:101` (`tests/fixtures/shannon_frankenstein_2k.txt`), `run_cmd.rs:1583` (walks exe ancestors for `crates/larql-experts/target/wasm32-wasip1/release`). Fix: gate them as research or make the args required.
- **P2** `run_cmd.rs:723-730`: the comment says only `standard`/`boundary_kv` are supported with remote MoE and that compression engines "silently drop experts". The code (:734-742) and the error text allow 7 engines. Either the comment is stale or the gate is wrong. Resolve it with the larql-kv authority from §1.
- **P2** `compile_cmd/save.rs:120-170`: every copy and write is `let _ =`, so a failed tokenizer/config copy still gives a "successful" compile. Propagate errors.
- **P2** `ffn_latency_cmd.rs:18` defaults to `http://127.0.0.1:9183`. Check that it matches `larql_server::DEFAULT_PORT` and use the constant.
- **P2** 7 non-test `process::exit` calls in the CLI bypass `Drop`/error reporting. Return an error instead.
- OK: no absolute `/Users/...` paths outside tests (`publish_cmd/collections.rs:296,316` are test fixtures). `server/announce.rs:175` and `bootstrap/mod.rs:655` default the shard store to `/tmp/larql-shards` (**P2**, macOS clears /tmp; use `dirs::cache_dir()` or require `--vindex-store`).

### Demos / fixture
- Demos: see §2 and §5. There are no tests, which is fine for examples, but CI should at least `cargo build --examples`.
- Continuation fixture: see H8 (the ABI stamp doesn't include features).
