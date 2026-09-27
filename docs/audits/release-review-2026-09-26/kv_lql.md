# Pre-release review: larql-kv, larql-lql, larql-python

Read-only review, 2026-09-26, branch `vindex3/qkv-attention-bias`. Paths are relative to `crates/`.
Severity: **P0** means memory-unsafe or UB, fix before release. **P1** means a correctness or architectural defect, fix before release if possible. **P2** is hygiene.

---

## 0. Top findings

| # | Sev | Where | Finding | Fix |
|---|-----|-------|---------|-----|
| 1 | P0 | `larql-python/src/walk.rs:170,212,237` | `Vec::from_raw_parts` over **mmap'd (read-only, non-allocator) memory**, then `mem::forget(arr.clone())` to stop the free. This is UB: Vec requires global-allocator memory, and it casts `*const`→`*mut` on a `Mmap`. | Use `ArrayView2::from_shape_ptr` tied to the mmap lifetime, or the vindex's own mmap-backed `WeightArray`. Better, delete `load_mmap_weights` and call `larql_vindex::load_model_weights*` (see #5). |
| 2 | P0 | `larql-python/src/walk.rs:208-212` (embed), `:233` (gate) | The embed zero-copy path never checks `vocab*hidden*4 <= embed_data.len()`, so a truncated file gives an OOB read. On the gate path, `ptr.add(float_offset)` runs **before** the bounds check at `:234`, which is UB by itself. The norm-vector path `:188-190` reads `entry.shape[0]` floats with no check against `raw.len()`. `offset + length` (`:144`) can overflow. | Bounds-check with `checked_add` and `checked_mul` before any pointer arithmetic, and return `Err` instead of `continue`. |
| 3 | P0 | `larql-python/src/trace_py.rs:14-27, 415-417`; created from `walk.rs:544` | `PyResidualTrace` keeps `*const ModelWeights` / `*const Tokenizer` borrowed from a `WalkModel`. Python code like `t = m.trace(..); del m; t.answer_trajectory(..)` is a **use-after-free**. `unsendable` does not help. | Hold a `Py<PyWalkModel>` (or `Arc<ModelWeights>` / `Arc<Tokenizer>`) in the trace instead of raw pointers. |
| 4 | P1 | `larql-kv/src/generation.rs:576-590`, used at `:261,281,331,347,415,436,482,501` + `masked_argmax :706-717` | Stop detection matches **decoded strings** against a hardcoded family list (`<eos>`, `<\|im_end\|>`, `<end_of_turn>`, …). The token is decoded with `decode(&[id], true)`, i.e. `skip_special_tokens=true`, so a special EOS decodes to `""` and never matches. Generation likely runs to `max_new_tokens` on real tokenizers. This is also an architecture hardcode, and it is duplicated in `larql-cli/.../walk_cmd.rs:1324`. | Use the canonical `larql_inference::layer_graph::generate::eos::EosConfig` (`eos_token_ids`) and compare **ids**. Add a test with a real special-token tokenizer. |
| 5 | P1 | `larql-python/src/walk.rs:45-285` | Python **reimplements weight loading and arch detection**. It builds a synthetic config JSON from a subset of fields (drops rope scaling, attention bias, softcap, PLE, MoE, etc.), then calls `detect_from_json`. It also hand-builds `ModelWeights`. Any architecture that needs a dropped field is silently mis-detected in Python only. | Replace with `larql_vindex::load_model_weights_with_opts` (the same path the CLI and LQL use). |
| 6 | P1 | `larql-kv/src/engines/turbo_quant/engine.rs:461,755,904` | `self.layers[layer]` is indexed with no prefill guard. Calling `decode_step`, `decode_step_resident`, `decode_step_quant_*` or `*_via_executor` before `prefill`, or with a model whose layer count differs, **panics**. Other engines return `Err`/`None`, and there is no `*_without_prefill` test here. | Return `EngineError::NotPrefilled`-style errors when `self.layers.len() != weights.num_layers`, and add the missing tests. |
| 7 | P1 | `larql-kv/src/lib.rs:238-244` | `boundary-per-layer` defaults `num_layers` to **34 ("Gemma 3 4B")**. This is an architecture hardcode, and on any other model the bare spec errors at prefill. | Make `num_layers: Option<usize>` and resolve it from `weights.num_layers` at prefill, or require `layers=`. |
| 8 | P1 | `larql-kv/src/lib.rs:148-278` (`EngineKind::from_name`) | Silent fallbacks on user input: an unparsable `window=abc` becomes `None` (unbounded); `bits=abc` becomes 4; `bits=260 as u8` wraps to 4 (`:204`); unknown keys (`windw=512`) are ignored; `chunk_tokens=x` becomes 512. `split_specs` then uses "does it parse" to merge pieces, which is fragile. | Return `Result<Self, SpecError>`. Reject unknown keys and unparsable or out-of-range values, and use `u8::try_from`. |
| 9 | P1 | `larql-kv/src/lib.rs:514,523,529,542` | `EngineKind::build_with_profiling` panics (`expect`/`panic!`) in library code on a user-reachable spec (`semantic-promotion:base=apollo`, or a bad `layers=`). | Return `Result<AnyEngine, EngineError>`. |
| 10 | P1 | `larql-python/src/vindex.rs:26-60, 741-884, 755-782`; LQL `executor/helpers.rs:132,268` | Python **reimplements DESCRIBE**: its own `is_readable_token` ("matches LQL executor logic"), a magic `20.0` score cutoff, garbage-label heuristics, and layer bands. Its band fallback (`num_layers/3`, `*5/6`) **diverges** from LQL `resolve_bands` (whole range after `LayerBands::for_family`). Python and LQL therefore give different answers on the same vindex. | Expose a `larql_lql` (or vindex) `describe()` API that returns structured edges, and have Python call it. |
| 11 | P1 | `larql-python` (no `py.detach` / `allow_threads` anywhere) | Every forward pass, generation and KNN holds the GIL, so a long `infer()` freezes all Python threads. | Wrap compute in `py.detach(...)`. This needs the captured state to be `Send`, so move `unsendable` state behind `Arc`. |

---

## 1. Decoupling

- **P2 – larql-kv → larql-inference is not an inversion, but the crate is misnamed.** `KvEngine`, `AnyEngine` and `EngineError` live in `larql-inference::kv_engine` (lib.rs:57-59 re-exports them), and inference's dispatch loop consumes the trait. That makes kv a plugin layer above inference, which is layered correctly. In practice larql-kv is the *decode-engine* crate: it uses `forward`, `attention`, `ffn`, `layer_executor`, `WeightsView` (~190 `WeightsView` refs) and runs whole forward passes. Recommendation: document it as "engines" (or rename it `larql-engines` post-release). Keep pure cache state (`cache.rs`, `CanonicalKvState`) free of forward logic.
- **P2 – dev-dependency cycles**: `larql-inference` and `larql-vindex` both dev-depend on `larql-kv` (`larql-inference/Cargo.toml:124`, `larql-vindex/Cargo.toml:96`). This works, but it can produce "two copies of type X" errors in tests and slows builds. Move those parity harnesses into `larql-kv/tests` or a dedicated integration crate.
- **P2 – dead dependencies**: `larql-compute-metal` is optional in both `larql-kv` and `larql-lql`, but neither crate references `larql_compute_metal` in source. The `gpu` feature only needs to forward to `larql-inference/gpu`, so drop `dep:larql-compute-metal` from both. `larql-core` is declared in `larql-lql/Cargo.toml` but never used in `src/`. Remove it.
- **P1 – engine code duplication.** Engines share the trait but not the driving loop. The sequence embed → `precompute_per_layer_inputs` → `for layer { executor.run_prefill_layer / run_decode_layer; apply_ple_and_layer_scalar }` → `last_row` is copied in `turbo_quant/engine.rs:680-800` (6 embed sites), `markov_residual/engine.rs:337-420`, `markov_residual_codec/executor.rs`, `windowed_checkpoint/engine.rs:750+`, `apollo/engine.rs`, and twice in `generation.rs` (`:601-704` hand-rolls the attention/FFN/PLE loop without any engine). Overall: 19 `apply_ple_and_layer_scalar` calls, and 7 `prefill_quant_via_executor` / `decode_step_quant_via_executor` pairs that differ only in "what to do with the layer's K/V". Fix: add `engines::drive::{prefill_layers, decode_layers}(weights, executor, ffn, scratch, tokens, on_layer: impl FnMut(layer, &h, kv) -> Result<..>)` next to `layer_ffn.rs` so each engine supplies only its state policy. `generate_cached_constrained` should drive a `StandardEngine` rather than its own loop.
- **P2 – Fused-executor silent downgrade**: `markov_residual/engine.rs:358,468` (and the same in the turbo_quant and windowed engines) fall back from `*_via_executor` to legacy `prefill_quant` when the executor is `Fused`. The legacy path *ignores the caller's FFN* (comment at turbo_quant `:672-676`), so a remote-FFN request silently runs local FFN. Refuse with an `EngineError` instead, or add the `requires_per_layer_dispatch()` hook the comment promises.
- **OK – LQL parser/AST separation is clean.** `ast.rs`, `lexer.rs` and `parser/*` import only `crate::ast` / `crate::lexer`, with no executor, inference or vindex types. `capability.rs` depends only on `ast` + `error`.
- **P2 – `relations.rs` at crate root** is executor logic (uses `larql_inference` / `larql_vindex`, `relations.rs:6-9`) sitting next to the pure front end. Move it to `executor/relations.rs`.
- **P2 – Executor returns `Vec<String>`** (`executor/mod.rs:134`). Execution and presentation are fused, so Python and the server can only get text and must re-derive structure. That is part of why Python reimplements DESCRIBE (#10). Post-release: return a typed `QueryResult` and render it in `repl.rs`.
- **OK – `larql-python/src/session.rs`** is a thin wrapper over `larql_lql::Session`. Nit: USE is built by string-escaping a path and re-parsing it (`session.rs:27-34`); construct `Statement::Use` directly.
- **P1 – `larql-python/src/vindex.rs` and `walk.rs` are not thin.** They reimplement loading (#5), DESCRIBE (#10), band resolution, and token filters.

## 2. Modularity / file size (>800 lines)

Most kv/lql oversize comes from **inline `#[cfg(test)] mod tests`**. Moving tests alone fixes 13 of the 29 files. Line = first `#[cfg(test)]`:

| File | Total | Non-test | Proposed split |
|---|---|---|---|
| `larql-lql/src/executor/tests.rs` | 5054 | 0 | Move to `larql-lql/tests/executor/{query,mutation,lifecycle,patch,remote,introspection}.rs` (by statement family). Needs `pub(crate)` → a `test-utils` feature or public test hooks. |
| `larql-lql/src/parser/tests.rs` | 2015 | 0 | Move to `tests/parser/{query,mutation,lifecycle,patch,trace,errors}.rs`. The parser is fully reachable through `larql_lql::parse`. |
| `larql-kv/src/engines/standard.rs` | 2287 | 1096 | `engines/standard/{mod.rs (struct+ctors+BackendSlot :52-160), prefill_decode.rs (:199-530), quant.rs (prefill_quant/decode_step_quant :605-790), per_layer_access.rs (PerLayerKvAccess :858-1075)}`, with tests in `tests/standard/`. |
| `larql-kv/src/engines/turbo_quant/engine.rs` | 1781 | 1007 | `turbo_quant/{codec.rs (TurboQuant + CompressedLayer :53-363), engine.rs (trait impl), executor_path.rs (:680-800), quant_cpu.rs (:802-1006)}`, with tests moved out. |
| `larql-kv/src/engines/markov_residual/engine.rs` | 1806 | 640 | Move tests only. |
| `larql-kv/src/generation.rs` | 1680 | 720 | `generation/{mod.rs, cached.rs (generate_cached*), engine.rs (generate_with_engine*), constrained.rs, argmax.rs}`. The sibling `generation/` dir already exists. |
| `larql-kv/src/engines/windowed_checkpoint/engine.rs` | 1512 | 912 | Split quant/executor paths (`:750+`) into `executor_path.rs` and move tests out. |
| `larql-kv/src/lib.rs` | 1352 | 548 | `EngineKind` (`:65-548`) → `engine_kind.rs` (+ `engine_kind/spec_parse.rs`). Move `tests` + `compliance_tests` (`:549-1352`) to `tests/engine_kind.rs`. |
| `larql-kv/src/engines/apollo/engine.rs` | 1258 | 603 | Move tests. |
| `larql-kv/src/engines/boundary_per_layer/engine.rs` | 1175 | 442 | Move tests. |
| `larql-kv/src/engines/markov_residual_codec/engine.rs` | 1132 | 352 | Move tests. |
| `larql-kv/src/engines/apollo/store.rs` | 1090 | 424 | Move tests. |
| `larql-kv/src/engines/markov_residual/compute.rs` | 1067 | ~64 before first cfg(test) (test hooks interleaved) | Move test-only override plumbing into `compute/test_hooks.rs` and the tests out. |
| `larql-kv/src/engines/boundary_kv/engine.rs` | 1061 | 391 | Move tests. |
| `larql-kv/src/engines/semantic_promotion/qualification.rs` | 969 | 540 | Move tests. |
| `larql-kv/src/engines/windowed_checkpoint/extend.rs` | 965 | 509 | Move tests. |
| `larql-kv/src/engines/markov_residual/walk.rs` | 925 | 541 | Move tests. |
| `larql-kv/src/engines/markov_residual_codec/walk.rs` | 850 | 440 | Move tests. |
| `larql-lql/src/executor/vindex3.rs` | 1086 | 1027 | `executor/vindex3/{mod.rs (bind, compose_overrides, helpers :867-1027), infer.rs (:75-261), stats.rs (:262-462), explain_trace.rs (:463-660), show.rs (:688-866)}`. |
| `larql-lql/src/lexer.rs` | 1046 | 674 | Move tests. `Keyword` table → `lexer/keywords.rs`. |
| `larql-lql/src/executor/remote/mod.rs` | 1005 | 220 | Move tests. |
| `larql-lql/src/relations.rs` | 804 | 316 | Move tests, then move the file to executor/. |
| `larql-python/src/vindex.rs` | 1487 | 1487 | `vindex/{mod.rs (PyVindex+props), types.rs (DescribeEdge/Relation/FeatureMeta/WalkHit :157-356), embeddings.rs, gates.rs, knn.rs, describe.rs, relations.rs, mutation.rs, infer.rs}`. The existing `═══` section banners at `:495-1216` are the cut lines. |
| `larql-python/src/walk.rs` | 1105 | 1105 | `walk/{load.rs (delete in favour of the vindex loader, #5), model.rs (PyWalkModel core), interp.rs (:555-926 lazarus surface), lens.rs (:927+)}`. |
| Test files in `tests/` over 800 (`cov_remote_mockito.rs` 1087, `vindex3_v2_parity.rs` 1032, `cov_parser_lexer.rs` 989, `continuation_mem_1/main.rs` 941, `cov_tier_a_synthetic.rs` 820) | | | Split by statement family (P2). |

## 3. File/folder structure

- **P1 – Tests in src/** (project rule): 90 kv files and 35 lql files carry `#[cfg(test)]`. Beyond inline modules there are 8.2k (kv) and 7.3k (lql) lines in `src/**/tests.rs` or `src/**/tests/` dirs (`larql-kv/src/{vindex3,model_walk,engines/semantic_promotion}/tests/`, `generation/kv_run/tests.rs`, `markov_residual{,_codec}/step/tests.rs`). Migrate to `crates/*/tests/`, starting with the two giant lql files.
- **P2 – test-only code in production modules**: `engines/mod.rs:127-199` (W10 thread-local override, `Q4kFlagGuard`) and the `markov_residual/compute.rs` env-override map. Move them into a `#[cfg(any(test, feature = "test-utils"))] mod test_hooks`.
- **P2 – mixed module layout**: `generation.rs` + `generation/`, `generation/kv_run.rs` + `generation/kv_run/`. Pick `mod.rs` or sibling-file style consistently when splitting.
- **P2 – `engines/no_expert_route.rs` / `layer_ffn.rs`** are shared helpers inside `engines/`. Group them as `engines/shared/` along with the proposed `drive.rs`.
- **P2 – `accuracy.rs`, `accuracy_suite/`, `vindex_compare.rs`, `profiler.rs`, `model_walk/`** are measurement tooling in the library's public API (`pub mod`). Put them behind a `bench`/`measure` feature so the release library surface is only engines + cache.

## 4. Hardcoding to a module/backend

- **P1 – engine selection by string match**: `EngineKind::from_name` (`lib.rs:148-278`) is a single alias table with per-arm param parsing. That is acceptable as a CLI parser, but it is the *only* registry: `display_name`, `supported_names`, `bench_specs` and `build` are four parallel matches kept in sync by tests. Suggestion: add `EngineSpec` trait objects with `name/aliases/parse/build`, register them in one table, and make the matches mechanical. At minimum return `Result` (#8).
- **P2 – env-var knobs in library code** (hidden, global, untyped):
  - `LARQL_MEMIT_RIDGE`, `LARQL_MEMIT_TARGET_DELTA` and `LARQL_MEMIT_SPREAD` are each parsed twice: `larql-lql/src/executor/lifecycle/compile/into_vindex.rs:167-178` and `into_model.rs:80-99`. Bad values silently fall back to defaults. Add a single `MemitOpts::from_env()` (or better, LQL `WITH (...)` options) that errors on unparsable values.
  - `LARQL_KNN_EARLY_EXIT` (`executor/query/infer.rs:56`), `LARQL_FR3_EXPLICIT` (`query/select/edges.rs:279`).
  - kv: `LARQL_W10_DISABLE` (`engines/mod.rs:140`), `LARQL_INSTRUMENT_UNLIMITED` (`windowed_checkpoint/extend.rs:339`), `LARQL_INSTRUMENT_MARKOV` (`markov_residual/walk.rs:159`).
  - Route all of them through `larql_compute::options` (the existing registry) so they are enumerable and documented.
- **P2 – Q4K lm_head silent fallback**: `generation.rs:548-573` falls back to f32 when the flag is on but no Q4_K head view exists. Record which path ran (decode-stage / dispatch-path fact) so a benchmark can't mistake one for the other.
- **OK**: no inline Metal paths in kv/lql/python. Backends come through the `EngineBackend` / `ComputeBackend` traits.
- **P2 – LQL `Backend` enum double-match**: 15+ `let Backend::Vindex3 {..} = &self.backend else { unreachable!("caller matched the backend") }` (`executor/vindex3.rs:89,270,402,465,589`, `mutation/insert/compose_v3/balance.rs:96,128,157,190,199`, `mutation/rebalance.rs:195,269`, `insert/knn_v3.rs:42`, `lifecycle/compile/into_vindex_v3.rs:37`, `mutation/update.rs:88`, `delete.rs:57`). Destructure once in the dispatcher and pass a `&Vindex3Binding` struct, which removes every `unreachable!`.

## 5. Hardcoding to an architecture

Literal hits (`arch_literals.txt`) in scope are all tests or docstrings (`boundary_kv/identity.rs:43-47`, `lexer.rs:693`, `executor/query/mod.rs:96-102`, `remote/*`, `python/lib.rs:733,744`). No action needed. Non-literal assumptions:

- **P1** `larql-kv/src/lib.rs:242` has `num_layers` default 34 (Gemma 3 4B); see #7.
- **P1** `larql-kv/src/generation.rs:576-590` has the hardcoded family stop-token strings; see #4.
- **P2** `larql-kv/src/engines/apollo/entry.rs:65` has `injection_layer: 30` ("Apollo 11 demo manifest", effectively Gemma 3 4B). It fails closed on smaller models (`apollo/engine.rs:87`), which is good, but a bare `apollo` spec is model-specific. Require `layer=` or derive it from the store's crystal layer.
- **P2** `larql-python/src/vindex.rs:760-782` uses a layer-band fallback of `num_layers/3` and `*5/6`, a hardcoded depth prior that disagrees with `LayerBands::for_family`. Use `resolve_bands`. `:774` `num_layers - 1` underflows when `num_layers == 0`.
- **P2** `larql-python/src/vindex.rs:862` magic score `20.0`, `:900` "garbage label" `len() > 20 && contains('/')`, `:30` token length `30`. These are model-scale-dependent. Name them as constants, or pull them from the LQL implementation.
- **OK**: sliding-window logic reads `weights.arch.is_sliding_window_layer` (`semantic_promotion/exclusion.rs:32`), PLE is driven by arch presence, and there is no fixed `head_dim` in kv.

## 6. General code review

### Correctness / panics
- **P1** `larql-kv/src/engines/turbo_quant/engine.rs:461,755,904`: decode before prefill panics (#6).
- **P1** `larql-kv/src/lib.rs:514-542`: `build` panics (#9).
- **P1** `larql-lql/src/parser/helpers.rs:409-411`: `expect_u32` does `n as u32` on an `i64`, so `LAYER 4294967296` silently becomes `0` and `LIMIT 4294967297` becomes `1`. User input silently wraps. Use `u32::try_from(n).map_err(..)`.
- **P1** `larql-lql/src/executor/query/walk.rs:33` / `explain.rs:27`: `embed.row(last_tok as usize)` panics if the tokenizer's vocab exceeds the embedding rows (tokenizer/vindex mismatch, or an added-token id). Check `last_tok < embed.nrows()` and return `LqlError`.
- **P2** `larql-lql/src/executor/query/describe/moe.rs:152`: `partial_cmp(..).unwrap()` panics on NaN. Use `total_cmp`. `:153` `dedup_by` only removes *adjacent* case-duplicates after a score sort, so duplicates survive. Dedup through a `HashSet` of lowercase keys.
- **P2** `larql-python/src/vindex.rs:880`: `partial_cmp().unwrap()` NaN → `PanicException`; use `total_cmp`. `larql-python/src/lib.rs:152-154`: `d.get_item("s")?.unwrap()` panics on a missing key; return `PyKeyError`.
- **P2** `larql-lql/src/executor/mutation/insert/knn.rs:114`: `layer_hint.min(num_layers-1)` silently clamps an out-of-range user `AT LAYER` value. Refuse it.
- **P2** `larql-python/src/trace_py.rs:397-400`: `positions` other than `"all"` silently means `Last` (a typo `"al"` gives Last). Validate the value.
- **P2** `larql-python/src/walk.rs:139-146, 234, 246`: tensors that fail a bounds check are `continue`d and silently missing, so the failure surfaces much later as a missing key. Error out.
- **P2** `larql-kv/src/generation.rs:622-632, 662-664`: an attention failure returns an empty or partial `Vec` with no error. The public `generate_*` functions can't distinguish "EOS" from "backend failed". Return `Result`.
- **P2** `larql-kv/src/lib.rs:441`: `let _ = profiling;` is dead, and the comment says profiling is ignored while the next lines use it. Remove both.

### Parser robustness (LQL is user input)
- The lexer and parser are panic-free on the paths checked: `peek` is bounds-safe (`parser/helpers.rs:354`), there is no recursion (conditions are a flat `AND` list, and pipes are a single level, so `a |> b |> c` gets a clear "trailing token" error, not a crash), and there are no raw indexing panics in `lexer.rs` beyond guarded `self.pos < len` accesses. Good.
- **P2** `lexer.rs:572-580`: an escape followed by a non-ASCII byte (`"\é"`) pushes the lead byte as a Latin-1 char, and the continuation bytes then become U+FFFD. Decode escapes on `char`s (iterate `str::char_indices`).
- **P2** `lexer.rs:516-519`: an unexpected non-ASCII char is reported as a mojibake byte, with a byte offset rather than a char or column. Also, error positions after `|` / `!` are reported *after* advancing.
- **P2** `parser/helpers.rs:354`: `peek()` clones the token (including `String`s) on every call, and `check_*` calls `peek`. Return `&Token`.
- **P2**: there is no fuzz or property test for `parse()`. Add a `proptest` / `cargo fuzz` target asserting `parse(any_string)` never panics. This is cheap and fits a user-input surface.

### Duplication / dead code
- **P1**: layer-band resolution is copy-pasted 3× in LQL (`executor/query/mod.rs:55` `resolve_bands`, `mutation/insert/knn.rs:105-112`, `mutation/insert/plan.rs:58-66`), plus a divergent 4th copy in Python. Call `resolve_bands` everywhere (and move it to `larql_vindex::LayerBands::resolve(&config)`).
- **P1**: engine loop duplication (see §1).
- **P2**: MEMIT env parsing duplicated in `into_vindex.rs` / `into_model.rs` (§4).
- **P2**: stop-token list duplicated in `larql-cli/src/commands/extraction/walk_cmd.rs:1324` and `larql-kv/src/generation.rs:576`.
- **P2** `larql-lql/src/executor/backend.rs:55-56`: `#[allow(dead_code)] path` on `Backend::Vindex3`. Remove it or use it.

### unsafe / PyO3
- **P0**: #1, #2, #3.
- **P2** `larql-python/src/walk.rs:394,444,488`: `slice::from_raw_parts(x_bytes.as_ptr() as *const f32, ..)` on Python `bytes`. The length is checked, but **alignment is not**: `PyBytes` data is not guaranteed 4-byte aligned, so this is UB on a misaligned buffer. Use `bytemuck::try_cast_slice`, or copy with `f32::from_le_bytes` chunks. The output direction (`:402`) is fine.
- **P1**: no GIL release (#11).
- **P2**: all stateful classes are `unsendable`, so any cross-thread access from Python raises. That is acceptable, but document it in the Python stubs.

### Public API hygiene
- **P2** `larql-kv/src/lib.rs:19-41`: exposes `accuracy`, `accuracy_suite`, `profiler`, `vindex_compare`, `model_walk` and every engine module as `pub`, plus `pub use` of each engine module. That is a very wide semver surface for a release. Feature-gate the measurement modules and consider `#[doc(hidden)]` on internals.
- **P2** `larql-lql/src/lib.rs:4-8`: `pub mod executor`, `relations` and `repl` expose internals (for example `executor::Backend` is `pub(crate)`, but the module tree is public). Re-export only `Session`, `parse`, `Statement`, `LqlError`, `CapabilityProfile` and `run_*`.
- **P2** `EngineKind::from_name` returns `Option`, so callers can't say *why* a spec failed (#8).
