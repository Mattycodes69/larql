# Pre-release review — larql-models, larql-factory, larql-core, larql-boundary, larql-execution, larql-experts, model-compute

Read-only review of the working tree as it stands, including the uncommitted qkv-attention-bias diff in larql-models. All paths are relative to `crates/`. Severity: P0 = must fix before release, P1 = should fix, P2 = cleanup.

---

## 0. larql-models abstraction verdict (the headline question)

**How the abstraction is built: mostly right, but too big, and dispatch is split across several places.**

- `ModelArchitecture` (`larql-models/src/config/architecture.rs:63`) is a real trait. It has defaults, families override them, and per-family facts such as `default_norm_eps`, `qkv_bias` and `rope_position_divisor_for_layer` are trait defaults that families override. That matches the project rule "config fact → trait default". `architectures/*.rs` holds 27 families, all under 510 lines. Inventory (`inventory/*`) is clean: no family string compares, and it goes through `find_architecture` and `arch.family()`.
- **It is a god-trait:** about 200 methods and 1,920 lines in one `trait` block (lines 63–1980). It covers tensor keys, norms, attention, RoPE and position, PLE, MoE (about 50 methods), MLA, DSA, multimodal and KV. Every consumer depends on all of it.
- **It is not "one file + one registration".** Adding a family today takes up to six edits:
  1. `architectures/<x>.rs`
  2. `architectures/mod.rs`
  3. the `use` list plus a match arm in `detect/mod.rs:127-224` (`detect_from_json`, a 25-arm `match` of `starts_with` guards, order-sensitive)
  4. a row in `detect/registry/table.rs` (`ARCHITECTURE_REGISTRY`, a **second, parallel source of truth**; its own doc at `detect/registry/mod.rs:4-14` admits CI cannot catch a missing row)
  5. a `lib.rs` re-export
  6. for GGUF, the alias `match` in `loading/gguf/loader.rs:236-247`, plus family branches in `to_config_json` and `normalize_gguf_key_for_arch`.
- **Where family facts leak out of `architectures/`:**
  - `detect/parser.rs` (generic config parser): `is_gemma` rope default and head_dim 256 (`:132-137`, `:200`), gpt2 4× intermediate (`:153`), crate-wide `DEFAULT_NUM_ATTENTION_HEADS=8` / `DEFAULT_NUM_KV_HEADS=4` (`:31-34`, applied at `:206`, `:231`).
  - `detect/config_io.rs:124` gpt2 / mamba2 special-casing.
  - `loading/gguf/loader.rs`: the gemma4 block `:252-367` (hardcoded head_dim 256 and `partial_rotary_factor 0.25`), `:604` gemma norm layout.
  - `connectors/projector.rs:29,74,80` (Gemma-3 `mm_input_projection_weight` names) and `encoders/vision_tower.rs:31` (SigLIP prefix) sit in generic folders.
- **Dependencies:** clean apart from one smell. `larql-vindex-spec` is pulled in only for `QuantFormat` in `detect/registry/entry.rs:4`, so a model-description crate is advertising vindex storage formats (see §1).

**Recommended target shape:**
- `ArchitectureEntry { patterns, construct: fn(ModelConfig, &Value) -> Box<dyn ModelArchitecture>, gguf_arch_aliases: &[&str], ... }` in one table, with `detect_from_json = registry.iter().find(matches).construct(..)` and the `match` deleted.
- Per-family config-parse defaults and GGUF translation hooks become trait or associated fns: `fn gguf_config_overrides(meta, &mut Value)`, `fn parse_defaults() -> ParseDefaults`.

---

## 1. Decoupling / layering

| Sev | Location | Finding | Fix |
|---|---|---|---|
| P1 | `model-compute/` (whole crate) | **Dead crate.** Nothing in the workspace depends on it (no Cargo.toml, `.rs` or CI reference outside its own dir), yet it sits in `members` **and** `default-members` (`Cargo.toml:28,48`), so every build compiles it. Its optional wasmtime host (`src/wasm/runtime.rs`, `session.rs`) duplicates the wasmtime host in `larql-inference/src/experts` (`larql-inference/Cargo.toml:78`), with a different ABI. | Delete it, or move it out of the workspace to the promised sibling repo. At minimum drop it from `default-members`. |
| P1 | `larql-experts/` | Nested standalone workspace (own `Cargo.lock`, 19 guest crates). Not a workspace member and **never built in CI** (no `.github/workflows` reference). Its only consumers, `larql-inference/tests/test_expert_dispatch.rs:17-23` and `test_llm_dispatch.rs` / `test_trie_dispatch.rs`, `eprintln!("skip: …")` and **pass** when the wasm dir is missing. That is a false green and breaks the "tests must not skip" rule. It has 0 `#[test]` of its own. | Add a CI job: `cargo build --target wasm32-wasip1 --release` in `larql-experts` followed by the dispatch tests, and make a missing dir a hard failure under `CI=1`. |
| OK | `larql-execution/` | 200-line contract leaf (`RefusalKind`, `ExecutionRefusal`) with no deps. Used by compute, kv, inference and vindex. It is justified as a cycle-breaker. | Keep it. Its inline `#[cfg(test)]` (`lib.rs:124`) should move to `tests/`. |
| OK | `larql-boundary/` | Leaf crate, used by larql-kv and larql-demos. Correctly separate. | — |
| P2 | `larql-models/src/detect/registry/entry.rs:4`, `table.rs` `quant_formats` | The model-description crate depends on `larql-vindex-spec` only to say which vindex quant formats each architecture supports. That is a vindex/factory capability fact. | Move the `quant_formats` column to `larql-factory::capabilities` (or vindex) and drop the dependency. |
| P2 | `larql-models/src/quant/ggml/q4_k.rs`, `q6_k.rs` (`q4k_row_dot`, `q*_row_scaled_add`) | Compute kernels (dot / axpy) live in the model-description crate and are consumed by `larql-compute/src/cpu/ops/q4k_q8k_dot/q4k_asm.rs` and `larql-vindex/src/quant/registry.rs`. | Keep dequant (a format fact) here. Move the row-dot/axpy kernels to larql-compute. |
| P2 | `larql-models/src/loading/{gguf/loader.rs:465-548, safetensors/mod.rs:358-417}` | Two loaders each hand-assemble `ModelWeights` (embed lookup, lm_head policy, vocab resolution, 20-field literal). They have **already diverged** (see §6 P1 on tied lm_head). | One `assemble_model_weights(arch, tensors, vectors, raw, …)` used by both. |

---

## 2. Modularity / file size (>800 lines)

| Sev | File (lines) | Proposed split |
|---|---|---|
| P1 | `larql-models/src/config/architecture.rs` (2119) | Turn the god-trait into supertraits, one file each under `config/architecture/`: `identity.rs` (family/config/validate), `tensor_keys.rs` (attn/ffn/norm key fns, ~250), `norms.rs` (norm type/eps/offsets/specs), `attention.rs` (scale, softcap, sinks, gate, sliding, KV sharing), `position.rs` (rope/yarn/llama3/linear, position policy), `ple.rs`, `moe.rs` (~450, expert keys, router, shared expert, latent MoE), `mla_dsa.rs`, `multimodal.rs`. `pub trait ModelArchitecture: TensorKeys + Norms + … {}`. Consumers take the narrow bound. Low-risk alternative: keep one trait but move default bodies into free fns in these files. |
| P1 | `larql-models/src/loading/gguf/loader.rs` (1835) | Move tests (`:617-1835`) to `loading/gguf/tests/loader.rs`, leaving about 617 lines. Extract `to_config_json` (`:196-418`) into `gguf/config_translate.rs`, with the gemma4 and MLA sections as per-family hooks. |
| P1 | `larql-models/src/test_fixtures.rs` (1379) | Split into `test_fixtures/{dense,moe,mla,gguf,multimodal}.rs`. |
| P1 | `larql-models/src/quant/ggml/mod.rs` (1211) | Tests from `:330` go to `quant/ggml/tests.rs`, leaving about 330. |
| P1 | `larql-models/src/config/tests.rs` (1035) | Move to `tests/` or split by topic (`config/tests/{rope,moe,norm}.rs`). |
| P1 | `larql-models/src/quant/ggml/tq.rs` (975) | Tests from `:542` to a sibling file. |
| P1 | `larql-models/src/inventory/report.rs` (956) | Tests from `:787` out. Split the report types from the render/serialise code. |
| P1 | `larql-models/src/detect/parser.rs` (874, no tests) | Split into `detect/parser/{topology.rs, attention.rs, rope.rs, moe.rs, mla.rs, mod.rs}`. `parse_model_config` is a single ~700-line fn (`:113-~870`). |
| P1 | `larql-models/src/config/position.rs` (853) | Tests from `:468` out. |
| P1 | `larql-models/src/loading/gguf/orient.rs` (831) | Tests from `:272` out, including `synth_gpt2_config`. |
| P1 | `larql-models/src/encoders/vision_tower.rs` (818) | Tests from `:339` out. |
| P1 | `larql-models/tests/test_architectures.rs` (2154), `tests/test_loading.rs` (1538) | Split per family (`tests/architectures/<family>.rs`) and per format (`tests/loading/{gguf,safetensors,mxfp4}.rs`). |
| P1 | `larql-models/src/config/model_config.rs` `ModelConfig` (about 120 pub fields, `#[derive(Debug, Clone)]` only, no `Default`) | The uncommitted `qkv_bias` field forced edits to every struct literal: `gemma3.rs:263`, `granite.rs:312`, `orient.rs:348`, `orient.rs:643`, `larql-server/tests/test_expert_endpoint.rs`, plus larql-vindex fixtures. Derive `Default` (or add a test builder) and use `..Default::default()` in fixtures. |

---

## 3. File / folder structure

| Sev | Location | Finding | Fix |
|---|---|---|---|
| P1 | `larql-models/src/**` | 66 source files carry inline `#[cfg(test)]` modules, against the "tests in tests/" rule. The good pattern already exists (`detect/tests/`, `inventory/tests/`, `config/interleave/tests.rs`). | Mechanically move them to `#[cfg(test)] #[path] mod tests;` sibling dirs, or to crate `tests/` when they only use the pub API. |
| P2 | `larql-models/src/config/{mamba2_tests.rs, kda_geometry_tests.rs, mla_geometry_tests.rs}`, `quant/ggml/type_id_conformance_tests.rs` | `*_tests.rs` files are loose among the source files. | Put them under `config/tests/`. |
| P2 | `larql-core/src`, `larql-boundary/src`, `larql-factory/src` (57 files with `#[cfg(test)]` across these + model-compute); `larql-factory/src/{build,validate}/tests.rs`, `test_support.rs` | Tests live in `src/`. | Move them to `tests/`. |
| P2 | `larql-models/src/connectors/projector.rs`, `encoders/vision_tower.rs` | Family-specific weight layouts (Gemma-3 projector, SigLIP) sit under generic folder names. | Rename to `connectors/gemma3_projector.rs` / `encoders/siglip.rs`, or have the architecture supply the key names. |

---

## 4. Hardcoding to a module / backend

No Metal, CUDA or ANE references in scope. Clean.

| Sev | Location | Finding | Fix |
|---|---|---|---|
| P1 | `larql-factory/src/estimate/preset_weights.rs:12-25` | The slice-preset vocabulary (`full`/`client`/`server`/`expert-server`/…) is a **third copy** (also `larql-cli/src/commands/primary/slice_cmd.rs:135-141` and `publish_cmd/collections.rs:137`). An unknown preset silently estimates **0 bytes** (`_ => &[]`). | One `SlicePreset` enum in `larql-vindex-spec` with `FromStr` returning an error, used by all three. |

---

## 5. Architecture hardcoding outside the architecture layer

(The literal hits in `arch_literals.txt` for these crates are almost all tests or docs. The real leaks are below.)

| Sev | Location | Finding | Fix |
|---|---|---|---|
| P1 | `larql-models/src/loading/gguf/loader.rs:236-247` | GGUF→HF `model_type` alias `match` sits in the loader. `phi*`→`phi` goes to Generic. `gemma` (Gemma 1) maps to `Gemma2Arch` in `detect/mod.rs:130`, while `:597-601` says Gemma 1 has the llama norm layout. That inconsistency needs verifying. | Put `gguf_arch_aliases` on the registry entry. |
| P1 | `larql-models/src/loading/gguf/loader.rs:252-367` | A whole `if arch == "gemma4"` block: `GEMMA4_GGUF_HEAD_DIM=256` (`constants.rs:62`) **overrides** the file's `key_length`, `partial_rotary_factor = 0.25` is hardcoded (`:338`), plus the `*_swa` keys. | Add a `fn gguf_config_overrides(&GgufMeta, &mut Value)` hook on the Gemma4 architecture module. |
| P1 | `larql-models/src/loading/gguf/loader.rs:603-616` | `normalize_gguf_key_for_arch` uses `matches!(arch, "gemma2" \| "gemma3") \|\| starts_with("gemma4")`. | Put `gguf_key_replacements()` on the registry entry. |
| P1 | `larql-models/src/detect/parser.rs:31-43,132-137,153,200-231` | The generic parser picks defaults by `model_type.starts_with("gemma")`: rope 1e6 (`defaults.rs:13`, wrong for Gemma 1/2, which use 10000 when a config omits it), head_dim 256, and gpt2 `4*hidden`. **Every** non-SSM config missing `num_attention_heads` or `num_key_value_heads` silently gets **8 / 4**: an assumed default, not a checked one. | Return `None`/0 and let `validate()` refuse, or move the defaults to per-family `parse_defaults()`. |
| P2 | `larql-models/src/loading/gguf/constants.rs:61` `DEFAULT_GGUF_VOCAB_SIZE = 262_144` | Gemma's vocab is used as the crate-wide fallback (`loader.rs:523`). | Error instead (`MissingConfig("vocab_size")`); the embed-shape fallback already covers real files. |
| P2 | `larql-models/src/architectures/qwen.rs:45` (uncommitted) | `QKV_BIAS_FAMILIES` uses exact `contains`, while its sibling `stores_norm_weight_as_offset` (`:34-37`) uses `starts_with`. Qwen2-lineage text sub-types (`qwen2_vl`, `qwen2_5_vl` text configs) will fall through to `None`. Also, when `attention_bias: false` is declared, the result is `None` rather than `Some(false)`. The design itself (family fact as trait override, declaration wins) is correct. | Decide exact vs prefix deliberately, and add a test for `qwen2_5_vl`. |
| P2 | `larql-models/src/connectors/projector.rs:29,74,80`; `encoders/vision_tower.rs:31,49-72` | Gemma-3 projector and SigLIP keys are hardcoded, and `#[serde(default = …)]` silently defaults `num_channels`, `layer_norm_eps`, `hidden_act` and `norm_type`. | Architecture-supplied keys, and explicit `Option` fields with checked defaults. |

---

## 6. General code review

### Untrusted input (GGUF / safetensors / config)

| Sev | Location | Finding | Fix |
|---|---|---|---|
| **P0** | `larql-models/src/loading/gguf/reader.rs:62-64` | `read_string`: `let len = read_u64(r)? as usize; vec![0u8; len]`. A crafted GGUF with `len = 2^63` **aborts the process** (capacity overflow / OOM) before `read_exact` can fail. This is reachable from `larql-server` and the CLI on any user-supplied `.gguf`. | Cap `len` (e.g. ≤ remaining file bytes, or 1 MiB for keys) and use `try_reserve`. Error on overflow. |
| **P0** | `larql-models/src/loading/gguf/reader.rs:86-87`, `parser.rs:245-263` | `Vec::with_capacity(len)` for metadata arrays, `n_tensors` and `n_dims`, all taken from untrusted u64/u32. Same abort. | Bound by remaining bytes / a sane maximum before reserving. |
| P1 | `larql-models/src/loading/gguf/loader.rs:124-127` | `info.dims.iter().product()` on u64 panics in debug and wraps in release. `n_elements as usize` truncates on 32-bit/wasm (planner-core wasm builds). | `checked_mul` fold, then `usize::try_from`. |
| P1 | `larql-models/src/quant/ggml/mod.rs:212-216` | `tensor_data_size`: `n_elements * 4` and `* 2` are unchecked. Block types floor-divide, so a non-multiple count under-reports the size. | Use `checked_mul`, and require `is_multiple_of(block)` as I2_S already does. |
| P1 | `larql-models/src/loading/gguf/parser.rs:275-277` | Data alignment is hardcoded to 32. `general.alignment` metadata is ignored, although GGUF allows other values. A file with a different alignment reads garbage offsets. | Read `general.alignment` (default 32) and validate it is a power of two. |
| P1 | `larql-models/src/loading/gguf/loader.rs:180-189` and `safetensors/mod.rs:348` | Tensors of rank ≥ 3 fall through `_ => {}` and are **silently dropped, not recorded in `skipped_tensors`**. On GGUF this includes every MoE `*_exps` expert bank (there is no `_exps` handling anywhere in `loading/gguf`), so a DeepSeek/Kimi GGUF (whose config the loader explicitly translates) loads with no experts. `skipped_tensors` is always `Vec::new()` on the GGUF path (`:529`). | Push rank ≥ 3 to `skipped_tensors` (or error), and populate it on GGUF. |
| P1 | `larql-models/src/loading/gguf/loader.rs:196-248` | `to_config_json` silently turns missing metadata into 0 or "": `unwrap_or("")` arch, `get_arch_u32 → 0` for hidden/layers/heads. `load_gguf` / `load_gguf_keep_quant` use the **unvalidated** `detect_from_json` (`:486`), so a header-less GGUF becomes a Generic arch with 0 dims. | Validate on every public entry, or error when `general.architecture` or `embedding_length` is absent. |
| P1 | `larql-models/src/loading/gguf/loader.rs:506-510` | GGUF lm_head silently ties to the embedding when `output.weight` is missing, **even when the config declares untied**. The safetensors path refuses this (`safetensors/mod.rs:371-386`, "GPT-OSS and OLMoE both declare false"). The two loaders have diverged. | Share the lm_head policy (see the §1 shared assembly). |
| P2 | `larql-models/src/loading/safetensors/mod.rs:139-150` | A directory containing any `.gguf` silently loads the **largest** one, even when safetensors are also present or an `mmproj-*.gguf` sits beside it. | Error on ambiguity, or prefer an explicit file. |
| P2 | `larql-models/src/detect/parser.rs` (throughout, e.g. `:589` new `qkv_bias`) | `text_config["k"].as_bool()` / `as_f64()` treats a mistyped declaration (`"true"` as a string) exactly like an absent one. That is a checked-default violation. | Add a `declared::<T>(v, key) -> Result<Option<T>>` helper that errors on the wrong type. |
| P2 | `larql-models/src/detect/parser.rs:161-165` | `ffn_intermediate_size_by_layer` uses `filter_map(as_u64)`, silently dropping non-integer entries, so the length check downstream sees a shorter array. | Error on a non-u64 element. |
| P2 | `larql-core/src/io/packed.rs:85-87` | String-table `len` (u32) is allocated before bounds checking: up to 4 GiB per string on a crafted file. The rest of the deserialiser is well hardened. | Check `len <= remaining` before allocating. |
| P2 | `larql-boundary/src/codec/int8.rs:61-66`, `codec/bf16.rs:27-30` | Wire decoders `assert!`/panic on short or odd-length payloads. `larql-kv/src/engines/markov_residual_codec/codec.rs:76` decodes stored bytes through this. | Return `Result<_, CodecError>`. |
| P2 | `larql-factory/src/estimate/dims.rs:34` | `vocab_size.unwrap_or(0)`: a missing vocab silently estimates embed/lm_head at 0 bytes (documented, but still an assumed default). The `detect_from_json` there is unvalidated. | Report "unknown" instead of 0. |

### Dead code / duplication / hygiene

- **P1** `larql-models/src/detect/mod.rs:127-224` vs `detect/registry/table.rs`: dual dispatch tables (see §0). Collapse them into one table carrying constructors.
- **P2** `larql-models/src/loading/gguf/loader.rs:205` `_get_u32` is an unused closure. Delete it.
- **P2** `larql-models/src/loading/gguf/loader.rs:94-100` and `:196-201` read `general.architecture` twice with the same `unwrap_or("")`. Make it one accessor that errors when absent.
- **P2** `larql-models/src/lib.rs:36-60` re-exports about 25 concrete `*Arch` structs as public API. Consumers should use `Box<dyn ModelArchitecture>`. Narrowing this would let families be added without widening the pub surface.
- **P2** `unsafe { memmap2::Mmap::map }` at `loading/gguf/loader.rs:110` and `safetensors/mod.rs:233` has no `// SAFETY:` note about external file mutation.

### Uncommitted qkv-bias diff (larql-models)

The design is correct: a new `ModelConfig.qkv_bias`, a trait default that reads the declaration, and a Qwen override that answers the family fact only when the checkpoint is silent. Tests are in `detect/tests/qwen_qkv_bias.rs`, `inventory/config_keys.rs:+1` and `report.rs:+6`. Only the §5 P2 on exact-vs-prefix matching applies, plus the fixture churn caused by `ModelConfig` having no `Default` (§2).
