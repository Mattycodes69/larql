# Pre-release review: `crates/larql-vindex/src/format/vindex3/`

Scope: 677 `.rs` files, ~241.7k lines. About 132k of those lines (55%) are test code inside `src/`: 87 sibling `*_tests.rs`/`tests.rs`/`tests_*.rs` files (~35k) plus 23 `tests/` subdirectories (~95k).
48 files exceed 800 lines. 28 are source, 20 are test/fixture.
Paths below are relative to `crates/larql-vindex/src/format/vindex3/`. Severity: P0 = release blocker, P1 = should fix, P2 = nice to have.

---

## 1. Decoupling / layering

| # | Location | Sev | Finding | Fix |
|---|---|---|---|---|
| D1 | `opplan/exec/` (≈60 modules + `cpu/`, `weights/`, `routed_experts/`) | P1 | A complete execution engine lives under `format/`. It includes CPU backends, decode sessions, KV state, prefetch, Metal dispatch, and NEON/SDOT integer kernels (`cpu/integer.rs`, `cpu/nvfp4_q8.rs`, `cpu/stationary.rs`, 6.5k lines in `cpu/`). This inverts AGENTS.md's layering: kernels belong in `larql-compute`, and execution belongs in a runtime crate, not the format crate. | Move `opplan/exec` to a sibling top-level `larql-vindex::exec` module now. Move the `cpu/` kernels into `larql-compute`. Keep `format::vindex3` limited to schema, read/write, and plan. |
| D2 | `represent/codec/codecs/{float,fp8_block,kquant,nvfp4}.rs`, `represent/codec/residency.rs` import `opplan::exec::cpu::physical::PhysicalProjectionPlan`; `codecs/{float,bf16_zlib}.rs` import `opplan::exec::operands::widen` | P1 | The codec (storage) layer depends on the CPU executor. | Move `PhysicalProjectionPlan` and `widen` into a leaf module (e.g. `represent/codec/physical_plan.rs`, `format/dtype.rs`). Exec then imports from codec, never the reverse. |
| D3 | Module cycles: `opplan ⇄ represent` (16 files each way), `graph ⇄ opplan` (`graph/roles.rs:808` doc link), `graph → represent` (`graph/build.rs:1052` pulls `DTYPE_MXFP4` from codec), `encode ⇄ opplan` | P1 | Circular module dependencies between format sub-layers. The intended order is graph → plan → opplan → represent/exec. | Move dtype string constants (`DTYPE_MXFP4`, `FloatDtype`) to a `format::dtype` leaf. Break `opplan/planned.rs:28-30`'s use of `represent::codec::{RepresentationExtent, RequiredAccess}` by moving those types down. |
| D4 | `opplan/exec/operands.rs:62,74,125,134,464-486` | P1 | `OperandStore` reaches directly into `represent::map`, `represent::plan_roles`, and `representation_attestations::{AttestationTable, recognition}` using fully qualified paths. It is a god-object that knows every sibling. | Inject a `RepresentationContext` trait/struct built by `represent` and passed into `OperandStore::open`. |
| D5 | `opplan/exec/kimi_source.rs:121-180` | P1 | Re-parses `system_graph.json` as an untyped `serde_json::Value` (`comp["execution"]["kda"]`, `moe["routing_policy"].as_str().unwrap_or("")`) and bypasses the typed `graph::` model. | Deserialize through `graph::Component` / `ExecutionSurface`, the same way `plan_component_ops` does. |
| D6 | `opplan/exec/stack_metal.rs:36-42,518`, `kimi_source.rs:29-33,390,613`, `kda_metal.rs:49-51`, `represent/measure/teacher_forced.rs:37` | P1 | Code in the format crate names concrete `larql_compute_metal::trait_impl::kimi_layer::*` types and calls `metal.kimi_decoder_layers_with_head(...)`. That bypasses the `PlanBackend` / `ComputeBackend` trait. | Hide these behind a `PlanBackend` capability (e.g. `fused_decoder_layers`), or move them into `larql-compute-metal` / a runtime crate. |
| D7 | Process-global env knobs read in library hot paths: `opplan/exec/mod.rs:2083-2110` (`LARQL_FFN_MULTI_POSITION` plus a mutable global setter `set_multi_position_ffn`), `cpu/integer.rs:128,209,227,1010,1075`, `cpu/physical.rs:239,268,687,758`, `cpu/stationary.rs:111`, `cpu/executor.rs:398`, `weights/staged.rs:99,112,189`, `represent/measure/teacher_forced.rs:446` | P1 | Behaviour changes silently with the environment, is cached in statics, and races between tests (see memory note "process-global test instruments race"). This is configuration by side channel. | Add one `ExecConfig` struct, parsed once at the CLI/server edge and threaded through `PreparedOperands` / the backends. Delete `set_multi_position_ffn`. |
| D8 | `represent/physical.rs:46-60`, `opplan/exec/weights/mod.rs:39` (`DEVICE_PAGE_ALIGN = 16384`) | P2 | Metal constants are copied by hand (`NOT_RESIDENT`, `WEIGHT_BINDING_ALIGN`, `PAGE_SIZE`). `kimi_source.rs:613` uses the real `larql_compute_metal::buffers::PAGE_SIZE`, so there are two sources of truth. | Put the binding contract constants in the `larql-execution` / `larql-vindex-spec` leaf and have both sides import them. |

## 2. Modularity / file size (>800 lines)

There are 28 oversized source files in scope. Split proposals for all of them follow, largest first.

| File (lines) | Proposed split |
|---|---|
| `opplan/exec/prepared.rs` (3770; tests from 2818) | `prepared/residual.rs`: PreparedNorm, PreparedHc*, PreparedAttnRes* (255-702). `prepared/mixers.rs`: GatedDelta/Mamba2/ConvQkv/Kda/Mla operands (845-1480). `prepared/selection.rs`: `select_realizations*`, budget_refusal, select_records, dependency_pins, pins, BankPin (1484-2036). `prepared/head.rs`: SelectedOutputHead/SelectedProjection (2077-2280). `prepared/operands.rs`: `impl PreparedOperands`, itself about 1380 lines, split again into load/traverse/dense-image. `prepared/census.rs`: AllocationCensus, ResidencyCensus (3662+). Move the inline test module (~950 lines) to `tests/prepared.rs`. |
| `opplan/build.rs` (3437; tests from 2790) | `plan_component_ops` is a single ~1370-line function (204-1576). Split it per op family: `build/attention.rs`, `build/ffn.rs` (dense/routed/latent, 1100-1330), `build/mixer.rs` (Mamba2/KDA/MLA branches), `build/closure.rs` (hc_head/attn_res_exit closure, 1576-1730), `build/roles.rs` (LayerOps, `required_roles`, `absent_op`, 1754-2320), `build/shapes.rs` (StackGeometry, `expected_shape`, packed/scales shapes, 2323-2790), `build/messages.rs` (the 20 refusal-string consts, 51-180). Move inline tests (~650 lines) out. |
| `plan/carriage.rs` (2582) | `carriage/rules.rs`: `CARRIAGE_RULES` data table (195-1352, ~1150 lines). Consider splitting it per cluster (rope/moe/mamba/hc). `carriage/probes/{rope,moe,mamba2,conv_qkv,hc,layers}.rs`: the ~60 `probe_*` fns (1400-2582). `carriage/mod.rs`: Carriage, CarriageRule, ProbeContext, CompanionGate, `rule_for`, `canonical_declared`. |
| `opplan/exec/mod.rs` (2390) | mod.rs should only declare modules. `exec/trace.rs`: Plane, LayerTrace, FinalState, ExecutionTrace, PlaneEvent (125-410). `exec/entry.rs`: `execute_*` / `prefill_*` (414-800). `exec/traverse.rs`: `traverse` + `execute_layer` (802-1690, 900 lines, needs a further split into per-sublayer helpers). `exec/batch_site.rs`: BatchSite* (1691-2080). `exec/attention_operands.rs`: AttentionOperands/AttentionBiases/project (2119-2390). The FFN-shape knob moves to ExecConfig (D7). |
| `opplan/exec/cpu/integer.rs` (1618; tests from 703) | `integer/activation.rs` (quantise_activation*, env knobs). `integer/asym.rs` (portable asym rows). `integer/sdot_neon.rs` (all `unsafe` NEON fns). `integer/k3k4.rs` (k3/k4 register variants, 704-1010). Move tests out. |
| `opplan/exec/decode.rs` (1597) | `impl DecodeSession` is 198-1243. Split into `decode/session.rs` (ctor/state), `decode/step.rs` (per-token step), `decode/site.rs` (enter/leave site, boundary_event, attention-residual site, 1243-1597), `decode/slots.rs` (OperandsSlot/KvSlot/Entry/Carrier). |
| `opplan/exec/production.rs` (1576; tests from 1037) | `production/attention_conditioning.rs` (qk_norm, condition_qk/v, biases, 254-530). `production/routing.rs` (router_input, select_experts, sigmoid_select, expert_inner, 531-700). `production/backend.rs` (ProductionBackend + PlanBackend impl). Move tests out. |
| `opplan/exec/operands.rs` (1565) | `operands/store.rs` (OperandStore open/resolve, the ~900-line impl at 256-1150). `operands/auxiliary.rs` (AuxiliaryExtents, attach_auxiliaries). `operands/overrides.rs` (OperandEdit/OperandOverrides, 1240-1430). `operands/source.rs` (OperandSource/SourceStamp). `widen` goes to a dtype leaf (D2). |
| `plan/semantics.rs` (1325) | `semantics/keys.rs` (the key tables, 17-800, pure data). `semantics/clusters.rs` (SemanticCluster + CLUSTER_KEYS, 913-1250). `semantics/classify.rs` (classify_key*, inert_at_value, component_of/leaf_of). |
| `opplan/exec/backend.rs` (1309) | `backend/format.rs` (WeightFormat, activations, MatrixClass, WeightFormats). `backend/slice.rs` (WeightSlice, 252-600). `backend/calls.rs` (all *Call / *Out structs, 600-906). `backend/trait.rs` (PlanBackend + Arc/Debug impls). |
| `opplan/exec/experts.rs` (1258) | `experts/latent.rs` (enter/exit_latent, LatentOperands). `experts/dense.rs` (`impl DenseOperands`). `experts/routed.rs` (`impl RoutedOperands`, 636-1048). `experts/load.rs` (load_packed, from_f32, bank_facts). |
| `graph/roles.rs` (1208) | `roles/operand_role.rs` (enum, 24-425). `roles/tables.rs` (ROLE_TABLE and the KDA/MLA/Mamba2/ConvQkv tables). `roles/residual.rs` (attn-res and HC head classifiers, 714-870). `roles/classify.rs` (classify_stack_tensor*, norm-placement evidence). |
| `graph/build.rs` (1194) | `build/groups.rs` (GroupClass, modality, classify_group). `build/declared.rs` (declares_*, recurrence_kind, uses_mla, declared_taps). `build/attention_table.rs` (781+). Keep `build_from_inventories` in mod. |
| `opplan/exec/kimi_source.rs` (1160) | Rename it to a family-neutral name (see A2). Split into `source/geometry.rs`, `source/model.rs`, and `source/overlay.rs`. |
| `opplan/exec/accounting.rs` (1151) | `accounting/residency.rs` (resident_profile, requantised bytes, Expectation). `accounting/budget.rs` (ResidencyBudget, RepresentationFloor, Throughput, Deficit). `accounting/ledger.rs` (Resources, PrepareReads, ResourceLedger). `accounting/sysmem.rs` (`physical_memory_bytes` and its unsafe libc/win32). |
| `plan/mod.rs` (1108) | Move the `*_findings` fns (237-1080) into `plan/findings/{identity,config_keys,carriage,placement,surface}.rs`. Keep `plan_system*` in mod.rs. |
| `opplan/exec/weights/mod.rs` (1107) | `weights/aligned.rs` (AlignedBytes). `weights/loaded.rs` (LoadedWeight). `weights/load.rs` (`load_weight`, `load_fp8_block`, conformance). `weights/quantize.rs` (quantize_mxfp4/nvfp4, nearest code). |
| `opplan/exec/reference.rs` (1088) | `reference/routing.rs` (select_experts_* refs, 598-720). `reference/backend.rs`. |
| `graph/surface.rs` (1069) | Split `surface_from_nested` and the head/norm evidence helpers from the ExecutionSurface types. |
| `opplan/exec/realization.rs` (1060), `represent/mod.rs` (1048), `opplan/mod.rs` (1037) | `opplan/mod.rs`: move the op structs into `opplan/ops/{attention,ffn,layer}.rs`. `represent/mod.rs`: move `compile_inner` and its encode helpers into `represent/compile_driver.rs`, keeping mod.rs as declarations and re-exports. |
| `represent/measure/teacher_forced.rs` (1012), `represent/physical.rs` (913), `plan/report.rs` (906), `represent/quality.rs` (870), `opplan/exec/kda.rs` (836), `opplan/exec/cpu/physical.rs` (806) | Near the limit. `quality.rs`: move the Kimi gate constructors into data/config (A3), which removes about 200 lines. `teacher_forced.rs`: move it out of lib (G3). The rest need one extract each. |

Test/fixture files over 800 lines (20). These move with S1 below, split per topic: `represent/tests.rs` (2171), `opplan/exec/tests/kimi_layer_metal.rs` (1254), `cpu/tests/integer.rs` (1248), `plan/tests/carriage.rs` (1152), `tests/coverage_device.rs` (1116), `tests/attn_res_2b_batch.rs`, `represent/ingest/tests.rs`, `opplan/tests/wave18_hc_carriage.rs`, `tests/kda_q8_real.rs`, `encode/tests.rs`, `represent/compiler_tests.rs`, `calibration/tests.rs`, `fixtures.rs` (927), `tests/wave19_hc_decode.rs`, `tests/accounting.rs`, `tests/kimi_two_layer.rs`, `plan/tests_support.rs`, `tests/head_observation.rs`, `represent/state/fixtures.rs`, `tests/realization.rs`.

## 3. File / folder structure

| # | Location | Sev | Finding | Fix |
|---|---|---|---|---|
| S1 | All of scope | P1 | Tests make up about 55% of `src/`: 87 sibling `*_tests.rs` files and 23 `tests/` subdirectories. This conflicts with the project rule that tests live in `tests/` folders. `represent/` alone has about 40 flat `foo.rs` + `foo_tests.rs` pairs. | Move black-box tests to `crates/larql-vindex/tests/vindex3/`. Keep a white-box test in `src/` only when it needs private items, and then as a `tests/` subdir per module, never a flat sibling. |
| S2 | `mod.rs:49-52,72` (`fixtures`, `fixtures_kimi`, `fixtures_qwen`, `fixtures_routed`, `test_support`; ~2.4k lines) | P1 | Test fixtures are unconditional `pub mod`s and ship in the release library. The crate already has a `test-utils` feature. | Gate them with `#[cfg(any(test, feature = "test-utils"))]` and group them under `vindex3/fixtures/{mod,kimi,qwen,routed}.rs`. |
| S3 | `opplan/exec/token.rs`, `stack.rs`, `kda_metal.rs` | P1 | These are `pub mod`s whose only callers are tests (`kda_metal` has none outside `tests/kda_metal.rs`; `token::embed` only from `tests/token_*.rs`). `token::embed` panics on an unknown id (`token.rs:62`). | Move them to `#[cfg(test)]` or delete them. At minimum, make `embed` return a `Result`. |
| S4 | mod.rs files with large logic: `opplan/exec/mod.rs` (2390), `plan/mod.rs` (1108), `opplan/exec/weights/mod.rs` (1107), `represent/mod.rs` (1048), `opplan/mod.rs` (1037), `represent/measure/plan/mod.rs` (625), `represent/state/propose/mod.rs` (600), `encode/mod.rs` (581) | P1 | mod.rs used as a god-module. | Follow the splits in section 2. |
| S5 | `represent/kda_candidate.rs:376` uses `#[path = "kda_candidate_real.rs"]`; `represent/measure/teacher_forced.rs` is a lib module but is really a real-model experiment driver | P2 | `#[path]` indirection hides test files. | Rename to `represent/kda_candidate/tests/real.rs`, or move to `tests/`. |
| S6 | `opplan/exec/` has 62 files in one flat directory | P2 | Concerns are mixed at one level: `continuation_*` (5 files), `observe_*` (4), `intervene*` (2), `kimi_*` (5). | Add sub-directories: `exec/continuation/`, `exec/observe/`, `exec/intervene/`, `exec/mixers/{kda,mla,mamba2,gated_delta,conv_qkv}`. |
| S7 | `representation_attestations/` sits beside `represent/` | P2 | Confusingly close names; the executor imports both. | Move it under `represent/attestations/`. |

## 4. Hardcoding to a module / backend

| # | Location | Sev | Finding | Fix |
|---|---|---|---|---|
| M1 | `opplan/exec/stack_metal.rs:518` `metal.kimi_decoder_layers_with_head`; `kimi_source.rs` `MetalBackend` concrete | P1 | Concrete `MetalBackend` type instead of the backend trait (see D6). | Add a trait capability, or move this to the runtime crate. |
| M2 | `represent/codec/*` → `exec::cpu::physical::PhysicalProjectionPlan` | P1 | The codec registry is hard-wired to the CPU backend's projection plan (see D2). A Metal or other backend cannot supply its own plan. | Add `CodecRegistry` → backend-neutral `ProjectionPlan` trait. |
| M3 | `opplan/exec/production.rs:74-96` | P2 | Three `ProductionBackend` identities as hard-coded name/family/revision triples. | Fine as data. Consider one table keyed by format. |

## 5. Hardcoding to an architecture

| # | Location | Sev | Finding | Fix |
|---|---|---|---|---|
| A1 | `opplan/exec/{kimi_kda_layer,kimi_mla_layer,kimi_moe_block,kimi_source,stack,stack_metal,token}.rs` | P1 | There is a second, Kimi-only execution path next to the generic `prepared` / `production` path. The operators are generic (KDA, MLA, MoE), but the files are family-named. `stack.rs:307` panics with a Kimi-specific invariant: `"no Kimi layer is MLA+dense — first_k_dense_replace=1 excludes only layer 0"`. | Fold it into the generic plan executor, or rename it to operator names (`kda_layer`, `mla_layer`, `moe_block`). Replace the panic with a `VindexError` refusal driven by the plan. |
| A2 | `opplan/exec/kimi_router.rs` | P1 | This is the generic sigmoid + bias-corrected router (DeepSeek-V3 / GLM share it; `examples/glm_moe_residency.rs` and generic `mla.rs` import it), but it is named after one family. It is also the third implementation of sigmoid routing, alongside `production.rs:578-660` and `reference.rs:598-720`, each with its own `1e-20` epsilon. | Rename to `sigmoid_router.rs`. Keep exactly one production implementation plus the reference oracle, and delete the third. |
| A3 | `represent/quality.rs:680-720` (`gate_by_id` matching `"kimi-logit-v1..v3"`, `kimi_logit_*()` constructors with `positions_min: 4096`, `top10_change_max: Some(82)` etc.) | P1 | Model-specific acceptance gates are compiled into the generic `represent` library. | Load gates as declared data (a JSON gate file with a registry), or scope them to a `gates/kimi.rs` data module named neutrally by instrument. |
| A4 | `represent/reading.rs:33-165` (`ReadingKind::Kimi`, `Reading::Kimi(Box<QualityBank>)`, `GateKind::Kimi`) | P1 | Enum variants named after a family for what is actually the `teacher-forced-two-arm/v1` procedure. | Rename to `TeacherForcedV1` (serde tag unchanged). |
| A5 | `represent/measure/teacher_forced.rs:60-62,126` (`"kimi-source-expert-bank"` etc.; `format!("/tmp/kimi_{label}_report.json")`), `represent/execution_cost.rs:380-401` (`"Kimi-Linear-48B-A3B-Instruct"`, provenance log paths) | P1 | A library path writes to a hard-coded `/tmp` path and embeds a model identity. | Take the output path and identity as parameters. Move the experiment driver to `examples/` or `tests/`. |
| A6 | `gguf/{export,metadata/mod,preflight/mod,plan/mod,vocab,walk/mod}.rs` (`ARCHITECTURE = "qwen35"`, `VOCAB_PRE = "qwen2"`, `qwen35_metadata`, `Qwen35Lowering`, `QwenExport`, `export_qwen35`) | P2 | The GGUF export path supports exactly one target family, which is baked into module-level pub names. This is acceptable as a declared lowering target but not extensible. | Add a `GgufTarget` trait (tensor names, metadata keys, transforms, vocab pre), with `Qwen35` as one impl under `gguf/targets/qwen35.rs`. |
| A7 | `opplan/build.rs:1131,1211-1218,2050,2213`, `production.rs:539-585`, `reference.rs:601,673` (`MoeRouterKind::Gemma4Hybrid`) | P2 | The router variant is named after a family, although the value is declared in config (larql-models). Branches check `== Gemma4Hybrid` to decide whether router scale, per-expert scale, and norm are required. | Derive the required operands from declared router facts (`router.has_scale` etc.), or rename the variant to its semantics (`SoftmaxWithExpertScale`) in larql-models. |
| A8 | `plan/carriage.rs:342,348` ("DeepSeek's mscale extension"), `plan/semantics.rs:697` (`GLM_SPARSE_INDEXER`), `opplan/exec/kernels.rs:330` (`llama3_frequencies`) | P2 | Family names in strings and function names. These are mostly legitimate names for published rope/indexer schemes. | OK as scheme names. Just make sure no branch keys on them. |
| A9 | `opplan/exec/weights/mod.rs:39` `DEVICE_PAGE_ALIGN = 16384`, `opplan/exec/prefetch.rs:50` fallback `4096` | P2 | Hardware (Apple GPU) page sizes baked into the format layer. | Get them from the backend or `sysconf`. The prefetch fallback should be a named const. |

No hard-coded model dims (151936 / 262144 / head counts / rope theta) were found in non-test source. The only `500000.0` is in the `kernels.rs` test module.

## 6. General code review

| # | Location | Sev | Finding | Fix |
|---|---|---|---|---|
| G1 | `graph/surface.rs:908,916-918` | P1 (likely bug) | `let hidden = nested.hidden_size.unwrap_or(0)`. If `hidden_size` is undeclared and `head_dim` is absent while `heads > 0`, then `0.is_multiple_of(heads)` is true, so `head_dim = 0` is accepted silently and nothing is pushed to `missing`. | Push `"hidden_size"` to `missing` when it is `None`, and only derive `head_dim` from a declared hidden size. |
| G2 | `opplan/build.rs:239,933,987,1031,1071,1085,1097,1109` | P1 | Eight `panic!` / `expect` calls in `plan_component_ops` that rely on "closure should have refused this". `pub fn plan_component_ops` can be called without closure (e.g. from tests or other crates), so a malformed container panics the server. | Return `VindexError::Parse` (the messages already exist). |
| G3 | `opplan/exec/stack.rs:74,307,310`, `stack_metal.rs:228,504-538`, `routed_experts/worker.rs:68,76,142,148,221,350` (bare `unreachable!()`), `token.rs:62,92,120,141-142` | P1 | Panics in pub exec APIs. `worker.rs` is on the distributed-worker path (V3-FFN-SLICE-1), so a panic there kills a worker on bad input. | Convert to `Result`. At minimum, give each `unreachable!` a message. |
| G4 | `opplan/exec/cpu/kernels.rs:27,55,95,141,177,237,427,437` | P2 | Kernel impls `panic!` when handed the wrong `WeightRows` variant. It is an internal invariant, but reachable through a mis-paired projector. | Make the projector generic over its row type, or return `Err`. |
| G5 | `opplan/exec/device.rs:149,256,278,306,376` `.lock().expect(...)`, `remote/hydrate.rs:179-266`, `represent/physical.rs:373-420` `.lock().unwrap()` | P2 | A poisoned mutex cascades into a panic across every caller. | Use `parking_lot::Mutex`, or `lock().unwrap_or_else(PoisonError::into_inner)` for stats counters. |
| G6 | `opplan/exec/prepared.rs:2383` | P2 | `layer_plan.ffn.as_ref().unwrap()` re-derives something already matched as `LayerFfn::Dense(op)` at 2362. | Pass `op` / the matched ffn directly. |
| G7 | `opplan/build.rs:502` (`intermediate: inter_for(layer).unwrap_or(0)`), `:1544` (`vocab_size: vocab.unwrap_or(0)`), `:640,919,1106` (`bank_id … unwrap_or_default()`), `graph/build.rs:431-432`, `opplan/planned.rs:286`, `opplan/exec/experts.rs:916` (`router.shape.get(1).unwrap_or(0)`) | P2 | `0` or `""` used as an "undeclared" sentinel in plan structs, so a later consumer cannot tell "absent" from "zero". This violates the "checked default, never assumed" rule. | Use `Option<usize>` in the op structs, or refuse at build time. |
| G8 | `represent/measure/teacher_forced.rs` (pub, `cfg(gpu, macos)`) | P1 | A library module reads env vars (`LARQL_RESIDENCY_SET`, `env_dir`) and writes `/tmp` files. It is test/experiment harness code compiled into the lib. | Move to `examples/` or `tests/`. |
| G9 | `opplan/exec/mod.rs:2108` `pub fn set_multi_position_ffn` | P1 | A public global mutable switch; its own doc says "Not for production code". | Make it `#[cfg(test)]`, or remove it via ExecConfig (D7). |
| G10 | Public API breadth: `opplan/exec/mod.rs:21-84` exposes ~55 `pub mod`s, including internals (`kernels`, `narrow`, `payload_prefix`, `quantise`, `prefetch`, `timing`) | P2 | This freezes internals as API before release. | Make them `pub(crate)` by default and re-export a curated surface from `exec/mod.rs`. |
| G11 | `unsafe` blocks (`read.rs:121`, `prefetch.rs`, `accounting.rs:598-613`, `cpu/*` NEON, `kda_metal.rs:70`) | OK | Each has a `SAFETY` comment within 4 lines. No action. | — |

---

## Counts per criterion

| Criterion | P0 | P1 | P2 |
|---|---|---|---|
| 1 Decoupling | 0 | 7 | 1 |
| 2 File size | 0 | 28 source files | 20 test files |
| 3 Structure | 0 | 4 | 3 |
| 4 Module hardcoding | 0 | 2 | 1 |
| 5 Architecture hardcoding | 0 | 5 | 4 |
| 6 General | 0 | 5 | 5 |

There are no P0 blockers. G1 (silent `head_dim = 0`) and G2/G3 (panics on container-derived input in pub APIs) are the closest to one.
