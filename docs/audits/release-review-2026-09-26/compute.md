# Pre-release review — larql-compute + larql-compute-metal

Scope: `crates/larql-compute` (CPU substrate + forward runtime) and `crates/larql-compute-metal` (Metal peer, 69 shader modules as MSL-in-Rust strings). Total ~116K lines of `.rs` across both `src/` trees. Read-only review; line numbers are from the working tree on `vindex3/qkv-attention-bias` (2026-09-26).

Severity: **P0** = blocks release / hides failures; **P1** = should fix before release; **P2** = cleanup.

---

## 1. Decoupling

### What is good
- `larql-compute/src/backend/mod.rs:49` `ComputeBackend: MatMul + QuantMatVec + DecodeBackend` is a real trait. `backend/factory.rs` has a `BackendKind` + ctor registry (`backend_from_spec`) that already names cuda/vulkan (`factory.rs:104`), and `larql-cli/src/backend_select.rs` is the one place the CLI picks a crate. No caller outside compute-metal downcasts to `MetalBackend` via `as_any` (0 hits). **Legacy serving path: decoupled.**
- `larql-compute` has no `metal`/`objc` dependency (Cargo.toml). Metal words in compute src are only the `BackendKind::Metal` enum and a `RowLocation::LocalGpu { backend: "metal" }` literal (`state_handle.rs:402`).

### Findings
| Sev | Location | Finding | Fix |
|---|---|---|---|
| **P0** | `larql-compute-metal/src/lowering/*`, `trait_impl/{kda,mla,kimi_layer,grouped_experts,bf16_*}` used from `larql-vindex/src/format/vindex3/opplan/exec/{stack_metal,kda_metal,kimi_source}.rs`, `represent/{physical,measure/teacher_forced}.rs`, `encode/segment.rs`, and 12 `larql-cli/src/commands/primary/vindex3_cmd/**` files + `larql-server/src/{vindex3.rs,routes/expert/metal.rs,routes/walk_ffn/q8k.rs}` | **The VINDEX3 execution path (the product path) has no backend trait at all.** It calls inherent `MetalBackend` methods (`encode_matvec`, `encode_attention`, `kda_attention_step`, `kimi_decoder_layer`, …) and holds `metal::Buffer`/`CommandBuffer` (re-exported as `lowering::DeviceBuffer`/`DeviceCommandBuffer`, `lowering/mod.rs:47-50` — a type alias, not an abstraction). Top imports by callers: `MetalBackend` ×37, `trait_impl::kda` ×17, `grouped_experts` ×16, `kimi_layer` ×15, `lowering::*` ×15. Adding CUDA means duplicating every `*_metal.rs` in vindex and every `lowered/*` CLI command. | Define a `LoweringBackend` (encoder-level) trait in `larql-compute` with associated `Buffer`/`CommandBuffer`/`Encoder` types covering the primitive set in `lowering/mod.rs` (matvec, matmul_rows, rms_norm_rows, rope, residual, attention, kda step, grouped experts, router select); implement it for `MetalBackend`; make vindex/CLI/server generic over it. |
| **P1** | `larql-compute/src/backend/decode.rs:224-548`, `backend/quant_matvec.rs:239,307`, `backend/capability.rs:24-101` | `DecodeBackend` is self-described as "Metal-shaped" (`backend/mod.rs:11`). Trait surface is format- and kernel-exploded: `full_pipeline_q4`, `multi_layer_q4_ffn`, `decode_token_q4k_moe`, `q4_matvec_pair_batch`, **`q4k_matvec_stride32`** (a Metal kernel variant name leaked into the trait; doc talks of `simd_sum` across 32 lanes), `wire_resident` (macOS VM-wiring concept). `Capability` has `FullPipelineQ4`, `DecodeQ4KMoe`, `PrefillQ4` … | Collapse onto `quant_matvec(format, …)` + a `ReductionOrder::Stable` hint; move `q4k_matvec_stride32`/`wire_resident` to Metal-inherent; key capabilities by operator+format, not by legacy method name. |
| **P1** | `larql-compute-metal/src/kv_dispatch_impl.rs:28-130`, `async_compute_backend_impl.rs:28-90` | `impl KvDispatch for MetalBackend` and `impl AsyncComputeBackend for MetalBackend` delegate **every** method to `const CPU: CpuBackend` ("Step 4 scaffold … when real Metal kernels land (Step 5)"). A caller who selected "metal" silently runs CPU for KV/attention. | Don't implement the traits for Metal until real; or make them return a typed `Unsupported` and have `supports(KvHandleNative)` etc. false (verify), so callers route explicitly. |
| **P1** | `larql-compute/Cargo.toml` (`larql-models` dep); 37 non-test src files use `ModelWeights`/`WeightsView`/`ModelArchitecture` (e.g. `residual.rs:33`, `forward/*`, `attention/block.rs`, `attention/decode/*`, `kquant_forward/*`, `pipeline_layer/*`, `ffn/*`, `forward_overrides.rs:102-281`) | **Why compute depends on models:** after ADR-0022 the crate is not "compute", it is the whole CPU forward runtime (embed, layer, predict, lens, PLE, vision tower, projector, MoE routing policy from `MoeRouterKind`). The kernel layer itself (`cpu/ops/*`, `backend/*`) only needs quant block constants (`quant::ggml::*_BLOCK_ELEMS`, `nvfp4`, `mxfp4::FusedHalf`, `half`). | Split: `larql-compute` (backend traits + `cpu/ops` kernels; depend on a tiny `larql-quant-spec` leaf or on `larql-vindex-spec` for block constants) and `larql-forward`/`larql-cpu-runtime` (everything taking `WeightsView`/`ModelArchitecture`). |
| **P2** | `larql-compute-metal/Cargo.toml`; `kv_dispatch_impl.rs:25`, `async_compute_backend_impl.rs:37-87`, `trait_impl/grouped_experts.rs:164` (`KdaGateForm`), `YarnRopeScaling` | compute-metal → models is mostly quant constants plus `WeightsView` only because of the CPU-delegating trait impls above, and `config::KdaGateForm` (a model-config enum used as a kernel selector). | Once the KvDispatch scaffold goes, the only need is block constants + a compute-owned `DecayGateForm` enum mapped at the caller. |
| **P2** | `larql-compute/src/backend/helpers.rs:1-40`, `attention/gpu.rs:1-20` | `dot_proj_gpu`/`matmul_gpu`/`run_attention_block_gpu` take `Option<&dyn ComputeBackend>` and fall back to ndarray on `None` — a second CPU path parallel to `CpuBackend`, and "gpu" naming in a backend-agnostic crate. | Require `&dyn ComputeBackend` (pass `&CpuBackend`), rename `*_via_backend`. |

---

## 2. Modularity / file size (>800 lines)

Source files (not tests) over the limit, with proposed splits:

| Lines | File | Proposed split |
|---|---|---|
| 1360 | `metal/src/trait_impl/matmul.rs` | tests at :908 → `tests/test_matmul_trait_impl.rs` (−450). Then `matmul/{trait_impl.rs (MatMul impl :10-447), gemv_encode.rs (encode_f32/f16_gemv), topk.rs (topk1/topk + reduce_* :506-820)}`. |
| 1111 | `metal/src/shaders/nvfp4_matvec.rs` | `nvfp4/{base.rs (SHADER v1/v2), sweep.rs (SWEEP_SHADER macro family :242-420), x2.rs (x2m/seg3/seg3t :423-700), matmul_sgk.rs (:694-1000), kernels.rs (marker structs :1005+)}`. Also retire losing sweep arms (see §6). |
| 1086 | `metal/src/shaders/mxfp4_grouped_experts.rs` | tests at :936 out; `mxfp4_grouped/{prelude.rs, split_lut16.rs (:127-580), down_combine.rs (:584-700), decode_variants.rs (DECODE_* :702-790), kernels.rs}`. |
| 1017 | `metal/src/ops/kv_cache.rs` | tests at :529 out (−490). Then `kv_cache/{cache.rs (LayerKVCache/KVCache), encode.rs (append/attend/seqpar)}`. |
| 1004 | `metal/src/trait_impl/kimi_layer/mod.rs` | move types (:56-360: `ExpertEncoding`, `ExpertAddressing`, `ProjectionBank`, `EncodedRegion`) → `moe_layer/types.rs` (they are shared by kda/mla, see §3); `chain.rs` (encode_layer_chain/recycle :415-600); `layer.rs` (encode_kimi_layer, scratch); `validate.rs` (:830+). |
| 960 | `metal/src/lowering/mod.rs` | mod.rs → re-exports only; `lowering/{matvec.rs (:83-600 targets + encode_matvec/matmul_rows/nvfp4/f16), norm.rs (qk_norm, rms_norm_rows, branch_norm :598-930), elementwise.rs (scale/rope/bias/residual :691-800), resources.rs (lowering_scratch/upload/readback/register :517-597)}`. |
| 947 | `metal/src/trait_impl/decode.rs` | `decode/{prefill.rs (prefill_kquant*, capture_pre_wo :215-557), kv.rs (has/populate/reset/truncate/preallocate :558-667), token.rs (decode_token* :668-947)}`. |
| 921 | `metal/src/trait_impl/kda/mod.rs` | types (`KdaShape`, `KdaDeviceWeights`, `KdaDeviceState`, planes, :58-330) → `kda/types.rs`; `validate.rs` (:444-572); `encode.rs` (:599-913). |
| 916 | `metal/src/lowering/attention.rs` | `attention/{types.rs (:46-182), single.rs (encode_attention + kv/splitk :183-603), rows.rs (:604-916)}`. |
| 902 | `metal/src/decode/encode_attn.rs` | **one 750-line fn** `encode_attention_block` (:151). Split per arm: `fused_attn.rs`, `qk_norm_rope.rs`, `kv_append_attend.rs`, `seqpar.rs`, `unfused.rs`, with a small selector returning an enum. |
| 872 | `compute/src/cpu/ops/ternary_matvec.rs` | tests at :470 out (−400). Optional: `ternary/{weight.rs, scalar.rs, neon.rs}`. |
| 862 | `metal/src/decode/token.rs` | **one 846-line fn** `decode_token_with_moe_split_fn` (:15). Extract per-layer body into `encode_layer(..)` + setup/readback phases (the stage modules already exist under `decode/encode_*`). |
| 853 | `compute/src/cpu/ops/moe/forward.rs` | tests at :406 out (−450); `cpu_moe_forward` is 381 lines → `route.rs` + `experts.rs` + `combine.rs`. |
| 838 | `compute/src/backend/decode.rs` | tests at :550 out (−290); `DecodeStateDump`/`ProfileTimings` → `backend/decode_types.rs`. |

Other long functions (>200 lines, non-test): `ops/full_pipeline/dispatch.rs:137 dispatch_full_pipeline` (659), `decode_hybrid.rs:28 decode_attention_layer` (537), `moe_gpu_route/encode.rs:54` (393), `moe_dispatch/dispatch.rs:27` (364), `moe_zero_copy.rs:161` (350), `ops/full_pipeline/stages.rs:116` (302), `compute/attention/block.rs:292 run_attention_block_core` (300), `decode/moe_interleave.rs:170` (283), `backend/mod.rs:273 with_options` (204). **P1** for the three >500.

Test files over 800 (P2, split by kernel family): `metal/tests/test_metal_shaders.rs` (2444), `test_pipeline_and_moe.rs` (1210), `test_kernel_vindex_integration.rs` (1066), `test_kernel_decode_attention_sinks.rs` (886), `test_lowering_attention_rows.rs` (879), `test_lowering_ffn_parity.rs` (848), `test_kernel_fused_ops_norms.rs` (828); in-src test files `compute/src/cpu/ops/q4k_q8k_dot/tests.rs` (1166), `q4_common/tests.rs` (969), `kquant_forward/tests.rs` (944), `metal/src/decode/moe_interleave/tests.rs` (1013), `trait_impl/kimi_layer/tests/mod.rs` (991), `trait_impl/kda/tests.rs` (833). Examples: `compare_ollama.rs` (1085), `kda_q4_trajectory_real.rs` (893).

---

## 3. File / folder structure

| Sev | Location | Finding | Fix |
|---|---|---|---|
| **P1** | both crates | Project rule "tests in `tests/`" is broadly violated: compute has 90 files with inline `#[cfg(test)]` + 32 test files under `src/` (12.3K lines); metal 62 inline + 31 files under `src/` (9.9K lines). | Migrate in bulk per module; for `pub(crate)` internals expose via a `#[cfg(feature="test-internals")]` shim or keep only true white-box tests inline. |
| P2 | `metal/src/{cb_status.rs + cb_status/, kv_dispatch_impl.rs + kv_dispatch_impl/, moe_gpu_route.rs + moe_gpu_route/, moe_zero_copy.rs + moe_zero_copy/, route_guard.rs + route_guard/, decode/diag.rs + decode/diag/, decode/head.rs + head/, ops/kv_seqpar.rs + kv_seqpar/, ops/attention_geometry.rs + attention_geometry/, decode/moe_interleave.rs + moe_interleave/}` | Mixed `foo.rs` + `foo/` layout everywhere (dir usually only holds tests). | Pick one style; move tests to `tests/` and delete the sibling dirs. |
| P2 | `metal/src/` root: 16 loose modules (`decode_hybrid.rs`, `direct_ops.rs`, `f32_ops.rs`, `calibration.rs`, `submission_clock.rs`, `route_witness.rs`, `kv_residency_contract.rs`, `moe_descriptor.rs`, …) alongside `decode/`, `ops/`, `stages/`, `lowering/`, `trait_impl/`, `moe_dispatch/`, `moe_gpu_route/`, `moe_zero_copy` | Three MoE top-level modules and four "encode a layer" hierarchies (`decode/`, `ops/full_pipeline/`, `decode_hybrid.rs`, `lowering/`) — structure mirrors history, not concepts. | Group: `moe/{dispatch,gpu_route,zero_copy,descriptor}`, `attention/`, `ffn/`, `runtime/{decode,prefill,lowering}`. |
| P2 | `metal/src/trait_impl/kimi_layer/mod.rs:122` | `ExpertEncoding` (generic) lives in the Kimi module and is imported by `kda/mod.rs:40` and `mla/mod.rs:38`. | Move to a neutral `moe_layer/types.rs`. |
| P2 | `metal/src/kernels/kda.rs:52` | `KimiLayerKernels` and `MlaKernels` defined in `kda.rs`. | One file per registry. |
| P2 | `metal/src/diag/` (4.5K lines incl. `shader_bench/config.rs` CLI arg parsing, `kernel_profile/all.rs` 627-line fn) | Bench/profiling harness compiled into the library. | Move to `examples/` or a `larql-compute-metal-bench` bin; keep only reusable timing primitives in lib. |
| P2 | `metal/src/shaders/mod.rs:84` `all_shaders()` | All 69 MSL modules concatenated into one string and compiled with `new_library_with_source` on **every** `MetalBackend::new()` (including experimental `turboquant_*`, `q4_sparse_matvec`, `graph_walk_knn`, which have no production caller). No `.metal` files / offline `metallib`. | Build a `.metallib` in `build.rs` (xcrun metal) with runtime-source fallback; move experimental kernels behind a `diag` feature. |
| P2 | `compute/src/test_fixtures.rs` | Test support in src (check it is feature-gated like models' `test-utils`). | Gate behind `test-utils`. |

---

## 4. Hardcoding to a module / backend

| Sev | Location | Finding | Fix |
|---|---|---|---|
| **P0** | see §1 row 1 | Callers (`larql-vindex` opplan exec, CLI `vindex3_cmd/lowered/*`, `bench/vindex3_runtime.rs:42,312`, `measure.rs:244-290`, `prepare.rs:258-365`, server `routes/expert/metal.rs`) name `MetalBackend::new()` directly; 9 CLI sites bypass `backend_select.rs`. | Route all construction through `backend_for_kind`; make lowered commands generic over the §1 trait. |
| P1 | `larql-cli/src/commands/primary/vindex3_cmd/lowered/run.rs:212-216` | CLI reads `larql_compute_metal::route_witness::LOWERED_ATTEND_{SERIAL,SEQPAR,SPLITK}` global atomics directly. | Expose route witnesses via a backend-neutral `RouteWitness` report on the trait. |
| P2 | `metal/src/stages/layer_scalar.rs:77,127` | Only raw-string kernel lookups left (`get_function("scale_vector")`, in tests). Otherwise kernel binding is typed via `ShaderKernel`/`TiledKernel` (`kernels/traits.rs`) — good. | Use `shaders::…::ScaleVectorKernel::KERNEL_NAME`. |
| P2 | `metal/src/options/mod.rs` (+ 77 distinct `LARQL_*` env vars across both crates; 14 non-test `env::var` sites incl. `compute/src/forward_overrides.rs`) | Kernel variant selection by env (`LARQL_GATE_UP_8SG`, `LARQL_F16_ACC` (requires `…8SG=0`), `LARQL_FUSED_*`, `LARQL_MXFP4_ARM`, `LARQL_KV_SEQPAR`, `LARQL_Q6K_8SG`). `LARQL_FUSED_Q6K_DOWN` is documented as "currently no-op" (`kernels/ffn.rs:15-17`). | Keep `BackendOptions` as the only surface; delete no-op flags; delete losing variants (see §6). |

---

## 5. Hardcoding to an architecture

| Sev | Location | Finding | Fix |
|---|---|---|---|
| **P1** | `metal/src/shaders/kimi_layer.rs:78,89-129,264-274`; `trait_impl/kimi_layer/*`; `kernels/kda.rs:52`; `backend/mod.rs:188,406` | Family-named operator: `kimi_router_select` / `kimi_moe_combine` / `kimi_expert_addresses`, `KimiLayerKernels`, `MetalBackend::kimi_decoder_layer`. The operator (sigmoid + correction-bias select, unbiased renorm weights, noaux_tc) is shared with DeepSeek-V3/GLM. It also **duplicates** `shaders/moe_router_select.rs` (both `MAX_EXPERTS = 256`). | Rename to operator names (`sigmoid_bias_topk_router`, `moe_layer`); fold into `moe_router_select` with a policy flag. |
| **P1** | `shaders/kimi_layer.rs:78`, `shaders/moe_router_select.rs:142-145` | Routed expert cap `E ≤ 256`, `K ≤ 32` (threadgroup arrays). Host guards exist (`trait_impl/kimi_layer/mod.rs:839`, `moe_gpu_route/forward.rs:76`), but there is **no generic fallback** for E > 256 (e.g. Kimi-K2's 384 experts): `forward.rs:76` returns `None` (caller silently takes the non-GPU route), `kimi_layer` refuses with the wrong error variant (`GroupedError::SlotCountMismatch`, `mod.rs:840`). | Two-pass/tiled top-k for E > 256; add `GroupedError::TooManyExperts{max,found}`. |
| **P1** | `metal/src/ops/kv_cache.rs:12-17,367-384,446-460`; `decode/mod.rs:30` `DEFAULT_KV_CACHE_MAX_SEQ = 4096`; `shaders/kv_attention.rs:143,212,556` (`tg_scores[1024]`/`[4096]`) | ComputeBackend decode/prefill attention is capped at **4096** positions: exceeding it hits `assert!` panics (`kv_cache.rs:379`, `:453`, `encode_kv_append` :314). Split-K (`ops/kv_splitk.rs`) exists only on the lowering path; `decode/` never uses it. | Wire `kv_splitk` into `decode/encode_attn.rs` for span > `LONG_ATTENTION_SPAN`; size KV cache from config `max_position_embeddings`; return an error rather than panic. |
| P2 | `shaders/attn_fused.rs:97-98` (`tg_q[256]`) | head_dim ≤ 256 — correctly gated by `MAX_HEAD_DIM_SINGLE_SG` in `decode/encode_attn.rs:122` with an unfused fallback. OK. `qk_norm*`/`v_norm` use strided loops with `tg_partial[512]` bounded by the host clamp — OK. | — |
| P2 | `compute/src/pipeline/moe.rs:349` `MoeRoutingPolicy::gemma4_hybrid()`; `pipeline_layer/moe_build.rs:245` | Family-named policy constructor. | Name by operator (`softmax_topk_per_expert_scale`). |
| P2 | `compute/src/pipeline/layer.rs:59-160` | `FullPipelineLayer` is a flat struct that gains a field per family (softcap, v_norm, layer_scalar, PLE, kv_shared_source, residual_multiplier…); sentinels instead of `Option` (`layer_scalar: 0.0 = disabled`, `residual_multiplier 1.0 = no-op`). | Group into operator sub-specs (`NormSpec`, `ResidualSpec`, `RopeSpec`) with `Option`s. |
| P2 | `metal/src/diag/kernel_profile/mod.rs:26` `GEMMA3_4B_KV_DIM`, `diag/shader_bench/shapes.rs:30`, `census.rs:58-59` | Bench shapes hardcoded to gemma3-4b. Harmless in a bench, but it is in the library (see §3). | Move with diag; read shapes from a vindex config. |

---

## 6. General code review

### Tests that silently pass with no Metal (project rule violation)
| Sev | Location | Finding | Fix |
|---|---|---|---|
| **P0** | 12 files / 51 sites in `metal/tests/` (`test_matmul_trait_arms.rs` ×~20, `test_backend_arms.rs:121,137,155,221`, `test_kernel_q4k_matmul.rs:62,96,131,167,201`, `test_kernel_q6k_matvec_8sg.rs:68,126,181`, `test_kernel_q4k_ffn_gate_up_f16acc.rs:140,194`, `test_kernel_q4k_matmul_perf.rs:37`, `test_lowering_gemma4_arms.rs:240,273,308,388,593`, `test_lowering_representations_routed.rs:122,195,594,649`, `test_kernel_q4k_*_8sg.rs`, `metal_decode_synthetic/common.rs`) + src `stages/layer_scalar.rs:66-68,116-118`, `trait_impl/matmul.rs`, `trait_impl/decode.rs`, `moe_dispatch/entry.rs`, `decode/gpu_timing.rs` | `let Some(gpu) = … else { return }` / `None => return`. **Worse than a missing device:** `MetalBackend::with_options` (`backend/mod.rs:305-311` + every `?` on pipeline creation) returns `None` on **shader compile failure or any missing kernel**, so a broken shader library makes these tests green. | One shared `fn metal() -> MetalBackend` that `panic!`s on macOS (`#[cfg(target_os="macos")]`) and gate the test files on `cfg(target_os="macos")` instead of runtime skipping. Make `with_options` return `Result<Self, MetalInitError{NoDevice, Compile(String), MissingKernel(&'static str)}>` so only `NoDevice` is skippable. |
| P1 | `metal/src/lib.rs:167-174` | `metal_backend_returns_some_on_apple_silicon_or_none_off_host` and `…with_options_threads_through…` assert nothing (`let _ = …`). | Assert `is_some()` under `cfg(target_os="macos")`. |
| P1 | `metal/src/backend/mod.rs:311` vs `kernels/norm.rs:115` (`compile_required`) | Mixed init failure policy: some pipelines `?`→`None` (silently "no Metal"), others panic via `compile_required`. | Single `Result` path (above). |
| P2 | `metal/coverage-policy.json` note (2026-08-27) | CI runner `macos-14`: residency/profile tests take the `else { return }` arm and pass having run nothing — recorded in the policy, i.e. known green-without-execution. | Mark them `#[ignore = "needs macOS 15"]` so the skip is visible in test output. |

### Correctness
| Sev | Location | Finding | Fix |
|---|---|---|---|
| **P1** | `metal/src/moe_zero_copy.rs:214-218` vs `:260`, `:316` | `single_base` checks `u32::try_from(offset)` for the expert's base offset only, but the dispatched offset is `(r.gate_up.1 + half * gate_half_bytes) as u32` — for the up half this can exceed `u32::MAX` while the check passed, and `as u32` **silently truncates**, reading another expert's (or layer's) bytes. Reachable with multi-GB per-layer regions (large MoE). | Check `offset + gate_half_bytes` (the max offset actually dispatched) in `single_base`, and use `u32::try_from(..).expect(..)` at :260/:316. |
| P1 | `metal/src/moe_gpu_route/forward.rs:74-80, 211` | Geometry/limit violations return `None` → caller silently falls back to a non-GPU route (no log/route witness). | Return a typed refusal; bump a route-witness counter at minimum. |
| P2 | 447 `as u32` casts in metal non-test src; only 6 `try_from` | Dims/counts narrowed unchecked (e.g. `moe_descriptor.rs:312`, `lowering/nvfp4.rs:642`, `moe_gpu_route/encode.rs:112 gate_half_bytes as u32`). Most are safe dims; byte-offset ones are not. | `fn u32_of(x: usize, what) -> u32` with a checked conversion for every byte offset/element count bound to a kernel. |
| P2 | 36 raw `std::slice::from_raw_parts(buf.contents(), n)` in metal non-test src (e.g. `moe_descriptor.rs:391,397`, `decode/moe_interleave.rs:290,405,413`) | Bypass the existing checked helper `buffers::try_read_buffer_f32` (`buffers/mod.rs:386-410`, which checks null + `buf.length()`). No length check at these sites. | Route through the helper (add `u32`/descriptor variants). |

### unsafe / unwrap
| Sev | Location | Finding | Fix |
|---|---|---|---|
| P2 | `compute/src/cpu/ops/q4k_q8k_dot/q4k_neon.rs` (48 `unsafe {}` blocks, 0 SAFETY) | Bounds are enforced by slicing before `as_ptr()` + `KernelShapeError::check`, but undocumented. Metal non-test: 76 unsafe blocks / 39 SAFETY comments; undocumented sites in `moe_gpu_route/forward.rs`, `moe_descriptor.rs`, `decode/moe_interleave.rs`, `decode/diag.rs`, `decode/moe_combine.rs`. | Add `// SAFETY:` per block (or one per fn with `#[deny(clippy::undocumented_unsafe_blocks)]` in both crates). |
| P2 | `metal/src/trait_impl/decode.rs:227,474,571,580,605,626,633,645,657,677,732,766,877` | `self.kv_cache.lock().unwrap()` ×13 (+ `cache_guard.as_mut().unwrap()` :580) — one panic poisons decode forever. | `lock().unwrap_or_else(PoisonError::into_inner)` (already used in `backend/mod.rs:299`) via a `kv_cache()` accessor. |

### Duplication / dead code
| Sev | Location | Finding | Fix |
|---|---|---|---|
| **P1** | `decode/encode_attn.rs`, `ops/full_pipeline/*`, `decode_hybrid.rs:28`, `lowering/attention.rs` | Four independent attention/layer encoders; `lowering/mod.rs:20-26` states serving encoders were "deliberately not reused" with "intended end state … share primitives". Split-K/row variants exist only in one of them (§5). | Make `lowering` primitives the single encoder layer; re-express `decode/` and `full_pipeline/` on top. |
| P2 | `shaders/q4k_matvec{,_8sg,_stride32}`, `q4k_ffn_gate_up{,_8sg,_coop,_f16acc}`, `q4kf_*`, `q4k_qkv_proj/q4kf_qkv_proj/q4k_q6k_qkv_proj`, `nvfp4_matvec.rs` sweep arms (`SGF_PRODUCTION_ARM` :1067), `mxfp4_grouped_experts.rs` 6 `split_lut16*` variants | Many A/B variants kept alive as env opt-ins; all compiled and pipelined at init. | Keep production + one reference per op; move the rest to a `diag` feature. |
| P2 | `turboquant_{encode,decode}.rs`, `q4_sparse_matvec.rs`, `graph_walk_knn.rs` | "experimental, diag-only" but in the production library. | Feature-gate. |
| P2 | `ops/full_pipeline/stages.rs:29`, `decode/gpu_timing.rs:107`, `diag/kernel_profile/measure.rs:47`, `compute/.../q4k_neon.rs:53 prefetch_l1_keep`, `q6k.rs:332`, `kv_dispatch/cpu/handles.rs:22` | `#[allow(dead_code)]` items. `lib.rs:45` blanket `allow(dead_code)` off-macOS hides real dead code on Linux CI. | Delete or `cfg`-gate precisely. |

### Magic numbers
| Sev | Location | Finding | Fix |
|---|---|---|---|
| P2 | `kv_attention.rs` `tg_scores[1024]`/`[4096]`, `tg_sg_vals[8]`/`[32]`; `plan_glue.rs:48-237` `[32]`; `fused_ops.rs` `tg_p[8]`; `kda.rs:114,248` `partial[128]`; `q6k_geglu_down.rs:47` `[256]` | Threadgroup sizes as literals duplicated between MSL and Rust constants (only some are templated, e.g. `KIMI_MAX_EXPERTS`, `f32_gemv`). A test pins `kda.rs` by string match (`shaders/kda.rs:309`). | Template every bound from the Rust constant (as `kimi_layer.rs` does with `format!`). |
| P2 | `ops/full_pipeline/buffers.rs:283-286` (`HIDDEN_SMALL 1024`, `Q_DIM_LARGER_THAN_HIDDEN 4096`) | Named but model-shaped. | Check intent; derive from config. |

---

## Summary counts
- Files >800 lines: 14 source, 13 test/src-test, 2 examples.
- Functions >500 lines: 5 (`decode_token_with_moe_split_fn` 846, `encode_attention_block` 750, `dispatch_full_pipeline` 659, `profile_all` 627, `decode_attention_layer` 537).
- Silent Metal skips: 51 sites in 12 test files + ~6 in-src.
- Inline `#[cfg(test)]` modules: 90 (compute) + 62 (metal).
