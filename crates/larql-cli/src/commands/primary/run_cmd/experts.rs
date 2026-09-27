//! `--experts` wiring: load registry, wrap prompt, generate, dispatch.
//!
//! Self-contained — does not call into `walk_cmd` because we need the raw
//! generated text for op-call extraction (walk_cmd streams to stdout).
//!
//! Backend matrix:
//!
//! | vindex quant | `--metal` | strategy                                    |
//! |--------------|-----------|---------------------------------------------|
//! | Q4_K         | yes       | `layer_graph::generate` (KV-cached, fast)   |
//! | Q4_K         | no        | `vindex::generate_kquant_cpu` (per-step, slow) |
//! | f32          | any       | `forward::generate_cached` (CPU, F32)       |
//!
//! Chat mode (no prompt): drops into a stdin REPL over the same loaded model.

use super::*;
use larql_inference::experts::{
    DispatchOutcome, DispatchSkip, Dispatcher, ExpertRegistry, ExpertSession, FilteredDispatcher,
    OpNameMask,
};
use larql_inference::prompt::ChatTemplate;
use larql_inference::WeightFfn;
use larql_vindex::{load_vindex_tokenizer, SilentLoadCallbacks, VectorIndex};

type BoxErr = Box<dyn std::error::Error>;

/// Which decode strategy to use for this `--experts` invocation.
enum Strategy {
    /// Q4_K vindex + Metal backend. KV-cached decode via `layer_graph::generate`.
    MetalQ4K,
    /// Q4_K vindex, no Metal. Loops `predict_kquant` per token (O(N²)).
    CpuQ4K,
    /// Non-quantised vindex. CPU `generate_cached` with full f32 weights.
    CpuF32,
}

impl Strategy {
    fn name(&self) -> &'static str {
        match self {
            Self::MetalQ4K => "metal-q4k",
            Self::CpuQ4K => "cpu-q4k",
            Self::CpuF32 => "cpu-f32",
        }
    }
}

/// Resolved runtime — model + index + backend + chosen strategy. Lives
/// across REPL turns so loads (and Metal init) only happen once.
struct Runtime {
    backend: Box<dyn larql_compute::ComputeBackend>,
    weights: larql_inference::ModelWeights,
    tokenizer: tokenizers::Tokenizer,
    index: Option<VectorIndex>,
    strategy: Strategy,
}

/// Teacher-forced prefix that drops the model into the op-name field
/// of an op-call JSON immediately. Used by `--constrained`.
const OP_CALL_PREFIX: &str = r#"{"op":""#;

impl Runtime {
    /// Generate text from `wrapped`. When `mask_op_names` is `Some`,
    /// constrained decoding (a) injects [`OP_CALL_PREFIX`] into the
    /// prompt as teacher-forcing and (b) restricts the op-name field
    /// of the generated text to a prefix of one of those op names.
    /// `None` is unconstrained generation.
    ///
    /// Returns the generated text. When constrained, the returned
    /// string includes the teacher-forced prefix so downstream
    /// `parse_op_call` sees a complete `{"op":"..."}` block.
    fn generate(
        &mut self,
        wrapped: &str,
        max_tokens: usize,
        mask_op_names: Option<&[String]>,
    ) -> Result<String, BoxErr> {
        // Teacher-force the JSON prefix when constrained — the model
        // never has to "decide" to emit the op-call.
        let effective_prompt: String = if mask_op_names.is_some() {
            format!("{wrapped}{OP_CALL_PREFIX}")
        } else {
            wrapped.to_string()
        };
        let token_ids =
            larql_inference::encode_prompt(&self.tokenizer, &*self.weights.arch, &effective_prompt)
                .map_err(|e| format!("tokenize: {e}"))?;

        let text = match self.strategy {
            Strategy::MetalQ4K => {
                let index = self.index.as_ref().expect("metal-q4k needs index");
                let backend = self.backend.as_ref();
                let cached_layers =
                    larql_inference::layer_graph::CachedLayerGraph::from_residuals(Vec::new());
                let num_layers = self.weights.num_layers;
                let result = if let Some(ops) = mask_op_names {
                    let mut mask = OpNameMask::new(ops.to_vec(), &self.tokenizer);
                    mask.set_seed_text(OP_CALL_PREFIX);
                    larql_inference::layer_graph::generate_constrained(
                        &mut self.weights,
                        &self.tokenizer,
                        &token_ids,
                        max_tokens,
                        index,
                        backend,
                        &cached_layers,
                        0..num_layers,
                        |ids, logits| mask.apply(ids, logits),
                    )
                } else {
                    larql_inference::layer_graph::generate(
                        &mut self.weights,
                        &self.tokenizer,
                        &token_ids,
                        max_tokens,
                        index,
                        backend,
                        &cached_layers,
                        0..num_layers,
                    )
                };
                result.tokens.iter().map(|(t, _)| t.as_str()).collect()
            }
            Strategy::CpuQ4K => {
                let index = self.index.as_ref().expect("cpu-q4k needs index");
                let toks = if let Some(ops) = mask_op_names {
                    let mut mask = OpNameMask::new(ops.to_vec(), &self.tokenizer);
                    mask.set_seed_text(OP_CALL_PREFIX);
                    larql_inference::vindex::generate_kquant_cpu_constrained(
                        &mut self.weights,
                        &self.tokenizer,
                        &token_ids,
                        max_tokens,
                        index,
                        |ids, logits| mask.apply(ids, logits),
                    )
                } else {
                    larql_inference::vindex::generate_kquant_cpu(
                        &mut self.weights,
                        &self.tokenizer,
                        &token_ids,
                        max_tokens,
                        index,
                    )
                };
                toks.into_iter().map(|(t, _)| t).collect()
            }
            Strategy::CpuF32 => {
                let ffn = WeightFfn {
                    weights: &self.weights,
                };
                let mut text = String::new();
                if let Some(ops) = mask_op_names {
                    let mut mask = OpNameMask::new(ops.to_vec(), &self.tokenizer);
                    mask.set_seed_text(OP_CALL_PREFIX);
                    larql_kv::generation::generate_cached_constrained(
                        &self.weights,
                        &self.tokenizer,
                        &ffn,
                        &token_ids,
                        max_tokens,
                        |ids, logits| mask.apply(ids, logits),
                        |_id, tok| text.push_str(tok),
                    );
                } else {
                    larql_kv::generation::generate_cached(
                        &self.weights,
                        &self.tokenizer,
                        &ffn,
                        &token_ids,
                        max_tokens,
                        |_id, tok| text.push_str(tok),
                    );
                }
                text
            }
        };
        // When constrained, prepend the teacher-forced prefix so the
        // dispatcher sees a complete op-call JSON block.
        let result = if mask_op_names.is_some() {
            format!("{OP_CALL_PREFIX}{text}")
        } else {
            text
        };
        Ok(result)
    }
}

/// Locate the WASM experts directory.
///
/// Search order:
///   1. `--experts-dir <PATH>` flag (if provided).
///   2. `LARQL_EXPERTS_DIR` env var.
///   3. Workspace build dir relative to the running CLI binary location.
fn resolve_experts_dir(args: &RunArgs) -> Result<PathBuf, BoxErr> {
    resolve_experts_dir_inner(
        args.experts_dir.clone(),
        std::env::var("LARQL_EXPERTS_DIR").ok().map(PathBuf::from),
        std::env::current_exe().ok(),
    )
}

/// Pure version of [`resolve_experts_dir`] — env var + current exe are
/// passed in. Lets unit tests exercise the precedence chain without
/// mutating shared process state.
fn resolve_experts_dir_inner(
    arg_dir: Option<PathBuf>,
    env_dir: Option<PathBuf>,
    exe_path: Option<PathBuf>,
) -> Result<PathBuf, BoxErr> {
    if let Some(p) = arg_dir {
        if !p.is_dir() {
            return Err(format!("--experts-dir does not exist: {}", p.display()).into());
        }
        return Ok(p);
    }
    if let Some(path) = env_dir {
        if path.is_dir() {
            return Ok(path);
        }
    }
    if let Some(exe) = exe_path {
        for ancestor in exe.ancestors() {
            let candidate = ancestor.join("crates/larql-experts/target/wasm32-wasip1/release");
            if candidate.is_dir() {
                return Ok(candidate);
            }
        }
    }
    Err(
        "could not locate WASM experts directory; pass --experts-dir or set LARQL_EXPERTS_DIR"
            .into(),
    )
}

/// Detect the chat template from a vindex.
///
/// Vindexes ship their family in `index.json` (no `config.json` — that
/// only exists in raw safetensors directories), so we read it directly.
/// Falls back to [`larql_models::detect_architecture`] for non-vindex
/// model dirs, then to `Plain` if neither resolves.
fn detect_template(vindex_path: &Path) -> ChatTemplate {
    // Try vindex index.json first.
    let index_path = vindex_path.join(INDEX_JSON);
    if let Ok(text) = std::fs::read_to_string(&index_path) {
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) {
            if let Some(family) = value.get("family").and_then(|v| v.as_str()) {
                return ChatTemplate::for_family(family);
            }
            // Fall back to model id → for_model_id heuristic if family is absent.
            if let Some(id) = value.get("model").and_then(|v| v.as_str()) {
                return ChatTemplate::for_model_id(id);
            }
        }
    }
    // Fall back to safetensors-style config.json detection.
    match larql_models::detect_architecture(vindex_path) {
        Ok(arch) => ChatTemplate::for_family(arch.family()),
        Err(_) => ChatTemplate::Plain,
    }
}

/// Pure strategy selector: given the vindex quant format and whether
/// the constructed backend has a fused Q4 decode pipeline, pick a
/// decode strategy.
fn pick_strategy(quant: larql_vindex::QuantFormat, metal_ready: bool) -> Strategy {
    match (quant, metal_ready) {
        (larql_vindex::QuantFormat::Q4K, true) => Strategy::MetalQ4K,
        (larql_vindex::QuantFormat::Q4K, false) => Strategy::CpuQ4K,
        _ => Strategy::CpuF32,
    }
}

/// Load the runtime: weights + tokenizer + q4 index (when needed).
fn load_runtime(vindex_path: &Path, args: &RunArgs) -> Result<Runtime, BoxErr> {
    let mut cb = SilentLoadCallbacks;
    let cfg = larql_vindex::load_vindex_config(vindex_path)?;
    // Build the backend first, then probe the *instance* for the fused
    // Q4 decode pipeline (the canonical PrefillQ4 + DecodeToken pair).
    // The old `metal_ready_for_q4` probed `default_backend()` — always
    // CPU since ADR-019, whose `supports_quant(Q4_K)` is `true` — so it
    // reduced to `== args.metal` and the "metal-q4k" strategy then ran
    // on a CPU backend.
    let backend = crate::backend_select::backend_for_metal_flag(args.metal)?;
    let fused_q4_ready = backend.supports(larql_compute::Capability::PrefillQ4)
        && backend.supports(larql_compute::Capability::DecodeToken);
    let strategy = pick_strategy(cfg.quant, fused_q4_ready);

    if args.verbose {
        eprintln!(
            "strategy: {} (quant={:?}, metal_requested={})",
            strategy.name(),
            cfg.quant,
            args.metal
        );
    }

    let (weights, index) = match strategy {
        Strategy::MetalQ4K | Strategy::CpuQ4K => {
            let weights = larql_vindex::load_model_weights_kquant(vindex_path, &mut cb)?;
            let mut idx = VectorIndex::load_vindex(vindex_path, &mut cb)?;
            idx.load_attn_kquant(vindex_path)?;
            idx.load_interleaved_kquant(vindex_path)?;
            let _ = idx.load_lm_head_kquant(vindex_path);
            (weights, Some(idx))
        }
        Strategy::CpuF32 => {
            let weights = larql_vindex::load_model_weights_with_opts(
                vindex_path,
                &mut cb,
                larql_vindex::LoadWeightsOptions::default(),
            )?;
            (weights, None)
        }
    };
    let tokenizer = load_vindex_tokenizer(vindex_path)?;
    Ok(Runtime {
        backend,
        weights,
        tokenizer,
        index,
        strategy,
    })
}

/// Print a single dispatch outcome (or skip reason) to stdout/stderr.
fn print_dispatch(
    model_output: &str,
    outcome: Result<DispatchOutcome, DispatchSkip>,
) -> Result<(), BoxErr> {
    match outcome {
        Ok(DispatchOutcome { call, result }) => {
            println!(
                "{}",
                serde_json::json!({
                    "op": call.op,
                    "args": call.args,
                    "value": result.value,
                    "expert_id": result.expert_id,
                })
            );
            Ok(())
        }
        Err(DispatchSkip::NoOpCall) => {
            eprintln!("no op-call extracted; raw output:");
            println!("{model_output}");
            Ok(())
        }
        Err(DispatchSkip::UnknownOp(op)) => {
            Err(format!("model emitted unknown op `{op}`; raw output: {model_output}").into())
        }
        Err(DispatchSkip::ExpertDeclined { op, args }) => {
            Err(format!("expert `{op}` declined args {args}; raw output: {model_output}").into())
        }
    }
}

pub fn run(vindex_path: &Path, args: &RunArgs) -> Result<(), BoxErr> {
    // ── Load experts ──
    let experts_dir = resolve_experts_dir(args)?;
    if args.verbose {
        eprintln!("experts: loading from {}", experts_dir.display());
    }
    let registry = ExpertRegistry::load_dir(&experts_dir)?;
    if args.verbose {
        eprintln!(
            "experts: loaded {} modules ({} ops)",
            registry.len(),
            registry.ops().len()
        );
    }

    // Optionally narrow the registry to a focused subset — small models
    // pick the right op far more reliably with 5–15 options than 126.
    let dispatcher: Box<dyn larql_inference::experts::Dispatcher> = if args.ops.is_empty() {
        Box::new(registry)
    } else {
        if args.verbose {
            eprintln!("experts: filtering to {} ops", args.ops.len());
        }
        Box::new(FilteredDispatcher::new(registry, args.ops.clone()))
    };
    let mut session = ExpertSession::new(dispatcher);

    // ── Detect template + load model ──
    let template = detect_template(vindex_path);
    if args.verbose {
        eprintln!("template: {}", template.name());
    }
    let mut runtime = load_runtime(vindex_path, args)?;

    if let Some(prompt) = args.prompt.as_deref() {
        run_one(&mut session, &mut runtime, prompt, template, args)
    } else {
        run_chat(&mut session, &mut runtime, template, args)
    }
}

/// Single dispatch: wrap → generate → dispatch → print.
fn run_one(
    session: &mut ExpertSession<Box<dyn larql_inference::experts::Dispatcher>>,
    runtime: &mut Runtime,
    prompt: &str,
    template: ChatTemplate,
    args: &RunArgs,
) -> Result<(), BoxErr> {
    let wrapped = session.build_prompt(prompt, template);
    let mask_op_names: Option<Vec<String>> = if args.constrained {
        Some(
            session
                .registry()
                .op_specs()
                .into_iter()
                .map(|s| s.name)
                .collect(),
        )
    } else {
        None
    };
    let model_output = runtime.generate(&wrapped, args.max_tokens, mask_op_names.as_deref())?;
    if args.verbose {
        eprintln!("model output: {model_output:?}");
    }
    print_dispatch(&model_output, session.dispatch(&model_output))
}

/// REPL: read line → run_one → repeat. Loads model exactly once.
fn run_chat(
    session: &mut ExpertSession<Box<dyn larql_inference::experts::Dispatcher>>,
    runtime: &mut Runtime,
    template: ChatTemplate,
    args: &RunArgs,
) -> Result<(), BoxErr> {
    eprintln!("larql experts chat — Ctrl-D to exit");
    let stdin = io::stdin();
    let mut stderr = io::stderr();
    loop {
        write!(stderr, "> ")?;
        stderr.flush()?;
        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) => {
                eprintln!();
                return Ok(());
            }
            Ok(_) => {}
            Err(e) => return Err(Box::new(e)),
        }
        let prompt = line.trim();
        if prompt.is_empty() {
            continue;
        }
        // Per-turn errors don't kill the REPL — print and continue.
        if let Err(e) = run_one(session, runtime, prompt, template, args) {
            eprintln!("error: {e}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_vindex::QuantFormat;

    // ── pick_strategy ──────────────────────────────────────────────────

    #[test]
    fn pick_strategy_q4k_with_metal_picks_metal() {
        assert!(matches!(
            pick_strategy(QuantFormat::Q4K, true),
            Strategy::MetalQ4K
        ));
    }

    #[test]
    fn pick_strategy_q4k_without_metal_picks_cpu_q4k() {
        assert!(matches!(
            pick_strategy(QuantFormat::Q4K, false),
            Strategy::CpuQ4K
        ));
    }

    #[test]
    fn pick_strategy_non_q4k_with_metal_falls_back_to_f32() {
        // Metal can't help with non-Q4K weights — backend has no f32 path.
        assert!(matches!(
            pick_strategy(QuantFormat::None, true),
            Strategy::CpuF32
        ));
    }

    #[test]
    fn pick_strategy_non_q4k_without_metal_picks_cpu_f32() {
        assert!(matches!(
            pick_strategy(QuantFormat::None, false),
            Strategy::CpuF32
        ));
    }

    // ── resolve_experts_dir_inner ──────────────────────────────────────

    #[test]
    fn resolve_arg_dir_when_valid() {
        let dir = tempfile::tempdir().expect("tempdir");
        let p = dir.path().to_path_buf();
        let resolved = resolve_experts_dir_inner(Some(p.clone()), None, None).expect("ok");
        assert_eq!(resolved, p);
    }

    #[test]
    fn resolve_arg_dir_invalid_errors() {
        let bogus = PathBuf::from("/this/path/does/not/exist/xyz");
        let err = resolve_experts_dir_inner(Some(bogus.clone()), None, None).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("--experts-dir does not exist"), "got: {msg}");
        assert!(
            msg.contains(bogus.to_str().unwrap()),
            "msg should name the path; got: {msg}"
        );
    }

    #[test]
    fn resolve_falls_through_to_env_dir() {
        let env = tempfile::tempdir().expect("tempdir");
        let resolved =
            resolve_experts_dir_inner(None, Some(env.path().to_path_buf()), None).expect("ok");
        assert_eq!(resolved, env.path());
    }

    #[test]
    fn resolve_skips_invalid_env_dir_falls_to_workspace_walk() {
        // env dir doesn't exist; workspace walk must then succeed.
        // Build a fake "exe" inside a workspace-shaped tempdir tree.
        let root = tempfile::tempdir().expect("tempdir");
        let wasm_dir = root
            .path()
            .join("crates/larql-experts/target/wasm32-wasip1/release");
        std::fs::create_dir_all(&wasm_dir).unwrap();
        // exe is conceptually somewhere inside root, e.g. target/debug/larql.
        let exe = root.path().join("target/debug/larql");
        std::fs::create_dir_all(exe.parent().unwrap()).unwrap();

        let resolved =
            resolve_experts_dir_inner(None, Some(PathBuf::from("/nonexistent/env/dir")), Some(exe))
                .expect("ok");
        assert_eq!(
            resolved.canonicalize().unwrap(),
            wasm_dir.canonicalize().unwrap()
        );
    }

    #[test]
    fn resolve_returns_error_when_nothing_resolves() {
        let err = resolve_experts_dir_inner(
            None,
            Some(PathBuf::from("/nope/env")),
            Some(PathBuf::from("/nope/exe")),
        )
        .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("could not locate"), "got: {msg}");
        assert!(
            msg.contains("--experts-dir"),
            "should hint at the flag; got: {msg}"
        );
    }

    // ── print_dispatch ─────────────────────────────────────────────────

    #[test]
    fn print_dispatch_unknown_op_errors() {
        let outcome = Err(DispatchSkip::UnknownOp("foo".into()));
        let err = print_dispatch("raw model output", outcome).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("unknown op `foo`"), "got: {msg}");
        assert!(
            msg.contains("raw model output"),
            "should include raw output; got: {msg}"
        );
    }

    #[test]
    fn print_dispatch_expert_declined_errors() {
        let outcome = Err(DispatchSkip::ExpertDeclined {
            op: "gcd".into(),
            args: serde_json::json!({"bad": true}),
        });
        let err = print_dispatch("output", outcome).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("expert `gcd` declined"), "got: {msg}");
    }

    #[test]
    fn print_dispatch_no_op_call_succeeds() {
        // No op-call is a soft case — print raw, return Ok.
        let outcome = Err(DispatchSkip::NoOpCall);
        assert!(print_dispatch("free text", outcome).is_ok());
    }
}
