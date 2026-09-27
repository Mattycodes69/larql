//! Build a .vindex from model weights — the extraction/clustering pipeline.
//!
//! Single entry point: `build_vindex` (full pipeline from weights). For
//! mid-run resume, see the streaming pipeline's checkpoint mechanism in
//! `extract::streaming` — the older `build_vindex_resume` path was
//! removed 2026-05-09 (it read the legacy `down_meta.jsonl` format that
//! nothing produces any more).
//!
//! `build_vindex` is structured around a `BuildContext` that holds the
//! shared inputs + accumulator state across the stages:
//!   1. `write_gate_vectors`            — gate matrices per layer (handles MoE)
//!   2. `write_embeddings`              — embedding table
//!   3. `write_down_meta_and_clusters`  — per-feature top-k tokens + collect
//!                                        offset directions for clustering
//!   4. `run_clustering`                — k-means + label clusters
//!   5. `write_tokenizer`
//!   6. `write_index_json`              — config + provenance + checksums
//!
//! Stage 3 lives in [`down_meta`], stage 6 in [`index_json`].
//! Discrete helpers live in `super::build_helpers`.

mod down_meta;
mod index_json;

use crate::extract::stage_labels::*;
use std::io::BufWriter;
use std::path::Path;

use larql_models::{FfnType, ModelWeights};

use crate::config::dtype::{write_floats, StorageDtype};
use crate::config::VindexLayerInfo;
use crate::error::VindexError;
use crate::format::filenames::*;

use super::build_helpers::{run_clustering_pipeline, ClusterData};

pub use crate::extract::callbacks::IndexBuildCallbacks;

pub(super) fn knowledge_layer_range(family: &str, num_layers: usize) -> Option<(usize, usize)> {
    crate::LayerBands::for_family(family, num_layers).map(|bands| {
        let start = bands.knowledge.0.min(num_layers);
        let end = bands.knowledge.1.saturating_add(1).min(num_layers);
        (start, end)
    })
}

// ═══════════════════════════════════════════════════════════════════════
// BuildContext — shared state across pipeline stages
// ═══════════════════════════════════════════════════════════════════════

/// Holds the inputs + accumulators for the build pipeline. Each stage
/// method on `BuildContext` reads inputs and mutates the accumulators
/// (`layer_infos`, `cluster_*`); the derived constants are set in `new`.
pub(super) struct BuildContext<'a> {
    // Inputs
    pub(super) weights: &'a ModelWeights,
    pub(super) tokenizer: &'a tokenizers::Tokenizer,
    pub(super) output_dir: &'a Path,
    pub(super) callbacks: &'a mut dyn IndexBuildCallbacks,
    pub(super) dtype: StorageDtype,
    pub(super) down_top_k: usize,

    // Derived constants
    pub(super) num_layers: usize,
    pub(super) hidden_size: usize,
    pub(super) intermediate_size: usize,
    pub(super) vocab_size: usize,
    pub(super) embed_scale: f32,
    pub(super) is_moe: bool,
    pub(super) n_experts: usize,

    // Stage 1 → Stage 6 (consumed by `write_index_json`)
    pub(super) layer_infos: Vec<VindexLayerInfo>,

    // Stage 3 collects → Stage 4 drains (`run_clustering`).
    pub(super) cluster_directions: Vec<f32>,
    pub(super) cluster_features: Vec<(usize, usize)>,
    pub(super) cluster_top_tokens: Vec<String>,
    pub(super) cluster_input_tokens: Vec<String>,
    pub(super) cluster_output_tokens: Vec<String>,

    /// Dense-only BitNet build: when set, `write_index_json` writes
    /// the weight manifest with attention + FFN projections skipped
    /// (only norms + embed + lm_head), since the I2_S projections
    /// live in the `bitnet/` artifacts.
    pub(super) dense_only: bool,
}

impl<'a> BuildContext<'a> {
    fn new(
        weights: &'a ModelWeights,
        tokenizer: &'a tokenizers::Tokenizer,
        output_dir: &'a Path,
        callbacks: &'a mut dyn IndexBuildCallbacks,
        dtype: StorageDtype,
        down_top_k: usize,
    ) -> Self {
        Self {
            num_layers: weights.num_layers,
            hidden_size: weights.hidden_size,
            intermediate_size: weights.intermediate_size,
            vocab_size: weights.vocab_size,
            embed_scale: weights.arch.embed_scale_multiplier(),
            is_moe: weights.arch.is_moe(),
            n_experts: weights.arch.num_experts(),
            weights,
            tokenizer,
            output_dir,
            callbacks,
            dtype,
            down_top_k,
            layer_infos: Vec::new(),
            cluster_directions: Vec::new(),
            cluster_features: Vec::new(),
            cluster_top_tokens: Vec::new(),
            cluster_input_tokens: Vec::new(),
            cluster_output_tokens: Vec::new(),
            dense_only: false,
        }
    }

    /// Stage 1 — write `gate_vectors.bin` (one matrix per layer; MoE
    /// concatenates each expert's matrix). Populates `layer_infos`.
    fn write_gate_vectors(&mut self) -> Result<(), VindexError> {
        self.callbacks.on_stage(STAGE_GATE_VECTORS);
        let gate_path = self.output_dir.join(GATE_VECTORS_BIN);
        let mut gate_file = BufWriter::new(std::fs::File::create(&gate_path)?);
        let mut offset: u64 = 0;

        for layer in 0..self.num_layers {
            self.callbacks
                .on_layer_start(COMP_GATE, layer, self.num_layers);
            let start = std::time::Instant::now();

            if self.is_moe && self.n_experts > 0 {
                // MoE: write each expert's gate matrix contiguously
                let mut total_features = 0usize;
                let mut layer_bytes = 0u64;
                let mut features_per_expert = 0usize;

                for expert in 0..self.n_experts {
                    let gate_key = match self.weights.arch.expert_ffn_gate_key(layer, expert) {
                        Some(k) => k,
                        None => continue,
                    };
                    let w_gate = match self.weights.tensors.get(&gate_key) {
                        Some(w) => w,
                        None => continue,
                    };
                    features_per_expert = w_gate.shape()[0];
                    total_features += features_per_expert;
                    let data = w_gate.as_slice().unwrap();
                    layer_bytes += write_floats(&mut gate_file, data, self.dtype)?;
                }

                // Also include shared expert if present
                if let Some(shared_key) = self.weights.arch.shared_expert_gate_key(layer) {
                    if let Some(w_gate) = self.weights.tensors.get(&shared_key) {
                        let n = w_gate.shape()[0];
                        total_features += n;
                        let data = w_gate.as_slice().unwrap();
                        layer_bytes += write_floats(&mut gate_file, data, self.dtype)?;
                    }
                }

                if total_features > 0 {
                    self.layer_infos.push(VindexLayerInfo {
                        layer,
                        num_features: total_features,
                        offset,
                        length: layer_bytes,
                        num_experts: Some(self.n_experts),
                        num_features_per_expert: Some(features_per_expert),
                    });
                    offset += layer_bytes;
                }
            } else {
                // Dense: single feature-input-direction matrix per layer.
                // Gated FFN routes through `ffn_gate`; non-gated FFN (GPT-2,
                // StarCoder2) reuses `ffn_up` for the same role.
                let gate_key = match self.weights.arch.ffn_type() {
                    FfnType::Gated => self.weights.arch.ffn_gate_key(layer),
                    FfnType::Standard => self.weights.arch.ffn_up_key(layer),
                };
                let w_gate = match self.weights.tensors.get(&gate_key) {
                    Some(w) => w,
                    None => continue,
                };
                let num_features = w_gate.shape()[0];
                let data = w_gate.as_slice().unwrap();
                let length = write_floats(&mut gate_file, data, self.dtype)?;
                self.layer_infos.push(VindexLayerInfo {
                    layer,
                    num_features,
                    offset,
                    length,
                    num_experts: None,
                    num_features_per_expert: None,
                });
                offset += length;
            }

            self.callbacks
                .on_layer_done(COMP_GATE, layer, start.elapsed().as_secs_f64() * 1000.0);
        }
        self.callbacks.on_stage_done(STAGE_GATE_VECTORS, 0.0);
        Ok(())
    }

    /// Stage 2 — write `embeddings.bin`.
    fn write_embeddings(&mut self) -> Result<(), VindexError> {
        self.callbacks.on_stage(STAGE_EMBEDDINGS);
        let embed_path = self.output_dir.join(EMBEDDINGS_BIN);
        let embed_data = self.weights.embed.as_slice().unwrap();
        let embed_bytes = crate::config::dtype::encode_floats(embed_data, self.dtype);
        std::fs::write(&embed_path, &embed_bytes)?;
        self.callbacks.on_stage_done(STAGE_EMBEDDINGS, 0.0);
        Ok(())
    }

    /// Stage 4 — k-means + label the collected cluster directions.
    /// Drains the `cluster_*` accumulators.
    fn run_clustering(&mut self) -> Result<(), VindexError> {
        run_clustering_pipeline(
            ClusterData {
                directions: std::mem::take(&mut self.cluster_directions),
                features: std::mem::take(&mut self.cluster_features),
                top_tokens: std::mem::take(&mut self.cluster_top_tokens),
                input_tokens: std::mem::take(&mut self.cluster_input_tokens),
                output_tokens: std::mem::take(&mut self.cluster_output_tokens),
            },
            self.hidden_size,
            self.weights,
            self.tokenizer,
            self.output_dir,
            self.callbacks,
        )
    }

    /// Stage 5 — copy the tokenizer JSON.
    fn write_tokenizer(&mut self) -> Result<(), VindexError> {
        self.callbacks.on_stage(STAGE_TOKENIZER);
        let tokenizer_json = self
            .tokenizer
            .to_string(true)
            .map_err(|e| VindexError::Parse(format!("tokenizer serialize: {e}")))?;
        std::fs::write(self.output_dir.join(TOKENIZER_JSON), tokenizer_json)?;
        self.callbacks.on_stage_done(STAGE_TOKENIZER, 0.0);
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Entry points
// ═══════════════════════════════════════════════════════════════════════

/// Build a .vindex from model weights and write it to disk.
///
/// Reads gate vectors and down projections directly from safetensors,
/// projects down vectors to vocabulary for top-k token metadata,
/// writes everything to a self-contained directory.
#[allow(clippy::too_many_arguments)]
pub fn build_vindex(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    model_name: &str,
    output_dir: &Path,
    down_top_k: usize,
    extract_level: crate::ExtractLevel,
    dtype: StorageDtype,
    callbacks: &mut dyn IndexBuildCallbacks,
) -> Result<(), VindexError> {
    // Refuse the extract before any output is created when the requested
    // tier needs attention tensors that the writer cannot represent
    // (e.g. MLA on the standard Q/K/V/O manifests). A late failure
    // inside the writer leaves a half-written vindex on disk.
    crate::format::weights::ensure_extract_level_supported(&*weights.arch, extract_level)?;

    std::fs::create_dir_all(output_dir)?;
    let mut ctx = BuildContext::new(weights, tokenizer, output_dir, callbacks, dtype, down_top_k);
    ctx.write_gate_vectors()?;
    ctx.write_embeddings()?;
    ctx.write_down_meta_and_clusters()?;
    ctx.run_clustering()?;
    ctx.write_tokenizer()?;
    ctx.write_index_json(model_name, extract_level)?;
    Ok(())
}

/// Build a *dense-only* BitNet vindex: embeddings + norms + lm_head +
/// tokenizer + index.json, skipping the gate-vector and clustering
/// stages entirely.
///
/// The walk-mode FFN (`write_gate_vectors`) and the
/// HNSW + Wikidata clustering (`run_clustering`) are the two
/// expensive stages of a normal inference-level build (they write
/// ~2 GB of f32 gate vectors and spend 20-30 min building per-layer
/// HNSW indices on a 2 B model).  Native-ternary BitNet inference
/// (`predict_bitnet`) does not use any of that: its forward pass
/// reads only the dense norms / embeddings / lm_head plus the
/// `bitnet/` I2_S artifacts (written separately by
/// `bitnet_writer::write_bitnet_artifacts`).  So for the
/// edge-deployable BitNet case we skip both stages.
///
/// The resulting vindex has **no** `gate_vectors.bin`; the server's
/// `VectorIndex::load_vindex_with_range` tolerates that (it loads a
/// degenerate empty gate index), and `load_bitnet_model` reads the
/// manifested dense weights with `skip_ffn = true`, so walk / browse
/// endpoints simply return nothing useful while `/v1/infer` (dense
/// mode) works at the native-ternary footprint (~1.4 GB resident:
/// embeddings F32 + I2_S BitLinears, no 2 GB gate matrix).
///
/// `--keep-quant` callers pass this for `--dense-only`; the BitNet
/// artifacts are stamped into index.json by the convert command
/// after this returns (same as the full path).
pub fn build_vindex_dense_only(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    model_name: &str,
    output_dir: &Path,
    dtype: StorageDtype,
    callbacks: &mut dyn IndexBuildCallbacks,
) -> Result<(), VindexError> {
    // Inference level so `write_index_json` calls `write_model_weights`
    // (norms + embed + lm_head into the manifest) and stamps
    // `has_model_weights = true`.
    let extract_level = crate::ExtractLevel::Inference;
    crate::format::weights::ensure_extract_level_supported(&*weights.arch, extract_level)?;

    std::fs::create_dir_all(output_dir)?;
    let mut ctx = BuildContext::new(weights, tokenizer, output_dir, callbacks, dtype, 0);
    ctx.dense_only = true;
    // Deliberately NO write_gate_vectors / write_down_meta_and_clusters
    // / run_clustering — leaves `layer_infos` empty, so index.json
    // carries zero gate layers and no gate_vectors.bin is produced.
    ctx.write_embeddings()?;
    ctx.write_tokenizer()?;
    ctx.write_index_json(model_name, extract_level)?;
    Ok(())
}

#[cfg(test)]
mod tests;
