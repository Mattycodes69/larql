//! On-disk residual capture pool for the DEC replay loadgen.
//!
//! A capture directory holds two files:
//!   * `manifest.json` — [`CaptureManifest`]: model identity, shape, and
//!     per-prompt step counts.
//!   * `residuals.bin` — f32 LE, layout `[prompt][step][layer][hidden]`,
//!     `steps` = the minimum step count across prompts (ragged tails are
//!     truncated at write time so every `(step, layer)` cell has a row from
//!     every prompt).
//!
//! Rows are captured **pre-normed** (see `ResidualCaptureSink`), i.e. exactly
//! the bytes the f32 wire carries and exactly the input Q8K quantisation
//! consumes — so replay needs no model weights and the pool is portable
//! across hosts (Mac capture → x86 replay).
//!
//! ## Optional routed-experts sidecars (additive, manifest stays v1)
//!
//! A pool captured with `--routing` additionally holds:
//!   * `raw.bin`     — raw post-attention residuals, same
//!     `[prompt][step][layer][hidden]` f32-LE layout as `residuals.bin`
//!     (the routed f32 endpoints want this; the server applies
//!     pre_experts_norm itself).
//!   * `normed.bin`  — pre-experts-normed residuals, same layout (input to
//!     the routed q8k frame quantisation).
//!   * `routing.bin` — `[prompt][step][layer][top_k × (u32 expert_id LE +
//!     f32 weight LE)]`. Fixed `top_k` records; layers with no MoE routing
//!     hold `top_k` sentinel entries ([`ROUTING_SENTINEL_EXPERT`], 0.0).
//!     Zero-weight pairs are stripped at write time (they cost nothing
//!     server-side; accounting for them would overcount): real pairs are
//!     compacted to the front of the record and the tail is sentinel-padded,
//!     so readers take the prefix before the first sentinel.
//!
//! The manifest gains an *optional* `routing` block
//! (`{ top_k, has_raw, has_normed }`). Absent block = walk-ffn-only pool:
//! the shipped 330M pool keeps opening and replaying unchanged
//! (`CaptureManifest` tolerates unknown fields, so old binaries also open
//! routed pools and simply ignore the sidecars).

use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::Path;

/// Bump when the binary layout changes. `open` rejects other versions.
/// The routing sidecars are additive — version stays 1.
pub const CAPTURE_VERSION: u32 = 1;

const MANIFEST_FILE: &str = "manifest.json";
const RESIDUALS_FILE: &str = "residuals.bin";
const RAW_FILE: &str = "raw.bin";
const NORMED_FILE: &str = "normed.bin";
const ROUTING_FILE: &str = "routing.bin";

/// Expert-id sentinel in `routing.bin`: marks an unused slot in a fixed
/// `top_k`-record (non-MoE layer, or a slot freed by zero-weight stripping).
pub const ROUTING_SENTINEL_EXPERT: u32 = u32::MAX;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptMeta {
    pub id: usize,
    pub text: String,
    /// Steps this prompt actually produced before truncation to `steps`.
    pub steps_captured: usize,
}

/// Optional manifest block describing the routed-experts sidecars.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingManifest {
    /// Fixed record length of `routing.bin` (entries per `[prompt][step][layer]`
    /// cell). From the model arch's `num_experts_per_token` at capture time.
    pub top_k: usize,
    /// `raw.bin` present.
    pub has_raw: bool,
    /// `normed.bin` present.
    pub has_normed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptureManifest {
    pub version: u32,
    /// Model path / id the pool was captured from.
    pub model: String,
    pub hidden_size: usize,
    pub num_layers: usize,
    /// Replayable steps: min over prompts of steps captured.
    pub steps: usize,
    /// Always `"f32-le"` at version 1.
    pub dtype: String,
    pub prompts: Vec<PromptMeta>,
    pub created_unix: u64,
    /// Routed-experts sidecar block. `None` = walk-ffn-only pool (the
    /// pre-routing format — old pools deserialise to `None`, and the block
    /// is omitted from JSON on write so non-routed pools stay byte-stable).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub routing: Option<RoutingManifest>,
}

impl CaptureManifest {
    /// Expected byte length for `residuals.bin` — and for `raw.bin` /
    /// `normed.bin`, which share the exact layout.
    pub fn expected_bytes(&self) -> u64 {
        self.prompts.len() as u64
            * self.steps as u64
            * self.num_layers as u64
            * self.hidden_size as u64
            * 4
    }

    /// Expected `routing.bin` byte length (`None` when the pool has no
    /// routing block). 8 bytes per entry: u32 expert id + f32 weight.
    pub fn expected_routing_bytes(&self) -> Option<u64> {
        self.routing.as_ref().map(|r| {
            self.prompts.len() as u64
                * self.steps as u64
                * self.num_layers as u64
                * r.top_k as u64
                * 8
        })
    }
}

/// Routed-experts capture input for [`CapturePool::write_with_routing`].
/// All planes use the same `[prompt][step][layer]` nesting as `per_prompt`;
/// `routing[p][s][l]` holds up to `top_k` `(expert_id, weight)` pairs
/// (empty for layers with no MoE routing).
pub struct RoutingCapture {
    pub top_k: usize,
    /// Raw post-attention residual rows (length = hidden each).
    pub raw: Vec<Vec<Vec<Vec<f32>>>>,
    /// Pre-experts-normed residual rows (length = hidden each).
    pub normed: Vec<Vec<Vec<Vec<f32>>>>,
    /// Per-layer routing pairs, final post-renorm post-scale weights.
    pub routing: Vec<Vec<Vec<Vec<(u32, f32)>>>>,
}

/// An open, validated capture pool.
pub struct CapturePool {
    pub manifest: CaptureManifest,
    /// Raw `residuals.bin` contents; decoded to f32 on demand in [`Self::rows`].
    data: Vec<u8>,
    // The routed planes and their accessors are consumed by the Phase C
    // routed replay arm (Endpoint seam); until it lands, unit tests are the
    // only readers, so the bin target sees them as dead code.
    /// `raw.bin` contents when the pool has routing with `has_raw`.
    #[allow(dead_code)]
    raw: Option<Vec<u8>>,
    /// `normed.bin` contents when the pool has routing with `has_normed`.
    #[allow(dead_code)]
    normed: Option<Vec<u8>>,
    /// Decoded `routing.bin` (fixed `top_k` records incl. sentinels).
    #[allow(dead_code)]
    routing: Option<Vec<(u32, f32)>>,
}

impl CapturePool {
    /// Write a pool from per-prompt captured steps.
    ///
    /// `per_prompt[p][step][layer]` is the pre-normed residual for prompt `p`.
    /// Prompts with more steps than the shortest are truncated so the binary
    /// layout is rectangular; the pre-truncation count is kept in
    /// [`PromptMeta::steps_captured`].
    #[allow(dead_code)] // Phase C routed-replay consumer pending; tests pin it
    pub fn write(
        dir: &Path,
        model: &str,
        hidden_size: usize,
        num_layers: usize,
        prompt_texts: &[String],
        per_prompt: &[Vec<Vec<Vec<f32>>>],
        created_unix: u64,
    ) -> Result<CaptureManifest, String> {
        Self::write_with_routing(
            dir,
            model,
            hidden_size,
            num_layers,
            prompt_texts,
            per_prompt,
            None,
            created_unix,
        )
    }

    /// [`Self::write`] plus the optional routed-experts sidecars. With
    /// `routing: None` this is exactly the walk-ffn-only write path.
    #[allow(clippy::too_many_arguments)]
    pub fn write_with_routing(
        dir: &Path,
        model: &str,
        hidden_size: usize,
        num_layers: usize,
        prompt_texts: &[String],
        per_prompt: &[Vec<Vec<Vec<f32>>>],
        routing: Option<&RoutingCapture>,
        created_unix: u64,
    ) -> Result<CaptureManifest, String> {
        if per_prompt.is_empty() {
            return Err("capture pool: no prompts captured".into());
        }
        if per_prompt.len() != prompt_texts.len() {
            return Err(format!(
                "capture pool: {} prompt texts but {} capture sinks",
                prompt_texts.len(),
                per_prompt.len()
            ));
        }
        let steps = per_prompt.iter().map(|s| s.len()).min().unwrap_or(0);
        if steps == 0 {
            return Err("capture pool: a prompt produced zero decode steps".into());
        }
        for (p, prompt_steps) in per_prompt.iter().enumerate() {
            for (s, layers) in prompt_steps.iter().take(steps).enumerate() {
                if layers.len() != num_layers {
                    return Err(format!(
                        "capture pool: prompt {p} step {s} has {} layers, expected {num_layers}",
                        layers.len()
                    ));
                }
                for (l, row) in layers.iter().enumerate() {
                    if row.len() != hidden_size {
                        return Err(format!(
                            "capture pool: prompt {p} step {s} layer {l} has {} floats, \
                             expected hidden {hidden_size}",
                            row.len()
                        ));
                    }
                }
            }
        }
        if let Some(rc) = routing {
            validate_routing_capture(rc, per_prompt.len(), steps, num_layers, hidden_size)?;
        }

        let manifest = CaptureManifest {
            version: CAPTURE_VERSION,
            model: model.to_string(),
            hidden_size,
            num_layers,
            steps,
            dtype: "f32-le".into(),
            prompts: prompt_texts
                .iter()
                .enumerate()
                .map(|(id, text)| PromptMeta {
                    id,
                    text: text.clone(),
                    steps_captured: per_prompt[id].len(),
                })
                .collect(),
            created_unix,
            routing: routing.map(|rc| RoutingManifest {
                top_k: rc.top_k,
                has_raw: true,
                has_normed: true,
            }),
        };

        std::fs::create_dir_all(dir).map_err(|e| format!("capture pool: mkdir: {e}"))?;
        write_plane(&dir.join(RESIDUALS_FILE), per_prompt, steps)?;
        if let Some(rc) = routing {
            write_plane(&dir.join(RAW_FILE), &rc.raw, steps)?;
            write_plane(&dir.join(NORMED_FILE), &rc.normed, steps)?;
            write_routing_file(&dir.join(ROUTING_FILE), rc, steps)?;
        }

        let manifest_json = serde_json::to_string_pretty(&manifest)
            .map_err(|e| format!("capture pool: manifest serialize: {e}"))?;
        std::fs::write(dir.join(MANIFEST_FILE), manifest_json)
            .map_err(|e| format!("capture pool: write manifest: {e}"))?;
        Ok(manifest)
    }

    /// Open and validate a pool directory.
    pub fn open(dir: &Path) -> Result<Self, String> {
        let manifest_path = dir.join(MANIFEST_FILE);
        let manifest_json = std::fs::read_to_string(&manifest_path)
            .map_err(|e| format!("capture pool: read {}: {e}", manifest_path.display()))?;
        let manifest: CaptureManifest = serde_json::from_str(&manifest_json)
            .map_err(|e| format!("capture pool: parse manifest: {e}"))?;
        if manifest.version != CAPTURE_VERSION {
            return Err(format!(
                "capture pool: version {} unsupported (expected {CAPTURE_VERSION})",
                manifest.version
            ));
        }
        let read_exact_len = |file: &str, expect: u64| -> Result<Vec<u8>, String> {
            let path = dir.join(file);
            let data = std::fs::read(&path)
                .map_err(|e| format!("capture pool: read {}: {e}", path.display()))?;
            if data.len() as u64 != expect {
                return Err(format!(
                    "capture pool: {file} is {} bytes, manifest expects {expect}",
                    data.len(),
                ));
            }
            Ok(data)
        };
        let data = read_exact_len(RESIDUALS_FILE, manifest.expected_bytes())?;

        let (raw, normed, routing) = match &manifest.routing {
            None => (None, None, None),
            Some(rb) => {
                if rb.top_k == 0 {
                    return Err("capture pool: routing block has top_k 0".into());
                }
                let raw = if rb.has_raw {
                    Some(read_exact_len(RAW_FILE, manifest.expected_bytes())?)
                } else {
                    None
                };
                let normed = if rb.has_normed {
                    Some(read_exact_len(NORMED_FILE, manifest.expected_bytes())?)
                } else {
                    None
                };
                let expect = manifest
                    .expected_routing_bytes()
                    .expect("routing block present");
                let bytes = read_exact_len(ROUTING_FILE, expect)?;
                let decoded: Vec<(u32, f32)> = bytes
                    .chunks_exact(8)
                    .map(|c| {
                        (
                            u32::from_le_bytes(c[..4].try_into().unwrap()),
                            f32::from_le_bytes(c[4..].try_into().unwrap()),
                        )
                    })
                    .collect();
                (raw, normed, Some(decoded))
            }
        };

        Ok(Self {
            manifest,
            data,
            raw,
            normed,
            routing,
        })
    }

    /// Number of prompts in the pool — the maximum replay batch size.
    pub fn num_prompts(&self) -> usize {
        self.manifest.prompts.len()
    }

    /// Whether the pool carries the routed-experts sidecars.
    #[allow(dead_code)] // Phase C routed-replay consumer pending; tests pin it
    pub fn has_routing(&self) -> bool {
        self.routing.is_some()
    }

    /// Fixed `routing.bin` record length (`None` for walk-ffn-only pools).
    #[allow(dead_code)] // Phase C routed-replay consumer pending; tests pin it
    pub fn routing_top_k(&self) -> Option<usize> {
        self.manifest.routing.as_ref().map(|r| r.top_k)
    }

    /// Bounds-check a `(batch, step, layer)` cell request.
    fn check_cell(&self, batch: usize, step: usize, layer: usize) -> Result<(), String> {
        let m = &self.manifest;
        if batch == 0 || batch > m.prompts.len() {
            return Err(format!(
                "capture pool: batch {batch} out of range (pool has {} prompts)",
                m.prompts.len()
            ));
        }
        if step >= m.steps {
            return Err(format!(
                "capture pool: step {step} out of range (pool has {} steps)",
                m.steps
            ));
        }
        if layer >= m.num_layers {
            return Err(format!(
                "capture pool: layer {layer} out of range (pool has {} layers)",
                m.num_layers
            ));
        }
        Ok(())
    }

    /// Decode a `batch × hidden` row block for `(step, layer)` out of one
    /// `[prompt][step][layer][hidden]` plane.
    fn plane_rows(
        &self,
        plane: &[u8],
        batch: usize,
        step: usize,
        layer: usize,
    ) -> Result<Vec<f32>, String> {
        self.check_cell(batch, step, layer)?;
        let m = &self.manifest;
        let row_bytes = m.hidden_size * 4;
        let mut out = Vec::with_capacity(batch * m.hidden_size);
        for prompt in 0..batch {
            let row_idx = (prompt * m.steps + step) * m.num_layers + layer;
            let start = row_idx * row_bytes;
            out.extend(
                plane[start..start + row_bytes]
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap())),
            );
        }
        Ok(out)
    }

    /// Build a `batch × hidden` contiguous row block for `(step, layer)`:
    /// row `i` is prompt `i`'s pre-normed residual at that step and layer.
    /// Distinct prompts per row keep MoE routing union realistic.
    pub fn rows(&self, batch: usize, step: usize, layer: usize) -> Result<Vec<f32>, String> {
        self.plane_rows(&self.data, batch, step, layer)
    }

    /// [`Self::rows`]-shaped block of **raw post-attention** residuals from
    /// `raw.bin`. Errors on walk-ffn-only pools (or `has_raw: false`).
    #[allow(dead_code)] // Phase C routed-replay consumer pending; tests pin it
    pub fn raw_rows(&self, batch: usize, step: usize, layer: usize) -> Result<Vec<f32>, String> {
        let plane = self
            .raw
            .as_ref()
            .ok_or("capture pool: raw.bin not present (walk-ffn-only pool)")?;
        self.plane_rows(plane, batch, step, layer)
    }

    /// [`Self::rows`]-shaped block of **pre-experts-normed** residuals from
    /// `normed.bin`. Errors on walk-ffn-only pools (or `has_normed: false`).
    #[allow(dead_code)] // Phase C routed-replay consumer pending; tests pin it
    pub fn normed_rows(&self, batch: usize, step: usize, layer: usize) -> Result<Vec<f32>, String> {
        let plane = self
            .normed
            .as_ref()
            .ok_or("capture pool: normed.bin not present (walk-ffn-only pool)")?;
        self.plane_rows(plane, batch, step, layer)
    }

    /// Routing pairs for one `(prompt, step, layer)` cell:
    /// `Ok(Some(pairs))` — the non-sentinel `(expert_id, weight)` prefix
    /// (zero-weight pairs were stripped at write time), `Ok(None)` — layer
    /// carried no MoE routing (all-sentinel record), `Err` — walk-ffn-only
    /// pool or out-of-range cell.
    #[allow(dead_code)] // Phase C routed-replay consumer pending; tests pin it
    pub fn routing(
        &self,
        prompt: usize,
        step: usize,
        layer: usize,
    ) -> Result<Option<&[(u32, f32)]>, String> {
        let entries = self
            .routing
            .as_ref()
            .ok_or("capture pool: routing.bin not present (walk-ffn-only pool)")?;
        if prompt >= self.manifest.prompts.len() {
            return Err(format!(
                "capture pool: prompt {prompt} out of range (pool has {} prompts)",
                self.manifest.prompts.len()
            ));
        }
        // Reuse the shared step/layer bounds checks (batch=1 is always valid).
        self.check_cell(1, step, layer)?;
        let m = &self.manifest;
        let k = self
            .manifest
            .routing
            .as_ref()
            .expect("routing data implies manifest block")
            .top_k;
        let base = ((prompt * m.steps + step) * m.num_layers + layer) * k;
        let record = &entries[base..base + k];
        let n = record
            .iter()
            .position(|&(id, _)| id == ROUTING_SENTINEL_EXPERT)
            .unwrap_or(k);
        Ok(if n == 0 { None } else { Some(&record[..n]) })
    }
}

/// Validate one routed sidecar plane set against the pool shape.
fn validate_routing_capture(
    rc: &RoutingCapture,
    num_prompts: usize,
    steps: usize,
    num_layers: usize,
    hidden_size: usize,
) -> Result<(), String> {
    if rc.top_k == 0 {
        return Err("capture pool: routing capture has top_k 0".into());
    }
    for (name, plane) in [("raw", &rc.raw), ("normed", &rc.normed)] {
        if plane.len() != num_prompts {
            return Err(format!(
                "capture pool: routing {name} plane has {} prompts, expected {num_prompts}",
                plane.len()
            ));
        }
        for (p, prompt_steps) in plane.iter().enumerate() {
            if prompt_steps.len() < steps {
                return Err(format!(
                    "capture pool: routing {name} plane prompt {p} has {} steps, needs ≥ {steps}",
                    prompt_steps.len()
                ));
            }
            for (s, layers) in prompt_steps.iter().take(steps).enumerate() {
                if layers.len() != num_layers {
                    return Err(format!(
                        "capture pool: routing {name} plane prompt {p} step {s} has {} layers, \
                         expected {num_layers}",
                        layers.len()
                    ));
                }
                for (l, row) in layers.iter().enumerate() {
                    if row.len() != hidden_size {
                        return Err(format!(
                            "capture pool: routing {name} plane prompt {p} step {s} layer {l} \
                             has {} floats, expected hidden {hidden_size}",
                            row.len()
                        ));
                    }
                }
            }
        }
    }
    if rc.routing.len() != num_prompts {
        return Err(format!(
            "capture pool: routing pairs have {} prompts, expected {num_prompts}",
            rc.routing.len()
        ));
    }
    for (p, prompt_steps) in rc.routing.iter().enumerate() {
        if prompt_steps.len() < steps {
            return Err(format!(
                "capture pool: routing pairs prompt {p} has {} steps, needs ≥ {steps}",
                prompt_steps.len()
            ));
        }
        for (s, layers) in prompt_steps.iter().take(steps).enumerate() {
            if layers.len() != num_layers {
                return Err(format!(
                    "capture pool: routing pairs prompt {p} step {s} has {} layers, \
                     expected {num_layers}",
                    layers.len()
                ));
            }
            for (l, pairs) in layers.iter().enumerate() {
                if pairs.len() > rc.top_k {
                    return Err(format!(
                        "capture pool: prompt {p} step {s} layer {l} has {} routing pairs, \
                         top_k is {}",
                        pairs.len(),
                        rc.top_k
                    ));
                }
                if pairs.iter().any(|&(id, _)| id == ROUTING_SENTINEL_EXPERT) {
                    return Err(format!(
                        "capture pool: prompt {p} step {s} layer {l} uses reserved expert id \
                         {ROUTING_SENTINEL_EXPERT}"
                    ));
                }
            }
        }
    }
    Ok(())
}

/// Write one `[prompt][step][layer][hidden]` f32-LE plane, truncated to
/// `steps` per prompt. Shared by `residuals.bin` / `raw.bin` / `normed.bin`.
fn write_plane(path: &Path, per_prompt: &[Vec<Vec<Vec<f32>>>], steps: usize) -> Result<(), String> {
    let f = std::fs::File::create(path)
        .map_err(|e| format!("capture pool: create {}: {e}", path.display()))?;
    let mut w = std::io::BufWriter::new(f);
    for prompt_steps in per_prompt {
        for layers in prompt_steps.iter().take(steps) {
            for row in layers {
                for &v in row {
                    w.write_all(&v.to_le_bytes())
                        .map_err(|e| format!("capture pool: write: {e}"))?;
                }
            }
        }
    }
    w.flush().map_err(|e| format!("capture pool: flush: {e}"))
}

/// Write `routing.bin`: fixed `top_k` records; zero-weight pairs stripped
/// (compacted out) and the record sentinel-padded to length.
fn write_routing_file(path: &Path, rc: &RoutingCapture, steps: usize) -> Result<(), String> {
    let f = std::fs::File::create(path)
        .map_err(|e| format!("capture pool: create {}: {e}", path.display()))?;
    let mut w = std::io::BufWriter::new(f);
    let mut put = |id: u32, weight: f32| -> Result<(), String> {
        w.write_all(&id.to_le_bytes())
            .and_then(|()| w.write_all(&weight.to_le_bytes()))
            .map_err(|e| format!("capture pool: write: {e}"))
    };
    for prompt_steps in &rc.routing {
        for layers in prompt_steps.iter().take(steps) {
            for pairs in layers {
                let mut written = 0usize;
                for &(id, weight) in pairs {
                    if weight != 0.0 {
                        put(id, weight)?;
                        written += 1;
                    }
                }
                for _ in written..rc.top_k {
                    put(ROUTING_SENTINEL_EXPERT, 0.0)?;
                }
            }
        }
    }
    w.flush().map_err(|e| format!("capture pool: flush: {e}"))
}

#[cfg(test)]
mod tests;
