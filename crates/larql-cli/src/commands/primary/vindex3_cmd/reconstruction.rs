//! `vindex3 sensitivity --reconstruction DIR`: the f32 tensors a compiled
//! representation would put in front of a matmul, written for an outside
//! engine.
//!
//! AUTO-REP-PRIOR-1 (`docs/auto-rep-prior-1.md`) scores MLX's gradient
//! prior against LARQL's codec, not MLX's. Its `W_low` must therefore be
//! exactly what LARQL executes, and the outside engine must be able to
//! prove it received that and nothing else.
//!
//! The values are `round_trip(source)`, i.e. `dequantize(quantize(source))`.
//! A compiled pack stores `encode(quantize(source))` and a reader decodes it
//! with the same `dequantize_into`, so the export and a decoded pack are one
//! function of the source. The manifest names the quantiser rather than
//! asserting the equivalence silently.
//!
//! One safetensors file per tensor, each holding a single `weight`, so a
//! reader can stream them one at a time rather than holding the whole stack.

use std::path::{Path, PathBuf};

use safetensors::tensor::TensorView;
use safetensors::Dtype;
use sha2::{Digest, Sha256};

/// The key each per-tensor file stores its values under.
pub(super) const WEIGHT_KEY: &str = "weight";

/// The manifest's name inside the export directory.
pub(super) const MANIFEST_NAME: &str = "manifest.json";

/// The function that produced the values, named in the manifest.
const QUANTISER: &str = "larql_models::quant::nvfp4::round_trip";

/// Manifest schema, bumped when a field changes meaning.
const MANIFEST_SCHEMA: &str = "larql.vindex3.reconstruction/v1";

#[derive(serde::Serialize)]
struct ExportedTensor {
    object: String,
    tensor: String,
    file: String,
    shape: Vec<usize>,
    /// SHA-256 of the file's bytes as written.
    file_sha256: String,
    /// `‖W − Q(W)‖² / ‖W‖²` over these exact values, so a reader can check
    /// its own copy of the source against the one this was made from.
    rel_error: f64,
}

#[derive(serde::Serialize)]
struct Manifest<'a> {
    schema: &'static str,
    container: String,
    quantiser: &'static str,
    codec: String,
    encoder: String,
    /// `payload_sha256` of every source representation read, by object.
    source_payloads: &'a std::collections::BTreeMap<String, String>,
    tensors: &'a [ExportedTensor],
}

/// Collects per-tensor files, then writes the manifest last so an
/// interrupted export has no manifest and cannot be mistaken for a complete
/// one.
pub(super) struct ReconstructionExport {
    dir: PathBuf,
    tensors: Vec<ExportedTensor>,
    source_payloads: std::collections::BTreeMap<String, String>,
}

impl ReconstructionExport {
    pub(super) fn create(dir: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        if dir.join(MANIFEST_NAME).exists() {
            return Err(format!(
                "{} already holds a reconstruction manifest; refusing to overwrite it",
                dir.display()
            )
            .into());
        }
        std::fs::create_dir_all(dir)?;
        Ok(Self {
            dir: dir.to_path_buf(),
            tensors: Vec::new(),
            source_payloads: Default::default(),
        })
    }

    pub(super) fn source(&mut self, object: &str, payload_sha256: &str) {
        self.source_payloads
            .insert(object.to_string(), payload_sha256.to_string());
    }

    pub(super) fn write(
        &mut self,
        object: &str,
        tensor: &str,
        shape: &[usize],
        values: &[f32],
        rel_error: f64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let view = TensorView::new(Dtype::F32, shape.to_vec(), &raw)
            .map_err(|e| format!("{tensor}: {e}"))?;
        let blob = safetensors::tensor::serialize([(WEIGHT_KEY, view)], None)
            .map_err(|e| format!("{tensor}: {e}"))?;
        let file = format!("{tensor}.safetensors");
        std::fs::write(self.dir.join(&file), &blob)?;
        self.tensors.push(ExportedTensor {
            object: object.to_string(),
            tensor: tensor.to_string(),
            file,
            shape: shape.to_vec(),
            file_sha256: hex(&Sha256::digest(&blob)),
            rel_error,
        });
        Ok(())
    }

    pub(super) fn finish(
        self,
        container: &Path,
        codec: String,
        encoder: String,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        let manifest = Manifest {
            schema: MANIFEST_SCHEMA,
            container: container.display().to_string(),
            quantiser: QUANTISER,
            codec,
            encoder,
            source_payloads: &self.source_payloads,
            tensors: &self.tensors,
        };
        std::fs::write(
            self.dir.join(MANIFEST_NAME),
            serde_json::to_string_pretty(&manifest)?,
        )?;
        Ok(self.tensors.len())
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
