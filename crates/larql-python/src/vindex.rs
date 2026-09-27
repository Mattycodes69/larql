//! Python bindings for the vindex — the queryable model format.
//!
//! Exposes VectorIndex + embeddings + tokenizer as a single Python object
//! with numpy array returns for gate vectors, embeddings, and KNN results.
//!
//! Two access patterns:
//! - Direct API: gate_vector(), embed(), gate_knn() — raw numpy arrays
//! - High-level: describe(), entity_knn(), insert() — string in, results out
//!
//! Threading: every call that touches the index runs inside
//! `Python::detach`, so other Python threads keep running while a KNN,
//! DESCRIBE or INFER executes. The index sits behind an `RwLock` (INSERT and
//! the `set_*` primitives take the write side) and the lazily loaded
//! inference weights behind a `Mutex`, because the GIL no longer serialises
//! those accesses. See [`crate::sync`] for the locking rules.

use ndarray::Array1;
use pyo3::prelude::*;
use std::sync::{Mutex, RwLock};

use larql_vindex::patch::knn_store::KnnStore;
use larql_vindex::{
    format::filenames::KNN_STORE_BIN, load_vindex_config, load_vindex_embeddings,
    load_vindex_tokenizer, tokenizers, SilentLoadCallbacks, VectorIndex, VindexConfig,
};

use larql_lql::relations::RelationClassifier;

mod describe;
mod infer;
mod lookup;
mod mutation;
mod types;
pub use types::*;

/// Auto-labelled clusters sometimes carry a long slash-joined token
/// list (`"a/b/c/…"`) instead of a relation name; `relations()` hides them.
const PATH_LIKE_LABEL_SEPARATOR: char = '/';
const PATH_LIKE_LABEL_MIN_LEN: usize = 20;

fn is_path_like_label(label: &str) -> bool {
    label.contains(PATH_LIKE_LABEL_SEPARATOR) && label.len() > PATH_LIKE_LABEL_MIN_LEN
}

// ── PyVindex ──

#[pyclass(name = "Vindex", frozen)]
pub struct PyVindex {
    /// Read by every query; written by INSERT / DELETE / `set_*`.
    pub(crate) index: RwLock<VectorIndex>,
    pub(crate) embeddings: ndarray::Array2<f32>,
    pub(crate) embed_scale: f32,
    pub(crate) tokenizer: tokenizers::Tokenizer,
    pub(crate) config: VindexConfig,
    pub(crate) path: String,
    pub(crate) classifier: Option<RelationClassifier>,
    /// Arch-B retrieval-override store. Loaded from `knn_store.bin` at
    /// open time if present. `infer()` captures residuals and consults
    /// this store before returning the raw model prediction; a stored
    /// key with `cos > KNN_COSINE_THRESHOLD` overrides the top-1
    /// prediction with the stored target token. Matches the LQL INFER
    /// query path (`executor/query/infer.rs`).
    pub(crate) knn_store: Option<KnnStore>,
    /// Lazy-loaded mmap'd weights for infer(). Created on first call, reused
    /// after. A `Mutex` because `InferenceWeights::infer_patched` takes
    /// `&mut self`, so concurrent INFER calls on one Vindex serialise here.
    pub(crate) walk_model: Mutex<Option<crate::walk::InferState>>,
}

impl PyVindex {
    /// Load a vindex from a directory path (Rust-callable).
    pub fn open(path: &str) -> PyResult<Self> {
        let dir = std::path::Path::new(path);

        let config = load_vindex_config(dir)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        let mut callbacks = SilentLoadCallbacks;
        let index = VectorIndex::load_vindex(dir, &mut callbacks)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        let (embeddings, embed_scale) = load_vindex_embeddings(dir)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        let tokenizer = load_vindex_tokenizer(dir)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        // Load relation classifier (clusters + labels) if available
        let classifier = RelationClassifier::from_vindex(dir);

        // Load the arch-B KNN store if the compiled vindex bundled one.
        let knn_path = dir.join(KNN_STORE_BIN);
        let knn_store = if knn_path.exists() {
            match KnnStore::load(&knn_path) {
                Ok(store) => Some(store),
                Err(e) => {
                    eprintln!("warning: failed to load knn_store.bin: {e}");
                    None
                }
            }
        } else {
            None
        };

        Ok(Self {
            index: RwLock::new(index),
            embeddings,
            embed_scale,
            tokenizer,
            config,
            path: path.to_string(),
            classifier,
            knn_store,
            walk_model: Mutex::new(None),
        })
    }

    /// Load a vindex with the GIL released (the load mmaps and parses
    /// every index file).
    pub fn open_detached(py: Python<'_>, path: &str) -> PyResult<Self> {
        py.detach(|| Self::open(path))
    }

    /// Run `f` against the index under the read lock, GIL released.
    pub(crate) fn read_index<R, F>(&self, py: Python<'_>, f: F) -> R
    where
        R: Send,
        F: FnOnce(&VectorIndex) -> R + Send,
    {
        py.detach(|| f(&crate::sync::read(&self.index)))
    }

    /// Run `f` against the index under the write lock, GIL released. The
    /// whole closure is one critical section, so a read-modify-write (find
    /// a free slot, then fill it) is atomic with respect to other threads.
    pub(crate) fn write_index<R, F>(&self, py: Python<'_>, f: F) -> R
    where
        R: Send,
        F: FnOnce(&mut VectorIndex) -> R + Send,
    {
        py.detach(|| f(&mut crate::sync::write(&self.index)))
    }

    /// Run `f` with the lazily-loaded inference weights and the index, GIL
    /// released. Loads the weights on first call; later calls reuse them.
    fn with_walk_model<R, F>(&self, py: Python<'_>, f: F) -> PyResult<R>
    where
        R: Send,
        F: FnOnce(&mut crate::walk::InferState, &VectorIndex) -> PyResult<R> + Send,
    {
        py.detach(|| {
            let mut state = crate::sync::lock(&self.walk_model);
            if state.is_none() {
                let dir = std::path::Path::new(&self.path);
                *state = Some(
                    crate::walk::InferState::load(dir, &self.config).map_err(|e| {
                        pyo3::exceptions::PyRuntimeError::new_err(format!(
                            "Failed to load model weights: {e}"
                        ))
                    })?,
                );
            }
            let index = crate::sync::read(&self.index);
            f(
                state.as_mut().expect("inference weights loaded above"),
                &index,
            )
        })
    }

    /// Compute scaled embedding for entity text. Multi-token entities are averaged.
    fn compute_embed(&self, text: &str) -> PyResult<Array1<f32>> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let ids = encoding.get_ids();
        if ids.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Empty tokenization",
            ));
        }

        let hidden = self.config.hidden_size;
        let mut sum = Array1::<f32>::zeros(hidden);
        let mut count = 0usize;

        for &tid in ids {
            let id = tid as usize;
            if id < self.embeddings.shape()[0] {
                sum += &self.embeddings.row(id);
                count += 1;
            }
        }

        if count == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "No valid token IDs",
            ));
        }

        let avg = sum / count as f32;
        Ok(avg * self.embed_scale)
    }
}

#[pymethods]
impl PyVindex {
    /// Load a vindex from a directory path.
    #[staticmethod]
    fn load(py: Python<'_>, path: &str) -> PyResult<Self> {
        Self::open_detached(py, path)
    }

    // ══════════════════════════════════════════════
    //  Properties
    // ══════════════════════════════════════════════

    #[getter]
    fn num_layers(&self) -> usize {
        self.config.num_layers
    }

    #[getter]
    fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }

    #[getter]
    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    #[getter]
    fn model(&self) -> &str {
        &self.config.model
    }

    #[getter]
    fn family(&self) -> &str {
        &self.config.family
    }

    #[getter]
    fn is_mmap(&self, py: Python<'_>) -> bool {
        self.read_index(py, |index| index.is_mmap())
    }

    #[getter]
    fn total_gate_vectors(&self, py: Python<'_>) -> usize {
        self.read_index(py, |index| index.total_gate_vectors())
    }

    #[getter]
    fn loaded_layers(&self, py: Python<'_>) -> Vec<usize> {
        self.read_index(py, |index| index.loaded_layers())
    }

    #[getter]
    fn embed_scale_value(&self) -> f32 {
        self.embed_scale
    }

    /// Number of features at a layer.
    fn num_features(&self, py: Python<'_>, layer: usize) -> usize {
        self.read_index(py, |index| index.num_features(layer))
    }
}
