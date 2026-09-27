//! Python bindings for the vindex — the queryable model format.
//!
//! Exposes VectorIndex + embeddings + tokenizer as a single Python object
//! with numpy array returns for gate vectors, embeddings, and KNN results.
//!
//! Two access patterns:
//! - Direct API: gate_vector(), embed(), gate_knn() — raw numpy arrays
//! - High-level: describe(), entity_knn(), insert() — string in, results out

use ndarray::Array1;
use pyo3::prelude::*;

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

#[pyclass(name = "Vindex", unsendable)]
pub struct PyVindex {
    pub(crate) index: VectorIndex,
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
    /// Lazy-loaded mmap'd weights for infer(). Created on first call, reused after.
    pub(crate) walk_model: std::cell::RefCell<Option<crate::walk::InferState>>,
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
            index,
            embeddings,
            embed_scale,
            tokenizer,
            config,
            path: path.to_string(),
            classifier,
            knn_store,
            walk_model: std::cell::RefCell::new(None),
        })
    }

    /// Run a closure with a mutable reference to the lazily-loaded walk FFN state.
    /// Loads on first call; subsequent calls reuse the mmap'd weights.
    fn with_walk_model<F, R>(&self, f: F) -> PyResult<R>
    where
        F: FnOnce(&mut crate::walk::InferState) -> PyResult<R>,
    {
        {
            let mut state = self.walk_model.borrow_mut();
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
        }
        let mut state = self.walk_model.borrow_mut();
        f(state.as_mut().unwrap())
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
    fn load(path: &str) -> PyResult<Self> {
        Self::open(path)
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
    fn is_mmap(&self) -> bool {
        self.index.is_mmap()
    }

    #[getter]
    fn total_gate_vectors(&self) -> usize {
        self.index.total_gate_vectors()
    }

    #[getter]
    fn loaded_layers(&self) -> Vec<usize> {
        self.index.loaded_layers()
    }

    #[getter]
    fn embed_scale_value(&self) -> f32 {
        self.embed_scale
    }

    /// Number of features at a layer.
    fn num_features(&self, layer: usize) -> usize {
        self.index.num_features(layer)
    }
}
