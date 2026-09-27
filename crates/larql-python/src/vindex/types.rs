//! Python value types returned by the Vindex API.

use larql_vindex::{FeatureMeta, WalkHit};
use pyo3::types::PyDict;

#[allow(unused_imports)]
use super::*;

// ── PyDescribeEdge ──

#[pyclass(name = "DescribeEdge", from_py_object)]
#[derive(Clone)]
pub struct PyDescribeEdge {
    #[pyo3(get)]
    pub relation: Option<String>,
    #[pyo3(get)]
    pub source: String,
    #[pyo3(get)]
    pub target: String,
    #[pyo3(get)]
    pub gate_score: f32,
    #[pyo3(get)]
    pub layer: usize,
    #[pyo3(get)]
    pub feature: usize,
    #[pyo3(get)]
    pub confidence: f32,
    #[pyo3(get)]
    pub also: Vec<String>,
}

#[pymethods]
impl PyDescribeEdge {
    pub(super) fn __repr__(&self) -> String {
        let rel = self.relation.as_deref().unwrap_or("?");
        format!(
            "DescribeEdge(relation='{}', target='{}', score={:.1}, layer={}, source='{}')",
            rel, self.target, self.gate_score, self.layer, self.source
        )
    }
}

// ── PyRelation ──

#[pyclass(name = "Relation", from_py_object)]
#[derive(Clone)]
pub struct PyRelation {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub cluster_id: usize,
    #[pyo3(get)]
    pub count: usize,
    #[pyo3(get)]
    pub top_tokens: Vec<String>,
}

#[pymethods]
impl PyRelation {
    pub(super) fn __repr__(&self) -> String {
        format!(
            "Relation(name='{}', count={}, cluster={})",
            self.name, self.count, self.cluster_id
        )
    }
}

// ── PyFeatureMeta ──

#[pyclass(name = "FeatureMeta", from_py_object)]
#[derive(Clone)]
pub struct PyFeatureMeta {
    pub(super) inner: FeatureMeta,
}

#[pymethods]
impl PyFeatureMeta {
    #[getter]
    pub(super) fn top_token(&self) -> &str {
        &self.inner.top_token
    }

    #[getter]
    pub(super) fn top_token_id(&self) -> u32 {
        self.inner.top_token_id
    }

    #[getter]
    pub(super) fn c_score(&self) -> f32 {
        self.inner.c_score
    }

    #[getter]
    pub(super) fn top_k<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let mut result = Vec::new();
        for entry in &self.inner.top_k {
            let dict = PyDict::new(py);
            dict.set_item("token", &entry.token)?;
            dict.set_item("token_id", entry.token_id)?;
            dict.set_item("logit", entry.logit)?;
            result.push(dict);
        }
        Ok(result)
    }

    pub(super) fn __repr__(&self) -> String {
        format!(
            "FeatureMeta(token='{}', id={}, c={:.4})",
            self.inner.top_token, self.inner.top_token_id, self.inner.c_score
        )
    }
}

// ── PyWalkHit ──

#[pyclass(name = "WalkHit", from_py_object)]
#[derive(Clone)]
pub struct PyWalkHit {
    pub(super) inner_layer: usize,
    pub(super) inner_feature: usize,
    pub(super) inner_gate_score: f32,
    pub(super) inner_meta: FeatureMeta,
    // Runtime-trace fields (2026-07-30 review, item 17): populated
    // when the hit comes from an executed walk trace, `None` on
    // post-hoc KNN views such as `Vindex.walk`.
    pub(super) inner_up_score: Option<f32>,
    pub(super) inner_activation: Option<f32>,
    pub(super) inner_down_row_norm: Option<f32>,
    pub(super) inner_rank: Option<usize>,
}

#[pymethods]
impl PyWalkHit {
    #[getter]
    pub(super) fn layer(&self) -> usize {
        self.inner_layer
    }

    #[getter]
    pub(super) fn feature(&self) -> usize {
        self.inner_feature
    }

    #[getter]
    pub(super) fn gate_score(&self) -> f32 {
        self.inner_gate_score
    }

    #[getter]
    pub(super) fn meta(&self) -> PyFeatureMeta {
        PyFeatureMeta {
            inner: self.inner_meta.clone(),
        }
    }

    #[getter]
    pub(super) fn up_score(&self) -> Option<f32> {
        self.inner_up_score
    }

    #[getter]
    pub(super) fn activation(&self) -> Option<f32> {
        self.inner_activation
    }

    #[getter]
    pub(super) fn down_row_norm(&self) -> Option<f32> {
        self.inner_down_row_norm
    }

    #[getter]
    pub(super) fn rank(&self) -> Option<usize> {
        self.inner_rank
    }

    #[getter]
    pub(super) fn top_token(&self) -> &str {
        &self.inner_meta.top_token
    }

    #[getter]
    pub(super) fn target(&self) -> &str {
        &self.inner_meta.top_token
    }

    pub(super) fn __repr__(&self) -> String {
        format!(
            "WalkHit(L{}:F{} score={:.4} token='{}')",
            self.inner_layer, self.inner_feature, self.inner_gate_score, self.inner_meta.top_token
        )
    }
}

impl From<WalkHit> for PyWalkHit {
    fn from(h: WalkHit) -> Self {
        Self {
            inner_layer: h.layer,
            inner_feature: h.feature,
            inner_gate_score: h.gate_score,
            inner_meta: h.meta,
            inner_up_score: h.up_score,
            inner_activation: h.activation,
            inner_down_row_norm: h.down_row_norm,
            inner_rank: h.rank,
        }
    }
}
