//! Python graph types (Edge, Node, Graph) over `larql_core`.

use larql_core as lq;
use pyo3::types::PyDict;

#[allow(unused_imports)]
use super::*;

// ── Helpers ──

pub(super) fn parse_source(s: &str) -> lq::SourceType {
    match s {
        "parametric" => lq::SourceType::Parametric,
        "document" => lq::SourceType::Document,
        "installed" => lq::SourceType::Installed,
        "wikidata" => lq::SourceType::Wikidata,
        "manual" => lq::SourceType::Manual,
        _ => lq::SourceType::Unknown,
    }
}

pub(super) fn parse_merge_strategy(s: &str) -> lq::MergeStrategy {
    match s {
        "union" => lq::MergeStrategy::Union,
        "source_priority" => lq::MergeStrategy::SourcePriority,
        _ => lq::MergeStrategy::MaxConfidence,
    }
}

// ── PyEdge ──

/// A key a compact dict must carry; a missing one is a `KeyError`.
pub(super) fn required_item<'py>(d: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    d.get_item(key)?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(key.to_string()))
}

#[pyclass(name = "Edge", from_py_object)]
#[derive(Clone)]
pub struct PyEdge {
    pub(super) inner: lq::Edge,
}

#[pymethods]
impl PyEdge {
    #[new]
    #[pyo3(signature = (subject, relation, object, confidence=1.0, source="unknown", metadata=None, injection=None))]
    pub(super) fn new(
        subject: &str,
        relation: &str,
        object: &str,
        confidence: f64,
        source: &str,
        metadata: Option<&Bound<'_, PyDict>>,
        injection: Option<(usize, f64)>,
    ) -> PyResult<Self> {
        let mut edge = lq::Edge::new(subject, relation, object)
            .with_confidence(confidence)
            .with_source(parse_source(source));

        if let Some(meta) = metadata {
            for (k, v) in meta.iter() {
                let key: String = k.extract()?;
                let val_str: String = v.str()?.to_string();
                edge = edge.with_metadata(&key, serde_json::Value::String(val_str));
            }
        }

        edge.injection = injection;

        Ok(Self { inner: edge })
    }

    #[getter]
    pub(super) fn subject(&self) -> &str {
        &self.inner.subject
    }

    #[getter]
    pub(super) fn relation(&self) -> &str {
        &self.inner.relation
    }

    #[getter]
    pub(super) fn object(&self) -> &str {
        &self.inner.object
    }

    #[getter]
    pub(super) fn confidence(&self) -> f64 {
        self.inner.confidence
    }

    #[getter]
    pub(super) fn source(&self) -> &str {
        self.inner.source.as_str()
    }

    #[getter]
    pub(super) fn metadata<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        match &self.inner.metadata {
            None => Ok(None),
            Some(meta) => {
                let json_str = serde_json::to_string(meta)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                let json_mod = py.import("json")?;
                let result = json_mod.call_method1("loads", (json_str,))?;
                Ok(Some(result))
            }
        }
    }

    #[getter]
    pub(super) fn injection(&self) -> Option<(usize, f64)> {
        self.inner.injection
    }

    pub(super) fn triple(&self) -> (String, String, String) {
        let t = self.inner.triple();
        (t.0, t.1, t.2)
    }

    pub(super) fn to_compact<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("s", &self.inner.subject)?;
        dict.set_item("r", &self.inner.relation)?;
        dict.set_item("o", &self.inner.object)?;
        dict.set_item("c", self.inner.confidence)?;
        if self.inner.source != lq::SourceType::Unknown {
            dict.set_item("src", self.inner.source.as_str())?;
        }
        if let Some(meta) = &self.inner.metadata {
            let json_str = serde_json::to_string(meta)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            let json_mod = py.import("json")?;
            let parsed = json_mod.call_method1("loads", (json_str,))?;
            dict.set_item("meta", parsed)?;
        }
        if let Some(inj) = self.inner.injection {
            dict.set_item("inj", vec![inj.0 as f64, inj.1])?;
        }
        Ok(dict)
    }

    #[staticmethod]
    pub(super) fn from_compact(d: &Bound<'_, PyDict>) -> PyResult<PyEdge> {
        let s: String = required_item(d, "s")?.extract()?;
        let r: String = required_item(d, "r")?.extract()?;
        let o: String = required_item(d, "o")?.extract()?;
        let c: f64 = d
            .get_item("c")?
            .map(|v| v.extract().unwrap_or(1.0))
            .unwrap_or(1.0);
        let src: String = d
            .get_item("src")?
            .map(|v| v.extract().unwrap_or_else(|_| "unknown".to_string()))
            .unwrap_or_else(|| "unknown".to_string());

        let mut edge = lq::Edge::new(s, r, o)
            .with_confidence(c)
            .with_source(parse_source(&src));

        // Parse metadata dict
        if let Some(meta_obj) = d.get_item("meta")? {
            let json_mod = d.py().import("json")?;
            let meta_str: String = json_mod.call_method1("dumps", (meta_obj,))?.extract()?;
            if let Ok(meta_map) = serde_json::from_str::<
                std::collections::HashMap<String, serde_json::Value>,
            >(&meta_str)
            {
                edge.metadata = Some(meta_map);
            }
        }

        // Parse injection
        if let Some(inj) = d.get_item("inj")? {
            let vals: Vec<f64> = inj.extract()?;
            if vals.len() == 2 {
                edge.injection = Some((vals[0] as usize, vals[1]));
            }
        }

        Ok(PyEdge { inner: edge })
    }

    pub(super) fn __repr__(&self) -> String {
        format!(
            "{} --{}--> {} ({:.2})",
            self.inner.subject, self.inner.relation, self.inner.object, self.inner.confidence
        )
    }

    pub(super) fn __eq__(&self, other: &PyEdge) -> bool {
        self.inner == other.inner
    }

    pub(super) fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.inner.hash(&mut hasher);
        hasher.finish()
    }
}

// ── PyNode ──

#[pyclass(name = "Node", from_py_object)]
#[derive(Clone)]
pub struct PyNode {
    pub(super) inner: lq::core::node::Node,
}

#[pymethods]
impl PyNode {
    #[getter]
    pub(super) fn name(&self) -> &str {
        &self.inner.name
    }

    /// Returns the node type string, or "unknown" if not inferred.
    /// Matches Python NodeType enum values.
    #[getter]
    pub(super) fn node_type(&self) -> &str {
        self.inner.node_type.as_deref().unwrap_or("unknown")
    }

    #[getter]
    pub(super) fn degree(&self) -> usize {
        self.inner.degree
    }

    #[getter]
    pub(super) fn out_degree(&self) -> usize {
        self.inner.out_degree
    }

    #[getter]
    pub(super) fn in_degree(&self) -> usize {
        self.inner.in_degree
    }

    pub(super) fn __repr__(&self) -> String {
        let ntype = self.inner.node_type.as_deref().unwrap_or("unknown");
        format!(
            "Node({}, type={}, degree={})",
            self.inner.name, ntype, self.inner.degree
        )
    }
}

// ── PyGraph ──

#[pyclass(name = "Graph", unsendable)]
pub struct PyGraph {
    pub(super) inner: lq::Graph,
}

#[pymethods]
impl PyGraph {
    #[new]
    pub(super) fn new() -> Self {
        Self {
            inner: lq::Graph::new(),
        }
    }

    // ── Construction ──

    pub(super) fn add_edge(&mut self, edge: &PyEdge) {
        self.inner.add_edge(edge.inner.clone());
    }

    pub(super) fn add_edges(&mut self, edges: Vec<PyEdge>) {
        self.inner.add_edges(edges.into_iter().map(|e| e.inner));
    }

    pub(super) fn remove_edge(&mut self, subject: &str, relation: &str, object_: &str) -> bool {
        self.inner.remove_edge(subject, relation, object_)
    }

    #[pyo3(signature = (strategy="max_confidence"))]
    pub(super) fn deduplicate(&mut self, strategy: &str) -> usize {
        self.inner.deduplicate(parse_merge_strategy(strategy))
    }

    // ── Queries ──

    #[pyo3(signature = (subject, relation=None))]
    pub(super) fn select(&self, subject: &str, relation: Option<&str>) -> Vec<PyEdge> {
        self.inner
            .select(subject, relation)
            .into_iter()
            .map(|e| PyEdge { inner: e.clone() })
            .collect()
    }

    #[pyo3(signature = (object_, relation=None))]
    pub(super) fn select_reverse(&self, object_: &str, relation: Option<&str>) -> Vec<PyEdge> {
        self.inner
            .select_reverse(object_, relation)
            .into_iter()
            .map(|e| PyEdge { inner: e.clone() })
            .collect()
    }

    /// Matches Python: returns {"entity", "type", "outgoing", "incoming"}
    pub(super) fn describe<'py>(
        &self,
        py: Python<'py>,
        entity: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        let result = self.inner.describe(entity);
        let node_type = self
            .inner
            .node(entity)
            .and_then(|n| n.node_type)
            .unwrap_or_else(|| "unknown".to_string());

        let dict = PyDict::new(py);
        dict.set_item("entity", &result.entity)?;
        dict.set_item("type", node_type)?;
        dict.set_item(
            "outgoing",
            result
                .outgoing
                .into_iter()
                .map(|e| PyEdge { inner: e })
                .collect::<Vec<_>>(),
        )?;
        dict.set_item(
            "incoming",
            result
                .incoming
                .into_iter()
                .map(|e| PyEdge { inner: e })
                .collect::<Vec<_>>(),
        )?;
        Ok(dict)
    }

    /// Matches Python: exists(subject, relation=None, object_=None)
    #[pyo3(signature = (subject, relation=None, object_=None))]
    pub(super) fn exists(
        &self,
        subject: &str,
        relation: Option<&str>,
        object_: Option<&str>,
    ) -> bool {
        match (relation, object_) {
            (Some(r), Some(o)) => self.inner.exists(subject, r, o),
            (Some(r), None) => !self.inner.select(subject, Some(r)).is_empty(),
            (None, Some(o)) => self
                .inner
                .select(subject, None)
                .iter()
                .any(|e| e.object == o),
            (None, None) => !self.inner.select(subject, None).is_empty(),
        }
    }

    /// Matches Python: returns (None, path) on failure instead of None
    pub(super) fn walk(
        &self,
        subject: &str,
        relations: Vec<String>,
    ) -> (Option<String>, Vec<PyEdge>) {
        let refs: Vec<&str> = relations.iter().map(|s| s.as_str()).collect();
        match self.inner.walk(subject, &refs) {
            Some((dest, path)) => (
                Some(dest),
                path.into_iter().map(|e| PyEdge { inner: e }).collect(),
            ),
            None => (None, Vec::new()),
        }
    }

    #[pyo3(signature = (query, max_results=10))]
    pub(super) fn search(&self, query: &str, max_results: usize) -> Vec<PyEdge> {
        self.inner
            .search(query, max_results)
            .into_iter()
            .map(|e| PyEdge { inner: e.clone() })
            .collect()
    }

    #[pyo3(signature = (entity, depth=2))]
    pub(super) fn subgraph(&self, entity: &str, depth: u32) -> PyGraph {
        PyGraph {
            inner: self.inner.subgraph(entity, depth),
        }
    }

    // ── Count / Node ──

    #[pyo3(signature = (relation=None, source=None))]
    pub(super) fn count(&self, relation: Option<&str>, source: Option<&str>) -> usize {
        let source_type = source.map(parse_source);
        self.inner.count(relation, source_type.as_ref())
    }

    pub(super) fn node(&self, name: &str) -> Option<PyNode> {
        self.inner.node(name).map(|n| PyNode { inner: n })
    }

    pub(super) fn nodes(&self) -> Vec<PyNode> {
        self.inner
            .nodes()
            .into_iter()
            .map(|n| PyNode { inner: n })
            .collect()
    }

    // ── Accessors ──

    #[getter]
    pub(super) fn edge_count(&self) -> usize {
        self.inner.edge_count()
    }

    #[getter]
    pub(super) fn node_count(&self) -> usize {
        self.inner.node_count()
    }

    #[getter]
    pub(super) fn edges(&self) -> Vec<PyEdge> {
        self.inner
            .edges()
            .iter()
            .map(|e| PyEdge { inner: e.clone() })
            .collect()
    }

    /// Matches Python: list_entities(entity_type=None)
    #[pyo3(signature = (entity_type=None))]
    pub(super) fn list_entities(&self, entity_type: Option<&str>) -> Vec<String> {
        match entity_type {
            None => self.inner.list_entities(),
            Some(t) => self
                .inner
                .nodes()
                .into_iter()
                .filter(|n| n.node_type.as_deref().unwrap_or("unknown") == t)
                .map(|n| n.name)
                .collect(),
        }
    }

    pub(super) fn list_relations(&self) -> Vec<String> {
        self.inner.list_relations()
    }

    // ── Stats ──

    pub(super) fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let s = self.inner.stats();
        let dict = PyDict::new(py);
        dict.set_item("entities", s.entities)?;
        dict.set_item("edges", s.edges)?;
        dict.set_item("relations", s.relations)?;
        dict.set_item("avg_confidence", s.avg_confidence)?;
        dict.set_item("connected_components", s.connected_components)?;
        dict.set_item("avg_degree", s.avg_degree)?;
        let sources = PyDict::new(py);
        for (k, v) in &s.sources {
            sources.set_item(k, v)?;
        }
        dict.set_item("sources", sources)?;
        Ok(dict)
    }

    // ── Serialization ──

    pub(super) fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let json_val = self.inner.to_json_value();
        let json_str = serde_json::to_string(&json_val)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let json_mod = py.import("json")?;
        json_mod.call_method1("loads", (json_str,))
    }

    #[staticmethod]
    pub(super) fn from_dict(data: &Bound<'_, PyAny>) -> PyResult<PyGraph> {
        let json_mod = data.py().import("json")?;
        let json_str: String = json_mod.call_method1("dumps", (data,))?.extract()?;
        let json_val: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let graph = lq::Graph::from_json_value(&json_val)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(PyGraph { inner: graph })
    }

    // ── Dunder ──

    pub(super) fn __len__(&self) -> usize {
        self.inner.edge_count()
    }

    pub(super) fn __repr__(&self) -> String {
        format!(
            "Graph(edges={}, nodes={})",
            self.inner.edge_count(),
            self.inner.node_count()
        )
    }

    pub(super) fn __contains__(&self, edge: &PyEdge) -> bool {
        self.inner.exists(
            &edge.inner.subject,
            &edge.inner.relation,
            &edge.inner.object,
        )
    }
}
