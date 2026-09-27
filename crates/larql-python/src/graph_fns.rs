//! Top-level graph functions: I/O, merge, diff, traversal, and weight/attention walks.

use larql_core as lq;
use larql_vindex as lv;
use pyo3::types::PyDict;

#[allow(unused_imports)]
use super::*;

// ── Free functions ──

#[pyfunction]
pub(super) fn load(path: &str) -> PyResult<PyGraph> {
    let graph = lq::load(path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    Ok(PyGraph { inner: graph })
}

#[pyfunction]
pub(super) fn save(graph: &PyGraph, path: &str) -> PyResult<()> {
    lq::save(&graph.inner, path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
}

#[pyfunction]
pub(super) fn shortest_path(graph: &PyGraph, from: &str, to: &str) -> Option<(f64, Vec<PyEdge>)> {
    lq::shortest_path(&graph.inner, from, to).map(|(cost, path)| {
        (
            cost,
            path.into_iter().map(|e| PyEdge { inner: e }).collect(),
        )
    })
}

#[pyfunction]
pub(super) fn merge_graphs(target: &mut PyGraph, other: &PyGraph) -> usize {
    lq::merge_graphs(&mut target.inner, &other.inner)
}

#[pyfunction]
#[pyo3(signature = (target, other, strategy="union"))]
pub(super) fn merge_graphs_with_strategy(
    target: &mut PyGraph,
    other: &PyGraph,
    strategy: &str,
) -> usize {
    let s = match strategy {
        "max_confidence" => lq::MergeStrategy::MaxConfidence,
        "source_priority" => lq::MergeStrategy::SourcePriority,
        _ => lq::MergeStrategy::Union,
    };
    lq::merge_graphs_with_strategy(&mut target.inner, &other.inner, s)
}

#[pyfunction]
pub(super) fn diff<'py>(
    py: Python<'py>,
    old: &PyGraph,
    new: &PyGraph,
) -> PyResult<Bound<'py, PyDict>> {
    let result = lq::diff(&old.inner, &new.inner);
    let dict = PyDict::new(py);
    dict.set_item(
        "added",
        result
            .added
            .into_iter()
            .map(|e| PyEdge { inner: e })
            .collect::<Vec<_>>(),
    )?;
    dict.set_item(
        "removed",
        result
            .removed
            .into_iter()
            .map(|e| PyEdge { inner: e })
            .collect::<Vec<_>>(),
    )?;
    dict.set_item("changed", result.changed.len())?;
    Ok(dict)
}

#[pyfunction]
#[pyo3(signature = (graph, damping=0.85, max_iterations=100, tolerance=1e-6))]
pub(super) fn pagerank<'py>(
    py: Python<'py>,
    graph: &PyGraph,
    damping: f64,
    max_iterations: usize,
    tolerance: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let result = lq::pagerank(&graph.inner, damping, max_iterations, tolerance);
    let dict = PyDict::new(py);
    let ranks = PyDict::new(py);
    for (k, v) in &result.ranks {
        ranks.set_item(k, v)?;
    }
    dict.set_item("ranks", ranks)?;
    dict.set_item("iterations", result.iterations)?;
    dict.set_item("converged", result.converged)?;
    Ok(dict)
}

#[pyfunction]
#[pyo3(signature = (graph, source, max_depth=10))]
pub(super) fn bfs_traversal(
    graph: &PyGraph,
    source: &str,
    max_depth: usize,
) -> (Vec<String>, Vec<PyEdge>) {
    let result = lq::bfs_traversal(&graph.inner, source, max_depth);
    (
        result.nodes,
        result
            .edges
            .into_iter()
            .map(|e| PyEdge { inner: e })
            .collect(),
    )
}

#[pyfunction]
#[pyo3(signature = (graph, source, max_depth=10))]
pub(super) fn dfs_traversal(
    graph: &PyGraph,
    source: &str,
    max_depth: usize,
) -> (Vec<String>, Vec<PyEdge>) {
    let result = lq::dfs(&graph.inner, source, max_depth);
    (
        result.nodes,
        result
            .edges
            .into_iter()
            .map(|e| PyEdge { inner: e })
            .collect(),
    )
}

#[pyfunction]
pub(super) fn load_csv(path: &str) -> PyResult<PyGraph> {
    let graph =
        lq::load_csv(path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    Ok(PyGraph { inner: graph })
}

#[pyfunction]
pub(super) fn save_csv(graph: &PyGraph, path: &str) -> PyResult<()> {
    lq::save_csv(&graph.inner, path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
}

/// Walk FFN weights from a model directory. Returns a Graph of extracted edges.
///
/// Args:
///     model_path: Path to model directory (safetensors + tokenizer.json)
///     output_path: Where to save the result (.larql.json or .larql.bin)
///     layer: Optional single layer to walk (default: all)
///     top_k: Top-k tokens per feature (default: 5)
///     min_score: Minimum activation score (default: 0.02)
#[pyfunction]
#[pyo3(signature = (model_path, output_path=None, layer=None, top_k=5, min_score=0.02))]
pub(super) fn weight_walk(
    model_path: &str,
    output_path: Option<&str>,
    layer: Option<usize>,
    top_k: usize,
    min_score: f32,
) -> PyResult<PyGraph> {
    let config = lv::WalkConfig { top_k, min_score };
    let layers: Option<Vec<usize>> = layer.map(|l| vec![l]);

    let mut graph = lq::Graph::new();
    graph.metadata.insert(
        "model".to_string(),
        serde_json::Value::String(model_path.to_string()),
    );
    graph.metadata.insert(
        "method".to_string(),
        serde_json::Value::String("weight-walk".to_string()),
    );

    let mut callbacks = lv::walker::weight_walker::SilentWalkCallbacks;

    lv::walk_model(
        model_path,
        layers.as_deref(),
        &config,
        &mut graph,
        &mut callbacks,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    if let Some(path) = output_path {
        lq::save(&graph, path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    }

    Ok(PyGraph { inner: graph })
}

/// Walk attention OV circuits from a model. Returns a Graph of routing edges.
#[pyfunction]
#[pyo3(signature = (model_path, output_path=None, layer=None, top_k=3, min_score=0.0))]
pub(super) fn attention_walk(
    model_path: &str,
    output_path: Option<&str>,
    layer: Option<usize>,
    top_k: usize,
    min_score: f32,
) -> PyResult<PyGraph> {
    let walker = lv::AttentionWalker::load(model_path)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let config = lv::WalkConfig { top_k, min_score };

    let mut graph = lq::Graph::new();
    graph.metadata.insert(
        "model".to_string(),
        serde_json::Value::String(model_path.to_string()),
    );
    graph.metadata.insert(
        "method".to_string(),
        serde_json::Value::String("attention-walk".to_string()),
    );

    let layers: Vec<usize> = match layer {
        Some(l) => vec![l],
        None => (0..walker.num_layers()).collect(),
    };

    let mut callbacks = lv::walker::weight_walker::SilentWalkCallbacks;
    for &l in &layers {
        walker
            .walk_layer(l, &config, &mut graph, &mut callbacks)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    }

    if let Some(path) = output_path {
        lq::save(&graph, path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    }

    Ok(PyGraph { inner: graph })
}
