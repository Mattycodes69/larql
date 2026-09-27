//! gRPC streaming DESCRIBE.

use crate::band_utils::PROBE_RELATION_SOURCE;
use tonic::Status;

#[allow(unused_imports)]
use super::*;

pub(super) fn grpc_stream_describe(
    model: &crate::state::LoadedModel,
    req: &DescribeRequest,
    tx: &tokio::sync::mpsc::Sender<Result<DescribeLayerEvent, Status>>,
) {
    let encoding = match model.tokenizer.encode(req.entity.as_str(), false) {
        Ok(e) => e,
        Err(_) => return,
    };
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();
    if token_ids.is_empty() {
        let _ = tx.blocking_send(Ok(DescribeLayerEvent {
            layer: 0,
            edges: vec![],
            done: true,
            total_edges: 0,
            latency_ms: 0.0,
        }));
        return;
    }

    let hidden = model.embeddings.shape()[1];
    let query = if token_ids.len() == 1 {
        model
            .embeddings
            .row(token_ids[0] as usize)
            .mapv(|v| v * model.embed_scale)
    } else {
        let mut avg = larql_vindex::ndarray::Array1::<f32>::zeros(hidden);
        for &tok in &token_ids {
            avg += &model
                .embeddings
                .row(tok as usize)
                .mapv(|v| v * model.embed_scale);
        }
        avg /= token_ids.len() as f32;
        avg
    };

    let start = std::time::Instant::now();
    let patched = model.patched.blocking_read();
    let all_layers = patched.loaded_layers();
    let entity_lower = req.entity.to_lowercase();
    let mut total_edges = 0u32;

    for &layer in &all_layers {
        let hits = patched.gate_knn(layer, &query, 20);
        let mut edges = Vec::new();

        for (feature, gate_score) in &hits {
            if *gate_score < 5.0 {
                continue;
            }
            if let Some(meta) = patched.feature_meta(layer, *feature) {
                let tok = meta.top_token.trim();
                if tok.is_empty() || tok.len() < 2 || tok.to_lowercase() == entity_lower {
                    continue;
                }
                let (relation, source) = model
                    .probe_labels
                    .get(&(layer, *feature))
                    .map(|r| (r.clone(), PROBE_RELATION_SOURCE.to_string()))
                    .unwrap_or_default();
                edges.push(DescribeEdge {
                    target: tok.to_string(),
                    gate_score: *gate_score,
                    layer: layer as u32,
                    relation,
                    source,
                    also: vec![],
                    layer_min: 0,
                    layer_max: 0,
                    count: 0,
                });
            }
        }

        total_edges += edges.len() as u32;

        if tx
            .blocking_send(Ok(DescribeLayerEvent {
                layer: layer as u32,
                edges,
                done: false,
                total_edges: 0,
                latency_ms: 0.0,
            }))
            .is_err()
        {
            return;
        }
    }

    let _ = tx.blocking_send(Ok(DescribeLayerEvent {
        layer: 0,
        edges: vec![],
        done: true,
        total_edges,
        latency_ms: start.elapsed().as_secs_f64() as f32 * 1000.0,
    }));
}
