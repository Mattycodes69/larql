//! gRPC walk-ffn: gate-KNN features and full FFN output.

use crate::routes::limits;
use tonic::Status;

#[allow(unused_imports)]
use super::*;

pub(super) fn grpc_walk_ffn(
    model: &crate::state::LoadedModel,
    req: &WalkFfnRequest,
) -> Result<WalkFfnResponse, Status> {
    let start = std::time::Instant::now();
    let hidden = model.config.hidden_size;
    let seq_len = if req.seq_len == 0 {
        1
    } else {
        req.seq_len as usize
    };

    let expected_len = if req.full_output {
        seq_len
            .checked_mul(hidden)
            .ok_or_else(|| Status::invalid_argument("seq_len * hidden overflow"))?
    } else {
        hidden
    };
    if req.residual.len() != expected_len {
        return Err(Status::invalid_argument(format!(
            "residual has {} elements, expected {expected_len} (seq_len={} * hidden={hidden})",
            req.residual.len(),
            if req.full_output { seq_len } else { 1 },
        )));
    }

    let scan_layers: Vec<usize> = if !req.layers.is_empty() {
        req.layers.iter().map(|l| *l as usize).collect()
    } else {
        vec![req.layer as usize]
    };

    let results = if req.full_output {
        grpc_walk_ffn_full_output(model, &scan_layers, &req.residual, seq_len, hidden)?
    } else {
        let top_k = limits::proto_count(
            "top_k",
            req.top_k,
            crate::routes::walk_ffn::types::DEFAULT_WALK_FFN_TOP_K,
            limits::MAX_TOP_K,
        )
        .map_err(Status::invalid_argument)?;
        grpc_walk_ffn_features_only(model, &scan_layers, &req.residual, top_k)
    };

    Ok(WalkFfnResponse {
        results,
        latency_ms: start.elapsed().as_secs_f64() as f32 * 1000.0,
    })
}

fn grpc_walk_ffn_features_only(
    model: &crate::state::LoadedModel,
    scan_layers: &[usize],
    residual: &[f32],
    top_k: usize,
) -> Vec<WalkFfnLayerResult> {
    let patched = model.patched.blocking_read();
    let query = larql_vindex::ndarray::Array1::from_vec(residual.to_vec());

    scan_layers
        .iter()
        .map(|&layer| {
            let hits = patched.gate_knn(layer, &query, top_k);
            WalkFfnLayerResult {
                layer: layer as u32,
                features: hits.iter().map(|(f, _)| *f as u32).collect(),
                scores: hits.iter().map(|(_, s)| *s).collect(),
                output: Vec::new(),
                seq_len: 0,
            }
        })
        .collect()
}

fn grpc_walk_ffn_full_output(
    model: &crate::state::LoadedModel,
    scan_layers: &[usize],
    residual: &[f32],
    seq_len: usize,
    hidden: usize,
) -> Result<Vec<WalkFfnLayerResult>, Status> {
    use larql_inference::ffn::FfnBackend;
    use larql_vindex::ndarray::Array2;

    let weights_guard = model
        .get_or_load_weights()
        .map_err(Status::failed_precondition)?;
    let weights: &larql_inference::ModelWeights = &weights_guard;

    let patched = model.patched.blocking_read();
    let walk_ffn = larql_inference::vindex::WalkFfn::new_unlimited(weights, &*patched);

    let x = Array2::from_shape_vec((seq_len, hidden), residual.to_vec())
        .map_err(|e| Status::internal(format!("reshape residual: {e}")))?;

    let mut results = Vec::with_capacity(scan_layers.len());
    for &layer in scan_layers {
        if layer >= model.config.num_layers {
            return Err(Status::invalid_argument(format!(
                "layer {layer} out of range (num_layers = {})",
                model.config.num_layers
            )));
        }
        let out = walk_ffn.forward(layer, &x);
        let output: Vec<f32> = out.into_iter().collect();
        debug_assert_eq!(output.len(), seq_len * hidden);
        results.push(WalkFfnLayerResult {
            layer: layer as u32,
            features: Vec::new(),
            scores: Vec::new(),
            output,
            seq_len: seq_len as u32,
        });
    }
    Ok(results)
}
