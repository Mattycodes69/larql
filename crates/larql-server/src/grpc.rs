//! gRPC service implementation for VindexService.

use crate::routes::limits;
use std::sync::Arc;

use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};

use crate::band_utils::{
    HEALTH_STATUS_OK, INFER_MODE_COMPARE, INFER_MODE_DENSE, INFER_MODE_WALK, PROBE_RELATION_SOURCE,
};
use crate::state::AppState;

pub mod proto {
    tonic::include_proto!("vindex");
}

use proto::vindex_service_server::VindexService;
use proto::*;

mod stream;
mod walk_ffn;
use stream::*;
use walk_ffn::*;

pub struct VindexGrpcService {
    pub state: Arc<AppState>,
}

impl VindexGrpcService {
    /// The bound VINDEX2 model, in gRPC's vocabulary: nothing bound is
    /// `NotFound`, a VINDEX3 container is `Unimplemented` naming the
    /// generation — never `NotFound`, which would present a bound model
    /// as absent.
    fn v2_model(&self) -> Result<Arc<crate::state::LoadedModel>, Status> {
        self.state.v2_or_unsupported(None).map_err(|e| match e {
            crate::error::ServerError::Unsupported(msg) => Status::unimplemented(msg),
            other => Status::not_found(other.message().to_string()),
        })
    }
}

#[tonic::async_trait]
impl VindexService for VindexGrpcService {
    async fn health(
        &self,
        _request: Request<HealthRequest>,
    ) -> Result<Response<HealthResponse>, Status> {
        self.state.bump_requests();
        let uptime = self.state.started_at.elapsed().as_secs();
        let served = self
            .state
            .requests_served
            .load(std::sync::atomic::Ordering::Relaxed);
        Ok(Response::new(HealthResponse {
            status: HEALTH_STATUS_OK.into(),
            uptime_seconds: uptime,
            requests_served: served,
        }))
    }

    async fn get_stats(
        &self,
        _request: Request<StatsRequest>,
    ) -> Result<Response<StatsResponse>, Status> {
        self.state.bump_requests();
        let model = self.v2_model()?;

        let config = &model.config;
        let total_features: usize = config.layers.iter().map(|l| l.num_features).sum();
        let fpl = config.layers.first().map(|l| l.num_features).unwrap_or(0);

        let has_inference = config.extract_level == larql_vindex::ExtractLevel::Inference
            || config.extract_level == larql_vindex::ExtractLevel::All
            || config.has_model_weights;

        let bands = config.layer_bands.as_ref().map(|b| LayerBands {
            syntax: vec![b.syntax.0 as u32, b.syntax.1 as u32],
            knowledge: vec![b.knowledge.0 as u32, b.knowledge.1 as u32],
            output: vec![b.output.0 as u32, b.output.1 as u32],
        });

        Ok(Response::new(StatsResponse {
            model: config.model.clone(),
            family: config.family.clone(),
            layers: config.num_layers as u32,
            features: total_features as u32,
            features_per_layer: fpl as u32,
            hidden_size: config.hidden_size as u32,
            vocab_size: config.vocab_size as u32,
            extract_level: config.extract_level.to_string(),
            dtype: config.dtype.to_string(),
            layer_bands: bands,
            loaded: Some(LoadedStatus {
                browse: true,
                inference: has_inference && !model.infer_disabled,
            }),
        }))
    }

    async fn describe(
        &self,
        request: Request<DescribeRequest>,
    ) -> Result<Response<DescribeResponse>, Status> {
        self.state.bump_requests();
        let req = request.into_inner();
        let model = self.v2_model()?;

        let result = tokio::task::spawn_blocking(move || grpc_describe(&model, &req))
            .await
            .map_err(|e| Status::internal(e.to_string()))??;

        Ok(Response::new(result))
    }

    async fn walk(&self, request: Request<WalkRequest>) -> Result<Response<WalkResponse>, Status> {
        self.state.bump_requests();
        let req = request.into_inner();
        let model = self.v2_model()?;

        let result = tokio::task::spawn_blocking(move || grpc_walk(&model, &req))
            .await
            .map_err(|e| Status::internal(e.to_string()))??;

        Ok(Response::new(result))
    }

    async fn select(
        &self,
        request: Request<SelectRequest>,
    ) -> Result<Response<SelectResponse>, Status> {
        self.state.bump_requests();
        let req = request.into_inner();
        let model = self.v2_model()?;

        let result = tokio::task::spawn_blocking(move || grpc_select(&model, &req))
            .await
            .map_err(|e| Status::internal(e.to_string()))??;

        Ok(Response::new(result))
    }

    async fn infer(
        &self,
        request: Request<InferRequest>,
    ) -> Result<Response<InferResponse>, Status> {
        self.state.bump_requests();
        let req = request.into_inner();
        let model = self.v2_model()?;

        if model.infer_disabled {
            return Err(Status::unavailable("inference disabled (--no-infer)"));
        }

        let result = tokio::task::spawn_blocking(move || grpc_infer(&model, &req))
            .await
            .map_err(|e| Status::internal(e.to_string()))??;

        Ok(Response::new(result))
    }

    async fn get_relations(
        &self,
        _request: Request<RelationsRequest>,
    ) -> Result<Response<RelationsResponse>, Status> {
        self.state.bump_requests();
        let model = self.v2_model()?;

        let result = tokio::task::spawn_blocking(move || grpc_relations(&model))
            .await
            .map_err(|e| Status::internal(e.to_string()))??;

        Ok(Response::new(result))
    }

    async fn walk_ffn(
        &self,
        request: Request<WalkFfnRequest>,
    ) -> Result<Response<WalkFfnResponse>, Status> {
        self.state.bump_requests();
        let req = request.into_inner();
        let model = self.v2_model()?;

        let result = tokio::task::spawn_blocking(move || grpc_walk_ffn(&model, &req))
            .await
            .map_err(|e| Status::internal(e.to_string()))??;

        Ok(Response::new(result))
    }

    type StreamDescribeStream = ReceiverStream<Result<DescribeLayerEvent, Status>>;

    async fn stream_describe(
        &self,
        request: Request<DescribeRequest>,
    ) -> Result<Response<Self::StreamDescribeStream>, Status> {
        self.state.bump_requests();
        let req = request.into_inner();
        let model = self.v2_model()?;

        let (tx, rx) = tokio::sync::mpsc::channel(64);

        tokio::task::spawn_blocking(move || {
            grpc_stream_describe(&model, &req, &tx);
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }
}

/// Compare two f32 scores in **descending** order. NaN compares as `Equal`
/// rather than panicking — corrupted vindex data or future patched scoring
/// paths must not be able to take a gRPC worker down via `sort_by`.
#[inline]
fn cmp_score_desc(a: f32, b: f32) -> std::cmp::Ordering {
    b.partial_cmp(&a).unwrap_or(std::cmp::Ordering::Equal)
}

// ── Blocking handler implementations ──

fn grpc_describe(
    model: &crate::state::LoadedModel,
    req: &DescribeRequest,
) -> Result<DescribeResponse, Status> {
    let start = std::time::Instant::now();

    let encoding = model
        .tokenizer
        .encode(req.entity.as_str(), false)
        .map_err(|e| Status::internal(format!("tokenize error: {e}")))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();

    if token_ids.is_empty() {
        return Ok(DescribeResponse {
            entity: req.entity.clone(),
            model: model.config.model.clone(),
            edges: vec![],
            latency_ms: 0.0,
        });
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

    let patched = model.patched.blocking_read();
    let all_layers = patched.loaded_layers();
    let limit = limits::proto_count(
        "limit",
        req.limit,
        crate::routes::describe::DEFAULT_DESCRIBE_LIMIT,
        limits::MAX_RESULT_ROWS,
    )
    .map_err(Status::invalid_argument)?;
    let min_score = if req.min_score > 0.0 {
        req.min_score
    } else {
        5.0
    };

    let trace = patched.walk(&query, &all_layers, limit);
    let entity_lower = req.entity.to_lowercase();

    let mut edges = Vec::new();
    for (layer, hits) in &trace.layers {
        for hit in hits {
            if hit.gate_score < min_score {
                continue;
            }
            let tok = hit.meta.top_token.trim();
            if tok.is_empty() || tok.len() < 2 || tok.to_lowercase() == entity_lower {
                continue;
            }

            let (relation, source) = model
                .probe_labels
                .get(&(*layer, hit.feature))
                .map(|r| (r.clone(), PROBE_RELATION_SOURCE.to_string()))
                .unwrap_or_default();

            edges.push(DescribeEdge {
                target: tok.to_string(),
                gate_score: hit.gate_score,
                layer: *layer as u32,
                relation,
                source,
                also: vec![],
                layer_min: 0,
                layer_max: 0,
                count: 0,
            });
        }
    }

    edges.sort_by(|a, b| cmp_score_desc(a.gate_score, b.gate_score));
    edges.truncate(limit);

    Ok(DescribeResponse {
        entity: req.entity.clone(),
        model: model.config.model.clone(),
        edges,
        latency_ms: start.elapsed().as_secs_f64() as f32 * 1000.0,
    })
}

fn grpc_walk(model: &crate::state::LoadedModel, req: &WalkRequest) -> Result<WalkResponse, Status> {
    let start = std::time::Instant::now();
    let top_k = limits::proto_count(
        "top",
        req.top,
        crate::routes::walk::DEFAULT_WALK_TOP,
        limits::MAX_TOP_K,
    )
    .map_err(Status::invalid_argument)?;

    let encoding = model
        .tokenizer
        .encode(req.prompt.as_str(), true)
        .map_err(|e| Status::internal(format!("tokenize error: {e}")))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();
    if token_ids.is_empty() {
        return Err(Status::invalid_argument("empty prompt"));
    }

    let last_tok = *token_ids.last().unwrap();
    let query = model
        .embeddings
        .row(last_tok as usize)
        .mapv(|v| v * model.embed_scale);

    let patched = model.patched.blocking_read();
    let all_layers = patched.loaded_layers();
    let trace = patched.walk(&query, &all_layers, top_k);

    let hits: Vec<WalkHit> = trace
        .layers
        .iter()
        .flat_map(|(layer, hits)| {
            hits.iter().map(move |hit| {
                let relation = model
                    .probe_labels
                    .get(&(*layer, hit.feature))
                    .cloned()
                    .unwrap_or_default();
                proto::WalkHit {
                    layer: *layer as u32,
                    feature: hit.feature as u32,
                    gate_score: hit.gate_score,
                    target: hit.meta.top_token.trim().to_string(),
                    relation,
                }
            })
        })
        .collect();

    Ok(WalkResponse {
        prompt: req.prompt.clone(),
        hits,
        latency_ms: start.elapsed().as_secs_f64() as f32 * 1000.0,
    })
}

fn grpc_select(
    model: &crate::state::LoadedModel,
    req: &SelectRequest,
) -> Result<SelectResponse, Status> {
    let start = std::time::Instant::now();
    let patched = model.patched.blocking_read();
    let all_layers = patched.loaded_layers();
    let limit = limits::proto_count(
        "limit",
        req.limit,
        crate::routes::select::DEFAULT_SELECT_LIMIT,
        limits::MAX_RESULT_ROWS,
    )
    .map_err(Status::invalid_argument)?;

    let scan_layers: Vec<usize> = if req.layer > 0 {
        vec![req.layer as usize]
    } else {
        all_layers
    };

    let mut edges = Vec::new();
    for &layer in &scan_layers {
        if let Some(metas) = patched.down_meta_at(layer) {
            for (feat_idx, meta_opt) in metas.iter().enumerate() {
                if let Some(meta) = meta_opt {
                    if !req.entity.is_empty()
                        && !meta
                            .top_token
                            .to_lowercase()
                            .contains(&req.entity.to_lowercase())
                    {
                        continue;
                    }
                    if req.min_confidence > 0.0 && meta.c_score < req.min_confidence {
                        continue;
                    }
                    let relation = model
                        .probe_labels
                        .get(&(layer, feat_idx))
                        .cloned()
                        .unwrap_or_default();
                    if !req.relation.is_empty()
                        && !relation
                            .to_lowercase()
                            .contains(&req.relation.to_lowercase())
                    {
                        continue;
                    }
                    edges.push(SelectEdge {
                        layer: layer as u32,
                        feature: feat_idx as u32,
                        target: meta.top_token.trim().to_string(),
                        c_score: meta.c_score,
                        relation,
                    });
                }
            }
        }
    }

    edges.sort_by(|a, b| cmp_score_desc(a.c_score, b.c_score));
    let total = edges.len() as u32;
    edges.truncate(limit);

    Ok(SelectResponse {
        edges,
        total,
        latency_ms: start.elapsed().as_secs_f64() as f32 * 1000.0,
    })
}

fn grpc_infer(
    model: &crate::state::LoadedModel,
    req: &InferRequest,
) -> Result<InferResponse, Status> {
    let weights_guard = model.get_or_load_weights().map_err(Status::unavailable)?;
    let weights: &larql_inference::ModelWeights = &weights_guard;

    let encoding = model
        .tokenizer
        .encode(req.prompt.as_str(), true)
        .map_err(|e| Status::internal(format!("tokenize error: {e}")))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();
    if token_ids.is_empty() {
        return Err(Status::invalid_argument("empty prompt"));
    }

    let top_k = limits::proto_count(
        "top",
        req.top,
        crate::routes::infer::DEFAULT_INFER_TOP,
        limits::MAX_RESULT_ROWS,
    )
    .map_err(Status::invalid_argument)?;
    let start = std::time::Instant::now();
    let mode = if req.mode.is_empty() {
        INFER_MODE_WALK
    } else {
        &req.mode
    };

    let to_preds = |preds: &[(String, f64)]| -> Vec<Prediction> {
        preds
            .iter()
            .map(|(t, p)| Prediction {
                token: t.clone(),
                probability: *p,
            })
            .collect()
    };

    match mode {
        INFER_MODE_COMPARE => {
            let patched = model.patched.blocking_read();
            let walk_pred = larql_inference::infer_patched(
                weights,
                &model.tokenizer,
                &*patched,
                Some(&patched.knn_store),
                &token_ids,
                top_k,
                &larql_inference::KnnRouteMode::from_env(),
            );
            let walk_ms = walk_pred.walk_ms as f32;

            let ds = std::time::Instant::now();
            let dense_pred = larql_inference::predict(weights, &model.tokenizer, &token_ids, top_k);
            let dense_ms = ds.elapsed().as_secs_f64() as f32 * 1000.0;

            Ok(InferResponse {
                prompt: req.prompt.clone(),
                predictions: vec![],
                mode: INFER_MODE_COMPARE.into(),
                walk_predictions: to_preds(&walk_pred.predictions),
                dense_predictions: to_preds(&dense_pred.predictions),
                walk_ms,
                dense_ms,
                latency_ms: start.elapsed().as_secs_f64() as f32 * 1000.0,
            })
        }
        INFER_MODE_DENSE => {
            let pred = larql_inference::predict(weights, &model.tokenizer, &token_ids, top_k);
            Ok(InferResponse {
                prompt: req.prompt.clone(),
                predictions: to_preds(&pred.predictions),
                mode: INFER_MODE_DENSE.into(),
                walk_predictions: vec![],
                dense_predictions: vec![],
                walk_ms: 0.0,
                dense_ms: 0.0,
                latency_ms: start.elapsed().as_secs_f64() as f32 * 1000.0,
            })
        }
        _ => {
            let patched = model.patched.blocking_read();
            let pred = larql_inference::infer_patched(
                weights,
                &model.tokenizer,
                &*patched,
                Some(&patched.knn_store),
                &token_ids,
                top_k,
                &larql_inference::KnnRouteMode::from_env(),
            );
            Ok(InferResponse {
                prompt: req.prompt.clone(),
                predictions: to_preds(&pred.predictions),
                mode: INFER_MODE_WALK.into(),
                walk_predictions: vec![],
                dense_predictions: vec![],
                walk_ms: 0.0,
                dense_ms: 0.0,
                latency_ms: start.elapsed().as_secs_f64() as f32 * 1000.0,
            })
        }
    }
}

fn grpc_relations(model: &crate::state::LoadedModel) -> Result<RelationsResponse, Status> {
    let start = std::time::Instant::now();
    let patched = model.patched.blocking_read();
    let all_layers = patched.loaded_layers();

    let mut counts: std::collections::HashMap<String, (usize, String)> =
        std::collections::HashMap::new();
    for &layer in &all_layers {
        if let Some(metas) = patched.down_meta_at(layer) {
            for meta in metas.iter().flatten() {
                let tok = meta.top_token.trim();
                if tok.len() >= 2 && meta.c_score >= 0.2 {
                    let example = meta
                        .top_k
                        .first()
                        .map(|t| t.token.trim().to_string())
                        .unwrap_or_default();
                    let entry = counts.entry(tok.to_string()).or_insert((0, example));
                    entry.0 += 1;
                }
            }
        }
    }

    let mut relations: Vec<RelationInfo> = counts
        .into_iter()
        .map(|(name, (count, example))| RelationInfo {
            name,
            count: count as u32,
            example,
        })
        .collect();
    relations.sort_by_key(|r| std::cmp::Reverse(r.count));
    let total = relations.len() as u32;
    relations.truncate(50);

    Ok(RelationsResponse {
        relations,
        total,
        latency_ms: start.elapsed().as_secs_f64() as f32 * 1000.0,
    })
}

#[cfg(test)]
mod tests;
