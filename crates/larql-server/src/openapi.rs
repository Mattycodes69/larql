//! OpenAPI / Swagger UI aggregation.
//!
//! Spec JSON is served at `/v1/openapi.json` and the browse-friendly
//! Swagger UI at `/swagger-ui`. Both can be disabled with `--no-docs`.
//!
//! Handlers are annotated in place with `#[utoipa::path]`. This module
//! owns:
//! - `ApiDoc` — the aggregator `#[derive(OpenApi)]` struct.
//! - `schemas` — synthetic response structs for handlers that return
//!   `Json<serde_json::Value>` (most of the browse/inference surface).
//! - `params` — shared request parameters (e.g. `model_id`).
//! - `swagger_router()` — helper that returns a ready-to-merge router
//!   hosting both the UI and the spec JSON.

use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

use crate::error::ErrorBody;

pub mod params {
    use utoipa::IntoParams;

    /// Path parameter selecting which vindex to target in multi-model mode.
    #[derive(IntoParams)]
    #[into_params(parameter_in = Path)]
    #[allow(dead_code)]
    pub struct ModelIdParam {
        /// The id of a loaded vindex, e.g. `gemma-3-1b-it`.
        pub model_id: String,
    }
}

pub mod schemas;

#[derive(OpenApi)]
#[openapi(
    info(
        title = "larql-server",
        version = env!("CARGO_PKG_VERSION"),
        description = "HTTP API for vindex knowledge queries, inference, and remote MoE expert shards.",
    ),
    tags(
        (name = "browse",    description = "Knowledge graph browse (no weights required)"),
        (name = "inference", description = "Forward passes, explain, insert, warmup"),
        (name = "openai",    description = "OpenAI-compatible endpoints"),
        (name = "expert",    description = "Remote MoE shard endpoints (binary wire)"),
        (name = "patches",   description = "Runtime patch overlay"),
        (name = "sessions",  description = "Session observability and eviction"),
        (name = "admin",     description = "Health, models, embed, tokens, WebSocket"),
    ),
    paths(
        // browse
        crate::routes::describe::handle_describe,
        crate::routes::walk::handle_walk,
        crate::routes::relations::handle_relations,
        crate::routes::stats::handle_stats,
        crate::routes::topology::handle_topology,
        crate::routes::models::handle_models,
        // inference
        crate::routes::select::handle_select,
        crate::routes::infer::handle_infer,
        crate::routes::explain::handle_explain,
        crate::routes::insert::handle_insert,
        crate::routes::warmup::handle_warmup,
        // sessions
        crate::routes::sessions::handle_list_sessions,
        crate::routes::sessions::handle_get_session,
        crate::routes::sessions::handle_delete_session,
        // patches
        crate::routes::patches::handle_apply_patch,
        crate::routes::patches::handle_list_patches,
        crate::routes::patches::handle_remove_patch,
        // admin
        crate::routes::health::handle_health,
        crate::routes::capabilities::handle_capabilities,
        crate::routes::plan::handle_plan,
        crate::routes::runtime::handle_runtime,
        crate::routes::runtime_lifecycle::handle_load_model,
        crate::routes::runtime_lifecycle::handle_unload_model,
        crate::routes::embed::handle_embed,
        crate::routes::embed::handle_embed_single,
        crate::routes::embed::handle_logits,
        crate::routes::embed::handle_token_encode,
        crate::routes::embed::handle_token_decode,
        crate::routes::stream::handle_stream,
        // openai
        crate::routes::openai::embeddings::handle_embeddings,
        crate::routes::openai::completions::handle_completions,
        crate::routes::openai::chat::handle_chat_completions,
        crate::routes::openai::responses::handler::handle_responses,
        crate::routes::openai::responses::retrieve::handle_get_response,
        crate::routes::openai::responses::retrieve::handle_delete_response,
        crate::routes::models::handle_model_retrieve,
        // expert
        crate::routes::walk_ffn::handle_walk_ffn,
        crate::routes::walk_ffn::handle_walk_ffn_q8k,
        crate::routes::expert::single::handle_expert,
        crate::routes::expert::batch_legacy::handle_expert_batch,
        crate::routes::expert::layer_batch::handle_experts_layer_batch,
        crate::routes::expert::layer_batch::handle_experts_layer_batch_f16,
        crate::routes::expert::multi_layer_batch::handle_experts_multi_layer_batch,
        crate::routes::expert::multi_layer_batch::handle_experts_multi_layer_batch_q8k,
        // multi-model variants — same handlers with a `{model_id}` path prefix
        crate::routes::describe::handle_describe_multi,
        crate::routes::walk::handle_walk_multi,
        crate::routes::relations::handle_relations_multi,
        crate::routes::stats::handle_stats_multi,
        crate::routes::select::handle_select_multi,
        crate::routes::infer::handle_infer_multi,
        crate::routes::explain::handle_explain_multi,
        crate::routes::insert::handle_insert_multi,
        crate::routes::patches::handle_apply_patch_multi,
        crate::routes::patches::handle_list_patches_multi,
        crate::routes::patches::handle_remove_patch_multi,
        crate::routes::embed::handle_embed_multi,
        crate::routes::embed::handle_embed_single_multi,
        crate::routes::embed::handle_logits_multi,
        crate::routes::embed::handle_token_encode_multi,
        crate::routes::embed::handle_token_decode_multi,
    ),
    components(schemas(
        ErrorBody,
        crate::routes::openai::error::OpenAIErrorBody,
        crate::routes::openai::error::OpenAIErrorPayload,
        // browse
        schemas::DescribeEdge,
        schemas::DescribeResponse,
        schemas::WalkHit,
        schemas::WalkResponse,
        schemas::RelationEntry,
        schemas::RelationsResponse,
        schemas::LayerBands,
        schemas::LoadedCapabilities,
        schemas::CapabilitiesResponse,
        crate::routes::plan::PlanRequest,
        schemas::StatsResponse,
        schemas::ModelEntry,
        schemas::ModelsListResponse,
        crate::routes::topology::TopologyResponse,
        // inference
        crate::routes::select::SelectRequest,
        schemas::SelectRow,
        schemas::SelectResponse,
        crate::routes::infer::InferRequest,
        schemas::Prediction,
        schemas::InferResponse,
        crate::routes::explain::ExplainRequest,
        schemas::ExplainLayerEntry,
        schemas::ExplainResponse,
        crate::routes::insert::InsertRequest,
        schemas::InsertResponse,
        crate::routes::warmup::WarmupRequest,
        crate::routes::warmup::WarmupResponse,
        // patches
        schemas::ApplyPatchBody,
        schemas::ApplyPatchResponse,
        schemas::PatchEntry,
        schemas::ListPatchesResponse,
        schemas::RemovePatchResponse,
        // sessions
        schemas::SessionPatches,
        schemas::SessionContinuation,
        schemas::SessionResponse,
        schemas::SessionListResponse,
        schemas::SessionDeletedResponse,
        // admin
        schemas::HealthResponse,
        schemas::RuntimeModel,
        schemas::RuntimeBackend,
        schemas::RuntimeMemory,
        schemas::RuntimePerformance,
        schemas::RuntimeGeneration,
        schemas::RuntimeResponse,
        crate::routes::runtime_lifecycle::LoadModelRequest,
        schemas::TokenEncodeResponse,
        schemas::TokenDecodeResponse,
        schemas::EmbedSingleJsonResponse,
        crate::routes::embed::EmbedRequest,
        crate::routes::embed::EmbedResponse,
        crate::routes::embed::LogitsRequest,
        crate::routes::embed::LogitsResponse,
        crate::routes::embed::TokenProb,
        // openai
        schemas::OpenAiEmbeddingsRequest,
        schemas::OpenAiEmbeddingObject,
        schemas::OpenAiEmbeddingsResponse,
        schemas::OpenAiCompletionsRequest,
        schemas::OpenAiCompletionsResponse,
        schemas::OpenAiChatRequest,
        schemas::OpenAiChatResponse,
        schemas::OpenAiResponsesRequest,
        schemas::OpenAiResponsesResponse,
        // expert
        crate::routes::expert::SingleExpertRequest,
        crate::routes::expert::SingleExpertResponse,
        crate::routes::expert::BatchExpertItem,
        crate::routes::expert::BatchExpertRequest,
        crate::routes::expert::BatchExpertResult,
        crate::routes::expert::BatchExpertResponse,
    )),
)]
pub struct ApiDoc;

/// Build a router hosting Swagger UI at `/swagger-ui` and the spec at
/// `/v1/openapi.json`. Merge into the main app router.
pub fn swagger_router() -> axum::Router {
    SwaggerUi::new("/swagger-ui")
        .url("/v1/openapi.json", ApiDoc::openapi())
        .into()
}
