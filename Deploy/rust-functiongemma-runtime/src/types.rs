use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Semaphore;

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: Option<String>,
    pub messages: Vec<Message>,
    pub tools: Option<Value>,
    pub tool_choice: Option<Value>,
    pub temperature: Option<f64>,
    pub max_tokens: Option<u32>,
    // Additional OpenAI-compatible parameters
    pub top_p: Option<f64>,
    pub top_k: Option<u32>,
    pub frequency_penalty: Option<f64>,
    pub presence_penalty: Option<f64>,
    pub stop: Option<Value>, // Can be string or array of strings
    pub seed: Option<i64>,
    pub user: Option<String>,
    pub n: Option<u32>, // Number of completions (only 1 supported currently)
    #[serde(default)]
    pub stream: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Message {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Value>,
}

#[derive(Debug, Serialize)]
pub(crate) struct Choice {
    pub(crate) index: u32,
    pub(crate) message: Message,
    pub(crate) finish_reason: String,
}

#[derive(Debug, Serialize)]
pub(crate) struct Usage {
    pub(crate) prompt_tokens: u32,
    pub(crate) completion_tokens: u32,
    pub(crate) total_tokens: u32,
}

#[derive(Debug, Serialize)]
pub(crate) struct ChatCompletionResponse {
    pub(crate) id: String,
    pub(crate) object: String,
    pub(crate) created: u64,
    pub(crate) model: String,
    pub(crate) choices: Vec<Choice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) usage: Option<Usage>,
}

#[derive(Debug, Serialize)]
pub(crate) struct ChatCompletionChunk {
    pub(crate) id: String,
    pub(crate) object: String, // always "chat.completion.chunk"
    pub(crate) created: u64,
    pub(crate) model: String,
    pub(crate) choices: Vec<ChunkChoice>,
}

#[derive(Debug, Serialize)]
pub(crate) struct ChunkChoice {
    pub(crate) index: u32,
    pub(crate) delta: ChunkDelta,
    pub(crate) finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub(crate) struct ChunkDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tool_calls: Option<Value>,
}

#[derive(Debug, Serialize)]
pub(crate) struct ModelInfo {
    pub(crate) id: String,
    pub(crate) object: String,
    pub(crate) owned_by: String,
    pub(crate) metadata: RouterMetadata,
}

#[derive(Debug, Serialize)]
pub(crate) struct ModelList {
    pub(crate) object: String,
    pub(crate) data: Vec<ModelInfo>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct ToolsMetadata {
    pub(crate) path: String,
    pub(crate) count: usize,
    pub(crate) loaded: bool,
}

#[derive(Debug, Serialize)]
pub(crate) struct LoraMetadata {
    pub(crate) path: String,
    pub(crate) r: usize,
}

#[derive(Debug, Serialize)]
pub(crate) struct RouterMetadata {
    pub(crate) version: String,
    pub(crate) model: String,
    pub(crate) tools: ToolsMetadata,
    pub(crate) engine: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) device: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) lora: Option<LoraMetadata>,
}

#[derive(Debug, Serialize)]
pub(crate) struct HealthResponse {
    pub(crate) status: String,
    pub(crate) metadata: RouterMetadata,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) metrics: Option<SystemMetrics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) router_stats: Option<RouterStats>,
}

#[derive(Debug, Default)]
pub(crate) struct RouterPrompt {
    pub(crate) mode: Option<String>,
    pub(crate) system_prompt: Option<String>,
    pub(crate) user_request: String,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct SystemMetrics {
    pub(crate) process_rss_mb: u64,
    pub(crate) process_vmem_mb: u64,
    pub(crate) total_memory_mb: u64,
    pub(crate) free_memory_mb: u64,
    pub(crate) cpu_usage: f32,
    pub(crate) uptime_sec: u64,
}

#[derive(Debug, Serialize)]
pub(crate) struct RouterStats {
    pub(crate) requests: u64,
    pub(crate) tokens_generated: u64,
    pub(crate) avg_tokens_per_sec: f64,
    pub(crate) avg_latency_ms: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RouterEngine {
    Heuristic,
    Model,
}

#[derive(Debug)]
pub(crate) struct InferenceResult {
    pub(crate) content: Option<String>,
    pub(crate) tool_calls: Option<Value>,
}

#[derive(Clone)]
pub(crate) struct AppState {
    pub(crate) semaphore: Arc<Semaphore>,
    pub(crate) request_timeout: Duration,
}
