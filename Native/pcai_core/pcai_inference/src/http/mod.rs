//! HTTP server with OpenAI-compatible API

use axum::{
    extract::State,
    http::StatusCode,
    response::{sse::Event, IntoResponse, Response, Sse},
    routing::{get, post},
    Json, Router,
};
use futures::stream;
use serde::{Deserialize, Serialize};
use std::{convert::Infallible, fs, path::PathBuf, sync::Arc, sync::OnceLock};
use tokio::sync::mpsc;
use tokio::sync::RwLock;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use uuid::Uuid;

use crate::{
    backends::{FinishReason, GenerateRequest, InferenceBackend},
    config::ServerConfig,
    Error, Result,
};

#[derive(Debug, Deserialize, Default)]
struct LlmConfigFile {
    #[serde(default)]
    router: RouterConfigFile,
}

#[derive(Debug, Deserialize, Default)]
#[allow(dead_code)]
struct RouterConfigFile {
    enabled: Option<bool>,
    provider: Option<String>,
    #[serde(rename = "baseUrl")]
    base_url: Option<String>,
    model: Option<String>,
    #[serde(rename = "toolsPath")]
    tools_path: Option<String>,
    strict: Option<bool>,
    force: Option<bool>,
    disable: Option<bool>,
    #[serde(rename = "defaultTemperature")]
    default_temperature: Option<f64>,
}

#[derive(Debug, Clone)]
struct RouterSettings {
    base_url: String,
    model: String,
    tools_path: String,
    strict: bool,
    force: bool,
    disable: bool,
    default_temperature: f64,
}

static ROUTER_SETTINGS: OnceLock<RouterSettings> = OnceLock::new();

fn config_path() -> PathBuf {
    PathBuf::from("Config/llm-config.json")
}

fn load_router_settings() -> RouterSettings {
    let raw = fs::read_to_string(config_path()).ok();
    let config = raw
        .and_then(|text| serde_json::from_str::<LlmConfigFile>(&text).ok())
        .unwrap_or_default();

    let router = config.router;
    let enabled = router.enabled.unwrap_or(true);
    let base_url = router
        .base_url
        .filter(|v| !v.trim().is_empty())
        .or_else(|| std::env::var("PCAI_ROUTER_BASE_URL").ok())
        .unwrap_or_default();
    let model = router
        .model
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| "functiongemma-270m-it".to_string());
    let tools_path = router
        .tools_path
        .filter(|v| !v.trim().is_empty())
        .or_else(|| std::env::var("PCAI_TOOLS_PATH").ok())
        .unwrap_or_default();
    let disable = router.disable.unwrap_or(!enabled) || base_url.trim().is_empty();

    RouterSettings {
        base_url,
        model,
        tools_path,
        strict: router.strict.unwrap_or(false),
        force: router.force.unwrap_or(false),
        disable,
        default_temperature: router.default_temperature.unwrap_or(0.2),
    }
}

fn router_settings() -> &'static RouterSettings {
    ROUTER_SETTINGS.get_or_init(load_router_settings)
}

/// Shared application state
pub struct AppState {
    backend: Arc<RwLock<Box<dyn InferenceBackend>>>,
}

/// Run the HTTP server
pub async fn run_server(config: ServerConfig, backend: Box<dyn InferenceBackend>) -> Result<()> {
    tracing::info!("Starting HTTP server on {}:{}", config.host, config.port);

    let state = AppState {
        backend: Arc::new(RwLock::new(backend)),
    };

    let mut app = Router::new()
        .route("/health", get(health_check))
        .route("/v1/models", get(list_models))
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(Arc::new(state))
        .layer(TraceLayer::new_for_http());

    if config.cors {
        app = app.layer(CorsLayer::permissive());
    }

    let addr = format!("{}:{}", config.host, config.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;

    tracing::info!("Server listening on {}", addr);

    axum::serve(listener, app).await.map_err(|e| Error::Other(e.into()))?;

    Ok(())
}

/// Health check endpoint
async fn health_check(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let backend = state.backend.read().await;
    let status = if backend.is_loaded() { "ready" } else { "not_ready" };

    Json(serde_json::json!({
        "status": status,
        "backend": backend.backend_name(),
    }))
}

/// OpenAI-compatible models list endpoint
async fn list_models(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let backend = state.backend.read().await;
    let model_id = if backend.is_loaded() {
        backend.backend_name()
    } else {
        "pcai-inference"
    };

    Json(serde_json::json!({
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "owned_by": "pcai"
            }
        ]
    }))
}

/// OpenAI-compatible completions endpoint
async fn completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CompletionRequest>,
) -> std::result::Result<Response, AppError> {
    let prompt_tokens = estimate_tokens(&req.prompt);
    let stop = req.stop.clone();
    let generate_req = GenerateRequest {
        prompt: req.prompt,
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
        stop: stop.clone().unwrap_or_default(),
    };

    if req.stream.unwrap_or(false) {
        let created = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock is before UNIX_EPOCH")
            .as_secs();
        let id = format!("cmpl-{}", Uuid::new_v4());
        let model = req.model.unwrap_or_else(|| "pcai-inference".to_string());

        let sse = stream_completions(state.backend.clone(), generate_req, stop, id, model, created).await?;
        return Ok(sse.into_response());
    }

    let backend = state.backend.read().await;

    if !backend.is_loaded() {
        return Err(AppError::ModelNotLoaded);
    }

    let response = backend.generate(generate_req).await?;

    let (text, finish_reason) = apply_stop_sequences(&response.text, &stop, response.finish_reason);

    let completion_response = CompletionResponse {
        id: format!("cmpl-{}", Uuid::new_v4()),
        object: "text_completion".to_string(),
        created: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock is before UNIX_EPOCH")
            .as_secs(),
        model: req.model.unwrap_or_else(|| "pcai-inference".to_string()),
        choices: vec![Choice {
            text,
            index: 0,
            finish_reason: Some(finish_reason_to_string(finish_reason)),
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens: response.tokens_generated,
            total_tokens: prompt_tokens + response.tokens_generated,
        },
    };

    Ok(Json(completion_response).into_response())
}

/// Chat completions endpoint
async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> std::result::Result<Response, AppError> {
    if should_route_to_functiongemma(&req) {
        match call_functiongemma(&req).await {
            Ok(router_response) => {
                if req.stream.unwrap_or(false) {
                    let sse = stream_functiongemma_response(router_response)?;
                    return Ok(sse.into_response());
                }
                return Ok(Json(router_response).into_response());
            }
            Err(err) => {
                if router_settings().strict {
                    return Err(err);
                }
                tracing::warn!("Router failed, falling back to local inference: {}", err.0);
            }
        }
    }

    let prompt = build_chat_prompt(&req.messages)?;
    let prompt_tokens = estimate_tokens(&prompt);

    let stop = req.stop.clone();
    let generate_req = GenerateRequest {
        prompt,
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
        stop: stop.clone().unwrap_or_default(),
    };

    if req.stream.unwrap_or(false) {
        let created = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock is before UNIX_EPOCH")
            .as_secs();
        let id = format!("chatcmpl-{}", Uuid::new_v4());
        let model = req.model.unwrap_or_else(|| "pcai-inference".to_string());

        let sse = stream_chat_completions(state.backend.clone(), generate_req, stop, id, model, created).await?;
        return Ok(sse.into_response());
    }

    let backend = state.backend.read().await;

    if !backend.is_loaded() {
        return Err(AppError::ModelNotLoaded);
    }

    let response = backend.generate(generate_req).await?;
    let (content, finish_reason) = apply_stop_sequences(&response.text, &stop, response.finish_reason);

    let completion_response = ChatCompletionResponse {
        id: format!("chatcmpl-{}", Uuid::new_v4()),
        object: "chat.completion".to_string(),
        created: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock is before UNIX_EPOCH")
            .as_secs(),
        model: req.model.unwrap_or_else(|| "pcai-inference".to_string()),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessageResponse {
                role: "assistant".to_string(),
                content,
            },
            finish_reason: Some(finish_reason_to_string(finish_reason)),
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens: response.tokens_generated,
            total_tokens: prompt_tokens + response.tokens_generated,
        },
    };

    Ok(Json(completion_response).into_response())
}

// Request/Response types
#[derive(Debug, Deserialize)]
struct CompletionRequest {
    #[serde(default)]
    model: Option<String>,
    prompt: String,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    stop: Option<Vec<String>>,
    #[serde(default)]
    stream: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionRequest {
    model: Option<String>,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    stop: Option<Vec<String>>,
    #[serde(default)]
    stream: Option<bool>,
    #[serde(default)]
    tools: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    tool_choice: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct CompletionResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<Choice>,
    usage: Usage,
}

#[derive(Debug, Serialize)]
struct Choice {
    text: String,
    index: usize,
    finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

#[derive(Debug, Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<ChatChoice>,
    usage: Usage,
}

#[derive(Debug, Serialize)]
struct ChatChoice {
    index: usize,
    message: ChatMessageResponse,
    finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
struct ChatMessageResponse {
    role: String,
    content: String,
}

// Error handling
#[derive(Debug)]
struct AppError(Error);

impl From<Error> for AppError {
    fn from(err: Error) -> Self {
        AppError(err)
    }
}

impl AppError {
    #[allow(non_upper_case_globals)]
    const ModelNotLoaded: Self = AppError(Error::ModelNotLoaded);
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, message) = match &self.0 {
            Error::ModelNotLoaded => (StatusCode::SERVICE_UNAVAILABLE, "Model not loaded"),
            Error::Backend(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg.as_str()),
            Error::InvalidInput(msg) => (StatusCode::BAD_REQUEST, msg.as_str()),
            _ => (StatusCode::INTERNAL_SERVER_ERROR, "Internal server error"),
        };

        let body = Json(serde_json::json!({
            "error": {
                "message": message,
                "type": "server_error",
            }
        }));

        (status, body).into_response()
    }
}

fn build_chat_prompt(messages: &[ChatMessage]) -> std::result::Result<String, AppError> {
    if messages.is_empty() {
        return Err(AppError(Error::InvalidInput("messages must not be empty".to_string())));
    }

    let mut prompt = String::new();
    for message in messages {
        let role = message.role.to_lowercase();
        let role_label = match role.as_str() {
            "system" => "System",
            "user" => "User",
            "assistant" => "Assistant",
            "tool" => "Tool",
            _ => "User",
        };

        if !prompt.is_empty() {
            prompt.push('\n');
        }

        prompt.push_str(role_label);
        prompt.push_str(": ");
        prompt.push_str(message.content.trim());
    }

    prompt.push_str("\nAssistant: ");
    Ok(prompt)
}

fn estimate_tokens(text: &str) -> usize {
    if text.is_empty() {
        return 0;
    }

    let char_count = text.chars().count();
    let char_tokens = char_count.div_ceil(4);
    let word_tokens = text.split_whitespace().count();
    char_tokens.max(word_tokens).max(1)
}

fn apply_stop_sequences(text: &str, stop: &Option<Vec<String>>, finish_reason: FinishReason) -> (String, FinishReason) {
    let Some(stops) = stop else {
        return (text.to_string(), finish_reason);
    };

    let mut earliest: Option<usize> = None;
    for stop_seq in stops {
        if stop_seq.is_empty() {
            continue;
        }
        if let Some(idx) = text.find(stop_seq) {
            earliest = Some(match earliest {
                Some(prev) => prev.min(idx),
                None => idx,
            });
        }
    }

    match earliest {
        Some(idx) => (text[..idx].to_string(), FinishReason::Stop),
        None => (text.to_string(), finish_reason),
    }
}

fn finish_reason_to_string(reason: FinishReason) -> String {
    match reason {
        FinishReason::Stop => "stop".to_string(),
        FinishReason::Length => "length".to_string(),
        FinishReason::Error => "error".to_string(),
    }
}

async fn stream_completions(
    backend: Arc<RwLock<Box<dyn InferenceBackend>>>,
    request: GenerateRequest,
    stop: Option<Vec<String>>,
    id: String,
    model: String,
    created: u64,
) -> std::result::Result<Sse<impl futures::Stream<Item = std::result::Result<Event, Infallible>>>, AppError> {
    let (tx, rx) = mpsc::unbounded_channel::<StreamItem>();
    let tx_tokens = tx.clone();

    tokio::spawn(async move {
        let backend_guard = backend.read().await;
        if !backend_guard.is_loaded() {
            let _ = tx.send(StreamItem::Error("Model not loaded".to_string()));
            let _ = tx.send(StreamItem::Done);
            return;
        }

        let mut tracker = StopTracker::new(stop.clone(), 64);
        let mut callback = |token: String| {
            for chunk in tracker.push(&token) {
                let _ = tx_tokens.send(StreamItem::Token(chunk));
            }
        };

        let result = backend_guard.generate_streaming(request, &mut callback).await;
        match result {
            Ok(response) => {
                let finish_reason = if tracker.stop_hit() {
                    FinishReason::Stop
                } else {
                    response.finish_reason
                };
                let _ = tx.send(StreamItem::Final(finish_reason));
            }
            Err(err) => {
                let _ = tx.send(StreamItem::Error(err.to_string()));
            }
        }

        let _ = tx.send(StreamItem::Done);
    });

    let id = Arc::new(id);
    let model = Arc::new(model);

    let stream = stream::unfold(rx, move |mut rx| {
        let id = Arc::clone(&id);
        let model = Arc::clone(&model);
        async move {
            match rx.recv().await {
                Some(item) => {
                    let data = match item {
                        StreamItem::Token(token) => serde_json::json!({
                            "id": id.as_str(),
                            "object": "text_completion.chunk",
                            "created": created,
                            "model": model.as_str(),
                            "choices": [
                                {
                                    "index": 0,
                                    "text": token,
                                    "finish_reason": null
                                }
                            ]
                        })
                        .to_string(),
                        StreamItem::Final(reason) => serde_json::json!({
                            "id": id.as_str(),
                            "object": "text_completion.chunk",
                            "created": created,
                            "model": model.as_str(),
                            "choices": [
                                {
                                    "index": 0,
                                    "text": "",
                                    "finish_reason": finish_reason_to_string(reason)
                                }
                            ]
                        })
                        .to_string(),
                        StreamItem::Error(msg) => serde_json::json!({
                            "error": {
                                "message": msg,
                                "type": "server_error"
                            }
                        })
                        .to_string(),
                        StreamItem::Done => "[DONE]".to_string(),
                    };
                    Some((Ok(Event::default().data(data)), rx))
                }
                None => None,
            }
        }
    });

    Ok(Sse::new(stream))
}

async fn stream_chat_completions(
    backend: Arc<RwLock<Box<dyn InferenceBackend>>>,
    request: GenerateRequest,
    stop: Option<Vec<String>>,
    id: String,
    model: String,
    created: u64,
) -> std::result::Result<Sse<impl futures::Stream<Item = std::result::Result<Event, Infallible>>>, AppError> {
    let (tx, rx) = mpsc::unbounded_channel::<StreamItem>();
    let tx_tokens = tx.clone();

    tokio::spawn(async move {
        let backend_guard = backend.read().await;
        if !backend_guard.is_loaded() {
            let _ = tx.send(StreamItem::Error("Model not loaded".to_string()));
            let _ = tx.send(StreamItem::Done);
            return;
        }

        let mut tracker = StopTracker::new(stop.clone(), 64);
        let mut callback = |token: String| {
            for chunk in tracker.push(&token) {
                let _ = tx_tokens.send(StreamItem::Token(chunk));
            }
        };

        let result = backend_guard.generate_streaming(request, &mut callback).await;
        match result {
            Ok(response) => {
                let finish_reason = if tracker.stop_hit() {
                    FinishReason::Stop
                } else {
                    response.finish_reason
                };
                let _ = tx.send(StreamItem::Final(finish_reason));
            }
            Err(err) => {
                let _ = tx.send(StreamItem::Error(err.to_string()));
            }
        }

        let _ = tx.send(StreamItem::Done);
    });

    let id = Arc::new(id);
    let model = Arc::new(model);

    let stream = stream::unfold(rx, move |mut rx| {
        let id = Arc::clone(&id);
        let model = Arc::clone(&model);
        async move {
            match rx.recv().await {
                Some(item) => {
                    let data = match item {
                        StreamItem::Token(token) => serde_json::json!({
                            "id": id.as_str(),
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model.as_str(),
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": { "content": token },
                                    "finish_reason": null
                                }
                            ]
                        })
                        .to_string(),
                        StreamItem::Final(reason) => serde_json::json!({
                            "id": id.as_str(),
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model.as_str(),
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": finish_reason_to_string(reason)
                                }
                            ]
                        })
                        .to_string(),
                        StreamItem::Error(msg) => serde_json::json!({
                            "error": {
                                "message": msg,
                                "type": "server_error"
                            }
                        })
                        .to_string(),
                        StreamItem::Done => "[DONE]".to_string(),
                    };
                    Some((Ok(Event::default().data(data)), rx))
                }
                None => None,
            }
        }
    });

    Ok(Sse::new(stream))
}

fn stream_functiongemma_response(
    response: serde_json::Value,
) -> std::result::Result<Sse<impl futures::Stream<Item = std::result::Result<Event, Infallible>>>, AppError> {
    let mut events: Vec<std::result::Result<Event, Infallible>> = Vec::new();

    if response.get("error").is_some() {
        events.push(Ok(Event::default().data(response.to_string())));
        events.push(Ok(Event::default().data("[DONE]")));
        return Ok(Sse::new(stream::iter(events)));
    }

    let id = response
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("chatcmpl-functiongemma");
    let created = response.get("created").and_then(|v| v.as_u64()).unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock is before UNIX_EPOCH")
            .as_secs()
    });
    let model = response
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("functiongemma-270m-it");

    let choice = response
        .get("choices")
        .and_then(|v| v.as_array())
        .and_then(|v| v.first())
        .ok_or_else(|| AppError(Error::Backend("Router response missing choices".to_string())))?;

    let finish_reason = choice.get("finish_reason").and_then(|v| v.as_str()).unwrap_or_else(|| {
        if choice.get("message").and_then(|m| m.get("tool_calls")).is_some() {
            "tool_calls"
        } else {
            "stop"
        }
    });

    let message = choice
        .get("message")
        .ok_or_else(|| AppError(Error::Backend("Router response missing message".to_string())))?;

    // initial role delta
    let role_payload = serde_json::json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": { "role": "assistant" },
                "finish_reason": null
            }
        ]
    });
    events.push(Ok(Event::default().data(role_payload.to_string())));

    if let Some(content) = message.get("content").and_then(|v| v.as_str()) {
        for chunk in chunk_text(content, 64) {
            let payload = serde_json::json!({
                "id": id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": { "content": chunk },
                        "finish_reason": null
                    }
                ]
            });
            events.push(Ok(Event::default().data(payload.to_string())));
        }
    }

    if let Some(tool_calls) = message.get("tool_calls").and_then(|v| v.as_array()) {
        for (index, tool_call) in tool_calls.iter().enumerate() {
            let id_value = tool_call.get("id").and_then(|v| v.as_str()).unwrap_or("call_0");
            let func = tool_call.get("function").cloned().unwrap_or_else(|| {
                serde_json::json!({
                    "name": "unknown",
                    "arguments": "{}"
                })
            });

            let payload = serde_json::json!({
                "id": id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": index,
                                    "id": id_value,
                                    "type": "function",
                                    "function": func
                                }
                            ]
                        },
                        "finish_reason": null
                    }
                ]
            });
            events.push(Ok(Event::default().data(payload.to_string())));
        }
    }

    let final_payload = serde_json::json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": finish_reason
            }
        ]
    });
    events.push(Ok(Event::default().data(final_payload.to_string())));
    events.push(Ok(Event::default().data("[DONE]")));

    Ok(Sse::new(stream::iter(events)))
}

fn should_route_to_functiongemma(req: &ChatCompletionRequest) -> bool {
    if router_settings().disable {
        return false;
    }

    if router_settings().force {
        return true;
    }

    if let Some(tools) = &req.tools {
        if !tools.is_empty() {
            return true;
        }
    }

    let mut content = String::new();
    for message in &req.messages {
        content.push_str(&message.content);
        content.push('\n');
    }

    let content = content.to_lowercase();
    let keywords = [
        "diagnose",
        "diagnostic",
        "report",
        "usb",
        "disk",
        "drive",
        "wsl",
        "docker",
        "network",
        "latency",
        "performance",
        "gpu",
        "driver",
        "event log",
        "error",
        "smart",
    ];

    keywords.iter().any(|k| content.contains(k))
}

async fn call_functiongemma(req: &ChatCompletionRequest) -> std::result::Result<serde_json::Value, AppError> {
    let settings = router_settings();
    let base_url = settings.base_url.clone();
    let model = settings.model.clone();

    let tools = match &req.tools {
        Some(tools) => Some(tools.clone()),
        None => load_default_tools(),
    };

    let payload = serde_json::json!({
        "model": model,
        "messages": req.messages.iter().map(|m| serde_json::json!({
            "role": m.role,
            "content": m.content
        })).collect::<Vec<_>>(),
        "tools": tools,
        "tool_choice": req.tool_choice.clone().unwrap_or(serde_json::Value::String("auto".to_string())),
        "temperature": req.temperature.unwrap_or(settings.default_temperature as f32),
    });

    let client = reqwest::Client::new();
    let url = format!("{}/v1/chat/completions", base_url.trim_end_matches('/'));
    let response = client
        .post(url)
        .json(&payload)
        .send()
        .await
        .map_err(|e| AppError(Error::Backend(format!("Router request failed: {}", e))))?;

    let status = response.status();
    let value = response
        .json::<serde_json::Value>()
        .await
        .map_err(|e| AppError(Error::Backend(format!("Router response invalid: {}", e))))?;

    if !status.is_success() {
        return Err(AppError(Error::Backend(format!(
            "Router returned {}: {}",
            status, value
        ))));
    }

    Ok(value)
}

fn load_default_tools() -> Option<Vec<serde_json::Value>> {
    let path = router_settings().tools_path.clone();
    let content = std::fs::read_to_string(&path).ok()?;
    let doc: serde_json::Value = serde_json::from_str(&content).ok()?;
    doc.get("tools")
        .and_then(|tools| tools.as_array())
        .map(|arr| arr.to_vec())
}

#[derive(Debug)]
struct StopTracker {
    stops: Vec<String>,
    buffer: String,
    emitted_len: usize,
    stopped: bool,
    chunk_size: usize,
}

impl StopTracker {
    fn new(stops: Option<Vec<String>>, chunk_size: usize) -> Self {
        Self {
            stops: stops.unwrap_or_default(),
            buffer: String::new(),
            emitted_len: 0,
            stopped: false,
            chunk_size: chunk_size.max(1),
        }
    }

    fn stop_hit(&self) -> bool {
        self.stopped
    }

    fn push(&mut self, token: &str) -> Vec<String> {
        if self.stopped {
            return vec![];
        }

        self.buffer.push_str(token);

        if self.stops.is_empty() {
            let new_text = &self.buffer[self.emitted_len..];
            self.emitted_len = self.buffer.len();
            return chunk_text(new_text, self.chunk_size);
        }

        let mut earliest: Option<usize> = None;
        for stop in &self.stops {
            if stop.is_empty() {
                continue;
            }
            if let Some(idx) = self.buffer.find(stop) {
                earliest = Some(match earliest {
                    Some(prev) => prev.min(idx),
                    None => idx,
                });
            }
        }

        match earliest {
            Some(idx) => {
                let allowed = &self.buffer[..idx];
                let new_text = if self.emitted_len < allowed.len() {
                    &allowed[self.emitted_len..]
                } else {
                    ""
                };
                self.emitted_len = allowed.len();
                self.stopped = true;
                chunk_text(new_text, self.chunk_size)
            }
            None => {
                let new_text = &self.buffer[self.emitted_len..];
                self.emitted_len = self.buffer.len();
                chunk_text(new_text, self.chunk_size)
            }
        }
    }
}

#[derive(Debug)]
enum StreamItem {
    Token(String),
    Final(FinishReason),
    Error(String),
    Done,
}
fn chunk_text(text: &str, max_chars: usize) -> Vec<String> {
    if text.is_empty() {
        return vec![];
    }

    let mut chunks = Vec::new();
    let mut current = String::new();

    for ch in text.chars() {
        current.push(ch);
        if current.len() >= max_chars {
            chunks.push(current);
            current = String::new();
        }
    }

    if !current.is_empty() {
        chunks.push(current);
    }

    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_chat_prompt() {
        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: "You are helpful.".to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: "Hello".to_string(),
            },
        ];

        let prompt = build_chat_prompt(&messages).expect("TODO: Verify unwrap");
        assert!(prompt.contains("System: You are helpful."));
        assert!(prompt.contains("User: Hello"));
        assert!(prompt.ends_with("Assistant: "));
    }

    #[test]
    fn test_estimate_tokens() {
        assert_eq!(estimate_tokens(""), 0);
        assert!(estimate_tokens("hello") >= 1);
        assert!(estimate_tokens("hello world") >= 2);
    }

    #[test]
    fn test_apply_stop_sequences() {
        let input = "Hello world STOP and more";
        let stops = Some(vec!["STOP".to_string()]);
        let (text, reason) = apply_stop_sequences(input, &stops, FinishReason::Length);
        assert_eq!(text.trim_end(), "Hello world");
        assert!(matches!(reason, FinishReason::Stop));
    }

    // === build_chat_prompt edge cases ===

    #[test]
    fn test_build_chat_prompt_empty_messages() {
        let messages: Vec<ChatMessage> = vec![];
        let result = build_chat_prompt(&messages);
        assert!(result.is_err());
    }

    #[test]
    fn test_build_chat_prompt_unknown_role() {
        let messages = vec![ChatMessage {
            role: "custom_role".to_string(),
            content: "Hi".to_string(),
        }];
        let prompt = build_chat_prompt(&messages).expect("TODO: Verify unwrap");
        // Unknown role should map to "User"
        assert!(prompt.contains("User: Hi"));
        assert!(prompt.ends_with("Assistant: "));
    }

    #[test]
    fn test_build_chat_prompt_tool_role() {
        let messages = vec![ChatMessage {
            role: "tool".to_string(),
            content: "result data".to_string(),
        }];
        let prompt = build_chat_prompt(&messages).expect("TODO: Verify unwrap");
        assert!(prompt.contains("Tool: result data"));
    }

    #[test]
    fn test_build_chat_prompt_single_user() {
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "What is 2+2?".to_string(),
        }];
        let prompt = build_chat_prompt(&messages).expect("TODO: Verify unwrap");
        assert_eq!(prompt, "User: What is 2+2?\nAssistant: ");
    }

    // === estimate_tokens edge cases ===

    #[test]
    fn test_estimate_tokens_long_text() {
        let short_tokens = estimate_tokens("hello");
        let long_tokens = estimate_tokens("hello world this is a much longer sentence with many words");
        assert!(long_tokens > short_tokens);
    }

    #[test]
    fn test_estimate_tokens_single_char() {
        let tokens = estimate_tokens("x");
        assert!(tokens >= 1);
    }

    // === apply_stop_sequences edge cases ===

    #[test]
    fn test_apply_stop_no_stops() {
        let (text, reason) = apply_stop_sequences("hello world", &None, FinishReason::Length);
        assert_eq!(text, "hello world");
        assert!(matches!(reason, FinishReason::Length));
    }

    #[test]
    fn test_apply_stop_empty_stop_list() {
        let stops = Some(vec![]);
        let (text, reason) = apply_stop_sequences("hello world", &stops, FinishReason::Length);
        assert_eq!(text, "hello world");
        assert!(matches!(reason, FinishReason::Length));
    }

    #[test]
    fn test_apply_stop_multiple_picks_earliest() {
        let stops = Some(vec!["LATE".to_string(), "EARLY".to_string()]);
        let input = "text EARLY then LATE end";
        let (text, reason) = apply_stop_sequences(input, &stops, FinishReason::Length);
        assert_eq!(text, "text ");
        assert!(matches!(reason, FinishReason::Stop));
    }

    #[test]
    fn test_apply_stop_at_beginning() {
        let stops = Some(vec!["STOP".to_string()]);
        let (text, reason) = apply_stop_sequences("STOP rest of text", &stops, FinishReason::Length);
        assert_eq!(text, "");
        assert!(matches!(reason, FinishReason::Stop));
    }

    // === chunk_text tests ===

    #[test]
    fn test_chunk_text_empty() {
        let chunks = chunk_text("", 10);
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_chunk_text_exact_fit() {
        let chunks = chunk_text("abcde", 5);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "abcde");
    }

    #[test]
    fn test_chunk_text_splits() {
        let chunks = chunk_text("abcdefghij", 3);
        assert_eq!(chunks.len(), 4); // "abc", "def", "ghi", "j"
        assert_eq!(chunks[0], "abc");
        assert_eq!(chunks[1], "def");
        assert_eq!(chunks[2], "ghi");
        assert_eq!(chunks[3], "j");
    }

    #[test]
    fn test_chunk_text_remainder() {
        let chunks = chunk_text("abcdefg", 4);
        assert_eq!(chunks.len(), 2); // "abcd", "efg"
        assert_eq!(chunks[0], "abcd");
        assert_eq!(chunks[1], "efg");
    }

    // === StopTracker tests ===

    #[test]
    fn test_stop_tracker_no_stops() {
        let mut tracker = StopTracker::new(None, 64);
        let result = tracker.push("hello ");
        assert_eq!(result, vec!["hello "]);
        let result2 = tracker.push("world");
        assert_eq!(result2, vec!["world"]);
        assert!(!tracker.stop_hit());
    }

    #[test]
    fn test_stop_tracker_detects_stop() {
        let mut tracker = StopTracker::new(Some(vec!["STOP".to_string()]), 64);
        let result = tracker.push("hello STOP world");
        assert_eq!(result, vec!["hello "]);
        assert!(tracker.stop_hit());
    }

    #[test]
    fn test_stop_tracker_stop_across_tokens() {
        // StopTracker emits text eagerly - when "hello ST" arrives, the tracker
        // doesn't know "STOP" is coming, so it emits "hello ST". When "OP world"
        // arrives, the buffer becomes "hello STOP world" and the stop is found at
        // index 6. Since 8 chars were already emitted (past the stop point),
        // no new text is emitted on the second push.
        let mut tracker = StopTracker::new(Some(vec!["STOP".to_string()]), 64);
        let r1 = tracker.push("hello ST");
        assert!(!tracker.stop_hit());
        assert_eq!(r1, vec!["hello ST"]);
        let r2 = tracker.push("OP world");
        assert!(tracker.stop_hit());
        // No new text emitted after stop detected
        assert!(r2.is_empty());
    }

    #[test]
    fn test_stop_tracker_already_stopped() {
        let mut tracker = StopTracker::new(Some(vec!["X".to_string()]), 64);
        let _ = tracker.push("aXb");
        assert!(tracker.stop_hit());
        let result = tracker.push("more text");
        assert!(result.is_empty());
    }
}
