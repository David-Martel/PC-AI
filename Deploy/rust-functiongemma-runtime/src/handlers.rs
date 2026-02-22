use crate::config::{default_model, router_engine};
use crate::error::ErrorResponse;
use crate::metrics::{router_metadata, router_stats, system_metrics};
use crate::routing::{heuristic_route, last_user_message, parse_router_prompt};
use crate::types::{
    AppState, ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, Choice,
    ChunkChoice, ChunkDelta, HealthResponse, Message, ModelInfo, ModelList, RouterEngine, Usage,
};
use axum::{
    extract::State,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

/// Maximum request body size (1 MB). Prevents DoS via oversized payloads.
pub(crate) const MAX_BODY_SIZE: usize = 1024 * 1024;

/// Hard cap on max_tokens to prevent resource exhaustion.
pub(crate) const MAX_TOKENS_CAP: u32 = 4096;

pub(crate) async fn health() -> Json<Value> {
    let metadata = router_metadata();
    let metrics = system_metrics();
    let stats = router_stats();
    Json(json!(HealthResponse {
        status: "ok".to_string(),
        metadata,
        metrics,
        router_stats: stats
    }))
}

pub(crate) async fn list_models() -> Json<ModelList> {
    let model = default_model();
    let metadata = router_metadata();
    Json(ModelList {
        object: "list".to_string(),
        data: vec![ModelInfo {
            id: model,
            object: "model".to_string(),
            owned_by: "pcai".to_string(),
            metadata,
        }],
    })
}

pub(crate) fn estimate_tokens(text: &str) -> u32 {
    // Simple estimation: ~4 chars per token for English
    (text.len() as u32 / 4).max(1)
}

pub(crate) fn validate_request(req: &ChatCompletionRequest) -> Result<(), ErrorResponse> {
    if req.messages.is_empty() {
        return Err(ErrorResponse::bad_request(
            "messages array is required and must not be empty",
            Some("messages".to_string()),
        ));
    }

    // Validate max_tokens upper bound
    if let Some(mt) = req.max_tokens {
        if mt > MAX_TOKENS_CAP {
            return Err(ErrorResponse::bad_request(
                format!(
                    "max_tokens {} exceeds maximum allowed value of {}",
                    mt, MAX_TOKENS_CAP
                ),
                Some("max_tokens".to_string()),
            ));
        }
    }

    // Validate message roles
    let valid_roles = ["user", "assistant", "system", "tool", "developer"];
    for (i, msg) in req.messages.iter().enumerate() {
        if !valid_roles.contains(&msg.role.as_str()) {
            return Err(ErrorResponse::bad_request(
                format!("Invalid role '{}' at messages[{}]. Must be one of: user, assistant, system, tool, developer", msg.role, i),
                Some(format!("messages[{}].role", i)),
            ));
        }
    }

    // Validate tool_choice if present
    if let Some(tc) = &req.tool_choice {
        match tc {
            Value::String(s) if s != "none" && s != "auto" && s != "required" => {
                return Err(ErrorResponse::bad_request(
                    format!("Invalid tool_choice value '{}'. Must be 'none', 'auto', 'required', or an object", s),
                    Some("tool_choice".to_string()),
                ));
            }
            Value::Object(map) => {
                if !map.contains_key("function") && !map.contains_key("type") {
                    return Err(ErrorResponse::bad_request(
                        "tool_choice object must have 'function' or 'type' field",
                        Some("tool_choice".to_string()),
                    ));
                }
            }
            _ => {}
        }
    }

    Ok(())
}

pub(crate) async fn chat(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    // Acquire semaphore permit with timeout to bound concurrency
    let _permit = match tokio::time::timeout(
        state.request_timeout,
        state.semaphore.clone().acquire_owned(),
    )
    .await
    {
        Ok(Ok(permit)) => permit,
        Ok(Err(_)) => {
            return ErrorResponse::service_unavailable("Server shutting down").into_response()
        }
        Err(_) => {
            return ErrorResponse::too_many_requests("Server busy, try again later").into_response()
        }
    };

    // Validate request
    if let Err(e) = validate_request(&req) {
        return e.into_response();
    }

    if req.stream {
        return chat_stream(req).into_response();
    }
    chat_normal(req).into_response()
}

pub(crate) fn chat_normal(
    req: ChatCompletionRequest,
) -> Result<Json<ChatCompletionResponse>, ErrorResponse> {
    let model = req.model.clone().unwrap_or_else(default_model);
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let message_text = last_user_message(&req.messages).unwrap_or_default();
    let prompt = parse_router_prompt(message_text);

    // Estimate prompt tokens
    let prompt_tokens = req
        .messages
        .iter()
        .filter_map(|m| m.content.as_ref())
        .map(|c| estimate_tokens(c))
        .sum::<u32>();

    let result = match router_engine() {
        RouterEngine::Heuristic => heuristic_route(&req, &prompt),
        RouterEngine::Model => {
            #[cfg(feature = "model")]
            {
                match crate::inference::infer_with_model(&req) {
                    Ok(res) => res,
                    Err(err) => {
                        tracing::warn!("model inference failed, falling back to heuristic: {err}");
                        heuristic_route(&req, &prompt)
                    }
                }
            }
            #[cfg(not(feature = "model"))]
            {
                tracing::warn!("model engine requested but runtime built without model feature; using heuristic");
                heuristic_route(&req, &prompt)
            }
        }
    };

    // Determine finish_reason based on whether tool was called
    let finish_reason = if result.tool_calls.is_some() {
        "tool_calls".to_string()
    } else {
        "stop".to_string()
    };

    // Estimate completion tokens
    let completion_tokens = result
        .content
        .as_ref()
        .map(|c| estimate_tokens(c))
        .unwrap_or(0)
        + result
            .tool_calls
            .as_ref()
            .map(|tc| estimate_tokens(&tc.to_string()))
            .unwrap_or(0);

    let message = Message {
        role: "assistant".to_string(),
        content: result.content,
        tool_calls: result.tool_calls,
    };

    Ok(Json(ChatCompletionResponse {
        id: format!("pcai-router-{}", created),
        object: "chat.completion".to_string(),
        created,
        model,
        choices: vec![Choice {
            index: 0,
            message,
            finish_reason,
        }],
        usage: Some(Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        }),
    }))
}

pub(crate) fn chat_stream(
    req: ChatCompletionRequest,
) -> axum::response::sse::Sse<
    impl futures::Stream<Item = Result<axum::response::sse::Event, std::convert::Infallible>>,
> {
    use axum::response::sse;

    let model = req.model.clone().unwrap_or_else(default_model);
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let id = format!("pcai-router-{}", created);
    let message_text = last_user_message(&req.messages).unwrap_or_default();
    let prompt = parse_router_prompt(message_text);

    let result = match router_engine() {
        RouterEngine::Heuristic => heuristic_route(&req, &prompt),
        RouterEngine::Model => {
            #[cfg(feature = "model")]
            {
                match crate::inference::infer_with_model(&req) {
                    Ok(res) => res,
                    Err(err) => {
                        tracing::warn!("model inference failed in stream, falling back: {err}");
                        heuristic_route(&req, &prompt)
                    }
                }
            }
            #[cfg(not(feature = "model"))]
            {
                heuristic_route(&req, &prompt)
            }
        }
    };

    let mut events: Vec<Result<sse::Event, std::convert::Infallible>> = Vec::new();

    // First chunk: role delta only
    let role_chunk = ChatCompletionChunk {
        id: id.clone(),
        object: "chat.completion.chunk".to_string(),
        created,
        model: model.clone(),
        choices: vec![ChunkChoice {
            index: 0,
            delta: ChunkDelta {
                role: Some("assistant".to_string()),
                content: None,
                tool_calls: None,
            },
            finish_reason: None,
        }],
    };
    events.push(Ok(
        sse::Event::default().data(serde_json::to_string(&role_chunk).unwrap_or_default())
    ));

    // Content / tool_calls chunk
    let finish_reason = if result.tool_calls.is_some() {
        "tool_calls"
    } else {
        "stop"
    };
    let content_chunk = ChatCompletionChunk {
        id: id.clone(),
        object: "chat.completion.chunk".to_string(),
        created,
        model: model.clone(),
        choices: vec![ChunkChoice {
            index: 0,
            delta: ChunkDelta {
                role: None,
                content: result.content,
                tool_calls: result.tool_calls,
            },
            finish_reason: None,
        }],
    };
    events.push(Ok(
        sse::Event::default().data(serde_json::to_string(&content_chunk).unwrap_or_default())
    ));

    // Final chunk with finish_reason and empty delta
    let done_chunk = ChatCompletionChunk {
        id,
        object: "chat.completion.chunk".to_string(),
        created,
        model,
        choices: vec![ChunkChoice {
            index: 0,
            delta: ChunkDelta {
                role: None,
                content: None,
                tool_calls: None,
            },
            finish_reason: Some(finish_reason.to_string()),
        }],
    };
    events.push(Ok(
        sse::Event::default().data(serde_json::to_string(&done_chunk).unwrap_or_default())
    ));

    // [DONE] sentinel
    events.push(Ok(sse::Event::default().data("[DONE]")));

    sse::Sse::new(futures::stream::iter(events))
}
