use crate::types::{ChatCompletionRequest, InferenceResult, RouterPrompt};
use serde_json::{json, Value};

pub(crate) fn last_user_message(messages: &[crate::types::Message]) -> Option<&str> {
    messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .and_then(|m| m.content.as_deref())
}

pub(crate) fn parse_router_prompt(text: &str) -> RouterPrompt {
    let mut prompt = RouterPrompt::default();
    if let Some(idx) = text.find("[MODE]") {
        let rest = &text[idx + 6..];
        if let Some(end) = rest.find('\n') {
            prompt.mode = Some(rest[..end].trim().to_string());
        }
    }
    if let Some(idx) = text.find("[SYSTEM_PROMPT]") {
        let rest = &text[idx + 15..];
        if let Some(end) = rest.find("[USER_REQUEST]") {
            prompt.system_prompt = Some(rest[..end].trim().to_string());
        }
    }
    if let Some(idx) = text.find("[USER_REQUEST]") {
        prompt.user_request = text[idx + 14..].trim().to_string();
    } else {
        prompt.user_request = text.trim().to_string();
    }
    prompt
}

pub(crate) fn extract_args_from_text(text: &str) -> Option<Value> {
    if let Some(idx) = text.find("Arguments:") {
        let tail = text[idx + "Arguments:".len()..].trim();
        if let Ok(val) = serde_json::from_str::<Value>(tail) {
            return Some(val);
        }
    }
    let start = text.find('{')?;
    let end = text.rfind('}')?;
    if end > start {
        serde_json::from_str::<Value>(&text[start..=end]).ok()
    } else {
        None
    }
}

pub(crate) fn select_tool_by_name(text: &str, tools: &Value) -> Option<String> {
    let tools_arr = tools.as_array()?;
    let text_lower = text.to_lowercase();
    for tool in tools_arr {
        let name = tool
            .get("function")
            .and_then(|f| f.get("name"))
            .and_then(|n| n.as_str())?;
        if text_lower.contains(&name.to_lowercase()) {
            return Some(name.to_string());
        }
    }
    None
}

pub(crate) fn resolve_tool_choice(tool_choice: &Value) -> Option<String> {
    match tool_choice {
        Value::String(s) => {
            if s == "none" || s == "auto" {
                None
            } else {
                Some(s.to_string())
            }
        }
        Value::Object(map) => map
            .get("function")
            .and_then(|f| f.get("name"))
            .and_then(|n| n.as_str())
            .map(|s| s.to_string()),
        _ => None,
    }
}

pub(crate) fn tool_choice_requires_tool(tool_choice: &Value) -> bool {
    match tool_choice {
        Value::String(s) => s.eq_ignore_ascii_case("required"),
        Value::Object(map) => {
            let is_function = map
                .get("type")
                .and_then(|v| v.as_str())
                .map(|t| t.eq_ignore_ascii_case("function"))
                .unwrap_or(false);
            is_function && map.get("function").is_none()
        }
        _ => false,
    }
}

pub(crate) fn first_tool_name(tools: &Value) -> Option<String> {
    tools.as_array().and_then(|arr| {
        arr.iter()
            .filter_map(|tool| {
                tool.get("function")
                    .and_then(|f| f.get("name"))
                    .and_then(|n| n.as_str())
                    .map(|s| s.to_string())
            })
            .next()
    })
}

pub(crate) fn build_tool_calls(tool_name: &str, args: Value, call_id: &str) -> Value {
    json!([
        {
            "id": call_id,
            "type": "function",
            "function": {
                "name": tool_name,
                "arguments": args
            }
        }
    ])
}

pub(crate) fn heuristic_route(req: &ChatCompletionRequest, prompt: &RouterPrompt) -> InferenceResult {
    let mut tool_name: Option<String> = None;
    let mut tool_args: Value = json!({});
    let requires_tool = req.tool_choice.as_ref().map(tool_choice_requires_tool).unwrap_or(false);

    if let Some(choice) = req.tool_choice.as_ref().and_then(resolve_tool_choice) {
        tool_name = Some(choice);
        tool_args = extract_args_from_text(&prompt.user_request).unwrap_or_else(|| json!({}));
    } else if let Some(tools) = req.tools.as_ref() {
        tool_name = select_tool_by_name(&prompt.user_request, tools);
        if tool_name.is_none() && requires_tool {
            tool_name = first_tool_name(tools);
        }
        tool_args = extract_args_from_text(&prompt.user_request).unwrap_or_else(|| json!({}));
    }

    if let Some(name) = tool_name {
        let calls = build_tool_calls(&name, tool_args, "call_heuristic");
        InferenceResult {
            content: None,
            tool_calls: Some(calls),
        }
    } else {
        InferenceResult {
            content: Some("NO_TOOL".to_string()),
            tool_calls: None,
        }
    }
}
