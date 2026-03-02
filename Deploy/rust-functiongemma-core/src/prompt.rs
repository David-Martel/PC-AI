use serde_json::{Map, Value};

pub fn format_escape_value(val: &Value) -> String {
    match val {
        Value::String(s) => s.clone(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        _ => serde_json::to_string(val).unwrap_or_default(),
    }
}

pub fn format_function_call(name: &str, args: &Map<String, Value>) -> String {
    if args.is_empty() {
        return format!("<start_function_call>call:{}{{}}<end_function_call>", name);
    }
    let mut parts = Vec::new();
    for (key, val) in args {
        let value = format_escape_value(val);
        parts.push(format!("{key}:<escape>{value}<escape>"));
    }
    let args_text = parts.join(",");
    format!("<start_function_call>call:{}{{{}}}<end_function_call>", name, args_text)
}

pub fn parse_escape_args(args_text: &str) -> Value {
    let mut map = serde_json::Map::new();
    let mut rest = args_text.trim();
    let escape = "<escape>";
    while let Some(key_end) = rest.find(":<escape>") {
        let key = rest[..key_end].trim();
        let val_start = key_end + ":<escape>".len();
        let remainder = &rest[val_start..];
        let val_end = remainder.find(escape).unwrap_or(remainder.len());
        let raw_val = remainder[..val_end].trim();
        let value = if raw_val.eq_ignore_ascii_case("true") {
            Value::Bool(true)
        } else if raw_val.eq_ignore_ascii_case("false") {
            Value::Bool(false)
        } else if let Ok(n) = raw_val.parse::<i64>() {
            Value::Number(n.into())
        } else if let Ok(n) = raw_val.parse::<f64>() {
            Value::Number(serde_json::Number::from_f64(n).expect("TODO: Verify unwrap"))
        } else {
            Value::String(raw_val.to_string())
        };
        map.insert(key.to_string(), value);

        let next = &remainder[val_end + escape.len()..];
        rest = next.trim_start_matches(',').trim();
        if rest.is_empty() {
            break;
        }
    }
    Value::Object(map)
}

/// Returns `true` when the model output is empty or contains only whitespace
/// and invisible Unicode characters (ZWNJ, ZWJ, BOM).
pub fn is_degenerate_output(text: &str) -> bool {
    for ch in text.chars() {
        if ch.is_whitespace() {
            continue;
        }
        if matches!(ch, '\u{200c}' | '\u{200d}' | '\u{feff}') {
            continue;
        }
        return false;
    }
    true
}

pub fn parse_function_call(output: &str) -> Option<(String, Value)> {
    let start_tag = "<start_function_call>call:";
    let end_tag = "<end_function_call>";
    let start = output.find(start_tag)?;
    let rest = &output[start + start_tag.len()..];
    let end = rest.find(end_tag)?;
    let body = &rest[..end];
    let name_end = body.find('{')?;
    let name = body[..name_end].trim();
    let args_text = body[name_end + 1..].trim().trim_end_matches('}');
    let args = if args_text.is_empty() {
        serde_json::json!({})
    } else {
        parse_escape_args(args_text)
    };
    Some((name.to_string(), args))
}

/// Trim a token-ID sequence to `max_len` by keeping the first half and the
/// last half, dropping the middle.
///
/// This preserves the start-of-sequence (system prompt / BOS) and the
/// end-of-sequence (most recent user turn) while discarding the least
/// relevant middle portion.
///
/// Returns the input unchanged when `max_len` is 0 or the sequence is
/// already within bounds.
pub fn trim_input_ids(ids: Vec<u32>, max_len: usize) -> Vec<u32> {
    if max_len == 0 || ids.len() <= max_len {
        return ids;
    }
    let head = max_len / 2;
    let tail = max_len.saturating_sub(head);
    let mut trimmed = Vec::with_capacity(max_len);
    trimmed.extend_from_slice(&ids[..head]);
    trimmed.extend_from_slice(&ids[ids.len() - tail..]);
    trimmed
}
