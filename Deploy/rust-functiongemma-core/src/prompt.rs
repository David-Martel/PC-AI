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

/// Parse a complete escaped argument list. Malformed arguments return null.
pub fn parse_escape_args(args_text: &str) -> Value {
    let mut map = Map::new();
    let mut rest = args_text.trim();
    while !rest.is_empty() {
        let Some((key, remainder)) = rest.split_once(":<escape>") else {
            return Value::Null;
        };
        let key = key.trim();
        if key.is_empty() || key.contains([':', ',', '{', '}', '<', '>']) {
            return Value::Null;
        }
        let Some((raw_val, next)) = remainder.split_once("<escape>") else {
            return Value::Null;
        };
        let raw_val = raw_val.trim();
        let value = if raw_val.eq_ignore_ascii_case("true") {
            Value::Bool(true)
        } else if raw_val.eq_ignore_ascii_case("false") {
            Value::Bool(false)
        } else if let Ok(n) = raw_val.parse::<i64>() {
            Value::Number(n.into())
        } else if let Ok(n) = raw_val.parse::<f64>() {
            serde_json::Number::from_f64(n)
                .map(Value::Number)
                .unwrap_or_else(|| Value::String(raw_val.to_owned()))
        } else {
            Value::String(raw_val.to_string())
        };
        if map.insert(key.to_owned(), value).is_some() {
            return Value::Null;
        }

        let next = next.trim();
        if next.is_empty() {
            break;
        }
        let Some(next) = next.strip_prefix(',') else {
            return Value::Null;
        };
        rest = next.trim();
        if rest.is_empty() {
            return Value::Null;
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
    let (_, rest) = output.split_once("<start_function_call>call:")?;
    let (body, _) = rest.split_once("<end_function_call>")?;
    let (name, rest) = body.split_once('{')?;
    let name = name.trim();
    let args = parse_escape_args(rest.trim_end().strip_suffix('}')?);
    (!name.is_empty() && args.is_object()).then(|| (name.to_owned(), args))
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // Protects valid tool calls and JSON scalar conversion.
    // Detects split delimiters inside escaped text or a fabricated numeric zero.
    // Needs only in-memory input. Breadcrumb: parse_function_call and parse_escape_args.
    #[test]
    fn complete_tool_calls_preserve_values() {
        let arguments = json!({"label": "café, east: {ready}", "enabled": true, "count": 3, "ratio": 0.25});
        let Some(map) = arguments.as_object() else {
            panic!("fixture must contain an object");
        };
        let output = format_function_call("lookup", map);
        assert_eq!(parse_function_call(&output), Some(("lookup".to_owned(), arguments)));
        assert_eq!(parse_escape_args("value:<escape>NaN<escape>"), json!({"value": "NaN"}));
        assert_eq!(
            parse_function_call(&format_function_call("lookup", &Map::new())),
            Some(("lookup".to_owned(), json!({})))
        );
    }

    // Protects the router from emitting a partial tool call.
    // Detects accepting argument prefixes as complete lists.
    // Needs only a formatted in-memory call. Breadcrumb: parse_escape_args.
    #[test]
    fn incomplete_arguments_are_rejected_without_partial_results() {
        let arguments = "a:<escape>true<escape>,b:<escape>ready<escape>";
        for end in arguments.char_indices().map(|(index, _)| index) {
            let prefix = &arguments[..end];
            let output = format!("<start_function_call>call:lookup{{{prefix}}}<end_function_call>");
            let closes_a_value = prefix.ends_with("<escape>") && prefix.matches("<escape>").count() % 2 == 0;
            if !prefix.is_empty() && !closes_a_value {
                assert!(parse_function_call(&output).is_none());
            }
        }
        assert!(parse_escape_args("a:<escape>1<escape>,a:<escape>2<escape>").is_null());
    }
}
