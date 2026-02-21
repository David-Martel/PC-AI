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
    format!(
        "<start_function_call>call:{}{{{}}}<end_function_call>",
        name, args_text
    )
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
            Value::Number(serde_json::Number::from_f64(n).unwrap())
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
