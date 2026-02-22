use serde_json::{Value, Map};
use itertools::Itertools;
use std::collections::HashSet;
use crate::data_gen::{TrainingItem, Message};

pub fn generate_arg_sets(parameters: &Map<String, Value>, max_cases: usize) -> Vec<Map<String, Value>> {
    let props = parameters.get("properties").and_then(|v| v.as_object());
    if props.is_none() {
        return vec![Map::new()];
    }
    let props = props.expect("TODO: Verify unwrap");

    let required: HashSet<&str> = parameters.get("required")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
        .unwrap_or_default();

    let mut required_specs = Vec::new();
    let mut optional_specs = Vec::new();

    for (name, schema) in props {
        if required.contains(name.as_str()) {
            required_specs.push((name.clone(), schema));
        } else {
            optional_specs.push((name.clone(), schema));
        }
    }

    let mut arg_sets: Vec<Map<String, Value>> = Vec::new();

    // Generate base sets from required parameters
    let mut required_values = Vec::new();
    for (_name, schema) in &required_specs {
        required_values.push(values_for_param(schema));
    }

    if required_values.is_empty() {
        arg_sets.push(Map::new());
    } else {
        for values in required_values.iter().multi_cartesian_product() {
            let mut set = Map::new();
            for (i, val) in values.iter().enumerate() {
                set.insert(required_specs[i].0.clone(), (*val).clone());
            }
            arg_sets.push(set);
        }
    }

    // Expand with optional parameters one by one to limit growth
    for (name, schema) in optional_specs {
        let candidates = values_for_param(schema);
        let mut new_sets = Vec::new();
        for base in &arg_sets {
            for val in &candidates {
                let mut enriched = base.clone();
                enriched.insert(name.clone(), val.clone());
                new_sets.push(enriched);
            }
        }
        arg_sets.extend(new_sets);
    }

    // Deduplicate and cap
    let mut seen = HashSet::new();
    let mut unique = Vec::new();
    for set in arg_sets {
        // Use a stable key for deduplication
        let key = serde_json::to_string(&set).expect("TODO: Verify unwrap");
        if seen.insert(key) {
            unique.push(set);
        }
        if unique.len() >= max_cases {
            break;
        }
    }

    if unique.is_empty() {
        unique.push(Map::new());
    }
    unique
}

fn values_for_param(schema: &Value) -> Vec<Value> {
    if let Some(enum_vals) = schema.get("enum").and_then(|v| v.as_array()) {
        return enum_vals.clone();
    }

    let param_type = schema.get("type").and_then(|v| v.as_str()).unwrap_or("string");
    match param_type {
        "boolean" => vec![Value::Bool(true), Value::Bool(false)],
        "integer" | "number" => {
            let min = schema.get("minimum").and_then(|v| v.as_f64());
            let max = schema.get("maximum").and_then(|v| v.as_f64());
            let values = if let (Some(min), Some(max)) = (min, max) {
                let mid = (min + max) / 2.0;
                vec![min, mid, max]
            } else {
                vec![0.0, 1.0]
            };
            if param_type == "integer" {
                values.into_iter().map(|v| Value::Number((v as i64).into())).collect()
            } else {
                values.into_iter().map(|v| Value::Number(serde_json::Number::from_f64(v).expect("TODO: Verify unwrap"))).collect()
            }
        },
        "array" => {
            let items = schema.get("items");
            let first_val = items.map(values_for_param).and_then(|v| v.into_iter().next()).unwrap_or(Value::String("item".to_string()));
            vec![Value::Array(vec![first_val])]
        },
        "object" => vec![Value::Object(Map::new())],
        _ => vec![schema.get("default").cloned().unwrap_or(Value::String("example".to_string()))],
    }
}

pub fn generate_negative_cases(tools: &Value) -> Vec<TrainingItem> {
    let mut items = Vec::new();
    let prompts = [
        "Hello, how are you today?",
        "What is the capital of France?",
        "Tell me a joke.",
        "How do I cook pasta?",
        "Write a poem about the sea.",
    ];

    for prompt in prompts {
        items.push(TrainingItem {
            messages: vec![
                Message {
                    role: "user".to_string(),
                    content: Some(prompt.to_string()),
                    tool_calls: None,
                },
                Message {
                    role: "model".to_string(),
                    content: Some("I'm sorry, I cannot perform any tool calls for that request. How else can I help you?".to_string()),
                    tool_calls: None,
                },
            ],
            tools: tools.clone(),
        });
    }
    items
}
