use rust_functiongemma_train::router_dataset::{
    build_router_dataset, build_tool_test_vectors, write_jsonl, write_test_vectors, RouterDatasetConfig,
};
use serde_json::json;
use std::fs;
use tempfile::tempdir;

#[test]
fn builds_router_dataset_and_writes_jsonl() {
    let dir = tempdir().expect("temp dir");
    let tools_path = dir.path().join("tools.json");
    let diagnose_path = dir.path().join("DIAGNOSE.md");
    let chat_path = dir.path().join("CHAT.md");
    let scenarios_path = dir.path().join("scenarios.json");
    let output_path = dir.path().join("out.jsonl");
    let vectors_path = dir.path().join("vectors.json");

    let tools = json!({
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "pcai_get_status",
                    "description": "Get status",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "mode": { "type": "string", "enum": ["diagnose", "chat"] }
                        },
                        "required": ["mode"]
                    }
                }
            }
        ]
    });
    fs::write(&tools_path, serde_json::to_string(&tools).expect("serialization should succeed")).expect("write tools");
    fs::write(&diagnose_path, "DIAGNOSE PROMPT").expect("write diagnose");
    fs::write(&chat_path, "CHAT PROMPT").expect("write chat");
    fs::write(
        &scenarios_path,
        serde_json::to_string(&json!({
            "scenarios": [
                {
                    "mode": "chat",
                    "user_content": "Explain WSL.",
                    "assistant_content": "NO_TOOL"
                }
            ]
        }))
        .expect("test operation should succeed"),
    )
    .expect("write scenarios");

    let cfg = RouterDatasetConfig {
        output: output_path.clone(),
        tools_path,
        diagnose_prompt: diagnose_path,
        chat_prompt: chat_path,
        scenarios_path: Some(scenarios_path),
        include_tool_coverage: true,
        max_cases: 1,
    };

    let items = build_router_dataset(&cfg).expect("build dataset");
    assert!(items.len() >= 2);
    assert!(items
        .iter()
        .any(|item| { item.messages.iter().any(|m| m.tool_calls.is_some()) }));
    assert!(items
        .iter()
        .any(|item| { item.messages.iter().any(|m| m.content.as_deref() == Some("NO_TOOL")) }));

    write_jsonl(&output_path, &items).expect("write jsonl");
    let output = fs::read_to_string(&output_path).expect("read jsonl");
    let lines: Vec<&str> = output.lines().collect();
    assert_eq!(lines.len(), items.len());

    let vectors = build_tool_test_vectors(&cfg.tools_path, cfg.max_cases).expect("build vectors");
    write_test_vectors(&vectors_path, &vectors).expect("write vectors");
    let vectors_raw = fs::read_to_string(&vectors_path).expect("read vectors");
    let vectors_json: serde_json::Value = serde_json::from_str(&vectors_raw).expect("parse vectors");
    assert!(vectors_json.as_array().map(|v| !v.is_empty()).unwrap_or(false));
}
