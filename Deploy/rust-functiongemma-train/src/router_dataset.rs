use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::data_gen::{Message, Scenario, TrainingItem};
use crate::schema_utils::generate_arg_sets;
use rust_functiongemma_core::prompt::format_function_call;

#[derive(Debug, Clone)]
pub struct RouterDatasetConfig {
    pub output: PathBuf,
    pub tools_path: PathBuf,
    pub diagnose_prompt: PathBuf,
    pub chat_prompt: PathBuf,
    pub scenarios_path: Option<PathBuf>,
    pub include_tool_coverage: bool,
    pub max_cases: usize,
}

#[derive(Debug, Serialize, Clone)]
pub struct ToolTestVector {
    pub tool: String,
    pub arguments: Map<String, Value>,
}

fn load_tools(path: &Path) -> Result<Vec<Value>> {
    let raw = fs::read_to_string(path).context("Failed to read tools path")?;
    let tools_json: Value = serde_json::from_str(&raw).context("Failed to parse tools JSON")?;
    let tools = tools_json
        .get("tools")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    Ok(tools)
}

fn load_prompt(path: &Path) -> String {
    if path.exists() {
        fs::read_to_string(path).unwrap_or_default()
    } else {
        String::new()
    }
}

fn load_scenarios(path: Option<&Path>) -> Vec<Scenario> {
    if let Some(p) = path {
        if p.exists() {
            if let Ok(raw) = fs::read_to_string(p) {
                if let Ok(val) = serde_json::from_str::<Value>(&raw) {
                    let items = if val.is_array() {
                        val.as_array().cloned().unwrap_or_default()
                    } else {
                        val.get("scenarios")
                            .and_then(|v| v.as_array())
                            .cloned()
                            .unwrap_or_default()
                    };
                    let mut scenarios = Vec::new();
                    for item in items {
                        if let Ok(s) = serde_json::from_value::<Scenario>(item) {
                            scenarios.push(s);
                        }
                    }
                    if !scenarios.is_empty() {
                        return scenarios;
                    }
                }
            }
        }
    }

    vec![
        Scenario {
            mode: "diagnose".to_string(),
            user_content: "Run a WSL network diagnosis and summarize any failures.".to_string(),
            tool_name: Some("pcai_run_wsl_network_tool".to_string()),
            tool_arguments: Map::from_iter([("mode".to_string(), Value::String("diagnose".to_string()))]),
            assistant_content: None,
            tool_sequence: Vec::new(),
        },
        Scenario {
            mode: "diagnose".to_string(),
            user_content: "Check the WSL/Docker environment health and report status.".to_string(),
            tool_name: Some("pcai_get_wsl_health".to_string()),
            tool_arguments: Map::new(),
            assistant_content: None,
            tool_sequence: Vec::new(),
        },
        Scenario {
            mode: "diagnose".to_string(),
            user_content: "Restart WSL because networking is stuck.".to_string(),
            tool_name: Some("pcai_restart_wsl".to_string()),
            tool_arguments: Map::new(),
            assistant_content: None,
            tool_sequence: Vec::new(),
        },
        Scenario {
            mode: "diagnose".to_string(),
            user_content: "Check Docker Desktop health and return a summary.".to_string(),
            tool_name: Some("pcai_get_docker_status".to_string()),
            tool_arguments: Map::new(),
            assistant_content: None,
            tool_sequence: Vec::new(),
        },
        Scenario {
            mode: "chat".to_string(),
            user_content: "Explain what WSL is and when to use it.".to_string(),
            tool_name: None,
            tool_arguments: Map::new(),
            assistant_content: Some("NO_TOOL".to_string()),
            tool_sequence: Vec::new(),
        },
        Scenario {
            mode: "chat".to_string(),
            user_content: "What does vLLM do and when should I use it?".to_string(),
            tool_name: None,
            tool_arguments: Map::new(),
            assistant_content: Some("NO_TOOL".to_string()),
            tool_sequence: Vec::new(),
        },
    ]
}

fn build_system_prompt(mode: &str, diagnose_prompt: &str, chat_prompt: &str) -> String {
    let router_rules = "You are a tool-calling router for PC-AI. Use only the tools provided in the schema. If a tool call is required, return tool_calls only. If no tool is needed, respond with NO_TOOL.";
    if mode.eq_ignore_ascii_case("chat") {
        format!("{}\n\n{}", chat_prompt, router_rules)
    } else {
        format!("{}\n\n{}", diagnose_prompt, router_rules)
    }
}

fn build_tool_prompt(name: &str, description: &str, args: &Map<String, Value>) -> String {
    let args_text = serde_json::to_string(args).unwrap_or_else(|_| "{}".to_string());
    format!(
        "Use {} to perform the task: {}. Arguments: {}",
        name, description, args_text
    )
}

fn tool_call_payload(name: &str, args: &Map<String, Value>) -> Value {
    json!([
        {
            "type": "function",
            "function": {
                "name": name,
                "arguments": args
            }
        }
    ])
}

fn build_conversation(scenario: &Scenario, tools: &[Value], diagnose_prompt: &str, chat_prompt: &str) -> TrainingItem {
    let system_msg = build_system_prompt(&scenario.mode, diagnose_prompt, chat_prompt);
    let mut messages = vec![
        Message {
            role: "developer".to_string(),
            content: Some(system_msg),
            tool_calls: None,
        },
        Message {
            role: "user".to_string(),
            content: Some(scenario.user_content.clone()),
            tool_calls: None,
        },
    ];

    if let Some(tool_name) = &scenario.tool_name {
        let function_call = format_function_call(tool_name, &scenario.tool_arguments);
        messages.push(Message {
            role: "assistant".to_string(),
            content: Some(function_call),
            tool_calls: Some(tool_call_payload(tool_name, &scenario.tool_arguments)),
        });
    } else {
        messages.push(Message {
            role: "assistant".to_string(),
            content: Some(
                scenario
                    .assistant_content
                    .clone()
                    .unwrap_or_else(|| "NO_TOOL".to_string()),
            ),
            tool_calls: None,
        });
    }

    TrainingItem {
        messages,
        tools: Value::Array(tools.to_vec()),
    }
}

pub fn build_router_dataset(cfg: &RouterDatasetConfig) -> Result<Vec<TrainingItem>> {
    let tools = load_tools(&cfg.tools_path)?;
    let scenarios = load_scenarios(cfg.scenarios_path.as_deref());
    let diagnose_prompt = load_prompt(&cfg.diagnose_prompt);
    let chat_prompt = load_prompt(&cfg.chat_prompt);

    let mut items: Vec<TrainingItem> = scenarios
        .iter()
        .map(|s| build_conversation(s, &tools, &diagnose_prompt, &chat_prompt))
        .collect();

    if cfg.include_tool_coverage {
        for tool in &tools {
            let fn_obj = tool.get("function").context("Tool has no function object")?;
            let name = fn_obj
                .get("name")
                .and_then(|v| v.as_str())
                .context("Tool has no name")?;
            let description = fn_obj.get("description").and_then(|v| v.as_str()).unwrap_or("");
            let empty_params = Map::new();
            let params = fn_obj
                .get("parameters")
                .and_then(|v| v.as_object())
                .unwrap_or(&empty_params);

            for args in generate_arg_sets(params, cfg.max_cases) {
                let scenario = Scenario {
                    mode: "diagnose".to_string(),
                    user_content: build_tool_prompt(name, description, &args),
                    tool_name: Some(name.to_string()),
                    tool_arguments: args,
                    assistant_content: None,
                    tool_sequence: Vec::new(),
                };
                items.push(build_conversation(&scenario, &tools, &diagnose_prompt, &chat_prompt));
            }
        }
    }

    Ok(items)
}

pub fn write_jsonl_streaming(cfg: &RouterDatasetConfig) -> Result<usize> {
    let tools = load_tools(&cfg.tools_path)?;
    let scenarios = load_scenarios(cfg.scenarios_path.as_deref());
    let diagnose_prompt = load_prompt(&cfg.diagnose_prompt);
    let chat_prompt = load_prompt(&cfg.chat_prompt);

    if let Some(parent) = cfg.output.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }

    let file = fs::File::create(&cfg.output)?;
    let mut writer = BufWriter::new(file);
    let mut count = 0usize;

    for scenario in &scenarios {
        let item = build_conversation(scenario, &tools, &diagnose_prompt, &chat_prompt);
        serde_json::to_writer(&mut writer, &item)?;
        writer.write_all(b"\n")?;
        count += 1;
    }

    if cfg.include_tool_coverage {
        for tool in &tools {
            let fn_obj = tool.get("function").context("Tool has no function object")?;
            let name = fn_obj
                .get("name")
                .and_then(|v| v.as_str())
                .context("Tool has no name")?;
            let description = fn_obj.get("description").and_then(|v| v.as_str()).unwrap_or("");
            let empty_params = Map::new();
            let params = fn_obj
                .get("parameters")
                .and_then(|v| v.as_object())
                .unwrap_or(&empty_params);

            for args in generate_arg_sets(params, cfg.max_cases) {
                let scenario = Scenario {
                    mode: "diagnose".to_string(),
                    user_content: build_tool_prompt(name, description, &args),
                    tool_name: Some(name.to_string()),
                    tool_arguments: args,
                    assistant_content: None,
                    tool_sequence: Vec::new(),
                };
                let item = build_conversation(&scenario, &tools, &diagnose_prompt, &chat_prompt);
                serde_json::to_writer(&mut writer, &item)?;
                writer.write_all(b"\n")?;
                count += 1;
            }
        }
    }

    writer.flush()?;
    Ok(count)
}

pub fn build_tool_test_vectors(tools_path: &Path, max_cases: usize) -> Result<Vec<ToolTestVector>> {
    let tools = load_tools(tools_path)?;
    let mut vectors = Vec::new();

    for tool in &tools {
        let fn_obj = tool.get("function").context("Tool has no function object")?;
        let name = fn_obj
            .get("name")
            .and_then(|v| v.as_str())
            .context("Tool has no name")?;
        let empty_params = Map::new();
        let params = fn_obj
            .get("parameters")
            .and_then(|v| v.as_object())
            .unwrap_or(&empty_params);

        for args in generate_arg_sets(params, max_cases) {
            vectors.push(ToolTestVector {
                tool: name.to_string(),
                arguments: args,
            });
        }
    }

    Ok(vectors)
}

pub fn write_jsonl(path: &Path, items: &[TrainingItem]) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let mut out = String::new();
    for item in items {
        out.push_str(&serde_json::to_string(item)?);
        out.push('\n');
    }
    fs::write(path, out)?;
    Ok(())
}

pub fn write_test_vectors(path: &Path, vectors: &[ToolTestVector]) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let out = serde_json::to_string_pretty(vectors)?;
    fs::write(path, out)?;
    Ok(())
}
