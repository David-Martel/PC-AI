use crate::schema_utils::generate_arg_sets;
use anyhow::{Context, Result};
use rust_functiongemma_core::prompt::format_function_call;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::fs;
use std::path::Path;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Message {
    pub role: String,
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Value>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TrainingItem {
    pub messages: Vec<Message>,
    pub tools: Value,
}

impl TrainingItem {
    pub fn to_prompt(&self) -> String {
        let mut prompt = String::new();
        for msg in &self.messages {
            let role = map_role(&msg.role);
            prompt.push_str("<start_of_turn>");
            prompt.push_str(role);
            prompt.push('\n');
            if let Some(c) = &msg.content {
                prompt.push_str(c);
            }
            prompt.push_str("<end_of_turn>\n");
        }
        prompt
    }
}

pub(crate) fn map_role(role: &str) -> &str {
    match role {
        "assistant" => "model",
        "system" => "developer",
        _ => role,
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct Scenario {
    pub mode: String,
    pub user_content: String,
    pub tool_name: Option<String>,
    #[serde(default)]
    pub tool_arguments: Map<String, Value>,
    pub assistant_content: Option<String>,
    /// Ordered list of tool names for multi-tool scenarios.
    #[serde(default)]
    pub tool_sequence: Vec<String>,
}

pub struct DataGenerator {
    pub tools: Value,
    pub default_system_msg: String,
}

impl DataGenerator {
    pub fn new(tools_path: &Path, system_prompt_path: Option<&Path>) -> Result<Self> {
        let tools_raw = fs::read_to_string(tools_path).context("Failed to read tools path")?;
        let tools_json: Value =
            serde_json::from_str(&tools_raw).context("Failed to parse tools JSON")?;
        let tools = tools_json.get("tools").cloned().unwrap_or(json!([]));

        let mut default_system_msg =
            "You are a model that can do function calling with the following functions".to_string();
        if let Some(p) = system_prompt_path {
            if p.exists() {
                let text = fs::read_to_string(p)?;
                default_system_msg = format!(
                    "{}\n\nYou are a tool-calling router. Use only the provided tools.",
                    text
                );
            }
        }

        Ok(Self {
            tools,
            default_system_msg,
        })
    }

    pub fn generate_from_schema(&self, max_cases: usize) -> Result<Vec<TrainingItem>> {
        let mut items = Vec::new();
        let tools_arr = self.tools.as_array().context("Tools is not an array")?;

        for tool in tools_arr {
            let fn_obj = tool
                .get("function")
                .context("Tool has no function object")?;
            let name = fn_obj
                .get("name")
                .and_then(|v| v.as_str())
                .context("Tool has no name")?;
            let description = fn_obj
                .get("description")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let params = fn_obj
                .get("parameters")
                .and_then(|v| v.as_object())
                .context("Tool has no parameters")?;

            for args in generate_arg_sets(params, max_cases) {
                let args_text = serde_json::to_string(&args)?;
                let user_prompt = format!(
                    "Use {} to perform the task: {}. Arguments: {}",
                    name, description, args_text
                );

                let context_aware_prompt = format!(
                    "[NATIVE_CONTEXT]\n{{\"telemetry\": \"active\", \"tool\": \"{}\"}}\n\n[USER_REQUEST]\n{}",
                    name, user_prompt
                );

                let thought_process = format!(
                    "<thought>\nUser request: \"{}\".\nReasoning: Tool '{}' performs \"{}\".\nDecision: I will call '{}' to satisfy the request.\n</thought>\n",
                    user_prompt, name, description, name
                );
                let function_call = format_function_call(name, &args);

                items.push(TrainingItem {
                    messages: vec![
                        Message {
                            role: "user".to_string(),
                            content: Some(format!(
                                "{}\n\n{}",
                                self.default_system_msg, context_aware_prompt
                            )),
                            tool_calls: None,
                        },
                        Message {
                            role: "assistant".to_string(),
                            content: Some(format!("{}\n{}", thought_process, function_call)),
                            tool_calls: Some(json!([
                                {
                                    "type": "function",
                                    "function": {
                                        "name": name,
                                        "arguments": args,
                                    }
                                }
                            ])),
                        },
                    ],
                    tools: self.tools.clone(),
                });
            }
        }

        // Add negative cases
        let negative_items = crate::schema_utils::generate_negative_cases(&self.tools);
        items.extend(negative_items);

        Ok(items)
    }

    /// Generate training items from a single scenarios JSON file.
    ///
    /// Handles three scenario modes:
    /// - `multi_tool`: produces a sequenced multi-function-call training item.
    /// - `no_tool` / `chat`: produces a NO_TOOL training item.
    /// - Scenarios with `tool_name` set (single-tool): skipped here because
    ///   they are already covered by `generate_from_schema`.
    pub fn generate_from_scenarios(&self, scenarios_path: &Path) -> Result<Vec<TrainingItem>> {
        let raw = fs::read_to_string(scenarios_path)?;
        let scenarios_val: Value = serde_json::from_str(&raw)?;

        let items_val = if scenarios_val.is_array() {
            scenarios_val.as_array().unwrap().clone()
        } else {
            scenarios_val
                .get("scenarios")
                .and_then(|v| v.as_array())
                .cloned()
                .unwrap_or_default()
        };

        let mut items = Vec::new();
        for val in items_val {
            let scenario: Scenario = serde_json::from_value(val)?;

            if !scenario.tool_sequence.is_empty() {
                // Multi-tool scenario: emit one training item that calls every
                // tool in the sequence with empty arguments.
                let calls_text: Vec<String> = scenario
                    .tool_sequence
                    .iter()
                    .map(|tool_name| format_function_call(tool_name, &Map::new()))
                    .collect();

                let thought = format!(
                    "<thought>\nUser request: \"{}\".\nReasoning: This requires multiple tools: {}.\nDecision: I will call {} tools in sequence.\n</thought>\n",
                    scenario.user_content,
                    scenario.tool_sequence.join(", "),
                    scenario.tool_sequence.len()
                );

                let tool_calls_json: Vec<Value> = scenario
                    .tool_sequence
                    .iter()
                    .map(|name| {
                        json!({
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": {}
                            }
                        })
                    })
                    .collect();

                items.push(TrainingItem {
                    messages: vec![
                        Message {
                            role: "user".to_string(),
                            content: Some(format!(
                                "{}\n\n{}",
                                self.default_system_msg, scenario.user_content
                            )),
                            tool_calls: None,
                        },
                        Message {
                            role: "assistant".to_string(),
                            content: Some(format!("{}\n{}", thought, calls_text.join("\n"))),
                            tool_calls: Some(json!(tool_calls_json)),
                        },
                    ],
                    tools: self.tools.clone(),
                });
                continue;
            }

            if scenario.tool_name.is_some() {
                // Single-tool scenarios are covered by schema gen or require
                // custom handling; skip them here.
                continue;
            }

            // NO_TOOL scenario (chat / no_tool modes).
            items.push(TrainingItem {
                messages: vec![
                    Message {
                        role: "user".to_string(),
                        content: Some(format!(
                            "{}\n\n{}",
                            self.default_system_msg, scenario.user_content
                        )),
                        tool_calls: None,
                    },
                    Message {
                        role: "assistant".to_string(),
                        content: Some(
                            scenario
                                .assistant_content
                                .unwrap_or_else(|| "NO_TOOL".to_string()),
                        ),
                        tool_calls: None,
                    },
                ],
                tools: self.tools.clone(),
            });
        }
        Ok(items)
    }

    /// Load and generate training items from every `*.json` file found
    /// directly inside `dir`.  Non-JSON files are silently ignored.
    pub fn generate_from_scenario_dir(&self, dir: &Path) -> Result<Vec<TrainingItem>> {
        let mut items = Vec::new();
        if dir.is_dir() {
            for entry in std::fs::read_dir(dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) == Some("json") {
                    items.extend(self.generate_from_scenarios(&path)?);
                }
            }
        }
        Ok(items)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_tools_json() -> Value {
        json!([{
            "type": "function",
            "function": {
                "name": "pcai_get_disk_status",
                "description": "Get disk health status",
                "parameters": { "type": "object", "properties": {} }
            }
        }])
    }

    fn make_generator() -> DataGenerator {
        DataGenerator {
            tools: make_tools_json(),
            default_system_msg: "test".into(),
        }
    }

    #[test]
    fn test_scenario_no_tool() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("neg.json");
        std::fs::write(
            &path,
            r#"{"scenarios":[{"mode":"no_tool","user_content":"Hello","assistant_content":"NO_TOOL"}]}"#,
        )
        .unwrap();
        let gen = make_generator();
        let items = gen.generate_from_scenarios(&path).unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].messages[1].content.as_deref(), Some("NO_TOOL"));
        assert!(items[0].messages[1].tool_calls.is_none());
    }

    #[test]
    fn test_scenario_no_tool_missing_assistant_content_defaults_to_no_tool() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("neg.json");
        // assistant_content omitted - should default to "NO_TOOL"
        std::fs::write(
            &path,
            r#"{"scenarios":[{"mode":"no_tool","user_content":"Hello"}]}"#,
        )
        .unwrap();
        let gen = make_generator();
        let items = gen.generate_from_scenarios(&path).unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].messages[1].content.as_deref(), Some("NO_TOOL"));
    }

    #[test]
    fn test_scenario_chat_mode_treated_as_no_tool() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("chat.json");
        std::fs::write(
            &path,
            r#"{"scenarios":[{"mode":"chat","user_content":"Explain WSL","assistant_content":"NO_TOOL"}]}"#,
        )
        .unwrap();
        let gen = make_generator();
        let items = gen.generate_from_scenarios(&path).unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].messages[1].content.as_deref(), Some("NO_TOOL"));
    }

    #[test]
    fn test_scenario_single_tool_skipped() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("single.json");
        std::fs::write(
            &path,
            r#"{"scenarios":[{"mode":"diagnose","user_content":"Check disk","tool_name":"pcai_get_disk_status","tool_arguments":{}}]}"#,
        )
        .unwrap();
        let gen = make_generator();
        let items = gen.generate_from_scenarios(&path).unwrap();
        // Single-tool scenarios are skipped in generate_from_scenarios.
        assert_eq!(items.len(), 0);
    }

    #[test]
    fn test_scenario_multi_tool() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("multi.json");
        std::fs::write(
            &path,
            r#"{"scenarios":[{"mode":"multi_tool","user_content":"Check disk and USB","tool_sequence":["pcai_get_disk_status","pcai_get_usb_status"]}]}"#,
        )
        .unwrap();
        let gen = make_generator();
        let items = gen.generate_from_scenarios(&path).unwrap();
        assert_eq!(items.len(), 1);
        let tc = items[0].messages[1].tool_calls.as_ref().unwrap();
        let arr = tc.as_array().unwrap();
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0]["function"]["name"], "pcai_get_disk_status");
        assert_eq!(arr[1]["function"]["name"], "pcai_get_usb_status");
    }

    #[test]
    fn test_scenario_multi_tool_three_tools() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("multi3.json");
        std::fs::write(
            &path,
            r#"{"scenarios":[{"mode":"multi_tool","user_content":"Full check","tool_sequence":["pcai_get_network_status","pcai_get_disk_status","pcai_get_usb_status"]}]}"#,
        )
        .unwrap();
        let gen = make_generator();
        let items = gen.generate_from_scenarios(&path).unwrap();
        assert_eq!(items.len(), 1);
        let arr = items[0].messages[1]
            .tool_calls
            .as_ref()
            .unwrap()
            .as_array()
            .unwrap();
        assert_eq!(arr.len(), 3);
    }

    #[test]
    fn test_scenario_multi_tool_thought_mentions_tool_count() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("thought.json");
        std::fs::write(
            &path,
            r#"{"scenarios":[{"mode":"multi_tool","user_content":"Disk and USB","tool_sequence":["pcai_get_disk_status","pcai_get_usb_status"]}]}"#,
        )
        .unwrap();
        let gen = make_generator();
        let items = gen.generate_from_scenarios(&path).unwrap();
        let content = items[0].messages[1].content.as_deref().unwrap();
        assert!(
            content.contains("2 tools"),
            "thought should mention tool count"
        );
        assert!(content.contains("<thought>"));
    }

    #[test]
    fn test_scenario_dir_loads_all_json() {
        let dir = TempDir::new().unwrap();
        std::fs::write(
            dir.path().join("a.json"),
            r#"{"scenarios":[{"mode":"no_tool","user_content":"Hi"}]}"#,
        )
        .unwrap();
        std::fs::write(
            dir.path().join("b.json"),
            r#"{"scenarios":[{"mode":"no_tool","user_content":"Bye"}]}"#,
        )
        .unwrap();
        std::fs::write(dir.path().join("not.txt"), "ignored").unwrap();
        let gen = make_generator();
        let items = gen.generate_from_scenario_dir(dir.path()).unwrap();
        assert_eq!(items.len(), 2);
    }

    #[test]
    fn test_scenario_dir_empty_dir_returns_empty() {
        let dir = TempDir::new().unwrap();
        let gen = make_generator();
        let items = gen.generate_from_scenario_dir(dir.path()).unwrap();
        assert!(items.is_empty());
    }

    #[test]
    fn test_scenario_dir_mixes_modes() {
        let dir = TempDir::new().unwrap();
        std::fs::write(
            dir.path().join("no_tool.json"),
            r#"{"scenarios":[{"mode":"no_tool","user_content":"Hello"},{"mode":"no_tool","user_content":"Thanks"}]}"#,
        )
        .unwrap();
        std::fs::write(
            dir.path().join("multi.json"),
            r#"{"scenarios":[{"mode":"multi_tool","user_content":"Check all","tool_sequence":["pcai_get_disk_status","pcai_get_usb_status"]}]}"#,
        )
        .unwrap();
        let gen = make_generator();
        let items = gen.generate_from_scenario_dir(dir.path()).unwrap();
        // 2 no_tool + 1 multi_tool = 3 total
        assert_eq!(items.len(), 3);
    }
}
