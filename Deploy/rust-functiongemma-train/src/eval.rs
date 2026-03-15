use crate::data_gen::TrainingItem;
use anyhow::{Context, Result};
use rust_functiongemma_core::prompt::parse_function_call;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct EvaluationMetrics {
    pub total: usize,
    pub tool_name_correct: usize,
    pub arg_exact_match: usize,
    pub no_tool_correct: usize,
    pub schema_failures: usize,
    #[serde(default)]
    pub predictions: Vec<(String, String)>,
}

impl EvaluationMetrics {
    pub fn tool_accuracy(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.tool_name_correct as f64 / self.total as f64
        }
    }

    pub fn arg_accuracy(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.arg_exact_match as f64 / self.total as f64
        }
    }

    pub fn record_prediction(&mut self, expected: &str, predicted: &str) {
        self.predictions.push((expected.to_string(), predicted.to_string()));
    }

    pub fn confusion_matrix(&self) -> HashMap<(String, String), u32> {
        let mut cm = HashMap::new();
        for (exp, pred) in &self.predictions {
            *cm.entry((exp.clone(), pred.clone())).or_insert(0) += 1;
        }
        cm
    }

    pub fn per_tool_f1(&self) -> HashMap<String, f64> {
        let mut tp: HashMap<String, u32> = HashMap::new();
        let mut fp: HashMap<String, u32> = HashMap::new();
        let mut fn_count: HashMap<String, u32> = HashMap::new();

        for (exp, pred) in &self.predictions {
            if exp == pred {
                *tp.entry(exp.clone()).or_insert(0) += 1;
            } else {
                *fp.entry(pred.clone()).or_insert(0) += 1;
                *fn_count.entry(exp.clone()).or_insert(0) += 1;
            }
        }

        let mut all_labels: HashSet<String> = HashSet::new();
        all_labels.extend(tp.keys().cloned());
        all_labels.extend(fp.keys().cloned());
        all_labels.extend(fn_count.keys().cloned());

        let mut f1_map = HashMap::new();
        for label in all_labels {
            let t = *tp.get(&label).unwrap_or(&0) as f64;
            let f = *fp.get(&label).unwrap_or(&0) as f64;
            let fn_ = *fn_count.get(&label).unwrap_or(&0) as f64;

            let precision = if t + f > 0.0 { t / (t + f) } else { 0.0 };
            let recall = if t + fn_ > 0.0 { t / (t + fn_) } else { 0.0 };
            let f1 = if precision + recall > 0.0 {
                2.0 * precision * recall / (precision + recall)
            } else {
                0.0
            };
            f1_map.insert(label, f1);
        }
        f1_map
    }

    pub fn summary_report(&self) -> String {
        let mut report = String::new();
        report.push_str(&format!(
            "Evaluation Summary:\n  Total: {}\n  Tool Accuracy: {:.1}%\n  Arg Accuracy: {:.1}%\n  NO_TOOL Correct: {}\n  Schema Failures: {}\n",
            self.total,
            self.tool_accuracy() * 100.0,
            self.arg_accuracy() * 100.0,
            self.no_tool_correct,
            self.schema_failures,
        ));

        if !self.predictions.is_empty() {
            report.push_str("\nPer-Tool F1 Scores:\n");
            let f1 = self.per_tool_f1();
            let mut sorted: Vec<_> = f1.iter().collect();
            sorted.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
            for (tool, score) in sorted {
                report.push_str(&format!("  {}: {:.3}\n", tool, score));
            }
        }
        report
    }
}

#[derive(Debug)]
pub struct EvalSampleResult {
    pub tool_match: bool,
    pub arg_match: bool,
    pub no_tool_match: bool,
    pub schema_valid: bool,
    pub expected_label: String,
    pub predicted_label: String,
}

pub fn evaluate_sample(
    output: &str,
    expected: &TrainingItem,
    fast_eval: bool,
    schema_validate: bool,
) -> Result<EvalSampleResult> {
    let expected_assistant = expected
        .messages
        .iter()
        .find(|m| m.role == "assistant")
        .context("No assistant message in expected item")?;

    let expected_tool = extract_expected_tool(expected_assistant.tool_calls.as_ref());
    let parsed_tool =
        parse_function_call(output).or_else(|| parse_tool_call_json(output).and_then(extract_parsed_tool));

    let mut tool_match = false;
    let mut arg_match = false;
    let mut no_tool_match = false;
    let mut schema_valid = true;
    let mut expected_label = String::new();
    let mut predicted_label = String::new();

    match (expected_tool, parsed_tool) {
        (Some((exp_name, exp_args)), Some((act_name, act_args))) => {
            expected_label = exp_name.clone();
            predicted_label = act_name.clone();
            tool_match = exp_name == act_name;
            if tool_match {
                arg_match = exp_args == act_args;
            }
            if schema_validate {
                schema_valid = validate_tool_call_schema(&act_name, &act_args, &expected.tools)?;
            }
        }
        (None, Some((act_name, act_args))) => {
            expected_label = "NO_TOOL".to_string();
            predicted_label = act_name.clone();
            no_tool_match = false;
            if schema_validate {
                schema_valid = validate_tool_call_schema_from_any(&act_args, &expected.tools)?;
            }
        }
        (Some((exp_name, _)), None) => {
            expected_label = exp_name;
            predicted_label = "NO_TOOL".to_string();
            tool_match = false;
            arg_match = false;
            no_tool_match = false;
        }
        (None, None) => {
            no_tool_match = output.contains("NO_TOOL") || !output.contains("<start_function_call>");
            expected_label = "NO_TOOL".to_string();
            predicted_label = if no_tool_match {
                "NO_TOOL".to_string()
            } else {
                "UNKNOWN".to_string()
            };
        }
    }

    if fast_eval {
        arg_match = tool_match;
    }

    Ok(EvalSampleResult {
        tool_match,
        arg_match,
        no_tool_match,
        schema_valid,
        expected_label,
        predicted_label,
    })
}

pub fn parse_tool_call(output: &str) -> Option<Value> {
    // Attempt to find JSON array or object in output
    let start = output.find('[').or_else(|| output.find('{'))?;
    let end = output.rfind(']').or_else(|| output.rfind('}'))?;

    if end > start {
        serde_json::from_str(&output[start..=end]).ok()
    } else {
        None
    }
}

fn parse_tool_call_json(output: &str) -> Option<Value> {
    parse_tool_call(output)
}

fn extract_expected_tool(tool_calls: Option<&Value>) -> Option<(String, Value)> {
    let tool_calls = tool_calls?.as_array()?;
    let tc = tool_calls.get(0)?;
    let func = tc.get("function")?;
    let name = func.get("name")?.as_str()?.to_string();
    let args = func
        .get("arguments")
        .cloned()
        .unwrap_or(Value::Object(Default::default()));
    Some((name, args))
}

fn extract_parsed_tool(value: Value) -> Option<(String, Value)> {
    if value.is_array() {
        let arr = value.as_array()?;
        let tc = arr.get(0)?;
        let func = tc.get("function")?;
        let name = func.get("name")?.as_str()?.to_string();
        let args = func
            .get("arguments")
            .cloned()
            .unwrap_or(Value::Object(Default::default()));
        Some((name, args))
    } else if value.is_object() {
        let func = value.get("function")?;
        let name = func.get("name")?.as_str()?.to_string();
        let args = func
            .get("arguments")
            .cloned()
            .unwrap_or(Value::Object(Default::default()));
        Some((name, args))
    } else {
        None
    }
}

fn validate_tool_call_schema(tool_name: &str, args: &Value, tools: &Value) -> Result<bool> {
    let tools_arr = match tools.as_array() {
        Some(arr) => arr,
        None => return Ok(false),
    };
    for tool in tools_arr {
        let func = tool.get("function").context("Tool missing function")?;
        let name = func.get("name").and_then(|v| v.as_str()).unwrap_or("");
        if name != tool_name {
            continue;
        }
        let params = match func.get("parameters").and_then(|v| v.as_object()) {
            Some(p) => p,
            None => return Ok(false),
        };
        return Ok(validate_args_against_schema(args, params));
    }
    Ok(false)
}

fn validate_tool_call_schema_from_any(args: &Value, tools: &Value) -> Result<bool> {
    let tools_arr = match tools.as_array() {
        Some(arr) => arr,
        None => return Ok(false),
    };
    for tool in tools_arr {
        let func = tool.get("function").context("Tool missing function")?;
        let params = match func.get("parameters").and_then(|v| v.as_object()) {
            Some(p) => p,
            None => continue,
        };
        if validate_args_against_schema(args, params) {
            return Ok(true);
        }
    }
    Ok(false)
}

fn validate_args_against_schema(args: &Value, params: &serde_json::Map<String, Value>) -> bool {
    let arg_obj = match args.as_object() {
        Some(obj) => obj,
        None => return false,
    };
    let required: Vec<String> = params
        .get("required")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
        .unwrap_or_default();
    for req in required {
        if !arg_obj.contains_key(&req) {
            return false;
        }
    }
    let props = match params.get("properties").and_then(|v| v.as_object()) {
        Some(p) => p,
        None => return true,
    };
    for (key, val) in arg_obj {
        if let Some(schema) = props.get(key) {
            if !validate_value(schema, val) {
                return false;
            }
        }
    }
    true
}

fn validate_value(schema: &Value, val: &Value) -> bool {
    if let Some(enum_vals) = schema.get("enum").and_then(|v| v.as_array()) {
        return enum_vals.iter().any(|v| v == val);
    }
    let typ = schema.get("type").and_then(|v| v.as_str()).unwrap_or("string");
    match typ {
        "string" => val.is_string(),
        "boolean" => val.is_boolean(),
        "integer" => val.as_i64().is_some(),
        "number" => val.as_f64().is_some(),
        "array" => val.is_array(),
        "object" => val.is_object(),
        _ => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_prediction() {
        let mut m = EvaluationMetrics::default();
        m.record_prediction("tool_a", "tool_a");
        m.record_prediction("tool_a", "tool_b");
        assert_eq!(m.predictions.len(), 2);
    }

    #[test]
    fn test_confusion_matrix_counts() {
        let mut m = EvaluationMetrics::default();
        m.record_prediction("A", "A");
        m.record_prediction("A", "A");
        m.record_prediction("A", "B");
        m.record_prediction("B", "A");
        let cm = m.confusion_matrix();
        assert_eq!(cm[&("A".into(), "A".into())], 2);
        assert_eq!(cm[&("A".into(), "B".into())], 1);
        assert_eq!(cm[&("B".into(), "A".into())], 1);
    }

    #[test]
    fn test_confusion_matrix_empty() {
        let m = EvaluationMetrics::default();
        assert!(m.confusion_matrix().is_empty());
    }

    #[test]
    fn test_per_tool_f1_perfect() {
        let mut m = EvaluationMetrics::default();
        m.record_prediction("A", "A");
        m.record_prediction("A", "A");
        m.record_prediction("B", "B");
        let f1 = m.per_tool_f1();
        assert!((f1["A"] - 1.0).abs() < 1e-9);
        assert!((f1["B"] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_per_tool_f1_with_errors() {
        let mut m = EvaluationMetrics::default();
        // A: 2 TP, 1 FN (predicted as B), 0 FP
        // B: 1 TP, 0 FN, 1 FP (was actually A)
        m.record_prediction("A", "A");
        m.record_prediction("A", "A");
        m.record_prediction("A", "B"); // FN for A, FP for B
        m.record_prediction("B", "B");
        let f1 = m.per_tool_f1();
        // A: precision=2/2=1.0, recall=2/3=0.667, F1=0.8
        assert!((f1["A"] - 0.8).abs() < 0.01, "A F1: {}", f1["A"]);
        // B: precision=1/2=0.5, recall=1/1=1.0, F1=0.667
        assert!((f1["B"] - 0.6667).abs() < 0.01, "B F1: {}", f1["B"]);
    }

    #[test]
    fn test_per_tool_f1_empty() {
        let m = EvaluationMetrics::default();
        assert!(m.per_tool_f1().is_empty());
    }

    #[test]
    fn test_summary_report_includes_f1() {
        let mut m = EvaluationMetrics::default();
        m.total = 3;
        m.tool_name_correct = 2;
        m.record_prediction("A", "A");
        m.record_prediction("A", "A");
        m.record_prediction("B", "A");
        let report = m.summary_report();
        assert!(report.contains("Per-Tool F1"));
        assert!(report.contains("Tool Accuracy"));
    }

    #[test]
    fn test_summary_report_no_predictions() {
        let m = EvaluationMetrics::default();
        let report = m.summary_report();
        assert!(!report.contains("Per-Tool F1"));
    }

    #[test]
    fn test_eval_sample_result_labels() {
        use crate::data_gen::{Message, TrainingItem};
        use serde_json::json;

        let item = TrainingItem {
            messages: vec![
                Message {
                    role: "user".into(),
                    content: Some("test".into()),
                    tool_calls: None,
                },
                Message {
                    role: "assistant".into(),
                    content: Some("<start_function_call>test_tool({})<end_function_call>".into()),
                    tool_calls: Some(json!([{
                        "type": "function",
                        "function": {
                            "name": "test_tool",
                            "arguments": {}
                        }
                    }])),
                },
            ],
            tools: json!([{
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "description": "test",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            }]),
        };

        let result = evaluate_sample(
            "<start_function_call>test_tool({})<end_function_call>",
            &item,
            true,
            false,
        )
        .expect("evaluate_sample failed on well-formed input");

        assert_eq!(result.expected_label, "test_tool");
        assert_eq!(result.predicted_label, "test_tool");
        assert!(result.tool_match);
    }
}
