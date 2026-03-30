use anyhow::Result;
use chrono::{Duration, NaiveDateTime, Utc};
use serde::Serialize;
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

#[derive(Debug, Serialize)]
pub struct ProcessLassoSnapshot {
    pub generated_at: String,
    pub config_path: String,
    pub log_path: String,
    pub sections: BTreeMap<String, BTreeMap<String, String>>,
    pub summary: ProcessLassoSummary,
    pub log_summary: ProcessLassoLogSummary,
}

#[derive(Debug, Serialize)]
pub struct ProcessLassoSummary {
    pub start_with_power_plan: Option<String>,
    pub gaming_mode_enabled: Option<bool>,
    pub target_power_plan: Option<String>,
    pub ooc_exclusions: Vec<String>,
    pub smart_trim_exclusions: Vec<String>,
    pub efficiency_mode_off: Vec<String>,
    pub default_priorities: BTreeMap<String, String>,
    pub log_efficiency_mode: Option<bool>,
    pub log_cpu_sets: Option<bool>,
}

#[derive(Debug, Serialize, Default)]
pub struct ProcessLassoLogSummary {
    pub lookback_minutes: u32,
    pub total_events: usize,
    pub efficiency_mode_events: usize,
    pub cpu_set_events: usize,
    pub smart_trim_events: usize,
    pub power_profile_events: usize,
    pub actions: BTreeMap<String, usize>,
    pub processes: BTreeMap<String, usize>,
}

pub fn collect_snapshot(config_path: &str, log_path: &str, lookback_minutes: u32) -> Result<ProcessLassoSnapshot> {
    let sections = parse_ini(config_path)?;
    let log_summary = parse_log(log_path, lookback_minutes)?;

    let summary = ProcessLassoSummary {
        start_with_power_plan: get_value(&sections, "PowerManagement", "StartWithPowerPlan"),
        gaming_mode_enabled: get_value(&sections, "GamingMode", "GamingModeEnabled").and_then(|v| parse_bool(&v)),
        target_power_plan: get_value(&sections, "GamingMode", "TargetPowerPlan"),
        ooc_exclusions: split_csv_field(
            get_value(&sections, "OutOfControlProcessRestraint", "OocExclusions").as_deref(),
        ),
        smart_trim_exclusions: split_csv_field(
            get_value(&sections, "MemoryManagement", "SmartTrimExclusions").as_deref(),
        ),
        efficiency_mode_off: parse_efficiency_mode(
            get_value(&sections, "ProcessAllowances", "EfficiencyMode").as_deref(),
        ),
        default_priorities: parse_priority_pairs(
            get_value(&sections, "ProcessDefaults", "DefaultPriorities").as_deref(),
        ),
        log_efficiency_mode: get_value(&sections, "Logging", "LogEfficiencyMode").and_then(|v| parse_bool(&v)),
        log_cpu_sets: get_value(&sections, "Logging", "LogCPUSets").and_then(|v| parse_bool(&v)),
    };

    Ok(ProcessLassoSnapshot {
        generated_at: Utc::now().to_rfc3339(),
        config_path: config_path.to_string(),
        log_path: log_path.to_string(),
        sections,
        summary,
        log_summary,
    })
}

fn parse_ini(config_path: &str) -> Result<BTreeMap<String, BTreeMap<String, String>>> {
    let text = read_text_file(config_path)?;
    let mut sections = BTreeMap::<String, BTreeMap<String, String>>::new();
    let mut current = String::from("global");

    for raw_line in text.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with(';') || line.starts_with('#') {
            continue;
        }

        if line.starts_with('[') && line.ends_with(']') && line.len() > 2 {
            current = line[1..line.len() - 1].trim().to_string();
            sections.entry(current.clone()).or_default();
            continue;
        }

        if let Some((key, value)) = line.split_once('=') {
            sections
                .entry(current.clone())
                .or_default()
                .insert(key.trim().to_string(), value.trim().to_string());
        }
    }

    Ok(sections)
}

fn parse_log(log_path: &str, lookback_minutes: u32) -> Result<ProcessLassoLogSummary> {
    if !Path::new(log_path).exists() {
        return Ok(ProcessLassoLogSummary {
            lookback_minutes,
            ..ProcessLassoLogSummary::default()
        });
    }

    let text = read_text_file(log_path)?;
    let cutoff = Utc::now().naive_utc() - Duration::minutes(i64::from(lookback_minutes));
    let mut summary = ProcessLassoLogSummary {
        lookback_minutes,
        ..ProcessLassoLogSummary::default()
    };

    for line in text.lines() {
        let fields = parse_csv_line(line);
        if fields.len() < 9 {
            continue;
        }

        let timestamp = match NaiveDateTime::parse_from_str(fields[1].as_str(), "%Y-%m-%d %H:%M:%S") {
            Ok(value) => value,
            Err(_) => continue,
        };

        if timestamp < cutoff {
            continue;
        }

        let process = fields[5].trim().to_ascii_lowercase();
        let action = fields[7].trim().to_string();

        summary.total_events += 1;
        *summary.actions.entry(action.clone()).or_insert(0) += 1;
        *summary.processes.entry(process).or_insert(0) += 1;

        let lowered_action = action.to_ascii_lowercase();
        if lowered_action.contains("efficiency mode") {
            summary.efficiency_mode_events += 1;
        }
        if lowered_action.contains("cpu set") {
            summary.cpu_set_events += 1;
        }
        if lowered_action.contains("smarttrim") {
            summary.smart_trim_events += 1;
        }
        if lowered_action.contains("power profile") {
            summary.power_profile_events += 1;
        }
    }

    Ok(summary)
}

fn read_text_file(path: &str) -> Result<String> {
    let bytes = fs::read(path)?;
    if bytes.starts_with(&[0xFF, 0xFE]) {
        let utf16: Vec<u16> = bytes[2..]
            .chunks_exact(2)
            .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
            .collect();
        return Ok(String::from_utf16_lossy(&utf16));
    }

    if bytes.starts_with(&[0xFE, 0xFF]) {
        let utf16: Vec<u16> = bytes[2..]
            .chunks_exact(2)
            .map(|chunk| u16::from_be_bytes([chunk[0], chunk[1]]))
            .collect();
        return Ok(String::from_utf16_lossy(&utf16));
    }

    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

fn parse_csv_line(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;

    for ch in line.chars() {
        match ch {
            '"' => in_quotes = !in_quotes,
            ',' if !in_quotes => {
                fields.push(current.trim().trim_matches('"').to_string());
                current.clear();
            }
            _ => current.push(ch),
        }
    }

    if !current.is_empty() || line.ends_with(',') {
        fields.push(current.trim().trim_matches('"').to_string());
    }

    fields
}

fn get_value(sections: &BTreeMap<String, BTreeMap<String, String>>, section: &str, key: &str) -> Option<String> {
    sections.get(section).and_then(|s| s.get(key)).cloned()
}

fn parse_bool(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "true" => Some(true),
        "false" => Some(false),
        _ => None,
    }
}

fn split_csv_field(value: Option<&str>) -> Vec<String> {
    value
        .unwrap_or_default()
        .split(',')
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .map(ToString::to_string)
        .collect()
}

fn parse_efficiency_mode(value: Option<&str>) -> Vec<String> {
    let parts = split_csv_field(value);
    let mut executables = Vec::new();

    for chunk in parts.chunks(2) {
        if let Some(name) = chunk.first() {
            executables.push(name.clone());
        }
    }

    executables
}

fn parse_priority_pairs(value: Option<&str>) -> BTreeMap<String, String> {
    let parts = split_csv_field(value);
    let mut priorities = BTreeMap::new();

    for chunk in parts.chunks(2) {
        if let [name, priority] = chunk {
            priorities.insert(name.clone(), priority.clone());
        }
    }

    priorities
}
