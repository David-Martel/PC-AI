use anyhow::Result;
use chrono::{DateTime, Duration, Local, NaiveDateTime, Utc};
use regex::Regex;
use serde::Serialize;
use std::collections::{BTreeMap, VecDeque};
use std::fs;
use std::path::Path;
use std::sync::OnceLock;

#[derive(Debug, Serialize)]
pub struct ProcessLassoSnapshot {
    pub generated_at: String,
    pub config_path: String,
    pub log_path: String,
    pub config_exists: bool,
    pub log_exists: bool,
    pub config_modified_utc: Option<String>,
    pub log_modified_utc: Option<String>,
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
    /// Explicit code/eventcode labels only; CSV positions are not guessed.
    pub event_codes: BTreeMap<u32, usize>,
    /// Heuristic labels are diagnostics, not a provider-defined severity.
    pub inferred_levels: BTreeMap<String, usize>,
    pub recent_events: VecDeque<ProcessLassoEvent>,
    pub retained_events_truncated: bool,
}

#[derive(Debug, Serialize)]
pub struct ProcessLassoEvent {
    /// Preserve the log's naive timestamp without inventing a timezone.
    pub timestamp: String,
    pub process: String,
    pub action: String,
    pub code: Option<u32>,
    pub inferred_level: String,
    pub raw: String,
    pub raw_truncated: bool,
}

fn modified_utc(path: &str) -> Option<String> {
    fs::metadata(path)
        .ok()?
        .modified()
        .ok()
        .map(DateTime::<Utc>::from)
        .map(|v| v.to_rfc3339())
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
        config_exists: Path::new(config_path).is_file(),
        log_exists: Path::new(log_path).is_file(),
        config_modified_utc: modified_utc(config_path),
        log_modified_utc: modified_utc(log_path),
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
    // Process Lasso CSV timestamps have no zone, matching the PowerShell
    // fallback's local-clock interpretation. Do not compare them to UTC.
    Ok(parse_log_text(&text, lookback_minutes, Local::now().naive_local()))
}

fn parse_log_text(text: &str, lookback_minutes: u32, now: NaiveDateTime) -> ProcessLassoLogSummary {
    let cutoff = now - Duration::minutes(i64::from(lookback_minutes));
    let mut summary = ProcessLassoLogSummary {
        lookback_minutes,
        ..ProcessLassoLogSummary::default()
    };
    static CODE: OnceLock<Regex> = OnceLock::new();
    let code_pattern =
        CODE.get_or_init(|| Regex::new(r"(?i)\b(?:code|eventcode)\s*[:=]\s*(\d+)\b").expect("literal code regex")); // ast-grep-ignore: mandate-rust-panic // SAFETY: Fixed internal regex is validated by parser controls; callers cannot supply a pattern.

    for line in text.lines() {
        let fields = parse_csv_line(line);
        if fields.len() < 9 {
            continue;
        }

        let timestamp = match NaiveDateTime::parse_from_str(fields[1].as_str(), "%Y-%m-%d %H:%M:%S") {
            Ok(value) => value,
            Err(_) => continue,
        };

        if timestamp < cutoff || timestamp > now {
            continue;
        }

        let process = fields[5].trim().to_ascii_lowercase();
        let action = fields[7].trim().to_string();

        summary.total_events += 1;
        *summary.actions.entry(action.clone()).or_insert(0) += 1;
        *summary.processes.entry(process.clone()).or_insert(0) += 1;

        let code = code_pattern.captures(&action).and_then(|c| c[1].parse::<u32>().ok());
        if let Some(code) = code {
            *summary.event_codes.entry(code).or_insert(0) += 1;
        }
        let lower = action.to_ascii_lowercase();
        let level = if lower.contains("error") {
            "error"
        } else if lower.contains("warn") {
            "warning"
        } else {
            "info"
        };
        *summary.inferred_levels.entry(level.to_string()).or_insert(0) += 1;
        let raw: String = line.chars().take(4096).collect();
        let raw_truncated = raw.len() < line.len();
        if summary.recent_events.len() == 128 {
            summary.recent_events.pop_front();
            summary.retained_events_truncated = true;
        }
        summary.recent_events.push_back(ProcessLassoEvent {
            timestamp: fields[1].clone(),
            process,
            action: action.clone(),
            code,
            inferred_level: level.to_string(),
            raw,
            raw_truncated,
        });

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

    summary
}

fn read_text_file(path: &str) -> Result<String> {
    let bytes = fs::read(path)?;
    if bytes.starts_with(&[0xFF, 0xFE]) {
        let utf16: Vec<u16> = bytes[2..]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|chunk| u16::from_le_bytes(*chunk))
            .collect();
        return Ok(String::from_utf16_lossy(&utf16));
    }

    if bytes.starts_with(&[0xFE, 0xFF]) {
        let utf16: Vec<u16> = bytes[2..]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|chunk| u16::from_be_bytes(*chunk))
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct OwnedDirectory(PathBuf);
    impl OwnedDirectory {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "pcai-lasso-owned-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).expect("fresh owned directory");
            Self(path)
        }
    }
    impl Drop for OwnedDirectory {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).expect("remove only owned fixture");
        }
    }
    fn now() -> NaiveDateTime {
        NaiveDateTime::parse_from_str("2026-01-10 12:00:00", "%Y-%m-%d %H:%M:%S").unwrap()
    }
    fn event(time: &str, message: &str) -> String {
        format!("0,{time},0,0,0,Example.EXE,0,{message},0\n")
    }

    #[test]
    fn metadata_keeps_sections_and_actual_file_provenance_for_both_encodings() {
        let owned = OwnedDirectory::new();
        let config = owned.0.join("settings.ini");
        let log = owned.0.join("events.csv");
        let text = "[Logging]\nSame=first\nLogCPUSets=true\n[Other]\nSame=second\n";
        for utf16 in [false, true] {
            let data = if utf16 {
                [
                    vec![0xff, 0xfe],
                    text.encode_utf16().flat_map(u16::to_le_bytes).collect(),
                ]
                .concat()
            } else {
                text.as_bytes().to_vec()
            };
            fs::write(&config, data).unwrap();
            let result = collect_snapshot(config.to_str().unwrap(), log.to_str().unwrap(), 60).unwrap();
            assert_eq!(result.sections["Logging"]["Same"], "first");
            assert_eq!(result.sections["Other"]["Same"], "second");
            assert_eq!(result.summary.log_cpu_sets, Some(true));
            assert!(result.config_exists && result.config_modified_utc.is_some());
            assert!(!result.log_exists && result.log_modified_utc.is_none());
            assert_eq!(result.log_summary.total_events, 0);
            let json = serde_json::to_value(&result).unwrap();
            assert!(
                json.get("generated_at").is_some() && json.get("summary").is_some() && json.get("sections").is_some()
            );
        }
    }

    #[test]
    fn numeric_and_unknown_boolean_text_stay_unqualified_with_raw_settings_retained() {
        // Preserve the established true/false-only typed summary. A metadata
        // port does not establish provider semantics for numeric spellings.
        let owned = OwnedDirectory::new();
        let config = owned.0.join("settings.ini");
        let log = owned.0.join("absent.csv");
        fs::write(&config, "[Logging]\nLogCPUSets=1\nLogEfficiencyMode=unknown\n").unwrap();
        let result = collect_snapshot(config.to_str().unwrap(), log.to_str().unwrap(), 60).unwrap();
        assert_eq!(result.sections["Logging"]["LogCPUSets"], "1");
        assert_eq!(result.sections["Logging"]["LogEfficiencyMode"], "unknown");
        assert_eq!(result.summary.log_cpu_sets, None);
        assert_eq!(result.summary.log_efficiency_mode, None);
        assert_eq!(parse_bool(" TRUE "), Some(true));
        assert_eq!(parse_bool("false"), Some(false));
        assert_eq!(parse_bool("0"), None);
    }

    #[test]
    fn missing_config_keeps_the_canonical_read_error() {
        let owned = OwnedDirectory::new();
        let missing = owned.0.join("absent");
        assert!(collect_snapshot(missing.to_str().unwrap(), missing.to_str().unwrap(), 60).is_err());
    }

    #[test]
    fn log_boundary_and_malformed_rows_do_not_invent_recent_records() {
        let text = event("2026-01-10 11:00:00", "WARNING eventcode=12 CPU set")
            + &event("2026-01-10 10:59:59", "ERROR code=99")
            + "bad,row\n"
            + &event("not-a-date", "ERROR code=55");
        let result = parse_log_text(&text, 60, now());
        assert_eq!(result.total_events, 1);
        assert_eq!(result.cpu_set_events, 1);
        assert_eq!(result.event_codes.get(&12), Some(&1));
        assert_eq!(result.inferred_levels.get("warning"), Some(&1));
        assert_eq!(result.recent_events[0].timestamp, "2026-01-10 11:00:00");
        assert_eq!(result.recent_events[0].process, "example.exe");
        assert!(!result.retained_events_truncated);
    }

    #[test]
    fn raw_retention_is_bounded_without_changing_full_summary_counts() {
        let text = event("2026-01-10 12:00:00", &format!("ERROR code=7 {}", "x".repeat(5000))).repeat(130);
        let result = parse_log_text(&text, 60, now());
        assert_eq!(result.total_events, 130);
        assert_eq!(result.event_codes.get(&7), Some(&130));
        assert_eq!(result.recent_events.len(), 128);
        assert!(result.retained_events_truncated);
        assert!(result
            .recent_events
            .iter()
            .all(|v| v.raw.chars().count() == 4096 && v.raw_truncated));
    }

    #[test]
    fn diagnostic_labels_do_not_come_from_other_csv_columns_or_future_rows() {
        let text = "0,2026-01-10 12:00:00,0,0,0,error-code=99.exe,0,CPU set,warning code=23\n".to_string()
            + &event("2026-01-10 12:00:01", "ERROR code=55");
        let result = parse_log_text(&text, 60, now());
        assert_eq!(result.total_events, 1);
        assert!(result.event_codes.is_empty());
        assert_eq!(result.inferred_levels.get("info"), Some(&1));
    }
}
