use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::time::SystemTime;

use anyhow::{Context, Result};
use chrono::{DateTime, Duration, Local, NaiveDateTime, Utc};
use regex::Regex;
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct ProcessLassoSnapshot {
    pub config_path: String,
    pub log_path: String,
    pub config_exists: bool,
    pub log_exists: bool,
    pub config_modified_utc: Option<String>,
    pub log_modified_utc: Option<String>,
    pub settings: BTreeMap<String, String>,
    pub recent_events: Vec<ProcessLassoEvent>,
    pub recent_event_count: usize,
    pub event_codes: BTreeMap<String, usize>,
    pub levels: BTreeMap<String, usize>,
}

#[derive(Debug, Serialize)]
pub struct ProcessLassoEvent {
    pub timestamp_utc: Option<String>,
    pub code: Option<u32>,
    pub level: String,
    pub process: Option<String>,
    pub raw: String,
}

pub fn collect_snapshot(config_path: &str, log_path: &str, lookback_minutes: u32) -> Result<ProcessLassoSnapshot> {
    let config_file = Path::new(config_path);
    let log_file = Path::new(log_path);

    let settings = if config_file.exists() {
        parse_ini_settings(config_file)?
    } else {
        BTreeMap::new()
    };

    let recent_events = if log_file.exists() {
        parse_log_events(log_file, lookback_minutes)?
    } else {
        Vec::new()
    };

    let mut event_codes = BTreeMap::new();
    let mut levels = BTreeMap::new();
    for event in &recent_events {
        if let Some(code) = event.code {
            *event_codes.entry(code.to_string()).or_insert(0) += 1;
        }
        *levels.entry(event.level.clone()).or_insert(0) += 1;
    }

    Ok(ProcessLassoSnapshot {
        config_path: config_path.to_string(),
        log_path: log_path.to_string(),
        config_exists: config_file.exists(),
        log_exists: log_file.exists(),
        config_modified_utc: file_modified_utc(config_file),
        log_modified_utc: file_modified_utc(log_file),
        settings,
        recent_event_count: recent_events.len(),
        recent_events,
        event_codes,
        levels,
    })
}

fn parse_ini_settings(path: &Path) -> Result<BTreeMap<String, String>> {
    let text = read_text_file(path)?;
    let mut settings = BTreeMap::new();

    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with(';') || trimmed.starts_with('#') || trimmed.starts_with('[') {
            continue;
        }

        if let Some((key, value)) = trimmed.split_once('=') {
            settings.insert(key.trim().to_string(), value.trim().to_string());
        }
    }

    Ok(settings)
}

fn parse_log_events(path: &Path, lookback_minutes: u32) -> Result<Vec<ProcessLassoEvent>> {
    let text = read_text_file(path)?;
    let process_regex = Regex::new(r"(?i)\b([a-z0-9 _.\-]+\.exe)\b").context("compile process regex")?;
    let code_regex =
        Regex::new(r"\b(?:code|eventcode)\s*[:=]\s*(\d+)\b|\[(\d{1,4})\]").context("compile code regex")?;
    let threshold = Local::now() - Duration::minutes(i64::from(lookback_minutes));

    let mut events = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let timestamp = parse_timestamp(trimmed);
        if let Some(local_time) = timestamp {
            if local_time < threshold {
                continue;
            }
        }

        let code = code_regex.captures(trimmed).and_then(|caps| {
            caps.get(1)
                .or_else(|| caps.get(2))
                .and_then(|value| value.as_str().parse::<u32>().ok())
        });

        let process = process_regex
            .captures(trimmed)
            .and_then(|caps| caps.get(1).map(|value| value.as_str().trim().to_string()));

        let lowered = trimmed.to_ascii_lowercase();
        let level = if lowered.contains("error") {
            "error"
        } else if lowered.contains("warn") {
            "warning"
        } else if lowered.contains("start") || lowered.contains("begin") {
            "begin"
        } else if lowered.contains("end") || lowered.contains("stop") {
            "end"
        } else {
            "info"
        };

        events.push(ProcessLassoEvent {
            timestamp_utc: timestamp.map(|value| value.with_timezone(&Utc).to_rfc3339()),
            code,
            level: level.to_string(),
            process,
            raw: trimmed.to_string(),
        });
    }

    Ok(events)
}

fn parse_timestamp(line: &str) -> Option<DateTime<Local>> {
    let prefix = line.get(0..19)?;
    let naive = NaiveDateTime::parse_from_str(prefix, "%Y-%m-%d %H:%M:%S").ok()?;
    naive.and_local_timezone(Local).single()
}

fn file_modified_utc(path: &Path) -> Option<String> {
    let modified = fs::metadata(path).ok()?.modified().ok()?;
    let modified_utc: DateTime<Utc> = DateTime::<Utc>::from(system_time_or_now(modified));
    Some(modified_utc.to_rfc3339())
}

fn system_time_or_now(value: SystemTime) -> SystemTime {
    if value == SystemTime::UNIX_EPOCH {
        SystemTime::now()
    } else {
        value
    }
}

fn read_text_file(path: &Path) -> Result<String> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    if bytes.starts_with(&[0xFF, 0xFE]) {
        return Ok(decode_utf16(&bytes[2..], true));
    }
    if bytes.starts_with(&[0xFE, 0xFF]) {
        return Ok(decode_utf16(&bytes[2..], false));
    }

    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

fn decode_utf16(bytes: &[u8], little_endian: bool) -> String {
    let mut data = Vec::with_capacity(bytes.len() / 2);
    for chunk in bytes.chunks_exact(2) {
        let value = if little_endian {
            u16::from_le_bytes([chunk[0], chunk[1]])
        } else {
            u16::from_be_bytes([chunk[0], chunk[1]])
        };
        data.push(value);
    }

    String::from_utf16_lossy(&data)
}
