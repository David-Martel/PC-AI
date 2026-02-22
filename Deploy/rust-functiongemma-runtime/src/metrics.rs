use crate::config::{default_model, router_engine, runtime_config};
use crate::types::{LoraMetadata, RouterMetadata, RouterStats, SystemMetrics, ToolsMetadata};
use serde_json::Value;
use std::fs;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;
use sysinfo::System;

#[cfg(feature = "model")]
use crate::gpu::device_label;
#[cfg(feature = "model")]
use crate::inference::lora_metadata;
use crate::types::RouterEngine;

pub(crate) static TOTAL_REQUESTS: AtomicU64 = AtomicU64::new(0);
pub(crate) static TOTAL_TOKENS: AtomicU64 = AtomicU64::new(0);
pub(crate) static TOTAL_INFER_MS: AtomicU64 = AtomicU64::new(0);

/// Cached tools metadata (loaded once from disk)
pub(crate) static TOOLS_METADATA_CACHE: OnceLock<ToolsMetadata> = OnceLock::new();

/// Cached system metrics with refresh interval
pub(crate) static SYSTEM_METRICS_CACHE: OnceLock<Mutex<(Instant, SystemMetrics)>> = OnceLock::new();

/// How often to refresh system metrics (seconds)
pub(crate) const METRICS_REFRESH_SECS: u64 = 5;

pub(crate) fn tools_metadata() -> ToolsMetadata {
    TOOLS_METADATA_CACHE
        .get_or_init(|| {
            let path = runtime_config().tools_path.clone();
            let mut count = 0usize;
            let mut loaded = false;
            if let Ok(contents) = fs::read_to_string(&path) {
                if let Ok(doc) = serde_json::from_str::<Value>(&contents) {
                    if let Some(arr) = doc.get("tools").and_then(|v| v.as_array()) {
                        count = arr.len();
                        loaded = true;
                    }
                }
            }
            ToolsMetadata {
                path,
                count,
                loaded,
            }
        })
        .clone()
}

pub(crate) fn router_metadata() -> RouterMetadata {
    let model = default_model();
    let engine = match router_engine() {
        RouterEngine::Heuristic => "heuristic",
        RouterEngine::Model => "model",
    };

    #[cfg(feature = "model")]
    let device = Some(device_label());
    #[cfg(not(feature = "model"))]
    let device: Option<String> = None;

    #[cfg(feature = "model")]
    let lora = lora_metadata();
    #[cfg(not(feature = "model"))]
    let lora: Option<LoraMetadata> = None;

    RouterMetadata {
        version: env!("CARGO_PKG_VERSION").to_string(),
        model,
        tools: tools_metadata(),
        engine: engine.to_string(),
        device,
        lora,
    }
}

pub(crate) fn collect_system_metrics() -> Option<SystemMetrics> {
    let mut sys = System::new_all();
    sys.refresh_all();
    let pid = sysinfo::get_current_pid().ok()?;
    let process = sys.process(pid)?;

    let process_rss_mb = process.memory() / 1024;
    let process_vmem_mb = process.virtual_memory() / 1024;
    let total_memory_mb = sys.total_memory() / 1024;
    let free_memory_mb = sys.free_memory() / 1024;
    let cpu_usage = process.cpu_usage();
    let uptime_sec = System::uptime();

    Some(SystemMetrics {
        process_rss_mb,
        process_vmem_mb,
        total_memory_mb,
        free_memory_mb,
        cpu_usage,
        uptime_sec,
    })
}

pub(crate) fn system_metrics() -> Option<SystemMetrics> {
    let cache = SYSTEM_METRICS_CACHE.get_or_init(|| {
        let metrics = collect_system_metrics().unwrap_or(SystemMetrics {
            process_rss_mb: 0,
            process_vmem_mb: 0,
            total_memory_mb: 0,
            free_memory_mb: 0,
            cpu_usage: 0.0,
            uptime_sec: 0,
        });
        Mutex::new((Instant::now(), metrics))
    });

    let mut guard = cache.lock().ok()?;
    if guard.0.elapsed().as_secs() >= METRICS_REFRESH_SECS {
        if let Some(fresh) = collect_system_metrics() {
            *guard = (Instant::now(), fresh);
        }
    }
    Some(guard.1.clone())
}

pub(crate) fn router_stats() -> Option<RouterStats> {
    let requests = TOTAL_REQUESTS.load(Ordering::Relaxed);
    let tokens = TOTAL_TOKENS.load(Ordering::Relaxed);
    let infer_ms = TOTAL_INFER_MS.load(Ordering::Relaxed);

    if requests == 0 {
        return Some(RouterStats {
            requests: 0,
            tokens_generated: 0,
            avg_tokens_per_sec: 0.0,
            avg_latency_ms: 0.0,
        });
    }

    let seconds = (infer_ms as f64) / 1000.0;
    let avg_tokens_per_sec = if seconds > 0.0 {
        tokens as f64 / seconds
    } else {
        0.0
    };
    let avg_latency_ms = (infer_ms as f64) / (requests as f64);

    Some(RouterStats {
        requests,
        tokens_generated: tokens,
        avg_tokens_per_sec,
        avg_latency_ms,
    })
}
