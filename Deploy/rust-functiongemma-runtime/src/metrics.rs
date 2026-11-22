use crate::config::{default_model, router_engine, runtime_config};
use crate::types::{LoraMetadata, RouterMetadata, RouterStats, SystemMetrics, ToolsMetadata};
use serde_json::Value;
use std::fs;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};
use sysinfo::{MemoryRefreshKind, ProcessRefreshKind, ProcessesToUpdate, System};

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

#[derive(Default)]
struct SystemMetricsCache {
    system: Option<System>,
    snapshot: Option<SystemMetrics>,
    cpu_sample: Option<CpuSample>,
    refreshed_at: Option<Instant>,
}

static SYSTEM_METRICS_CACHE: OnceLock<Mutex<SystemMetricsCache>> = OnceLock::new();

#[derive(Clone, Copy)]
struct CpuSample {
    at: Instant,
    millis: u64,
}

impl CpuSample {
    fn usage_since(self, previous: Self) -> Option<f32> {
        let interval = self.at.checked_duration_since(previous.at)?;
        if interval < sysinfo::MINIMUM_CPU_UPDATE_INTERVAL {
            return None;
        }
        let cpu_time = Duration::from_millis(self.millis.checked_sub(previous.millis)?);
        Some((100.0 * cpu_time.as_secs_f64() / interval.as_secs_f64()) as f32)
    }
}

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
            ToolsMetadata { path, count, loaded }
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

fn collect_system_metrics(sys: &mut System, previous_cpu: Option<CpuSample>) -> Option<(SystemMetrics, CpuSample)> {
    const BYTES_PER_MIB: u64 = 1024 * 1024;

    let pid = sysinfo::get_current_pid().ok()?;
    sys.refresh_memory_specifics(MemoryRefreshKind::nothing().with_ram());
    let updated = sys.refresh_processes_specifics(
        ProcessesToUpdate::Some(&[pid]),
        true,
        ProcessRefreshKind::nothing().with_memory().with_cpu().without_tasks(),
    );
    if updated == 0 || sys.total_memory() == 0 {
        return None;
    }
    let process = sys.process(pid)?;

    let cpu = CpuSample {
        at: Instant::now(),
        millis: process.accumulated_cpu_time(),
    };
    let snapshot = SystemMetrics {
        process_rss_mb: process.memory() / BYTES_PER_MIB,
        process_vmem_mb: process.virtual_memory() / BYTES_PER_MIB,
        total_memory_mb: sys.total_memory() / BYTES_PER_MIB,
        free_memory_mb: sys.free_memory() / BYTES_PER_MIB,
        cpu_usage: previous_cpu.and_then(|previous| cpu.usage_since(previous)),
        uptime_sec: System::uptime().unwrap_or(0),
    };
    Some((snapshot, cpu))
}

pub(crate) fn system_metrics() -> Option<SystemMetrics> {
    let cache = SYSTEM_METRICS_CACHE.get_or_init(Mutex::default);
    let mut guard = cache.lock().ok()?;
    if guard
        .refreshed_at
        .is_none_or(|refreshed| refreshed.elapsed().as_secs() >= METRICS_REFRESH_SECS)
    {
        if guard.system.is_none() {
            guard.system = System::new().ok();
        }
        let previous_cpu = guard.cpu_sample;
        if let Some((fresh, cpu)) = guard
            .system
            .as_mut()
            .and_then(|system| collect_system_metrics(system, previous_cpu))
        {
            guard.snapshot = Some(fresh);
            guard.cpu_sample = Some(cpu);
        }
        guard.refreshed_at = Some(Instant::now());
    }
    guard.snapshot.clone()
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
    let avg_tokens_per_sec = if seconds > 0.0 { tokens as f64 / seconds } else { 0.0 };
    let avg_latency_ms = (infer_ms as f64) / (requests as f64);

    Some(RouterStats {
        requests,
        tokens_generated: tokens,
        avg_tokens_per_sec,
        avg_latency_ms,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Protects: scoped and correctly sized health telemetry. Detects: scanning
    // unrelated processes, reporting KiB as MiB, or inventing a cold CPU delta.
    // Needs: current-process and RAM counters. Breadcrumb: collect_system_metrics.
    #[test]
    fn cold_health_samples_one_process_and_reports_mib() -> anyhow::Result<()> {
        let mut system = System::new()?;
        let (metrics, _) = collect_system_metrics(&mut system, None)
            .ok_or_else(|| anyhow::anyhow!("the test process should be available"))?;
        let pid = sysinfo::get_current_pid().map_err(anyhow::Error::msg)?;
        let process = system
            .process(pid)
            .ok_or_else(|| anyhow::anyhow!("no current-process sample"))?;

        assert_eq!(system.processes().len(), 1);
        assert!(system.total_memory() > 0);
        assert_eq!(metrics.total_memory_mb, system.total_memory() >> 20);
        assert_eq!(metrics.free_memory_mb, system.free_memory() >> 20);
        assert_eq!(metrics.process_rss_mb, process.memory() >> 20);
        assert_eq!(metrics.process_vmem_mb, process.virtual_memory() >> 20);
        assert_eq!(metrics.cpu_usage, None);
        Ok(())
    }

    // Protects: interval CPU percentage. Detects: normalizing to one core,
    // measuring instantaneously, or wrapping a regressed CPU counter.
    // Needs: synthetic timestamps. Breadcrumb: CpuSample::usage_since.
    #[test]
    fn health_cpu_usage_requires_an_interval_and_allows_multiple_cores() {
        let start = CpuSample {
            at: Instant::now(),
            millis: 100,
        };
        let refreshed = CpuSample {
            at: start.at + Duration::from_secs(5),
            millis: 7_600,
        };
        let regressed = CpuSample {
            millis: 50,
            ..refreshed
        };

        assert_eq!(refreshed.usage_since(start), Some(150.0));
        assert_eq!(start.usage_since(start), None);
        assert_eq!(regressed.usage_since(start), None);
    }
}
