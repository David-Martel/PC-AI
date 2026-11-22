use crate::handshake::discover_thunderbolt_peers;
use crate::identity::{local_hostname, resolve_peer_identity};
use crate::platform::collect_platform_snapshot;
#[cfg(target_os = "windows")]
use crate::switch::run_bounded_command_with_deadline;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Instant;

const MAX_DISCOVERY_PROFILE_SAMPLES: usize = 5;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct PipelineProfileMetrics {
    pub pipeline_name: String,
    pub description: String,
    pub iterations: usize,
    pub min_latency_us: u64,
    pub median_latency_us: u64,
    pub mean_latency_us: u64,
    pub p95_latency_us: u64,
    pub max_latency_us: u64,
    pub items_discovered: usize,
    pub speedup_vs_legacy_baseline_x: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct ArchitectureBenchmarkReport {
    pub computer_name: String,
    pub os_family: String,
    pub architecture: String,
    pub deterministic_identity_latency_us: u64,
    #[serde(default)]
    pub legacy_baseline_status: LegacyBaselineStatus,
    #[serde(default)]
    pub legacy_baseline_attempts: usize,
    pub pipelines: Vec<PipelineProfileMetrics>,
}

pub fn compute_stats(mut samples_us: Vec<u64>) -> (u64, u64, u64, u64, u64) {
    if samples_us.is_empty() {
        return (0, 0, 0, 0, 0);
    }
    samples_us.sort_unstable();
    let min_v = samples_us[0];
    let max_v = *samples_us.last().unwrap_or(&0);
    let sum: u128 = samples_us.iter().map(|&x| u128::from(x)).sum();
    let mean_v = (sum / (samples_us.len() as u128)) as u64;
    let mid = samples_us.len() / 2;
    let median_v = samples_us[mid];
    let p95_idx = ((samples_us.len() as f64) * 0.95).floor() as usize;
    let p95_v = samples_us[p95_idx.min(samples_us.len() - 1)];
    (min_v, median_v, mean_v, p95_v, max_v)
}

#[cfg(any(target_os = "windows", test))]
fn parse_legacy_count(exit_code: Option<i32>, stdout: &str) -> Option<usize> {
    if exit_code != Some(0) {
        return None;
    }
    stdout.trim().parse().ok()
}

/// Availability of the requested PowerShell/CIM timings.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum LegacyBaselineStatus {
    /// An older report does not contain sample-availability evidence.
    #[default]
    Unknown,
    NotRequested,
    Unsupported,
    Failed,
    Partial,
    Measured,
}

fn collect_legacy_baseline(
    requested: bool,
    supported: bool,
    iterations: usize,
    mut probe: impl FnMut() -> Option<(u64, usize)>,
) -> (LegacyBaselineStatus, usize, Vec<u64>, usize) {
    if !requested {
        return (LegacyBaselineStatus::NotRequested, 0, vec![], 0);
    }
    if !supported {
        return (LegacyBaselineStatus::Unsupported, 0, vec![], 0);
    }
    let attempts = iterations.min(MAX_DISCOVERY_PROFILE_SAMPLES);
    let mut samples = Vec::with_capacity(attempts);
    let mut items = 0;
    for (elapsed, count) in (0..attempts).filter_map(|_| probe()) {
        samples.push(elapsed);
        items = count;
    }
    let status = if samples.is_empty() {
        LegacyBaselineStatus::Failed
    } else if samples.len() == attempts {
        LegacyBaselineStatus::Measured
    } else {
        LegacyBaselineStatus::Partial
    };
    (status, attempts, samples, items)
}

#[cfg(target_os = "windows")]
fn run_legacy_powershell_cim_probe() -> Option<(u64, usize)> {
    const CHILD_TIMEOUT_MS: u64 = 10_000;
    const CHILD_CAPTURE_LIMIT: usize = 4096;
    let start = Instant::now();
    let (exit_code, stdout, _) = run_bounded_command_with_deadline(
        Path::new("pwsh"),
        &[
            "-NoProfile",
            "-NonInteractive",
            "-InputFormat",
            "None",
            "-Command",
            "$ErrorActionPreference = 'Stop'; \
             $a = @(Get-NetAdapter | Where-Object { $_.InterfaceDescription -match 'USB4|Thunderbolt' }); \
             $n = @(if ($a.Count -gt 0) { Get-NetNeighbor | Where-Object { $_.InterfaceIndex -in $a.ifIndex } }); \
             $a.Count + $n.Count",
        ],
        CHILD_TIMEOUT_MS,
        CHILD_CAPTURE_LIMIT,
    )
    .ok()?;
    let count = parse_legacy_count(exit_code, &stdout)?;
    Some((start.elapsed().as_micros() as u64, count))
}

#[cfg(not(target_os = "windows"))]
fn run_legacy_powershell_cim_probe() -> Option<(u64, usize)> {
    None
}

pub async fn run_architecture_benchmarks(
    iterations: usize,
    timeout_ms: u64,
    include_legacy_baseline: bool,
    registry_path: Option<&Path>,
) -> Result<ArchitectureBenchmarkReport> {
    let iters = iterations.max(1);

    // 0. Deterministic SHA-256 identity resolution microbenchmark
    let id_start = Instant::now();
    for _ in 0..100 {
        let _ = resolve_peer_identity(Some("dtm-p1gen7"), Some("dtm-p1gen9"), registry_path)?;
    }
    let id_latency_us = (id_start.elapsed().as_micros() as u64) / 100;

    // 1. Pipeline A: NativeKernelFfi (GetAdaptersAddresses + GetIpNetTable2 + ConfigMgr32 / getifaddrs)
    let mut ffi_samples = Vec::with_capacity(iters);
    let mut ffi_items = 0usize;
    for _ in 0..iters {
        let t0 = Instant::now();
        let snap = collect_platform_snapshot(true)?;
        ffi_samples.push(t0.elapsed().as_micros() as u64);
        ffi_items = snap.adapters.len() + snap.usb4_devices.len();
    }
    let (ffi_min, ffi_med, ffi_mean, ffi_p95, ffi_max) = compute_stats(ffi_samples);

    // 2. Pipeline B: AsyncMultiplexedHandshake (Kernel FFI + Tokio JoinSet TCP + NTLM AV_PAIR + SSH Banner)
    let async_iters = iters.min(MAX_DISCOVERY_PROFILE_SAMPLES);
    let mut async_samples = Vec::with_capacity(async_iters);
    let mut async_verified_items = 0usize;
    for _ in 0..async_iters {
        let t0 = Instant::now();
        let peers = discover_thunderbolt_peers(false, timeout_ms, false, registry_path).await?;
        async_samples.push(t0.elapsed().as_micros() as u64);
        async_verified_items = peers.iter().filter(|p| p.verified_handshake).count();
    }
    let (as_min, as_med, as_mean, as_p95, as_max) = compute_stats(async_samples);

    // 3. Pipeline C: LegacySubprocessBaseline (PowerShell CIM Get-NetAdapter + Get-NetNeighbor)
    let (legacy_baseline_status, legacy_baseline_attempts, legacy_samples, leg_items) = collect_legacy_baseline(
        include_legacy_baseline,
        cfg!(windows),
        iters,
        run_legacy_powershell_cim_probe,
    );
    let legacy_iterations = legacy_samples.len();
    let (leg_min, leg_med, leg_mean, leg_p95, leg_max) = compute_stats(legacy_samples);

    // Process startup, inventory and network verification profile different workloads.
    let mut pipelines = vec![
        PipelineProfileMetrics {
            pipeline_name: "Pipeline1_NativeKernelFfi".to_string(),
            description: "In-process Win32 IpHelper (GetAdaptersAddresses + GetIpNetTable2 + GetIpInterfaceEntry) + CfgMgr32 / Linux getifaddrs + sysfs".to_string(),
            iterations: iters,
            min_latency_us: ffi_min,
            median_latency_us: ffi_med,
            mean_latency_us: ffi_mean,
            p95_latency_us: ffi_p95,
            max_latency_us: ffi_max,
            items_discovered: ffi_items,
            speedup_vs_legacy_baseline_x: None,
        },
        PipelineProfileMetrics {
            pipeline_name: "Pipeline2_AsyncMultiplexedHandshake".to_string(),
            description: "NativeKernelFfi + Tokio concurrent TCP socket pre-screen + binary NTLM Type-2 AV_PAIR TargetInfo decoder + SSH banner inspection".to_string(),
            iterations: async_iters,
            min_latency_us: as_min,
            median_latency_us: as_med,
            mean_latency_us: as_mean,
            p95_latency_us: as_p95,
            max_latency_us: as_max,
            items_discovered: async_verified_items,
            speedup_vs_legacy_baseline_x: None,
        },
    ];

    if include_legacy_baseline {
        pipelines.push(PipelineProfileMetrics {
            pipeline_name: "Pipeline3_LegacyPowerShellCimSubprocess".to_string(),
            description: "Legacy PowerShell WMI/CIM subprocess (Get-NetAdapter + Get-NetNeighbor)".to_string(),
            iterations: legacy_iterations,
            min_latency_us: leg_min,
            median_latency_us: leg_med,
            mean_latency_us: leg_mean,
            p95_latency_us: leg_p95,
            max_latency_us: leg_max,
            items_discovered: leg_items,
            speedup_vs_legacy_baseline_x: None,
        });
    }

    Ok(ArchitectureBenchmarkReport {
        computer_name: local_hostname(),
        os_family: std::env::consts::OS.to_string(),
        architecture: std::env::consts::ARCH.to_string(),
        deterministic_identity_latency_us: id_latency_us,
        legacy_baseline_status,
        legacy_baseline_attempts,
        pipelines,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_stats_handles_empty_and_populated_samples() {
        assert_eq!(compute_stats(vec![]), (0, 0, 0, 0, 0));
        let (min_v, med_v, mean_v, p95_v, max_v) = compute_stats(vec![500, 100, 300, 200, 400]);
        assert_eq!(min_v, 100);
        assert_eq!(med_v, 300);
        assert_eq!(mean_v, 300);
        assert_eq!(p95_v, 500);
        assert_eq!(max_v, 500);
    }

    // Protects: legacy discovery records repeated elapsed times on the same capped iteration budget as the handshake.
    // Detects: a one-shot legacy sample, excess child launches, or a failed sample counted as latency.
    // Needs: scripted child observations only. Breadcrumb: benchmark.rs::collect_legacy_baseline.
    #[test]
    fn legacy_samples_respect_handshake_cap_and_exclude_failures() {
        let mut input = [Some((30, 1)), None, Some((10, 2)), Some((20, 3)), None].into_iter();
        let (status, attempted, samples, items) =
            collect_legacy_baseline(true, true, 20, || input.next().expect("more than five child launches"));
        assert_eq!(attempted, 5);
        assert_eq!(status, LegacyBaselineStatus::Partial);
        assert_eq!(compute_stats(samples), (10, 20, 20, 30, 30));
        assert_eq!(items, 3);
        assert_eq!(input.len(), 0);
    }

    // Protects: user opt-out, an unavailable backend, and a requested backend failure remain distinguishable.
    // Detects: Linux substitute commands or failed child launches being reported as successful timings.
    // Needs: simulated platform eligibility and probe outcomes. Breadcrumb: benchmark.rs::collect_legacy_baseline.
    #[test]
    fn legacy_baseline_reports_skipped_unsupported_and_failed_distinctly() {
        assert_eq!(
            collect_legacy_baseline(false, true, 3, || panic!("skipped probe was run")),
            (LegacyBaselineStatus::NotRequested, 0, vec![], 0)
        );
        assert_eq!(
            collect_legacy_baseline(true, false, 3, || panic!("unsupported probe was run")),
            (LegacyBaselineStatus::Unsupported, 0, vec![], 0)
        );
        assert_eq!(
            collect_legacy_baseline(true, true, 3, || None),
            (LegacyBaselineStatus::Failed, 3, vec![], 0)
        );
        assert_eq!(
            collect_legacy_baseline(true, true, 2, || Some((11, 0))),
            (LegacyBaselineStatus::Measured, 2, vec![11, 11], 0)
        );
    }

    // Protects: zero discovered items is a valid successful sample.
    // Detects: missing, malformed, or nonzero-exit output relabeled as a zero-item measurement.
    // Needs: simulated exit status and captured text. Breadcrumb: benchmark.rs::parse_legacy_count.
    #[test]
    fn legacy_probe_accepts_zero_count_only_on_clean_exit() {
        assert_eq!(parse_legacy_count(Some(0), " 0\r\n"), Some(0));
        assert_eq!(parse_legacy_count(Some(0), "42\n"), Some(42));
        assert_eq!(parse_legacy_count(Some(1), "42\n"), None);
        assert_eq!(parse_legacy_count(None, "0\n"), None);
        assert_eq!(parse_legacy_count(Some(0), "unexpected\n0"), None);
        assert_eq!(parse_legacy_count(Some(0), "120f"), None);
    }
}
