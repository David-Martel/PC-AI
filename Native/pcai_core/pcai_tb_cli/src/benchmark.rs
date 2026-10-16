use crate::handshake::discover_thunderbolt_peers;
use crate::identity::{local_hostname, resolve_peer_identity};
use crate::platform::collect_platform_snapshot;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::process::Command;
use std::time::Instant;

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
    pub speedup_vs_legacy_baseline_x: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct ArchitectureBenchmarkReport {
    pub computer_name: String,
    pub os_family: String,
    pub architecture: String,
    pub deterministic_identity_latency_us: u64,
    pub pipelines: Vec<PipelineProfileMetrics>,
}

fn compute_stats(mut samples_us: Vec<u64>) -> (u64, u64, u64, u64, u64) {
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

fn run_legacy_powershell_cim_probe() -> (u64, usize) {
    let start = Instant::now();
    #[cfg(target_os = "windows")]
    {
        let out = Command::new("pwsh")
            .args([
                "-NoProfile",
                "-NonInteractive",
                "-InputFormat",
                "None",
                "-Command",
                "$a = @(Get-NetAdapter -ErrorAction SilentlyContinue | Where-Object { $_.InterfaceDescription -match 'USB4|Thunderbolt' }); $n = @(Get-NetNeighbor -ErrorAction SilentlyContinue | Where-Object { $_.InterfaceIndex -in $a.ifIndex }); $a.Count + $n.Count",
            ])
            .output();
        let elapsed = start.elapsed().as_micros() as u64;
        if let Ok(o) = out {
            let count = String::from_utf8_lossy(&o.stdout)
                .trim()
                .lines()
                .last()
                .and_then(|s| s.trim().parse::<usize>().ok())
                .unwrap_or(0);
            return (elapsed, count);
        }
        (elapsed, 0)
    }
    #[cfg(not(target_os = "windows"))]
    {
        let out = Command::new("sh")
            .args(["-c", "ip -j addr show && ip -j neigh show"])
            .output();
        let elapsed = start.elapsed().as_micros() as u64;
        let count = out.map(|o| if o.status.success() { 1 } else { 0 }).unwrap_or(0);
        (elapsed, count)
    }
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
    let mut async_samples = Vec::with_capacity(iters.min(5));
    let mut async_items = 0usize;
    let async_iters = iters.min(5);
    for _ in 0..async_iters {
        let t0 = Instant::now();
        let peers = discover_thunderbolt_peers(false, timeout_ms, false, registry_path).await?;
        async_samples.push(t0.elapsed().as_micros() as u64);
        async_items = peers.len();
    }
    let (as_min, as_med, as_mean, as_p95, as_max) = compute_stats(async_samples);

    // 3. Pipeline C: LegacySubprocessBaseline (PowerShell CIM Get-NetAdapter + Get-NetNeighbor)
    let (leg_min, leg_med, leg_mean, leg_p95, leg_max, leg_items) = if include_legacy_baseline {
        let (dur, cnt) = run_legacy_powershell_cim_probe();
        (dur, dur, dur, dur, dur, cnt)
    } else {
        (0, 0, 0, 0, 0, 0)
    };

    let baseline_us = if leg_med > 0 { leg_med as f64 } else { 1.0 };
    let ffi_speedup = if leg_med > 0 && ffi_med > 0 {
        ((baseline_us / (ffi_med as f64)) * 10.0).round() / 10.0
    } else {
        1.0
    };
    let async_speedup = if leg_med > 0 && as_med > 0 {
        ((baseline_us / (as_med as f64)) * 10.0).round() / 10.0
    } else {
        1.0
    };

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
            speedup_vs_legacy_baseline_x: ffi_speedup,
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
            items_discovered: async_items,
            speedup_vs_legacy_baseline_x: async_speedup,
        },
    ];

    if include_legacy_baseline {
        pipelines.push(PipelineProfileMetrics {
            pipeline_name: "Pipeline3_LegacyPowerShellCimSubprocess".to_string(),
            description: "Legacy PowerShell WMI/CIM subprocess (Get-NetAdapter + Get-NetNeighbor)".to_string(),
            iterations: 1,
            min_latency_us: leg_min,
            median_latency_us: leg_med,
            mean_latency_us: leg_mean,
            p95_latency_us: leg_p95,
            max_latency_us: leg_max,
            items_discovered: leg_items,
            speedup_vs_legacy_baseline_x: 1.0,
        });
    }

    Ok(ArchitectureBenchmarkReport {
        computer_name: local_hostname(),
        os_family: std::env::consts::OS.to_string(),
        architecture: std::env::consts::ARCH.to_string(),
        deterministic_identity_latency_us: id_latency_us,
        pipelines,
    })
}
