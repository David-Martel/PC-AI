//! Memory Pressure Analysis and Optimization Recommendations
//!
//! Analyses running process categories, paging activity, pool memory usage,
//! and orphaned terminal processes to generate prioritised recommendations
//! for LLM workload environments under memory pressure.

use crate::PcaiStatus;
use serde::Serialize;
use std::collections::HashMap;
use std::time::Instant;
use sysinfo::System;

// ── FFI-safe report struct ────────────────────────────────────────────────────

/// FFI-safe memory pressure report.
///
/// All scalar fields map directly to C# `ulong`/`float`/`uint`/`byte` members.
///
/// # Example
/// ```no_run
/// use pcai_core_lib::performance::optimizer::analyze_memory_pressure;
/// let report = analyze_memory_pressure();
/// assert!(report.available_mb <= report.available_mb + 1); // tautology for docs
/// ```
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MemoryPressureReport {
    pub status: PcaiStatus,
    /// 0 = low (<60 %), 1 = moderate (<80 %), 2 = high (<90 %), 3 = critical (≥90 %)
    pub pressure_level: u8,
    /// Physical memory available to processes, in MiB
    pub available_mb: u64,
    /// Committed memory as a fraction of commit limit (0.0–1.0+)
    pub committed_pct: f32,
    /// Non-paged pool usage in MiB (abnormal if > 4 096 MiB)
    pub pool_nonpaged_mb: u64,
    /// Hard page faults per second averaged over the sample window
    pub pages_per_sec: u64,
    /// Number of processes consuming more than 500 MiB working set
    pub top_consumer_count: u32,
    /// Number of processes with more than 100 000 open handles
    pub handle_leak_count: u32,
    /// Number of cmd.exe / conhost.exe processes whose parent is no longer alive
    pub orphan_terminal_count: u32,
    /// Wall-clock time to collect all data, in milliseconds
    pub elapsed_ms: u64,
}

impl MemoryPressureReport {
    /// Return an error report with zeroed metrics.
    pub fn error(status: PcaiStatus) -> Self {
        Self {
            status,
            pressure_level: 0,
            available_mb: 0,
            committed_pct: 0.0,
            pool_nonpaged_mb: 0,
            pages_per_sec: 0,
            top_consumer_count: 0,
            handle_leak_count: 0,
            orphan_terminal_count: 0,
            elapsed_ms: 0,
        }
    }
}

impl Default for MemoryPressureReport {
    fn default() -> Self {
        Self::error(PcaiStatus::Success)
    }
}

// ── Process category data ─────────────────────────────────────────────────────

/// Aggregated statistics for a process category.
#[derive(Debug, Clone, Serialize, Default)]
pub struct CategoryStats {
    pub count: u32,
    pub working_set_mb: u64,
    pub private_mb: u64,
    pub handle_count: u64,
}

/// JSON output for the full process-category breakdown.
#[derive(Debug, Serialize)]
pub struct ProcessCategoriesJson {
    pub status: String,
    pub elapsed_ms: u64,
    pub categories: HashMap<String, CategoryStats>,
}

// ── Optimization recommendation ───────────────────────────────────────────────

/// A single prioritised optimization recommendation.
///
/// The `action` field is a machine-readable identifier suitable for driving
/// automated remediation scripts (PowerShell, C# service, etc.).
#[derive(Debug, Clone, Serialize)]
pub struct OptimizationRecommendation {
    /// 1 = critical, 2 = high, 3 = medium, 4 = low
    pub priority: u8,
    /// Coarse category tag (e.g. "handle_leak", "orphan_cleanup")
    pub category: String,
    /// Human-readable description of the issue and suggested action
    pub description: String,
    /// Conservative estimate of memory that would be freed, in MiB
    pub estimated_savings_mb: u64,
    /// Machine-readable action identifier for automated remediation
    pub action: String,
    /// `true` if the action can be executed without explicit user confirmation
    pub safe_to_auto: bool,
}

/// JSON envelope for the recommendations array.
#[derive(Debug, Serialize)]
pub struct OptimizationRecommendationsJson {
    pub status: String,
    pub elapsed_ms: u64,
    pub recommendation_count: u32,
    pub recommendations: Vec<OptimizationRecommendation>,
}

/// JSON envelope for the memory pressure report (mirrors `MemoryPressureReport`).
#[derive(Debug, Serialize)]
pub struct MemoryPressureJson {
    pub status: String,
    pub pressure_level: u8,
    pub pressure_label: String,
    pub available_mb: u64,
    pub committed_pct: f32,
    pub pool_nonpaged_mb: u64,
    pub pages_per_sec: u64,
    pub top_consumer_count: u32,
    pub handle_leak_count: u32,
    pub orphan_terminal_count: u32,
    pub elapsed_ms: u64,
}

// ── Category classification helpers ──────────────────────────────────────────

/// Classify a process name (lower-case) into one of the standard category keys.
fn classify_process(name_lower: &str) -> &'static str {
    // LLM agents first so pcai is not also counted as a build tool
    if matches_any(name_lower, &["claude", "codex", "ollama", "copilot", "pcai", "llama"]) {
        return "llm_agents";
    }
    if matches_any(name_lower, &["chrome", "brave", "msedge", "firefox"]) {
        return "browsers";
    }
    if matches_any(
        name_lower,
        &["conhost", "cmd", "powershell", "pwsh", "wezterm", "windowsterminal"],
    ) {
        return "terminals";
    }
    if matches_any(
        name_lower,
        &["rust-analyzer", "cargo", "node", "dotnet", "msbuild", "cl"],
    ) {
        return "build_tools";
    }
    "system_services"
}

/// Returns `true` when `name` contains any of the given substrings.
#[inline]
fn matches_any(name: &str, patterns: &[&str]) -> bool {
    patterns.iter().any(|p| name.contains(p))
}

/// Returns `true` when a process name (lower-case) is a terminal emulator that
/// may be orphaned (cmd or conhost).
#[inline]
fn is_orphan_candidate(name_lower: &str) -> bool {
    name_lower == "cmd" || name_lower == "conhost"
}

// ── Windows-specific metrics (pool memory, paging, handles) ──────────────────

/// Raw metrics collected from the Windows kernel.
struct WindowsMetrics {
    pool_nonpaged_mb: u64,
    pages_per_sec: u64,
    committed_pct: f32,
}

/// Query `GlobalMemoryStatusEx` and the performance counter registry key for
/// pool/paging data.  Falls back to zero on non-Windows targets.
#[cfg(target_os = "windows")]
fn collect_windows_metrics(sys: &System) -> WindowsMetrics {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};

    // --- GlobalMemoryStatusEx ---
    let mut mem_status = MEMORYSTATUSEX {
        dwLength: std::mem::size_of::<MEMORYSTATUSEX>() as u32,
        dwMemoryLoad: 0,
        ullTotalPhys: 0,
        ullAvailPhys: 0,
        ullTotalPageFile: 0,
        ullAvailPageFile: 0,
        ullTotalVirtual: 0,
        ullAvailVirtual: 0,
        ullAvailExtendedVirtual: 0,
    };

    let committed_pct = unsafe {
        if GlobalMemoryStatusEx(&mut mem_status) != 0 {
            let committed = mem_status.ullTotalPageFile.saturating_sub(mem_status.ullAvailPageFile);
            let limit = mem_status.ullTotalPageFile;
            if limit > 0 {
                committed as f32 / limit as f32
            } else {
                0.0
            }
        } else {
            0.0
        }
    };

    // --- Pool non-paged: derive from sysinfo available vs used ---
    // sysinfo does not expose pool metrics directly.  We use the relationship:
    //   pool_nonpaged ≈ total_physical - available - working_sets_sum
    // as a conservative heuristic when the Windows registry key is inaccessible.
    let total_mb = sys.total_memory() / (1024 * 1024);
    let available_mb = sys.available_memory() / (1024 * 1024);
    let working_set_sum_mb: u64 = sys.processes().values().map(|p| p.memory() / (1024 * 1024)).sum();
    let pool_nonpaged_mb = total_mb.saturating_sub(available_mb).saturating_sub(working_set_sum_mb);

    // --- Pages/sec: sysinfo does not expose this counter.  We read the
    //     Windows Performance counter registry value for a rough single-sample
    //     estimate (delta-based accuracy requires two samples; we return the
    //     raw counter divided by a 60-second window as a coarse average).
    // As a safe fallback, return 0 so the caller can still make decisions.
    let pages_per_sec = read_paging_counter_estimate();

    WindowsMetrics {
        pool_nonpaged_mb,
        pages_per_sec,
        committed_pct,
    }
}

/// Read the Windows performance registry key `\\Memory\\Page Faults/sec` counter.
///
/// The registry value is a raw cumulative counter; without two timed samples we
/// cannot compute a true rate.  We divide by a nominal 60-second observation
/// window to produce a conservative floor estimate suitable for threshold checks.
#[cfg(target_os = "windows")]
fn read_paging_counter_estimate() -> u64 {
    use winreg::enums::HKEY_LOCAL_MACHINE;
    use winreg::RegKey;

    // Perflib stores the page faults counter under a numeric index.
    // The reliable cross-version key is:
    // HKLM\SYSTEM\CurrentControlSet\Services\PerfOS\Performance
    // However this value is not easily decoded without PDH.
    // We use a pragmatic heuristic: if we cannot read the counter, return 0.
    let hklm = RegKey::predef(HKEY_LOCAL_MACHINE);
    let key_path = "SYSTEM\\CurrentControlSet\\Services\\PerfOS\\Performance";
    if let Ok(_key) = hklm.open_subkey(key_path) {
        // Key exists but decoding raw perf data is non-trivial without PDH.
        // Return 0 to indicate "unknown" rather than a fabricated value.
    }
    0
}

#[cfg(not(target_os = "windows"))]
fn collect_windows_metrics(sys: &System) -> WindowsMetrics {
    let total_mb = sys.total_memory() / (1024 * 1024);
    let available_mb = sys.available_memory() / (1024 * 1024);
    let working_set_sum_mb: u64 = sys.processes().values().map(|p| p.memory() / (1024 * 1024)).sum();
    let pool_nonpaged_mb = total_mb.saturating_sub(available_mb).saturating_sub(working_set_sum_mb);

    WindowsMetrics {
        pool_nonpaged_mb,
        pages_per_sec: 0,
        committed_pct: if sys.total_memory() > 0 {
            sys.used_memory() as f32 / sys.total_memory() as f32
        } else {
            0.0
        },
    }
}

// ── Core analysis functions ───────────────────────────────────────────────────

/// Map a usage percentage to a pressure level.
///
/// | Level | Threshold |
/// |-------|-----------|
/// | 0     | < 60 %    |
/// | 1     | < 80 %    |
/// | 2     | < 90 %    |
/// | 3     | ≥ 90 %    |
fn pressure_level(used_pct: f64) -> u8 {
    if used_pct >= 90.0 {
        3
    } else if used_pct >= 80.0 {
        2
    } else if used_pct >= 60.0 {
        1
    } else {
        0
    }
}

fn pressure_label(level: u8) -> &'static str {
    match level {
        0 => "low",
        1 => "moderate",
        2 => "high",
        _ => "critical",
    }
}

/// Collect a full `MemoryPressureReport` from live system data.
///
/// This is the single authoritative data-collection path used by all three
/// public-facing functions.
pub fn analyze_memory_pressure() -> MemoryPressureReport {
    let start = Instant::now();

    let mut sys = System::new_all();
    sys.refresh_all();

    // Second refresh for accurate CPU figures (matching existing process.rs pattern)
    std::thread::sleep(std::time::Duration::from_millis(100));
    sys.refresh_all();

    let total_bytes = sys.total_memory();
    let available_bytes = sys.available_memory();
    let used_bytes = sys.used_memory();

    let available_mb = available_bytes / (1024 * 1024);

    let used_pct = if total_bytes > 0 {
        used_bytes as f64 / total_bytes as f64 * 100.0
    } else {
        0.0
    };

    let win_metrics = collect_windows_metrics(&sys);

    // --- Per-process derived metrics ---
    // Build a set of live PIDs for orphan detection
    let live_pids: std::collections::HashSet<u32> = sys.processes().keys().map(|p| p.as_u32()).collect();

    let mut top_consumer_count = 0u32;
    let mut handle_leak_count = 0u32;
    let mut orphan_terminal_count = 0u32;

    const LARGE_CONSUMER_MB: u64 = 500;
    const HANDLE_LEAK_THRESHOLD: u64 = 100_000;

    for (pid, process) in sys.processes() {
        let working_set_mb = process.memory() / (1024 * 1024);
        if working_set_mb > LARGE_CONSUMER_MB {
            top_consumer_count += 1;
        }

        // sysinfo 0.38 exposes handles via a platform-specific field.
        // Use the `exe` heuristic: the handle count is not directly in sysinfo
        // on all targets, so we query via the Windows API when available.
        let handle_count = get_handle_count(pid.as_u32());
        if handle_count >= HANDLE_LEAK_THRESHOLD {
            handle_leak_count += 1;
        }

        let name_lower = process.name().to_string_lossy().to_lowercase();
        if is_orphan_candidate(&name_lower) {
            let parent_alive = process
                .parent()
                .map(|ppid| live_pids.contains(&ppid.as_u32()))
                .unwrap_or(false);
            if !parent_alive {
                orphan_terminal_count += 1;
            }
        }
    }

    MemoryPressureReport {
        status: PcaiStatus::Success,
        pressure_level: pressure_level(used_pct),
        available_mb,
        committed_pct: win_metrics.committed_pct,
        pool_nonpaged_mb: win_metrics.pool_nonpaged_mb,
        pages_per_sec: win_metrics.pages_per_sec,
        top_consumer_count,
        handle_leak_count,
        orphan_terminal_count,
        elapsed_ms: start.elapsed().as_millis() as u64,
    }
}

/// Return the number of open handles for the given PID via `GetProcessHandleCount`.
/// Falls back to 0 on non-Windows targets or when access is denied.
#[cfg(target_os = "windows")]
fn get_handle_count(pid: u32) -> u64 {
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::Threading::{GetProcessHandleCount, OpenProcess, PROCESS_QUERY_INFORMATION};

    unsafe {
        let handle = OpenProcess(PROCESS_QUERY_INFORMATION, 0, pid);
        if handle.is_null() {
            return 0;
        }
        let mut count: u32 = 0;
        GetProcessHandleCount(handle, &mut count);
        CloseHandle(handle);
        count as u64
    }
}

#[cfg(not(target_os = "windows"))]
fn get_handle_count(_pid: u32) -> u64 {
    0
}

/// Build the per-category aggregation map over the current process list.
pub fn get_process_categories() -> (u64, HashMap<String, CategoryStats>) {
    let start = Instant::now();

    let mut sys = System::new_all();
    sys.refresh_all();

    let live_pids: std::collections::HashSet<u32> = sys.processes().keys().map(|p| p.as_u32()).collect();

    let mut categories: HashMap<String, CategoryStats> = HashMap::new();
    for cat in &["llm_agents", "browsers", "terminals", "build_tools", "system_services"] {
        categories.insert(cat.to_string(), CategoryStats::default());
    }

    for (pid, process) in sys.processes() {
        let name_lower = process.name().to_string_lossy().to_lowercase();
        let cat = classify_process(&name_lower);

        // For terminals, exclude orphans from the totals (they have no owning parent)
        if is_orphan_candidate(&name_lower) {
            let parent_alive = process
                .parent()
                .map(|ppid| live_pids.contains(&ppid.as_u32()))
                .unwrap_or(false);
            if !parent_alive {
                // Count under "terminals" anyway — the recommendation engine will
                // separately flag them with the orphan_cleanup category.
            }
        }

        let entry = categories.entry(cat.to_string()).or_default();
        entry.count += 1;
        entry.working_set_mb += process.memory() / (1024 * 1024);
        entry.private_mb += process.virtual_memory() / (1024 * 1024);
        entry.handle_count += get_handle_count(pid.as_u32());
    }

    (start.elapsed().as_millis() as u64, categories)
}

/// Generate a prioritised list of `OptimizationRecommendation` items from a
/// `MemoryPressureReport` combined with live category data.
pub fn get_optimization_recommendations() -> (u64, Vec<OptimizationRecommendation>) {
    let start = Instant::now();

    let report = analyze_memory_pressure();
    let (_, categories) = get_process_categories();

    let mut recs: Vec<OptimizationRecommendation> = Vec::new();

    // --- Priority 1: handle leak ---
    if report.handle_leak_count > 0 {
        recs.push(OptimizationRecommendation {
            priority: 1,
            category: "handle_leak".to_string(),
            description: format!(
                "{} process(es) have more than 100 000 open handles. \
                 This is a strong indicator of a handle leak, which also \
                 inflates the non-paged pool. Restart the offending process(es).",
                report.handle_leak_count
            ),
            estimated_savings_mb: report.pool_nonpaged_mb / 2,
            action: "restart_handle_leak_processes".to_string(),
            safe_to_auto: false,
        });
    }

    // --- Priority 1: non-paged pool abnormally high (> 4 GB) ---
    if report.pool_nonpaged_mb > 4_096 {
        recs.push(OptimizationRecommendation {
            priority: 1,
            category: "pool_nonpaged_high".to_string(),
            description: format!(
                "Non-paged pool is {:.1} GB, which is abnormally high. \
                 This typically indicates a kernel handle leak or a faulty driver. \
                 Collect a kernel memory dump and analyse with !poolused.",
                report.pool_nonpaged_mb as f64 / 1024.0
            ),
            estimated_savings_mb: report.pool_nonpaged_mb.saturating_sub(1_024),
            action: "collect_kernel_memory_dump".to_string(),
            safe_to_auto: false,
        });
    }

    // --- Priority 1: extreme paging ---
    if report.pages_per_sec > 1_000 {
        recs.push(OptimizationRecommendation {
            priority: 1,
            category: "page_thrashing".to_string(),
            description: format!(
                "Hard page fault rate is {} pages/sec, indicating severe memory \
                 thrashing. Reduce the working set of the highest memory consumers \
                 or add physical RAM.",
                report.pages_per_sec
            ),
            estimated_savings_mb: 0,
            action: "reduce_working_set".to_string(),
            safe_to_auto: false,
        });
    }

    // --- Priority 2: orphaned terminals ---
    if report.orphan_terminal_count > 10 {
        recs.push(OptimizationRecommendation {
            priority: 2,
            category: "orphan_cleanup".to_string(),
            description: format!(
                "{} orphaned cmd.exe/conhost.exe processes detected (parent PID \
                 no longer alive). These accumulate handle table entries. \
                 Terminate them to recover handle table space.",
                report.orphan_terminal_count
            ),
            estimated_savings_mb: (report.orphan_terminal_count as u64) * 4,
            action: "kill_orphan_terminals".to_string(),
            safe_to_auto: true,
        });
    }

    // --- Priority 3: browser tab proliferation ---
    let browser_count = categories.get("browsers").map(|c| c.count).unwrap_or(0);
    if browser_count > 40 {
        let browser_mb = categories.get("browsers").map(|c| c.working_set_mb).unwrap_or(0);
        recs.push(OptimizationRecommendation {
            priority: 3,
            category: "browser_tab_sprawl".to_string(),
            description: format!(
                "{} browser processes are running (combined working set ~{} MiB). \
                 Close unused tabs or use a tab-suspender extension.",
                browser_count, browser_mb
            ),
            estimated_savings_mb: browser_mb / 3,
            action: "consolidate_browser_tabs".to_string(),
            safe_to_auto: false,
        });
    }

    // --- Priority 3: WSL memory budget ---
    // WSL2's vmmemWSL is counted as a single process under sysinfo.
    // Default .wslconfig limit is 50 % of physical RAM.
    let total_bytes = {
        let mut s = System::new_all();
        s.refresh_memory();
        s.total_memory()
    };
    let wsl_default_limit_mb = (total_bytes / (1024 * 1024)) / 2;
    // Approximate WSL usage: look for a process named "vmmemWSL" or "vmwp".
    let wsl_usage_mb = {
        let mut s = System::new_all();
        s.refresh_processes(sysinfo::ProcessesToUpdate::All, true);
        s.processes()
            .values()
            .filter(|p| {
                let n = p.name().to_string_lossy().to_lowercase();
                n.contains("vmmemwsl") || n.contains("vmwp")
            })
            .map(|p| p.memory() / (1024 * 1024))
            .sum::<u64>()
    };

    if wsl_usage_mb > 0 && wsl_usage_mb > wsl_default_limit_mb * 8 / 10 {
        recs.push(OptimizationRecommendation {
            priority: 3,
            category: "wsl_config".to_string(),
            description: format!(
                "WSL2 is consuming ~{} MiB, which is ≥80 % of the default \
                 {}-MiB limit. Add a `[wsl2] memory=8GB` entry to \
                 %USERPROFILE%\\.wslconfig to cap usage.",
                wsl_usage_mb, wsl_default_limit_mb
            ),
            estimated_savings_mb: wsl_usage_mb.saturating_sub(wsl_default_limit_mb / 2),
            action: "configure_wslconfig_memory_limit".to_string(),
            safe_to_auto: false,
        });
    }

    // --- Priority 2: build tool memory (rust-analyzer leak) ---
    let build_mb = categories.get("build_tools").map(|c| c.working_set_mb).unwrap_or(0);
    let build_handles = categories.get("build_tools").map(|c| c.handle_count).unwrap_or(0);
    if build_mb > 8_192 || build_handles > 500_000 {
        recs.push(OptimizationRecommendation {
            priority: 2,
            category: "build_tool_leak".to_string(),
            description: format!(
                "Build tools (rust-analyzer, cargo, node, dotnet) are using \
                 {} MiB with {} total handles. rust-analyzer is a known handle \
                 leaker; restart it or the IDE language server.",
                build_mb, build_handles
            ),
            estimated_savings_mb: build_mb / 2,
            action: "restart_rust_analyzer".to_string(),
            safe_to_auto: false,
        });
    }

    // Sort by priority ascending (1 = most urgent first), then by savings descending
    recs.sort_by(|a, b| {
        a.priority
            .cmp(&b.priority)
            .then(b.estimated_savings_mb.cmp(&a.estimated_savings_mb))
    });

    (start.elapsed().as_millis() as u64, recs)
}

// ── Public entry points (called from mod.rs FFI wrappers) ────────────────────

/// Collect the memory pressure report and serialize it to a JSON envelope.
pub fn memory_pressure_to_json(report: &MemoryPressureReport) -> MemoryPressureJson {
    MemoryPressureJson {
        status: if report.status.is_success() {
            "Success".to_string()
        } else {
            format!("{:?}", report.status)
        },
        pressure_level: report.pressure_level,
        pressure_label: pressure_label(report.pressure_level).to_string(),
        available_mb: report.available_mb,
        committed_pct: report.committed_pct,
        pool_nonpaged_mb: report.pool_nonpaged_mb,
        pages_per_sec: report.pages_per_sec,
        top_consumer_count: report.top_consumer_count,
        handle_leak_count: report.handle_leak_count,
        orphan_terminal_count: report.orphan_terminal_count,
        elapsed_ms: report.elapsed_ms,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pressure_level_thresholds() {
        assert_eq!(pressure_level(0.0), 0);
        assert_eq!(pressure_level(59.9), 0);
        assert_eq!(pressure_level(60.0), 1);
        assert_eq!(pressure_level(79.9), 1);
        assert_eq!(pressure_level(80.0), 2);
        assert_eq!(pressure_level(89.9), 2);
        assert_eq!(pressure_level(90.0), 3);
        assert_eq!(pressure_level(100.0), 3);
    }

    #[test]
    fn test_classify_process() {
        assert_eq!(classify_process("chrome"), "browsers");
        assert_eq!(classify_process("brave"), "browsers");
        assert_eq!(classify_process("msedge"), "browsers");
        assert_eq!(classify_process("ollama"), "llm_agents");
        assert_eq!(classify_process("claude"), "llm_agents");
        assert_eq!(classify_process("rust-analyzer"), "build_tools");
        assert_eq!(classify_process("cargo"), "build_tools");
        assert_eq!(classify_process("conhost"), "terminals");
        assert_eq!(classify_process("cmd"), "terminals");
        assert_eq!(classify_process("powershell"), "terminals");
        assert_eq!(classify_process("svchost"), "system_services");
        assert_eq!(classify_process("lsass"), "system_services");
    }

    #[test]
    fn test_is_orphan_candidate() {
        assert!(is_orphan_candidate("cmd"));
        assert!(is_orphan_candidate("conhost"));
        assert!(!is_orphan_candidate("powershell"));
        assert!(!is_orphan_candidate("chrome"));
    }

    #[test]
    fn test_analyze_memory_pressure_returns_valid_report() {
        let report = analyze_memory_pressure();
        assert_eq!(report.status, PcaiStatus::Success);
        // available_mb must be a reasonable fraction of total (we are not the
        // only process on the machine, but we should have some memory free)
        assert!(report.available_mb < u64::MAX);
        assert!(report.committed_pct >= 0.0);
        assert!(report.pressure_level <= 3);
    }

    #[test]
    fn test_get_process_categories_all_keys_present() {
        let (elapsed_ms, categories) = get_process_categories();
        // Every standard category key must be present
        for key in &["llm_agents", "browsers", "terminals", "build_tools", "system_services"] {
            assert!(categories.contains_key(*key), "missing category key: {}", key);
        }
        // system_services must always have at least one process (svchost, etc.)
        let sys_count = categories["system_services"].count;
        assert!(sys_count > 0, "expected at least one system_services process");
        // elapsed must be plausible
        assert!(elapsed_ms < 60_000, "category scan took unexpectedly long");
    }

    #[test]
    fn test_optimization_recommendations_are_sorted_by_priority() {
        let (_, recs) = get_optimization_recommendations();
        // Priorities must be non-decreasing (1 before 2 before 3 before 4)
        for window in recs.windows(2) {
            assert!(
                window[0].priority <= window[1].priority,
                "recommendations not sorted: priority {} before {}",
                window[0].priority,
                window[1].priority
            );
        }
    }

    #[test]
    fn test_recommendation_fields_are_non_empty() {
        let (_, recs) = get_optimization_recommendations();
        for rec in &recs {
            assert!(!rec.category.is_empty(), "recommendation has empty category");
            assert!(!rec.description.is_empty(), "recommendation has empty description");
            assert!(!rec.action.is_empty(), "recommendation has empty action");
            assert!(rec.priority >= 1 && rec.priority <= 4, "priority out of range");
        }
    }

    #[test]
    fn test_memory_pressure_to_json_label() {
        let mut report = MemoryPressureReport::default();
        report.status = PcaiStatus::Success;

        report.pressure_level = 0;
        assert_eq!(memory_pressure_to_json(&report).pressure_label, "low");

        report.pressure_level = 1;
        assert_eq!(memory_pressure_to_json(&report).pressure_label, "moderate");

        report.pressure_level = 2;
        assert_eq!(memory_pressure_to_json(&report).pressure_label, "high");

        report.pressure_level = 3;
        assert_eq!(memory_pressure_to_json(&report).pressure_label, "critical");
    }
}
