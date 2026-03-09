# PC_AI Memory Optimization Proposal

**System:** Intel Core Ultra 7 155H (16c/22t), 64 GB RAM, RTX 5060 Ti (16GB) + RTX 2000 Ada (8GB, error state)
**Collected:** 2026-03-09 09:25 EDT | 5 samples @ 3s intervals | 837 processes analyzed
**Pressure Level:** CRITICAL (1.2-1.5 GB free of 64 GB = 84%+ utilization)

---

## Executive Summary

This system is under **severe memory pressure** while running LLM agents. With only ~1.3 GB average free RAM out of 64 GB, the system is actively thrashing (4,818 pages/sec vs healthy <100). A single rust-analyzer process has leaked 3.17 million handles and consumed 29 GB private memory, accounting for nearly half the system's RAM. Combined with WSL2 (5.4 GB), Claude agents (5.2 GB), and 94 browser processes, the system has zero headroom.

**Estimated reclaimable:** ~32 GB (50% of total RAM)
**Critical issues:** 4 (require immediate action)
**Effort:** Most gains come from 3 targeted actions requiring <5 minutes.

---

## Data Collection Methodology

| Source | Method | Samples |
|--------|--------|---------|
| System memory | `Win32_OperatingSystem` + perf counters | 5 snapshots @ 3s |
| Process data | `Get-Process` + `Win32_Process` WMI | Full enumeration (837 procs) |
| GPU | `nvidia-smi` + `Win32_VideoController` | Point-in-time |
| Pool memory | `\Memory\Pool Nonpaged Bytes` counter | 5 samples |
| Paging rate | `\Memory\Pages/sec` counter | 5 samples |
| Page file | `Win32_PageFileUsage` | Point-in-time |
| Handle analysis | Process handle counts | Full enumeration |
| Orphan detection | Parent PID existence check | Full enumeration |

---

## Findings by Priority

### P1 CRITICAL: rust-analyzer Handle/Memory Leak

| Metric | Value | Normal |
|--------|-------|--------|
| Handles | **3,168,135** | <1,000 |
| Private memory | **29.1 GB** | <2 GB |
| Working set | **4.0 GB** | <500 MB |
| Threads | 26 | ~10-15 |

**Root cause:** rust-analyzer is leaking OS handles at an extreme rate (3.17M vs typical <1K). Each handle consumes kernel pool nonpaged memory (~40-100 bytes), which explains the 10+ GB nonpaged pool. This is a known rust-analyzer issue when operating on large workspaces or when VS Code extensions trigger excessive file watches.

**Impact:** ~29 GB private memory + ~8 GB pool nonpaged = **~37 GB attributable** to this single process.

**Actions:**
1. **Immediate:** Restart rust-analyzer (VS Code: `Ctrl+Shift+P` → "Rust Analyzer: Restart Server")
2. **Preventive:** Add `rust-analyzer.files.excludeDirs` to VS Code settings for `node_modules`, `.git`, `target`, `build` directories
3. **Monitoring:** Process Lasso can be configured to alert when handle count exceeds 100K

### P1 CRITICAL: Nonpaged Pool Memory (10.2 GB)

| Metric | Value | Normal |
|--------|-------|--------|
| Pool Nonpaged Avg | **10.19 GB** | 1-2 GB |
| Pool Paged Avg | **8.41 GB** | 2-4 GB |

**Root cause:** Directly caused by rust-analyzer's 3.17M handles. Each handle creates a kernel object that lives in nonpaged pool (cannot be paged to disk). This memory is effectively "pinned" in RAM and cannot be reclaimed.

**Impact:** ~8 GB of nonpaged pool above normal, all locked in physical RAM.

**Action:** Resolves automatically when rust-analyzer is restarted.

### P1 CRITICAL: Excessive Paging (4,818 pages/sec)

| Metric | Value | Healthy |
|--------|-------|---------|
| Avg pages/sec | **4,818** | <100 |
| Peak pages/sec | **17,555** | <500 |
| Page file usage | **30.1 GB** | <10 GB |
| Available memory | **1,295 MB** avg | >4,096 MB |
| Committed bytes | **~113 GB** | <50 GB |

**Root cause:** System is overcommitted (113 GB committed vs 64 GB physical). The page file is absorbing 30 GB of overflow, causing constant disk I/O that degrades all applications.

**Impact:** All applications suffer latency spikes. LLM inference is particularly affected because model weights get paged out between requests.

**Actions:**
1. Address rust-analyzer leak first (reclaims ~29 GB immediately)
2. After reclaim, verify paging drops to <100/sec
3. If still elevated, reduce WSL memory limit (see P3)

### P2 HIGH: Claude Agent Memory (5.2 GB)

| Process | PID | Working Set | Private |
|---------|-----|-------------|---------|
| claude.exe | 67772 | 443 MB | **3.98 GB** |
| claude.exe | 86484 | 294 MB | **1.21 GB** |
| codex.exe | 73076 | 155 MB | 249 MB |
| **Total** | | **892 MB** | **5.45 GB** |

**Analysis:** Two Claude instances are running. PID 67772 has 4 GB private memory, which is high but expected for a frontier LLM agent with extensive conversation context. The gap between working set (443 MB) and private (3.98 GB) indicates most memory has been paged out.

**Actions:**
1. Close unused Claude sessions (each saves ~1-4 GB)
2. Use `claude /clear` to reset context in long-running sessions
3. Consider running only 1 Claude + 1 Codex concurrently instead of 2 Claude + 1 Codex

### P2 HIGH: Orphaned Terminal Processes

| Process | Count | Impact |
|---------|-------|--------|
| cmd.exe | 56 | Orphan spawns from build/agent processes |
| conhost.exe | 33 | Console host for orphaned cmd instances |

**Root cause:** LLM agents and build tools spawn cmd.exe subprocesses that persist after the parent exits. Each cmd+conhost pair uses ~5-15 MB.

**Impact:** ~500-800 MB in aggregate, plus handle/thread overhead.

**Actions:**
1. Automated cleanup script (safe: orphaned processes have no parent)
2. Process Lasso can be configured to terminate orphaned conhost after idle timeout

### P3 MEDIUM: WSL2 Memory (5.4 GB)

| Metric | Value | .wslconfig Setting |
|--------|-------|--------------------|
| vmmemWSL private | **5.4 GB** | `memory=34359738368` (32 GB limit) |
| vmmemWSL WS | 734 MB | — |
| Networking | VirtioProxy | — |
| Auto-reclaim | `Gradual` | Good |

**Analysis:** WSL2 is configured with a 32 GB limit but currently using only 5.4 GB. The `autoMemoryReclaim=Gradual` setting is correct for eventual reclaim. However, 32 GB is too generous given current pressure.

**Action:** Reduce `.wslconfig` memory limit:
```ini
[wsl2]
memory=16GB   # Was 32GB - halve it given LLM agent workload
processors=8  # Was 12 - reduce for agent-heavy workflows
```

### P3 MEDIUM: Browser Tab Sprawl (8.1 GB)

| Browser | Processes | Working Set | Private |
|---------|-----------|-------------|---------|
| Chrome | 43 | 4,383 MB | ~4,400 MB |
| Brave | 51 | 3,758 MB | ~3,200 MB |
| **Total** | **94** | **8,141 MB** | **~7,600 MB** |

**Actions:**
1. Install tab suspender extension (e.g., The Great Suspender) - saves 40-60% of browser RAM
2. Consolidate to one browser for daily use
3. Close unused tabs (target: <20 total across browsers)

### P4 LOW: RTX 2000 Ada in Error State

The RTX 2000 Ada (8 GB, GPU 1 — designated training GPU) shows `Status=Error` in WMI. Only the RTX 5060 Ti is visible to nvidia-smi (currently at 95% GPU util, 3 GB VRAM used, 56°C).

**Actions:**
1. Check Device Manager for error code
2. May need driver reinstall or BIOS update
3. This halves available GPU VRAM for training (8 GB lost)

---

## Process Lasso Configuration Recommendations

Process Lasso is installed and running. Configure these rules:

### Priority Rules for LLM Workloads
| Process | CPU Priority | I/O Priority | Memory Priority |
|---------|-------------|--------------|-----------------|
| pcai-inference.exe | High | High | High |
| ollama.exe | Above Normal | Normal | High |
| claude.exe | Normal | Normal | Normal |
| codex.exe | Normal | Normal | Normal |
| rust-analyzer.exe | Below Normal | Low | Low |
| chrome.exe | Below Normal | Low | Low |
| brave.exe | Below Normal | Low | Low |

### ProBalance Settings
- Enable ProBalance (auto-demotion of CPU-hogging processes)
- SmartTrim: Enable for browser processes (trims working sets when idle)
- Watchdog: Alert when any process exceeds 100K handles

### Recommended Process Lasso Rules
1. **CPU affinity for rust-analyzer**: Limit to P-cores 0-7 (prevent E-core scheduling overhead)
2. **Memory trim for browsers**: Periodic working set trim when tabs are background
3. **Terminate rule**: Kill conhost.exe processes idle >30 minutes with no parent

---

## Recommended Action Sequence

### Immediate (saves ~35 GB, <5 minutes)

| # | Action | Est. Savings | Risk |
|---|--------|-------------|------|
| 1 | Restart rust-analyzer in VS Code | **29 GB** + 8 GB pool | None |
| 2 | Close extra Claude session (PID 67772) | **4 GB** | Save work first |
| 3 | Run orphan terminal cleanup | **0.5-0.8 GB** | None |

### Short-term (saves ~5-8 GB, <30 minutes)

| # | Action | Est. Savings | Risk |
|---|--------|-------------|------|
| 4 | Reduce `.wslconfig` memory to 16 GB | **Up to 16 GB** cap | Restart WSL |
| 5 | Close 50% of browser tabs | **3-4 GB** | None |
| 6 | Install tab suspender extension | **2-3 GB** ongoing | None |

### Ongoing (preventive)

| # | Action | Benefit |
|---|--------|---------|
| 7 | Configure Process Lasso rules (above) | Auto-trim + alerts |
| 8 | Add rust-analyzer exclude dirs | Prevent handle leaks |
| 9 | Set up `Get-PcaiMemoryPressure` as a pre-flight check before launching agents | Prevents overcommit |
| 10 | Investigate RTX 2000 Ada error state | Recover 8 GB VRAM |

---

## New PC-AI Tooling (CSharp_RustDLL Pattern)

Three new PowerShell cmdlets have been created, following the Rust FFI → C# P/Invoke → PowerShell paradigm:

### Cmdlets

| Cmdlet | Purpose | Native Fallback |
|--------|---------|-----------------|
| `Get-PcaiMemoryPressure` | Memory pressure level with paging, pool, handle leak detection | Yes |
| `Get-PcaiProcessCategories` | LLM-workload-aware process classification | Yes |
| `Get-PcaiOptimizationPlan` | Prioritized recommendations with estimated savings | Yes |

### Architecture (CSharp_RustDLL)

```
Rust (pcai_core_lib.dll)          C# (PcaiNative)              PowerShell
─────────────────────────         ─────────────────             ──────────
optimizer.rs                      OptimizerModule.cs            Get-PcaiMemoryPressure
  pcai_analyze_memory_pressure()    AnalyzeMemoryPressure()     Get-PcaiProcessCategories
  pcai_get_memory_pressure_json()   GetMemoryPressureJson()     Get-PcaiOptimizationPlan
  pcai_get_process_categories_json()GetProcessCategoriesJson()
  pcai_get_optimization_recommendations_json()
                                    GetOptimizationRecommendationsJson()
```

### Data Collection Scripts

| Script | Purpose |
|--------|---------|
| `Tools/Collect-SystemPerformanceData.ps1` | Comprehensive system snapshot (JSON) |
| `Tools/Collect-DetailedProcessData.ps1` | LLM/browser/terminal deep analysis |
| `Tools/Show-PerfSummary.ps1` | Human-readable summary of collected data |
| `Tools/Test-Optimizer.ps1` | Validation test for optimizer cmdlets |

### Usage

```powershell
# Quick health check before launching agents
$pressure = Get-PcaiMemoryPressure
if ($pressure.PressureLevelCode -ge 2) {
    Write-Warning "Memory pressure is $($pressure.PressureLevel) - consider closing applications"
    Get-PcaiOptimizationPlan | Where-Object { $_.Priority -le 2 } | Format-Table
}

# Full analysis
Get-PcaiProcessCategories | Format-Table -AutoSize
Get-PcaiOptimizationPlan | Format-List

# Collect baseline data
.\Tools\Collect-SystemPerformanceData.ps1 -SampleCount 10 -SampleIntervalSec 5
.\Tools\Show-PerfSummary.ps1
```

---

## Appendix: Raw Data Summary

### Memory Samples (5 @ 3-second intervals)
| Time | Available MB | Committed GB | Pages/sec | Faults/sec |
|------|-------------|-------------|-----------|------------|
| 09:25:21 | 1,499 | 112.7 | 2,441 | 30,662 |
| 09:25:26 | 1,381 | 112.9 | 2,700 | 46,873 |
| 09:25:30 | 1,172 | 113.1 | 17,555 | 29,272 |
| 09:25:34 | 1,206 | 113.1 | 1,042 | 40,627 |
| 09:25:38 | 1,217 | 113.0 | 352 | 34,520 |

### Process Category Breakdown (live test run)
| Category | Procs | WS MB | Private MB | Handles | Top Process |
|----------|-------|-------|-----------|---------|-------------|
| Build_Tools | 16 | 2,229 | 31,620 | 3,449,985 | rust-analyzer |
| Other | 397 | 7,888 | 17,227 | 575,140 | nvcontainer |
| Browsers | 124 | 8,555 | 11,485 | 60,068 | chrome |
| WSL | 11 | 759 | 6,275 | 2,054 | vmmemWSL |
| Node/Electron | 56 | 1,091 | 5,999 | 11,184 | bun |
| LLM_Agents | 4 | 1,016 | 5,861 | 5,265 | claude |
| System | 119 | 2,245 | 3,827 | 89,646 | dwm |
| Terminals | 93 | 409 | 944 | 12,414 | WindowsTerminal |
| Shells | 7 | 279 | 617 | 4,129 | powershell |
| Python | 26 | 69 | 590 | 3,455 | python |
