# Benchmarking & Profiling Enhancement Design (Sub-project B)

> **Date:** 2026-03-27
> **Status:** Approved
> **Author:** Claude Opus 4.6 + David Martel
> **Repo:** David-Martel/PC-AI
> **Part of:** 4-part testing/benchmarking/LLM optimization initiative (A → **B** → C → D)

## Problem Statement

The repo has a solid benchmark suite (14 cases, 5 categories) but no automated regression detection. Profiling requires manual tool invocation. There's no way for CI to block a PR that makes performance worse.

## Components

### 1. `Tests/Invoke-BenchmarkGate.ps1`
CI benchmark regression detector. Compares current run against saved baseline, flags regressions exceeding a threshold.

### 2. `Tools/Invoke-RustProfile.ps1`
Unified profiling wrapper: flamegraph (CPU), cargo bench (criterion), memory profiling, and tool summary.

### 3. `.pcai/benchmarks/baseline.json`
Structured baseline format with per-case metrics, git hash, and timestamp.

## Success Criteria

1. `Invoke-BenchmarkGate.ps1 -SaveBaseline` creates a valid baseline
2. `Invoke-BenchmarkGate.ps1 -FailOnRegression` correctly detects >15% regressions
3. `Invoke-RustProfile.ps1 -Profile Summary` lists available profiling tools
4. Markdown output is suitable for PR comments
