<#
.SYNOPSIS
    Collects detailed process data focused on LLM agent terminals and memory hogs.
#>
param([string]$OutputPath = 'Reports/detailed-process-data.json')

$ErrorActionPreference = 'SilentlyContinue'

Write-Host "=== Detailed Process Analysis ===" -ForegroundColor Cyan

# 1. All node.exe processes with command lines
Write-Host "[1/6] Node.js/Electron processes..." -ForegroundColor Yellow
$nodeProcs = @()
Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'node|electron|bun|deno' } | ForEach-Object {
    $proc = Get-Process -Id $_.ProcessId -ErrorAction SilentlyContinue
    if ($proc) {
        $nodeProcs += @{
            Name = $_.Name
            PID = $_.ProcessId
            ParentPID = $_.ParentProcessId
            WorkingSet_MB = [math]::Round($proc.WorkingSet64 / 1MB, 1)
            Private_MB = [math]::Round($proc.PrivateMemorySize64 / 1MB, 1)
            Handles = $proc.HandleCount
            CommandLine = if ($_.CommandLine.Length -gt 300) { $_.CommandLine.Substring(0, 300) + '...' } else { $_.CommandLine }
        }
    }
}

# 2. Chrome/Brave tab analysis
Write-Host "[2/6] Browser processes..." -ForegroundColor Yellow
$browserProcs = @()
Get-Process -Name 'chrome', 'brave', 'msedge', 'firefox' -ErrorAction SilentlyContinue | ForEach-Object {
    $browserProcs += @{
        Name = $_.ProcessName
        PID = $_.Id
        WorkingSet_MB = [math]::Round($_.WorkingSet64 / 1MB, 1)
        Private_MB = [math]::Round($_.PrivateMemorySize64 / 1MB, 1)
        Handles = $_.HandleCount
    }
}
$browserSummary = @{
    ChromeCount = ($browserProcs | Where-Object { $_.Name -eq 'chrome' }).Count
    ChromeTotal_MB = [math]::Round(($browserProcs | Where-Object { $_.Name -eq 'chrome' } | Measure-Object -Property WorkingSet_MB -Sum).Sum, 0)
    ChromePrivate_MB = [math]::Round(($browserProcs | Where-Object { $_.Name -eq 'chrome' } | Measure-Object -Property Private_MB -Sum).Sum, 0)
    BraveCount = ($browserProcs | Where-Object { $_.Name -eq 'brave' }).Count
    BraveTotal_MB = [math]::Round(($browserProcs | Where-Object { $_.Name -eq 'brave' } | Measure-Object -Property WorkingSet_MB -Sum).Sum, 0)
    EdgeCount = ($browserProcs | Where-Object { $_.Name -eq 'msedge' }).Count
    EdgeTotal_MB = [math]::Round(($browserProcs | Where-Object { $_.Name -eq 'msedge' } | Measure-Object -Property WorkingSet_MB -Sum).Sum, 0)
}

# 3. rust-analyzer detail (massive handle count)
Write-Host "[3/6] rust-analyzer detail..." -ForegroundColor Yellow
$rustAnalyzer = @()
Get-Process -Name 'rust-analyzer' -ErrorAction SilentlyContinue | ForEach-Object {
    $rustAnalyzer += @{
        PID = $_.Id
        WorkingSet_MB = [math]::Round($_.WorkingSet64 / 1MB, 1)
        Private_MB = [math]::Round($_.PrivateMemorySize64 / 1MB, 1)
        Virtual_MB = [math]::Round($_.VirtualMemorySize64 / 1MB, 1)
        Handles = $_.HandleCount
        Threads = $_.Threads.Count
        StartTime = $_.StartTime.ToString('o')
        TotalCPU_Sec = [math]::Round($_.TotalProcessorTime.TotalSeconds, 1)
    }
}

# 4. WSL memory
Write-Host "[4/6] WSL memory..." -ForegroundColor Yellow
$wslMem = @()
Get-Process -Name 'vmmemWSL', 'vmmem', 'wsl', 'wslhost' -ErrorAction SilentlyContinue | ForEach-Object {
    $wslMem += @{
        Name = $_.ProcessName
        PID = $_.Id
        WorkingSet_MB = [math]::Round($_.WorkingSet64 / 1MB, 1)
        Private_MB = [math]::Round($_.PrivateMemorySize64 / 1MB, 1)
        Virtual_MB = [math]::Round($_.VirtualMemorySize64 / 1MB, 1)
    }
}
# Check .wslconfig
$wslConfig = $null
$wslConfigPath = Join-Path $env:USERPROFILE '.wslconfig'
if (Test-Path $wslConfigPath) {
    $wslConfig = Get-Content $wslConfigPath -Raw
}

# 5. Conhost/terminal analysis
Write-Host "[5/6] Terminal processes..." -ForegroundColor Yellow
$terminals = @()
Get-Process -Name 'conhost', 'wezterm*', 'WindowsTerminal', 'wt', 'cmd', 'powershell', 'pwsh' -ErrorAction SilentlyContinue | ForEach-Object {
    $terminals += @{
        Name = $_.ProcessName
        PID = $_.Id
        WorkingSet_MB = [math]::Round($_.WorkingSet64 / 1MB, 1)
        Private_MB = [math]::Round($_.PrivateMemorySize64 / 1MB, 1)
        Handles = $_.HandleCount
    }
}
$terminalSummary = @{}
$terminals | Group-Object { $_.Name } | ForEach-Object {
    $terminalSummary[$_.Name] = @{
        Count = $_.Count
        TotalWS_MB = [math]::Round(($_.Group | Measure-Object -Property WorkingSet_MB -Sum).Sum, 0)
        TotalPrivate_MB = [math]::Round(($_.Group | Measure-Object -Property Private_MB -Sum).Sum, 0)
    }
}

# 6. Claude/Codex/LLM agents
Write-Host "[6/6] LLM agent processes..." -ForegroundColor Yellow
$llmAgents = @()
Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'claude|codex|ollama|pcai|llama|copilot' } | ForEach-Object {
    $proc = Get-Process -Id $_.ProcessId -ErrorAction SilentlyContinue
    if ($proc) {
        $llmAgents += @{
            Name = $_.Name
            PID = $_.ProcessId
            WorkingSet_MB = [math]::Round($proc.WorkingSet64 / 1MB, 1)
            Private_MB = [math]::Round($proc.PrivateMemorySize64 / 1MB, 1)
            Handles = $proc.HandleCount
            Threads = $proc.Threads.Count
            CommandLine = if ($_.CommandLine.Length -gt 300) { $_.CommandLine.Substring(0, 300) + '...' } else { $_.CommandLine }
        }
    }
}

# Assemble
$report = @{
    CollectedAt = (Get-Date -Format 'o')
    NodeProcesses = ($nodeProcs | Sort-Object { $_.Private_MB } -Descending)
    BrowserSummary = $browserSummary
    RustAnalyzer = $rustAnalyzer
    WSLMemory = $wslMem
    WSLConfig = $wslConfig
    TerminalSummary = $terminalSummary
    LLMAgents = ($llmAgents | Sort-Object { $_.Private_MB } -Descending)
    # Pool memory analysis
    PoolAnalysis = @{
        Note = 'Pool nonpaged 10+ GB is abnormally high. Usually caused by driver leaks (NVIDIA, network, storage) or excessive handle counts.'
        TopHandleProcesses = (Get-Process | Sort-Object HandleCount -Descending | Select-Object -First 10 | ForEach-Object {
            @{ Name = $_.ProcessName; PID = $_.Id; Handles = $_.HandleCount; WS_MB = [math]::Round($_.WorkingSet64/1MB,1) }
        })
    }
}

$report | ConvertTo-Json -Depth 5 | Set-Content $OutputPath -Encoding UTF8
Write-Host "`nSaved to $OutputPath" -ForegroundColor Green

# Print key findings
Write-Host "`n=== Key Findings ===" -ForegroundColor Yellow
Write-Host "Browsers: Chrome=$($browserSummary.ChromeCount) procs ($($browserSummary.ChromeTotal_MB) MB WS / $($browserSummary.ChromePrivate_MB) MB Private), Brave=$($browserSummary.BraveCount) ($($browserSummary.BraveTotal_MB) MB)"
Write-Host "rust-analyzer: $($rustAnalyzer.Count) instance(s)"
foreach ($ra in $rustAnalyzer) {
    Write-Host "  PID $($ra.PID): WS=$($ra.WorkingSet_MB) MB, Private=$($ra.Private_MB) MB, Handles=$($ra.Handles), Threads=$($ra.Threads)"
}
Write-Host "WSL: $($wslMem.Count) processes"
foreach ($w in $wslMem) {
    Write-Host "  $($w.Name): WS=$($w.WorkingSet_MB) MB, Private=$($w.Private_MB) MB"
}
Write-Host "LLM Agents: $($llmAgents.Count) processes"
foreach ($a in $llmAgents) {
    Write-Host "  $($a.Name): WS=$($a.WorkingSet_MB) MB, Private=$($a.Private_MB) MB, PID=$($a.PID)"
}
Write-Host "`nTerminal Summary:"
foreach ($k in $terminalSummary.Keys | Sort-Object) {
    $v = $terminalSummary[$k]
    Write-Host "  $k`: $($v.Count) procs, WS=$($v.TotalWS_MB) MB, Private=$($v.TotalPrivate_MB) MB"
}
Write-Host "`nTop Handle Holders:"
foreach ($h in $report.PoolAnalysis.TopHandleProcesses) {
    Write-Host "  $($h.Name) PID=$($h.PID): $($h.Handles) handles, $($h.WS_MB) MB WS"
}
