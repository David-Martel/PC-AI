<#
.SYNOPSIS
    Collects comprehensive system performance data for RAM optimization analysis.
.DESCRIPTION
    Gathers memory usage, process details, GPU info, page file config, performance
    counters, and terminal/LLM process analysis. Outputs structured JSON report.
#>
[CmdletBinding()]
param(
    [int]$SampleCount = 5,
    [int]$SampleIntervalSec = 3,
    [string]$OutputPath = (Join-Path $PSScriptRoot '..' 'Reports' 'system-performance-data.json')
)

$ErrorActionPreference = 'SilentlyContinue'

Write-Host "=== PC_AI System Performance Data Collection ===" -ForegroundColor Cyan
Write-Host "Collecting $SampleCount samples at ${SampleIntervalSec}s intervals...`n"

# --- 1. Static system info ---
Write-Host "[1/7] System info..." -ForegroundColor Yellow
$os = Get-CimInstance Win32_OperatingSystem
$cs = Get-CimInstance Win32_ComputerSystem
$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1

$systemInfo = @{
    Hostname        = $env:COMPUTERNAME
    OS              = $os.Caption
    OSBuild         = $os.BuildNumber
    CPU             = $cpu.Name
    CPUCores        = $cpu.NumberOfCores
    CPULogical      = $cpu.NumberOfLogicalProcessors
    TotalRAM_GB     = [math]::Round($cs.TotalPhysicalMemory / 1GB, 2)
    CollectedAt     = (Get-Date -Format 'o')
}

# --- 2. GPU info ---
Write-Host "[2/7] GPU info..." -ForegroundColor Yellow
$gpuInfo = @()
$gpus = Get-CimInstance Win32_VideoController
foreach ($gpu in $gpus) {
    $gpuInfo += @{
        Name        = $gpu.Name
        VRAM_GB     = [math]::Round($gpu.AdapterRAM / 1GB, 1)
        Driver      = $gpu.DriverVersion
        Status      = $gpu.Status
    }
}

# NVIDIA specific — use PC-AI.Gpu module (NVML FFI primary, nvidia-smi fallback).
$nvidiaGpu = @()
$gpuModulePath = Join-Path (Split-Path $PSScriptRoot -Parent) 'Modules\PC-AI.Gpu\PC-AI.Gpu.psd1'
if (Test-Path $gpuModulePath -ErrorAction SilentlyContinue) {
    Import-Module $gpuModulePath -Force -ErrorAction SilentlyContinue
}

if (Get-Command Get-NvidiaGpuInventory -ErrorAction SilentlyContinue) {
    $gpuInventory = Get-NvidiaGpuInventory -ErrorAction SilentlyContinue
    foreach ($gpu in @($gpuInventory)) {
        $freeMB = if ($null -ne $gpu.MemoryTotalMB -and $null -ne $gpu.MemoryUsedMB) {
            $gpu.MemoryTotalMB - $gpu.MemoryUsedMB
        } else { $null }
        $nvidiaGpu += @{
            Name         = $gpu.Name
            TotalVRAM_MB = $gpu.MemoryTotalMB
            UsedVRAM_MB  = $gpu.MemoryUsedMB
            FreeVRAM_MB  = $freeMB
            GPU_Util_Pct = $gpu.Utilization   # $null on nvml-ffi source (static snapshot)
            Mem_Util_Pct = $null              # not in GpuInfo static snapshot
            Temp_C       = $gpu.Temperature
            Source       = $gpu.Source
        }
    }
}
elseif (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    # Hard fallback: nvidia-smi subprocess when module not available.
    $nvOut = nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu --format=csv,noheader,nounits 2>$null
    foreach ($line in $nvOut) {
        $parts = $line -split ',\s*'
        if ($parts.Count -ge 7) {
            $nvidiaGpu += @{
                Name         = $parts[0]
                TotalVRAM_MB = [int]$parts[1]
                UsedVRAM_MB  = [int]$parts[2]
                FreeVRAM_MB  = [int]$parts[3]
                GPU_Util_Pct = [int]$parts[4]
                Mem_Util_Pct = [int]$parts[5]
                Temp_C       = [int]$parts[6]
                Source       = 'nvidia-smi'
            }
        }
    }
}

# --- 3. Page file ---
Write-Host "[3/7] Page file config..." -ForegroundColor Yellow
$pageFiles = @()
$pfs = Get-CimInstance Win32_PageFileUsage
foreach ($pf in $pfs) {
    $pageFiles += @{
        Path            = $pf.Name
        AllocatedMB     = $pf.AllocateBaseSize
        CurrentUsageMB  = $pf.CurrentUsage
        PeakUsageMB     = $pf.PeakUsage
    }
}

# --- 4. Memory snapshot ---
Write-Host "[4/7] Memory snapshot..." -ForegroundColor Yellow
$memSnapshot = @{
    FreePhysical_GB     = [math]::Round($os.FreePhysicalMemory / 1MB, 2)
    UsedPhysical_GB     = [math]::Round(($cs.TotalPhysicalMemory - $os.FreePhysicalMemory * 1KB) / 1GB, 2)
    PercentUsed         = [math]::Round((1 - $os.FreePhysicalMemory * 1KB / $cs.TotalPhysicalMemory) * 100, 1)
    TotalVirtual_GB     = [math]::Round($os.TotalVirtualMemorySize / 1MB, 2)
    FreeVirtual_GB      = [math]::Round($os.FreeVirtualMemory / 1MB, 2)
}

# --- 5. Process analysis ---
Write-Host "[5/7] Process analysis (all + LLM-focused)..." -ForegroundColor Yellow

# Top 40 by working set
$top40 = Get-Process | Sort-Object WorkingSet64 -Descending | Select-Object -First 40
$topProcesses = @()
foreach ($p in $top40) {
    $cmdLine = ''
    try {
        $wmiProc = Get-CimInstance Win32_Process -Filter "ProcessId=$($p.Id)"
        $cmdLine = if ($wmiProc.CommandLine.Length -gt 200) { $wmiProc.CommandLine.Substring(0, 200) + '...' } else { $wmiProc.CommandLine }
    } catch {}
    $topProcesses += @{
        Name            = $p.ProcessName
        PID             = $p.Id
        WorkingSet_MB   = [math]::Round($p.WorkingSet64 / 1MB, 1)
        Private_MB      = [math]::Round($p.PrivateMemorySize64 / 1MB, 1)
        Virtual_MB      = [math]::Round($p.VirtualMemorySize64 / 1MB, 1)
        Handles         = $p.HandleCount
        Threads         = $p.Threads.Count
        CommandLine     = $cmdLine
    }
}

# LLM/Terminal process patterns
$llmPatterns = @('node', 'claude', 'ollama', 'wezterm', 'WindowsTerminal', 'wt',
    'powershell', 'pwsh', 'cmd', 'conhost', 'python', 'cargo', 'rust',
    'dotnet', 'Code', 'cursor', 'electron', 'chrome', 'msedge', 'firefox',
    'copilot', 'bun', 'deno', 'pcai', 'llama', 'mistral', 'ProcessLasso')

$llmProcesses = @()
$allProcs = Get-Process
foreach ($p in $allProcs) {
    $matched = $false
    foreach ($pat in $llmPatterns) {
        if ($p.ProcessName -like "*$pat*") { $matched = $true; break }
    }
    if ($matched) {
        $cmdLine = ''
        try {
            $wmiProc = Get-CimInstance Win32_Process -Filter "ProcessId=$($p.Id)"
            $cmdLine = if ($wmiProc.CommandLine.Length -gt 200) { $wmiProc.CommandLine.Substring(0, 200) + '...' } else { $wmiProc.CommandLine }
        } catch {}
        $llmProcesses += @{
            Name            = $p.ProcessName
            PID             = $p.Id
            WorkingSet_MB   = [math]::Round($p.WorkingSet64 / 1MB, 1)
            Private_MB      = [math]::Round($p.PrivateMemorySize64 / 1MB, 1)
            Handles         = $p.HandleCount
            Threads         = $p.Threads.Count
            CommandLine     = $cmdLine
        }
    }
}

# Aggregate by category
$categoryMap = @{
    'Browsers'      = @('chrome', 'msedge', 'firefox')
    'Terminals'     = @('wezterm', 'WindowsTerminal', 'wt', 'conhost', 'cmd')
    'Shells'        = @('powershell', 'pwsh')
    'Node/Electron' = @('node', 'electron', 'bun', 'deno')
    'IDEs'          = @('Code', 'cursor')
    'LLM/AI'        = @('ollama', 'claude', 'pcai', 'llama', 'mistral', 'copilot')
    'Build'         = @('cargo', 'rust', 'dotnet')
    'Python'        = @('python')
    'SystemTools'   = @('ProcessLasso')
}

$categoryAggregates = @{}
foreach ($cat in $categoryMap.Keys) {
    $catProcs = $llmProcesses | Where-Object {
        $name = $_.Name
        $categoryMap[$cat] | Where-Object { $name -like "*$_*" }
    }
    $categoryAggregates[$cat] = @{
        ProcessCount    = ($catProcs | Measure-Object).Count
        TotalWS_MB      = [math]::Round(($catProcs | Measure-Object -Property WorkingSet_MB -Sum).Sum, 1)
        TotalPrivate_MB  = [math]::Round(($catProcs | Measure-Object -Property Private_MB -Sum).Sum, 1)
    }
}

# --- 6. Time-series memory samples ---
Write-Host "[6/7] Collecting $SampleCount memory samples..." -ForegroundColor Yellow
$memorySamples = @()
for ($i = 1; $i -le $SampleCount; $i++) {
    Write-Host "  Sample $i/$SampleCount..."
    $counters = @(
        '\Memory\Available MBytes',
        '\Memory\Committed Bytes',
        '\Memory\Commit Limit',
        '\Memory\Cache Bytes',
        '\Memory\Pool Paged Bytes',
        '\Memory\Pool Nonpaged Bytes',
        '\Memory\Pages/sec',
        '\Memory\Page Faults/sec'
    )
    $sample = @{ Timestamp = (Get-Date -Format 'o') }
    $counterData = Get-Counter -Counter $counters -ErrorAction SilentlyContinue
    if ($counterData) {
        foreach ($cs2 in $counterData.CounterSamples) {
            $key = ($cs2.Path -split '\\')[-1] -replace '[/\s]', '_'
            $sample[$key] = [math]::Round($cs2.CookedValue, 0)
        }
    }
    $memorySamples += $sample
    if ($i -lt $SampleCount) { Start-Sleep -Seconds $SampleIntervalSec }
}

# --- 7. Services and startup impact ---
Write-Host "[7/7] Service and startup analysis..." -ForegroundColor Yellow
$heavyServices = @()
$services = Get-CimInstance Win32_Service | Where-Object { $_.State -eq 'Running' }
foreach ($svc in $services) {
    $proc = Get-Process -Id $svc.ProcessId -ErrorAction SilentlyContinue
    if ($proc -and $proc.WorkingSet64 -gt 50MB) {
        $heavyServices += @{
            ServiceName     = $svc.Name
            DisplayName     = $svc.DisplayName
            PID             = $svc.ProcessId
            StartMode       = $svc.StartMode
            WorkingSet_MB   = [math]::Round($proc.WorkingSet64 / 1MB, 1)
        }
    }
}
$heavyServices = $heavyServices | Sort-Object { $_.WorkingSet_MB } -Descending

# --- Process Lasso check ---
$processLassoInstalled = $false
$plPaths = @(
    'C:\Program Files\Process Lasso\ProcessLasso.exe',
    'C:\Program Files (x86)\Process Lasso\ProcessLasso.exe',
    "${env:ProgramFiles}\Process Lasso\ProcessLasso.exe"
)
foreach ($path in $plPaths) {
    if (Test-Path $path) { $processLassoInstalled = $true; break }
}
$plProcess = Get-Process -Name 'ProcessLasso*' -ErrorAction SilentlyContinue

# --- Assemble report ---
$report = @{
    SystemInfo          = $systemInfo
    GPUs                = $gpuInfo
    NvidiaGPUs          = $nvidiaGpu
    PageFiles           = $pageFiles
    MemorySnapshot      = $memSnapshot
    TopProcesses        = $topProcesses
    LLMTerminalProcs    = ($llmProcesses | Sort-Object { $_.WorkingSet_MB } -Descending)
    CategoryAggregates  = $categoryAggregates
    MemorySamples       = $memorySamples
    HeavyServices       = $heavyServices
    ProcessLasso        = @{
        Installed       = $processLassoInstalled
        Running         = ($null -ne $plProcess)
    }
    TotalProcessCount   = ($allProcs | Measure-Object).Count
    TotalWorkingSet_GB  = [math]::Round(($allProcs | Measure-Object -Property WorkingSet64 -Sum).Sum / 1GB, 2)
}

# Output
$outDir = Split-Path $OutputPath -Parent
if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir -Force | Out-Null }
$report | ConvertTo-Json -Depth 5 | Set-Content $OutputPath -Encoding UTF8

Write-Host "`n=== Collection Complete ===" -ForegroundColor Green
Write-Host "Report saved to: $OutputPath"
Write-Host "`n--- Quick Summary ---" -ForegroundColor Cyan
Write-Host "Total RAM: $($systemInfo.TotalRAM_GB) GB"
Write-Host "Used RAM:  $($memSnapshot.UsedPhysical_GB) GB ($($memSnapshot.PercentUsed)%)"
Write-Host "Free RAM:  $($memSnapshot.FreePhysical_GB) GB"
Write-Host "Total Processes: $($report.TotalProcessCount)"
Write-Host "Total Working Set: $($report.TotalWorkingSet_GB) GB"
Write-Host "Process Lasso: $(if($processLassoInstalled){'Installed'}else{'Not found'}) / $(if($plProcess){'Running'}else{'Not running'})"

Write-Host "`n--- Category RAM Usage ---" -ForegroundColor Cyan
foreach ($cat in ($categoryAggregates.Keys | Sort-Object)) {
    $agg = $categoryAggregates[$cat]
    if ($agg.ProcessCount -gt 0) {
        Write-Host ("  {0,-15} {1,3} procs  {2,8:N1} MB WS  {3,8:N1} MB Private" -f $cat, $agg.ProcessCount, $agg.TotalWS_MB, $agg.TotalPrivate_MB)
    }
}

Write-Host "`n--- Top 10 Processes by RAM ---" -ForegroundColor Cyan
$topProcesses | Select-Object -First 10 | ForEach-Object {
    Write-Host ("  {0,-30} {1,8:N1} MB WS  {2,8:N1} MB Private  PID {3}" -f $_.Name, $_.WorkingSet_MB, $_.Private_MB, $_.PID)
}

return $report
