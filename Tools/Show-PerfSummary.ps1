<#
.SYNOPSIS
    Displays summary of collected performance data.
#>
param([string]$Path = (Join-Path $PSScriptRoot '..' 'Reports' 'system-performance-data.json'))

$data = Get-Content $Path -Raw | ConvertFrom-Json

Write-Host "=== System Info ===" -ForegroundColor Cyan
Write-Host "  CPU: $($data.SystemInfo.CPU)"
Write-Host "  Cores: $($data.SystemInfo.CPUCores) physical, $($data.SystemInfo.CPULogical) logical"
Write-Host "  Total RAM: $($data.SystemInfo.TotalRAM_GB) GB"

Write-Host "`n=== Memory Snapshot ===" -ForegroundColor Cyan
Write-Host "  Used: $($data.MemorySnapshot.UsedPhysical_GB) GB ($($data.MemorySnapshot.PercentUsed)%)"
Write-Host "  Free: $($data.MemorySnapshot.FreePhysical_GB) GB"
Write-Host "  Virtual Total: $($data.MemorySnapshot.TotalVirtual_GB) GB"
Write-Host "  Virtual Free: $($data.MemorySnapshot.FreeVirtual_GB) GB"

Write-Host "`n=== GPUs ===" -ForegroundColor Cyan
foreach ($gpu in $data.GPUs) {
    Write-Host "  $($gpu.Name): VRAM=$($gpu.VRAM_GB)GB Status=$($gpu.Status)"
}
if ($data.NvidiaGPUs) {
    foreach ($ng in $data.NvidiaGPUs) {
        Write-Host "  NVIDIA Detail: $($ng.Name) - $($ng.UsedVRAM_MB)/$($ng.TotalVRAM_MB) MB used, GPU=$($ng.GPU_Util_Pct)%, Mem=$($ng.Mem_Util_Pct)%, Temp=$($ng.Temp_C)C"
    }
}

Write-Host "`n=== Page Files ===" -ForegroundColor Cyan
foreach ($pf in $data.PageFiles) {
    Write-Host "  $($pf.Path): Current=$($pf.CurrentUsageMB) MB, Peak=$($pf.PeakUsageMB) MB"
}

Write-Host "`n=== Category Aggregates ===" -ForegroundColor Cyan
foreach ($prop in $data.CategoryAggregates.PSObject.Properties) {
    $v = $prop.Value
    if ($v.ProcessCount -gt 0) {
        Write-Host ("  {0,-15} {1,3} procs  {2,8:N1} MB WS  {3,8:N1} MB Private" -f $prop.Name, $v.ProcessCount, $v.TotalWS_MB, $v.TotalPrivate_MB)
    }
}

Write-Host "`n=== Top 20 Processes ===" -ForegroundColor Cyan
$i = 0
foreach ($p in $data.TopProcesses) {
    $i++
    if ($i -gt 20) { break }
    Write-Host ("  {0,2}. {1,-35} {2,8:N1} MB WS  {3,8:N1} MB Priv  H={4} T={5}  PID {6}" -f $i, $p.Name, $p.WorkingSet_MB, $p.Private_MB, $p.Handles, $p.Threads, $p.PID)
}

Write-Host "`n=== Heavy Services ===" -ForegroundColor Cyan
foreach ($svc in $data.HeavyServices | Select-Object -First 10) {
    Write-Host ("  {0,-40} {1,8:N1} MB WS  Mode={2}" -f $svc.DisplayName, $svc.WorkingSet_MB, $svc.StartMode)
}

Write-Host "`n=== Memory Samples (time-series) ===" -ForegroundColor Cyan
foreach ($s in $data.MemorySamples) {
    $ts = ([datetime]$s.Timestamp).ToString('HH:mm:ss')
    Write-Host ("  {0}  Avail={1,5} MB  Committed={2,6:N0} MB  Pages/s={3,6}  Faults/s={4,6}" -f `
        $ts, $s.available_mbytes, [math]::Round($s.committed_bytes/1MB,0), $s.pages_sec, $s.page_faults_sec)
}

Write-Host "`n=== Process Lasso ===" -ForegroundColor Cyan
Write-Host "  Installed: $($data.ProcessLasso.Installed)"
Write-Host "  Running: $($data.ProcessLasso.Running)"

Write-Host "`n=== Totals ===" -ForegroundColor Cyan
Write-Host "  Total processes: $($data.TotalProcessCount)"
Write-Host "  Total working set: $($data.TotalWorkingSet_GB) GB"

# Compute key metrics
$poolPaged = [math]::Round(($data.MemorySamples | Measure-Object -Property pool_paged_bytes -Average).Average / 1GB, 2)
$poolNonpaged = [math]::Round(($data.MemorySamples | Measure-Object -Property pool_nonpaged_bytes -Average).Average / 1GB, 2)
$avgAvail = [math]::Round(($data.MemorySamples | Measure-Object -Property available_mbytes -Average).Average, 0)
$avgPagesSec = [math]::Round(($data.MemorySamples | Measure-Object -Property pages_sec -Average).Average, 0)

Write-Host "`n=== Derived Metrics ===" -ForegroundColor Yellow
Write-Host "  Avg Pool Paged: $poolPaged GB"
Write-Host "  Avg Pool Nonpaged: $poolNonpaged GB"
Write-Host "  Avg Available: $avgAvail MB"
Write-Host "  Avg Pages/sec: $avgPagesSec (>1000 = heavy paging)"
Write-Host "  Committed/Limit ratio: $([math]::Round(($data.MemorySamples | Measure-Object -Property committed_bytes -Average).Average / ($data.MemorySamples | Measure-Object -Property commit_limit -Average).Average * 100, 1))%"
