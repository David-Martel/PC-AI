function Get-PcaiOptimizationPlan {
    <#
    .SYNOPSIS
        Generates prioritized memory/performance optimization recommendations.
    .DESCRIPTION
        Analyzes current system state and generates actionable optimization
        recommendations with estimated memory savings. Uses native Rust FFI
        when available, falls back to PowerShell analysis.
    .OUTPUTS
        Array of recommendation objects sorted by priority.
    .EXAMPLE
        Get-PcaiOptimizationPlan
    .EXAMPLE
        Get-PcaiOptimizationPlan | Where-Object { $_.Priority -le 2 }
    .EXAMPLE
        Get-PcaiOptimizationPlan -AsJson
    #>
    [CmdletBinding()]
    param(
        [switch]$AsJson
    )

    Import-Module PC-AI.Common -ErrorAction SilentlyContinue
    $nativeAvailable = $false
    try { $nativeAvailable = Initialize-PcaiNative } catch {}

    if ($nativeAvailable) {
        $json = [PcaiNative.OptimizerModule]::GetOptimizationRecommendationsJson()
        if ($json) {
            if ($AsJson) { return $json }
            return $json | ConvertFrom-Json
        }
    }

    # Fallback: PowerShell-based recommendations
    Write-Verbose 'Native DLL unavailable, using PowerShell fallback'
    $recommendations = @()

    $os = Get-CimInstance Win32_OperatingSystem
    $cs = Get-CimInstance Win32_ComputerSystem
    $availMB = [math]::Round($os.FreePhysicalMemory / 1KB, 0)

    # 1. Handle leak detection
    $handleLeakers = Get-Process | Where-Object { $_.HandleCount -gt 100000 } |
        Sort-Object HandleCount -Descending
    foreach ($leaker in $handleLeakers) {
        $recommendations += [PSCustomObject]@{
            Priority          = 1
            Category          = 'handle_leak'
            Description       = "$($leaker.ProcessName) (PID $($leaker.Id)) has $($leaker.HandleCount) handles and $([math]::Round($leaker.PrivateMemorySize64/1GB,1)) GB private memory. Likely a handle/memory leak."
            EstimatedSavingsMB = [math]::Round($leaker.PrivateMemorySize64 / 1MB * 0.5, 0)
            Action            = "restart_process:$($leaker.ProcessName)"
            SafeToAuto        = $false
        }
    }

    # 2. Pool nonpaged analysis
    try {
        $poolNP = (Get-Counter '\Memory\Pool Nonpaged Bytes' -ErrorAction Stop).CounterSamples[0].CookedValue
        $poolNP_GB = [math]::Round($poolNP / 1GB, 1)
        if ($poolNP_GB -gt 4) {
            $recommendations += [PSCustomObject]@{
                Priority          = 1
                Category          = 'pool_nonpaged'
                Description       = "Pool nonpaged memory is $poolNP_GB GB (normal: 1-2 GB). Usually caused by driver leaks (NVIDIA, network, storage) or processes with excessive handles."
                EstimatedSavingsMB = [math]::Round(($poolNP_GB - 2) * 1024, 0)
                Action            = 'investigate_pool_nonpaged'
                SafeToAuto        = $false
            }
        }
    } catch {}

    # 3. Orphan terminals
    $allPids = (Get-Process).Id
    $orphanCmds = @()
    Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^(cmd|conhost)\.exe$' } | ForEach-Object {
        if ($_.ParentProcessId -notin $allPids) {
            $orphanCmds += $_
        }
    }
    if ($orphanCmds.Count -gt 5) {
        $orphanMB = 0
        foreach ($o in $orphanCmds) {
            $p = Get-Process -Id $o.ProcessId -ErrorAction SilentlyContinue
            if ($p) { $orphanMB += [math]::Round($p.WorkingSet64 / 1MB, 0) }
        }
        $recommendations += [PSCustomObject]@{
            Priority          = 2
            Category          = 'orphan_cleanup'
            Description       = "$($orphanCmds.Count) orphaned cmd/conhost processes detected (parent PID gone). Total: ~$orphanMB MB."
            EstimatedSavingsMB = $orphanMB
            Action            = 'kill_orphan_terminals'
            SafeToAuto        = $true
        }
    }

    # 4. Browser tab sprawl
    $browsers = @{
        'chrome' = (Get-Process -Name 'chrome' -ErrorAction SilentlyContinue | Measure-Object)
        'brave'  = (Get-Process -Name 'brave' -ErrorAction SilentlyContinue | Measure-Object)
        'msedge' = (Get-Process -Name 'msedge' -ErrorAction SilentlyContinue | Measure-Object)
    }
    foreach ($browser in $browsers.GetEnumerator()) {
        if ($browser.Value.Count -gt 30) {
            $totalMB = [math]::Round((Get-Process -Name $browser.Key -ErrorAction SilentlyContinue |
                Measure-Object -Property WorkingSet64 -Sum).Sum / 1MB, 0)
            $recommendations += [PSCustomObject]@{
                Priority          = 3
                Category          = 'browser_tabs'
                Description       = "$($browser.Key) has $($browser.Value.Count) processes using ~$totalMB MB. Consider closing unused tabs or using tab suspender."
                EstimatedSavingsMB = [math]::Round($totalMB * 0.4, 0)
                Action            = "reduce_browser_tabs:$($browser.Key)"
                SafeToAuto        = $false
            }
        }
    }

    # 5. WSL memory
    $wslProc = Get-Process -Name 'vmmemWSL' -ErrorAction SilentlyContinue
    if ($wslProc) {
        $wslPrivateMB = [math]::Round($wslProc.PrivateMemorySize64 / 1MB, 0)
        if ($wslPrivateMB -gt 4096) {
            $wslConfigPath = Join-Path $env:USERPROFILE '.wslconfig'
            $hasConfig = Test-Path $wslConfigPath
            $recommendations += [PSCustomObject]@{
                Priority          = 3
                Category          = 'wsl_memory'
                Description       = "WSL2 VM using $wslPrivateMB MB private memory. $(if($hasConfig){'Check .wslconfig memory limit.'}else{'Consider creating .wslconfig with memory limit.'})"
                EstimatedSavingsMB = [math]::Round($wslPrivateMB * 0.3, 0)
                Action            = 'tune_wsl_config'
                SafeToAuto        = $false
            }
        }
    }

    # 6. Paging rate
    try {
        $pagesSec = (Get-Counter '\Memory\Pages/sec' -ErrorAction Stop).CounterSamples[0].CookedValue
        if ($pagesSec -gt 1000) {
            $recommendations += [PSCustomObject]@{
                Priority          = 1
                Category          = 'excessive_paging'
                Description       = "System is paging at $([math]::Round($pagesSec,0)) pages/sec (healthy: <100). Available memory: $availMB MB. System is thrashing."
                EstimatedSavingsMB = 0
                Action            = 'reduce_memory_footprint'
                SafeToAuto        = $false
            }
        }
    } catch {}

    # 7. Large individual processes (>2GB private, not already flagged as handle leak)
    $leakerPids = $handleLeakers | ForEach-Object { $_.Id }
    Get-Process | Where-Object {
        $_.PrivateMemorySize64 -gt 2GB -and $_.Id -notin $leakerPids
    } | Sort-Object PrivateMemorySize64 -Descending | ForEach-Object {
        $recommendations += [PSCustomObject]@{
            Priority          = 2
            Category          = 'large_process'
            Description       = "$($_.ProcessName) (PID $($_.Id)) using $([math]::Round($_.PrivateMemorySize64/1GB,1)) GB private memory."
            EstimatedSavingsMB = [math]::Round($_.PrivateMemorySize64 / 1MB * 0.2, 0)
            Action            = "investigate_process:$($_.ProcessName):$($_.Id)"
            SafeToAuto        = $false
        }
    }

    $sorted = $recommendations | Sort-Object Priority, @{Expression={$_.EstimatedSavingsMB}; Descending=$true}

    if ($AsJson) {
        return $sorted | ConvertTo-Json -Depth 3
    }
    return $sorted
}
