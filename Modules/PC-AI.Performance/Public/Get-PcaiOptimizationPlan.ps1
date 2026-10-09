function Get-PcaiOptimizationPlan {
    <#
    .SYNOPSIS
        Generates prioritized memory/performance optimization recommendations.
    .DESCRIPTION
        Analyzes current system state and generates actionable optimization
        manual-only snapshot observations without measured savings. Uses native Rust FFI
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

    function Test-RecommendationNumber($Value) {
        if ($null -eq $Value -or $Value -is [bool] -or $Value -is [string] -or $Value -isnot [ValueType]) { return $false }
        try { $number=[double]$Value } catch { return $false }
        return -not [double]::IsNaN($number) -and -not [double]::IsInfinity($number) -and $number -ge 0
    }
    Import-Module PC-AI.Common -ErrorAction SilentlyContinue
    $nativeAvailable = $false
    try { $nativeAvailable = Initialize-PcaiNative } catch {}

    if ($nativeAvailable) {
        try {
            $json=[PcaiNative.OptimizerModule]::GetOptimizationRecommendationsJson()
            if ([string]::IsNullOrWhiteSpace($json) -or -not $json.TrimStart().StartsWith('{')) { throw 'Native recommendation object required.' }
            $raw=$json|ConvertFrom-Json -ErrorAction Stop
            if ($raw.status -cne 'Success' -or -not (Test-RecommendationNumber $raw.recommendation_count) -or $raw.recommendations -isnot [array]) { throw 'Native recommendation status/schema invalid.' }
            if ($raw.recommendation_count -ne $raw.recommendations.Count) { throw 'Native recommendation cardinality mismatch.' }
            $rows=@(foreach ($item in $raw.recommendations) {
                if ($item -isnot [pscustomobject] -or -not (Test-RecommendationNumber $item.priority) -or $item.priority -ne [math]::Truncate($item.priority)) { throw 'Native recommendation priority invalid.' }
                foreach($field in @('category','description','action')) {
                    if ($item.$field -isnot [string] -or [string]::IsNullOrWhiteSpace($item.$field)) { throw 'Native recommendation text invalid.' }
                }
                if ($item.safe_to_auto -isnot [bool] -or -not (Test-RecommendationNumber $item.estimated_savings_mb)) { throw 'Native recommendation metric invalid.' }
                [pscustomobject]@{Priority=$item.priority;Category=$item.category;Description="Native snapshot flagged '$($item.category)'. Review measurements before manual action; no causal diagnosis or savings has been established.";EstimatedSavingsMB=$null;Action=$item.action;SafeToAuto=$false;Source='PcaiNative.OptimizerModule';MeasurementStatus='NativeSnapshotObservationOnly'}
            })
            $rows=@($rows|Sort-Object Priority,Category)
            if ($AsJson) { return ConvertTo-Json -InputObject @($rows) -Depth 5 }
            return $rows
        } catch { Write-Verbose 'Native recommendation acquisition or schema unavailable; using fallback.' }
    }
    # Fallback: PowerShell-based recommendations
    Write-Verbose 'Native DLL unavailable, using PowerShell fallback'
    $recommendations = @()

    $os = Get-CimInstance Win32_OperatingSystem -ErrorAction Stop
    $cs = Get-CimInstance Win32_ComputerSystem -ErrorAction Stop
    if (-not (Test-RecommendationNumber $os.FreePhysicalMemory) -or -not (Test-RecommendationNumber $cs.TotalPhysicalMemory) -or $cs.TotalPhysicalMemory -le 0 -or [double]$os.FreePhysicalMemory*1024 -gt $cs.TotalPhysicalMemory) {
        throw [IO.InvalidDataException]::new('Physical-memory metadata is invalid.')
    }
    $availMB = [math]::Round($os.FreePhysicalMemory / 1KB, 0)

    # 1. Handle leak detection
    $handleLeakers = Get-Process | Where-Object { $_.HandleCount -gt 100000 } |
        Sort-Object HandleCount -Descending
    foreach ($leaker in $handleLeakers) {
        $recommendations += [PSCustomObject]@{
            Priority          = 1
            Category          = 'handle_leak'
            Description       = "$($leaker.ProcessName) (PID $($leaker.Id)) has $($leaker.HandleCount) handles and $([math]::Round($leaker.PrivateMemorySize64/1GB,1)) GB private memory. This is a single high-handle snapshot; collect interval evidence before drawing a causal conclusion."
            EstimatedSavingsMB = $null
            Action            = "restart_process:$($leaker.ProcessName)"
            SafeToAuto        = $false
        }
    }

    # 2. Pool nonpaged analysis
    try {
        $sample=Get-Counter '\Memory\Pool Nonpaged Bytes' -ErrorAction Stop
        if (@($sample.CounterSamples).Count -ne 1 -or -not (Test-RecommendationNumber $sample.CounterSamples[0].CookedValue)) { throw 'Pool counter invalid.' }
        if (-not (Test-RecommendationNumber $sample.CounterSamples[0].Status) -or $sample.CounterSamples[0].Status -notin @(0,1)) { throw 'Pool counter sample status invalid.' }
        $poolNP=$sample.CounterSamples[0].CookedValue
        $poolNP_GB = [math]::Round($poolNP / 1GB, 1)
        if ($poolNP_GB -gt 4) {
            $recommendations += [PSCustomObject]@{
                Priority          = 1
                Category          = 'pool_nonpaged'
                Description       = "Observed nonpaged pool: $poolNP_GB GB. This snapshot does not identify a leak or driver cause; review interval measurements."
                EstimatedSavingsMB = $null
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
            Description       = "$($orphanCmds.Count) cmd/conhost processes have absent parent PIDs in this snapshot, using ~$orphanMB MB working set. Ownership and disposability are unverified; review manually."
            EstimatedSavingsMB = $null
            Action            = 'kill_orphan_terminals'
            SafeToAuto        = $false
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
                Description       = "$($browser.Key) has $($browser.Value.Count) processes using ~$totalMB MB. Process count does not establish tab count or waste; review workload needs manually."
                EstimatedSavingsMB = $null
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
            $recommendations += [PSCustomObject]@{
                Priority          = 3
                Category          = 'wsl_memory'
                Description       = "Observed WSL2 VM private memory: $wslPrivateMB MB. Review workload requirements manually before considering configuration changes."
                EstimatedSavingsMB = $null
                Action            = 'tune_wsl_config'
                SafeToAuto        = $false
            }
        }
    }

    # 6. Paging rate
    try {
        $sample=Get-Counter '\Memory\Pages/sec' -ErrorAction Stop
        if (@($sample.CounterSamples).Count -ne 1 -or -not (Test-RecommendationNumber $sample.CounterSamples[0].CookedValue)) { throw 'Paging counter invalid.' }
        if (-not (Test-RecommendationNumber $sample.CounterSamples[0].Status) -or $sample.CounterSamples[0].Status -notin @(0,1)) { throw 'Paging counter sample status invalid.' }
        $pagesSec=$sample.CounterSamples[0].CookedValue
        if ($pagesSec -gt 1000) {
            $recommendations += [PSCustomObject]@{
                Priority          = 1
                Category          = 'excessive_paging'
                Description       = "Observed paging: $([math]::Round($pagesSec,0)) pages/sec. Available memory: $availMB MB. A single sample does not establish thrashing; review interval evidence."
                EstimatedSavingsMB = $null
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
            EstimatedSavingsMB = $null
            Action            = "investigate_process:$($_.ProcessName):$($_.Id)"
            SafeToAuto        = $false
        }
    }

    foreach ($item in $recommendations) {
        $item | Add-Member -NotePropertyName Source -NotePropertyValue 'PowerShell-Fallback'
        $item | Add-Member -NotePropertyName MeasurementStatus -NotePropertyValue 'SnapshotObservationOnly'
    }
    $sorted = @($recommendations | Sort-Object Priority,Category)

    if ($AsJson) {
        return ConvertTo-Json -InputObject @($sorted) -Depth 5
    }
    return $sorted
}
