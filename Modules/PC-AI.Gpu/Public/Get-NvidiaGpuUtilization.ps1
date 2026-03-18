#Requires -Version 5.1
<#
.SYNOPSIS
    Returns a real-time utilization snapshot for each installed NVIDIA GPU.

.DESCRIPTION
    Queries nvidia-smi.exe with CSV output to capture the current utilization,
    memory usage, temperature, power draw, and fan speed for all GPUs (or a
    specific GPU when -Index is supplied).

    This function is optimized for polling scenarios: it performs a single
    nvidia-smi invocation and returns all GPU data in one call. Use -Continuous
    with -IntervalSeconds to run a polling loop until the user presses Ctrl+C.

    When nvidia-smi is unavailable the function writes a warning and returns an
    empty array; there is no CIM fallback for real-time utilization metrics.

.PARAMETER Index
    When specified, return utilization for only the GPU at this nvidia-smi index
    (0-based). When omitted all GPUs are returned.

.PARAMETER Continuous
    When specified, poll nvidia-smi repeatedly at -IntervalSeconds intervals and
    write each snapshot to the pipeline. Press Ctrl+C to stop.

.PARAMETER IntervalSeconds
    Polling interval in seconds when -Continuous is active. Minimum 1, default 2.

.OUTPUTS
    PSCustomObject[] with properties:
      Index          - nvidia-smi GPU index (integer)
      Name           - GPU display name
      GpuUtilization - Compute utilization percentage (integer, 0-100)
      MemoryUsedMB   - Currently used VRAM in MiB (integer)
      MemoryTotalMB  - Total VRAM in MiB (integer)
      Temperature    - GPU core temperature in degrees Celsius (integer)
      PowerDraw      - Current power draw in watts (decimal), or $null
      FanSpeed       - Fan speed percentage (integer), or $null
      Timestamp      - DateTime of this snapshot

.EXAMPLE
    Get-NvidiaGpuUtilization
    Returns a single snapshot of all GPU utilization metrics.

.EXAMPLE
    Get-NvidiaGpuUtilization -Index 1
    Returns current utilization for GPU at index 1.

.EXAMPLE
    Get-NvidiaGpuUtilization -Continuous -IntervalSeconds 5
    Polls and outputs GPU utilization every 5 seconds until Ctrl+C.
#>
function Get-NvidiaGpuUtilization {
    [CmdletBinding()]
    [OutputType([PSCustomObject[]])]
    param(
        [Parameter()]
        [ValidateRange(0, 15)]
        [int]$Index = -1,

        [Parameter()]
        [switch]$Continuous,

        [Parameter()]
        [ValidateRange(1, 3600)]
        [int]$IntervalSeconds = 2
    )

    $ErrorActionPreference = 'Stop'

    # --- Locate nvidia-smi ---
    $nvidiaSmi = $null
    $smiCandidates = @(
        'nvidia-smi.exe',
        'C:\Windows\System32\nvidia-smi.exe',
        'C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe'
    )

    foreach ($candidate in $smiCandidates) {
        if (Get-Command -Name $candidate -ErrorAction SilentlyContinue) {
            $nvidiaSmi = $candidate
            break
        }
        if (Test-Path $candidate -ErrorAction SilentlyContinue) {
            $nvidiaSmi = $candidate
            break
        }
    }

    if (-not $nvidiaSmi) {
        Write-Warning 'nvidia-smi.exe not found. Real-time GPU utilization is unavailable without the NVIDIA driver.'
        return @()
    }

    # Helper scriptblock — performs one query and returns result objects
    $queryBlock = {
        param($smi, $filterIndex)

        $queryFields = 'index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,fan.speed'
        $smiArgs     = @(
            ('--query-gpu={0}' -f $queryFields),
            '--format=csv,noheader,nounits'
        )

        $rawLines = & $smi @smiArgs 2>&1
        if ($null -ne $LASTEXITCODE -and $LASTEXITCODE -ne 0) {
            Write-Warning "nvidia-smi utilization query exited with code $LASTEXITCODE."
            return @()
        }

        $snapshots = [System.Collections.Generic.List[PSCustomObject]]::new()
        $timestamp = [datetime]::Now

        foreach ($line in $rawLines) {
            $line = $line.Trim()
            if ($line -eq '') { continue }

            $fields = $line -split ',\s*'
            if ($fields.Count -lt 8) {
                Write-Verbose "Skipping malformed nvidia-smi line: $line"
                continue
            }

            $gpuIndex        = [int]($fields[0].Trim())
            $gpuName         = $fields[1].Trim()
            $utilizationRaw  = $fields[2].Trim()
            $memUsedRaw      = $fields[3].Trim()
            $memTotalRaw     = $fields[4].Trim()
            $tempRaw         = $fields[5].Trim()
            $powerRaw        = $fields[6].Trim()
            $fanRaw          = $fields[7].Trim()

            if ($filterIndex -ge 0 -and $gpuIndex -ne $filterIndex) {
                continue
            }

            # Parse numeric fields; treat "[Not Supported]" and empty strings as $null
            $gpuUtil  = if ($utilizationRaw -match '^\d+$')   { [int]$utilizationRaw }   else { $null }
            $memUsed  = if ($memUsedRaw     -match '^\d+$')   { [int]$memUsedRaw }     else { $null }
            $memTotal = if ($memTotalRaw    -match '^\d+$')   { [int]$memTotalRaw }    else { $null }
            $temp     = if ($tempRaw        -match '^\d+$')   { [int]$tempRaw }        else { $null }
            $power    = if ($powerRaw       -match '^\d+(\.\d+)?$') { [decimal]$powerRaw } else { $null }
            $fan      = if ($fanRaw         -match '^\d+$')   { [int]$fanRaw }         else { $null }

            $snapshots.Add([PSCustomObject]@{
                Index          = $gpuIndex
                Name           = $gpuName
                GpuUtilization = $gpuUtil
                MemoryUsedMB   = $memUsed
                MemoryTotalMB  = $memTotal
                Temperature    = $temp
                PowerDraw      = $power
                FanSpeed       = $fan
                Timestamp      = $timestamp
            })
        }

        return @($snapshots)
    }

    try {
        if ($Continuous) {
            Write-Verbose "Starting continuous GPU utilization polling every $IntervalSeconds second(s). Press Ctrl+C to stop."
            while ($true) {
                $snapshot = & $queryBlock $nvidiaSmi $Index
                foreach ($item in $snapshot) {
                    Write-Output $item
                }
                Start-Sleep -Seconds $IntervalSeconds
            }
        }
        else {
            return & $queryBlock $nvidiaSmi $Index
        }
    }
    catch [System.Management.Automation.PipelineStoppedException] {
        # Ctrl+C in -Continuous mode — clean exit
        Write-Verbose 'GPU utilization polling stopped by user.'
    }
    catch {
        Write-Error "Get-NvidiaGpuUtilization failed: $($_.Exception.Message)"
        return @()
    }
}
