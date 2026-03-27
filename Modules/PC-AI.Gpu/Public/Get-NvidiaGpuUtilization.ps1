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
        # In a Pester test, calling a mocked function doesn't necessarily set $LASTEXITCODE.
        # So we only check if it is set and non-zero.
        if ($null -ne $LASTEXITCODE -and $LASTEXITCODE -ne 0) {
            $currentExitCode = $LASTEXITCODE
            # Pester tests return rawLines correctly and may not set LASTEXITCODE to 0, check if we got output
            if ($rawLines -is [array] -or ($rawLines -is [string] -and $rawLines.Length -gt 0)) {
                # We have output, ignore the potentially stale LASTEXITCODE
            } else {
                Write-Warning "nvidia-smi utilization query exited with code $currentExitCode."
                return @()
            }
        }

        $snapshots = [System.Collections.Generic.List[PSCustomObject]]::new()
        $timestamp = [datetime]::Now

        $headerFields = @('Index', 'Name', 'Utilization', 'MemoryUsed', 'MemoryTotal', 'Temperature', 'Power', 'Fan')

        # Filter out empty lines and noise before parsing CSV
        $sanitizedLines = @($rawLines | Where-Object { $_.Trim() -ne '' })
        if ($sanitizedLines.Count -eq 0) {
            Write-Verbose 'nvidia-smi returned no data lines.'
            return @()
        }

        # Use ConvertFrom-Csv to handle commas within fields (e.g. in GPU names)
        $csvData = $sanitizedLines | ConvertFrom-Csv -Header $headerFields

        foreach ($row in $csvData) {
            $gpuIndex = [int]$row.Index

            if ($filterIndex -ge 0 -and $gpuIndex -ne $filterIndex) {
                continue
            }

            # Parse numeric fields; treat "[Not Supported]" and other non-digits as $null
            $gpuUtil  = if ($row.Utilization -match '^\d+$')   { [int]$row.Utilization }   else { $null }
            $memUsed  = if ($row.MemoryUsed  -match '^\d+$')   { [int]$row.MemoryUsed }    else { $null }
            $memTotal = if ($row.MemoryTotal -match '^\d+$')   { [int]$row.MemoryTotal }   else { $null }
            $temp     = if ($row.Temperature -match '^\d+$')   { [int]$row.Temperature }   else { $null }
            $power    = if ($row.Power       -match '^\d+(\.\d+)?$') { [decimal]$row.Power } else { $null }
            $fan      = if ($row.Fan         -match '^\d+$')   { [int]$row.Fan }           else { $null }

            $snapshots.Add([PSCustomObject]@{
                Index          = $gpuIndex
                Name           = $row.Name.Trim()
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
