#Requires -Version 7.0

if (-not $script:PcaiPerfWorkerState) {
    $script:PcaiPerfWorkerState = [PSCustomObject]@{
        Process  = $null
        Input    = $null
        Output   = $null
        ToolPath = $null
        LastUse  = [datetime]::MinValue
    }
}

function Test-PcaiPerfWorkerHealthy {
    [CmdletBinding()]
    param(
        [string]$ToolPath
    )

    $state = $script:PcaiPerfWorkerState
    if (-not $state -or -not $state.Process) {
        return $false
    }

    if ($state.Process.HasExited) {
        return $false
    }

    if ($ToolPath -and $state.ToolPath -and -not [string]::Equals($ToolPath, $state.ToolPath, [System.StringComparison]::OrdinalIgnoreCase)) {
        return $false
    }

    return ($state.Input -and $state.Output)
}

function Stop-PcaiPerfWorker {
    [CmdletBinding()]
    param()

    $state = $script:PcaiPerfWorkerState
    if (-not $state) {
        return
    }

    if ($state.Input) {
        try { $state.Input.Dispose() } catch {}
    }

    if ($state.Output) {
        try { $state.Output.Dispose() } catch {}
    }

    if ($state.Process) {
        try {
            if (-not $state.Process.HasExited) {
                $state.Process.Kill($true)
                $state.Process.WaitForExit(1000) | Out-Null
            }
        } catch {}
        try { $state.Process.Dispose() } catch {}
    }

    $script:PcaiPerfWorkerState = [PSCustomObject]@{
        Process  = $null
        Input    = $null
        Output   = $null
        ToolPath = $null
        LastUse  = [datetime]::MinValue
    }
}

function Start-PcaiPerfWorker {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$ToolPath
    )

    if (Test-PcaiPerfWorkerHealthy -ToolPath $ToolPath) {
        $script:PcaiPerfWorkerState.LastUse = Get-Date
        return $script:PcaiPerfWorkerState
    }

    Stop-PcaiPerfWorker

    $startInfo = [System.Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = $ToolPath
    $startInfo.Arguments = 'worker'
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardInput = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.StandardOutputEncoding = [System.Text.Encoding]::UTF8
    $startInfo.StandardInputEncoding = [System.Text.Encoding]::UTF8

    $process = [System.Diagnostics.Process]::new()
    $process.StartInfo = $startInfo

    if (-not $process.Start()) {
        throw "Failed to start pcai-perf worker: $ToolPath"
    }

    $script:PcaiPerfWorkerState = [PSCustomObject]@{
        Process  = $process
        Input    = $process.StandardInput
        Output   = $process.StandardOutput
        ToolPath = $ToolPath
        LastUse  = Get-Date
    }

    return $script:PcaiPerfWorkerState
}

function Invoke-PcaiPerfWorkerRequest {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$ToolPath,
        [Parameter(Mandatory)]
        [string]$Command,
        [hashtable]$Payload = @{}
    )

    $state = Start-PcaiPerfWorker -ToolPath $ToolPath
    $request = @{ command = $Command }
    foreach ($entry in $Payload.GetEnumerator()) {
        $request[$entry.Key] = $entry.Value
    }

    $requestJson = $request | ConvertTo-Json -Compress -Depth 8

    try {
        $state.Input.WriteLine($requestJson)
        $state.Input.Flush()
        $responseLine = $state.Output.ReadLine()
        if ([string]::IsNullOrWhiteSpace($responseLine)) {
            Stop-PcaiPerfWorker
            return $null
        }

        $response = $responseLine | ConvertFrom-Json -Depth 8
        if (-not $response.ok) {
            throw ($response.error ?? 'pcai-perf worker request failed')
        }

        $script:PcaiPerfWorkerState.LastUse = Get-Date
        return $response.result
    } catch {
        Stop-PcaiPerfWorker
        throw
    }
}
