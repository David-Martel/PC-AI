#Requires -Version 5.1

function Invoke-NukeNulCleanup {
    <#
    .SYNOPSIS
        Runs the NukeNul reserved filename cleanup tool and returns JSON results.

    .DESCRIPTION
        Locates NukeNul.exe (Rust DLL + C# CLI hybrid) and executes a scan/delete
        against the target path. Parses JSON output for structured results.

    .PARAMETER Path
        Target directory to scan (default: current directory).

    .PARAMETER ExePath
        Optional explicit path to NukeNul.exe.

    .PARAMETER Force
        Skip interactive confirmation.

    .OUTPUTS
        PSCustomObject with parsed results and execution metadata.
    #>
    [CmdletBinding(SupportsShouldProcess = $true)]
    param(
        [Parameter(Position = 0)]
        [string]$Path = (Get-Location).Path,

        [Parameter()]
        [ValidateScript({ Test-Path $_ -PathType Leaf })]
        [string]$ExePath,

        [Parameter()]
        [switch]$Force
    )

    Set-StrictMode -Version Latest

    $repoRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
    $nativeRoot = Join-Path $repoRoot 'Native\NukeNul'

    $candidates = @(
        $ExePath,
        (Join-Path $env:USERPROFILE 'bin\NukeNul.exe'),
        (Join-Path $nativeRoot 'bin\Release\net8.0\win-x64\NukeNul.exe'),
        (Join-Path $nativeRoot 'bin\Release\net8.0\win-x64\publish\NukeNul.exe')
    ) | Where-Object { $_ -and (Test-Path $_ -PathType Leaf) }

    if (-not $candidates -or $candidates.Count -eq 0) {
        throw "NukeNul.exe not found. Build or install it in $env:USERPROFILE\bin or $nativeRoot."
    }

    $exe = $candidates[0]
    $dll = Join-Path (Split-Path $exe -Parent) 'nuker_core.dll'
    if (-not (Test-Path $dll -PathType Leaf)) {
        throw "nuker_core.dll not found next to $exe"
    }

    if (-not (Test-Path $Path)) {
        throw "Target path not found: $Path"
    }

    $action = "NukeNul cleanup on $Path"
    if (-not $Force) {
        if (-not $PSCmdlet.ShouldProcess($Path, $action)) {
            return [PSCustomObject]@{
                Status = 'Skipped'
                Path = $Path
                ExePath = $exe
            }
        }
    }

    $raw = & $exe $Path 2>&1
    $exitCode = $LASTEXITCODE

    $parsed = $null
    try {
        $parsed = $raw | ConvertFrom-Json -ErrorAction Stop
    }
    catch {
        $parsed = $null
    }

    return [PSCustomObject]@{
        Status = if ($exitCode -eq 0) { 'Success' } else { 'Error' }
        ExitCode = $exitCode
        Path = $Path
        ExePath = $exe
        Raw = $raw
        Parsed = $parsed
    }
}
