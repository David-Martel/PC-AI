#requires -Version 7.0
<#
.SYNOPSIS
Launch the detached GitOps worker without blocking the pre-push hook.
.DESCRIPTION
Launch rejection is visible, but the hook retains exit zero. The worker owns its
per-checkout lease and finite request deadlines; do not put it in a kill-on-close job.
#>
[CmdletBinding()]
param()
try {
    $root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
    $monitor = Join-Path $PSScriptRoot 'Invoke-GitOpsMonitors.ps1'
    $executable = Join-Path $PSHOME $(if ($IsWindows) { 'pwsh.exe' } else { 'pwsh' })
    if (-not $IsWindows) { throw 'This detached hook launcher requires Windows.' }
    $command = '"{0}" -NoProfile -WindowStyle Hidden -File "{1}" -RepoRoot "{2}"' -f $executable, $monitor, $root
    # The WMI provider does not inherit process-only routing or credential variables.
    # Pass a sorted Unicode environment in memory; never put its values in logs or arguments.
    $environment = [Environment]::GetEnvironmentVariables('Process')
    $keys = [string[]]@($environment.Keys)
    [Array]::Sort($keys, [StringComparer]::OrdinalIgnoreCase)
    $variables = [string[]]@($keys | ForEach-Object { $_ + '=' + [string]$environment[$_] })
    # Unicode + breakaway. ShowWindow keeps the console hidden; DETACHED_PROCESS
    # did not execute the qualified PowerShell fixture on this host.
    $startup = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ EnvironmentVariables = $variables; CreateFlags = [uint32]16778240; ShowWindow = [uint16]0 }
    try {
        $result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $command; CurrentDirectory = $root; ProcessStartupInformation = $startup } -ErrorAction Stop
    }
    finally {
        $startup = $null; $variables = $null; $environment = $null
    }
    if ($null -eq $result -or $result.ReturnValue -ne 0 -or $result.ProcessId -le 0) {
        Write-Warning 'GitOps worker launch was rejected; the push continues.'
    }
}
catch {
    Write-Warning 'GitOps worker launch failed; the push continues.'
}
exit 0
