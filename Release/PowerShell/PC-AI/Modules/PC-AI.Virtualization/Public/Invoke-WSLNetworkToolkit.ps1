#Requires -Version 5.1
<#
.SYNOPSIS
    Wrapper for the external WSL network toolkit script.

.PARAMETER ScriptPath
    Path to the WSL network recovery script (default: C:\Scripts\wsl-network-recovery.ps1)

.PARAMETER Distribution
    WSL distribution name (passed through)

.PARAMETER Check
    Run basic connectivity checks

.PARAMETER Diagnose
    Run detailed diagnostics

.PARAMETER Repair
    Run repair sequence

.PARAMETER Full
    Run full repair sequence (including HNS reset)
#>
function Invoke-WSLNetworkToolkit {
    [CmdletBinding()]
    [OutputType([PSCustomObject])]
    param(
        [Parameter()]
        [string]$ScriptPath = 'C:\Scripts\wsl-network-recovery.ps1',

        [Parameter()]
        [ValidateSet('check','diagnose','repair','full')]
        [string]$Mode,

        [Parameter()]
        [string]$Distribution,

        [Parameter()]
        [switch]$Check,

        [Parameter()]
        [switch]$Diagnose,

        [Parameter()]
        [switch]$Repair,

        [Parameter()]
        [switch]$Full,

        [Parameter()]
        [switch]$Optimize,

        [Parameter()]
        [switch]$ApplyConfig,

        [Parameter()]
        [switch]$TestNetworkingMode,

        [Parameter()]
        [ValidateSet('nat','mirrored','virtioproxy')]
        [string]$NetworkingMode,

        [Parameter()]
        [switch]$FixDns,

        [Parameter()]
        [switch]$RestartWsl,

        [Parameter()]
        [switch]$ResetAdapters,

        [Parameter()]
        [switch]$ResetWinsock,

        [Parameter()]
        [switch]$RestartHns,

        [Parameter()]
        [switch]$RestartWslService,

        [Parameter()]
        [switch]$DisableVmqOnWsl,

        [Parameter()]
        [switch]$Force
    )

    if ($Mode) {
        $Check = $false
        $Diagnose = $false
        $Repair = $false
        $Full = $false

        switch ($Mode) {
            'check' { $Check = $true }
            'diagnose' { $Diagnose = $true }
            'repair' { $Repair = $true }
            'full' { $Full = $true }
        }
    }

    $scriptExists = Test-Path $ScriptPath

    $args = @()
    if ($Distribution) { $args += @('-Distro', $Distribution) }

    if ($Check -or $Diagnose -or $Repair -or $Full -or $Optimize -or $ApplyConfig -or $TestNetworkingMode) {
        if ($Check) { $args += '-Check' }
        if ($Diagnose) { $args += '-Diagnose' }
        if ($Repair) { $args += '-Repair' }
        if ($Full) { $args += '-Full' }
        if ($Optimize) { $args += '-Optimize' }
        if ($ApplyConfig) { $args += '-ApplyConfig' }
        if ($TestNetworkingMode) { $args += '-TestNetworkingMode' }
    }
    else {
        $args += '-Check'
    }

    if ($NetworkingMode) { $args += @('-NetworkingMode', $NetworkingMode) }
    if ($FixDns) { $args += '-FixDns' }
    if ($RestartWsl) { $args += '-RestartWsl' }
    if ($ResetAdapters) { $args += '-ResetAdapters' }
    if ($ResetWinsock) { $args += '-ResetWinsock' }
    if ($RestartHns) { $args += '-RestartHns' }
    if ($RestartWslService) { $args += '-RestartWslService' }
    if ($DisableVmqOnWsl) { $args += '-DisableVmqOnWsl' }
    if ($Force) { $args += '-Force' }

    if ($scriptExists) {
        $output = & powershell -NoProfile -ExecutionPolicy Bypass -File $ScriptPath @args 2>&1
        $exitCode = $LASTEXITCODE

        return [PSCustomObject]@{
            ScriptPath = $ScriptPath
            Arguments  = $args
            ExitCode   = $exitCode
            Output     = ($output | Out-String).Trim()
            Success    = ($exitCode -eq 0)
            Fallback   = $false
        }
    }

    $requestedFlags = @()
    if ($Check) { $requestedFlags += 'Check' }
    if ($Diagnose) { $requestedFlags += 'Diagnose' }
    if ($Repair) { $requestedFlags += 'Repair' }
    if ($Full) { $requestedFlags += 'Full' }
    if ($Optimize) { $requestedFlags += 'Optimize' }
    if ($ApplyConfig) { $requestedFlags += 'ApplyConfig' }
    if ($TestNetworkingMode) { $requestedFlags += 'TestNetworkingMode' }
    if ($FixDns) { $requestedFlags += 'FixDns' }
    if ($RestartWsl) { $requestedFlags += 'RestartWsl' }
    if ($ResetAdapters) { $requestedFlags += 'ResetAdapters' }
    if ($ResetWinsock) { $requestedFlags += 'ResetWinsock' }
    if ($RestartHns) { $requestedFlags += 'RestartHns' }
    if ($RestartWslService) { $requestedFlags += 'RestartWslService' }
    if ($DisableVmqOnWsl) { $requestedFlags += 'DisableVmqOnWsl' }

    if ($requestedFlags.Count -eq 0) {
        $requestedFlags += 'Check'
    }

    $unsupported = @()
    if ($Optimize -or $ApplyConfig -or $TestNetworkingMode -or $FixDns -or $ResetAdapters -or $ResetWinsock -or $RestartHns -or $RestartWslService -or $DisableVmqOnWsl -or $NetworkingMode) {
        $unsupported = @(
            if ($Optimize) { 'Optimize' }
            if ($ApplyConfig) { 'ApplyConfig' }
            if ($TestNetworkingMode) { 'TestNetworkingMode' }
            if ($FixDns) { 'FixDns' }
            if ($ResetAdapters) { 'ResetAdapters' }
            if ($ResetWinsock) { 'ResetWinsock' }
            if ($RestartHns) { 'RestartHns' }
            if ($RestartWslService) { 'RestartWslService' }
            if ($DisableVmqOnWsl) { 'DisableVmqOnWsl' }
            if ($NetworkingMode) { "NetworkingMode=$NetworkingMode" }
        ) | Where-Object { $_ }
    }

    $fallbackResult = $null
    $fallbackOutput = $null
    $fallbackExitCode = 0
    try {
        if ($Repair -or $Full) {
            $fallbackResult = Repair-WSLNetworking -RestartWSL:$RestartWsl -Force:$Force
        }
        else {
            $quick = $Check -and -not $Diagnose
            if ($Distribution) {
                $fallbackResult = Get-WSLEnvironmentHealth -Quick:$quick -Distribution $Distribution
            }
            else {
                $fallbackResult = Get-WSLEnvironmentHealth -Quick:$quick
            }
        }
        $fallbackOutput = $fallbackResult | ConvertTo-Json -Depth 6
    }
    catch {
        $fallbackExitCode = 1
        $fallbackOutput = "Fallback execution failed: $($_.Exception.Message)"
    }

    return [PSCustomObject]@{
        ScriptPath      = $ScriptPath
        Arguments       = $args
        ExitCode        = $fallbackExitCode
        Output          = $fallbackOutput
        Success         = ($fallbackExitCode -eq 0)
        Fallback        = $true
        FallbackMode    = if ($Repair -or $Full) { 'Repair-WSLNetworking' } else { 'Get-WSLEnvironmentHealth' }
        RequestedFlags  = $requestedFlags
        Unsupported     = $unsupported
    }
}
