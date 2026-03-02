#Requires -Version 5.1
<#
.SYNOPSIS
    Wrapper for the WSL/Docker health check script in C:\Scripts\Startup.
#>
function Invoke-WSLDockerHealthCheck {
    [CmdletBinding()]
    [OutputType([PSCustomObject])]
    param(
        [Parameter()]
        [string]$ScriptPath = 'C:\Scripts\Startup\wsl-docker-health-check.ps1',

        [Parameter()]
        [switch]$AutoRecover,

        [Parameter()]
        [switch]$Quick
    )

    $scriptExists = Test-Path $ScriptPath

    $args = @()
    $verboseRequested = $PSBoundParameters.ContainsKey('Verbose')
    if ($AutoRecover) { $args += '-AutoRecover' }
    if ($verboseRequested) { $args += '-Verbose' }
    if ($Quick) { $args += '-Quick' }

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

    $fallbackExitCode = 0
    $fallbackOutput = $null
    try {
        $fallbackResult = Get-WSLEnvironmentHealth -AutoRecover:$AutoRecover -Quick:$Quick
        $fallbackOutput = $fallbackResult | ConvertTo-Json -Depth 6
    }
    catch {
        $fallbackExitCode = 1
        $fallbackOutput = "Fallback execution failed: $($_.Exception.Message)"
    }

    return [PSCustomObject]@{
        ScriptPath   = $ScriptPath
        Arguments    = $args
        ExitCode     = $fallbackExitCode
        Output       = $fallbackOutput
        Success      = ($fallbackExitCode -eq 0)
        Fallback     = $true
        FallbackMode = 'Get-WSLEnvironmentHealth'
    }
}
