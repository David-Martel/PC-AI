<#
.SYNOPSIS
    Establishes a validated Bitwarden session through the installed machine backend.
.DESCRIPTION
    Reuses the protected, bounded machine bootstrap rather than handling credentials
    or invoking Bitwarden directly. No vault sync, export, or machine-cache hydration
    is performed. Dot-sourcing defines the functions without initializing secrets.
.PARAMETER Quiet
    Suppresses informational output; failures still produce a sanitized warning.
.PARAMETER Force
    Requests a fresh session from the validated backend.
.PARAMETER PersistSession
    Persists only the validated process session as a User environment variable.
.PARAMETER DryRun
    Describes the operation without loading backends or accessing credentials.
.PARAMETER Help
    Displays usage without loading backends or accessing credentials.
#>
[CmdletBinding(SupportsShouldProcess)]
param(
    [switch]$Quiet,
    [switch]$Force,
    [switch]$PersistSession,
    [switch]$DryRun,
    [Alias('h', '-help')][switch]$Help
)

function Set-BwVaultUserSession {
    [CmdletBinding(SupportsShouldProcess)]
    param([Parameter(Mandatory)][string]$Session)
    if ($PSCmdlet.ShouldProcess('User BW_SESSION', 'Persist validated session')) {
        [Environment]::SetEnvironmentVariable('BW_SESSION', $Session, 'User')
    }
}

function Invoke-BwVaultUnlock {
    [CmdletBinding(SupportsShouldProcess)]
    [OutputType([int])]
    param(
        [switch]$Quiet,
        [switch]$Force,
        [switch]$PersistSession,
        [switch]$DryRun
    )

    if ($DryRun) {
        if (-not $Quiet) { Write-Information 'Would establish a validated Bitwarden session.' -InformationAction Continue }
        return 0
    }
    if (-not $PSCmdlet.ShouldProcess('Bitwarden session', 'Initialize through the validated machine backend')) { return 0 }

    $previousSession = $env:BW_SESSION
    try {
        if (-not (Get-Command Initialize-BitwardenSessionFromBackends -CommandType Function -ErrorAction SilentlyContinue)) {
            $backendPath = Join-Path $env:USERPROFILE '.machine\SecretBackendUtilities.ps1'
            . $backendPath *> $null
        }
        # Backend diagnostics may contain credential values: expose only our fixed messages.
        $result = Initialize-BitwardenSessionFromBackends -Refresh:$Force -Quiet -ErrorAction Stop 2>$null 3>$null 4>$null 5>$null 6>$null
        if ($null -eq $result -or $result.Success -isnot [bool] -or -not $result.Success -or
            [string]::IsNullOrWhiteSpace($env:BW_SESSION)) {
            throw 'The backend did not establish a validated session.'
        }
        if ($PersistSession) { Set-BwVaultUserSession -Session $env:BW_SESSION -ErrorAction Stop }
        if (-not $Quiet) { Write-Information 'Bitwarden session validated.' -InformationAction Continue }
        return 0
    }
    catch {
        $env:BW_SESSION = $previousSession
        $PSCmdlet.WriteWarning('Bitwarden session initialization failed; no session was persisted by this launcher.')
        return 1
    }
}

if ($MyInvocation.InvocationName -ne '.') {
    if ($Help) {
        Write-Output 'Usage: Unlock-BwVault.ps1 [-Quiet] [-Force] [-PersistSession] [-DryRun] [-WhatIf] [-Help|-h]'
        exit 0
    }
    $exitCode = Invoke-BwVaultUnlock -Quiet:$Quiet -Force:$Force -PersistSession:$PersistSession -DryRun:$DryRun -WhatIf:$WhatIfPreference
    exit $exitCode
}
