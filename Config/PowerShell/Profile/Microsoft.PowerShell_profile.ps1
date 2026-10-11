# Keep SSH command and subsystem sessions silent for binary transfer protocols.
if ($env:SSH_CONNECTION -and [Environment]::CommandLine -match '(?i)(?:^|\s)-(?:command|c|encodedcommand|ec|file|f|noninteractive)\b') {
    return
}

# Avoid PSReadLine's background key reader after console input has been lost.
# PSReadLine can already be loaded by ConsoleHost before this profile runs.
$psReadLineOptOut = $env:PCAI_DISABLE_PSREADLINE
if ($null -eq $psReadLineOptOut) {
    # Existing terminal parents may not have inherited a new persisted setting.
    $psReadLineOptOut = [Environment]::GetEnvironmentVariable('PCAI_DISABLE_PSREADLINE', 'User')
}
if ($psReadLineOptOut -eq '1' -or [Console]::IsInputRedirected) {
    try {
        if (Microsoft.PowerShell.Core\Get-Module -Name PSReadLine) {
            Microsoft.PowerShell.Core\Remove-Module -Name PSReadLine -ErrorAction Stop
        }
    } catch {
        Microsoft.PowerShell.Utility\Write-Warning "Could not remove PSReadLine for console safety: $_"
    }

    if ($Host.Name -eq 'ConsoleHost') {
        # This documented host extension bypasses module command auto-loading.
        function global:PSConsoleHostReadLine {
            [Console]::ReadLine()
        }
    }
}

# pc_accel shim: keep the cloud profile path tiny and load the real profile locally.

$localProfileRoot = Microsoft.PowerShell.Management\Join-Path $HOME '.config\powershell'
$localModulesRoot = Microsoft.PowerShell.Management\Join-Path $HOME 'Documents\PowerShell\Modules'
$localProfile = Microsoft.PowerShell.Management\Join-Path $localProfileRoot 'Microsoft.PowerShell_profile.ps1'

$env:POWERSHELL_PROFILE_ROOT = $localProfileRoot
$env:POWERSHELL_MODULES_PATH = $localModulesRoot

if (-not $env:PROFILEUTILITIES_PCAI_AUTO_PATH_OPTIMIZE) {
    $env:PROFILEUTILITIES_PCAI_AUTO_PATH_OPTIMIZE = '0'
}
if (-not $env:PS_SKIP_PROFILE_ACCELERATOR) {
    $env:PS_SKIP_PROFILE_ACCELERATOR = '1'
}
if (-not $env:PS_SKIP_PSMODULEPATH_OPTIMIZE) {
    $env:PS_SKIP_PSMODULEPATH_OPTIMIZE = '1'
}

if (Microsoft.PowerShell.Management\Test-Path -LiteralPath $localProfile) {
    . $localProfile
    return
}

# Qualify built-ins: wildcard-export module manifests can otherwise cause
# command discovery to import unrelated modules during early profile startup.
Microsoft.PowerShell.Utility\Write-Warning "Canonical local PowerShell profile not found: $localProfile"
