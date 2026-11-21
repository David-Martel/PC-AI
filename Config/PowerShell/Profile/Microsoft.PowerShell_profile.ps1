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

# Fast paths. A Claude agent-home (...\agent-homes\<id>\home) is an isolated sandbox with no private
# profile; PCAI_PROFILE_FAST=1 is the explicit opt-out for any other tool that wants a bare shell
# without passing -NoProfile. Both return before touching the environment, the disk or the network.
if ($env:PCAI_PROFILE_FAST -eq '1' -or $HOME -match '[\\/]agent-homes[\\/]') {
    return
}

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
}
else {
    # Qualify built-ins: wildcard-export module manifests can otherwise cause
    # command discovery to import unrelated modules during early profile startup.
    Microsoft.PowerShell.Utility\Write-Warning "Canonical local PowerShell profile not found: $localProfile"
}

# Lazy WinGet "command not found" suggestions, interactive shells only. The PowerToys installer
# appends an eager `Import-Module Microsoft.WinGet.CommandNotFound` after this bootstrap. That import
# cost 1.4-31 s (the module is a cloud-only OneDrive file) and throws where winget is missing, so
# every path ends in `return` and the appended block is unreachable. This one-shot hook
# restores the feature: it imports the module on the first unresolved command, and only if winget
# is on PATH. The imported module registers its own feedback provider. It is registered AFTER the
# private profile has run, and only if that profile left no handler: command discovery misses during
# private startup (a Get-Command probe) would otherwise fire it and pay the import every start.
if ($Host.Name -eq 'ConsoleHost' -and -not [Console]::IsInputRedirected -and [Environment]::CommandLine -notmatch '(?i)(?:^|\s)-noni' -and $null -eq $ExecutionContext.InvokeCommand.CommandNotFoundAction) {
    $ExecutionContext.InvokeCommand.CommandNotFoundAction = {
        param($CommandName, $CommandLookupEventArgs)
        $ExecutionContext.InvokeCommand.CommandNotFoundAction = $null
        try {
            if (Microsoft.PowerShell.Core\Get-Command -Name winget -CommandType Application -ErrorAction Ignore) {
                Microsoft.PowerShell.Core\Import-Module -Name Microsoft.WinGet.CommandNotFound -ErrorAction Ignore
            }
        } catch {
            $null = $_
        }
    }
}

return
