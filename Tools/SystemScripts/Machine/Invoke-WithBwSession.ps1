<#
.SYNOPSIS
    Runs one command with a Bitwarden session scoped to this process, then drops the session.
.DESCRIPTION
    Unlocks through Unlock-BwVault.ps1 (machine backend) and runs the given command with
    $env:BW_SESSION set, so a child process can call `bw get <field> <item-id>` itself and use
    the value in-process (for example, type it into a browser form) without it ever passing
    through command text, logs or the console. The session is never persisted and is cleared
    when the command exits. The child is responsible for not printing the values it reads.
    Every argument is passed through verbatim (no parameter binding), so the wrapped command's own
    flags such as -NoProfile or --no-project are not mistaken for parameters of this script.
.EXAMPLE
    ~\.machine\Invoke-WithBwSession.ps1 uv run --no-project --with playwright python mcdp.py `
        fill-secret 'input[name=passwd]' <item-id> password
#>
$Command = @($args)
if ($Command.Count -eq 0) { Write-Warning 'Usage: Invoke-WithBwSession.ps1 <exe> [args...]'; exit 2 }
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Unlock-BwVault.ps1')
try {
    if ((Invoke-BwVaultUnlock -Quiet) -ne 0 -or -not $env:BW_SESSION) { Write-Warning 'Bitwarden unlock failed.'; exit 1 }
    $exe, $rest = $Command
    & $exe @rest
    exit $LASTEXITCODE
}
finally { $env:BW_SESSION = $null }
