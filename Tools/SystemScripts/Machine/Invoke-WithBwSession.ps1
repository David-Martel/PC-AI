<#
.SYNOPSIS
    Runs one command with a Bitwarden session scoped to this process, then drops the session.
.DESCRIPTION
    Unlocks through Unlock-BwVault.ps1 (machine backend) and runs the given command with
    $env:BW_SESSION set, so a child process can call `bw get <field> <item-id>` itself and use
    the value in-process (for example, type it into a browser form) without it ever passing
    through command text, logs or the console. This script never persists the session itself and
    clears $env:BW_SESSION when the command exits. The unlock goes through the machine backend,
    which keeps its shared protected session file (~/.bwdata/session.txt) under its own contract:
    a fresh process recovers from that file, and on a host without a password source it is the
    only way to unlock. The child is responsible for not printing the values it reads.
    Every argument after the executable is passed through verbatim (no parameter binding), so the
    wrapped command's own flags such as -NoProfile or --no-project are not mistaken for parameters
    of this script. Only the first argument is checked for this script's own options:
      -h, --help, -Help, -?   print usage and exit 0, without unlocking.
      -DryRun                 name the command that would run and exit 0, without unlocking.
.EXAMPLE
    ~\.machine\Invoke-WithBwSession.ps1 uv run --no-project --with playwright python mcdp.py `
        fill-secret 'input[name=passwd]' <item-id> password
#>
$Command = @($args)
$usage = 'Usage: Invoke-WithBwSession.ps1 [-DryRun] <exe> [args...]   (-h for help)'
if ($Command.Count -eq 0) { Write-Warning $usage; exit 2 }
# Help and dry-run must answer before the unlock, which can write the backend's session file.
if ($Command[0] -in @('-h', '--help', '-Help', '-?')) { Write-Output $usage; exit 0 }
if ($Command[0] -eq '-DryRun') {
    if ($Command.Count -lt 2) { Write-Warning $usage; exit 2 }
    Write-Output "Would unlock Bitwarden and run: $($Command[1]) (+$($Command.Count - 2) argument(s), not shown)"
    exit 0
}
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Unlock-BwVault.ps1')
try {
    if ((Invoke-BwVaultUnlock -Quiet) -ne 0 -or -not $env:BW_SESSION) { Write-Warning 'Bitwarden unlock failed.'; exit 1 }
    # Index, don't destructure: `$exe, $rest = $Command` leaves $rest a [string] when one
    # argument follows the executable, and splatting a string passes it per character.
    $exe = $Command[0]
    [string[]]$rest = @($Command | Select-Object -Skip 1)
    & $exe @rest
    exit $LASTEXITCODE
}
finally { $env:BW_SESSION = $null }
