<#
.SYNOPSIS
    Execute a command on a DTM infrastructure host via SSH.

.DESCRIPTION
    General-purpose SSH wrapper for any host defined in ~/.ssh/config.
    Defaults to the UDM Pro SE (alias 'udmpro'). Supports structured JSON
    output for LLM agent consumption, file transfer via SCP, and batch
    command execution.

    Replaces the UDM-only Invoke-UdmSsh with a multi-host design.

.PARAMETER Command
    Remote shell command to execute. Required unless -Interactive is set.

.PARAMETER HostAlias
    SSH config alias or user@host string. Default: 'udmpro'.

.PARAMETER Interactive
    Allocate a TTY for interactive sessions. -Command is optional.

.PARAMETER Timeout
    SSH ConnectTimeout in seconds. Default: 10.

.PARAMETER IdentityFile
    Path to SSH private key. Default: empty (use SSH config).

.PARAMETER BindAddress
    Local source address for ssh -b. Default: empty (auto).

.PARAMETER Json
    Wrap output in a JSON envelope with host, command, exit_code, stdout,
    stderr, and duration_ms fields.

.PARAMETER Raw
    Return stdout without trimming trailing whitespace.

.EXAMPLE
    Invoke-DtmSsh -Command "uptime"

.EXAMPLE
    Invoke-DtmSsh -HostAlias dtm-work -Command "whoami"

.EXAMPLE
    Invoke-DtmSsh -Json -Command "free -m"

.EXAMPLE
    Send-DtmFile -LocalPath .\script.sh -RemotePath /data/scripts/script.sh

.EXAMPLE
    Receive-DtmFile -RemotePath /etc/hosts -LocalPath .\hosts.txt
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-DtmSsh {
    [CmdletBinding()]
    [OutputType([string])]
    param(
        [Parameter(Position = 0)]
        [string]$Command = "",

        [Alias("Host")]
        [string]$HostAlias = "udmpro",

        [switch]$Interactive,

        [int]$Timeout = 10,

        [string]$IdentityFile = "",

        [string]$BindAddress = "",

        [switch]$Json,

        [switch]$Raw
    )

    if (-not $Interactive -and [string]::IsNullOrWhiteSpace($Command)) {
        throw "Parameter -Command is required when -Interactive is not set."
    }

    $sshArgs = [System.Collections.Generic.List[string]]::new()

    # Identity file (only if explicitly provided)
    if (-not [string]::IsNullOrWhiteSpace($IdentityFile)) {
        $resolvedKey = $IdentityFile
        if ($resolvedKey.StartsWith("~/") -or $resolvedKey.StartsWith('~\')) {
            $resolvedKey = Join-Path $HOME $resolvedKey.Substring(2)
        }
        $sshArgs.AddRange([string[]]@("-i", $resolvedKey))
    }

    # Bind address (only if explicitly provided)
    if (-not [string]::IsNullOrWhiteSpace($BindAddress)) {
        $sshArgs.AddRange([string[]]@("-b", $BindAddress))
    }

    # Connection options — suppress noise for LLM/script use
    $sshArgs.AddRange([string[]]@(
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=$Timeout",
        "-o", "LogLevel=ERROR",
        "-o", "VisualHostKey=no"
    ))

    if (-not $Interactive) {
        $sshArgs.AddRange([string[]]@("-o", "RequestTTY=no", "-o", "BatchMode=yes"))
    }

    # Target host (SSH config alias carries user/host/port/key)
    $sshArgs.Add($HostAlias)

    if (-not [string]::IsNullOrWhiteSpace($Command)) {
        $sshArgs.Add($Command)
    }

    Write-Verbose "ssh $($sshArgs -join ' ')"

    if ($Interactive) {
        & ssh @sshArgs
        if ($LASTEXITCODE -ne 0) {
            throw "SSH interactive session exited with code $LASTEXITCODE."
        }
        return
    }

    # Non-interactive: capture via unique temp files (unpredictable names
    # to prevent symlink attacks — security: temp file race condition fix)
    $stdoutPath = [System.IO.Path]::GetTempFileName()
    $stderrPath = [System.IO.Path]::GetTempFileName()

    try {
        $sw = [System.Diagnostics.Stopwatch]::StartNew()

        $proc = Start-Process `
            -FilePath "ssh" `
            -ArgumentList $sshArgs `
            -NoNewWindow `
            -PassThru `
            -Wait `
            -RedirectStandardOutput $stdoutPath `
            -RedirectStandardError  $stderrPath

        $sw.Stop()
        $durationMs = $sw.ElapsedMilliseconds

        $stdout = if (Test-Path $stdoutPath) { [System.IO.File]::ReadAllText($stdoutPath) } else { "" }
        $stderr = if (Test-Path $stderrPath) { [System.IO.File]::ReadAllText($stderrPath) } else { "" }

        if ($Json) {
            # JSON envelope
            $envelope = @{
                host        = $HostAlias
                command     = $Command
                exit_code   = $proc.ExitCode
                stdout      = $stdout.TrimEnd()
                stderr      = $stderr.TrimEnd()
                duration_ms = $durationMs
            }
            return ($envelope | ConvertTo-Json -Compress)
        }

        # Surface stderr
        if (-not [string]::IsNullOrWhiteSpace($stderr)) {
            Write-Error $stderr.TrimEnd() -ErrorAction Continue
        }

        if ($proc.ExitCode -ne 0) {
            throw "SSH command failed (exit code $($proc.ExitCode))$(if ($stderr.Trim()) { ": $($stderr.Trim())" })"
        }

        if ($Raw) { return $stdout }
        return $stdout.TrimEnd()
    }
    finally {
        Remove-Item $stdoutPath, $stderrPath -Force -ErrorAction SilentlyContinue
    }
}

function Send-DtmFile {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory, Position = 0)]
        [string]$LocalPath,

        [Parameter(Mandatory, Position = 1)]
        [string]$RemotePath,

        [string]$HostAlias = "udmpro",

        [int]$Timeout = 30
    )

    $scpArgs = @(
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=$Timeout",
        "-o", "LogLevel=ERROR",
        "-o", "VisualHostKey=no",
        "-o", "BatchMode=yes"
    )

    if (Test-Path $LocalPath -PathType Container) {
        $scpArgs += "-r"
    }

    $scpArgs += @($LocalPath, "${HostAlias}:${RemotePath}")

    Write-Verbose "scp $($scpArgs -join ' ')"
    & scp @scpArgs
    if ($LASTEXITCODE -ne 0) {
        throw "SCP upload failed (exit code $LASTEXITCODE)"
    }
}

function Receive-DtmFile {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory, Position = 0)]
        [string]$RemotePath,

        [Parameter(Mandatory, Position = 1)]
        [string]$LocalPath,

        [string]$HostAlias = "udmpro",

        [int]$Timeout = 30
    )

    $scpArgs = @(
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=$Timeout",
        "-o", "LogLevel=ERROR",
        "-o", "VisualHostKey=no",
        "-o", "BatchMode=yes",
        "${HostAlias}:${RemotePath}",
        $LocalPath
    )

    Write-Verbose "scp $($scpArgs -join ' ')"
    & scp @scpArgs
    if ($LASTEXITCODE -ne 0) {
        throw "SCP download failed (exit code $LASTEXITCODE)"
    }
}

function Invoke-DtmSshBatch {
    [CmdletBinding()]
    [OutputType([string])]
    param(
        [Parameter(Mandatory, Position = 0, ValueFromRemainingArguments)]
        [string[]]$Commands,

        [string]$HostAlias = "udmpro",

        [switch]$Json
    )

    $chained = $Commands -join "; echo '---'; "
    Invoke-DtmSsh -HostAlias $HostAlias -Command $chained -Json:$Json
}

# ── backward compatibility ──────────────────────────────────────────────────

function Invoke-UdmSsh {
    [CmdletBinding()]
    [OutputType([string])]
    param(
        [Parameter(Position = 0)]
        [string]$Command = "",
        [switch]$Interactive,
        [int]$Timeout = 10,
        [string]$BindAddress = "",
        [string]$IdentityFile = "",
        [switch]$Raw
    )

    $params = @{ HostAlias = "udmpro" }
    if ($Command)      { $params.Command = $Command }
    if ($Interactive)   { $params.Interactive = $true }
    if ($Timeout -ne 10) { $params.Timeout = $Timeout }
    if ($BindAddress)   { $params.BindAddress = $BindAddress }
    if ($IdentityFile)  { $params.IdentityFile = $IdentityFile }
    if ($Raw)           { $params.Raw = $true }

    Invoke-DtmSsh @params
}
