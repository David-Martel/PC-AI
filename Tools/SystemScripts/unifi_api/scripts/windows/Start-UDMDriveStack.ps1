<#
.SYNOPSIS
Starts and validates the UDM SMB/rclone drive stack.

.DESCRIPTION
Runs dependency checks, optional network reachability checks, rclone remote
setup, rclone mount startup, SMB credential/link setup, health checks,
transcript logging, structured JSON/NDJSON logging, and Application event-log
integration for the UDM drive stack.

.PARAMETER DryRun
Run prerequisite validation and logging, then exit before changing rclone
configuration, starting mounts, setting SMB credentials, or creating SMB links.
The long CLI form `--DryRun` is also accepted.

.PARAMETER Help
Print script help and exit. The aliases `-h` and `--help` are also accepted.
#>
param(
    [string]$TargetHost = "192.168.1.1",
    [string]$SmbShareName = "udmpro_data",
    [string]$SmbMountPath = "F:\udm_smb",
    [string]$RcloneMountPath = "W:\udm_rclone",
    [string]$RcloneRemote = "udm_sftp",
    [string]$RcloneRemotePath = "/data",
    [string]$RcloneLogPath = "C:\Users\david\unifi_api\docs\commands\rclone_udm_mount.log",
    [string]$SshHostAlias = "udmpro",
    [string]$SshUser = "root",
    [string]$SmbCredentialUser = "root",
    [string]$SmbCredentialPassword = "",
    [string]$IdentityFile = "",
    [string]$LogRoot = "",
    [string]$EventSource = "UDMDriveStack",
    [switch]$EnsureSmbEnabled = $true,
    [switch]$SkipNetworkCheck,
    [int]$RetryCount = 6,
    [int]$RetryDelaySeconds = 10,
    [int]$NetworkTimeoutMs = 2000,
    [int]$RcloneLogMaxAgeHours = 24,
    [switch]$DryRun,
    [Alias('h', '?')]
    [switch]$Help,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CliArgs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$CliArgs = @($CliArgs | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
if (@($CliArgs) -contains '--help') {
    $Help = $true
    $CliArgs = @($CliArgs | Where-Object { $_ -ne '--help' })
}
if (@($CliArgs) -contains '--DryRun') {
    $DryRun = $true
    $CliArgs = @($CliArgs | Where-Object { $_ -ne '--DryRun' })
}
if ($Help) {
    $helpMatch = [regex]::Match((Get-Content -LiteralPath $PSCommandPath -Raw), '(?s)<#\s*(.*?)\s*#>')
    if ($helpMatch.Success) { $helpMatch.Groups[1].Value.Trim() } else { Get-Help -Detailed $PSCommandPath }
    return
}
if (@($CliArgs).Count -gt 0) {
    throw "Unknown CLI argument(s): $($CliArgs -join ', ')"
}

$script:ExitCodes = [ordered]@{
    Success             = 0
    HostUnreachable     = 20
    MissingDependency   = 30
    MountFailed         = 40
    PartialHealth       = 50
    UnexpectedException = 90
}

$script:EventIds = [ordered]@{
    Start    = 4400
    Skip     = 4401
    Retry    = 4402
    Degraded = 4403
    Success  = 4404
    Failure  = 4405
}

$script:FailureExitCode = $null
$script:TranscriptStarted = $false
$script:EventSourceReady = $false
$script:RcloneExe = $null
$script:SshExe = $null
$script:RcloneRunLogPath = $null
$script:RcloneProcessSummary = $null
$script:RunStamp = Get-Date -Format "yyyyMMdd-HHmmss"
$script:StartedAt = Get-Date
$script:ScriptDirectory = Split-Path -Parent $PSCommandPath
$script:RepoRoot = Split-Path -Parent (Split-Path -Parent $script:ScriptDirectory)

if (-not $LogRoot) {
    $LogRoot = Join-Path $script:RepoRoot "logs\udm-drive-stack"
}

$script:RunLogDirectory = Join-Path $LogRoot $script:RunStamp
New-Item -ItemType Directory -Path $script:RunLogDirectory -Force | Out-Null

$script:TranscriptPath = Join-Path $script:RunLogDirectory "transcript.log"
$script:StructuredLogPath = Join-Path $script:RunLogDirectory "events.ndjson"
$script:ResultPath = Join-Path $script:RunLogDirectory "result.json"

try {
    Start-Transcript -Path $script:TranscriptPath -UseMinimalHeader -ErrorAction Stop | Out-Null
    $script:TranscriptStarted = $true
}
catch {
    Start-Transcript -Path $script:TranscriptPath -ErrorAction Stop | Out-Null
    $script:TranscriptStarted = $true
}

$script:RunResult = [ordered]@{
    run_id               = $script:RunStamp
    started_at           = $script:StartedAt.ToString("o")
    completed_at         = $null
    status               = "running"
    exit_code            = $null
    exit_code_name       = $null
    script_path          = $PSCommandPath
    log_directory        = $script:RunLogDirectory
    transcript_path      = $script:TranscriptPath
    structured_log_path  = $script:StructuredLogPath
    target_host          = $TargetHost
    smb_mount_path       = $SmbMountPath
    rclone_mount_path    = $RcloneMountPath
    rclone_remote        = $RcloneRemote
    rclone_remote_path   = $RcloneRemotePath
    legacy_rclone_log    = $RcloneLogPath
    run_rclone_log       = $null
    rclone_process       = $null
    dry_run              = [bool]$DryRun
    checks               = @()
    retries              = @()
    events               = @()
    error                = $null
}

function Redact-SensitiveText {
    param([AllowNull()][string]$Text)

    if ($null -eq $Text) {
        return $null
    }

    $redacted = $Text -replace "(?i)(pass(word)?|token|secret|key)\s+[`"'][^`"']+[`"']", '$1 <redacted>'
    $redacted = $redacted -replace "(?i)(/pass:|--password[=\s]+|password=)[^\s]+", '$1<redacted>'
    return $redacted
}

function ConvertTo-LogObject {
    param(
        [string]$Kind,
        [hashtable]$Data
    )

    $record = [ordered]@{
        timestamp = (Get-Date).ToString("o")
        kind      = $Kind
    }

    foreach ($key in $Data.Keys) {
        $record[$key] = $Data[$key]
    }

    return $record
}

function Add-RunRecord {
    param(
        [string]$Kind,
        [hashtable]$Data
    )

    $record = ConvertTo-LogObject -Kind $Kind -Data $Data
    $record | ConvertTo-Json -Depth 8 -Compress | Add-Content -LiteralPath $script:StructuredLogPath -Encoding UTF8
    return $record
}

function Add-CheckResult {
    param(
        [string]$Name,
        [string]$Status,
        [string]$Message,
        [hashtable]$Data = @{}
    )

    $check = ConvertTo-LogObject -Kind "check" -Data @{
        name    = $Name
        status  = $Status
        message = Redact-SensitiveText $Message
        data    = $Data
    }
    $script:RunResult.checks += $check
    $check | ConvertTo-Json -Depth 8 -Compress | Add-Content -LiteralPath $script:StructuredLogPath -Encoding UTF8
    return $check
}

function Ensure-EventSource {
    param([string]$Source)

    try {
        if (-not [System.Diagnostics.EventLog]::SourceExists($Source)) {
            New-EventLog -LogName Application -Source $Source -ErrorAction Stop
        }
        $script:EventSourceReady = $true
        Add-CheckResult -Name "event_source" -Status "ok" -Message "Event source is available." -Data @{ source = $Source } | Out-Null
    }
    catch {
        $script:EventSourceReady = $false
        Add-CheckResult -Name "event_source" -Status "degraded" -Message "Event source is not writable: $($_.Exception.Message)" -Data @{ source = $Source } | Out-Null
    }
}

function Write-StackEvent {
    param(
        [ValidateSet("Start", "Skip", "Retry", "Degraded", "Success", "Failure")]
        [string]$Name,
        [string]$Message,
        [System.Diagnostics.EventLogEntryType]$EntryType = [System.Diagnostics.EventLogEntryType]::Information,
        [hashtable]$Data = @{}
    )

    $eventId = $script:EventIds[$Name]
    $safeMessage = Redact-SensitiveText $Message
    $eventRecord = Add-RunRecord -Kind "event" -Data @{
        name       = $Name
        event_id   = $eventId
        entry_type = $EntryType.ToString()
        message    = $safeMessage
        data       = $Data
    }
    $script:RunResult.events += $eventRecord

    if ($script:EventSourceReady) {
        try {
            Write-EventLog -LogName Application -Source $EventSource -EntryType $EntryType -EventId $eventId -Message $safeMessage -ErrorAction Stop
        }
        catch {
            Add-CheckResult -Name "event_write" -Status "degraded" -Message "Failed to write Windows event log entry ${eventId}: $($_.Exception.Message)" -Data @{ event_id = $eventId } | Out-Null
        }
    }
}

function Write-Step {
    param([string]$Message)

    $safeMessage = Redact-SensitiveText $Message
    Write-Host "[udm-drive] $safeMessage"
    Add-RunRecord -Kind "step" -Data @{ message = $safeMessage } | Out-Null
}

function Complete-Run {
    param(
        [string]$Status,
        [string]$ExitCodeName,
        [string]$Message = ""
    )

    $exitCode = $script:ExitCodes[$ExitCodeName]
    $script:RunResult.completed_at = (Get-Date).ToString("o")
    $script:RunResult.status = $Status
    $script:RunResult.exit_code = $exitCode
    $script:RunResult.exit_code_name = $ExitCodeName
    $script:RunResult.run_rclone_log = $script:RcloneRunLogPath
    $script:RunResult.rclone_process = $script:RcloneProcessSummary

    if ($Message) {
        $script:RunResult.error = Redact-SensitiveText $Message
    }

    $script:RunResult | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $script:ResultPath -Encoding UTF8

    if ($script:TranscriptStarted) {
        Stop-Transcript | Out-Null
    }

    exit $exitCode
}

function Throw-RunFailure {
    param(
        [ValidateSet("HostUnreachable", "MissingDependency", "MountFailed", "PartialHealth", "UnexpectedException")]
        [string]$ExitCodeName,
        [string]$Message
    )

    $script:FailureExitCode = $ExitCodeName
    throw $Message
}

function Test-HostReachable {
    param(
        [string]$HostName,
        [int]$TimeoutMs
    )

    try {
        $ping = [System.Net.NetworkInformation.Ping]::new()
        $reply = $ping.Send($HostName, $TimeoutMs)
        $ping.Dispose()
        if ($reply.Status -eq [System.Net.NetworkInformation.IPStatus]::Success) {
            return $true
        }
    }
    catch {
    }

    try {
        $client = [System.Net.Sockets.TcpClient]::new()
        $async = $client.BeginConnect($HostName, 22, $null, $null)
        $connected = $async.AsyncWaitHandle.WaitOne($TimeoutMs)
        if ($connected) {
            $client.EndConnect($async)
            $client.Dispose()
            return $true
        }
        $client.Dispose()
    }
    catch {
    }

    return $false
}

function Resolve-CommandPath {
    param([string]$Name)

    $command = Get-Command $Name -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $command) {
        return $null
    }
    return $command.Source
}

function Resolve-IdentityFile {
    param(
        [string]$Preferred,
        [string]$HostAlias
    )

    if ($Preferred) {
        if ($Preferred.StartsWith("~/")) {
            return (Join-Path $HOME ($Preferred.Substring(2)))
        }
        return [Environment]::ExpandEnvironmentVariables($Preferred)
    }

    $resolved = & $script:SshExe -G $HostAlias 2>$null
    if ($LASTEXITCODE -eq 0) {
        foreach ($line in $resolved) {
            if ($line -match "^identityfile\s+(.+)$") {
                $path = $matches[1].Trim()
                if ($path.StartsWith("~/")) {
                    return (Join-Path $HOME ($path.Substring(2)))
                }
                return [Environment]::ExpandEnvironmentVariables($path)
            }
        }
    }

    Throw-RunFailure -ExitCodeName MissingDependency -Message "Unable to resolve SSH identity file for host alias '$HostAlias'."
}

function Test-SmbCredentialAvailable {
    param(
        [string]$HostName,
        [string]$User,
        [string]$Password
    )

    if ($Password) {
        return @{ available = $true; source = "supplied_password" }
    }

    $targets = @($HostName, "MicrosoftAccount:$HostName", "Domain:target=$HostName")
    foreach ($target in $targets) {
        $output = & cmdkey /list:$target 2>&1
        if (($LASTEXITCODE -eq 0) -and (($output -join "`n") -match "(?i)Target:")) {
            return @{ available = $true; source = "windows_credential_manager"; target = $target }
        }
    }

    return @{ available = $false; source = "missing"; user = $User }
}

function Test-WinFspAvailable {
    $service = Get-Service -Name "WinFsp.Launcher" -ErrorAction SilentlyContinue
    if (-not $service) {
        return @{ available = $false; message = "WinFsp.Launcher service is missing." }
    }
    if ($service.Status -ne "Running") {
        return @{ available = $false; message = "WinFsp.Launcher service is $($service.Status)." }
    }
    return @{ available = $true; message = "WinFsp.Launcher is running."; status = $service.Status.ToString() }
}

function Test-MountPathReady {
    param(
        [string]$Path,
        [string]$Label
    )

    $parent = Split-Path -Parent $Path
    if (-not $parent) {
        Throw-RunFailure -ExitCodeName MissingDependency -Message "$Label mount path '$Path' does not have a parent directory."
    }

    if (-not (Test-Path $parent)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
        Add-CheckResult -Name "${Label}_mount_parent" -Status "ok" -Message "Created mount parent directory." -Data @{ path = $parent } | Out-Null
    }
    else {
        Add-CheckResult -Name "${Label}_mount_parent" -Status "ok" -Message "Mount parent exists." -Data @{ path = $parent } | Out-Null
    }

    if (Test-Path $Path) {
        $item = Get-Item -LiteralPath $Path -Force
        if ($item.PSIsContainer) {
            $children = Get-ChildItem -LiteralPath $Path -Force -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($children) {
                Add-CheckResult -Name "${Label}_mount_path" -Status "degraded" -Message "Mount path exists and is not empty." -Data @{ path = $Path } | Out-Null
                return $false
            }
            Add-CheckResult -Name "${Label}_mount_path" -Status "ok" -Message "Mount path exists and is empty." -Data @{ path = $Path } | Out-Null
            return $true
        }
        Throw-RunFailure -ExitCodeName MissingDependency -Message "$Label mount path '$Path' exists and is not a directory."
    }

    Add-CheckResult -Name "${Label}_mount_path" -Status "ok" -Message "Mount path is available for creation." -Data @{ path = $Path } | Out-Null
    return $true
}

function Find-RcloneMountProcess {
    param(
        [string]$RemoteSpec,
        [string]$MountPath
    )

    Get-CimInstance Win32_Process -Filter "Name='rclone.exe'" -ErrorAction SilentlyContinue | Where-Object {
        $_.CommandLine -like "*mount*" -and
        $_.CommandLine -like "*$RemoteSpec*" -and
        $_.CommandLine -like "*$MountPath*"
    } | Select-Object -First 1
}

function Rotate-RcloneLog {
    param(
        [string]$LegacyLogPath,
        [string]$RunLogPath,
        [int]$MaxAgeHours
    )

    $logDir = Split-Path -Parent $LegacyLogPath
    if (-not (Test-Path $logDir)) {
        New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    }

    if (Test-Path $LegacyLogPath) {
        $existing = Get-Item -LiteralPath $LegacyLogPath -Force
        $ageHours = ((Get-Date) - $existing.LastWriteTime).TotalHours
        $rotatedPath = "$LegacyLogPath.$($script:RunStamp).old"
        Move-Item -LiteralPath $LegacyLogPath -Destination $rotatedPath -Force
        $status = if ($ageHours -gt $MaxAgeHours) { "degraded" } else { "ok" }
        Add-CheckResult -Name "rclone_log_rotation" -Status $status -Message "Rotated prior rclone log." -Data @{
            previous_log = $LegacyLogPath
            rotated_log  = $rotatedPath
            age_hours    = [math]::Round($ageHours, 2)
            stale_after  = $MaxAgeHours
        } | Out-Null
    }
    else {
        Add-CheckResult -Name "rclone_log_rotation" -Status "ok" -Message "No prior rclone log existed." -Data @{ previous_log = $LegacyLogPath } | Out-Null
    }

    $runLogDir = Split-Path -Parent $RunLogPath
    if (-not (Test-Path $runLogDir)) {
        New-Item -ItemType Directory -Path $runLogDir -Force | Out-Null
    }
}

function Test-RcloneConfigReadable {
    $remotes = & $script:RcloneExe listremotes 2>$null
    if ($LASTEXITCODE -ne 0) {
        Throw-RunFailure -ExitCodeName MissingDependency -Message "rclone listremotes failed; verify rclone configuration is readable for this user."
    }
    Add-CheckResult -Name "rclone_config" -Status "ok" -Message "rclone configuration is readable." -Data @{
        remote_present = ($remotes -contains "$RcloneRemote`:")
    } | Out-Null
}

function Test-Dependencies {
    $script:RcloneExe = Resolve-CommandPath -Name "rclone"
    if (-not $script:RcloneExe) {
        Throw-RunFailure -ExitCodeName MissingDependency -Message "Missing dependency: rclone.exe is not on PATH."
    }
    Add-CheckResult -Name "dependency_rclone" -Status "ok" -Message "rclone.exe found." -Data @{ path = $script:RcloneExe } | Out-Null

    $script:SshExe = Resolve-CommandPath -Name "ssh"
    if (-not $script:SshExe) {
        Throw-RunFailure -ExitCodeName MissingDependency -Message "Missing dependency: ssh.exe is not on PATH."
    }
    Add-CheckResult -Name "dependency_ssh" -Status "ok" -Message "ssh.exe found." -Data @{ path = $script:SshExe } | Out-Null

    $winFsp = Test-WinFspAvailable
    if (-not $winFsp.available) {
        Add-CheckResult -Name "dependency_winfsp" -Status "failed" -Message $winFsp.message -Data @{} | Out-Null
        Throw-RunFailure -ExitCodeName MissingDependency -Message $winFsp.message
    }
    Add-CheckResult -Name "dependency_winfsp" -Status "ok" -Message $winFsp.message -Data @{ status = $winFsp.status } | Out-Null

    Test-RcloneConfigReadable

    $credential = Test-SmbCredentialAvailable -HostName $TargetHost -User $SmbCredentialUser -Password $SmbCredentialPassword
    if (-not $credential.available) {
        Add-CheckResult -Name "smb_credential" -Status "degraded" -Message "No SMB credential was supplied or found in Windows Credential Manager." -Data @{
            host = $TargetHost
            user = $SmbCredentialUser
        } | Out-Null
    }
    else {
        Add-CheckResult -Name "smb_credential" -Status "ok" -Message "SMB credential material is available." -Data @{
            host   = $TargetHost
            source = $credential.source
        } | Out-Null
    }

    $script:Identity = Resolve-IdentityFile -Preferred $IdentityFile -HostAlias $SshHostAlias
    if (-not (Test-Path $script:Identity)) {
        Throw-RunFailure -ExitCodeName MissingDependency -Message "SSH identity file not found: $($script:Identity)"
    }
    Add-CheckResult -Name "ssh_identity" -Status "ok" -Message "SSH identity file exists." -Data @{ path = $script:Identity } | Out-Null

    $rcloneRemoteSpec = "$RcloneRemote`:$RcloneRemotePath"
    $existingProc = Find-RcloneMountProcess -RemoteSpec $rcloneRemoteSpec -MountPath $RcloneMountPath
    if ($existingProc) {
        $script:RcloneProcessSummary = [ordered]@{
            pid          = $existingProc.ProcessId
            command_line = Redact-SensitiveText $existingProc.CommandLine
            already_running = $true
        }
        Add-CheckResult -Name "rclone_existing_process" -Status "ok" -Message "rclone mount process is already running." -Data @{
            pid = $existingProc.ProcessId
        } | Out-Null
    }
    else {
        if (-not (Test-MountPathReady -Path $RcloneMountPath -Label "rclone")) {
            Throw-RunFailure -ExitCodeName MountFailed -Message "rclone mount path '$RcloneMountPath' exists and is not empty while no matching rclone process is running."
        }
    }

    Test-MountPathReady -Path $SmbMountPath -Label "smb" | Out-Null
}

function Ensure-SmbPathLink {
    param(
        [string]$LinkPath,
        [string]$RemotePath
    )

    $parent = Split-Path -Parent $LinkPath
    if (-not (Test-Path $parent)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }

    if (Test-Path $LinkPath) {
        $item = Get-Item -LiteralPath $LinkPath -Force
        $targetText = if ($item.Target -is [Array]) { ($item.Target -join ";") } else { [string]$item.Target }
        if ($item.LinkType) {
            if ($targetText -and $targetText -ine $RemotePath) {
                Write-Step "SMB link path exists with target '$targetText' (expected '$RemotePath'); reusing existing link."
            }
            else {
                Write-Step "SMB link already present: $LinkPath -> $RemotePath"
            }
            return
        }
        Throw-RunFailure -ExitCodeName MountFailed -Message "Path '$LinkPath' already exists and is not the expected SMB link target."
    }

    $mklink = & cmd /c "mklink /d `"$LinkPath`" `"$RemotePath`"" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Throw-RunFailure -ExitCodeName MountFailed -Message "Failed to create SMB symlink ($LASTEXITCODE): $($mklink -join ' ')"
    }
    Write-Step "Created SMB link: $LinkPath -> $RemotePath"
}

function Ensure-SmbCredential {
    param(
        [string]$TargetHost,
        [string]$User,
        [string]$Password
    )

    if (-not $Password) {
        return
    }

    & cmdkey /add:$TargetHost /user:$User /pass:$Password | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Throw-RunFailure -ExitCodeName MissingDependency -Message "Failed to set SMB credential for $TargetHost."
    }
    Write-Step "Updated Windows Credential Manager entry for SMB host $TargetHost."
}

function Ensure-RcloneRemote {
    param(
        [string]$RemoteName,
        [string]$TargetHost,
        [string]$User,
        [string]$KeyFile
    )

    $remoteKey = "${RemoteName}:"
    $existing = & $script:RcloneExe listremotes 2>$null
    if ($LASTEXITCODE -ne 0) {
        Throw-RunFailure -ExitCodeName MissingDependency -Message "rclone listremotes failed."
    }

    if ($existing -contains $remoteKey) {
        & $script:RcloneExe config update $RemoteName host $TargetHost user $User key_file $KeyFile shell_type unix | Out-Null
        if ($LASTEXITCODE -ne 0) {
            Throw-RunFailure -ExitCodeName MissingDependency -Message "Failed to update rclone remote '$RemoteName'."
        }
        Write-Step "Updated rclone remote '$RemoteName'."
        return
    }

    & $script:RcloneExe config create $RemoteName sftp host $TargetHost user $User key_file $KeyFile shell_type unix | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Throw-RunFailure -ExitCodeName MissingDependency -Message "Failed to create rclone remote '$RemoteName'."
    }
    Write-Step "Created rclone remote '$RemoteName'."
}

function Ensure-RcloneMount {
    param(
        [string]$RemoteName,
        [string]$RemotePath,
        [string]$MountPath,
        [string]$LegacyLogPath
    )

    $mountParent = Split-Path -Parent $MountPath
    if (-not (Test-Path $mountParent)) {
        New-Item -ItemType Directory -Path $mountParent -Force | Out-Null
    }

    $remoteSpec = "$RemoteName`:$RemotePath"
    $existingProc = Find-RcloneMountProcess -RemoteSpec $remoteSpec -MountPath $MountPath
    if ($existingProc) {
        $script:RcloneProcessSummary = [ordered]@{
            pid          = $existingProc.ProcessId
            command_line = Redact-SensitiveText $existingProc.CommandLine
            already_running = $true
        }
        Write-Step "rclone mount process already running (PID $($existingProc.ProcessId))."
        return
    }

    $script:RcloneRunLogPath = Join-Path $script:RunLogDirectory "rclone_mount.log"
    Rotate-RcloneLog -LegacyLogPath $LegacyLogPath -RunLogPath $script:RcloneRunLogPath -MaxAgeHours $RcloneLogMaxAgeHours

    if (Test-Path $MountPath) {
        $existingChildren = Get-ChildItem -LiteralPath $MountPath -Force -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($existingChildren) {
            Throw-RunFailure -ExitCodeName MountFailed -Message "Mount path '$MountPath' exists and is not empty."
        }
        Remove-Item -LiteralPath $MountPath -Force -Recurse
    }

    $mountArgs = @(
        "mount",
        "$RemoteName`:$RemotePath",
        $MountPath,
        "--links",
        "--vfs-cache-mode", "full",
        "--vfs-cache-max-age", "1h",
        "--vfs-cache-max-size", "512M",
        "--vfs-read-ahead", "64M",
        "--vfs-fast-fingerprint",
        "--dir-cache-time", "2m",
        "--buffer-size", "32M",
        "--transfers", "4",
        "--retries", "5",
        "--low-level-retries", "10",
        "--retries-sleep", "2s",
        "--sftp-concurrency", "8",
        "--sftp-idle-timeout", "60s",
        "--log-file", $script:RcloneRunLogPath,
        "--log-level", "INFO"
    )

    $proc = Start-Process -FilePath $script:RcloneExe -ArgumentList $mountArgs -WindowStyle Hidden -PassThru
    Start-Sleep -Seconds 3

    if ($proc.HasExited) {
        Throw-RunFailure -ExitCodeName MountFailed -Message "rclone mount process exited immediately (exit=$($proc.ExitCode)). Check log: $($script:RcloneRunLogPath)"
    }

    $process = Get-CimInstance Win32_Process -Filter "ProcessId=$($proc.Id)" -ErrorAction SilentlyContinue
    $script:RcloneProcessSummary = [ordered]@{
        pid             = $proc.Id
        command_line    = Redact-SensitiveText $(if ($process) { $process.CommandLine } else { "$($script:RcloneExe) $($mountArgs -join ' ')" })
        already_running = $false
    }

    Add-RunRecord -Kind "rclone_process" -Data @{
        pid          = $script:RcloneProcessSummary.pid
        command_line = $script:RcloneProcessSummary.command_line
        log_path     = $script:RcloneRunLogPath
    } | Out-Null
    Write-Step "Started rclone mount process (PID $($proc.Id)) at $MountPath."
}

function Test-MountHealth {
    param(
        [string]$MountPath,
        [string]$Label
    )

    if (-not (Test-Path $MountPath)) {
        Write-Step "WARNING: $Label mount path not found: $MountPath"
        Add-CheckResult -Name "${Label}_health" -Status "failed" -Message "$Label mount path not found." -Data @{ path = $MountPath } | Out-Null
        return $false
    }

    try {
        $entries = Get-ChildItem -LiteralPath $MountPath -Force -ErrorAction Stop | Select-Object -First 5
        if ($entries.Count -gt 0) {
            Write-Step "$Label mount healthy ($($entries.Count)+ entries visible)."
            Add-CheckResult -Name "${Label}_health" -Status "ok" -Message "$Label mount has visible entries." -Data @{ path = $MountPath; visible_entries = $entries.Count } | Out-Null
            return $true
        }
        Write-Step "WARNING: $Label mount is empty: $MountPath"
        Add-CheckResult -Name "${Label}_health" -Status "degraded" -Message "$Label mount is empty." -Data @{ path = $MountPath } | Out-Null
        return $false
    }
    catch {
        Write-Step "WARNING: $Label mount read failed: $($_.Exception.Message)"
        Add-CheckResult -Name "${Label}_health" -Status "failed" -Message "$Label mount read failed: $($_.Exception.Message)" -Data @{ path = $MountPath } | Out-Null
        return $false
    }
}

function Invoke-WithRetry {
    param(
        [scriptblock]$Action,
        [int]$Attempts,
        [int]$DelaySeconds,
        [string]$Name
    )

    for ($i = 1; $i -le $Attempts; $i++) {
        try {
            & $Action
            return
        }
        catch {
            $message = Redact-SensitiveText $_.Exception.Message
            if ($i -eq $Attempts) {
                throw
            }
            $retryRecord = Add-RunRecord -Kind "retry" -Data @{
                name       = $Name
                attempt    = $i
                attempts   = $Attempts
                delay_sec  = $DelaySeconds
                error      = $message
            }
            $script:RunResult.retries += $retryRecord
            Write-StackEvent -Name Retry -EntryType Warning -Message "$Name failed (attempt $i/$Attempts): $message" -Data @{
                name    = $Name
                attempt = $i
            }
            Start-Sleep -Seconds $DelaySeconds
        }
    }
}

try {
    Ensure-EventSource -Source $EventSource
    Write-StackEvent -Name Start -Message "Starting UDM drive stack for $TargetHost. Logs: $script:RunLogDirectory" -Data @{
        target_host = $TargetHost
        log_dir     = $script:RunLogDirectory
    }

    if ($SmbCredentialPassword) {
        Write-Step "SMB password was supplied as an argument. It will not be written to structured logs; prefer Windows Credential Manager for scheduled runs."
    }

    if (-not $SkipNetworkCheck) {
        Write-Step "Checking network reachability of $TargetHost..."
        $reachable = Test-HostReachable -HostName $TargetHost -TimeoutMs $NetworkTimeoutMs
        if (-not $reachable) {
            Add-CheckResult -Name "network_reachability" -Status "failed" -Message "$TargetHost is not reachable." -Data @{ host = $TargetHost; timeout_ms = $NetworkTimeoutMs } | Out-Null
            Write-StackEvent -Name Skip -EntryType Warning -Message "Skipping UDM drive stack: $TargetHost is not reachable." -Data @{ host = $TargetHost }
            Complete-Run -Status "skipped" -ExitCodeName HostUnreachable -Message "$TargetHost is not reachable."
        }
        Add-CheckResult -Name "network_reachability" -Status "ok" -Message "$TargetHost is reachable." -Data @{ host = $TargetHost; timeout_ms = $NetworkTimeoutMs } | Out-Null
        Write-Step "Host $TargetHost is reachable."
    }
    else {
        Add-CheckResult -Name "network_reachability" -Status "skipped" -Message "Network check skipped by caller." -Data @{ host = $TargetHost } | Out-Null
    }

    Test-Dependencies

    if ($DryRun) {
        Write-Step "Dry run requested. Dependency validation completed; mount and SMB changes will not be applied."
        Write-StackEvent -Name Skip -Message "Dry run completed for UDM drive stack. No mount or SMB changes were applied." -Data @{
            target_host = $TargetHost
            rclone_mount = $RcloneMountPath
            smb_mount = $SmbMountPath
        }
        Complete-Run -Status "dry-run" -ExitCodeName Success
    }

    if ($EnsureSmbEnabled) {
        $smbScript = Join-Path $script:RepoRoot "scripts\udm_boot\Set-UDMSmbBootState.ps1"
        if (-not (Test-Path $smbScript)) {
            Write-Step "WARNING: SMB state script not found: $smbScript (skipping server-side setup)"
            Add-CheckResult -Name "smb_server_setup_script" -Status "degraded" -Message "SMB state script not found." -Data @{ path = $smbScript } | Out-Null
        }
        else {
            Write-Step "Ensuring UDM SMB boot state is enabled."
            & powershell -ExecutionPolicy Bypass -File $smbScript -Target "$SshUser@$TargetHost" -State enable -AuthMode user -SmbUser $SshUser
            if ($LASTEXITCODE -ne 0) {
                Write-Step "WARNING: Failed to enable SMB boot state (non-fatal, continuing)."
                Add-CheckResult -Name "smb_server_setup" -Status "degraded" -Message "Failed to enable SMB boot state." -Data @{ exit_code = $LASTEXITCODE } | Out-Null
            }
            else {
                Add-CheckResult -Name "smb_server_setup" -Status "ok" -Message "UDM SMB boot state command completed." -Data @{} | Out-Null
            }
        }
    }

    Invoke-WithRetry -Name "rclone remote setup" -Attempts $RetryCount -DelaySeconds $RetryDelaySeconds -Action {
        Ensure-RcloneRemote -RemoteName $RcloneRemote -TargetHost $TargetHost -User $SshUser -KeyFile $script:Identity
    }

    Invoke-WithRetry -Name "rclone mount" -Attempts $RetryCount -DelaySeconds $RetryDelaySeconds -Action {
        Ensure-RcloneMount -RemoteName $RcloneRemote -RemotePath $RcloneRemotePath -MountPath $RcloneMountPath -LegacyLogPath $RcloneLogPath
    }

    Invoke-WithRetry -Name "SMB credential setup" -Attempts $RetryCount -DelaySeconds $RetryDelaySeconds -Action {
        Ensure-SmbCredential -TargetHost $TargetHost -User $SmbCredentialUser -Password $SmbCredentialPassword
    }

    Invoke-WithRetry -Name "SMB link setup" -Attempts $RetryCount -DelaySeconds $RetryDelaySeconds -Action {
        Ensure-SmbPathLink -LinkPath $SmbMountPath -RemotePath "\\$TargetHost\$SmbShareName"
    }

    $rcloneHealthy = Test-MountHealth -MountPath $RcloneMountPath -Label "rclone"
    $smbHealthy = Test-MountHealth -MountPath $SmbMountPath -Label "SMB"

    if ($rcloneHealthy -and $smbHealthy) {
        Write-Step "Drive stack ready. rclone: $RcloneMountPath | SMB: $SmbMountPath"
        Write-StackEvent -Name Success -Message "UDM drive stack ready. rclone: $RcloneMountPath | SMB: $SmbMountPath" -Data @{
            rclone_mount = $RcloneMountPath
            smb_mount    = $SmbMountPath
        }
        Complete-Run -Status "success" -ExitCodeName Success
    }
    elseif ($rcloneHealthy -or $smbHealthy) {
        Write-Step "Drive stack partial. rclone: $rcloneHealthy | SMB: $smbHealthy"
        Write-StackEvent -Name Degraded -EntryType Warning -Message "UDM drive stack is partially healthy. rclone=$rcloneHealthy SMB=$smbHealthy" -Data @{
            rclone_healthy = $rcloneHealthy
            smb_healthy    = $smbHealthy
        }
        Complete-Run -Status "degraded" -ExitCodeName PartialHealth -Message "Partial health: rclone=$rcloneHealthy SMB=$smbHealthy"
    }
    else {
        Write-Step "WARNING: Drive stack unhealthy. Check logs: $($script:RcloneRunLogPath)"
        Throw-RunFailure -ExitCodeName MountFailed -Message "Drive stack health check failed for both rclone and SMB mounts."
    }
}
catch {
    $message = Redact-SensitiveText $_.Exception.Message
    $exitCodeName = if ($script:FailureExitCode) { $script:FailureExitCode } else { "UnexpectedException" }
    $entryType = [System.Diagnostics.EventLogEntryType]::Error
    Write-Step "ERROR: $message"
    Write-StackEvent -Name Failure -EntryType $entryType -Message "UDM drive stack failed ($exitCodeName): $message" -Data @{ exit_code_name = $exitCodeName }
    Complete-Run -Status "failed" -ExitCodeName $exitCodeName -Message $message
}
