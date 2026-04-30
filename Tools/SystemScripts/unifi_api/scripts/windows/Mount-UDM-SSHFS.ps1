param(
    [string]$TargetHost = "192.168.1.1",
    [string]$User = "root",
    [string]$RemotePath = "/",
    [ValidatePattern("^[A-Za-z]$")]
    [string]$DriveLetter = "U",
    [string]$SshfsPath = "C:\Program Files\SSHFS-Win\bin\sshfs.exe",
    [switch]$Unmount
)

# Check WinFsp installation (check directory, not specific binary — layout varies by version)
$winFspDir = "C:\Program Files\WinFsp"
if (-not (Test-Path $winFspDir)) {
    Write-Warning @"
WinFsp not installed. Install from: https://winfsp.dev/rel/
  winget install -e --id WinFsp.WinFsp
Then install SSHFS-Win:
  winget install -e --id SSHFS-Win.SSHFS-Win
"@
    if (-not (Test-Path $SshfsPath)) { exit 1 }
}

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-ExternalCommand {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$ArgumentList
    )

    $stdoutPath = Join-Path $env:TEMP "unifi_api_cmd_${PID}_out.txt"
    $stderrPath = Join-Path $env:TEMP "unifi_api_cmd_${PID}_err.txt"
    Remove-Item $stdoutPath, $stderrPath -Force -ErrorAction SilentlyContinue

    try {
        $proc = Start-Process -FilePath $FilePath -ArgumentList $ArgumentList -NoNewWindow -PassThru -Wait `
            -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath

        $stdout = if (Test-Path $stdoutPath) { [string](Get-Content -Path $stdoutPath -Raw -ErrorAction SilentlyContinue) } else { "" }
        $stderr = if (Test-Path $stderrPath) { [string](Get-Content -Path $stderrPath -Raw -ErrorAction SilentlyContinue) } else { "" }
        $stdout = if ($null -eq $stdout) { "" } else { $stdout }
        $stderr = if ($null -eq $stderr) { "" } else { $stderr }
        $combined = (($stdout.TrimEnd(), $stderr.TrimEnd()) | Where-Object { $_ }) -join "`n"

        return [pscustomobject]@{
            ExitCode = $proc.ExitCode
            Output = $combined
        }
    }
    finally {
        Remove-Item $stdoutPath, $stderrPath -Force -ErrorAction SilentlyContinue
    }
}

$drive = "$($DriveLetter.ToUpperInvariant()):"

if ($Unmount) {
    & net use $drive /delete /y | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "No active mapping for $drive (net use)."
    }
    exit 0
}

if (-not (Test-Path $SshfsPath)) {
    throw "sshfs.exe not found at '$SshfsPath'. Install WinFsp + SSHFS-Win first."
}

Write-Host "[check] validating key-based SSH auth for $User@$TargetHost"
$sshResult = Invoke-ExternalCommand -FilePath "ssh" -ArgumentList @(
    "-o", "BatchMode=yes",
    "-o", "PreferredAuthentications=publickey",
    "$User@$TargetHost",
    "echo KEY_AUTH_OK"
)
if ($sshResult.ExitCode -ne 0 -or $sshResult.Output -notmatch "KEY_AUTH_OK") {
    throw "Key-based SSH auth failed for $User@$TargetHost. Configure SSH keys first."
}

# Enforce key-based workflow and keep connection resilient without password prompts.
$mountArgs = @(
    "$User@$TargetHost`:$RemotePath",
    $drive,
    "-o",
    "reconnect,ServerAliveInterval=10,ServerAliveCountMax=6,StrictHostKeyChecking=accept-new,PreferredAuthentications=publickey,idmap=user,Ciphers=aes128-ctr,Compression=no"
)

Write-Host ("[mount] {0}@{1}:{2} -> {3}" -f $User, $TargetHost, $RemotePath, $drive)
$mountResult = Invoke-ExternalCommand -FilePath $SshfsPath -ArgumentList $mountArgs
if ($mountResult.ExitCode -ne 0) {
    throw "SSHFS mount failed (exit=$($mountResult.ExitCode)). Output: $($mountResult.Output)"
}

Write-Host "[ok] mounted $drive"
