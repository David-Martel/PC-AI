param(
    [string]$ConfigPath = (Join-Path $HOME ".ssh/config"),
    [string]$UdmAlias = "udmpro",
    [string]$UdmHost = "192.168.1.1",
    [string]$UdmUser = "root",
    [string]$IdentityFile = "~/.ssh/id_ed25519",
    [string]$UnvrAlias = "unvr",
    [string]$UnvrHost = "",
    [string]$UnvrUser = "root",
    [switch]$IncludeProxyTemplate = $true
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-PathLikeSsh([string]$PathValue) {
    if (-not $PathValue) { return "" }
    if ($PathValue.StartsWith("~/")) {
        return Join-Path $HOME ($PathValue.Substring(2))
    }
    if ($PathValue.StartsWith('~\')) {
        return Join-Path $HOME ($PathValue.Substring(2))
    }
    return [Environment]::ExpandEnvironmentVariables($PathValue)
}

function Find-DefaultIdentityFile {
    $sshDir = Join-Path $HOME ".ssh"
    if (-not (Test-Path $sshDir)) { return "" }

    $files = Get-ChildItem -Path $sshDir -File | Where-Object {
        $_.Name -notmatch "\.pub$" -and
        $_.Name -notmatch "^(config|known_hosts|known_hosts\.old|authorized_keys|pageant\.conf)$"
    }

    $preferred = @("ed25519", "ecdsa", "rsa", "dsa")
    foreach ($algo in $preferred) {
        $match = $files | Where-Object { $_.Name -match $algo } | Select-Object -First 1
        if ($match) {
            return "~/.ssh/$($match.Name)"
        }
    }

    $fallback = $files | Select-Object -First 1
    if ($fallback) {
        return "~/.ssh/$($fallback.Name)"
    }

    return ""
}

$dir = Split-Path -Parent $ConfigPath
if (-not (Test-Path $dir)) {
    New-Item -ItemType Directory -Path $dir -Force | Out-Null
}

if (-not (Test-Path $ConfigPath)) {
    New-Item -ItemType File -Path $ConfigPath -Force | Out-Null
}

$existing = Get-Content -Path $ConfigPath -Raw -Encoding UTF8
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$backupPath = "$ConfigPath.bak.$timestamp"
Copy-Item -Path $ConfigPath -Destination $backupPath -Force

$beginMarker = "# BEGIN unifi_api managed ssh block"
$endMarker = "# END unifi_api managed ssh block"
$resolvedIdentityPath = Resolve-PathLikeSsh $IdentityFile
if ($IdentityFile -and -not (Test-Path $resolvedIdentityPath)) {
    $IdentityFile = Find-DefaultIdentityFile
}
$hasIdentityFile = -not [string]::IsNullOrWhiteSpace($IdentityFile)
$identitiesOnlyValue = if ($hasIdentityFile) { "yes" } else { "no" }

$managed = @()
$managed += $beginMarker
$managed += "Host $UdmAlias"
$managed += "  HostName $UdmHost"
$managed += "  User $UdmUser"
$managed += "  IdentitiesOnly $identitiesOnlyValue"
if ($hasIdentityFile) {
    $managed += "  IdentityFile $IdentityFile"
}
$managed += "  PreferredAuthentications publickey"
$managed += "  ServerAliveInterval 20"
$managed += "  ServerAliveCountMax 3"
$managed += "  TCPKeepAlive yes"
$managed += "  ConnectTimeout 10"
$managed += "  Compression yes"
$managed += "  StrictHostKeyChecking accept-new"
$managed += ""

if ($UnvrHost) {
    $managed += "Host $UnvrAlias"
    $managed += "  HostName $UnvrHost"
    $managed += "  User $UnvrUser"
    $managed += "  IdentitiesOnly $identitiesOnlyValue"
    if ($hasIdentityFile) {
        $managed += "  IdentityFile $IdentityFile"
    }
    $managed += "  PreferredAuthentications publickey"
    $managed += "  ServerAliveInterval 20"
    $managed += "  ServerAliveCountMax 3"
    $managed += "  TCPKeepAlive yes"
    $managed += "  ConnectTimeout 10"
    $managed += "  Compression yes"
    $managed += "  StrictHostKeyChecking accept-new"
    $managed += ""
}

if ($IncludeProxyTemplate) {
    $managed += "Host unifi-device-*"
    $managed += "  User root"
    $managed += "  ProxyJump $UdmAlias"
    $managed += "  IdentitiesOnly $identitiesOnlyValue"
    if ($hasIdentityFile) {
        $managed += "  IdentityFile $IdentityFile"
    }
    $managed += "  PreferredAuthentications publickey"
    $managed += "  ServerAliveInterval 20"
    $managed += "  ServerAliveCountMax 3"
    $managed += "  ConnectTimeout 8"
    $managed += "  StrictHostKeyChecking accept-new"
    $managed += ""
}

$managed += "# ControlMaster is intentionally omitted on Windows OpenSSH due unstable behavior in local tests."
$managed += $endMarker
$managedBlock = ($managed -join "`n") + "`n"

$escapedBegin = [regex]::Escape($beginMarker)
$escapedEnd = [regex]::Escape($endMarker)
$pattern = "(?ms)^$escapedBegin.*?^$escapedEnd\r?\n?"
if ($existing -match $pattern) {
    $updated = [regex]::Replace($existing, $pattern, $managedBlock)
}
else {
    if ($existing.Length -gt 0 -and -not $existing.EndsWith("`n")) {
        $existing += "`n"
    }
    $updated = $existing + "`n" + $managedBlock
}

Set-Content -Path $ConfigPath -Value $updated -Encoding UTF8 -NoNewline

Write-Host "[ok] updated $ConfigPath"
Write-Host "[ok] backup  $backupPath"
