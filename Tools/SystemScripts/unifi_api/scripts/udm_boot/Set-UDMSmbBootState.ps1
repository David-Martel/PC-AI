param(
    [string]$Target = "root@192.168.1.1",
    [ValidateSet("enable", "disable")]
    [string]$State = "disable",
    [string]$ShareName = "udmpro_data",
    [string]$SharePath = "/data/shared",
    [string]$AllowedSubnets = "127.0.0.0/8 10.0.0.0/8 192.168.0.0/16",
    [ValidateSet("user", "guest")]
    [string]$AuthMode = "user",
    [string]$SmbUser = "root",
    [string]$SmbPassword = "",
    [switch]$InstallSamba,
    [switch]$NoImmediateRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$installer = Join-Path $scriptRoot "install_udm_smb_boot.ps1"
if (-not (Test-Path $installer)) {
    throw "Installer not found: $installer"
}

$params = @{
    Target = $Target
    ShareName = $ShareName
    SharePath = $SharePath
    AllowedSubnets = $AllowedSubnets
    AuthMode = $AuthMode
    SmbUser = $SmbUser
}
if ($SmbPassword) { $params.SmbPassword = $SmbPassword }
if ($InstallSamba) { $params.InstallSamba = $true }
if ($NoImmediateRun) { $params.NoImmediateRun = $true }
if ($State -eq "enable") { $params.EnableSmb = $true }

Write-Host "[step] applying SMB boot state '$State' on $Target"
& $installer @params
if ($LASTEXITCODE -ne 0) {
    throw "Failed to apply SMB boot state '$State'."
}
Write-Host "[ok] SMB boot state '$State' applied."
