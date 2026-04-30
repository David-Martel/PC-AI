param(
    [string]$SmbMountPath = "F:\udm_smb",
    [string]$RcloneMountPath = "W:\udm_rclone",
    [string]$RcloneRemote = "udm_sftp",
    [string]$RcloneRemotePath = "/data"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$targetSpec = "mount $RcloneRemote`:$RcloneRemotePath $RcloneMountPath"
$procs = Get-CimInstance Win32_Process -Filter "Name='rclone.exe'" -ErrorAction SilentlyContinue | Where-Object {
    $_.CommandLine -like "*$targetSpec*"
}

foreach ($proc in $procs) {
    Stop-Process -Id $proc.ProcessId -Force -ErrorAction SilentlyContinue
    Write-Host "[ok] stopped rclone PID $($proc.ProcessId)"
}

if (Test-Path $SmbMountPath) {
    $item = Get-Item -LiteralPath $SmbMountPath -Force
    if ($item.LinkType) {
        Remove-Item -LiteralPath $SmbMountPath -Force
        Write-Host "[ok] removed SMB symlink $SmbMountPath"
    }
}
