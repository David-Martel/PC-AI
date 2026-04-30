param(
    [string]$Target = "root@192.168.1.1",
    [string]$BackupPath = "/data/scripts/backups/samba/smb.conf.pre_unifi_api",
    [switch]$RestartServices = $true
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$remote = @'
set -eu
BACKUP_PATH="$1"
RESTART_SERVICES="$2"

if [ ! -f "$BACKUP_PATH" ]; then
  echo "Backup not found: $BACKUP_PATH" >&2
  exit 2
fi

cp "$BACKUP_PATH" /etc/samba/smb.conf
chmod 0644 /etc/samba/smb.conf || true

if [ "$RESTART_SERVICES" = "1" ] && command -v systemctl >/dev/null 2>&1; then
  systemctl restart smbd || true
  systemctl restart nmbd || true
fi

echo "Restored /etc/samba/smb.conf from $BACKUP_PATH"
'@

$encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($remote))
$restartFlag = if ($RestartServices) { "1" } else { "0" }
$cmd = "echo $encoded | base64 -d > /tmp/unifi_api_restore_smb.sh && chmod +x /tmp/unifi_api_restore_smb.sh && /tmp/unifi_api_restore_smb.sh '$BackupPath' '$restartFlag' ; rc=`$?; rm -f /tmp/unifi_api_restore_smb.sh; exit `$rc"

Write-Host "[step] restoring Samba config on $Target from $BackupPath"
ssh -o LogLevel=ERROR -o VisualHostKey=no $Target $cmd
if ($LASTEXITCODE -ne 0) {
    throw "Samba restore failed."
}
Write-Host "[ok] restore completed."
