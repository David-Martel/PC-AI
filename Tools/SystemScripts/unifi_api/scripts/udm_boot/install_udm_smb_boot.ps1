param(
    [string]$Target = "root@192.168.1.1",
    [string]$ShareName = "udmpro_data",
    [string]$SharePath = "/data/shared",
    [string]$AllowedSubnets = "127.0.0.0/8 10.0.0.0/8 192.168.0.0/16",
    [ValidateSet("user", "guest")]
    [string]$AuthMode = "user",
    [string]$SmbUser = "root",
    [string]$SmbPassword = "",
    [switch]$EnableSmb,
    [switch]$InstallSamba,
    [switch]$NoImmediateRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$templateRunner = Join-Path $scriptRoot "templates/run_on_boot.sh"
$templateSmb = Join-Path $scriptRoot "templates/40-smb-server.sh"

if (-not (Test-Path $templateRunner) -or -not (Test-Path $templateSmb)) {
    throw "Missing templates in $scriptRoot/templates"
}

$remoteTmp = "/tmp/unifi_api_udm_smb_boot"

Write-Host "[step] upload templates"
ssh -o LogLevel=ERROR -o VisualHostKey=no $Target "mkdir -p $remoteTmp"
scp -q $templateRunner "${Target}:${remoteTmp}/run_on_boot.sh"
scp -q $templateSmb "${Target}:${remoteTmp}/40-smb-server.sh"

$installFlag = if ($InstallSamba) { "1" } else { "0" }
$runNowFlag = if ($NoImmediateRun) { "0" } else { "1" }
$enableSmbFlag = if ($EnableSmb) { "1" } else { "0" }

$remoteScript = @'
set -eu

REMOTE_TMP="$1"
SHARE_NAME="$2"
SHARE_PATH="$3"
ALLOWED_SUBNETS="$4"
AUTH_MODE="$5"
SMB_USER="$6"
SMB_PASSWORD="$7"
INSTALL_SAMBA_IF_MISSING="$8"
RUN_NOW="$9"
ENABLE_SMB="${10}"

DATA_ROOT="/data"
if [ ! -d "$DATA_ROOT" ] && [ -d "/mnt/data" ]; then
  DATA_ROOT="/mnt/data"
fi

if [ ! -d "$DATA_ROOT" ]; then
  echo "No persistent data root found (/data or /mnt/data)." >&2
  exit 1
fi

mkdir -p "${DATA_ROOT}/scripts" "${DATA_ROOT}/on_boot.d" "$SHARE_PATH"
chmod 0755 "${DATA_ROOT}/scripts"
chmod 0755 "${DATA_ROOT}/on_boot.d"
chmod 0775 "$SHARE_PATH" || true

install -m 0755 "${REMOTE_TMP}/run_on_boot.sh" "${DATA_ROOT}/scripts/run_on_boot.sh"
install -m 0755 "${REMOTE_TMP}/40-smb-server.sh" "${DATA_ROOT}/on_boot.d/40-smb-server.sh"

cat > "${DATA_ROOT}/scripts/udm_smb.env" <<EOF
SHARE_NAME="${SHARE_NAME}"
SHARE_PATH="${SHARE_PATH}"
ALLOWED_SUBNETS="${ALLOWED_SUBNETS}"
AUTH_MODE="${AUTH_MODE}"
SMB_USER="${SMB_USER}"
INSTALL_SAMBA_IF_MISSING="${INSTALL_SAMBA_IF_MISSING}"
ENABLE_SMB="${ENABLE_SMB}"
EOF

if [ -n "$SMB_PASSWORD" ]; then
  cat >> "${DATA_ROOT}/scripts/udm_smb.env" <<EOF
SMB_PASSWORD="${SMB_PASSWORD}"
EOF
fi
chmod 0600 "${DATA_ROOT}/scripts/udm_smb.env"

if systemctl cat udm-boot.service >/dev/null 2>&1 || systemctl cat udmboot.service >/dev/null 2>&1; then
  echo "Detected existing udm-boot service; using existing on_boot framework."
else
  cat > /etc/systemd/system/udm-custom-onboot.service <<EOF
[Unit]
Description=Run custom /data/on_boot.d scripts
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
ExecStart=${DATA_ROOT}/scripts/run_on_boot.sh
RemainAfterExit=yes
User=root
Group=root

[Install]
WantedBy=multi-user.target
EOF
  systemctl daemon-reload
  systemctl enable udm-custom-onboot.service
  systemctl start udm-custom-onboot.service || true
fi

if [ "$RUN_NOW" = "1" ]; then
  "${DATA_ROOT}/on_boot.d/40-smb-server.sh" || true
fi

if command -v systemctl >/dev/null 2>&1; then
  systemctl is-enabled smbd >/dev/null 2>&1 || true
  systemctl is-active smbd >/dev/null 2>&1 || true
fi

echo "SMB boot automation configured under ${DATA_ROOT}/on_boot.d."
'@

Write-Host "[step] install on target"
$encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($remoteScript))
$remoteCmd = "echo $encoded | base64 -d > $remoteTmp/install.sh && chmod +x $remoteTmp/install.sh && $remoteTmp/install.sh '$remoteTmp' '$ShareName' '$SharePath' '$AllowedSubnets' '$AuthMode' '$SmbUser' '$SmbPassword' '$installFlag' '$runNowFlag' '$enableSmbFlag' && rm -f $remoteTmp/install.sh"
ssh -o LogLevel=ERROR -o VisualHostKey=no $Target $remoteCmd

Write-Host "[step] verify services"
ssh -o LogLevel=ERROR -o VisualHostKey=no $Target "ls -la /data/on_boot.d 2>/dev/null || true; systemctl --no-pager status smbd 2>/dev/null | sed -n '1,80p' || true"

Write-Host "[done] configured SMB on-boot automation for $Target"
