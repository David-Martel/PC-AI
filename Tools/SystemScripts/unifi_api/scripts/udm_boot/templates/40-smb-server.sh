#!/bin/sh
set -eu

DATA_ROOT="/data"
if [ ! -d "$DATA_ROOT" ] && [ -d "/mnt/data" ]; then
  DATA_ROOT="/mnt/data"
fi

ENV_FILE="${DATA_ROOT}/scripts/udm_smb.env"
if [ -f "$ENV_FILE" ]; then
  # shellcheck disable=SC1090
  . "$ENV_FILE"
fi

SHARE_NAME="${SHARE_NAME:-udmpro_data}"
SHARE_PATH="${SHARE_PATH:-${DATA_ROOT}/shared}"
ALLOWED_SUBNETS="${ALLOWED_SUBNETS:-127.0.0.0/8 10.0.0.0/8 192.168.0.0/16}"
AUTH_MODE="${AUTH_MODE:-user}"          # user | guest
SMB_USER="${SMB_USER:-root}"
INSTALL_SAMBA_IF_MISSING="${INSTALL_SAMBA_IF_MISSING:-0}"
ENABLE_SMB="${ENABLE_SMB:-0}"
LOG_FILE="${DATA_ROOT}/scripts/udm-smb.log"
BACKUP_ROOT="${DATA_ROOT}/scripts/backups/samba"
BACKUP_CONF="${BACKUP_ROOT}/smb.conf.pre_unifi_api"

mkdir -p "$(dirname "$LOG_FILE")"
touch "$LOG_FILE"
mkdir -p "$BACKUP_ROOT"

log() {
  echo "$(date -Iseconds) [smb] $*" >>"$LOG_FILE"
}

if ! command -v smbd >/dev/null 2>&1; then
  if [ "$INSTALL_SAMBA_IF_MISSING" = "1" ] && command -v apt-get >/dev/null 2>&1; then
    log "samba missing; installing"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update >>"$LOG_FILE" 2>&1 || true
    apt-get install -y samba >>"$LOG_FILE" 2>&1 || true
  fi
fi

if ! command -v smbd >/dev/null 2>&1; then
  log "samba not installed; skipping"
  exit 0
fi

mkdir -p "$SHARE_PATH"
chmod 0775 "$SHARE_PATH" || true

SMB_CONF="/etc/samba/smb.conf"
if [ ! -f "$SMB_CONF" ]; then
  log "missing ${SMB_CONF}; skipping"
  exit 0
fi

if [ ! -f "$BACKUP_CONF" ]; then
  cp "$SMB_CONF" "$BACKUP_CONF"
  chmod 0600 "$BACKUP_CONF" || true
  log "created backup ${BACKUP_CONF}"
fi

BEGIN_MARKER="# BEGIN UDM SMB BOOT SHARE (managed by unifi_api)"
END_MARKER="# END UDM SMB BOOT SHARE (managed by unifi_api)"
TMP_CONF="$(mktemp)"
TMP_BLOCK="$(mktemp)"

if [ "$ENABLE_SMB" != "1" ]; then
  awk -v begin="$BEGIN_MARKER" -v end="$END_MARKER" '
    $0 == begin {in_block=1; next}
    $0 == end {in_block=0; next}
    !in_block {print}
  ' "$SMB_CONF" >"$TMP_CONF"
  cp "$TMP_CONF" "$SMB_CONF"
  rm -f "$TMP_CONF" "$TMP_BLOCK"
  if command -v systemctl >/dev/null 2>&1; then
    systemctl restart smbd >>"$LOG_FILE" 2>&1 || true
    systemctl restart nmbd >>"$LOG_FILE" 2>&1 || true
  fi
  log "ENABLE_SMB=0; managed share disabled"
  exit 0
fi

{
  echo "$BEGIN_MARKER"
  echo "[$SHARE_NAME]"
  echo "path = $SHARE_PATH"
  echo "comment = UDM Pro persistent SMB share"
  echo "browseable = yes"
  echo "create mask = 0664"
  echo "directory mask = 0775"
  echo "hosts allow = $ALLOWED_SUBNETS"
  echo "hosts deny = all"
  if [ "$AUTH_MODE" = "guest" ]; then
    echo "guest ok = yes"
    echo "read only = no"
    echo "force user = root"
  else
    echo "guest ok = no"
    echo "read only = no"
    echo "valid users = $SMB_USER"
  fi
  echo "$END_MARKER"
} >"$TMP_BLOCK"

awk -v begin="$BEGIN_MARKER" -v end="$END_MARKER" '
  $0 == begin {in_block=1; next}
  $0 == end {in_block=0; next}
  !in_block {print}
' "$SMB_CONF" >"$TMP_CONF"

cat "$TMP_BLOCK" >>"$TMP_CONF"
cp "$TMP_CONF" "$SMB_CONF"
rm -f "$TMP_CONF" "$TMP_BLOCK"

if [ -n "${SMB_PASSWORD:-}" ] && [ "$AUTH_MODE" = "user" ]; then
  if pdbedit -L | grep -q "^${SMB_USER}:"; then
    log "updating samba password for ${SMB_USER}"
    printf '%s\n%s\n' "$SMB_PASSWORD" "$SMB_PASSWORD" | smbpasswd -s "$SMB_USER" >>"$LOG_FILE" 2>&1 || true
  else
    log "creating samba user ${SMB_USER}"
    printf '%s\n%s\n' "$SMB_PASSWORD" "$SMB_PASSWORD" | smbpasswd -s -a "$SMB_USER" >>"$LOG_FILE" 2>&1 || true
  fi
fi

if command -v testparm >/dev/null 2>&1; then
  testparm -s >>"$LOG_FILE" 2>&1 || true
fi

if command -v systemctl >/dev/null 2>&1; then
  systemctl enable smbd >>"$LOG_FILE" 2>&1 || true
  systemctl restart smbd >>"$LOG_FILE" 2>&1 || true
  systemctl restart nmbd >>"$LOG_FILE" 2>&1 || true
fi

log "share applied name=${SHARE_NAME} path=${SHARE_PATH} auth_mode=${AUTH_MODE}"
