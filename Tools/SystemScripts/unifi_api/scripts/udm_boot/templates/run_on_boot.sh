#!/bin/sh
set -eu

DATA_ROOT="/data"
if [ ! -d "$DATA_ROOT" ] && [ -d "/mnt/data" ]; then
  DATA_ROOT="/mnt/data"
fi

ON_BOOT_DIR="${DATA_ROOT}/on_boot.d"
LOG_FILE="${DATA_ROOT}/scripts/udm-on-boot.log"
LOCK_FILE="/var/lock/udm-on-boot.lock"

mkdir -p "$(dirname "$LOG_FILE")"
touch "$LOG_FILE"

{
  echo "$(date -Iseconds) [runner] starting"

  if [ ! -d "$ON_BOOT_DIR" ]; then
    echo "$(date -Iseconds) [runner] missing ${ON_BOOT_DIR}; exiting"
    exit 0
  fi

  if command -v flock >/dev/null 2>&1; then
    exec 9>"$LOCK_FILE"
    if ! flock -n 9; then
      echo "$(date -Iseconds) [runner] lock busy; exiting"
      exit 0
    fi
  fi

  find "$ON_BOOT_DIR" -mindepth 1 -maxdepth 1 -type f | sort | while IFS= read -r script; do
    if [ -x "$script" ]; then
      echo "$(date -Iseconds) [runner] exec $script"
      "$script" || echo "$(date -Iseconds) [runner] script failed $script"
    else
      case "$script" in
        *.sh)
          echo "$(date -Iseconds) [runner] source $script"
          # shellcheck disable=SC1090
          . "$script" || echo "$(date -Iseconds) [runner] source failed $script"
          ;;
        *)
          echo "$(date -Iseconds) [runner] skip $script"
          ;;
      esac
    fi
  done

  echo "$(date -Iseconds) [runner] complete"
} >>"$LOG_FILE" 2>&1
