#!/usr/bin/env bash
# Starts pcai-link-health.service when a Thunderbolt/USB4 network interface appears, disappears,
# or changes link or address. Runs as a systemd --user service; needs no root.
# Docker veth churn is ignored: only interfaces whose driver is thunderbolt-net (or that are
# already gone and were named thunderbolt*) trigger a run. A burst is coalesced into one run that
# starts QUIET_SECONDS after its first Thunderbolt event.
set -u
QUIET_SECONDS="${PCAI_LINK_HEALTH_QUIET_SECONDS:-10}"
UNIT="${PCAI_LINK_HEALTH_UNIT:-pcai-link-health.service}"

is_usb4_ifname() {
    local name="$1" drv
    [ -z "$name" ] && return 1
    if [ -e "/sys/class/net/$name/device/driver" ]; then
        drv="$(basename "$(readlink -f "/sys/class/net/$name/device/driver")")"
        [ "$drv" = "thunderbolt-net" ] || [ "$drv" = "thunderbolt_net" ]
        return
    fi
    case "$name" in thunderbolt*|tb*) return 0 ;; esac
    return 1
}

pending=0
fire_at=0
# `ip -o monitor` prints one line per event: "<n>: <ifname>[@...]: <...>" for link events and
# "<n>: <ifname>    inet ..." for address events. Deleted links are prefixed with "Deleted".
# The quiet window starts at the first Thunderbolt event and is not extended by unrelated
# events (Docker veth churn can arrive more often than the window), so a deadline is kept.
while true; do
    if [ "$pending" -eq 1 ]; then
        timeout_s=$((fire_at - SECONDS))
        [ "$timeout_s" -lt 1 ] && timeout_s=1
    else
        timeout_s=3600
    fi
    # Capture read's own status: after an if without else, $? is 0, not read's timeout status.
    if IFS= read -r -t "$timeout_s" line; then
        status=0
    else
        status=$?
    fi
    if [ "$status" -eq 0 ]; then
        line="${line#Deleted }"
        name="$(printf '%s\n' "$line" | sed -n 's/^[0-9][0-9]*: \([^:@ ]*\).*/\1/p')"
        if is_usb4_ifname "$name" && [ "$pending" -eq 0 ]; then
            pending=1
            fire_at=$((SECONDS + QUIET_SECONDS))
        fi
    elif [ "$status" -le 128 ]; then
        # The monitor exited; let systemd restart this service.
        exit 1
    fi
    if [ "$pending" -eq 1 ] && [ "$SECONDS" -ge "$fire_at" ]; then
        systemctl --user start --no-block "$UNIT" || true
        pending=0
    fi
done < <(exec ip -o monitor link address)
