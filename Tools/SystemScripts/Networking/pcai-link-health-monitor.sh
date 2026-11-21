#!/usr/bin/env bash
# Starts pcai-link-health.service when a Thunderbolt/USB4 network interface appears, disappears,
# or changes link or address. Runs as a systemd --user service; needs no root.
# Docker veth churn is ignored: only interfaces whose driver is thunderbolt-net (or that are
# already gone and were named thunderbolt*) trigger a run. Bursts are coalesced by a quiet window.
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
# `ip -o monitor` prints one line per event: "<n>: <ifname>[@...]: <...>" for link events and
# "<n>: <ifname>    inet ..." for address events. Deleted links are prefixed with "Deleted".
while true; do
    # Capture read's own status: after an if without else, $? is 0, not read's timeout status.
    if IFS= read -r -t "$QUIET_SECONDS" line; then
        status=0
    else
        status=$?
    fi
    if [ "$status" -eq 0 ]; then
        line="${line#Deleted }"
        name="$(printf '%s\n' "$line" | sed -n 's/^[0-9][0-9]*: \([^:@ ]*\).*/\1/p')"
        if is_usb4_ifname "$name"; then pending=1; fi
        continue
    fi
    if [ "$status" -gt 128 ]; then
        # Read timed out: the quiet window elapsed.
        if [ "$pending" -eq 1 ]; then
            systemctl --user start --no-block "$UNIT" || true
            pending=0
        fi
        continue
    fi
    # The monitor exited; let systemd restart this service.
    exit 1
done < <(exec ip -o monitor link address)
