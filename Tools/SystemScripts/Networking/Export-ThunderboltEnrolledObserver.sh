#!/bin/sh
# Fixed read-only forced command for a dedicated Windows SYSTEM observer key.
# Install root-owned at /usr/local/libexec/pcai-usb4-windows-observe.
# Contract: Invoke-ThunderboltEnrolledPeer.ps1 and docs/usb4/windows-boot.md.
set -eu
export LC_ALL=C
test -n "${SSH_CONNECTION:-}"
printf 'PCAI_NODE\n'
/usr/bin/cat /etc/pcai-usb4/node-id
printf '\nPCAI_CONNECTION\n%s\nPCAI_OBSERVATION\n' "$SSH_CONNECTION"
/usr/bin/sudo -n /usr/local/libexec/pcai-usb4-agent observe-current /etc/pcai-usb4/policy.json
printf '\nPCAI_ADDRESSES\n'
/usr/sbin/ip -j address show
