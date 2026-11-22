#Requires -Version 7.0
# Run explicitly with PCAI_TB_TEST_SSH_ALIAS set to a key-authenticated Linux host.
# Executes the genuine generated Bash logic with only hardware/NM I/O replaced.
# Fixtures never invoke real sudo or NetworkManager and remove their temporary files.
BeforeAll {
    if (-not $env:PCAI_TB_TEST_SSH_ALIAS) { throw 'Set PCAI_TB_TEST_SSH_ALIAS for this explicit Linux-runtime fixture suite.' }
    $repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
    . (Join-Path $repoRoot 'Tools/SystemScripts/Networking/Invoke-ThunderboltLinuxPeer.ps1')
    $profile = @{ expectedHostname = 'fixture-peer'; peerAddress = '172.31.240.2'; mtuBytes = 1500 }
    $generated = New-TbLinuxConfigureScript -Profile $profile -Name fixture-peer -Interface thunderbolt0
    function Invoke-ProfileFixture([string]$Scenario) {
        $fixture = @'
set -euo pipefail
fixture_dir=$(mktemp -d)
trap 'rm -f "$fixture_dir/events" "$fixture_dir/output" "$fixture_dir/error"; rmdir "$fixture_dir"' EXIT
events="$fixture_dir/events"
touch "$events"
hostname() { printf '%s\n' fixture-peer; }
readlink() { printf '%s\n' /sys/bus/thunderbolt/drivers/thunderbolt-net; }
cat() { if [ "$1" = /sys/class/net/thunderbolt0/carrier ]; then printf '1\n'; else command cat "$@"; fi; }
sudo() { test "$1" = -n && test "$2" = nmcli || return 97; shift 2; nmcli "$@"; }
nmcli() {
    if [ "$1" = -t ]; then
        if [ "$scenario" != new ]; then printf '11111111-1111-1111-1111-111111111111:pcai-tb-fixture-peer\n'; fi
        if [ "$scenario" = duplicate ]; then printf '22222222-2222-2222-2222-222222222222:pcai-tb-fixture-peer\n'; fi
    elif [ "$1" = -g ]; then
        test "$3 $4 $5 $6" = 'connection show uuid 11111111-1111-1111-1111-111111111111' || return 98
        case "$2" in
            connection.type) printf '802-3-ethernet\n';;
            connection.interface-name) printf 'thunderbolt0\n';;
            connection.autoconnect|ipv4.never-default|ipv4.ignore-auto-routes|ipv4.ignore-auto-dns|ipv6.never-default) printf 'yes\n';;
            ipv4.method) printf 'manual\n';;
            ipv4.addresses) printf '172.31.240.2/30\n';;
            ipv6.method) case "$scenario" in ipv6) printf 'auto\n';; legacy) printf 'disabled\n';; *) printf 'link-local\n';; esac;;
            ipv6.addresses) if [ "$scenario" = ipv6-address ]; then printf 'fd00::2/64\n'; fi;;
            ipv4.routes) if [ "$scenario" = routes ]; then printf '10.0.0.0/8 172.31.240.1\n'; fi;;
            ipv6.gateway) if [ "$scenario" = ipv6-gateway ]; then printf 'fe80::1\n'; fi;;
            ipv6.dns) if [ "$scenario" = ipv6-dns ]; then printf 'fd00::53\n'; fi;;
            ipv6.routes) if [ "$scenario" = ipv6-route ]; then printf 'fd00::/64 fe80::1\n'; fi;;
            ipv4.gateway|ipv4.dns) :;;
            *) return 99;;
        esac
    elif [ "$1" = connection ]; then printf '%s\n' "$*" >> "$events"
    else return 96
    fi
}
set +e
(
'@
        $tail = @'
) >"$fixture_dir/output" 2>"$fixture_dir/error"
result=$?
set -e
python3 - "$result" "$events" "$fixture_dir/error" <<'PY'
import json,sys
with open(sys.argv[2]) as stream: events=stream.read().splitlines()
with open(sys.argv[3]) as stream: error=stream.read()
print(json.dumps({'exitCode':int(sys.argv[1]),'events':events,'error':error}))
PY
'@
        $script = "scenario='$Scenario'`n" + $fixture + "`n" + $generated + "`n" + $tail
        Invoke-TbSsh -SshAlias $env:PCAI_TB_TEST_SSH_ALIAS -Script $script | ConvertFrom-Json
    }
}

Describe 'Generated Linux profile logic executed by Bash' {
    # Protects: migration of the original isolated profile and repeated configuration.
    # Detects: disabling IPv6 or mutating a profile through an ambiguous connection name.
    # Needs: SSH to a Linux host with Bash, Python 3 and fixture-only nmcli.
    # Breadcrumb: Tools/SystemScripts/Networking/Invoke-ThunderboltLinuxPeer.ps1::New-TbLinuxConfigureScript.
    It 'reconciles an isolated <Scenario> profile by UUID and enables link-local IPv6' -ForEach @(
        @{ Scenario = 'valid' }, @{ Scenario = 'legacy' }
    ) {
        $r = Invoke-ProfileFixture $Scenario
        $r.exitCode | Should -Be 0
        $r.events.Count | Should -Be 2
        $r.events[0] | Should -Be 'connection modify uuid 11111111-1111-1111-1111-111111111111 802-3-ethernet.mtu 1500 ipv4.gateway  ipv4.dns  ipv4.never-default yes ipv4.ignore-auto-routes yes ipv4.ignore-auto-dns yes ipv6.method link-local'
        $r.events[1] | Should -Be 'connection up uuid 11111111-1111-1111-1111-111111111111 ifname thunderbolt0'
    }
    # Protects: existing IPv4 routing policy from profile adoption.
    # Detects: activation of a profile with a foreign static route.
    # Needs: SSH to Linux with Bash, Python 3 and fixture-only nmcli.
    # Breadcrumb: Tools/SystemScripts/Networking/Invoke-ThunderboltLinuxPeer.ps1::New-TbLinuxConfigureScript.
    It 'refuses a stored static route before any mutation' {
        $r = Invoke-ProfileFixture routes
        $r.exitCode | Should -Be 1
        $r.events.Count | Should -Be 0
        $r.error | Should -Match 'unexpected ipv4.routes'
    }
    # Protects: a link-local profile with no automatic upstream configuration.
    # Detects: adoption of an IPv6 auto profile before any mutation.
    # Needs: SSH to Linux with Bash, Python 3 and fixture-only nmcli.
    # Breadcrumb: Tools/SystemScripts/Networking/Invoke-ThunderboltLinuxPeer.ps1::New-TbLinuxConfigureScript.
    It 'refuses automatic IPv6 before any mutation' {
        $r = Invoke-ProfileFixture ipv6
        $r.exitCode | Should -Be 1
        $r.events.Count | Should -Be 0
        $r.error | Should -Match 'unexpected ipv6.method'
    }
    # Protects: custody of IPv6 settings already stored in an existing profile.
    # Detects: migration over a manual IPv6 address, gateway, DNS, or route.
    # Needs: SSH to a Linux host with Bash, Python 3 and fixture-only nmcli.
    # Breadcrumb: Tools/SystemScripts/Networking/Invoke-ThunderboltLinuxPeer.ps1::New-TbLinuxConfigureScript.
    It 'refuses stored IPv6 <Field> before any mutation' -ForEach @(
        @{ Scenario = 'ipv6-address'; Field = 'addresses' }, @{ Scenario = 'ipv6-gateway'; Field = 'gateway' },
        @{ Scenario = 'ipv6-dns'; Field = 'dns' }, @{ Scenario = 'ipv6-route'; Field = 'routes' }
    ) {
        $r = Invoke-ProfileFixture $Scenario
        $r.exitCode | Should -Be 1
        $r.events.Count | Should -Be 0
        $r.error | Should -Match "unexpected ipv6\.$Field"
    }
    # Protects: unique profile ownership before NetworkManager operations.
    # Detects: adoption or modification of either of two identically named profiles.
    # Needs: SSH to Linux with Bash, Python 3 and fixture-only nmcli.
    # Breadcrumb: Tools/SystemScripts/Networking/Invoke-ThunderboltLinuxPeer.ps1::New-TbLinuxConfigureScript.
    It 'refuses duplicate connection names before any mutation' {
        $r = Invoke-ProfileFixture duplicate
        $r.exitCode | Should -Be 1
        $r.events.Count | Should -Be 0
        $r.error | Should -Match 'Ambiguous existing Thunderbolt profile'
    }
    # Protects: first installation of a manual IPv4 and link-local IPv6 profile.
    # Detects: a disabled IPv6 default, upstream route enablement, or reversed operation order.
    # Needs: SSH to Linux with Bash, Python 3 and fixture-only nmcli.
    # Breadcrumb: Tools/SystemScripts/Networking/Invoke-ThunderboltLinuxPeer.ps1::New-TbLinuxConfigureScript.
    It 'creates the isolated manual profile before modifying and activating it' {
        $r = Invoke-ProfileFixture new
        $r.exitCode | Should -Be 0
        $r.events.Count | Should -Be 3
        $r.events[0] | Should -Be 'connection add type ethernet ifname thunderbolt0 con-name pcai-tb-fixture-peer ipv4.method manual ipv4.addresses 172.31.240.2/30 ipv4.never-default yes ipv4.ignore-auto-routes yes ipv4.ignore-auto-dns yes ipv6.method link-local ipv6.never-default yes connection.autoconnect yes'
        $r.events[2] | Should -Be 'connection up id pcai-tb-fixture-peer ifname thunderbolt0'
    }
}
