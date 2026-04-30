param(
    [string]$HostAlias = "udmpro-capability-test",
    [string]$HostName = "192.168.1.1",
    [string]$User = "root"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$tmpCfg = Join-Path $env:TEMP "ssh-capabilities-$($PID).conf"

$configText = @"
Host jump-test
  HostName 10.0.0.10
  User jump

Host $HostAlias
  HostName $HostName
  User $User
  ServerAliveInterval 20
  ServerAliveCountMax 3
  ConnectTimeout 10
  Compression yes
  ProxyJump jump-test
  ControlMaster auto
  ControlPersist 10m
  ControlPath ~/.ssh/cm-%r@%h:%p
"@

Set-Content -Path $tmpCfg -Value $configText -Encoding ascii

try {
    $sshVersion = (cmd /c "ssh -V 2>&1" | Out-String).Trim()
    $resolved = ssh -G $HostAlias -F $tmpCfg 2>$null
    if (-not $resolved) {
        throw "Unable to resolve ssh config with ssh -G."
    }

    $kv = @{}
    foreach ($line in $resolved) {
        if ($line -match "^\s*([^\s]+)\s+(.+)$") {
            $kv[$matches[1].ToLowerInvariant()] = $matches[2]
        }
    }

    $report = [ordered]@{
        ssh_version = $sshVersion
        supports_server_alive = $kv.ContainsKey("serveraliveinterval") -and $kv.ContainsKey("serveralivecountmax")
        supports_proxy_jump = $kv.ContainsKey("proxyjump")
        supports_control_master = $kv.ContainsKey("controlmaster")
        supports_control_persist = $kv.ContainsKey("controlpersist")
        supports_control_path = $kv.ContainsKey("controlpath")
        resolved_values = [ordered]@{
            serveraliveinterval = $kv["serveraliveinterval"]
            serveralivecountmax = $kv["serveralivecountmax"]
            proxyjump = $kv["proxyjump"]
            controlmaster = $kv["controlmaster"]
            controlpersist = $kv["controlpersist"]
            controlpath = $kv["controlpath"]
            connecttimeout = $kv["connecttimeout"]
            compression = $kv["compression"]
        }
    }

    $report | ConvertTo-Json -Depth 4
}
finally {
    Remove-Item $tmpCfg -Force -ErrorAction SilentlyContinue
}
