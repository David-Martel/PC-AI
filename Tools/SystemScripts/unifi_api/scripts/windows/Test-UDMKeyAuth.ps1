param(
    [string[]]$Targets = @("root@192.168.1.1"),
    [string]$ProxyJump = "",
    [string]$OutputDir = "docs/commands",
    [switch]$NoOutputFile
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$results = @()
foreach ($target in $Targets) {
    $args = @(
        "-o", "BatchMode=yes",
        "-o", "PreferredAuthentications=publickey",
        "-o", "ConnectTimeout=8",
        "-o", "ServerAliveInterval=20",
        "-o", "ServerAliveCountMax=2"
    )
    if ($ProxyJump) {
        $args += @("-o", "ProxyJump=$ProxyJump")
    }
    $args += @($target, "echo KEY_AUTH_OK")

    $stdoutPath = Join-Path $env:TEMP "unifi_api_ssh_smoke_${PID}_out.txt"
    $stderrPath = Join-Path $env:TEMP "unifi_api_ssh_smoke_${PID}_err.txt"
    Remove-Item $stdoutPath, $stderrPath -Force -ErrorAction SilentlyContinue

    try {
        $start = Get-Date
        $proc = Start-Process -FilePath "ssh" -ArgumentList $args -NoNewWindow -PassThru -Wait `
            -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
        $durationMs = [math]::Round(((Get-Date) - $start).TotalMilliseconds, 2)
        $stdout = if (Test-Path $stdoutPath) { [string](Get-Content -Path $stdoutPath -Raw -ErrorAction SilentlyContinue) } else { "" }
        $stderr = if (Test-Path $stderrPath) { [string](Get-Content -Path $stderrPath -Raw -ErrorAction SilentlyContinue) } else { "" }
        $stdout = if ($null -eq $stdout) { "" } else { $stdout }
        $stderr = if ($null -eq $stderr) { "" } else { $stderr }
        $combinedParts = @()
        if (-not [string]::IsNullOrWhiteSpace($stdout)) { $combinedParts += $stdout.TrimEnd() }
        if (-not [string]::IsNullOrWhiteSpace($stderr)) { $combinedParts += $stderr.TrimEnd() }
        $combined = $combinedParts -join "`n"
        $ok = ($proc.ExitCode -eq 0 -and $combined -match "KEY_AUTH_OK")

        $results += [pscustomobject]@{
            target = $target
            proxy_jump = $ProxyJump
            ok = $ok
            exit_code = $proc.ExitCode
            duration_ms = $durationMs
            output = $combined
            timestamp = (Get-Date).ToString("o")
        }
    }
    finally {
        Remove-Item $stdoutPath, $stderrPath -Force -ErrorAction SilentlyContinue
    }
}

$results | Format-Table -AutoSize | Out-Host

if (-not $NoOutputFile) {
    New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $file = Join-Path $OutputDir "ssh_key_auth_smoketest_$stamp.json"
    $results | ConvertTo-Json -Depth 5 | Set-Content -Path $file -Encoding UTF8
    Write-Host "[ok] wrote $file"
}

if ($results.ok -contains $false) {
    exit 1
}
