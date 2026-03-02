#Requires -Version 5.1
<#
.SYNOPSIS
    Starts WinSocat HVSOCK<->TCP proxies based on configuration.
#>
function Start-HVSockProxy {
    [CmdletBinding()]
    [OutputType([PSCustomObject])]
    param(
        [Parameter()]
        [string]$ConfigPath = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path 'Config\hvsock-proxy.conf'),

        [Parameter()]
        [string]$StatePath = "$env:ProgramData\PC_AI\hvsock-proxy\state.json",

        [Parameter()]
        [switch]$Force,

        [Parameter()]
        [switch]$RegisterServices
    )

    if (-not (Test-Path $ConfigPath)) {
        throw "HVSOCK proxy config not found: $ConfigPath"
    }

    $winsocat = Get-Command winsocat.exe -ErrorAction SilentlyContinue
    if (-not $winsocat) {
        throw 'WinSocat not found. Run Install-HVSockProxy first.'
    }

    if ($RegisterServices) {
        try {
            Register-HVSockServices -ConfigPath $ConfigPath -Force:$Force | Out-Null
        } catch {
            Write-Warning "Failed to register HVSOCK services: $_"
        }
    }

    $stateDir = Split-Path $StatePath -Parent
    if (-not (Test-Path $stateDir)) {
        New-Item -ItemType Directory -Path $stateDir -Force | Out-Null
    }

    $entries = [System.Collections.Generic.List[PSObject]]::new()
    $lines = Get-Content $ConfigPath | ForEach-Object { $_.Trim() } | Where-Object { $_ -and -not $_.StartsWith('#') }

    foreach ($line in $lines) {
        $parts = $line.Split(':')
        if ($parts.Count -lt 4) { continue }

        $name = $parts[0]
        $serviceId = $parts[1]
        $tcpHost = $parts[2]
        $tcpPort = $parts[3]

        $args = "HVSock-LISTEN:$serviceId TCP:${tcpHost}:$tcpPort"
        $proc = Start-Process -FilePath $winsocat.Path -ArgumentList $args -PassThru -WindowStyle Hidden

        $entries.Add([PSCustomObject]@{
            Name = $name
            ServiceId = $serviceId
            TcpTarget = "${tcpHost}:$tcpPort"
            Pid = $proc.Id
            Command = "$($winsocat.Path) $args"
            Started = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
        })
    }

    $entries | ConvertTo-Json -Depth 4 | Set-Content -Path $StatePath -Encoding UTF8

    return [PSCustomObject]@{
        ConfigPath = $ConfigPath
        StatePath = $StatePath
        Count = $entries.Count
        Proxies = $entries
    }
}
