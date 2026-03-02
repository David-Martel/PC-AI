#Requires -Version 5.1
<#
.SYNOPSIS
    Registers Hyper-V socket (HVSOCK) services for WSL guests.

.DESCRIPTION
    Creates GuestCommunicationServices registry keys so WSL can connect to
    host HVSOCK services. Uses the VSock service GUID format described by
    Microsoft (0000xxxx-facb-11e6-bd58-64006a7986d3).
#>
function Register-HVSockServices {
    [CmdletBinding(SupportsShouldProcess)]
    [OutputType([PSCustomObject])]
    param(
        [Parameter()]
        [string]$ConfigPath,

        [Parameter()]
        [switch]$Force
    )

    if (-not $ConfigPath) {
        if (Get-Command Resolve-PcaiPath -ErrorAction SilentlyContinue) {
            $ConfigPath = Resolve-PcaiPath -PathType 'HVSockConfig'
        } elseif ($env:PCAI_ROOT) {
            $ConfigPath = Join-Path $env:PCAI_ROOT 'Config\hvsock-proxy.conf'
        } else {
            $repoRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
            $ConfigPath = Join-Path $repoRoot 'Config\hvsock-proxy.conf'
        }
    }

    $isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    if (-not $isAdmin) {
        throw 'Register-HVSockServices requires Administrator privileges.'
    }

    if (-not (Test-Path $ConfigPath)) {
        throw "HVSOCK proxy config not found: $ConfigPath"
    }

    $baseKey = 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Virtualization\GuestCommunicationServices'
    if (-not (Test-Path $baseKey)) {
        New-Item -Path $baseKey -Force | Out-Null
    }

    $results = @()
    $lines = Get-Content $ConfigPath | ForEach-Object { $_.Trim() } | Where-Object { $_ -and -not $_.StartsWith('#') }

    foreach ($line in $lines) {
        $parts = $line.Split(':')
        if ($parts.Count -lt 4) { continue }

        $name = $parts[0]
        $serviceIdRaw = $parts[1]
        $serviceGuid = $null
        $port = $null

        if ($serviceIdRaw -match '^vsock-(\d+)$') {
            $port = [int]$Matches[1]
            $hex = '{0:x8}' -f $port
            $serviceGuid = "$hex-facb-11e6-bd58-64006a7986d3"
        } elseif ($serviceIdRaw -match '^\d+$') {
            $port = [int]$serviceIdRaw
            $hex = '{0:x8}' -f $port
            $serviceGuid = "$hex-facb-11e6-bd58-64006a7986d3"
        } elseif ($serviceIdRaw -match '^[\{]?[0-9a-fA-F-]{36}[\}]?$') {
            $serviceGuid = $serviceIdRaw.Trim('{}')
        } else {
            Write-Warning "Unsupported service id format: $serviceIdRaw"
            continue
        }

        $keyPath = Join-Path $baseKey "{$serviceGuid}"
        $elementName = "PC_AI:$name"

        if ($PSCmdlet.ShouldProcess($keyPath, 'Register HVSOCK service')) {
            if (-not (Test-Path $keyPath)) {
                New-Item -Path $keyPath -Force | Out-Null
            }
            if ($Force -or -not (Get-ItemProperty -Path $keyPath -Name ElementName -ErrorAction SilentlyContinue)) {
                New-ItemProperty -Path $keyPath -Name ElementName -Value $elementName -PropertyType String -Force | Out-Null
            }
        }

        $results += [PSCustomObject]@{
            Name = $name
            ServiceId = $serviceIdRaw
            Guid = $serviceGuid
            Port = $port
            RegistryPath = $keyPath
            Registered = Test-Path $keyPath
        }
    }

    return $results
}
