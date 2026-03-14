#Requires -Version 7.0
[CmdletBinding()]
param(
    [Parameter()]
    [string[]]$ComputerNameCandidates = @(
        'DESKTOP-86FC90B',
        'dtm-supernuc',
        'dtm-work',
        'dtm-work-mobo'
    ),

    [Parameter()]
    [int]$WinRmPort = 5985,

    [Parameter()]
    [int]$TcpTimeoutMs = 1500,

    [Parameter()]
    [switch]$IncludeRaw
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$driversManifest = Join-Path $repoRoot 'Modules\PC-AI.Drivers\PC-AI.Drivers.psd1'
Import-Module $driversManifest -Force

Find-ThunderboltPeer -ComputerNameCandidates $ComputerNameCandidates -WinRmPort $WinRmPort -TcpTimeoutMs $TcpTimeoutMs -IncludeRaw:$IncludeRaw
