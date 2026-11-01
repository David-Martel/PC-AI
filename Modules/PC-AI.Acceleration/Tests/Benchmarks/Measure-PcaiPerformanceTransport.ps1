#Requires -Version 7.0
[CmdletBinding()]
param(
    [Parameter()]
    [int]$Iterations = 5,

    [Parameter()]
    [string]$DiskPath = (Join-Path $env:USERPROFILE 'Documents\PowerShell'),

    [Parameter()]
    [string]$PcaiRoot
)

if ($PcaiRoot) {
    $env:PCAI_ROOT = $PcaiRoot
}

$moduleRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Import-Module (Join-Path $moduleRoot 'PC-AI.Acceleration.psd1') -Force
$null = Initialize-PcaiNative -Force

$asm = [System.AppDomain]::CurrentDomain.GetAssemblies() |
    Where-Object { $_.GetName().Name -eq 'PcaiNative' } |
    Select-Object -First 1

function Measure-MeanMs {
    param(
        [Parameter(Mandatory)]
        [scriptblock]$Action,

        [Parameter()]
        [int]$Count = 5
    )

    $samples = [System.Collections.Generic.List[double]]::new()
    for ($index = 0; $index -lt $Count; $index++) {
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        & $Action | Out-Null
        $sw.Stop()
        $samples.Add([double]$sw.Elapsed.TotalMilliseconds)
    }

    [pscustomobject]@{
        Mean = [Math]::Round((($samples | Measure-Object -Average).Average), 2)
        Min  = [Math]::Round((($samples | Measure-Object -Minimum).Minimum), 2)
        Max  = [Math]::Round((($samples | Measure-Object -Maximum).Maximum), 2)
    }
}

$typedProc = Measure-MeanMs -Count $Iterations -Action {
    [PcaiNative.PerformanceModule]::GetTopProcesses([uint32]15, 'memory')
}

$jsonProc = Measure-MeanMs -Count $Iterations -Action {
    [PcaiNative.PerformanceModule]::GetTopProcessesJson([uint32]15, 'memory') | ConvertFrom-Json
}

$typedDisk = Measure-MeanMs -Count $Iterations -Action {
    [PcaiNative.PerformanceModule]::GetDiskUsageReport($DiskPath, [uint32]8)
}

$jsonDisk = Measure-MeanMs -Count $Iterations -Action {
    [PcaiNative.PerformanceModule]::GetDiskUsageJson($DiskPath, [uint32]8) | ConvertFrom-Json
}

$procResult = Get-ProcessesFast -Top 5 -SortBy mem
$diskResult = Get-DiskUsageFast -Path $DiskPath -Top 5

[pscustomobject]@{
    AssemblyPath = if ($asm) { $asm.Location } else { $null }
    Methods = if ($asm) {
        $asm.GetType('PcaiNative.PerformanceModule').GetMethods([System.Reflection.BindingFlags]'Public,Static') |
            Select-Object -ExpandProperty Name |
            Sort-Object -Unique
    } else {
        @()
    }
    TypedVsJson = [pscustomobject]@{
        Processes = [pscustomobject]@{
            Typed   = $typedProc
            Json    = $jsonProc
            Speedup = [Math]::Round($jsonProc.Mean / $typedProc.Mean, 2)
        }
        Disk = [pscustomobject]@{
            Typed   = $typedDisk
            Json    = $jsonDisk
            Speedup = [Math]::Round($jsonDisk.Mean / $typedDisk.Mean, 2)
        }
    }
    PublicCalls = [pscustomobject]@{
        ProcessesTool = @($procResult | Select-Object -First 1 -ExpandProperty Tool)
        DiskTool      = @($diskResult | Select-Object -First 1 -ExpandProperty Tool)
        ProcessCount  = @($procResult).Count
        DiskCount     = @($diskResult).Count
    }
}
