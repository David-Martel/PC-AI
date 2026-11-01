#Requires -Version 7.0
[CmdletBinding()]
param(
    [Parameter()]
    [ValidateSet('acceleration-import', 'native-init', 'chat-tui-help', 'service-host-provider-show')]
    [string]$Scenario = 'acceleration-import',

    [Parameter()]
    [string]$OutputRoot,

    [Parameter()]
    [switch]$PassThru
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

. (Join-Path $PSScriptRoot '..\Helpers\Resolve-TestRepoRoot.ps1')

function Resolve-BenchmarkCommandSpec {
    param(
        [Parameter(Mandatory)]
        [string]$Name,
        [Parameter(Mandatory)]
        [string[]]$ExecutableCandidates,
        [string[]]$DllCandidates = @()
    )

    foreach ($candidate in @($ExecutableCandidates | Where-Object { $_ })) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return [PSCustomObject]@{
                Name     = $Name
                Command  = $candidate
                BaseArgs = @()
                Kind     = 'executable'
            }
        }
    }

    foreach ($candidate in @($DllCandidates | Where-Object { $_ })) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return [PSCustomObject]@{
                Name     = $Name
                Command  = 'dotnet'
                BaseArgs = @($candidate)
                Kind     = 'dotnet'
            }
        }
    }

    return $null
}

function Invoke-ProfiledCommand {
    param(
        [Parameter(Mandatory)]
        [psobject]$Spec,
        [Parameter(Mandatory)]
        [string[]]$Arguments,
        [Parameter(Mandatory)]
        [string]$OutputDirectory
    )

    $traceTool = Get-Command 'dotnet-trace' -ErrorAction SilentlyContinue
    $tracePath = Join-Path $OutputDirectory ($Spec.Name + '.nettrace')
    $logPath = Join-Path $OutputDirectory ($Spec.Name + '.stdout.log')

    if ($traceTool) {
        & $traceTool.Source collect --output $tracePath --format NetTrace -- $Spec.Command @($Spec.BaseArgs + $Arguments) *> $logPath
        return [PSCustomObject]@{
            Profiler = 'dotnet-trace'
            Trace    = $tracePath
            Log      = $logPath
            Command  = $Spec.Command
        }
    }

    & $Spec.Command @($Spec.BaseArgs + $Arguments) *> $logPath
    return [PSCustomObject]@{
        Profiler = 'none'
        Trace    = $null
        Log      = $logPath
        Command  = $Spec.Command
    }
}

$PcaiRoot = Resolve-TestRepoRoot -StartPath $PSScriptRoot
$timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
if (-not $OutputRoot) {
    $OutputRoot = Join-Path $PcaiRoot ("Reports\tooling-benchmarks\$timestamp\profiles")
}

$OutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
$null = New-Item -ItemType Directory -Path $OutputRoot -Force

$result = switch ($Scenario) {
    'acceleration-import' {
        $tracePath = Join-Path $OutputRoot 'acceleration-import.trace.txt'
        Trace-Command -Name ModuleImport,CommandDiscovery,ParameterBinding -FilePath $tracePath -Expression {
            Import-Module (Join-Path $PcaiRoot 'Modules\PC-AI.Acceleration\PC-AI.Acceleration.psd1') -Force | Out-Null
        }

        [PSCustomObject]@{
            Scenario = $Scenario
            Profiler = 'Trace-Command'
            Trace    = $tracePath
        }
    }
    'native-init' {
        $tracePath = Join-Path $OutputRoot 'native-init.trace.txt'
        Trace-Command -Name ModuleImport,CommandDiscovery,ParameterBinding -FilePath $tracePath -Expression {
            Import-Module (Join-Path $PcaiRoot 'Modules\PC-AI.Acceleration\PC-AI.Acceleration.psd1') -Force | Out-Null
            Initialize-PcaiNative -Force | Out-Null
        }

        [PSCustomObject]@{
            Scenario = $Scenario
            Profiler = 'Trace-Command'
            Trace    = $tracePath
        }
    }
    'chat-tui-help' {
        $spec = Resolve-BenchmarkCommandSpec -Name 'PcaiChatTui' -ExecutableCandidates @(
            (Join-Path $PcaiRoot 'Native\PcaiChatTui\bin\Release\net8.0\win-x64\publish\PcaiChatTui.exe'),
            (Join-Path $PcaiRoot 'Native\PcaiChatTui\bin\Release\net8.0\win-x64\PcaiChatTui.exe'),
            (Join-Path $PcaiRoot '.pcai\build\artifacts\pcai-chattui\PcaiChatTui.exe')
        ) -DllCandidates @(
            (Join-Path $PcaiRoot 'Native\PcaiChatTui\bin\Release\net8.0\win-x64\PcaiChatTui.dll'),
            (Join-Path $PcaiRoot 'Native\PcaiChatTui\bin\Release\net8.0\PcaiChatTui.dll'),
            (Join-Path $PcaiRoot '.pcai\build\artifacts\pcai-chattui\PcaiChatTui.dll')
        )
        if (-not $spec) {
            throw 'PcaiChatTui build output was not found.'
        }

        $profile = Invoke-ProfiledCommand -Spec $spec -Arguments @('--help') -OutputDirectory $OutputRoot
        $profile | Add-Member -NotePropertyName Scenario -NotePropertyValue $Scenario
        $profile
    }
    'service-host-provider-show' {
        $spec = Resolve-BenchmarkCommandSpec -Name 'PcaiServiceHost' -ExecutableCandidates @(
            (Join-Path $PcaiRoot 'Native\PcaiServiceHost\bin\Release\net8.0\win-x64\publish\PcaiServiceHost.exe'),
            (Join-Path $PcaiRoot 'Native\PcaiServiceHost\bin\Release\net8.0\win-x64\PcaiServiceHost.exe'),
            (Join-Path $PcaiRoot '.pcai\build\artifacts\pcai-servicehost\PcaiServiceHost.exe')
        ) -DllCandidates @(
            (Join-Path $PcaiRoot 'Native\PcaiServiceHost\bin\Release\net8.0\win-x64\PcaiServiceHost.dll'),
            (Join-Path $PcaiRoot 'Native\PcaiServiceHost\bin\Release\net8.0\PcaiServiceHost.dll'),
            (Join-Path $PcaiRoot '.pcai\build\artifacts\pcai-servicehost\PcaiServiceHost.dll')
        )
        if (-not $spec) {
            throw 'PcaiServiceHost build output was not found.'
        }

        $profile = Invoke-ProfiledCommand -Spec $spec -Arguments @('provider', 'show') -OutputDirectory $OutputRoot
        $profile | Add-Member -NotePropertyName Scenario -NotePropertyValue $Scenario
        $profile
    }
}

if ($PassThru) {
    $result
}
else {
    $result | Format-List
}
