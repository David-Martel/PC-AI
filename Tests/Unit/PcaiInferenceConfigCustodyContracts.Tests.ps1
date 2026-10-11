#Requires -Version 5.1
Describe 'Inference configuration retention at the native launch boundary' -Tag 'Unit','Virtualization' -Skip:($PSVersionTable.PSEdition -ne 'Core' -or [Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT) {
    BeforeAll {
        $script:Source = if ($env:PCAI_CONFIG_CONTRACT_SOURCE) { $env:PCAI_CONFIG_CONTRACT_SOURCE } else { Join-Path $PSScriptRoot '../../Modules/PC-AI.Virtualization/Public/Invoke-PcaiServiceHost.ps1' }
        . $script:Source
        function Resolve-PcaiRepoRoot { param($StartPath) $script:CaseRoot }
        $script:WitnessRoot = if ($env:PCAI_CONFIG_WITNESS_ROOT) { $env:PCAI_CONFIG_WITNESS_ROOT } else { Join-Path $TestDrive 'retained-config-evidence' }
        $null = [IO.Directory]::CreateDirectory($script:WitnessRoot)
        $buildScript = Join-Path $TestDrive 'build-owned-config-reader.ps1'
        $script:ReaderPath = Join-Path $TestDrive 'owned-config-reader.exe'
        $readerCompiler = Join-Path $env:SystemRoot 'System32/WindowsPowerShell/v1.0/powershell.exe'
        $readerBuild = @'
param([Parameter(Mandatory)][string]$OutputAssembly)
$ErrorActionPreference = 'Stop'
if ([IO.File]::Exists($OutputAssembly)) { throw 'Retain existing owned executable.' }
Add-Type -TypeDefinition @"
using System;
using System.IO;
using System.Text;
public static class OwnedConfigReader {
    public static int Main(string[] args) {
        if (args.Length != 2 || args[0] != "--config") return 7;
        try { Console.WriteLine(File.ReadAllText(args[1], Encoding.UTF8)); return 0; }
        catch (Exception) { Console.Error.WriteLine("Synthetic config reader could not read its input."); return 7; }
    }
}
"@ -OutputAssembly $OutputAssembly -OutputType ConsoleApplication
'@
        [IO.File]::WriteAllText($buildScript, $readerBuild, [Text.UTF8Encoding]::new($false))
        & $readerCompiler -NoLogo -NoProfile -File $buildScript -OutputAssembly $script:ReaderPath
        if ($LASTEXITCODE -ne 0) { throw 'Owned config-reader build failed.' }
        [IO.File]::Copy($buildScript, (Join-Path $script:WitnessRoot 'fixture-reader-build.ps1'), $false)
        [IO.File]::Copy($script:ReaderPath, (Join-Path $script:WitnessRoot 'fixture-built-reader.exe'), $false)

        $script:CaseRevision = 0
        function Get-ConfigContractNamespace {
            param([string]$Root)
            $prefix = [IO.Path]::GetFullPath($Root).TrimEnd('\','/') + [IO.Path]::DirectorySeparatorChar
            @(Get-ChildItem -LiteralPath $Root -Recurse -Force | ForEach-Object {
                if (-not $_.FullName.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) { throw 'Namespace escaped owned root.' }
                [ordered]@{ Path=$_.FullName.Substring($prefix.Length); Directory=$_.PSIsContainer; Length=if ($_.PSIsContainer) { 0 } else { $_.Length }; Hash=if ($_.PSIsContainer) { $null } else { (Get-FileHash -LiteralPath $_.FullName).Hash } }
            })
        }
        function Save-ConfigContractObservation {
            param([string]$Name, $Observation)
            $snapshotRoot = Join-Path $script:WitnessRoot ($Name + '-files')
            $null = [IO.Directory]::CreateDirectory($snapshotRoot)
            $retained = @()
            $fileRevision = 0
            foreach ($file in (Get-ConfigContractNamespace $script:CaseRoot | Where-Object { -not $_.Directory })) {
                $fileRevision++
                $copy = Join-Path $snapshotRoot ('file-r' + $fileRevision + '.bin')
                [IO.File]::Copy((Join-Path $script:CaseRoot $file.Path), $copy, $false)
                $copiedHash = (Get-FileHash -LiteralPath $copy).Hash
                if ($copiedHash -cne $file.Hash) { throw 'Retained raw bytes differ.' }
                $retained += [ordered]@{ OriginalRelativePath=$file.Path; RetainedPath=$copy; Length=$file.Length; Hash=$copiedHash }
            }
            $Observation.Add('RetainedRawFiles', $retained)
            $Observation | ConvertTo-Json -Depth 9 | Set-Content -LiteralPath (Join-Path $script:WitnessRoot ($Name + '.json'))
        }
    }
    BeforeEach {
        $script:CaseRevision++
        $script:CaseRoot = Join-Path $TestDrive ('case-r' + $script:CaseRevision)
        $null = New-Item -ItemType Directory -Path (Join-Path $script:CaseRoot 'Native/pcai_core/pcai_inference/target/release') -Force
        $null = New-Item -ItemType Directory -Path (Join-Path $script:CaseRoot 'Config')
        Copy-Item -LiteralPath $script:ReaderPath -Destination (Join-Path $script:CaseRoot 'Native/pcai_core/pcai_inference/target/release/pcai-llamacpp.exe')
        $script:ModelA = Join-Path $script:CaseRoot 'synthetic-model-a.gguf'
        $script:ModelB = Join-Path $script:CaseRoot 'synthetic-model-b.gguf'
        [IO.File]::WriteAllText($script:ModelA, 'Synthetic config marker, never parsed or loaded.')
        [IO.File]::WriteAllText($script:ModelB, 'Second synthetic config marker, never parsed or loaded.')
        @{ providers=@{ 'pcai-native'=@{ gpuLayers=12; defaultMaxTokens=41; defaultTemperature=0.3 } }; router=@{ enabled=$false } } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $script:CaseRoot 'Config/llm-config.json')
        Mock Test-Path {
            $candidate = [string]$Path
            if ([string]::IsNullOrEmpty($candidate)) { $candidate = [string]$LiteralPath }
            $prefix = [IO.Path]::GetFullPath($script:CaseRoot).TrimEnd('\','/') + [IO.Path]::DirectorySeparatorChar
            if ($candidate.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) { [IO.File]::Exists($candidate) -or [IO.Directory]::Exists($candidate) }
            else { $false }
        }
        Mock Get-Date { '20000101_000000' } -ParameterFilter { $Format -eq 'yyyyMMdd_HHmmss' }
        Mock Invoke-RestMethod { throw 'Network probing is forbidden.' }
        Mock Start-Process { throw 'Unowned process launch is forbidden.' }
    }
    It 'hands the actual serialized configuration to the private native reader' {
        $before = Get-ConfigContractNamespace $script:CaseRoot
        $result = Start-RustInferenceServer -NativeBackend llamacpp -ModelPath $script:ModelA -Port 19081 -GpuLayers 0
        $received = $result.Output | ConvertFrom-Json
        Save-ConfigContractObservation single-call ([ordered]@{ Before=$before; After=(Get-ConfigContractNamespace $script:CaseRoot); ExitCode=$result.ExitCode; Args=$result.Args; Received=$received; NativeBoundary='Owned synthetic config-reader executable; not an inference runtime' })
        $result.ExitCode | Should -Be 0
        $received.server.port | Should -Be 19081
        $received.model.path | Should -BeExactly $script:ModelA
        $received.backend.n_gpu_layers | Should -Be 0
        $received.model.generation.max_tokens | Should -Be 41
        $received.router.enabled | Should -BeFalse
        (Get-Content -LiteralPath $result.Args[1] -Raw | ConvertFrom-Json).server.port | Should -Be 19081
    }
    It 'keeps two same-second launches bound to independent retained configuration bytes' {
        $before = Get-ConfigContractNamespace $script:CaseRoot
        $first = Start-RustInferenceServer -NativeBackend llamacpp -ModelPath $script:ModelA -Port 19081 -GpuLayers 0
        $firstHash = (Get-FileHash -LiteralPath $first.Args[1]).Hash
        $firstRetainedCopy = Join-Path $script:WitnessRoot 'same-second-first-config.bin'
        [IO.File]::Copy($first.Args[1], $firstRetainedCopy, $false)
        $second = Start-RustInferenceServer -NativeBackend llamacpp -ModelPath $script:ModelB -Port 19082 -GpuLayers 2
        $retainedHash = (Get-FileHash -LiteralPath $first.Args[1]).Hash
        Save-ConfigContractObservation same-second ([ordered]@{ Before=$before; After=(Get-ConfigContractNamespace $script:CaseRoot); FirstArgs=$first.Args; SecondArgs=$second.Args; FirstHash=$firstHash; FirstRetainedHash=$retainedHash; FirstRawCopy=$firstRetainedCopy; FirstRawCopyHash=(Get-FileHash -LiteralPath $firstRetainedCopy).Hash; FirstReceived=($first.Output|ConvertFrom-Json); SecondReceived=($second.Output|ConvertFrom-Json); ExitCodes=@($first.ExitCode,$second.ExitCode) })
        $first.ExitCode | Should -Be 0
        $second.ExitCode | Should -Be 0
        ($first.Output | ConvertFrom-Json).server.port | Should -Be 19081
        ($second.Output | ConvertFrom-Json).server.port | Should -Be 19082
        $second.Args[1] | Should -Not -BeExactly $first.Args[1]
        $retainedHash | Should -BeExactly $firstHash
        (Get-Content -LiteralPath $first.Args[1] -Raw | ConvertFrom-Json).model.path | Should -BeExactly $script:ModelA
    }
    It 'preserves both a preexisting timestamp config and a retained stable config' {
        $runtime = Join-Path $script:CaseRoot '.pcai/runtime/pcai-inference'
        $null = New-Item -ItemType Directory -Path $runtime -Force
        $dated = Join-Path $runtime 'config-20000101_000000.json'
        $stable = Join-Path $runtime 'config-r1.json'
        $directoryCollision = Join-Path $runtime 'config-r2.json'
        $null = [IO.Directory]::CreateDirectory($directoryCollision)
        [IO.File]::WriteAllText($dated, 'Unique retained timestamp config bytes.')
        [IO.File]::WriteAllText($stable, 'Unique retained stable config bytes.')
        [IO.File]::Copy($dated, (Join-Path $script:WitnessRoot 'existing-timestamp-config-before.bin'), $false)
        [IO.File]::Copy($stable, (Join-Path $script:WitnessRoot 'existing-stable-config-before.bin'), $false)
        $before = Get-ConfigContractNamespace $script:CaseRoot
        $datedHash = (Get-FileHash -LiteralPath $dated).Hash
        $stableHash = (Get-FileHash -LiteralPath $stable).Hash
        $result = Start-RustInferenceServer -NativeBackend llamacpp -ModelPath $script:ModelA -Port 19081 -GpuLayers 0
        Save-ConfigContractObservation existing-configs ([ordered]@{ Before=$before; After=(Get-ConfigContractNamespace $script:CaseRoot); Args=$result.Args; ExitCode=$result.ExitCode; DatedBefore=$datedHash; DatedAfter=(Get-FileHash -LiteralPath $dated).Hash; StableBefore=$stableHash; StableAfter=(Get-FileHash -LiteralPath $stable).Hash })
        $result.ExitCode | Should -Be 0
        (Get-FileHash -LiteralPath $dated).Hash | Should -BeExactly $datedHash
        (Get-FileHash -LiteralPath $stable).Hash | Should -BeExactly $stableHash
        [IO.Directory]::Exists($directoryCollision) | Should -BeTrue
        $result.Args[1] | Should -Not -BeIn @($dated,$stable,$directoryCollision)
    }
    It 'refuses a missing model marker before creating runtime configuration' {
        $before = Get-ConfigContractNamespace $script:CaseRoot
        { Start-RustInferenceServer -NativeBackend llamacpp -ModelPath (Join-Path $script:CaseRoot 'absent.gguf') -Port 19081 } | Should -Throw '*model file not found*'
        $expectedNamespace = $before | ConvertTo-Json -Depth 6
        $actualNamespace = Get-ConfigContractNamespace $script:CaseRoot | ConvertTo-Json -Depth 6
        Save-ConfigContractObservation missing-model ([ordered]@{ Before=$before; After=(Get-ConfigContractNamespace $script:CaseRoot) })
        $actualNamespace | Should -BeExactly $expectedNamespace
    }
    It 'fails closed on a non-collision I/O error without modifying retained bytes' {
        $runtimeParent = Join-Path $script:CaseRoot '.pcai/runtime'
        $null = [IO.Directory]::CreateDirectory($runtimeParent)
        [IO.File]::WriteAllText((Join-Path $runtimeParent 'pcai-inference'), 'Owned file blocking the runtime directory.')
        $before = Get-ConfigContractNamespace $script:CaseRoot
        { Start-RustInferenceServer -NativeBackend llamacpp -ModelPath $script:ModelA -Port 19081 } | Should -Throw
        $expectedNamespace = $before | ConvertTo-Json -Depth 6
        $actualNamespace = Get-ConfigContractNamespace $script:CaseRoot | ConvertTo-Json -Depth 6
        Save-ConfigContractObservation invalid-runtime ([ordered]@{ Before=$before; After=(Get-ConfigContractNamespace $script:CaseRoot) })
        $actualNamespace | Should -BeExactly $expectedNamespace
    }
}
