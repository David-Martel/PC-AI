#Requires -Version 7.0

BeforeAll {
    $repoRoot = Join-Path $PSScriptRoot '../..'
    $errors = $null
    $buildAst = [Management.Automation.Language.Parser]::ParseFile(
        (Join-Path $repoRoot 'Build.ps1'), [ref]$null, [ref]$errors)
    if ($errors.Count) { throw ($errors | Out-String) }

    $mainStart = $buildAst.EndBlock.Statements | Where-Object {
        $_.Extent.Text -eq "Write-BuildHeader 'PC_AI Build System'"
    } | Select-Object -First 1
    if (-not $mainStart) { throw 'The top-level build entry point was not found.' }
    $prelude = @'
param(
    [string]$Component = 'llamacpp',
    [switch]$Package,
    [switch]$Deploy,
    [switch]$RunTests,
    [switch]$BuildFails,
    [switch]$PreflightFails,
    [switch]$PackageThrows,
    [string]$TestOutcome = 'pass'
)
$ErrorActionPreference = 'Stop'
$SkipQualityGate = -not $PreflightFails
$Configuration = 'Release'
$FunctionGemmaArgs = @()
function Write-BuildHeader {}
function Write-BuildStep {}
function Write-BuildPhase {}
function Write-BuildResult {}
function Initialize-BuildDirectories {}
function Initialize-BuildVersion {}
function Initialize-CargoToolsDefaults {}
function Set-BuildAccelerationEnvironment {}
function Set-ReleaseBuildFlags {}
function Save-BuildPublicationOutputs { Write-Host 'PCAI_GATE:ARCHIVE' }
function Invoke-AstGrepCheck { Write-Host 'PCAI_GATE:PREFLIGHT'; return $false }
function ConvertTo-GateVerdict { param([bool]$Result) $Result }
function Write-BuildSummary { Write-Host 'PCAI_GATE:SUMMARY' }
function New-BuildManifest { Write-Host 'PCAI_GATE:MANIFEST' }
function New-ReleasePackages {
    Write-Host 'PCAI_GATE:PACKAGE'
    if ($PackageThrows) { throw 'Synthetic package writer failure' }
}
function New-DeployBundle { Write-Host 'PCAI_GATE:DEPLOY' }
function Invoke-InferenceBuild {
    Write-Host 'PCAI_GATE:BUILD'
    @{ Success = -not $BuildFails; Duration = [timespan]::Zero; Artifacts = @('fixture.exe') }
}
function Invoke-StandaloneTests {
    Write-Host 'PCAI_GATE:STANDALONE'
    @{ Success = $TestOutcome -eq 'pass' }
}
function Invoke-PostBuildTests {
    Write-Host 'PCAI_GATE:TEST'
    switch ($TestOutcome) {
        'pass' { @{ Success = $true; Failed = @() } }
        'failed' { @{ Success = $false; Failed = @('fixture') } }
        'empty' { @{ Success = $false; Failed = @() } }
        'missing' { return }
    }
}
'@
    $script:BuildEntryProbe = Join-Path $TestDrive 'build-entry.ps1'
    $suffix = $buildAst.Extent.Text.Substring($mainStart.Extent.StartOffset)
    [IO.File]::WriteAllText($script:BuildEntryProbe, $prelude + [Environment]::NewLine + $suffix)
    $script:BuildProbePwsh = (Get-Command pwsh -CommandType Application | Select-Object -First 1).Source

    $buildFunctionNames = @(
        'Invoke-InferenceBuild', 'Invoke-FunctionGemmaBuild', 'Get-FunctionGemmaCargoArguments',
        'Save-BuildPublicationOutputs', 'Get-BuildDeployBundleName'
    )
    foreach ($functionName in $buildFunctionNames) {
        $definition = $buildAst.Find({
                param($node)
                $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq $functionName
            }, $true)
        if ($definition) { . ([scriptblock]::Create($definition.Extent.Text)) }
    }
    $packageComponents = $buildAst.EndBlock.Statements | Where-Object {
        $_.Extent.Text -like '$script:BuildPackageComponents = *'
    }
    if ($packageComponents) { . ([scriptblock]::Create($packageComponents.Extent.Text)) }
    function Write-BuildStep {}
    function Resolve-CargoOutputDirectory {
        param([string]$ProjectDir, [string]$Configuration)
        $script:FixtureResolverCalls++
        if (-not $script:FixtureCargoOutput) { throw 'The build failed before artifact lookup.' }
        return $script:FixtureCargoOutput
    }
    function Invoke-RustBuildCommand {
        param([string]$Path, [string[]]$CargoArgs, [string]$LogFile)
        $script:FixtureRustBuildCalls++
        if ($Path -like '*rust-functiongemma-train') {
            if ($script:FixtureChildThrows) { throw 'synthetic FunctionGemma child exception' }
            return $false
        }
        return $true
    }

    $cudaAst = [Management.Automation.Language.Parser]::ParseFile(
        (Join-Path $repoRoot 'Native/pcai_core/pcai_inference/Invoke-PcaiBuild.ps1'), [ref]$null, [ref]$errors)
    if ($errors.Count) { throw ($errors | Out-String) }
    $cudaInit = $cudaAst.Find({
            param($node)
            $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Initialize-BuildEnvironment'
        }, $true)
    $cudaBranch = $cudaInit.Body.EndBlock.Statements | Where-Object {
        $_ -is [Management.Automation.Language.IfStatementAst] -and
        $_.Extent.Text -match '^if \(\$EnableCuda' -and
        $_.Extent.Text.Contains('Initialize-CudaEnvironment -Quiet')
    }
    if (@($cudaBranch).Count -ne 1) { throw 'The explicit CUDA selector was not found.' }
    $script:CudaProbe = [scriptblock]::Create($cudaBranch.Extent.Text)
}

Describe 'Build release output admission' -Tag 'Unit', 'Portable' {
    # Protects: real top-level build modes preserve diagnostics but skip every release writer on failure.
    # Detects: manifest/package/deploy calls before checking component or requested-test outcomes,
    # an exception leaving publication outputs current, or a missing/empty test-failure payload.
    # Needs: pwsh, Pester 5 and TestDrive. The real top-level statements call inert stubs in a child.
    # Breadcrumb: Build.ps1 main execution, post-build tests and release output calls.
    It 'routes <Case> to exit <ExitCode> with the expected release writers' -ForEach @(
        @{ Case = 'failed component with requested package'; Arguments = @('-Package', '-BuildFails'); ExitCode = 1; Markers = 'ARCHIVE,BUILD,SUMMARY' }
        @{ Case = 'failed requested tests with deploy'; Arguments = @('-Deploy', '-RunTests', '-TestOutcome', 'failed'); ExitCode = 1; Markers = 'ARCHIVE,BUILD,TEST,SUMMARY' }
        @{ Case = 'failed requested tests without failed-suite names'; Arguments = @('-Package', '-RunTests', '-TestOutcome', 'empty'); ExitCode = 1; Markers = 'ARCHIVE,BUILD,TEST,SUMMARY' }
        @{ Case = 'requested tests without a verdict'; Arguments = @('-Deploy', '-RunTests', '-TestOutcome', 'missing'); ExitCode = 1; Markers = 'ARCHIVE,BUILD,TEST,SUMMARY' }
        @{ Case = 'successful deploy'; Arguments = @('-Deploy', '-RunTests'); ExitCode = 0; Markers = 'ARCHIVE,BUILD,TEST,MANIFEST,PACKAGE,DEPLOY,SUMMARY' }
        @{ Case = 'failed diagnostic build without release outputs'; Arguments = @('-BuildFails'); ExitCode = 1; Markers = 'ARCHIVE,BUILD,SUMMARY' }
        @{ Case = 'failed preflight with deploy'; Arguments = @('-Deploy', '-PreflightFails'); ExitCode = 1; Markers = 'ARCHIVE,PREFLIGHT' }
        @{ Case = 'package writer throws'; Arguments = @('-Deploy', '-PackageThrows'); ExitCode = 1; Markers = 'ARCHIVE,BUILD,MANIFEST,PACKAGE,ARCHIVE,SUMMARY' }
        @{ Case = 'standalone failed tests with release switches'; Arguments = @('-Component', 'test', '-Package', '-Deploy', '-TestOutcome', 'failed'); ExitCode = 1; Markers = 'STANDALONE' }
    ) {
        $PSNativeCommandUseErrorActionPreference = $false
        $output = & $script:BuildProbePwsh -NoLogo -NoProfile -File $script:BuildEntryProbe @Arguments 2>&1
        $LASTEXITCODE | Should -Be $ExitCode -Because ($output -join [Environment]::NewLine)
        $selected = @($output | ForEach-Object {
                if ($_ -match '^PCAI_GATE:(\w+)$') { $Matches[1] }
            })
        ($selected -join ',') | Should -BeExactly $Markers
    }

    # Protects: failed inference subprocesses report no artifacts from the shared Cargo target.
    # Detects: artifact lookup or copy continuing after a nonzero child exit or a thrown error.
    # Needs: Pester 5 and TestDrive. An inert child exits or throws without compiling.
    # Breadcrumb: Build.ps1 Invoke-InferenceBuild artifact collection.
    It 'returns no artifacts when the inference child <Case>' -ForEach @(
        @{ Case = 'exits unsuccessfully'; ChildAction = 'exit 17'; ExpectedError = $null }
        @{ Case = 'throws'; ChildAction = "throw 'synthetic inference child exception'"; ExpectedError = 'synthetic inference child exception' }
    ) {
        $script:ProjectRoot = $TestDrive
        $script:BuildLogsDir = Join-Path $TestDrive 'logs'
        $childPath = Join-Path $TestDrive 'Native/pcai_core/pcai_inference/Invoke-PcaiBuild.ps1'
        New-Item -ItemType Directory -Path (Split-Path $childPath -Parent) -Force | Out-Null
        Set-Content -LiteralPath $childPath -Value @('Write-Output ''synthetic child failure''', $ChildAction)
        $script:FixtureResolverCalls = 0
        $script:FixtureCargoOutput = $null
        $priorExitCode = $global:LASTEXITCODE
        try {
            $global:LASTEXITCODE = 0
            $result = Invoke-InferenceBuild -Backend 'llamacpp' -Configuration 'Release' -EnableCuda $false
        } finally {
            $global:LASTEXITCODE = $priorExitCode
        }

        $result.Success | Should -BeFalse
        $result.Artifacts | Should -BeNullOrEmpty
        $script:FixtureResolverCalls | Should -Be 0
        $result.LogFile | Should -Not -BeNullOrEmpty
        if ($ExpectedError) {
            $result.Error | Should -BeExactly $ExpectedError
        } else {
            (Test-Path -LiteralPath $result.LogFile -PathType Leaf) | Should -BeTrue
        }
    }

    # Protects: prior standard outputs remain recoverable and leave the current publication paths.
    # Detects: stale manifest or package URLs, changed bytes, or an overly broad archive move.
    # Needs: TestDrive and a fixed fixture version. No installed bundles or actual builds.
    # Breadcrumb: Build.ps1 Save-BuildPublicationOutputs path and writer contracts.
    It 'archives only the last publication for the requested mode and variant' {
        $script:BuildRoot = Join-Path $TestDrive 'publication'
        $originalVersion = $env:PCAI_VERSION
        $env:PCAI_VERSION = '2.7.1+fixture'
        try {
            $outputs = [ordered]@{
                'artifacts/manifest.json' = 'last-good manifest'
                'packages/pcai-llamacpp-cpu-win64.zip' = 'last-good package'
                'packages/pc-ai-bundle-2.7.1_fixture-cpu-win64.zip' = 'last-good bundle archive'
                'deploy/pc-ai-bundle-2.7.1_fixture-cpu-win64/payload.txt' = 'last-good bundle'
            }
            $untouched = @(
                'packages/pcai-llamacpp-cuda-win64.zip',
                'packages/pc-ai-bundle-2.6.1_fixture-cpu-win64.zip',
                'packages/unrelated-cpu-win64.zip',
                'artifacts/pcai-llamacpp/cached.exe'
            )
            foreach ($entry in $outputs.GetEnumerator()) {
                $destination = Join-Path $script:BuildRoot $entry.Key
                New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
                Set-Content -LiteralPath $destination -Value $entry.Value
            }
            foreach ($relative in $untouched) {
                $destination = Join-Path $script:BuildRoot $relative
                New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
                Set-Content -LiteralPath $destination -Value 'unrelated'
            }

            Save-BuildPublicationOutputs -EnableCuda $false -Package $true -Deploy $true
            Save-BuildPublicationOutputs -EnableCuda $false -Package $true -Deploy $true

            $archive = @(Get-ChildItem -LiteralPath (Join-Path $script:BuildRoot 'previous-outputs') -Directory)
            $archive.Count | Should -Be 1
            foreach ($entry in $outputs.GetEnumerator()) {
                Test-Path -LiteralPath (Join-Path $script:BuildRoot $entry.Key) | Should -BeFalse
                Get-Content -LiteralPath (Join-Path $archive[0].FullName $entry.Key) | Should -BeExactly $entry.Value
            }
            foreach ($relative in $untouched) {
                Get-Content -LiteralPath (Join-Path $script:BuildRoot $relative) | Should -BeExactly 'unrelated'
            }
        } finally {
            $env:PCAI_VERSION = $originalVersion
        }
    }

    # Protects: an output root reached through a junction or symlink cannot move files outside its named path.
    # Detects: probing the build folder through a linked parent before detecting the link.
    # Needs: TestDrive and junctions on Windows or symlinks on Linux. All paths are owned fixture data.
    # Breadcrumb: Build.ps1 Save-BuildPublicationOutputs ancestor validation.
    It 'refuses an output root reached through a linked parent' {
        $target = Join-Path $TestDrive 'archive-junction-target'
        $alias = Join-Path $TestDrive 'archive-junction-parent'
        $artifacts = Join-Path $target 'build/artifacts'
        New-Item -ItemType Directory -Path $artifacts -Force | Out-Null
        $manifest = Join-Path $artifacts 'manifest.json'
        Set-Content -LiteralPath $manifest -Value 'still current'
        $linkType = if ($IsWindows) { 'Junction' } else { 'SymbolicLink' }
        New-Item -ItemType $linkType -Path $alias -Target $target | Out-Null
        try {
            $script:BuildRoot = Join-Path $alias 'build'
            { Save-BuildPublicationOutputs -EnableCuda $false -Package $false -Deploy $false } |
                Should -Throw '*A regular directory is required*'
            Get-Content -LiteralPath $manifest | Should -BeExactly 'still current'
            Test-Path -LiteralPath (Join-Path $target 'build/previous-outputs') | Should -BeFalse
        } finally {
            Remove-Item -LiteralPath $alias -Force
        }
    }

    # Protects: the FunctionGemma collector runs only after both child crates succeed.
    # Detects: cached executable or DLL selection after the second crate fails or throws.
    # Needs: TestDrive and mocked Rust child commands. No Cargo or compiler.
    # Breadcrumb: Build.ps1 Invoke-FunctionGemmaBuild child loop and collection.
    It 'returns no FunctionGemma artifacts when the training child <Case>' -ForEach @(
        @{ Case = 'reports failure'; Throws = $false }
        @{ Case = 'throws'; Throws = $true }
    ) {
        $script:ProjectRoot = Join-Path $TestDrive 'functiongemma-root'
        $script:BuildLogsDir = Join-Path $script:ProjectRoot 'logs'
        $script:BuildArtifactsDir = Join-Path $script:ProjectRoot 'artifacts'
        $cache = Join-Path $script:ProjectRoot 'cache'
        foreach ($relative in @('Deploy/rust-functiongemma-runtime', 'Deploy/rust-functiongemma-train', 'cache')) {
            New-Item -ItemType Directory -Path (Join-Path $script:ProjectRoot $relative) -Force | Out-Null
        }
        Set-Content -LiteralPath (Join-Path $cache 'cached.exe') -Value 'stale exe'
        Set-Content -LiteralPath (Join-Path $cache 'cached.dll') -Value 'stale dll'
        $script:FixtureRustBuildCalls = 0
        $script:FixtureResolverCalls = 0
        $script:FixtureChildThrows = $Throws
        $script:FixtureCargoOutput = $cache

        $result = Invoke-FunctionGemmaBuild -Configuration Release

        $result.Success | Should -BeFalse
        $result.Artifacts | Should -BeNullOrEmpty
        $result.LogFile | Should -Not -BeNullOrEmpty
        if ($Throws) { $result.Error | Should -BeExactly 'synthetic FunctionGemma child exception' }
        $script:FixtureRustBuildCalls | Should -Be 2
        $script:FixtureResolverCalls | Should -Be 0
        Test-Path -LiteralPath $script:BuildArtifactsDir | Should -BeFalse
    }
}

Describe 'Explicit inference CUDA build admission' -Tag 'Unit', 'Portable' {
    # Protects: explicit CUDA requests fail before native compilation if discovery is unavailable.
    # Detects: an absent toolkit or discovery script falling through to a CPU build.
    # Needs: Pester 5 and owned temp scripts; uses the real CUDA branch with an inert helper.
    # Breadcrumb: Invoke-PcaiBuild.ps1 Initialize-BuildEnvironment CUDA selection.
    It 'rejects explicit CUDA with <Case>' -ForEach @(
        @{ Case = 'missing helper'; Helper = $false; Message = '*toolkit discovery helper is missing*' }
        @{ Case = 'missing toolkit'; Helper = $true; Message = '*no complete CUDA Toolkit was found*' }
    ) {
        $ToolsDir = Join-Path $TestDrive $Case
        New-Item -ItemType Directory -Path $ToolsDir -Force | Out-Null
        if ($Helper) {
            Set-Content -LiteralPath (Join-Path $ToolsDir 'Initialize-CudaEnvironment.ps1') -Value @'
function Initialize-CudaEnvironment {
    param([switch]$Quiet)
    [pscustomobject]@{ Found = $false; CudaPath = $null }
}
'@
        }
        $EnableCuda = $true
        { & $script:CudaProbe } | Should -Throw $Message
    }

    # Protects: CPU-only builds are independent of the CUDA toolkit and its discovery helper.
    # Detects: attempting toolkit discovery even though the build never requested it.
    # Needs: Pester 5 and an inert helper that records a load in TestDrive.
    # Breadcrumb: Invoke-PcaiBuild.ps1 Initialize-BuildEnvironment conditional CUDA branch.
    It 'does not require CUDA when the switch is absent' {
        $ToolsDir = Join-Path $TestDrive 'cpu-only'
        New-Item -ItemType Directory -Path $ToolsDir -Force | Out-Null
        Set-Content -LiteralPath (Join-Path $ToolsDir 'Initialize-CudaEnvironment.ps1') -Value @'
Set-Content -LiteralPath (Join-Path $PSScriptRoot 'cuda-loaded') -Value 'loaded'
'@
        $EnableCuda = $false
        & $script:CudaProbe
        Test-Path -LiteralPath (Join-Path $ToolsDir 'cuda-loaded') | Should -BeFalse
    }
}
