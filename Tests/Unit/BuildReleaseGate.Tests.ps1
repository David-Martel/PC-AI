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
    [string]$TestOutcome = 'pass'
)
$ErrorActionPreference = 'Stop'
$SkipQualityGate = $true
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
function Write-BuildSummary { Write-Host 'PCAI_GATE:SUMMARY' }
function New-BuildManifest { Write-Host 'PCAI_GATE:MANIFEST' }
function New-ReleasePackages { Write-Host 'PCAI_GATE:PACKAGE' }
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

    $inferenceFunction = $buildAst.Find({
            param($node)
            $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Invoke-InferenceBuild'
        }, $true)
    . ([scriptblock]::Create($inferenceFunction.Extent.Text))
    function Write-BuildStep {}
    function Resolve-CargoOutputDirectory { throw 'Failed child must not resolve or copy cached artifacts.' }

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
    # or a missing/empty test-failure payload contributing exit status zero.
    # Needs: pwsh, Pester 5 and TestDrive. The real top-level statements call inert stubs in a child.
    # Breadcrumb: Build.ps1 main execution, post-build tests and release output calls.
    It 'routes <Case> to exit <ExitCode> with the expected release writers' -ForEach @(
        @{ Case = 'failed component with requested package'; Arguments = @('-Package', '-BuildFails'); ExitCode = 1; Markers = 'BUILD,SUMMARY' }
        @{ Case = 'failed requested tests with deploy'; Arguments = @('-Deploy', '-RunTests', '-TestOutcome', 'failed'); ExitCode = 1; Markers = 'BUILD,TEST,SUMMARY' }
        @{ Case = 'failed requested tests without failed-suite names'; Arguments = @('-Package', '-RunTests', '-TestOutcome', 'empty'); ExitCode = 1; Markers = 'BUILD,TEST,SUMMARY' }
        @{ Case = 'requested tests without a verdict'; Arguments = @('-Deploy', '-RunTests', '-TestOutcome', 'missing'); ExitCode = 1; Markers = 'BUILD,TEST,SUMMARY' }
        @{ Case = 'successful deploy'; Arguments = @('-Deploy', '-RunTests'); ExitCode = 0; Markers = 'BUILD,TEST,MANIFEST,PACKAGE,DEPLOY,SUMMARY' }
        @{ Case = 'failed diagnostic build without release outputs'; Arguments = @('-BuildFails'); ExitCode = 1; Markers = 'BUILD,MANIFEST,SUMMARY' }
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
    # Detects: artifact lookup or copy continuing after a child exit code other than zero.
    # Needs: Pester 5 and TestDrive; inert child script exits without compiling.
    # Breadcrumb: Build.ps1 Invoke-InferenceBuild artifact collection.
    It 'returns no artifacts when the inference child exits unsuccessfully' {
        $script:ProjectRoot = $TestDrive
        $script:BuildLogsDir = Join-Path $TestDrive 'logs'
        $child = Join-Path $TestDrive 'Native/pcai_core/pcai_inference/Invoke-PcaiBuild.ps1'
        New-Item -ItemType Directory -Path (Split-Path $child -Parent) -Force | Out-Null
        Set-Content -LiteralPath $child -Value @('Write-Output ''synthetic child failure''', 'exit 17')
        Mock Resolve-CargoOutputDirectory { throw 'The build failed before artifact lookup.' }

        $result = Invoke-InferenceBuild -Backend 'llamacpp' -Configuration 'Release' -EnableCuda $false

        $result.Success | Should -BeFalse
        $result.Artifacts | Should -BeNullOrEmpty
        Should -Invoke Resolve-CargoOutputDirectory -Exactly -Times 0
        (Test-Path -LiteralPath $result.LogFile -PathType Leaf) | Should -BeTrue
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
    # Detects: enforcing CUDA admission even though the build never requested it.
    # Needs: Pester 5 and a nonexistent path in TestDrive. No discovery script or hardware.
    # Breadcrumb: Invoke-PcaiBuild.ps1 Initialize-BuildEnvironment conditional CUDA branch.
    It 'does not require CUDA when the switch is absent' {
        $ToolsDir = Join-Path $TestDrive 'absent'
        $EnableCuda = $false
        { & $script:CudaProbe } | Should -Not -Throw
    }
}
