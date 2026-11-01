Describe 'Portable runner native argument boundary' -Tag 'Unit', 'Fast', 'Portable' -Skip:($PSVersionTable.PSVersion.Major -lt 7) {
    It 'passes the complete Cargo feature list as one native argument' {
        $source = Join-Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) 'Tests/Invoke-PortableTests.ps1'
        $tokens = $null; $parseErrors = $null
        $ast = [Management.Automation.Language.Parser]::ParseFile($source, [ref]$tokens, [ref]$parseErrors)
        @($parseErrors) | Should -HaveCount 0
        $commands = @($ast.FindAll({
                    param($node)
                    $node -is [Management.Automation.Language.CommandAst] -and $node.GetCommandName() -eq 'cargo' -and
                    $node.CommandElements.Count -gt 1 -and $node.CommandElements[1].Extent.Text -eq 'test'
                }, $true))
        $commands | Should -HaveCount 1
        $command = $commands[0]
        # Preserve the actual argument/redirection bytes; replace only Cargo's
        # executable with a real native pwsh receiver. No Cargo build is needed.
        $tail = $command.Extent.Text.Substring($command.CommandElements[1].Extent.StartOffset - $command.Extent.StartOffset)
        $receiver = Join-Path $TestDrive 'native-argv-receiver.ps1'
        [IO.File]::WriteAllText($receiver, 'ConvertTo-Json -InputObject @($args) -Compress')
        $driver = Join-Path $TestDrive 'native-argv-driver.ps1'
        $native = Join-Path $PSHOME $(if ($IsWindows) { 'pwsh.exe' } else { 'pwsh' })
        $driverText = "& '$($native.Replace("'", "''"))' -NoLogo -NoProfile -File '$($receiver.Replace("'", "''"))' $tail" + "`nexit `$LASTEXITCODE`n"
        [IO.File]::WriteAllText($driver, $driverText)
        $start = [Diagnostics.ProcessStartInfo]::new()
        $start.FileName = $native
        $start.UseShellExecute = $false
        $start.RedirectStandardOutput = $true
        $start.RedirectStandardError = $true
        foreach ($argument in @('-NoLogo', '-NoProfile', '-File', $driver)) { $start.ArgumentList.Add($argument) }
        $process = [Diagnostics.Process]::new()
        $process.StartInfo = $start
        try {
            [void]$process.Start()
            $stdout = $process.StandardOutput.ReadToEndAsync()
            $stderr = $process.StandardError.ReadToEndAsync()
            if (-not $process.WaitForExit(15000)) {
                $process.Kill($true)
                if (-not $process.WaitForExit(5000)) { throw 'Owned native argv child did not confirm termination.' }
                throw 'Owned native argv child exceeded its 15-second bound.'
            }
            $output = $stdout.GetAwaiter().GetResult()
            $process.ExitCode | Should -Be 0 -Because $stderr.GetAwaiter().GetResult()
            $observed = @($output | ConvertFrom-Json)
            Write-Information "Observed Cargo argv: $output" -InformationAction Continue
            $observed | Should -Be @('test', '--workspace', '--no-default-features', '--features', 'server,ffi') -Because $output
        } finally { $process.Dispose() }
    }
}

Describe 'test runner candidate module binding' -Tag 'Unit', 'Fast', 'Portable' -Skip:($PSVersionTable.PSVersion.Major -lt 7) {
    BeforeAll {
        $script:CandidateRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
        $script:BindingChild = Join-Path $TestDrive 'runner-binding-child.ps1'
        @'
param([string]$Runner,[string]$Kind,[int]$Fail)
$ErrorActionPreference='Stop'
Import-Module Pester -RequiredVersion 5.7.1
$global:BindingFailure=$Fail
$global:BindingObservation=$null
function global:Invoke-Pester {
    param($Configuration)
    $global:BindingObservation=[ordered]@{
        First=($env:PSModulePath -split [regex]::Escape([string][IO.Path]::PathSeparator))[0]
        Resolved=@(Get-Module PC-AI.Common -ListAvailable)[0].Path
    }
    if($global:BindingFailure){throw 'Synthetic runner failure before test execution'}
    [pscustomobject]@{TotalCount=1;PassedCount=1;FailedCount=0;SkippedCount=0;NotRunCount=0;FailedContainersCount=0;FailedBlocksCount=0;Duration=[TimeSpan]::Zero;CodeCoverage=$null}
}

$global:BindingDispatch=${function:Invoke-Pester}
function global:Import-Module {
    [CmdletBinding()]
    param([Parameter(Position=0)][string[]]$Name,[version]$MinimumVersion,[version]$RequiredVersion,[switch]$Force)
    Microsoft.PowerShell.Core\Import-Module @PSBoundParameters
    # The runner's real Pester import can re-export Invoke-Pester. Retain only
    # the inert dispatch boundary after forwarding the complete actual import.
    Set-Item -LiteralPath Function:\global:Invoke-Pester -Value $global:BindingDispatch
}
$original='inert-module-path-sentinel'+[IO.Path]::PathSeparator+$env:PSModulePath
$env:PSModulePath=$original
$originalLocation=(Get-Location).Path
$caught=$false
try {
    if($Kind -eq 'Pester'){& $Runner -Type Unit -Tag 'InertBindingFixtureOnly'}
    else {& $Runner -Suite Unit -Tag 'InertBindingFixtureOnly'}
} catch {$caught=$true}
'PCAI_BINDING_RESULT='+([ordered]@{Observed=$global:BindingObservation;Restored=($env:PSModulePath -ceq $original);LocationRestored=((Get-Location).Path -ceq $originalLocation);Caught=$caught}|ConvertTo-Json -Compress)
'@ | Set-Content -LiteralPath $script:BindingChild -Encoding utf8
    }

    # Execute each complete maintained runner with only its test-dispatch boundary
    # inert. Detect stale installed/junction module selection and environment leaks
    # on success and exceptional exit, without discovering/executing product tests.
    It 'pins candidate modules and restores process state for <Runner> failure=<Fail>' -ForEach @(
        @{Runner = '.pester.ps1'; Kind = 'Pester'; Fail = 0 },
        @{Runner = '.pester.ps1'; Kind = 'Pester'; Fail = 1 },
        @{Runner = 'Invoke-AllTests.ps1'; Kind = 'All'; Fail = 0 },
        @{Runner = 'Invoke-AllTests.ps1'; Kind = 'All'; Fail = 1 }
    ) {
        $runnerPath = Join-Path $script:CandidateRoot "Tests/$Runner"
        $start = [Diagnostics.ProcessStartInfo]::new()
        $start.FileName = Join-Path $PSHOME $(if ($IsWindows) { 'pwsh.exe' }else { 'pwsh' })
        $start.UseShellExecute = $false
        $start.WorkingDirectory = $TestDrive
        $start.RedirectStandardOutput = $true
        $start.RedirectStandardError = $true
        foreach ($argument in @('-NoLogo', '-NoProfile', '-File', $script:BindingChild, '-Runner', $runnerPath, '-Kind', $Kind, '-Fail', [string]$Fail)) {
            $start.ArgumentList.Add($argument)
        }
        $process = [Diagnostics.Process]::new()
        $process.StartInfo = $start
        try {
            [void]$process.Start()
            $stdout = $process.StandardOutput.ReadToEndAsync()
            $stderr = $process.StandardError.ReadToEndAsync()
            if (-not $process.WaitForExit(60000)) {
                $process.Kill($true)
                $process.WaitForExit()
                throw 'Owned runner-binding child exceeded its 60-second bound.'
            }
            $output = $stdout.GetAwaiter().GetResult()
            $errorText = $stderr.GetAwaiter().GetResult()
            $process.ExitCode | Should -Be 0 -Because $errorText
            $receiptLine = @($output -split '\r?\n' | Where-Object { $_ -like 'PCAI_BINDING_RESULT=*' })
            $receiptLine.Count | Should -Be 1 -Because $output
            $receipt = $receiptLine[0].Substring('PCAI_BINDING_RESULT='.Length) | ConvertFrom-Json
            $receipt.Observed.First | Should -Be (Join-Path $script:CandidateRoot 'Modules') -Because $output
            $receipt.Observed.Resolved | Should -Be (Join-Path $script:CandidateRoot 'Modules/PC-AI.Common/PC-AI.Common.psm1')
            $receipt.Restored | Should -BeTrue
            $receipt.LocationRestored | Should -BeTrue
            $receipt.Caught | Should -Be ([bool]$Fail)
        }
        finally {
            $process.Dispose()
        }
    }
}
Describe 'Canonical runner plans and preserved cohorts' -Tag 'Unit', 'Fast', 'Portable' {
    BeforeAll {
        $script:PlanRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
        $script:PlanRunner = Join-Path $script:PlanRoot 'Tests/.pester.ps1'
        $script:LegacyRunner = Join-Path $script:PlanRoot 'Tests/Invoke-AllTests.ps1'
    }

    It 'plans covered CI output without importing Pester, dispatching tests, writing or changing process state' {
        Mock Import-Module { throw 'Plan imported a runtime module.' }
        Mock Invoke-Pester { throw 'Plan dispatched Pester.' }
        $location = (Get-Location).Path
        $modulePath = $env:PSModulePath
        $output = Join-Path $TestDrive 'must-not-create'
        $plan = & $script:PlanRunner -Type Unit -Coverage -CI -OutputDirectory $output -Plan
        $plan.Profile | Should -BeExactly 'Unit'
        $plan.Configuration.CodeCoverage.Enabled | Should -BeTrue
        $plan.Configuration.CodeCoverage.Path | Should -Be @((Join-Path $script:PlanRoot 'Modules'))
        $plan.Configuration.CodeCoverage.CoveragePercentTarget | Should -Be 85
        Test-Path -LiteralPath $output | Should -BeFalse
        (Get-Location).Path | Should -BeExactly $location
        $env:PSModulePath | Should -BeExactly $modulePath
        Should -Invoke Import-Module -Times 0 -Exactly
        Should -Invoke Invoke-Pester -Times 0 -Exactly
    }

    It 'retains exactly the original Windows CI cohort and sequence with mandatory all-module coverage' {
        $expected = @(
            'Tests/Unit'
            'Tests/Evaluation/PC-AI.Evaluation.Tests.ps1'
            'Tests/Integration/ModuleLoading.Tests.ps1'
            'Tests/Boot/WorkloadResourceProfile.Tests.ps1'
            'Tests/Boot/CloudClientGate.Tests.ps1'
            'Tests/Boot/ArchiveSourceCustody.Tests.ps1'
            'Tests/Boot/CloudCacheRepairIdentity.Tests.ps1'
            'Tests/Boot/CloudMountGate.Tests.ps1'
            'Tests/Boot/CloudMountGateRunner.Tests.ps1'
            'Tests/Boot/MachineCapabilities.Tests.ps1'
            'Tests/Boot/MediaToolchain.Tests.ps1'
            'Tests/Boot/HardwareMeasurementContracts.Tests.ps1'
            'Tests/Boot/DevModuleDeployment.Tests.ps1'
            'Tests/Boot/FleetToolRouting.Tests.ps1'
            'Tests/Boot/TelemetryPromptContracts.Tests.ps1'
            'Tests/Boot/PackagedRuntimeRoot.Tests.ps1'
            'Tests/Boot/ProfileBootstrap.Tests.ps1'
            'Tests/Boot/CrashProfileSafety.Tests.ps1'
            'Tests/Boot/CrashTelemetry.Tests.ps1'
        )
        $plan = & $script:PlanRunner -Profile WindowsCI -Coverage:$false -Plan
        $plan.DeclaredPaths | Should -Be $expected
        $plan.Configuration.Run.Path | Should -Be @($expected | ForEach-Object { Join-Path $script:PlanRoot $_ })
        $plan.FullWindowsScope | Should -BeTrue
        $plan.Configuration.CodeCoverage.Enabled | Should -BeTrue
        $plan.Configuration.CodeCoverage.CoveragePercentTarget | Should -Be 85
        $plan.Configuration.CodeCoverage.RecursePaths | Should -BeTrue
    }

    It 'preserves the distinct two-cohort and flat four-cohort meanings of All' {
        $canonical = & $script:PlanRunner -Type All -Plan
        $legacy = & $script:LegacyRunner -Suite All -Plan
        $canonical.DeclaredPaths | Should -Be @('Tests/Unit', 'Tests/Integration')
        $expectedGlobs = @('Tests/Unit/*.Tests.ps1', 'Tests/Integration/*.Tests.ps1', 'Tests/E2E/*.Tests.ps1', 'Tests/Functional/*.Tests.ps1')
        $legacy.DeclaredPaths | Should -Be $expectedGlobs
        $expectedFiles = @(foreach ($glob in $expectedGlobs) { Get-ChildItem -Path (Join-Path $script:PlanRoot $glob) -File | ForEach-Object FullName })
        $legacy.Configuration.Run.Path | Should -Be $expectedFiles
    }

    It 'preserves legacy exclusions and report paths while intentionally broadening coverage to all Modules' {
        $plan = & $script:LegacyRunner -Suite Unit -CodeCoverage -Tag Contract -ExcludeTag Other -SkipSlow -OutputFormat JUnitXml -OutputFile 'owned-results/result.xml' -Plan
        $plan.Configuration.Filter.Tag | Should -Be @('Contract')
        $plan.Configuration.Filter.ExcludeTag | Should -Be @('Other', 'Performance', 'Model', 'Slow')
        $plan.Configuration.TestResult.OutputFormat | Should -BeExactly 'JUnitXml'
        $plan.Configuration.TestResult.OutputPath | Should -Be (Join-Path $script:PlanRoot 'Tests/owned-results/result.xml')
        $plan.Configuration.CodeCoverage.Path | Should -Be @((Join-Path $script:PlanRoot 'Modules'))
    }

    It 'selects only a literal requested file and passes the name filter without discovering a cohort' {
        $file = Join-Path $script:PlanRoot 'Tests/Unit/PesterCoverageGate.Tests.ps1'
        $plan = & $script:PlanRunner -Path $file -FullName '*coverage admission*' -Plan
        $plan.Configuration.Run.Path | Should -Be @($file)
        $plan.Configuration.Filter.FullName | Should -Be @('*coverage admission*')
        $plan.Targeted | Should -BeTrue
        $plan.FullWindowsScope | Should -BeFalse
        $plan.Configuration.CodeCoverage.Enabled | Should -BeFalse
    }

    It 'rejects a missing, wildcard or outside-repository target: <Target>' -ForEach @(
        @{Target = 'Tests/Unit/missing-owned.Tests.ps1' }, @{Target = 'Tests/Unit/*.Tests.ps1' }, @{Target = '../outside.Tests.ps1' }
    ) { { & $script:PlanRunner -Path $Target -Plan } | Should -Throw }

    It 'rejects narrowing the Windows acceptance profile' {
        { & $script:PlanRunner -Profile WindowsCI -Tag Unit -Plan } | Should -Throw '*cannot be narrowed*'
        { & $script:PlanRunner -Profile WindowsCI -FullName '*one*' -Plan } | Should -Throw '*cannot be narrowed*'
    }
}

Describe 'Canonical runner actual result propagation' -Tag 'Unit', 'Fast', 'Portable' {
    BeforeAll {
        $script:ResultRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
        $script:ResultRunner = Join-Path $script:ResultRoot 'Tests/.pester.ps1'
        $script:ResultTarget = Join-Path $script:ResultRoot 'Tests/Unit/PesterCoverageGate.Tests.ps1'
    }
    BeforeEach {
        $script:OwnDispatchFixture = $false
        if (Get-Variable -Name PcaiRunnerDispatchFixtureResult -Scope Global -ErrorAction SilentlyContinue) { throw 'Preserve a pre-existing dispatch fixture variable.' }
        $script:DispatchResult = [pscustomobject]@{ TotalCount = 1; PassedCount = 1; FailedCount = 0; SkippedCount = 0; NotRunCount = 0; FailedBlocksCount = 0; FailedContainersCount = 0; CodeCoverage = $null }
        $global:PcaiRunnerDispatchFixtureResult = $script:DispatchResult
        $script:OwnDispatchFixture = $true
        Mock Import-Module { } -ParameterFilter { $Name -eq 'Pester' }
        Mock Invoke-Pester { $global:PcaiRunnerDispatchFixtureResult }
    }
    AfterEach {
        if ($script:OwnDispatchFixture) { Remove-Variable -Name PcaiRunnerDispatchFixtureResult -Scope Global -ErrorAction Stop }
    }
    # The runner and its validator are real. Only its external test-dispatch
    # boundary is inert; these cases give no product or coverage qualification.
    It 'returns only the actual successful dispatch result with PassThru and restores caller state' {
        $location = (Get-Location).Path
        $modulePath = $env:PSModulePath
        $result = @(& $script:ResultRunner -Path $script:ResultTarget -PassThru)
        $result | Should -HaveCount 1
        [object]::ReferenceEquals($result[0], $script:DispatchResult) | Should -BeTrue
        $result[0].PcaiTestPlan.Configuration.Run.Path | Should -Be @($script:ResultTarget)
        (Get-Location).Path | Should -BeExactly $location
        $env:PSModulePath | Should -BeExactly $modulePath
        Should -Invoke Invoke-Pester -Times 1 -Exactly -ParameterFilter { $Configuration.Run.Path.Value -contains $script:ResultTarget }
    }
    It 'throws for real returned <Field>=<Value> even without CI or coverage' -ForEach @(
        @{Field = 'FailedCount'; Value = 1 }, @{Field = 'FailedBlocksCount'; Value = 1 }, @{Field = 'FailedContainersCount'; Value = 1 },
        @{Field = 'TotalCount'; Value = 0 }, @{Field = 'NotRunCount'; Value = 1 }
    ) {
        $script:DispatchResult.$Field = $Value
        $location = (Get-Location).Path
        $modulePath = $env:PSModulePath
        $expectedFailure = if ($Field -eq 'TotalCount') { '*No tests discovered*' } elseif ($Field -eq 'NotRunCount') { '*all discovered tests are NotRun*' } else { "*Pester failures: $Field=$Value*" }
        { & $script:ResultRunner -Path $script:ResultTarget -PassThru } | Should -Throw $expectedFailure
        (Get-Location).Path | Should -BeExactly $location
        $env:PSModulePath | Should -BeExactly $modulePath
        Should -Invoke Invoke-Pester -Times 1 -Exactly
    }
    It 'preserves an external dispatch exception and caller state' {
        Mock Invoke-Pester { throw 'owned dispatch exception sentinel' }
        $location = (Get-Location).Path
        $modulePath = $env:PSModulePath
        { & $script:ResultRunner -Path $script:ResultTarget -PassThru } | Should -Throw '*owned dispatch exception sentinel*'
        (Get-Location).Path | Should -BeExactly $location
        $env:PSModulePath | Should -BeExactly $modulePath
    }
}
