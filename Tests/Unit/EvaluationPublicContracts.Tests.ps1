#Requires -Version 7.0

# Public consumer controls. No resolver/body extraction or internal orchestration mocks.
BeforeAll {
    $script:evaluationRepoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
    $script:publicEvaluationModuleOwned = $false
    $script:publicEvaluationOwnedImportModules = @()
    $script:publicEvaluationSetupFailureOriginal = $null
    $script:publicEvaluationOwnershipFailureOriginals = [Collections.Generic.List[object]]::new()
    $script:publicModuleRemovalFailureOriginals = [Collections.Generic.List[object]]::new()
    function Remove-PublicEvaluationOwnedImports {
        $unresolved = [Collections.Generic.List[Management.Automation.PSModuleInfo]]::new()
        foreach ($ownedModule in $script:publicEvaluationOwnedImportModules) {
            try { Remove-Module -ModuleInfo $ownedModule -ErrorAction Stop }
            catch {
                $script:publicModuleRemovalFailureOriginals.Add($_)
                $unresolved.Add($ownedModule)
            }
        }
        $script:publicEvaluationOwnedImportModules = $unresolved.ToArray()
    }
    $script:publicAmbientEvaluationModules = @(Get-Module -Name PC-AI.Evaluation)
    $expectedModulePath = [IO.Path]::GetFullPath((Join-Path $evaluationRepoRoot 'Modules/PC-AI.Evaluation/PC-AI.Evaluation.psm1'))
    if ($publicAmbientEvaluationModules.Count -gt 1) {
        throw 'An ambiguous ambient evaluation module must be preserved.'
    }
    if ($publicAmbientEvaluationModules.Count -eq 1) {
        $script:publicEvaluationModule = $publicAmbientEvaluationModules[0]
        if ([IO.Path]::GetFullPath($publicEvaluationModule.Path) -cne $expectedModulePath) {
            throw 'A different ambient evaluation module must be preserved.'
        }
    }
    else {
        function Select-PublicEvaluationImportedModule {
            param([object[]]$ReturnedModules, [string]$RootModulePath, [string]$ValidationScriptPath)
            $matches = [Collections.Generic.List[Management.Automation.PSModuleInfo]]::new()
            foreach ($returnedModule in $ReturnedModules) {
                if ($returnedModule -isnot [Management.Automation.PSModuleInfo] -or [string]::IsNullOrWhiteSpace($returnedModule.Path)) {
                    throw 'Evaluation manifest import returned a nonmodule or pathless object.'
                }
                $returnedPath = [IO.Path]::GetFullPath($returnedModule.Path)
                if ($returnedPath -ceq $RootModulePath) { $matches.Add($returnedModule) }
                elseif ($returnedPath -cne $ValidationScriptPath) {
                    throw 'Evaluation manifest import returned an unexpected module path.'
                }
            }
            if ($matches.Count -ne 1) { throw 'Evaluation manifest import must return one exact root module.' }
            return $matches[0]
        }
        # ScriptsToProcess can add its own PSModuleInfo to PassThru output.
        $script:publicAmbientImportModules = @(Get-Module -All)
        $validationScriptPath = [IO.Path]::GetFullPath((Join-Path $evaluationRepoRoot 'Modules/PC-AI.Evaluation/ValidateDependencies.ps1'))
        $script:publicEvaluationImportResults = [Collections.Generic.List[object]]::new()
        $script:publicEvaluationPostImportModules = @()
        try {
            try {
                # Preserve each original output even if import subsequently terminates.
                Import-Module (Join-Path $evaluationRepoRoot 'Modules/PC-AI.Evaluation/PC-AI.Evaluation.psd1') -PassThru -WarningAction SilentlyContinue -ErrorAction Stop |
                    ForEach-Object { $script:publicEvaluationImportResults.Add($_) }
            }
            finally {
                # Loaded original references can survive an import failure without output.
                try { $script:publicEvaluationPostImportModules = @(Get-Module -All) }
                catch { $script:publicEvaluationOwnershipFailureOriginals.Add($_) }
                $observedModules = $publicEvaluationImportResults.ToArray() + $publicEvaluationPostImportModules
                $ownedModules = [Collections.Generic.List[Management.Automation.PSModuleInfo]]::new()
                foreach ($declaredPath in @($expectedModulePath, $validationScriptPath)) {
                    foreach ($observedModule in $observedModules) {
                        if ($observedModule -isnot [Management.Automation.PSModuleInfo] -or [string]::IsNullOrWhiteSpace($observedModule.Path)) { continue }
                        try { $observedPath = [IO.Path]::GetFullPath($observedModule.Path) }
                        catch { $script:publicEvaluationOwnershipFailureOriginals.Add($_); continue }
                        if ($observedPath -cne $declaredPath) { continue }
                        $alreadyKnown = $false
                        foreach ($knownModule in @($publicAmbientImportModules) + $ownedModules.ToArray()) {
                            if ([object]::ReferenceEquals($observedModule, $knownModule)) { $alreadyKnown = $true; break }
                        }
                        if (-not $alreadyKnown) { $ownedModules.Add($observedModule) }
                    }
                }
                $script:publicEvaluationOwnedImportModules = $ownedModules.ToArray()
            }
            if ($publicEvaluationOwnershipFailureOriginals.Count -gt 0) { throw $publicEvaluationOwnershipFailureOriginals[0] }
            $script:publicEvaluationModule = Select-PublicEvaluationImportedModule -ReturnedModules $publicEvaluationImportResults.ToArray() -RootModulePath $expectedModulePath -ValidationScriptPath $validationScriptPath
            $script:publicEvaluationModuleOwned = $true
        }
        catch {
            $script:publicEvaluationSetupFailureOriginal = $_
            Remove-PublicEvaluationOwnedImports
            throw $script:publicEvaluationSetupFailureOriginal
        }
    }

    function New-PublicObservedSuite {
        param([double]$Value = 0.5)
        $suite = New-EvaluationSuite -Name 'owned-observations' -Metrics accuracy
        $result = [Activator]::CreateInstance($suite.Results.GetType().GetGenericArguments()[0])
        $result.TestCaseId = 'owned-result'
        $result.Status = 'pass'
        $result.OverallScore = 0.75
        $result.Response = 'owned response'
        $result.Duration = [timespan]::FromMilliseconds(12)
        $result.Metrics = @{ accuracy = $Value }
        $suite.Results.Add($result)
        return $suite
    }

    function Write-PublicReference {
        param([string]$Json, [string]$Name = 'reference')
        [IO.File]::WriteAllText((Join-Path $script:publicBaselineRoot "$Name.json"), $Json, [Text.UTF8Encoding]::new($false))
    }

    function New-PublicHttpSuite {
        $suite = New-EvaluationSuite -Name 'owned-http-reference' -Metrics accuracy
        # Metric calculators are an advertised suite extension, not a command replacement.
        $suite.Metrics[0].Calculator = {
            param($result)
            if ($result.TestCaseId -eq 'first') { return 0.5 }
            return 1.0
        }
        $suite.AddTestCase((New-EvaluationTestCase -Id first -Prompt 'owned first prompt' -ExpectedOutput 'owned response'))
        $suite.AddTestCase((New-EvaluationTestCase -Id second -Prompt 'owned second prompt' -ExpectedOutput 'owned response'))
        return $suite
    }
}

AfterAll {
    # Setup may have failed before root selection; retry only unresolved originals.
    Remove-PublicEvaluationOwnedImports
    if ($publicModuleRemovalFailureOriginals.Count -gt 0) {
        if ($null -ne $publicEvaluationSetupFailureOriginal) { throw $publicEvaluationSetupFailureOriginal }
        throw $publicModuleRemovalFailureOriginals[0]
    }
}

Describe 'Evaluation public persistence and numerical admission' -Tag 'Unit', 'Evaluation', 'Portable' {
    BeforeEach {
        $script:publicEnvironmentBefore = @{}
        foreach ($environmentName in @('PCAI_ROOT', 'PCAI_ARTIFACTS_ROOT')) {
            $item = Get-Item -LiteralPath "Env:$environmentName" -ErrorAction SilentlyContinue
            $script:publicEnvironmentBefore[$environmentName] = @{ Exists = $null -ne $item; Value = $item.Value }
        }
        $script:publicOwnedRoot = Join-Path $TestDrive ([guid]::NewGuid().ToString('N'))
        $script:publicBaselineRoot = Join-Path $publicOwnedRoot 'baselines'
        $null = New-Item -ItemType Directory -Path $publicBaselineRoot -Force
        $env:PCAI_ROOT = $evaluationRepoRoot
        $env:PCAI_ARTIFACTS_ROOT = Join-Path $publicOwnedRoot 'artifacts'
        $script:publicStateBefore = & $publicEvaluationModule {
            @{
                Config = $script:EvaluationConfig; Baselines = $script:Baselines
                CurrentSuite = $script:CurrentSuite; RunState = $script:EvaluationRunState
                Process = $script:CompiledServerProcess; ConfigPath = $script:CompiledServerConfigPath
                BaseUrl = $script:CompiledServerBaseUrl; Backend = $script:CompiledServerBackend
            }
        }
        & $publicEvaluationModule {
            param($baselineRoot)
            $script:EvaluationConfig = @{} + $script:EvaluationConfig
            $script:EvaluationConfig.BaselinePath = $baselineRoot
            $script:EvaluationConfig.HttpBaseUrl = 'http://evaluation-contract.invalid:48080'
            $script:Baselines = @{}
            # An owned HTTP test must never retire a pre-existing compiled server.
            $script:CompiledServerProcess = $null
            $script:CompiledServerConfigPath = $null
            $script:CompiledServerBaseUrl = $null
            $script:CompiledServerBackend = $null
        } $publicBaselineRoot
    }

    AfterEach {
        $script:publicRestoreFailureOriginals = [Collections.Generic.List[object]]::new()
        try {
            & $publicEvaluationModule {
                param($previous)
                $script:EvaluationConfig = $previous.Config; $script:Baselines = $previous.Baselines
                $script:CurrentSuite = $previous.CurrentSuite; $script:EvaluationRunState = $previous.RunState
                $script:CompiledServerProcess = $previous.Process; $script:CompiledServerConfigPath = $previous.ConfigPath
                $script:CompiledServerBaseUrl = $previous.BaseUrl; $script:CompiledServerBackend = $previous.Backend
            } $publicStateBefore
        }
        catch { $script:publicRestoreFailureOriginals.Add($_) }
        finally {
            foreach ($name in $publicEnvironmentBefore.Keys) {
                $previous = $publicEnvironmentBefore[$name]
                try {
                    if ($previous.Exists) {
                        [Environment]::SetEnvironmentVariable($name, $previous.Value, 'Process')
                    }
                    elseif (Test-Path -LiteralPath "Env:$name") {
                        Remove-Item -LiteralPath "Env:$name" -ErrorAction Stop
                    }
                }
                catch { $script:publicRestoreFailureOriginals.Add($_) }
            }
        }
        # Every restoration is attempted before any assertion or rethrow.
        if ($publicRestoreFailureOriginals.Count -gt 0) { throw $publicRestoreFailureOriginals[0] }
        & $publicEvaluationModule {
            param($previous)
            foreach ($pair in @(
                @($script:EvaluationConfig, $previous.Config), @($script:Baselines, $previous.Baselines),
                @($script:CurrentSuite, $previous.CurrentSuite), @($script:EvaluationRunState, $previous.RunState),
                @($script:CompiledServerProcess, $previous.Process)
            )) {
                [object]::ReferenceEquals($pair[0], $pair[1]) | Should -BeTrue
            }
        } $publicStateBefore
        foreach ($name in $publicEnvironmentBefore.Keys) {
            $previous = $publicEnvironmentBefore[$name]
            $restored = Get-Item -LiteralPath "Env:$name" -ErrorAction SilentlyContinue
            ($null -ne $restored) | Should -Be $previous.Exists
            if ($previous.Exists) { $restored.Value | Should -BeExactly $previous.Value }
        }
    }

    # Protects: completed evaluation persistence and downstream regression reports.
    # Detects: missing observations, incorrect distributions or baseline rewrites.
    # Needs: Pester and the real module; the HTTP transport alone is mocked.
    # Breadcrumb: Modules/PC-AI.Evaluation/Public/New-BaselineSnapshot.ps1; Get-RegressionReport.ps1.
    It 'persists an actual completed HTTP evaluation and consumes its on-disk baseline in reports' {
        Mock -ModuleName PC-AI.Evaluation Invoke-RestMethod {
            if ($Method -eq 'Get') { return @{ status = 'ok' } }
            return @{ choices = @(@{ text = 'owned response' }) }
        }
        $suite = New-PublicHttpSuite
        $baseline = New-BaselineSnapshot -Name completed -Suite $suite -Backend http
        $path = Join-Path $publicBaselineRoot 'completed.json'
        $physical = Get-Content -LiteralPath $path -Raw | ConvertFrom-Json -AsHashtable
        $physical.TestCount | Should -Be 2
        $physical.Summary.Passed | Should -Be 1
        $physical.Summary.Failed | Should -Be 1
        $physical.Summary.Errors | Should -Be 0
        $physical.Metrics.accuracy.Mean | Should -Be 0.75
        $physical.DetailedResults.TestId | Should -Be @('first', 'second')
        $baseline.Metrics.accuracy.Mean | Should -Be $physical.Metrics.accuracy.Mean
        Should -Invoke -ModuleName PC-AI.Evaluation Invoke-RestMethod -Times 1 -Exactly -ParameterFilter { $Method -eq 'Get' }
        Should -Invoke -ModuleName PC-AI.Evaluation Invoke-RestMethod -Times 2 -Exactly -ParameterFilter { $Method -eq 'Post' }
        $before = (Get-FileHash -LiteralPath $path).Hash
        $suite.Results[0].Metrics.accuracy = 0.25; $suite.Results[0].OverallScore = 0.25
        $suite.Results[1].Metrics.accuracy = 0.5; $suite.Results[1].OverallScore = 0.5
        $report = Get-RegressionReport -Suite $suite -BaselineNames completed
        $report.BaselinesCompared | Should -Be 1
        $report.Summary.TotalRegressions | Should -Be 1
        $report.Reports[0].Regressions[0].Metric | Should -BeExactly 'accuracy'
        $report.Reports[0].Regressions[0].Change | Should -Be -50
        (Get-FileHash -LiteralPath $path).Hash | Should -BeExactly $before
    }

    # Protects: finite persisted metric distributions.
    # Detects: aggregate overflow that publishes an unusable baseline.
    # Needs: Pester and the real module; the HTTP transport alone is mocked.
    # Breadcrumb: Modules/PC-AI.Evaluation/Public/New-BaselineSnapshot.ps1.
    It 'refuses a nonfinite aggregate from individually finite completed observations before publishing a baseline' {
        Mock -ModuleName PC-AI.Evaluation Invoke-RestMethod {
            if ($Method -eq 'Get') { return @{ status = 'ok' } }
            return @{ choices = @(@{ text = 'owned response' }) }
        }
        $suite = New-PublicHttpSuite
        $suite.Metrics[0].Calculator = { param($result) return 1e308 }
        { New-BaselineSnapshot -Name overflow -Suite $suite -Backend http } |
            Should -Throw '*Baseline metric aggregation is incomplete or nonfinite*'
        $suite.Results.Count | Should -Be 2
        foreach ($result in $suite.Results) {
            [double]::IsFinite($result.Metrics.accuracy) | Should -BeTrue
            $result.Status | Should -BeExactly 'pass'
        }
        Test-Path -LiteralPath (Join-Path $publicBaselineRoot 'overflow.json') | Should -BeFalse
        (& $publicEvaluationModule { $script:Baselines.Count }) | Should -Be 0
        Should -Invoke -ModuleName PC-AI.Evaluation Invoke-RestMethod -Times 2 -Exactly -ParameterFilter { $Method -eq 'Post' }
    }

    # Protects: existing scalar baseline consumers and immutable reference bytes.
    # Detects: incorrect improvement calculations or destructive format migration.
    # Needs: Pester, the real module and an owned temporary reference file.
    # Breadcrumb: Modules/PC-AI.Evaluation/Public/Test-ForRegression.ps1.
    It 'reads a legacy scalar metric baseline without rewriting its physical representation' {
        Write-PublicReference '{"Timestamp":"2026-01-01T00:00:00Z","Metrics":{"accuracy":0.25}}'
        $path = Join-Path $publicBaselineRoot 'reference.json'
        $before = (Get-FileHash -LiteralPath $path).Hash
        $report = Test-ForRegression -BaselineName reference -Suite (New-PublicObservedSuite -Value 0.5)
        $report.HasRegressions | Should -BeFalse
        $report.Regressions.Count | Should -Be 0
        $report.Improvements.Count | Should -Be 1
        $report.Improvements[0].Baseline | Should -Be 0.25
        $report.Improvements[0].Current | Should -Be 0.5
        $report.Improvements[0].Change | Should -Be 100
        (Get-FileHash -LiteralPath $path).Hash | Should -BeExactly $before
    }

    # Protects: metric dictionary admission for the missing, scalar and array cases.
    # Detects: malformed references producing successful comparisons or rewrites.
    # Needs: Pester, the real module and owned temporary malformed references.
    # Breadcrumb: Modules/PC-AI.Evaluation/Public/Test-ForRegression.ps1.
    It 'rejects a baseline without a metric dictionary: <Shape>' -ForEach @(
        @{ Shape = 'missing'; Json = '{"Timestamp":"2026-01-01T00:00:00Z"}' }
        @{ Shape = 'scalar'; Json = '{"Metrics":42}' }
        @{ Shape = 'array'; Json = '{"Metrics":[0.5]}' }
    ) {
        Write-PublicReference $Json
        $path = Join-Path $publicBaselineRoot 'reference.json'
        $before = (Get-FileHash -LiteralPath $path).Hash
        { Test-ForRegression -BaselineName reference -Suite (New-PublicObservedSuite) } |
            Should -Throw '*Regression comparison requires metric distributions*'
        (Get-FileHash -LiteralPath $path).Hash | Should -BeExactly $before
    }

    # Protects: finite relative ratios and percentages in both overflow cases.
    # Detects: finite observations producing infinite summaries or rewritten references.
    # Needs: Pester, the real module and owned temporary numeric references.
    # Breadcrumb: Modules/PC-AI.Evaluation/Public/Test-ForRegression.ps1.
    It 'refuses a finite input pair whose relative summary exceeds the numerical domain: <Kind>' -ForEach @(
        @{ Kind = 'ratio overflow'; Baseline = 1e-308; Current = 10.0 }
        @{ Kind = 'percent overflow'; Baseline = 1e-306; Current = 10.0 }
    ) {
        [double]::IsFinite($Baseline) | Should -BeTrue
        [double]::IsFinite($Current) | Should -BeTrue
        $Baseline | Should -BeGreaterThan 0.0
        Write-PublicReference (@{ Metrics = @{ accuracy = @{ Mean = $Baseline } } } | ConvertTo-Json -Depth 4)
        $path = Join-Path $publicBaselineRoot 'reference.json'
        $before = (Get-FileHash -LiteralPath $path).Hash
        { Test-ForRegression -BaselineName reference -Suite (New-PublicObservedSuite -Value $Current) } |
            Should -Throw '*Relative regression exceeds the finite numerical domain*'
        (Get-FileHash -LiteralPath $path).Hash | Should -BeExactly $before
    }

    # Protects: reference containment for traversal, device, trailing-dot and literal-null names.
    # Detects: unsafe file access admitted before ordinary filename validation.
    # Needs: Pester, the real module and an owned empty baseline directory.
    # Breadcrumb: Modules/PC-AI.Evaluation/Public/Test-ForRegression.ps1.
    It 'refuses an unsafe regression reference name before loading another path: <Kind>' -ForEach @(
        @{ Kind = 'parent traversal'; Name = '../outside' }
        @{ Kind = 'reserved device'; Name = 'AUX.json' }
        @{ Kind = 'trailing dot'; Name = 'reference.' }
        @{ Kind = 'literal null sink'; Name = '$null' }
    ) {
        { Test-ForRegression -BaselineName $Name -Suite (New-PublicObservedSuite) } |
            Should -Throw '*Baseline name must be an ordinary filename leaf*'
        @(Get-ChildItem -LiteralPath $publicBaselineRoot -File).Count | Should -Be 0
    }

    # Protects: error-aware pipeline consumers when a baseline is absent.
    # Detects: a null success item or missing diagnostic under continuing error policy.
    # Needs: Pester, the real module and an owned directory without the reference.
    # Breadcrumb: Modules/PC-AI.Evaluation/Public/Test-ForRegression.ps1.
    It 'returns no successful comparison after a missing reference under nonterminating error policy' {
        $observedErrors = @()
        $comparison = @(Test-ForRegression -BaselineName absent -Suite (New-PublicObservedSuite) -ErrorAction SilentlyContinue -ErrorVariable observedErrors)
        $comparison.Count | Should -Be 0
        $observedErrors.Count | Should -Be 1
        $observedErrors[0].Exception.Message | Should -Match 'Baseline not found: absent'
    }

    # Protects: error-aware result consumers when no current suite exists.
    # Detects: a fabricated summary, null success item or missing diagnostic.
    # Needs: Pester and the real module; fixture restoration preserves ambient state.
    # Breadcrumb: Modules/PC-AI.Evaluation/Public/Get-EvaluationResults.ps1.
    It 'reports absent current suite without fabricating a successful summary' {
        & $publicEvaluationModule { $script:CurrentSuite = $null }
        $observedErrors = @()
        $summary = @(Get-EvaluationResults -ErrorAction SilentlyContinue -ErrorVariable observedErrors)
        $summary.Count | Should -Be 0
        $observedErrors.Count | Should -Be 1
        $observedErrors[0].Exception.Message | Should -Match 'No evaluation suite available'
    }
}


Describe 'Evaluation standalone import identity' -Tag 'Unit', 'Evaluation', 'Portable' {
    # Protects: standalone module state and transport mocks with the compatibility facade discoverable.
    # Detects: recursive aggregate autoload replacing the producer during runtime-helper bootstrap.
    # Needs: a fresh profile-free process; all external transport/provider boundaries are inert.
    It 'retains one producer when its module directory is on PSModulePath' {
        $probePath = Join-Path $TestDrive 'evaluation-import-identity.ps1'
        $runtimeRoot = Join-Path $TestDrive 'evaluation-import-runtime'
        $null = New-Item -ItemType Directory -Path $runtimeRoot -Force
        @'
param([Parameter(Mandatory)][string]$RepositoryRoot, [Parameter(Mandatory)][string]$OwnedRuntimeRoot)
$ErrorActionPreference = 'Stop'
$env:PCAI_CACHE_PROVIDER = 'memory'
$env:PCAI_DISABLE_NATIVE = '1'
$env:PCAI_ROOT = $RepositoryRoot
$env:PCAI_ARTIFACTS_ROOT = Join-Path $OwnedRuntimeRoot 'artifacts'
$env:LOCALAPPDATA = $OwnedRuntimeRoot
$env:PSModulePath = (Join-Path $RepositoryRoot 'Modules') + [IO.Path]::PathSeparator + $env:PSModulePath
$global:ForbiddenBoundaryCalls = 0
function global:Invoke-RestMethod { $global:ForbiddenBoundaryCalls++; throw 'HTTP is forbidden in the import fixture.' }
function global:Invoke-WebRequest { $global:ForbiddenBoundaryCalls++; throw 'HTTP is forbidden in the import fixture.' }
function global:bw { $global:ForbiddenBoundaryCalls++; throw 'Credential retrieval is forbidden in the import fixture.' }
function global:jules { $global:ForbiddenBoundaryCalls++; throw 'Jules execution is forbidden in the import fixture.' }
$rootPath = [IO.Path]::GetFullPath((Join-Path $RepositoryRoot 'Modules/PC-AI.Evaluation/PC-AI.Evaluation.psm1'))
$imports = @(Import-Module (Join-Path $RepositoryRoot 'Modules/PC-AI.Evaluation/PC-AI.Evaluation.psd1') -PassThru -WarningAction SilentlyContinue)
$returned = @($imports | Where-Object { $_.Path -and [IO.Path]::GetFullPath($_.Path) -ceq $rootPath })
$listed = @(Get-Module -Name PC-AI.Evaluation -All)
if ($returned.Count -ne 1 -or $listed.Count -ne 1 -or -not [object]::ReferenceEquals($returned[0], $listed[0])) {
    throw 'Standalone Evaluation import replaced its producer instance.'
}
foreach ($name in @('New-EvaluationSuite', 'New-BaselineSnapshot', 'Invoke-EvaluationSuite')) {
    $command = Get-Command -Name $name -CommandType Function -ListImported -ErrorAction Stop
    if (-not [object]::ReferenceEquals($command.Module, $returned[0]) -or
        -not [object]::ReferenceEquals($command.ScriptBlock.Module, $returned[0])) {
        throw 'An Evaluation public export is bound to a different producer.'
    }
}
$otherComponents = @(Get-Module -All | Where-Object { $_.Name -like 'PC-AI*' -and $_.Name -ne 'PC-AI.Evaluation' })
if ($otherComponents.Count -ne 0) { throw 'Standalone Evaluation import unexpectedly loaded compatibility components.' }
if ($global:ForbiddenBoundaryCalls -ne 0) { throw 'Import attempted an external transport or provider boundary.' }
[pscustomobject]@{ ProducerIdentity = $true; OtherComponentCount = $otherComponents.Count; ForbiddenBoundaryCalls = $global:ForbiddenBoundaryCalls } |
    ConvertTo-Json -Compress
'@ | Set-Content -LiteralPath $probePath -Encoding utf8NoBOM
        $start = [Diagnostics.ProcessStartInfo]::new()
        $start.FileName = Join-Path $PSHOME $(if ($IsWindows) { 'pwsh.exe' } else { 'pwsh' })
        $start.UseShellExecute = $false
        $start.CreateNoWindow = $true
        $start.RedirectStandardOutput = $true
        $start.RedirectStandardError = $true
        foreach ($argument in @('-NoLogo', '-NoProfile', '-NonInteractive', '-File', $probePath, '-RepositoryRoot', $evaluationRepoRoot, '-OwnedRuntimeRoot', $runtimeRoot)) {
            $start.ArgumentList.Add($argument)
        }
        $child = [Diagnostics.Process]::new()
        $child.StartInfo = $start
        try {
            $child.Start() | Should -BeTrue
            $stdout = $child.StandardOutput.ReadToEndAsync()
            $stderr = $child.StandardError.ReadToEndAsync()
            if (-not $child.WaitForExit(60000)) {
                $child.Kill($true)
                throw 'The owned Evaluation import probe exceeded its bounded timeout.'
            }
            $output = $stdout.GetAwaiter().GetResult()
            $null = $stderr.GetAwaiter().GetResult()
            $child.ExitCode | Should -Be 0 -Because 'standalone Evaluation import must retain the actual producer'
            $result = $output | ConvertFrom-Json
            $result.ProducerIdentity | Should -BeTrue
            $result.OtherComponentCount | Should -Be 0
            $result.ForbiddenBoundaryCalls | Should -Be 0
        }
        finally { $child.Dispose() }
    }
}
