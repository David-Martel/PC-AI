# Default maintained binding: repository wrapper and immutable predecessor fixture.
# Private qualification may override only the wrapper, with an explicit exact raw pin.
param(
    [string]$WrapperSourcePath,
    [string]$ExpectedWrapperSHA256
)

BeforeAll {
    $hasOverride = -not [string]::IsNullOrWhiteSpace($WrapperSourcePath)
    $hasExpected = -not [string]::IsNullOrWhiteSpace($ExpectedWrapperSHA256)
    if ($hasOverride -ne $hasExpected) { throw 'Wrapper override and exact SHA256 must be supplied together' }
    if ($hasOverride -and (-not [IO.Path]::IsPathRooted($WrapperSourcePath) -or $ExpectedWrapperSHA256 -notmatch '\A[0-9A-Fa-f]{64}\z')) {
        throw 'Private wrapper qualification requires an absolute source path and 64 hexadecimal SHA256 characters'
    }
    $repositoryRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    $script:wrapperSource = [IO.Path]::GetFullPath($(if ($hasOverride) { $WrapperSourcePath } else { Join-Path $repositoryRoot 'Modules/PC-AI.LLM/Public/Invoke-FunctionGemmaDataset.ps1' }))
    $script:predecessorSource = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../Fixtures/FunctionGemmaDataset.Predecessor.ps1'))
    $testSource = Join-Path $PSScriptRoot 'FunctionGemmaDatasetWrapper.Tests.ps1'
    $script:sourcePins = @(
        foreach ($source in @($script:wrapperSource, $script:predecessorSource, $testSource)) {
            [PSCustomObject]@{ Path=$source; SHA256=(Get-FileHash -LiteralPath $source -ErrorAction Stop).Hash }
        }
    )
    if ($script:sourcePins[1].SHA256 -cne '16D2A190301204A590193021917685CE195E1CC35CEB085C0F78FBAB356BD957') {
        throw 'SOURCE_DRIFT: immutable FunctionGemma predecessor fixture'
    }
    if ($hasOverride -and -not $script:sourcePins[0].SHA256.Equals($ExpectedWrapperSHA256, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'SOURCE_DRIFT: explicit private wrapper qualification pin'
    }
    $script:fixtureOrdinal = 0
    $script:helperText = @'
[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$ToolsPath, [string]$DiagnosePrompt, [string]$ChatPrompt,
    [string]$ScenariosPath, [string]$Output, [string]$TestVectors,
    [int]$MaxCases = 24, [switch]$NoToolCoverage, [switch]$Stream,
    [switch]$UseLld, [switch]$LlmDebug, [switch]$UseNative, [switch]$NativeOnly
)
$forwarded = [ordered]@{}
foreach ($key in @($PSBoundParameters.Keys | Sort-Object)) {
    $value = $PSBoundParameters[$key]
    $forwarded[$key] = if ($value -is [Management.Automation.SwitchParameter]) { $value.IsPresent } else { $value }
}
$record = [PSCustomObject]@{ ActualScript = $PSCommandPath; Parameters = $forwarded; EffectiveMaxCases = $MaxCases }
$logPath = Join-Path (Split-Path -Parent $PSScriptRoot) 'helper-calls.jsonl'
[IO.File]::AppendAllText($logPath, ($record | ConvertTo-Json -Depth 5 -Compress) + "`n", [Text.UTF8Encoding]::new($false))
$record
'@
    function New-WrapperFixture {
        param([Parameter(Mandatory)][string]$Source)
        $script:fixtureOrdinal++
        $root = [IO.Path]::GetFullPath((Join-Path $TestDrive ('wrapper-case-' + $script:fixtureOrdinal)))
        $drive = [IO.Path]::GetFullPath($TestDrive).TrimEnd([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar)
        if (-not $root.StartsWith($drive + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) { throw 'Fixture escaped TestDrive' }
        if (Test-Path -LiteralPath $root) { throw 'Fresh fixture root required' }
        $public = Join-Path $root 'Modules/PC-AI.LLM/Public'
        $tools = Join-Path $root 'Tools'
        $wrongTools = Join-Path $root 'Modules/Tools'
        [IO.Directory]::CreateDirectory($public) | Out-Null
        [IO.Directory]::CreateDirectory($tools) | Out-Null
        [IO.Directory]::CreateDirectory($wrongTools) | Out-Null
        $wrapper = Join-Path $public 'Invoke-FunctionGemmaDataset.ps1'
        [IO.File]::Copy($Source, $wrapper, $false)
        if ((Get-FileHash -LiteralPath $Source).Hash -cne (Get-FileHash -LiteralPath $wrapper).Hash) { throw 'Copied wrapper differs from source' }
        $helper = Join-Path $tools 'prepare-functiongemma-router-data.ps1'
        [IO.File]::WriteAllText($helper, $script:helperText, [Text.UTF8Encoding]::new($false))
        [IO.File]::WriteAllText((Join-Path $wrongTools 'prepare-functiongemma-router-data.ps1'), "throw 'DETECT:poisoned-Modules-Tools'", [Text.UTF8Encoding]::new($false))
        [PSCustomObject]@{ Root=$root; Wrapper=$wrapper; Helper=$helper; Log=(Join-Path $root 'helper-calls.jsonl') }
    }
    function Read-WrapperCalls {
        param([Parameter(Mandatory)][string]$Path)
        if (Test-Path -LiteralPath $Path) {
            foreach ($line in [IO.File]::ReadAllLines($Path, [Text.UTF8Encoding]::new($false, $true))) { $line | ConvertFrom-Json }
        }
    }
}


AfterAll {
    # Rehash the actual selected repository/private source, predecessor and test.
    # Teardown source drift is a failed block; callers must inspect block/container failures.
    foreach ($source in @($script:sourcePins)) {
        if ((Get-FileHash -LiteralPath $source.Path -ErrorAction Stop).Hash -cne $source.SHA256) {
            throw "SOURCE_DRIFT_AFTER: $($source.Path)"
        }
    }
}
Describe 'Copied FunctionGemma dataset wrapper repository boundary' {
    # Protects: Failure-first evidence that the shipped predecessor selects the wrong helper directory.
    # Detects: A predecessor substituted with fixed code, an unexercised route, or a baseline that reaches repository Tools.
    # Needs: Exact raw-pinned predecessor copied under the tiny Modules/PC-AI.LLM/Public layout and a throwing Modules/Tools poison.
    # Breadcrumb: Invoke-FunctionGemmaDataset repository-root expression; reproduces the predecessor selecting Modules/Tools.
    It 'reproduces the predecessor selecting Modules/Tools' {
        $fixture = New-WrapperFixture -Source $script:predecessorSource
        . $fixture.Wrapper
        { Invoke-FunctionGemmaDataset -NativeOnly } | Should -Throw -ExpectedMessage 'DETECT:poisoned-Modules-Tools'
        @(Read-WrapperCalls -Path $fixture.Log).Count | Should -Be 0
    }

    # Protects: The corrected wrapper selects the real repository Tools helper exactly once.
    # Detects: Wrong parent count, duplicate invocation, swallowed helper output, or failure to retain the helper default.
    # Needs: Exact candidate copy, inert parameter-recording Tools helper, and poison at the predecessor's path.
    # Breadcrumb: Invoke-FunctionGemmaDataset repository-root expression and final script invocation; selects repository Tools with omitted parameters.
    It 'selects repository Tools with omitted parameters' {
        $fixture = New-WrapperFixture -Source $script:wrapperSource
        . $fixture.Wrapper
        $result = @(Invoke-FunctionGemmaDataset)
        $calls = @(Read-WrapperCalls -Path $fixture.Log)
        $result.Count | Should -Be 1
        $calls.Count | Should -Be 1
        $calls[0].ActualScript | Should -Be $fixture.Helper
        @($calls[0].Parameters.PSObject.Properties).Count | Should -Be 0
        $calls[0].EffectiveMaxCases | Should -Be 24
        $result[0].ActualScript | Should -Be $fixture.Helper
    }

    # Protects: NativeOnly supplied alone reaches the corrected helper with its caller value.
    # Detects: A dropped NativeOnly switch, invented UseNative switch, or selection of the poisoned old route.
    # Needs: Exact candidate copy and inert helper ledger; this does not exercise native availability or CLI refusal.
    # Breadcrumb: Invoke-FunctionGemmaDataset PSBoundParameters splat; forwards NativeOnly alone.
    It 'forwards NativeOnly alone' {
        $fixture = New-WrapperFixture -Source $script:wrapperSource
        . $fixture.Wrapper
        Invoke-FunctionGemmaDataset -NativeOnly | Out-Null
        $calls = @(Read-WrapperCalls -Path $fixture.Log)
        $calls.Count | Should -Be 1
        $calls[0].Parameters.NativeOnly | Should -BeTrue
        @($calls[0].Parameters.PSObject.Properties).Count | Should -Be 1
        $calls[0].Parameters.PSObject.Properties.Name | Should -Be 'NativeOnly'
    }

    # Protects: UseNative supplied alone is passed without forcing NativeOnly.
    # Detects: Switch loss, NativeOnly accidentally synthesized, or failure before reaching repository Tools.
    # Needs: Exact candidate copy and inert helper ledger; no DLL loading or native fallback executes.
    # Breadcrumb: Invoke-FunctionGemmaDataset PSBoundParameters splat; forwards UseNative alone.
    It 'forwards UseNative alone' {
        $fixture = New-WrapperFixture -Source $script:wrapperSource
        . $fixture.Wrapper
        Invoke-FunctionGemmaDataset -UseNative | Out-Null
        $calls = @(Read-WrapperCalls -Path $fixture.Log)
        $calls.Count | Should -Be 1
        $calls[0].Parameters.UseNative | Should -BeTrue
        @($calls[0].Parameters.PSObject.Properties).Count | Should -Be 1
        $calls[0].Parameters.PSObject.Properties.Name | Should -Be 'UseNative'
    }

    # Protects: All six supplied path strings and the caller's MaxCases value cross the wrapper boundary without rewriting.
    # Detects: Missing fields, path resolution or splitting at spaces, hard-coded limits, and double invocation.
    # Needs: Exact candidate copy and recording helper with literal non-existent paths; input datasets are not read.
    # Breadcrumb: Invoke-FunctionGemmaDataset full explicit string/count parameters; preserves every supplied path and MaxCases.
    It 'preserves every supplied path and MaxCases' {
        $fixture = New-WrapperFixture -Source $script:wrapperSource
        . $fixture.Wrapper
        $arguments = @{
            ToolsPath='relative tools [fixture].json'; DiagnosePrompt='diagnose prompt.md'; ChatPrompt='chat prompt.md'
            ScenariosPath='scenario inputs.json'; Output='output dataset.jsonl'; TestVectors='test vectors.json'; MaxCases=41
        }
        Invoke-FunctionGemmaDataset @arguments | Out-Null
        $calls = @(Read-WrapperCalls -Path $fixture.Log)
        $calls.Count | Should -Be 1
        @($calls[0].Parameters.PSObject.Properties).Count | Should -Be $arguments.Count
        foreach ($key in $arguments.Keys) { $calls[0].Parameters.$key | Should -Be $arguments[$key] }
        $calls[0].EffectiveMaxCases | Should -Be 41
    }

    # Protects: Explicit false switches remain bound and true switches retain their presence.
    # Detects: Truthiness filtering of caller parameters, loss of false values, or automatic expansion of native policy.
    # Needs: Exact candidate copy and recording helper; flags are recorded and never execute builds, profiles or inference.
    # Breadcrumb: Invoke-FunctionGemmaDataset switch parameter forwarding; preserves mixed true and explicit false switches.
    It 'preserves mixed true and explicit false switches' {
        $fixture = New-WrapperFixture -Source $script:wrapperSource
        . $fixture.Wrapper
        $arguments = @{NativeOnly=$false; UseNative=$true; NoToolCoverage=$true; Stream=$false; UseLld=$false; LlmDebug=$true}
        Invoke-FunctionGemmaDataset @arguments | Out-Null
        $calls = @(Read-WrapperCalls -Path $fixture.Log)
        $calls.Count | Should -Be 1
        @($calls[0].Parameters.PSObject.Properties).Count | Should -Be $arguments.Count
        foreach ($key in $arguments.Keys) { $calls[0].Parameters.$key | Should -Be $arguments[$key] }
    }

    # Protects: Missing repository helper fails with the correct path and does not use the poisoned Modules/Tools alternative.
    # Detects: Silent fallback to an old directory, swallowed missing-helper errors, or unexpected helper execution.
    # Needs: Fresh TestDrive fixture whose one owned Tools helper is removed; the poison remains present.
    # Breadcrumb: Invoke-FunctionGemmaDataset Test-Path guard; refuses a missing repository helper.
    It 'refuses a missing repository helper' {
        $fixture = New-WrapperFixture -Source $script:wrapperSource
        . $fixture.Wrapper
        Remove-Item -LiteralPath $fixture.Helper -ErrorAction Stop
        $expected = 'prepare-functiongemma-router-data.ps1 not found at ' + $fixture.Helper
        { Invoke-FunctionGemmaDataset -NativeOnly } | Should -Throw -ExpectedMessage ([WildcardPattern]::Escape($expected))
        @(Read-WrapperCalls -Path $fixture.Log).Count | Should -Be 0
    }

    # Protects: A real selected helper failure reaches the caller without switching routes or manufacturing success.
    # Detects: Catch-and-ignore behavior or retries through the poisoned old helper directory.
    # Needs: Fresh fixture and inert throwing repository helper; no actual native or Build.ps1 path executes.
    # Breadcrumb: Invoke-FunctionGemmaDataset final invocation; propagates the selected helper exception.
    It 'propagates the selected helper exception' {
        $fixture = New-WrapperFixture -Source $script:wrapperSource
        . $fixture.Wrapper
        [IO.File]::WriteAllText($fixture.Helper, "throw 'DETECT:selected-helper-failure'", [Text.UTF8Encoding]::new($false))
        { Invoke-FunctionGemmaDataset -UseNative } | Should -Throw -ExpectedMessage 'DETECT:selected-helper-failure'
        @(Read-WrapperCalls -Path $fixture.Log).Count | Should -Be 0
    }
}
