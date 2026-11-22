#Requires -Version 7.0
<#
.SYNOPSIS
    Test assessment for PC_AI: inventory, reach, measured Pester coverage, quality counts and a ratcheted gate.

.DESCRIPTION
    Why this exists: this script used to count test files by name and write a Markdown report, which
    measured nothing (Reports/COVERAGE_REPORT.md was months stale), while the repo's real coverage
    setting could not fail. Measured on 1bbeaad2: Tests/PesterConfiguration.psd1 asked for
    './Modules/**/*.ps1', Pester treats `**` like `*`, and only 17 of 241 module files were analysed;
    `CoveragePercentTarget = 85` only colours console output.

    This tool measures instead, writes one JSON document and exits 1 when a gate fails:

      inventory  Pester It blocks per directory, Rust test attributes per crate, C# test attributes per project.
                 Rust and C# counts are static greps, so cfg(feature)-gated tests count whether or not the
                 feature is enabled.
      reach      which runner scripts and workflows would execute each *.Tests.ps1; files nothing runs are orphans.
      coverage   Pester command coverage over every Modules/**/*.ps1|psm1 file, overall and per module (not with -NoRun).
      quality    It blocks with no assertion, with only `Should -Not -Throw`, or with a constant-input assertion
                 (`$true | Should -BeTrue`), found through the PowerShell AST.
      gates      thresholds from Config/test-baseline.json, the violations, and a pass|fail verdict.

    The baseline is a ratchet. Its numbers start at what was measured, not at the 85% target (recorded as
    target_coverage_pct), and -UpdateBaseline can only raise the coverage minimums and lower the violation
    counts, orphan allowlist and known-failure budget.

    Writes nothing in the repository except -OutputPath (and Config/test-baseline.json with -UpdateBaseline).
    Pester's JaCoCo and result files go to a temporary directory that is removed afterwards.

    Dot-sourcing this file defines the functions and does nothing else; Tests/Unit/TestAssessment.Tests.ps1 does that.

.PARAMETER OutputPath
    JSON report path. Default: .pcai/assessment/test-assessment.json (gitignored).

.PARAMETER Tier
    Portable runs Tests/Unit with the Portable tag, the set that must also pass without hardware. Full runs the
    paths ci.yml passes to Invoke-Pester (read from Tests/Invoke-PcaiCi.ps1) with no tag filter.

.PARAMETER BaselinePath
    Ratchet file. Default: Config/test-baseline.json (committed).

.PARAMETER UpdateBaseline
    Raise the coverage minimums and lower the violation counts to what this run measured. It never loosens a
    threshold. It seeds a tier that has no baseline entry yet.

.PARAMETER NoRun
    Static inventory, reach and quality only. No Pester run, no coverage, fast and deterministic.

.PARAMETER RepoRoot
    Repository to assess. Default: the parent of this script's directory. A test seam for fixtures.

.EXAMPLE
    pwsh Tests/coverage-report.ps1 -Tier Portable
    Runs the Portable suite with coverage, writes the JSON and exits 1 on a gate failure.

.EXAMPLE
    pwsh Tests/coverage-report.ps1 -NoRun
    Static assessment only.

.LINK
    Tests/Helpers/Get-PesterRunVerdict.ps1
.LINK
    Tests/Unit/TestAssessment.Tests.ps1
.LINK
    Config/test-baseline.json
#>
[CmdletBinding()]
param(
    [string]$OutputPath,

    [ValidateSet('Portable', 'Full')]
    [string]$Tier = 'Portable',

    [string]$BaselinePath,

    [switch]$UpdateBaseline,

    [switch]$NoRun,

    [string]$RepoRoot
)

. (Join-Path $PSScriptRoot 'Helpers/Get-PesterRunVerdict.ps1')

$script:AssessmentSchemaVersion = 1
$script:AstNs = 'System.Management.Automation.Language'

#region file discovery

function Get-AssessmentFileList {
    <#
    .SYNOPSIS
        Lists repository files as forward-slash relative paths in ordinal order.
    .DESCRIPTION
        Uses `git ls-files --cached --others --exclude-standard` so it never walks worktrees/, target/ or
        node_modules/. Outside a git repository (a test fixture) it walks the directory and skips those.
    .PARAMETER Root
        Repository root.
    #>
    [CmdletBinding()]
    [OutputType([string[]])]
    param([Parameter(Mandatory)][string]$Root)

    $Root = [IO.Path]::GetFullPath($Root)
    $files = $null
    if (Get-Command git -ErrorAction SilentlyContinue) {
        $top = & git -C $Root rev-parse --show-toplevel 2>$null
        if ($LASTEXITCODE -eq 0 -and $top -and ([IO.Path]::GetFullPath($top).TrimEnd('/', '\') -eq $Root.TrimEnd('/', '\'))) {
            $listed = & git -C $Root -c core.quotepath=off ls-files --cached --others --exclude-standard 2>$null
            if ($LASTEXITCODE -eq 0) { $files = @($listed) }
        }
    }
    if ($null -eq $files) {
        $skip = '(^|/)(\.git|worktrees|target|node_modules|\.pcai)(/|$)'
        $files = @(Get-ChildItem -LiteralPath $Root -Recurse -File -Force -ErrorAction SilentlyContinue | ForEach-Object {
                [IO.Path]::GetRelativePath($Root, $_.FullName).Replace('\', '/')
            } | Where-Object { $_ -notmatch $skip })
    }
    # A path git still lists but the working tree no longer has (deleted, not yet staged) is not a file to assess.
    $set = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    foreach ($f in $files) {
        if ($f -and (Test-Path -LiteralPath (Join-Path $Root $f) -PathType Leaf)) { [void]$set.Add($f) }
    }
    $sorted = [string[]]::new($set.Count)
    $set.CopyTo($sorted)
    [Array]::Sort($sorted, [StringComparer]::Ordinal)
    $sorted
}

#endregion

#region Pester AST analysis

function Get-CommandParameterStrings {
    # String constants given to a command's named parameter, e.g. -Tag 'Unit', 'Portable'.
    param($Command, [string]$Name)
    $values = [System.Collections.Generic.List[string]]::new()
    $elements = $Command.CommandElements
    for ($i = 0; $i -lt $elements.Count; $i++) {
        $el = $elements[$i]
        if ($el -is [System.Management.Automation.Language.CommandParameterAst] -and $el.ParameterName -eq $Name) {
            $arg = if ($el.Argument) { $el.Argument } elseif ($i + 1 -lt $elements.Count) { $elements[$i + 1] } else { $null }
            if ($arg) {
                $found = $arg.FindAll({ param($n) $n -is [System.Management.Automation.Language.StringConstantExpressionAst] }, $true)
                foreach ($s in $found) { $values.Add([string]$s.Value) }
                if ($arg -is [System.Management.Automation.Language.StringConstantExpressionAst]) { $values.Add([string]$arg.Value) }
            }
        }
    }
    $values.ToArray()
}

function Test-StaticSkipParameter {
    # True for a literal -Skip or -Skip:$true. A -Skip:(condition) is decided at run time and is not static.
    param($Command)
    foreach ($el in $Command.CommandElements) {
        if ($el -is [System.Management.Automation.Language.CommandParameterAst] -and $el.ParameterName -eq 'Skip') {
            if ($null -eq $el.Argument) { return $true }
            if ($el.Argument -is [System.Management.Automation.Language.VariableExpressionAst] -and
                $el.Argument.VariablePath.UserPath -eq 'true') { return $true }
        }
    }
    $false
}

function Get-PesterFileAnalysis {
    <#
    .SYNOPSIS
        Parses one Pester file and classifies each It block.
    .DESCRIPTION
        Uses the PowerShell parser, so strings and comments that merely mention `Should` do not count.
        An It block is
          no_assertion      no Should, Assert-*, local assertion helper or throw inside it, and it is not statically skipped;
          not_throw_only    every Should in it is `Should -Not -Throw`, so it proves nothing about the result;
          tautology         it pipes a constant (`$true`, `$null`, a number or a plain string) into Should.
        A function defined in the same file whose body asserts counts as an assertion helper.
    .PARAMETER Path
        Path to the .Tests.ps1 file.
    .OUTPUTS
        PSCustomObject: ParseErrors, HasPortableTag, It (array of Name, Line, StaticSkip, NoAssertion, NotThrowOnly, Tautology).
    #>
    [CmdletBinding()]
    [OutputType([pscustomobject])]
    param([Parameter(Mandatory)][string]$Path)

    $tokens = $null
    $errors = $null
    $ast = [System.Management.Automation.Language.Parser]::ParseFile($Path, [ref]$tokens, [ref]$errors)
    $result = [pscustomobject]@{ ParseErrors = @($errors).Count; HasPortableTag = $false; It = @() }
    if ($null -eq $ast) { return $result }

    $isCommand = { param($n) $n -is [System.Management.Automation.Language.CommandAst] }
    $allCommands = $ast.FindAll($isCommand, $true)

    foreach ($c in $allCommands) {
        if ($c.GetCommandName() -in 'Describe', 'Context', 'It') {
            if ('Portable' -in (Get-CommandParameterStrings -Command $c -Name 'Tag')) { $result.HasPortableTag = $true }
        }
    }

    $helpers = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    $functions = $ast.FindAll({ param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] }, $true)
    foreach ($fn in $functions) {
        $asserts = $fn.Body.FindAll({
                param($n)
                ($n -is [System.Management.Automation.Language.ThrowStatementAst]) -or
                ($n -is [System.Management.Automation.Language.CommandAst] -and $n.GetCommandName() -eq 'Should')
            }, $true)
        if (@($asserts).Count -gt 0) { [void]$helpers.Add($fn.Name) }
    }

    $its = [System.Collections.Generic.List[object]]::new()
    foreach ($it in ($allCommands | Where-Object { $_.GetCommandName() -in 'It', 'Specify' })) {
        $body = $it.CommandElements | Where-Object { $_ -is [System.Management.Automation.Language.ScriptBlockExpressionAst] } | Select-Object -First 1
        $nameEl = $it.CommandElements | Select-Object -Skip 1 -First 1
        $name = if ($nameEl -is [System.Management.Automation.Language.StringConstantExpressionAst] -or
            $nameEl -is [System.Management.Automation.Language.ExpandableStringExpressionAst]) { [string]$nameEl.Value } else { '' }

        $skip = Test-StaticSkipParameter -Command $it
        for ($p = $it.Parent; $p -and -not $skip; $p = $p.Parent) {
            if ($p -is [System.Management.Automation.Language.CommandAst] -and $p.GetCommandName() -in 'Describe', 'Context') {
                $skip = Test-StaticSkipParameter -Command $p
            }
        }

        $shoulds = @(); $helperCalls = 0; $throws = 0; $tautology = $false
        if ($body) {
            $inner = $body.ScriptBlock.FindAll($isCommand, $true)
            $shoulds = @($inner | Where-Object { $_.GetCommandName() -eq 'Should' })
            $helperCalls = @($inner | Where-Object {
                    $n = $_.GetCommandName()
                    $n -and ($n -like 'Assert-*' -or $helpers.Contains($n))
                }).Count
            $throws = @($body.ScriptBlock.FindAll({ param($n) $n -is [System.Management.Automation.Language.ThrowStatementAst] }, $true)).Count
            foreach ($should in $shoulds) {
                $pipeline = $should.Parent
                if ($pipeline -is [System.Management.Automation.Language.PipelineAst] -and $pipeline.PipelineElements.Count -ge 2) {
                    $first = $pipeline.PipelineElements[0]
                    if ($first -is [System.Management.Automation.Language.CommandExpressionAst]) {
                        $e = $first.Expression
                        $pnames = @($should.CommandElements | Where-Object { $_ -is [System.Management.Automation.Language.CommandParameterAst] } | ForEach-Object ParameterName)
                        $argVar = $should.CommandElements | Select-Object -Skip 1 |
                            Where-Object { $_ -is [System.Management.Automation.Language.VariableExpressionAst] } |
                            ForEach-Object { $_.VariablePath.UserPath } | Select-Object -First 1
                        if ($e -is [System.Management.Automation.Language.VariableExpressionAst] -and $e.VariablePath.UserPath -in 'true', 'false') {
                            # `$false | Should -BeTrue` is the deliberate "should not reach here" failure marker, and
                            # `-Not` inverts the check, so only an assertion that is certain to pass is a tautology.
                            $value = $e.VariablePath.UserPath
                            $compares = ('Be' -in $pnames -or 'BeExactly' -in $pnames) -and $argVar -eq $value
                            $wanted = if ($value -eq 'true') { 'BeTrue' } else { 'BeFalse' }
                            if (('Not' -notin $pnames) -and ($wanted -in $pnames -or $compares)) { $tautology = $true }
                        } elseif (($e -is [System.Management.Automation.Language.ConstantExpressionAst]) -or
                            ($e -is [System.Management.Automation.Language.StringConstantExpressionAst]) -or
                            ($e -is [System.Management.Automation.Language.VariableExpressionAst] -and $e.VariablePath.UserPath -eq 'null')) {
                            $tautology = $true
                        }
                    }
                }
            }
        }
        $assertions = $shoulds.Count + $helperCalls + $throws
        $notThrow = $false
        if ($shoulds.Count -gt 0 -and $helperCalls -eq 0 -and $throws -eq 0) {
            $notThrow = (@($shoulds | Where-Object {
                        $names = @($_.CommandElements | Where-Object { $_ -is [System.Management.Automation.Language.CommandParameterAst] } | ForEach-Object ParameterName)
                        -not ('Not' -in $names -and 'Throw' -in $names)
                    }).Count -eq 0)
        }
        $its.Add([pscustomobject]@{
                Name         = $name
                Line         = $it.Extent.StartLineNumber
                StaticSkip   = $skip
                NoAssertion  = ($assertions -eq 0 -and -not $skip)
                NotThrowOnly = $notThrow
                Tautology    = $tautology
            })
    }
    $result.It = $its.ToArray()
    $result
}

#endregion

#region inventory

function Get-RustTestInventory {
    <#
    .SYNOPSIS
        Counts Rust test attributes per crate (static grep; cfg(feature)-gated tests count regardless of features).
    .PARAMETER Root
        Repository root.
    .PARAMETER Files
        Relative file list from Get-AssessmentFileList.
    #>
    [CmdletBinding()]
    param([string]$Root, [string[]]$Files)
    $crateDirs = @($Files | Where-Object { $_ -match '(^|/)Cargo\.toml$' } | ForEach-Object {
            $d = [IO.Path]::GetDirectoryName($_).Replace('\', '/'); $d } | Sort-Object { $_.Length } -Descending)
    $counts = @{}
    foreach ($f in ($Files | Where-Object { $_ -like '*.rs' })) {
        $crate = $crateDirs | Where-Object { $_ -eq '' -or $f.StartsWith($_ + '/', [StringComparison]::Ordinal) } | Select-Object -First 1
        if ($null -eq $crate) { continue }
        $text = [IO.File]::ReadAllText((Join-Path $Root $f))
        $n = [regex]::Matches($text, '(?m)^\s*#\[(?:[\w:]+::)?(?:test|rstest|test_case)\b').Count
        $key = if ($crate -eq '') { '.' } else { $crate }
        $counts[$key] = $n + [int]$counts[$key]
    }
    @($counts.Keys | Sort-Object { $_ } | ForEach-Object { [ordered]@{ crate = $_; tests = $counts[$_] } })
}

function Get-CSharpTestInventory {
    <#
    .SYNOPSIS
        Counts C# test attributes per project that contains any ([Fact], [Theory], [Test], [TestMethod], [TestCase]).
    .PARAMETER Root
        Repository root.
    .PARAMETER Files
        Relative file list from Get-AssessmentFileList.
    #>
    [CmdletBinding()]
    param([string]$Root, [string[]]$Files)
    $projects = @($Files | Where-Object { $_ -like '*.csproj' } | ForEach-Object { [IO.Path]::GetDirectoryName($_).Replace('\', '/') } |
            Sort-Object { $_.Length } -Descending)
    $counts = @{}
    foreach ($f in ($Files | Where-Object { $_ -like '*.cs' })) {
        $proj = $projects | Where-Object { $_ -eq '' -or $f.StartsWith($_ + '/', [StringComparison]::Ordinal) } | Select-Object -First 1
        if ($null -eq $proj) { continue }
        $text = [IO.File]::ReadAllText((Join-Path $Root $f))
        $n = [regex]::Matches($text, '(?m)^\s*\[(?:Fact|Theory|Test|TestMethod|TestCase)\b').Count
        if ($n -gt 0) { $counts[$proj] = $n + [int]$counts[$proj] }
    }
    @($counts.Keys | Sort-Object { $_ } | ForEach-Object { [ordered]@{ project = $_; tests = $counts[$_] } })
}

#endregion

#region reach

function Get-CiPesterPath {
    <#
    .SYNOPSIS
        Reads the Pester path list that Tests/Invoke-PcaiCi.ps1 hands to Invoke-Pester (the list ci.yml uses).
    .PARAMETER Root
        Repository root.
    #>
    [CmdletBinding()]
    [OutputType([string[]])]
    param([string]$Root)
    $script = Join-Path $Root 'Tests/Invoke-PcaiCi.ps1'
    if (-not (Test-Path -LiteralPath $script)) { return @() }
    $m = [regex]::Match([IO.File]::ReadAllText($script), '(?s)\$pesterPaths\s*=\s*@\((?<body>.*?)\)')
    if (-not $m.Success) { return @() }
    @([regex]::Matches($m.Groups['body'].Value, "'([^']+)'") | ForEach-Object { $_.Groups[1].Value })
}

function Get-TestReach {
    <#
    .SYNOPSIS
        Maps each *.Tests.ps1 file to the runner scripts and workflows that would execute it.
    .DESCRIPTION
        Runner rules mirror the runners' own path logic:
          Invoke-PortableTests  Tests/Unit (recursive) with the Portable tag
          Invoke-AllTests       direct children of Tests/Unit, Integration, E2E and Functional
          test.ps1              everything under Tests/
          .pester.ps1           Tests/Unit and Tests/Integration (recursive)
          Invoke-PcaiCi         the path list in Invoke-PcaiCi.ps1
        A workflow reaches a file when its text names the file, one of its directories (two segments or more), or
        the file's leaf listed in a list file the workflow reads (Tests/linux-ci-pester.txt). This is a text
        heuristic: it proves a workflow mentions the file, not that the job runs it.
    .PARAMETER Root
        Repository root.
    .PARAMETER Files
        Relative file list from Get-AssessmentFileList.
    .PARAMETER Analysis
        Hashtable of relative path to Get-PesterFileAnalysis result.
    #>
    [CmdletBinding()]
    param([string]$Root, [string[]]$Files, [hashtable]$Analysis)

    $ciPaths = Get-CiPesterPath -Root $Root
    $workflows = @{}
    foreach ($w in ($Files | Where-Object { $_ -like '.github/workflows/*.yml' -or $_ -like '.github/workflows/*.yaml' })) {
        $workflows[[IO.Path]::GetFileName($w)] = [IO.File]::ReadAllText((Join-Path $Root $w)).Replace('\', '/')
    }
    $listLeaves = @{}
    foreach ($l in ($Files | Where-Object { $_ -like 'Tests/*.txt' })) {
        $listLeaves[$l] = @([IO.File]::ReadAllLines((Join-Path $Root $l)) | ForEach-Object { $_.Trim() } |
                Where-Object { $_ -and -not $_.StartsWith('#') })
    }

    $rows = [System.Collections.Generic.List[object]]::new()
    foreach ($f in ($Files | Where-Object { $_ -like '*.Tests.ps1' })) {
        $leaf = [IO.Path]::GetFileName($f)
        $dir = [IO.Path]::GetDirectoryName($f).Replace('\', '/')
        $portable = $Analysis.ContainsKey($f) -and $Analysis[$f].HasPortableTag
        $runners = [System.Collections.Generic.List[string]]::new()
        if ($f.StartsWith('Tests/Unit/', [StringComparison]::Ordinal) -and $portable) { $runners.Add('Invoke-PortableTests') }
        if ($dir -in 'Tests/Unit', 'Tests/Integration', 'Tests/E2E', 'Tests/Functional') { $runners.Add('Invoke-AllTests') }
        if ($f.StartsWith('Tests/', [StringComparison]::Ordinal)) { $runners.Add('test.ps1') }
        if ($f.StartsWith('Tests/Unit/', [StringComparison]::Ordinal) -or $f.StartsWith('Tests/Integration/', [StringComparison]::Ordinal)) { $runners.Add('.pester.ps1') }
        if (@($ciPaths | Where-Object { $f -eq $_ -or $f.StartsWith($_.TrimEnd('/') + '/', [StringComparison]::Ordinal) }).Count -gt 0) { $runners.Add('Invoke-PcaiCi') }

        $segments = $f.Split('/')
        $ancestors = for ($i = 2; $i -lt $segments.Count; $i++) { ($segments[0..($i - 1)] -join '/') }
        $hit = [System.Collections.Generic.List[string]]::new()
        foreach ($name in ($workflows.Keys | Sort-Object)) {
            $text = $workflows[$name]
            $found = $text.Contains($f)
            if (-not $found) {
                foreach ($a in $ancestors) {
                    if ($text -match ('(?<![\w./-])' + [regex]::Escape($a) + '(?![\w-]|/[\w])')) { $found = $true; break }
                }
            }
            if (-not $found) {
                foreach ($list in $listLeaves.Keys) {
                    if ($text.Contains($list) -and $listLeaves[$list] -contains $leaf) { $found = $true; break }
                }
            }
            if ($found) { $hit.Add($name) }
        }
        $rows.Add([ordered]@{ file = $f; runners = @($runners | Sort-Object); workflows = @($hit) })
    }
    $rows.ToArray()
}

#endregion

#region coverage

function Get-ModuleNameFromPath {
    # Modules/PC-AI.Common/Public/x.ps1 -> PC-AI.Common; Modules/PcaiInference.psm1 -> PcaiInference.
    param([string]$RelativePath)
    $parts = $RelativePath.Split('/')
    if ($parts.Count -le 2) { [IO.Path]::GetFileNameWithoutExtension($parts[-1]) } else { $parts[1] }
}

function Get-ModuleSourceFile {
    <#
    .SYNOPSIS
        Lists the module source files coverage is measured against (Modules/**/*.ps1 and *.psm1 except *.Tests.ps1).
    .PARAMETER Files
        Relative file list from Get-AssessmentFileList.
    #>
    [CmdletBinding()]
    [OutputType([string[]])]
    param([string[]]$Files)
    @($Files | Where-Object {
            $_.StartsWith('Modules/', [StringComparison]::Ordinal) -and ($_ -like '*.ps1' -or $_ -like '*.psm1') -and $_ -notlike '*.Tests.ps1'
        })
}

function Get-CoverageSummary {
    <#
    .SYNOPSIS
        Reduces a Pester CodeCoverage object to overall, files-analysed ratio and per-module percentages.
    .PARAMETER CodeCoverage
        The CodeCoverage property of an Invoke-Pester result.
    .PARAMETER Root
        Repository root, used to make analysed paths relative.
    .PARAMETER ModuleFiles
        Relative module source files from Get-ModuleSourceFile; the denominator for the files-analysed ratio.
    #>
    [CmdletBinding()]
    param($CodeCoverage, [string]$Root, [string[]]$ModuleFiles)

    $rel = { param($p) [IO.Path]::GetRelativePath($Root, $p).Replace('\', '/') }
    $analyzed = @($CodeCoverage.FilesAnalyzed | ForEach-Object { & $rel $_ })
    $exec = @($CodeCoverage.CommandsExecuted)
    $missed = @($CodeCoverage.CommandsMissed)

    $byModule = [ordered]@{}
    foreach ($f in $ModuleFiles) {
        $m = Get-ModuleNameFromPath $f
        if (-not $byModule.Contains($m)) { $byModule[$m] = [ordered]@{ module = $m; files_total = 0; files_analyzed = 0; commands_analyzed = 0; commands_executed = 0; pct = 0.0 } }
        $byModule[$m].files_total++
    }
    foreach ($f in $analyzed) {
        $m = Get-ModuleNameFromPath $f
        if ($byModule.Contains($m)) { $byModule[$m].files_analyzed++ }
    }
    foreach ($c in $exec) { $m = Get-ModuleNameFromPath (& $rel $c.File); if ($byModule.Contains($m)) { $byModule[$m].commands_executed++; $byModule[$m].commands_analyzed++ } }
    foreach ($c in $missed) { $m = Get-ModuleNameFromPath (& $rel $c.File); if ($byModule.Contains($m)) { $byModule[$m].commands_analyzed++ } }
    foreach ($m in $byModule.Keys) {
        $row = $byModule[$m]
        $row.pct = if ($row.commands_analyzed -gt 0) { [math]::Round(100.0 * $row.commands_executed / $row.commands_analyzed, 2) } else { 0.0 }
    }

    $total = $ModuleFiles.Count
    $inScope = @($analyzed | Where-Object { $_ -in $ModuleFiles }).Count
    $cmdAnalyzed = [int]$CodeCoverage.CommandsAnalyzedCount
    $cmdExecuted = [int]$CodeCoverage.CommandsExecutedCount
    [ordered]@{
        module_files_total    = $total
        files_analyzed        = $inScope
        files_analyzed_ratio  = if ($total -gt 0) { [math]::Round($inScope / $total, 4) } else { 0.0 }
        commands_analyzed     = $cmdAnalyzed
        commands_executed     = $cmdExecuted
        pct                   = if ($cmdAnalyzed -gt 0) { [math]::Round(100.0 * $cmdExecuted / $cmdAnalyzed, 2) } else { 0.0 }
        by_module             = @($byModule.Values)
    }
}

#endregion

#region gate

function Get-GateViolation {
    # A gate violation: a stable code a reader can grep for, and the human sentence.
    param([string]$Code, [string]$Message)
    [ordered]@{ code = $Code; message = $Message }
}

function Test-AssessmentGate {
    <#
    .SYNOPSIS
        Applies the ratcheted baseline to an assessment and returns the violations.
    .DESCRIPTION
        Fails closed: a missing baseline, a missing tier entry or a missing Pester 5 is a violation, not a pass.
    .PARAMETER Assessment
        The assessment document (ordered hashtable) built by this script.
    .PARAMETER Baseline
        Parsed Config/test-baseline.json (PSCustomObject) or $null when absent.
    .PARAMETER Tier
        Portable or Full.
    .PARAMETER PesterResult
        Invoke-Pester result, or $null with -NoRun.
    .PARAMETER Ran
        Whether Pester was run for this assessment.
    #>
    [CmdletBinding()]
    param($Assessment, $Baseline, [string]$Tier, $PesterResult, [switch]$Ran)

    $v = [System.Collections.Generic.List[object]]::new()
    if ($null -eq $Baseline) {
        $v.Add((Get-GateViolation 'baseline_missing' 'no baseline file; run with -UpdateBaseline once to seed it, then commit it'))
        return $v.ToArray()
    }

    $q = $Assessment.quality
    foreach ($pair in @(
            @('no_assertion_it', 'no_assertion_it_max'), @('not_throw_only_it', 'not_throw_only_it_max'),
            @('tautology_it', 'tautology_it_max'), @('parse_error_files', 'parse_error_files_max'))) {
        $max = $Baseline.quality.($pair[1])
        if ($null -eq $max) { $max = 0 }
        if ([int]$q.($pair[0]) -gt [int]$max) {
            $v.Add((Get-GateViolation "quality_$($pair[0])" "$($pair[0]) is $($q.($pair[0])), baseline allows $max"))
        }
    }
    $allowed = @($Baseline.orphans_allowed)
    foreach ($o in $Assessment.reach.orphans) {
        if ($o -notin $allowed) { $v.Add((Get-GateViolation 'new_orphan' "$o is run by no runner script and no workflow, and is not in the orphan allowlist")) }
    }

    if (-not $Ran) { return $v.ToArray() }

    $tb = $Baseline.tiers.$Tier
    if ($null -eq $tb) {
        $v.Add((Get-GateViolation 'tier_baseline_missing' "baseline has no entry for tier $Tier; run with -UpdateBaseline to seed it"))
        return $v.ToArray()
    }
    $known = @{}
    if ($tb.known_failures.tests_by_file) { foreach ($p in $tb.known_failures.tests_by_file.PSObject.Properties) { $known[$p.Name] = [int]$p.Value } }
    $verdict = Get-PesterRunVerdict -Result $PesterResult -KnownFailures $known `
        -MaxFailedContainers ([int]$tb.known_failures.max_failed_containers) -MaxFailedBlocks ([int]$tb.known_failures.max_failed_blocks)
    foreach ($msg in $verdict.Violations) { $v.Add((Get-GateViolation 'pester_run' $msg)) }

    $cov = $Assessment.coverage.pester
    if ($cov) {
        if ($cov.pct -lt [double]$tb.coverage_min_pct) {
            $v.Add((Get-GateViolation 'coverage_below_min' "Pester coverage $($cov.pct)% is below the baseline minimum $($tb.coverage_min_pct)% (target $($Baseline.target_coverage_pct)%)"))
        }
        $minRatio = if ($null -ne $Baseline.files_analyzed_ratio_min) { [double]$Baseline.files_analyzed_ratio_min } else { 0.95 }
        if ($cov.files_analyzed_ratio -lt $minRatio) {
            $v.Add((Get-GateViolation 'coverage_files_ratio' "only $($cov.files_analyzed) of $($cov.module_files_total) module files were analysed ($($cov.files_analyzed_ratio)), minimum ratio $minRatio; the coverage path glob is too narrow"))
        }
        if ($tb.per_module_min_pct) {
            # Small modules move a point or two between runs (test order, machine state), so a module may sit up to
            # per_module_tolerance_pct under its minimum before it is a violation. The overall minimum has no tolerance.
            $tolerance = if ($null -ne $Baseline.per_module_tolerance_pct) { [double]$Baseline.per_module_tolerance_pct } else { 1.0 }
            foreach ($p in $tb.per_module_min_pct.PSObject.Properties) {
                $row = @($cov.by_module | Where-Object { $_.module -eq $p.Name }) | Select-Object -First 1
                if ($row -and $row.pct -lt ([double]$p.Value - $tolerance)) {
                    $v.Add((Get-GateViolation 'module_coverage_below_min' "$($p.Name) coverage $($row.pct)% is below its baseline minimum $($p.Value)% (tolerance $tolerance)"))
                }
            }
        }
    } else {
        $v.Add((Get-GateViolation 'coverage_missing' 'Pester ran but produced no coverage data'))
    }
    $v.ToArray()
}

function Update-AssessmentBaseline {
    <#
    .SYNOPSIS
        Ratchets the baseline: raises coverage minimums, lowers violation counts. Never loosens anything.
    .DESCRIPTION
        An absent baseline, or an absent tier entry, is seeded from the measurement. Afterwards a value moves
        only in the stricter direction: coverage minimums use max(old, floor(measured)); violation maximums,
        the orphan allowlist and the known-failure budget use min/intersection. A new orphan or a new failing
        file is never added, so the gate still fails until someone fixes it.
    .PARAMETER Baseline
        Existing baseline (PSCustomObject) or $null.
    .PARAMETER Assessment
        The assessment document.
    .PARAMETER Tier
        Portable or Full.
    .PARAMETER PesterResult
        Invoke-Pester result, or $null.
    .PARAMETER Ran
        Whether Pester was run.
    #>
    [CmdletBinding()]
    param($Baseline, $Assessment, [string]$Tier, $PesterResult, [switch]$Ran)

    $old = if ($Baseline) { $Baseline | ConvertTo-Json -Depth 12 | ConvertFrom-Json -AsHashtable } else { @{} }
    $new = [ordered]@{
        schema_version            = $script:AssessmentSchemaVersion
        target_coverage_pct       = if ($old.target_coverage_pct) { $old.target_coverage_pct } else { 85 }
        files_analyzed_ratio_min  = if ($old.files_analyzed_ratio_min) { $old.files_analyzed_ratio_min } else { 0.95 }
        per_module_tolerance_pct  = if ($null -ne $old.per_module_tolerance_pct) { $old.per_module_tolerance_pct } else { 1.0 }
    }

    $q = $Assessment.quality
    $oq = if ($old.quality) { $old.quality } else { @{} }
    $newQ = [ordered]@{}
    foreach ($k in 'no_assertion_it', 'not_throw_only_it', 'tautology_it', 'parse_error_files') {
        $cur = [int]$q.$k
        $newQ["${k}_max"] = if ($null -ne $oq["${k}_max"]) { [math]::Min([int]$oq["${k}_max"], $cur) } else { $cur }
    }
    $new.quality = $newQ

    $orphans = @($Assessment.reach.orphans)
    $kept = if ($old.orphans_allowed) { @($orphans | Where-Object { $_ -in @($old.orphans_allowed) }) } else { $orphans }
    $new.orphans_allowed = @($kept | Sort-Object { $_ } -CaseSensitive)

    $tiers = [ordered]@{}
    $oldTiers = if ($old.tiers) { $old.tiers } else { @{} }
    foreach ($name in (@($oldTiers.Keys) + $Tier | Sort-Object -Unique)) {
        if ($null -ne $oldTiers[$name]) { $tiers[$name] = $oldTiers[$name] }
    }
    if ($Ran -and $Assessment.coverage.pester) {
        $cov = $Assessment.coverage.pester
        $prev = $oldTiers[$Tier]
        $failedByFile = [ordered]@{}
        foreach ($t in @($PesterResult.Failed)) {
            $leaf = Split-Path -Leaf $t.ScriptBlock.File
            $failedByFile[$leaf] = 1 + [int]$failedByFile[$leaf]
        }
        $entry = [ordered]@{}
        if ($prev) {
            $entry.coverage_min_pct = [math]::Max([double]$prev.coverage_min_pct, [math]::Floor($cov.pct))
            $pm = [ordered]@{}
            foreach ($row in $cov.by_module) {
                $floor = [math]::Floor($row.pct)
                $was = $prev.per_module_min_pct[$row.module]
                $pm[$row.module] = if ($null -ne $was) { [math]::Max([double]$was, $floor) } else { $floor }
            }
            $entry.per_module_min_pct = $pm
            $kf = [ordered]@{}
            $kf.max_failed_containers = [math]::Min([int]$prev.known_failures.max_failed_containers, [int]$PesterResult.FailedContainersCount)
            $kf.max_failed_blocks = [math]::Min([int]$prev.known_failures.max_failed_blocks, [int]$PesterResult.FailedBlocksCount)
            $tbf = [ordered]@{}
            foreach ($leaf in ($prev.known_failures.tests_by_file.Keys | Sort-Object)) {
                $now = [int]$failedByFile[$leaf]
                if ($now -gt 0) { $tbf[$leaf] = [math]::Min([int]$prev.known_failures.tests_by_file[$leaf], $now) }
            }
            $kf.tests_by_file = $tbf
            $entry.known_failures = $kf
        } else {
            $entry.coverage_min_pct = [math]::Floor($cov.pct)
            $pm = [ordered]@{}
            foreach ($row in $cov.by_module) { $pm[$row.module] = [math]::Floor($row.pct) }
            $entry.per_module_min_pct = $pm
            $entry.known_failures = [ordered]@{
                max_failed_containers = [int]$PesterResult.FailedContainersCount
                max_failed_blocks     = [int]$PesterResult.FailedBlocksCount
                tests_by_file         = $failedByFile
            }
        }
        $entry.measured = [ordered]@{ pct = $cov.pct; files_analyzed = $cov.files_analyzed; module_files_total = $cov.module_files_total; failed_tests = [int]$PesterResult.FailedCount }
        $tiers[$Tier] = $entry
    }
    $new.tiers = $tiers
    $new
}

#endregion

#region runner

function Invoke-AssessmentPester {
    <#
    .SYNOPSIS
        Runs the tier's Pester suite with coverage over all of Modules/ and returns the Pester result.
    .DESCRIPTION
        Coverage uses Path 'Modules' with RecursePaths, not a '**' glob: Pester treats '**' like '*', which is how
        the old setting analysed 17 of 241 files. Coverage and result files go to a temp directory.
    .PARAMETER Root
        Repository root.
    .PARAMETER Tier
        Portable or Full.
    .PARAMETER ScratchDir
        Directory for Pester's own output files.
    #>
    [CmdletBinding()]
    param([string]$Root, [string]$Tier, [string]$ScratchDir)

    # Measure under the default preference. The tool itself runs with 'Stop', and that leaks into the tests: measured on
    # PcaiNativeInitializationContracts.Tests.ps1, a bare Invoke-Pester gives 1 failed test and Stop gives 3 failed
    # tests plus a failed block, so the tool would report numbers a developer's own Invoke-Pester never sees.
    $ErrorActionPreference = 'Continue'
    Import-Module Pester -MinimumVersion 5.0.0 -ErrorAction Stop
    $config = New-PesterConfiguration
    $config.Run.PassThru = $true
    $config.Run.Exit = $false
    $config.Output.Verbosity = 'Minimal'
    $config.TestResult.Enabled = $false
    if ($Tier -eq 'Portable') {
        $config.Run.Path = @('Tests/Unit')
        $config.Filter.Tag = @('Portable')
    } else {
        $config.Run.Path = @(Get-CiPesterPath -Root $Root)
    }
    $config.CodeCoverage.Enabled = $true
    $config.CodeCoverage.Path = @('Modules')
    $config.CodeCoverage.RecursePaths = $true
    $config.CodeCoverage.ExcludeTests = $true
    $config.CodeCoverage.OutputFormat = 'JaCoCo'
    $config.CodeCoverage.OutputPath = Join-Path $ScratchDir 'coverage.xml'
    Push-Location $Root
    try { Invoke-Pester -Configuration $config } finally { Pop-Location }
}

function ConvertTo-AssessmentJson {
    # Stable text: LF line endings, no BOM handled by the writer.
    param($InputObject)
    ($InputObject | ConvertTo-Json -Depth 12).Replace("`r`n", "`n") + "`n"
}

function Invoke-TestAssessment {
    <#
    .SYNOPSIS
        Builds the assessment document, applies the gate and writes the report. Returns the document.
    .PARAMETER Root
        Repository root.
    .PARAMETER Tier
        Portable or Full.
    .PARAMETER OutputPath
        JSON report path.
    .PARAMETER BaselinePath
        Baseline path.
    .PARAMETER UpdateBaseline
        Ratchet the baseline from this run.
    .PARAMETER NoRun
        Skip Pester and coverage.
    #>
    [CmdletBinding()]
    param([string]$Root, [string]$Tier, [string]$OutputPath, [string]$BaselinePath, [switch]$UpdateBaseline, [switch]$NoRun)

    $files = @(Get-AssessmentFileList -Root $Root)
    $testFiles = @($files | Where-Object { $_ -like '*.Tests.ps1' })
    $analysis = @{}
    foreach ($f in $testFiles) { $analysis[$f] = Get-PesterFileAnalysis -Path (Join-Path $Root $f) }

    $dirs = [ordered]@{}
    $quality = @{ no_assertion_it = @(); not_throw_only_it = @(); tautology_it = @() }
    $parseErrorFiles = [System.Collections.Generic.List[string]]::new()
    foreach ($f in $testFiles) {
        $a = $analysis[$f]
        $dir = [IO.Path]::GetDirectoryName($f).Replace('\', '/')
        if (-not $dirs.Contains($dir)) { $dirs[$dir] = [ordered]@{ dir = $dir; files = 0; it_blocks = 0; static_skips = 0; portable_files = 0 } }
        $d = $dirs[$dir]
        $d.files++
        $d.it_blocks += $a.It.Count
        $d.static_skips += @($a.It | Where-Object StaticSkip).Count
        if ($a.HasPortableTag) { $d.portable_files++ }
        if ($a.ParseErrors -gt 0) { $parseErrorFiles.Add($f) }
        foreach ($it in $a.It) {
            $row = [ordered]@{ file = $f; line = $it.Line; name = $it.Name }
            if ($it.NoAssertion) { $quality.no_assertion_it += , $row }
            if ($it.NotThrowOnly) { $quality.not_throw_only_it += , $row }
            if ($it.Tautology) { $quality.tautology_it += , $row }
        }
    }

    $reachRows = Get-TestReach -Root $Root -Files $files -Analysis $analysis
    $orphans = @($reachRows | Where-Object { $_.runners.Count -eq 0 -and $_.workflows.Count -eq 0 } | ForEach-Object { $_.file })

    $head = 'unknown'
    $dirty = $false
    if (Get-Command git -ErrorAction SilentlyContinue) {
        $sha = & git -C $Root rev-parse HEAD 2>$null
        if ($LASTEXITCODE -eq 0 -and $sha) { $head = [string]$sha }
        $porcelain = & git -C $Root status --porcelain 2>$null
        $dirty = ($LASTEXITCODE -eq 0 -and [bool]$porcelain)
    }

    $assessment = [ordered]@{
        schema_version = $script:AssessmentSchemaVersion
        head_sha       = $head
        dirty          = $dirty
        generated_at   = (Get-Date).ToUniversalTime().ToString('o')
        tier           = $Tier
        run            = (-not $NoRun)
        inventory      = [ordered]@{
            pester = [ordered]@{
                dirs   = @($dirs.Values)
                totals = [ordered]@{
                    files        = $testFiles.Count
                    it_blocks    = [int]($analysis.Values | ForEach-Object { $_.It.Count } | Measure-Object -Sum).Sum
                    static_skips = [int]($analysis.Values | ForEach-Object { @($_.It | Where-Object StaticSkip).Count } | Measure-Object -Sum).Sum
                }
            }
            rust   = [ordered]@{ note = 'static grep of #[test]/#[rstest]/#[test_case]; cfg(feature)-gated tests are counted whether or not the feature is enabled'; crates = @(Get-RustTestInventory -Root $Root -Files $files) }
            csharp = [ordered]@{ note = 'static grep of [Fact]/[Theory]/[Test]/[TestMethod]/[TestCase]'; projects = @(Get-CSharpTestInventory -Root $Root -Files $files) }
        }
        reach          = [ordered]@{
            runners   = @('Invoke-PortableTests', 'Invoke-AllTests', 'test.ps1', '.pester.ps1', 'Invoke-PcaiCi')
            files     = @($reachRows)
            orphans   = @($orphans)
        }
        quality        = [ordered]@{
            no_assertion_it   = $quality.no_assertion_it.Count
            not_throw_only_it = $quality.not_throw_only_it.Count
            tautology_it      = $quality.tautology_it.Count
            parse_error_files = $parseErrorFiles.Count
            details           = [ordered]@{
                no_assertion_it   = @($quality.no_assertion_it)
                not_throw_only_it = @($quality.not_throw_only_it)
                tautology_it      = @($quality.tautology_it)
                parse_error_files = @($parseErrorFiles)
            }
        }
    }

    $baseline = $null
    if (Test-Path -LiteralPath $BaselinePath) { $baseline = Get-Content -LiteralPath $BaselinePath -Raw | ConvertFrom-Json }

    $pester = $null
    $earlyViolations = [System.Collections.Generic.List[object]]::new()
    if (-not $NoRun) {
        $scratch = Join-Path ([IO.Path]::GetTempPath()) ("pcai-assess-" + [guid]::NewGuid().ToString('N'))
        [IO.Directory]::CreateDirectory($scratch) | Out-Null
        try {
            if (-not (Get-Module -ListAvailable -Name Pester | Where-Object Version -ge '5.0.0')) {
                $earlyViolations.Add((Get-GateViolation 'pester_missing' 'Pester 5+ is not installed, so nothing was measured'))
            } else {
                $pester = Invoke-AssessmentPester -Root $Root -Tier $Tier -ScratchDir $scratch
            }
        } catch {
            $earlyViolations.Add((Get-GateViolation 'pester_run' "Invoke-Pester threw: $($_.Exception.Message)"))
        } finally {
            Remove-Item -LiteralPath $scratch -Recurse -Force -ErrorAction SilentlyContinue
        }
        if ($pester) {
            $moduleFiles = @(Get-ModuleSourceFile -Files $files)
            $assessment.coverage = [ordered]@{ pester = Get-CoverageSummary -CodeCoverage $pester.CodeCoverage -Root $Root -ModuleFiles $moduleFiles }
            $assessment.run_summary = [ordered]@{
                result = [string]$pester.Result; total = [int]$pester.TotalCount; passed = [int]$pester.PassedCount
                failed = [int]$pester.FailedCount; skipped = [int]$pester.SkippedCount; not_run = [int]$pester.NotRunCount
                failed_blocks = [int]$pester.FailedBlocksCount; failed_containers = [int]$pester.FailedContainersCount
            }
        }
    }

    $ran = (-not $NoRun) -and ($null -ne $pester)

    # Ratchet first, then judge against the ratcheted baseline, so the printed verdict is the state the file now
    # records. A new orphan or a new failing file is never added to it, so those still fail.
    if ($UpdateBaseline) {
        $updated = Update-AssessmentBaseline -Baseline $baseline -Assessment $assessment -Tier $Tier -PesterResult $pester -Ran:$ran
        $updatedJson = ConvertTo-AssessmentJson $updated
        [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($BaselinePath)) | Out-Null
        [IO.File]::WriteAllText($BaselinePath, $updatedJson, [Text.UTF8Encoding]::new($false))
        $baseline = $updatedJson | ConvertFrom-Json
        $assessment.baseline_updated = $true
    }

    $violations = [System.Collections.Generic.List[object]]::new()
    foreach ($e in $earlyViolations) { $violations.Add($e) }
    foreach ($g in @(Test-AssessmentGate -Assessment $assessment -Baseline $baseline -Tier $Tier -PesterResult $pester -Ran:$ran)) { $violations.Add($g) }

    $tb = if ($baseline) { $baseline.tiers.$Tier } else { $null }
    $assessment.gates = [ordered]@{
        target_coverage_pct      = if ($baseline) { $baseline.target_coverage_pct } else { 85 }
        coverage_min             = if ($tb) { $tb.coverage_min_pct } else { $null }
        files_analyzed_ratio_min = if ($baseline) { $baseline.files_analyzed_ratio_min } else { 0.95 }
        per_module_min           = if ($tb) { $tb.per_module_min_pct } else { $null }
        baseline_path            = [IO.Path]::GetRelativePath($Root, $BaselinePath).Replace('\', '/')
    }
    $assessment.violations = @($violations)
    $assessment.verdict = if ($violations.Count -eq 0) { 'pass' } else { 'fail' }

    [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($OutputPath)) | Out-Null
    [IO.File]::WriteAllText($OutputPath, (ConvertTo-AssessmentJson $assessment), [Text.UTF8Encoding]::new($false))

    $assessment
}

#endregion

# Dot-sourcing (the unit tests) stops here: functions are defined, nothing runs.
if ($MyInvocation.InvocationName -eq '.') { return }

$ErrorActionPreference = 'Stop'

if (-not $RepoRoot) { $RepoRoot = Split-Path -Parent $PSScriptRoot }
$RepoRoot = [IO.Path]::GetFullPath($RepoRoot)
if (-not $OutputPath) { $OutputPath = Join-Path $RepoRoot '.pcai/assessment/test-assessment.json' }
if (-not $BaselinePath) { $BaselinePath = Join-Path $RepoRoot 'Config/test-baseline.json' }
$OutputPath = [IO.Path]::GetFullPath($OutputPath, (Get-Location).Path)
$BaselinePath = [IO.Path]::GetFullPath($BaselinePath, (Get-Location).Path)

$result = Invoke-TestAssessment -Root $RepoRoot -Tier $Tier -OutputPath $OutputPath -BaselinePath $BaselinePath -UpdateBaseline:$UpdateBaseline -NoRun:$NoRun

$q = $result.quality
Write-Host "Test assessment ($Tier, $(if ($NoRun) { 'static' } else { 'run' })) at $($result.head_sha.Substring(0, [math]::Min(8, $result.head_sha.Length)))$(if ($result.dirty) { ' (dirty)' })"
Write-Host ("  Pester: {0} files, {1} It blocks, {2} static skips" -f $result.inventory.pester.totals.files, $result.inventory.pester.totals.it_blocks, $result.inventory.pester.totals.static_skips)
Write-Host ("  Quality: no-assertion {0}, not-throw-only {1}, tautology {2}, parse-error files {3}" -f $q.no_assertion_it, $q.not_throw_only_it, $q.tautology_it, $q.parse_error_files)
Write-Host ("  Reach: {0} test files, {1} orphan(s)" -f @($result.reach.files).Count, @($result.reach.orphans).Count)
if ($result.Contains('coverage')) {
    $c = $result.coverage.pester
    Write-Host ("  Coverage: {0}% of {1} commands; {2}/{3} module files analysed (min {4}%, target {5}%)" -f $c.pct, $c.commands_analyzed, $c.files_analyzed, $c.module_files_total, $result.gates.coverage_min, $result.gates.target_coverage_pct)
    $r = $result.run_summary
    Write-Host ("  Run: {0}; passed {1}, failed {2}, skipped {3}, failed blocks {4}, failed containers {5}" -f $r.result, $r.passed, $r.failed, $r.skipped, $r.failed_blocks, $r.failed_containers)
}
foreach ($viol in $result.violations) { Write-Host "  VIOLATION [$($viol.code)] $($viol.message)" -ForegroundColor Red }
if ($result.Contains('baseline_updated')) { Write-Host "  Baseline ratcheted: $BaselinePath" -ForegroundColor Cyan }
Write-Host "Verdict: $($result.verdict.ToUpperInvariant())  Report: $OutputPath" -ForegroundColor $(if ($result.verdict -eq 'pass') { 'Green' } else { 'Red' })
if ($result.verdict -ne 'pass') { exit 1 }
