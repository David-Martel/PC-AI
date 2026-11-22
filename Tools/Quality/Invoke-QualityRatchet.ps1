#Requires -Version 7.0
<#
.SYNOPSIS
    Fail-closed static-quality ratchet: counts known-bad patterns and fails when any count rises.

.DESCRIPTION
    The repo carries a large backlog of panic-prone Rust (unwrap/expect/indexing), swallowed
    PowerShell errors and C# catch-alls. Fixing it all at once is not realistic, and a zero-tolerance
    gate would fail on day one. A ratchet records today's counts in Config/quality-baseline.json and
    fails a run when any count is higher than the baseline, so the backlog can only shrink.

    Sources (all counted per "<tool>|<rule>|<prod|test>"):
      ast-grep          every rule under rules/core, rules/rust and rules/csharp, loaded through
                        sgconfig.ratchet.yml (not sgconfig.yml: lefthook and the pre-build gate read that
                        one and most Rust rules are severity error).
      pssa              PSScriptAnalyzer with PSScriptAnalyzerSettings.ratchet.psd1, which re-enables
                        PSAvoidUsingEmptyCatchBlock (excluded by the repo settings) and
                        PSAvoidOverwritingBuiltInCmdlets. -IncludeRepoSettings merges in the repo
                        gate's rules too; that takes 8-12 minutes, and Build.ps1 -LintProfile powershell
                        already enforces them with zero tolerance.
      pcai-ps-ast       PowerShell AST checks, see PcaiPsAstChecks.psm1 for the rules and heuristic limits.
      pcai-cs-catch     C# catch-all heuristics, see PcaiPsAstChecks.psm1.

    Prod versus test is decided by path (tests/ benches/ examples/ Tests/ *.Tests/ *Tests.cs
    *.Tests.ps1) and, for Rust, by position after a `#[cfg(test)] mod`. vendor/, third_party/,
    worktrees/ and Tests/Fixtures/ are not scanned.

    Fail-closed conditions (exit 1): a count above the baseline; ast-grep or PSScriptAnalyzer missing; a
    tool whose MAJOR.MINOR differs from the baseline's; a tool that fails its canary (a known-bad probe
    file it must flag) or yields nothing where the baseline has findings; a missing or unreadable
    baseline; any file the checks could not read.

    Files scanned are the git-tracked files of RepoRoot (committed or staged), so untracked scratch
    files never change the counts. A RepoRoot that is not the top of a git repository (test fixtures)
    is walked instead.

.PARAMETER RepoRoot
    Tree to scan. Default: the repository containing this script.

.PARAMETER BaselinePath
    Baseline JSON. Default: Config/quality-baseline.json under RepoRoot.

.PARAMETER OutputPath
    Result JSON. Default: .pcai/assessment/quality.json under RepoRoot (.pcai/ is gitignored).

.PARAMETER Scope
    prod, test or all (default). Restricts which counts are compared.

.PARAMETER UpdateBaseline
    Lowers baseline counts to the current ones. Creates the file when it does not exist. Never raises a
    count, never changes tool versions, and refuses to run on a tree with modified tracked files.

.PARAMETER Rebaseline
    Rewrites the whole baseline from the current run, raising counts and recording tool versions as
    found. For a reviewed change that upgrades a tool or adds rules; every raised count is printed.

.PARAMETER BaseRef
    Ref used to list changed files for new_findings. Default origin/main.

.PARAMETER AstGrepCommand
    ast-grep executable. Default: `ast-grep`, then `sg` (only if it identifies itself as ast-grep; on
    Linux `sg` is the group-switching tool).

.PARAMETER PssaModuleName
    PSScriptAnalyzer module name. Overridable so tests can simulate it being absent.

.PARAMETER IncludeRepoSettings
    Also run the repo gate's PSScriptAnalyzer rules (slow).

.EXAMPLE
    pwsh -NoProfile -File Tools/Quality/Invoke-QualityRatchet.ps1

.EXAMPLE
    pwsh -NoProfile -File Tools/Quality/Invoke-QualityRatchet.ps1 -UpdateBaseline
#>
[CmdletBinding()]
param(
    [string]$RepoRoot,
    [string]$BaselinePath,
    [string]$OutputPath,
    [ValidateSet('prod', 'test', 'all')]
    [string]$Scope = 'all',
    [switch]$UpdateBaseline,
    [switch]$Rebaseline,
    [string]$BaseRef = 'origin/main',
    [string]$AstGrepCommand,
    [string]$PssaModuleName = 'PSScriptAnalyzer',
    [string]$SgConfigPath,
    [string]$PssaSettingsPath,
    [switch]$IncludeRepoSettings
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$script:SchemaVersion = 1
$script:ToolRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
# A tool whose baseline total is at least this must report something; zero means it silently did nothing.
$script:SuspiciousZeroThreshold = 10
$script:MaxNewFindings = 1000

Import-Module (Join-Path $PSScriptRoot 'PcaiPsAstChecks.psm1') -Force

function ConvertTo-PosixPath {
    param([string]$Path)
    return $Path.Replace('\', '/')
}

function Invoke-NativeTool {
    # Runs an executable and returns its exit code and output. Callers must look at ExitCode.
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [string[]]$Arguments = @(),
        [string]$WorkingDirectory
    )
    $errPath = [System.IO.Path]::GetTempFileName()
    if ($WorkingDirectory) { Push-Location -LiteralPath $WorkingDirectory }
    try {
        $lines = @(& $FilePath @Arguments 2> $errPath)
        $exitCode = $LASTEXITCODE
        $stderr = @([System.IO.File]::ReadAllLines($errPath))
    } finally {
        if ($WorkingDirectory) { Pop-Location }
        [System.IO.File]::Delete($errPath)
    }
    return [pscustomobject]@{ ExitCode = $exitCode; Output = $lines; Error = $stderr }
}

function Get-MajorMinor {
    param([string]$Version)
    if ($Version -match '^(\d+)\.(\d+)') { return "$($Matches[1]).$($Matches[2])" }
    return $Version
}

function Sort-Ordinal {
    # Ordinal sort of items by a string key; Sort-Object would apply culture-sensitive rules.
    param([object[]]$Items, [scriptblock]$KeyOf)
    if ($Items.Count -lt 2) { return , $Items }
    $keys = [string[]]($Items | ForEach-Object { & $KeyOf $_ })
    $copy = [object[]]$Items.Clone()
    [Array]::Sort($keys, $copy, [StringComparer]::Ordinal)
    return , $copy
}

function Write-StableJson {
    param([object]$Value, [string]$Path)
    $directory = Split-Path -Parent $Path
    if ($directory -and -not (Test-Path -LiteralPath $directory)) { [void][System.IO.Directory]::CreateDirectory($directory) }
    $json = ($Value | ConvertTo-Json -Depth 8).Replace("`r`n", "`n") + "`n"
    [System.IO.File]::WriteAllText($Path, $json, [System.Text.UTF8Encoding]::new($false))
}

function Get-GitInfo {
    param([string]$Root)
    $info = [ordered]@{ IsRepo = $false; HeadSha = $null; Dirty = $false; Tracked = $null }
    $git = Get-Command git -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $git) { return [pscustomobject]$info }
    $top = Invoke-NativeTool -FilePath $git.Source -Arguments @('-C', $Root, 'rev-parse', '--show-toplevel')
    if ($top.ExitCode -ne 0 -or $top.Output.Count -eq 0) { return [pscustomobject]$info }
    $topPath = (Resolve-Path -LiteralPath ([string]$top.Output[0])).Path
    $rootPath = (Resolve-Path -LiteralPath $Root).Path
    $comparison = if ($IsWindows) { [StringComparison]::OrdinalIgnoreCase } else { [StringComparison]::Ordinal }
    if (-not [string]::Equals($topPath.TrimEnd('\', '/'), $rootPath.TrimEnd('\', '/'), $comparison)) { return [pscustomobject]$info }
    $info.IsRepo = $true
    $head = Invoke-NativeTool -FilePath $git.Source -Arguments @('-C', $Root, 'rev-parse', 'HEAD')
    if ($head.ExitCode -eq 0 -and $head.Output.Count -gt 0) { $info.HeadSha = [string]$head.Output[0] }
    $status = Invoke-NativeTool -FilePath $git.Source -Arguments @('-C', $Root, 'status', '--porcelain', '--untracked-files=no')
    if ($status.ExitCode -ne 0) { throw "git status failed in ${Root}: $($status.Error -join ' ')" }
    $info.Dirty = @($status.Output | Where-Object { $_ }).Count -gt 0
    $list = Invoke-NativeTool -FilePath $git.Source -Arguments @('-C', $Root, '-c', 'core.quotepath=false', 'ls-files')
    if ($list.ExitCode -ne 0) { throw "git ls-files failed in ${Root}: $($list.Error -join ' ')" }
    $info.Tracked = @($list.Output | Where-Object { $_ } | ForEach-Object { ConvertTo-PosixPath $_ })
    $info['GitPath'] = $git.Source
    return [pscustomobject]$info
}

function Get-ChangedFileSet {
    # Files changed against BaseRef (merge-base diff) plus uncommitted tracked changes; $null when unknown.
    param([string]$Root, $Git)
    if (-not $Git.IsRepo) { return $null }
    $set = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    $committed = Invoke-NativeTool -FilePath $Git.GitPath -Arguments @('-C', $Root, '-c', 'core.quotepath=false', 'diff', '--name-only', "$BaseRef...HEAD")
    if ($committed.ExitCode -ne 0) { return $null }
    foreach ($name in $committed.Output) { if ($name) { [void]$set.Add((ConvertTo-PosixPath $name)) } }
    $working = Invoke-NativeTool -FilePath $Git.GitPath -Arguments @('-C', $Root, '-c', 'core.quotepath=false', 'diff', '--name-only', 'HEAD')
    if ($working.ExitCode -eq 0) { foreach ($name in $working.Output) { if ($name) { [void]$set.Add((ConvertTo-PosixPath $name)) } } }
    return $set
}

function Get-ScanFile {
    # Repo-relative, /-separated paths of the files the ratchet may look at.
    param([string]$Root, $Git)
    $excluded = [regex]::new('(^|/)(vendor|third_party|worktrees|\.pcai|\.git|node_modules|target|obj)/|^Tests/Fixtures/', 'IgnoreCase')
    $wanted = [regex]::new('\.(rs|cs|ps1|psm1)$', 'IgnoreCase')
    if ($Git.IsRepo) { $all = $Git.Tracked } else {
        $rootFull = (Resolve-Path -LiteralPath $Root).Path.TrimEnd('\', '/')
        $all = [System.Collections.Generic.List[string]]::new()
        # Walk by hand instead of Get-ChildItem -Recurse so excluded directories are never entered and no
        # junction or symlink is followed: Native/pcai_core/target/** is a junction to the shared cargo
        # cache on some hosts and must never be traversed.
        $skipNames = [System.Collections.Generic.HashSet[string]]::new([string[]]@('vendor', 'third_party', 'worktrees', '.pcai', '.git', 'node_modules', 'target', 'obj'), [StringComparer]::OrdinalIgnoreCase)
        $pending = [System.Collections.Generic.Stack[string]]::new()
        $pending.Push($rootFull)
        while ($pending.Count -gt 0) {
            $directory = $pending.Pop()
            foreach ($entry in (Get-ChildItem -LiteralPath $directory -Force)) {
                if ($entry.Attributes -band [System.IO.FileAttributes]::ReparsePoint) { continue }
                if ($entry.PSIsContainer) {
                    if (-not $skipNames.Contains($entry.Name)) { $pending.Push($entry.FullName) }
                } else {
                    $all.Add((ConvertTo-PosixPath $entry.FullName.Substring($rootFull.Length + 1)))
                }
            }
        }
    }
    $files = @($all | Where-Object { $wanted.IsMatch($_) -and -not $excluded.IsMatch($_) -and (Test-Path -LiteralPath (Join-Path $Root $_) -PathType Leaf) })
    return , (Sort-Ordinal -Items $files -KeyOf { param($f) $f })
}

function Resolve-AstGrep {
    param([string]$Requested)
    $candidates = if ($Requested) { @($Requested) } else { @('ast-grep', 'sg') }
    foreach ($candidate in $candidates) {
        $command = Get-Command $candidate -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
        if (-not $command) { continue }
        $probe = Invoke-NativeTool -FilePath $command.Source -Arguments @('--version')
        $text = ($probe.Output -join ' ')
        if ($probe.ExitCode -eq 0 -and $text -match '^\s*ast-grep\s+(\d+\.\d+\.\d+)') {
            return [pscustomobject]@{ Path = $command.Source; Version = $Matches[1] }
        }
    }
    return $null
}

function New-Finding {
    param([string]$Tool, [string]$Rule, [string]$Scope, [string]$File, [int]$Line, [string]$Message)
    return [pscustomobject]@{ tool = $Tool; rule = $Rule; scope = $Scope; file = $File; line = $Line; message = $Message }
}

function Get-RustTestStart {
    param([string]$Root, [string]$Relative, [hashtable]$Cache)
    if (-not $Cache.ContainsKey($Relative)) {
        $Cache[$Relative] = Get-PcaiRustTestStartLine -Path (Join-Path $Root $Relative)
    }
    return $Cache[$Relative]
}

function Invoke-AstGrepScan {
    param($Tool, [string]$Root, [string[]]$Files, [string]$ConfigPath)

    $scanFailures = [System.Collections.Generic.List[string]]::new()
    $findings = [System.Collections.Generic.List[object]]::new()
    if (-not (Test-Path -LiteralPath $ConfigPath)) {
        $scanFailures.Add("ast-grep config not found: $ConfigPath")
        return [pscustomobject]@{ Findings = $findings; Failures = $scanFailures }
    }

    # Canary: a probe file every loaded rule directory must flag, so a config or rule-dir problem that
    # makes ast-grep scan with no rules shows up as a failure instead of as an improvement.
    $probeDir = Join-Path ([System.IO.Path]::GetTempPath()) ("pcai-ratchet-probe-" + [guid]::NewGuid().ToString('N'))
    [void][System.IO.Directory]::CreateDirectory($probeDir)
    try {
        [System.IO.File]::WriteAllText((Join-Path $probeDir 'probe.rs'), "fn f(o: Option<u8>, tx: Tx) { let x = o.unwrap(); let _ = tx.send(1); }`n")
        $probe = Invoke-NativeTool -FilePath $Tool.Path -Arguments @('scan', '--config', $ConfigPath, '--json=stream', '.') -WorkingDirectory $probeDir
        $probeRules = @($probe.Output | Where-Object { $_ -like '{*' } | ForEach-Object { ($_ | ConvertFrom-Json).ruleId })
        foreach ($expected in 'unwrap-call', 'let-underscore-send') {
            if ($probeRules -notcontains $expected) { $scanFailures.Add("ast-grep canary did not flag rule '$expected' with $ConfigPath (exit $($probe.ExitCode)); rules are not loading") }
        }
    } finally {
        [System.IO.Directory]::Delete($probeDir, $true)
    }
    if ($scanFailures.Count -gt 0) { return [pscustomobject]@{ Findings = $findings; Failures = $scanFailures } }

    $paths = @($Files | ForEach-Object { $i = $_.IndexOf('/'); if ($i -lt 0) { $_ } else { $_.Substring(0, $i) } } | Select-Object -Unique)
    $paths = @($paths | Where-Object { $_ -notmatch '^(vendor|third_party|worktrees|\.pcai|\.git|node_modules)$' })
    $result = Invoke-NativeTool -FilePath $Tool.Path -Arguments (@('scan', '--config', $ConfigPath, '--json=stream') + $paths) -WorkingDirectory $Root
    $stderrText = ($result.Error -join "`n")
    # ast-grep exits 1 after a successful scan that found error-severity diagnostics.
    $benign = $stderrText -match '^\s*Error: \d+ error\(s\) found in code\.' -or [string]::IsNullOrWhiteSpace($stderrText)
    if ($result.ExitCode -gt 1 -or ($result.ExitCode -eq 1 -and -not $benign)) {
        $scanFailures.Add("ast-grep scan failed (exit $($result.ExitCode)): $($result.Error | Select-Object -First 3)")
        return [pscustomobject]@{ Findings = $findings; Failures = $scanFailures }
    }

    $fileSet = [System.Collections.Generic.HashSet[string]]::new([string[]]$Files, [StringComparer]::Ordinal)
    $rustStart = @{}
    foreach ($line in $result.Output) {
        if ([string]::IsNullOrWhiteSpace($line)) { continue }
        if ($line[0] -ne '{') { $scanFailures.Add("ast-grep produced a non-JSON line: $line"); break }
        $parsed = $line | ConvertFrom-Json
        $relative = ConvertTo-PosixPath ([string]$parsed.file)
        if ($relative.StartsWith('./')) { $relative = $relative.Substring(2) }
        if (-not $fileSet.Contains($relative)) { continue }
        if ($parsed.ruleId -eq 'unused-suppression') { continue }
        $scope = Get-PcaiQualityFileScope -RelativePath $relative
        $zeroBasedLine = [int]$parsed.range.start.line
        if ($scope -eq 'prod' -and $relative.EndsWith('.rs', [StringComparison]::OrdinalIgnoreCase) -and $zeroBasedLine -ge (Get-RustTestStart -Root $Root -Relative $relative -Cache $rustStart)) {
            $scope = 'test'
        }
        $findings.Add((New-Finding -Tool 'ast-grep' -Rule ([string]$parsed.ruleId) -Scope $scope -File $relative -Line ($zeroBasedLine + 1) -Message ([string]$parsed.message)))
    }
    return [pscustomobject]@{ Findings = $findings; Failures = $scanFailures }
}

function Get-PssaSetting {
    param([string]$RatchetPath, [bool]$MergeRepoSettings)
    if (-not $MergeRepoSettings) { return $RatchetPath }
    $repo = Import-PowerShellDataFile -LiteralPath (Join-Path $script:ToolRoot 'PSScriptAnalyzerSettings.psd1')
    $ratchet = Import-PowerShellDataFile -LiteralPath $RatchetPath
    $include = [System.Collections.Generic.List[string]]::new()
    foreach ($rule in @($repo.IncludeRules) + @($ratchet.IncludeRules)) { if ($rule -and -not $include.Contains($rule)) { $include.Add($rule) } }
    $merged = @{ Severity = [string[]]@('Error', 'Warning', 'Information'); IncludeRules = [string[]]$include; ExcludeRules = [string[]]@($repo.ExcludeRules | Where-Object { $ratchet.IncludeRules -notcontains $_ }) }
    $rules = @{}
    foreach ($source in @($repo.Rules, $ratchet.Rules)) { if ($source) { foreach ($key in $source.Keys) { $rules[$key] = $source[$key] } } }
    # The repo file's PowerShellVersion values ('5.1', '7.4') are not PSScriptAnalyzer profile names; the analyzer
    # rejects them as soon as the rule is enabled, so let the rule use its default profile.
    $rules.Remove('PSAvoidOverwritingBuiltInCmdlets')
    $merged['Rules'] = $rules
    return $merged
}

function Invoke-PssaScan {
    param([string]$Root, [string[]]$Files, $Settings)

    $scanFailures = [System.Collections.Generic.List[string]]::new()
    $findings = [System.Collections.Generic.List[object]]::new()
    $ps = @($Files | Where-Object { $_ -match '\.(ps1|psm1)$' })
    if ($ps.Count -eq 0) { return [pscustomobject]@{ Findings = $findings; Failures = $scanFailures } }

    # Canary: an empty catch the ruleset must flag.
    $probe = Join-Path ([System.IO.Path]::GetTempPath()) ("pcai-ratchet-probe-" + [guid]::NewGuid().ToString('N') + '.ps1')
    try {
        [System.IO.File]::WriteAllText($probe, "try { 1 } catch { }`n")
        $canary = @(Invoke-ScriptAnalyzer -Path $probe -Settings $Settings)
        if (@($canary | Where-Object { $_.RuleName -eq 'PSAvoidUsingEmptyCatchBlock' }).Count -ne 1) {
            $scanFailures.Add('PSScriptAnalyzer canary did not flag PSAvoidUsingEmptyCatchBlock; the ruleset is not loading')
            return [pscustomobject]@{ Findings = $findings; Failures = $scanFailures }
        }
    } finally {
        [System.IO.File]::Delete($probe)
    }

    $fileSet = [System.Collections.Generic.HashSet[string]]::new([string[]]$ps, [StringComparer]::Ordinal)
    $rootFull = (Resolve-Path -LiteralPath $Root).Path.TrimEnd('\', '/')
    $roots = @($ps | ForEach-Object { $i = $_.IndexOf('/'); if ($i -lt 0) { $_ } else { $_.Substring(0, $i) } } | Select-Object -Unique)
    foreach ($entry in $roots) {
        $target = Join-Path $Root $entry
        $isDirectory = Test-Path -LiteralPath $target -PathType Container
        $errors = $null
        $results = if ($isDirectory) { @(Invoke-ScriptAnalyzer -Path $target -Recurse -Settings $Settings -ErrorVariable errors -ErrorAction SilentlyContinue) }
        else { @(Invoke-ScriptAnalyzer -Path $target -Settings $Settings -ErrorVariable errors -ErrorAction SilentlyContinue) }
        if ($errors -and @($errors).Count -gt 0) { $scanFailures.Add("PSScriptAnalyzer reported an error scanning '$entry': $(@($errors)[0])") }
        foreach ($result in $results) {
            if (-not $result.ScriptPath) { continue }
            $full = (ConvertTo-PosixPath $result.ScriptPath)
            $prefix = (ConvertTo-PosixPath $rootFull) + '/'
            if (-not $full.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) { continue }
            $relative = $full.Substring($prefix.Length)
            if (-not $fileSet.Contains($relative)) { continue }
            $rule = if ([string]$result.Severity -eq 'ParseError') { 'ParseError' } else { [string]$result.RuleName }
            $findings.Add((New-Finding -Tool 'pssa' -Rule $rule -Scope (Get-PcaiQualityFileScope -RelativePath $relative) -File $relative -Line ([int]$result.Line) -Message ([string]$result.Message)))
        }
    }
    return [pscustomobject]@{ Findings = $findings; Failures = $scanFailures }
}

function Invoke-OwnChecks {
    param([string]$Root, [string[]]$Files)
    $scanFailures = [System.Collections.Generic.List[string]]::new()
    $findings = [System.Collections.Generic.List[object]]::new()
    foreach ($relative in $Files) {
        $full = Join-Path $Root $relative
        $scope = Get-PcaiQualityFileScope -RelativePath $relative
        try {
            if ($relative -match '\.(ps1|psm1)$') {
                foreach ($item in (Get-PcaiPsAstFinding -Path $full)) {
                    $findings.Add((New-Finding -Tool 'pcai-ps-ast' -Rule $item.Rule -Scope $scope -File $relative -Line $item.Line -Message $item.Message))
                }
            } elseif ($relative -match '\.cs$') {
                foreach ($item in (Get-PcaiCSharpCatchFinding -Path $full)) {
                    $findings.Add((New-Finding -Tool 'pcai-cs-catch' -Rule $item.Rule -Scope $scope -File $relative -Line $item.Line -Message $item.Message))
                }
            }
        } catch {
            $scanFailures.Add("could not scan ${relative}: $($_.Exception.Message)")
        }
    }
    return [pscustomobject]@{ Findings = $findings; Failures = $scanFailures }
}

function Read-Baseline {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    $data = [System.IO.File]::ReadAllText($Path) | ConvertFrom-Json -AsHashtable
    foreach ($required in 'schema_version', 'tools', 'counts') {
        if (-not $data.ContainsKey($required)) { throw "baseline $Path is missing '$required'" }
    }
    if ($data.schema_version -ne $script:SchemaVersion) { throw "baseline $Path has schema_version $($data.schema_version); this script reads $script:SchemaVersion" }
    return $data
}

function Invoke-QualityRatchet {
    $failures = [System.Collections.Generic.List[string]]::new()
    $notes = [System.Collections.Generic.List[string]]::new()

    $root = if ($RepoRoot) { (Resolve-Path -LiteralPath $RepoRoot).Path } else { $script:ToolRoot }
    $baselineFile = if ($BaselinePath) { $BaselinePath } else { Join-Path $root 'Config/quality-baseline.json' }
    $outputFile = if ($OutputPath) { $OutputPath } else { Join-Path $root '.pcai/assessment/quality.json' }
    $sgConfig = if ($SgConfigPath) { $SgConfigPath } else { Join-Path $script:ToolRoot 'sgconfig.ratchet.yml' }
    $pssaSettingsFile = if ($PssaSettingsPath) { $PssaSettingsPath } else { Join-Path $script:ToolRoot 'PSScriptAnalyzerSettings.ratchet.psd1' }
    if ($UpdateBaseline -and $Rebaseline) { throw '-UpdateBaseline and -Rebaseline are mutually exclusive' }

    $git = Get-GitInfo -Root $root
    $files = Get-ScanFile -Root $root -Git $git

    # --- tools -------------------------------------------------------------------------------
    $tools = [ordered]@{}
    $sg = Resolve-AstGrep -Requested $AstGrepCommand
    if ($sg) { $tools['ast-grep'] = $sg.Version } else { $failures.Add("ast-grep is not installed or not on PATH (tried '$(if ($AstGrepCommand) { $AstGrepCommand } else { 'ast-grep, sg' })'); install it, e.g. 'cargo install ast-grep --locked' or 'npm i -g @ast-grep/cli'") }
    $pssaModule = Get-Module -ListAvailable -Name $PssaModuleName | Sort-Object Version -Descending | Select-Object -First 1
    if ($pssaModule) { $tools['PSScriptAnalyzer'] = $pssaModule.Version.ToString() } else { $failures.Add("PowerShell module '$PssaModuleName' is not installed; run: Install-Module PSScriptAnalyzer -Scope CurrentUser") }
    $tools['pcai-quality-checks'] = Get-PcaiChecksVersion

    # --- baseline ----------------------------------------------------------------------------
    $baseline = $null
    try { $baseline = Read-Baseline -Path $baselineFile } catch { $failures.Add("baseline unreadable: $($_.Exception.Message)") }
    if ($null -eq $baseline -and -not (Test-Path -LiteralPath $baselineFile) -and -not $UpdateBaseline -and -not $Rebaseline) {
        $failures.Add("baseline not found: $baselineFile; create it with -UpdateBaseline from a clean checkout")
    }
    if ($baseline -and -not $Rebaseline) {
        foreach ($name in $baseline.tools.Keys) {
            if (-not $tools.Contains($name)) { continue }
            if ((Get-MajorMinor $tools[$name]) -ne (Get-MajorMinor ([string]$baseline.tools[$name]))) {
                $failures.Add("$name is $($tools[$name]) but the baseline was recorded with $($baseline.tools[$name]) (MAJOR.MINOR must match). Install the baseline's version, or in a reviewed change re-record with: pwsh Tools/Quality/Invoke-QualityRatchet.ps1 -Rebaseline")
            }
        }
    }

    # --- scans -------------------------------------------------------------------------------
    $findings = [System.Collections.Generic.List[object]]::new()
    if ($failures.Count -eq 0 -or ($sg -and $pssaModule)) {
        if ($sg) {
            $r = Invoke-AstGrepScan -Tool $sg -Root $root -Files $files -ConfigPath $sgConfig
            $findings.AddRange([object[]]@($r.Findings)); foreach ($f in $r.Failures) { $failures.Add($f) }
        }
        if ($pssaModule) {
            Import-Module $pssaModule.Path -Force -ErrorAction Stop
            $settings = Get-PssaSetting -RatchetPath $pssaSettingsFile -MergeRepoSettings $IncludeRepoSettings.IsPresent
            $r = Invoke-PssaScan -Root $root -Files $files -Settings $settings
            $findings.AddRange([object[]]@($r.Findings)); foreach ($f in $r.Failures) { $failures.Add($f) }
        }
        $r = Invoke-OwnChecks -Root $root -Files $files
        $findings.AddRange([object[]]@($r.Findings)); foreach ($f in $r.Failures) { $failures.Add($f) }
    }

    $inScope = @($findings | Where-Object { $Scope -eq 'all' -or $_.scope -eq $Scope })
    $sorted = Sort-Ordinal -Items $inScope -KeyOf { param($f) '{0}{1}{2:D8}{3}{4}' -f $f.file, [char]0, $f.line, $f.tool, $f.rule }

    $counts = [System.Collections.Generic.SortedDictionary[string, int]]::new([StringComparer]::Ordinal)
    foreach ($f in $sorted) {
        $key = '{0}|{1}|{2}' -f $f.tool, $f.rule, $f.scope
        $counts[$key] = $(if ($counts.ContainsKey($key)) { $counts[$key] + 1 } else { 1 })
    }

    # --- compare -----------------------------------------------------------------------------
    $baseCounts = if ($baseline) { $baseline.counts } else { @{} }
    $inScopeKey = { param($k) $Scope -eq 'all' -or $k.EndsWith("|$Scope", [StringComparison]::Ordinal) }
    $keys = [System.Collections.Generic.SortedSet[string]]::new([StringComparer]::Ordinal)
    foreach ($k in $counts.Keys) { if (& $inScopeKey $k) { [void]$keys.Add($k) } }
    foreach ($k in $baseCounts.Keys) { if (& $inScopeKey $k) { [void]$keys.Add($k) } }
    $delta = [ordered]@{}
    $regressions = [System.Collections.Generic.List[string]]::new()
    $improvements = [System.Collections.Generic.List[string]]::new()
    foreach ($k in $(if ($baseline) { $keys } else { @() })) {
        $current = if ($counts.ContainsKey($k)) { $counts[$k] } else { 0 }
        $base = if ($baseCounts.ContainsKey($k)) { [int]$baseCounts[$k] } else { 0 }
        if ($current -eq $base) { continue }
        $delta[$k] = [ordered]@{ baseline = $base; current = $current; change = $current - $base }
        if ($current -gt $base) { $regressions.Add("$k rose from $base to $current") } else { $improvements.Add("$k fell from $base to $current") }
    }
    if ($baseline -and $failures.Count -eq 0) {
        foreach ($prefix in 'ast-grep|', 'pssa|', 'pcai-ps-ast|', 'pcai-cs-catch|') {
            $baseTotal = 0
            foreach ($k in $baseCounts.Keys) { if ($k.StartsWith($prefix, [StringComparison]::Ordinal) -and (& $inScopeKey $k)) { $baseTotal += [int]$baseCounts[$k] } }
            $currentTotal = 0
            foreach ($k in $counts.Keys) { if ($k.StartsWith($prefix, [StringComparison]::Ordinal) -and (& $inScopeKey $k)) { $currentTotal += $counts[$k] } }
            if ($baseTotal -ge $script:SuspiciousZeroThreshold -and $currentTotal -eq 0) {
                $failures.Add("$($prefix.TrimEnd('|')) reported no findings but the baseline records $baseTotal; treating the tool as malfunctioning, not as a clean tree")
            }
        }
    }
    # -Rebaseline exists to accept raised counts (tool upgrade, new rules); every other mode fails on them.
    if (-not $Rebaseline) { foreach ($r in $regressions) { $failures.Add("count above baseline: $r") } }

    # --- new findings in changed files -------------------------------------------------------
    $changed = Get-ChangedFileSet -Root $root -Git $git
    $newFindings = @()
    $changedCount = $null
    if ($null -ne $changed) {
        $changedCount = $changed.Count
        $newFindings = @($sorted | Where-Object { $changed.Contains($_.file) })
    } else {
        $notes.Add("changed files unavailable (not a git root or '$BaseRef' not found); new_findings is empty")
    }

    $verdict = if ($failures.Count -eq 0) { 'pass' } else { 'fail' }

    # --- baseline writes ---------------------------------------------------------------------
    $baselineAction = 'none'
    if (($UpdateBaseline -or $Rebaseline) -and $verdict -eq 'pass') {
        if ($git.IsRepo -and $git.Dirty) {
            $failures.Add('refusing to write the baseline: tracked files have uncommitted changes; commit first so the baseline matches a commit')
            $verdict = 'fail'
        } elseif ($Scope -ne 'all' -and $Rebaseline) {
            $failures.Add('-Rebaseline needs -Scope all')
            $verdict = 'fail'
        } else {
            $newCounts = [System.Collections.Generic.SortedDictionary[string, int]]::new([StringComparer]::Ordinal)
            foreach ($k in $baseCounts.Keys) { if (-not $Rebaseline -and -not (& $inScopeKey $k)) { $newCounts[$k] = [int]$baseCounts[$k] } }
            foreach ($k in $counts.Keys) { if (& $inScopeKey $k) { $newCounts[$k] = $counts[$k] } }
            $newTools = if ($baseline -and -not $Rebaseline) { $baseline.tools } else { $tools }
            $record = [ordered]@{
                schema_version = $script:SchemaVersion
                recorded_from  = [ordered]@{ head_sha = $git.HeadSha; note = 'Counts of statically detectable risky patterns; may only go down. See Tools/Quality/Invoke-QualityRatchet.ps1.' }
                tools          = $newTools
                counts         = $newCounts
            }
            if ($Rebaseline -and $baseline) {
                foreach ($line in $regressions) { Write-Host "REBASELINE raises: $line" -ForegroundColor Yellow }
            }
            Write-StableJson -Value $record -Path $baselineFile
            $baselineAction = if ($baseline) { if ($Rebaseline) { 'rebaselined' } else { 'lowered' } } else { 'created' }
        }
    }

    $exitCode = if ($verdict -eq 'pass') { 0 } else { 1 }
    $result = [ordered]@{
        schema_version      = $script:SchemaVersion
        generated_at        = [DateTime]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ')
        head_sha            = $git.HeadSha
        dirty               = $git.Dirty
        scope               = $Scope
        baseline_path       = (ConvertTo-PosixPath ([System.IO.Path]::GetRelativePath($root, $baselineFile)))
        baseline_action     = $baselineAction
        tools               = $tools
        files_scanned       = $files.Count
        counts              = $counts
        delta               = $delta
        improvements        = @($improvements)
        changed_files_count = $changedCount
        new_findings_total  = @($newFindings).Count
        new_findings        = @($newFindings | Select-Object -First $script:MaxNewFindings)
        notes               = @($notes)
        reasons             = @($failures)
        verdict             = $verdict
        exit_code           = $exitCode
    }
    Write-StableJson -Value $result -Path $outputFile

    # --- human summary -----------------------------------------------------------------------
    Write-Host "Quality ratchet: $($verdict.ToUpperInvariant())  ($($files.Count) files, scope=$Scope, head=$($git.HeadSha), dirty=$($git.Dirty))"
    Write-Host ('Tools: ' + (($tools.Keys | ForEach-Object { "$_ $($tools[$_])" }) -join ', '))
    foreach ($k in $counts.Keys) {
        if (-not (& $inScopeKey $k)) { continue }
        $base = if ($baseCounts.ContainsKey($k)) { [int]$baseCounts[$k] } else { 0 }
        Write-Host ('  {0,-70} {1,6}  (baseline {2})' -f $k, $counts[$k], $base)
    }
    foreach ($line in $improvements) { Write-Host "IMPROVED: $line" -ForegroundColor Green }
    if ($improvements.Count -gt 0 -and $baselineAction -eq 'none') {
        Write-Host 'Lock in the improvement: pwsh Tools/Quality/Invoke-QualityRatchet.ps1 -UpdateBaseline (clean tree, then commit the baseline).' -ForegroundColor Green
    }
    if ($newFindings.Count -gt 0) { Write-Host "$($newFindings.Count) finding(s) are in files changed against $BaseRef (see new_findings in $outputFile)." }
    foreach ($line in $failures) { Write-Host "FAIL: $line" -ForegroundColor Red }
    if ($baselineAction -ne 'none') { Write-Host "Baseline $baselineAction : $baselineFile" }
    Write-Host "Result: $outputFile"
    return $exitCode
}

try {
    $code = Invoke-QualityRatchet
} catch {
    Write-Host "FAIL: quality ratchet crashed: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host $_.ScriptStackTrace
    $code = 1
}
exit $code
