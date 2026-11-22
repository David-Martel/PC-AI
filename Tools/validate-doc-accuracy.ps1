#Requires -Version 5.1
<#
.SYNOPSIS
    Failing-capable documentation-accuracy gate: checks what the docs claim against what the repository contains.

.DESCRIPTION
    Why this exists: CLAUDE.md, README.md and CI-CD-GUIDE.md are read by agents and humans as ground truth. They
    drifted (CI table listing disabled or missing workflows, a documented /version endpoint with no route, crate and
    component lists that no longer matched the code) because the previous version of this script printed mismatches
    and always exited 0. This version exits 1 when any check fails.

    The gate is deterministic, offline and idempotent: it reads the working tree (tracked and untracked-but-not-ignored
    files when .git exists, otherwise a filesystem walk), never calls the network or gh, and writes nothing except the
    optional -OutputPath JSON report. PowerShell cmdlets and module exports are resolved by parsing source with the
    PowerShell AST, never by Import-Module or Get-Command, so results are the same on Windows and Linux.

    Checks (ids accepted by -Check):
      build-components  Component lists in CLAUDE.md equal the Build.ps1 -Component ValidateSet.
      rust-crates       Crates named in the CLAUDE.md/README.md trees equal the Native/pcai_core workspace members.
      workflows         Every .github/workflows file is in the CLAUDE.md CI table and in CI-CD-GUIDE.md, and every
                        table row or guide reference names a file that exists. Whether a workflow is enabled is
                        repository state on GitHub and cannot be read offline, so the State column is not checked.
      endpoints         Documented pcai-inference HTTP endpoints exist as routes in pcai_inference/src/http/mod.rs.
      tool-schema       Tool names agree between Config/pcai-tools.json and Deploy/rust-functiongemma/TOOLS.md, and
                        each pcai_mapping resolves to a defined cmdlet or an existing script.
      relative-links    Relative links in repository markdown resolve (case-sensitively) to a tracked path.
      module-exports    Every FunctionsToExport entry in a Modules manifest is a function defined in that module.
      retired-hosts     Names of hosts that left the fleet must not appear in the scoped docs. Reported as warn by
                        default; -RetiredHostSeverity fail makes it fail.
      training-data     Tool calls in the FunctionGemma training data name tools that exist in the schema.

    Result JSON: schema_version, repo_head, generated_at, checks[] {id,status,file,line,claim,actual,fix_hint},
    summary {pass,fail,warn,skip}. A check with no findings yields one pass entry; each finding is its own entry.

.PARAMETER RepoRoot
    Repository root. Defaults to the parent of this script's directory.

.PARAMETER OutputPath
    Optional path for the JSON report. The only file this script writes.

.PARAMETER Check
    Run only these check ids (an array, or one comma-separated string under pwsh -File). Default: all.

.PARAMETER RetiredHostSeverity
    Status given to retired-host findings: warn (default) or fail.

.PARAMETER RetiredHostScope
    Repo-relative globs (** crosses directories) scanned by retired-hosts.

.PARAMETER RetiredHostAllowlist
    Repo-relative globs excluded from retired-hosts because they record history.

.EXAMPLE
    pwsh Tools/validate-doc-accuracy.ps1 -OutputPath $env:TEMP/doc-accuracy.json

.NOTES
    Exit code 0 when no check failed (warnings allowed); 1 otherwise.
    Tests: Tests/Unit/DocAccuracy.Tests.ps1 (positive controls for every check).
#>
[CmdletBinding()]
param(
    [string]$RepoRoot,

    [string]$OutputPath,

    [string[]]$Check,

    [ValidateSet('fail', 'warn')]
    [string]$RetiredHostSeverity = 'warn',

    [string[]]$RetiredHostScope = @('CLAUDE.md', 'README.md', 'Tools/**/README.md'),

    [string[]]$RetiredHostAllowlist = @('Reports/**', 'CHANGELOG.md')
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if (-not $RepoRoot) { $RepoRoot = Split-Path -Parent $PSScriptRoot }
$RepoRoot = [System.IO.Path]::GetFullPath($RepoRoot).TrimEnd('\', '/')
if (-not (Test-Path -LiteralPath $RepoRoot -PathType Container)) { throw "RepoRoot not found: $RepoRoot" }

$script:AllChecks = @('build-components', 'rust-crates', 'workflows', 'endpoints', 'tool-schema', 'relative-links',
    'module-exports', 'retired-hosts', 'training-data')
# `pwsh -File` passes "a,b" as one string, so split commas here instead of relying on a ValidateSet array.
if ($Check) {
    $Check = @($Check | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
    $unknown = @($Check | Where-Object { $_ -notin $script:AllChecks })
    if ($unknown.Count -gt 0) { throw "Unknown check id(s): $($unknown -join ', '). Valid: $($script:AllChecks -join ', ')" }
}
$script:Results = [System.Collections.Generic.List[object]]::new()
$script:CurrentCheck = ''
$script:CheckSummary = ''
$script:LineCache = @{}
$script:FuncCache = @{}

# ---------------------------------------------------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------------------------------------------------

function Add-Finding {
    param(
        [Parameter(Mandatory)][ValidateSet('pass', 'fail', 'warn', 'skip')][string]$Status,
        [string]$File,
        [int]$Line = 0,
        [string]$Claim,
        [string]$Actual,
        [string]$FixHint
    )
    $script:Results.Add([pscustomobject][ordered]@{
            id       = $script:CurrentCheck
            status   = $Status
            file     = if ($File) { $File } else { $null }
            line     = if ($Line -gt 0) { $Line } else { $null }
            claim    = $Claim
            actual   = $Actual
            fix_hint = $FixHint
        })
}

function Get-FullPath {
    param([string]$Relative)
    return [System.IO.Path]::GetFullPath((Join-Path $RepoRoot $Relative))
}

function Read-RepoLines {
    # Returns the file's lines (CRLF or LF) or $null when the file is absent.
    param([string]$Relative)
    if (-not $script:LineCache.ContainsKey($Relative)) {
        $full = Get-FullPath $Relative
        # Plain assignments, not an if-expression: an expression would unroll an empty array to $null.
        $read = $null
        if ([System.IO.File]::Exists($full)) { $read = [System.IO.File]::ReadAllLines($full) }
        $script:LineCache[$Relative] = $read
    }
    $lines = $script:LineCache[$Relative]
    if ($null -eq $lines) { return $null }
    # The comma keeps a one-line or empty file an array; a bare return would unroll it to a string or nothing.
    return , $lines
}

function Get-RepoFileList {
    $set = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::Ordinal)
    $viaGit = $false
    if (Test-Path -LiteralPath (Join-Path $RepoRoot '.git')) {
        $git = Get-Command git -ErrorAction SilentlyContinue
        if ($git) {
            try {
                $out = & git -C $RepoRoot -c core.quotepath=off ls-files --cached --others --exclude-standard 2>$null
                if ($LASTEXITCODE -eq 0 -and $out) {
                    foreach ($p in $out) {
                        if ($p -and [System.IO.File]::Exists((Get-FullPath $p))) { [void]$set.Add($p) }
                    }
                    $viaGit = $true
                }
            } catch {
                $viaGit = $false
            }
        }
    }
    if (-not $viaGit) {
        $skipDirs = '(^|/)(\.git|target|node_modules|worktrees)(/|$)'
        foreach ($f in Get-ChildItem -LiteralPath $RepoRoot -Recurse -File -Force -ErrorAction SilentlyContinue) {
            $rel = $f.FullName.Substring($RepoRoot.Length + 1).Replace('\', '/')
            if ($rel -notmatch $skipDirs) { [void]$set.Add($rel) }
        }
    }
    return $set
}

function ConvertTo-GlobRegex {
    param([string]$Glob)
    $r = [regex]::Escape($Glob)
    $r = $r.Replace('\*\*/', '(?:.*/)?').Replace('\*\*', '.*').Replace('\*', '[^/]*')
    return '^' + $r + '$'
}

function Test-GlobMatch {
    param([string]$Path, [string[]]$Globs)
    foreach ($g in $Globs) {
        if ($Path -cmatch (ConvertTo-GlobRegex $g)) { return $true }
    }
    return $false
}

function Get-FencedLines {
    # Yields objects {Number, Text, InFence} for every line, tracking ``` and ~~~ fences.
    param([string[]]$Lines)
    $inFence = $false
    $n = 0
    foreach ($l in $Lines) {
        $n++
        if ($l -match '^\s*(```|~~~)') {
            $inFence = -not $inFence
            [pscustomobject]@{ Number = $n; Text = $l; InFence = $true; IsFenceMarker = $true }
            continue
        }
        [pscustomobject]@{ Number = $n; Text = $l; InFence = $inFence; IsFenceMarker = $false }
    }
}

function Get-DefinedFunctionNames {
    # Function names defined anywhere in a PowerShell source file, from the AST (nothing is executed).
    param([string]$Relative)
    if ($script:FuncCache.ContainsKey($Relative)) { return $script:FuncCache[$Relative] }
    $tokens = $null
    $errors = $null
    $ast = [System.Management.Automation.Language.Parser]::ParseFile((Get-FullPath $Relative), [ref]$tokens, [ref]$errors)
    $names = @($ast.FindAll({ param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] }, $true) |
            ForEach-Object { $_.Name })
    $script:FuncCache[$Relative] = $names
    return $names
}

function Get-ModuleSourceFiles {
    # Source files that can define a module's functions: Modules/<Name>/**.ps1|psm1, else Modules/<Name>.psm1.
    param([string]$ModuleName, $FileSet)
    $prefix = "Modules/$ModuleName/"
    $inDir = @($FileSet | Where-Object { $_.StartsWith($prefix, [System.StringComparison]::Ordinal) -and $_ -match '\.(ps1|psm1)$' })
    if ($inDir.Count -gt 0) { return $inDir }
    $root = "Modules/$ModuleName.psm1"
    if ($FileSet.Contains($root)) { return @($root) }
    return @()
}

# ---------------------------------------------------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------------------------------------------------

function Test-BuildComponents {
    $build = 'Build.ps1'
    $claude = Read-RepoLines 'CLAUDE.md'
    if (-not [System.IO.File]::Exists((Get-FullPath $build)) -or $null -eq $claude) {
        Add-Finding -Status skip -Claim 'Build.ps1 and CLAUDE.md present' -Actual 'one of them is missing'
        return
    }
    $tokens = $null
    $errors = $null
    $ast = [System.Management.Automation.Language.Parser]::ParseFile((Get-FullPath $build), [ref]$tokens, [ref]$errors)
    $param = $ast.ParamBlock.Parameters | Where-Object { $_.Name.VariablePath.UserPath -eq 'Component' } | Select-Object -First 1
    $actual = @()
    if ($param) {
        $vs = $param.Attributes | Where-Object { $_.TypeName.Name -eq 'ValidateSet' } | Select-Object -First 1
        if ($vs) { $actual = @($vs.PositionalArguments | ForEach-Object { [string]$_.Value }) }
    }
    if ($actual.Count -eq 0) {
        Add-Finding -Status fail -File $build -Claim 'Build.ps1 declares a -Component ValidateSet' -Actual 'none found' -FixHint 'Restore the ValidateSet or update this gate.'
        return
    }
    $found = $false
    for ($i = 0; $i -lt $claude.Count; $i++) {
        if ($claude[$i] -match '-Component`?\s+values') {
            $found = $true
            $documented = @([regex]::Matches(($claude[$i] -replace '^[^:]*:', ''), '`([A-Za-z][\w-]*)`') | ForEach-Object { $_.Groups[1].Value })
            $missing = @($actual | Where-Object { $_ -notin $documented })
            $extra = @($documented | Where-Object { $_ -notin $actual })
            if ($missing.Count -gt 0) {
                Add-Finding -Status fail -File 'CLAUDE.md' -Line ($i + 1) -Claim 'Component list is complete' `
                    -Actual ('missing from docs: ' + ($missing -join ', ')) -FixHint 'Copy the Build.ps1 -Component ValidateSet into the list.'
            }
            if ($extra.Count -gt 0) {
                Add-Finding -Status fail -File 'CLAUDE.md' -Line ($i + 1) -Claim 'Every documented component exists' `
                    -Actual ('not in Build.ps1: ' + ($extra -join ', ')) -FixHint 'Remove components the ValidateSet no longer accepts.'
            }
        }
    }
    if (-not $found) {
        Add-Finding -Status fail -File 'CLAUDE.md' -Claim 'CLAUDE.md lists the -Component values' -Actual 'no "-Component values" line' `
            -FixHint 'Add an "Available -Component values" line listing each value in backticks.'
    }
    $script:CheckSummary = "$($actual.Count) Build.ps1 components match the CLAUDE.md list"
}

function Get-WorkspaceMembers {
    $toml = Read-RepoLines 'Native/pcai_core/Cargo.toml'
    if ($null -eq $toml) { return $null }
    $text = $toml -join "`n"
    $m = [regex]::Match($text, '(?s)members\s*=\s*\[(.*?)\]')
    if (-not $m.Success) { return @() }
    return @([regex]::Matches($m.Groups[1].Value, '"([^"]+)"') | ForEach-Object { $_.Groups[1].Value })
}

function Test-RustCrates {
    $members = Get-WorkspaceMembers
    if ($null -eq $members) {
        Add-Finding -Status skip -Claim 'Native/pcai_core/Cargo.toml present' -Actual 'missing'
        return
    }
    $docsSeen = 0
    foreach ($doc in 'CLAUDE.md', 'README.md') {
        $lines = Read-RepoLines $doc
        if ($null -eq $lines) { continue }
        $all = @(Get-FencedLines $lines)
        for ($i = 0; $i -lt $all.Count; $i++) {
            $row = $all[$i]
            if (-not $row.InFence -or $row.IsFenceMarker) { continue }
            $first = [regex]::Match($row.Text, '[A-Za-z0-9_]')
            if (-not $first.Success) { continue }
            $rootCol = $first.Index
            if ($row.Text.Substring($rootCol) -cnotmatch '^(Native/)?pcai_core/') { continue }
            $docsSeen++
            $documented = [System.Collections.Generic.List[string]]::new()
            for ($j = $i + 1; $j -lt $all.Count; $j++) {
                $child = $all[$j]
                if ($child.IsFenceMarker -or -not $child.InFence) { break }
                $c = [regex]::Match($child.Text, '[A-Za-z0-9_]')
                if (-not $c.Success) { continue }
                if ($c.Index -le $rootCol) { break }
                $nm = [regex]::Match($child.Text.Substring($c.Index), '^(pcai_[A-Za-z0-9_]+)/')
                if ($nm.Success) { $documented.Add($nm.Groups[1].Value) }
            }
            $missing = @($members | Where-Object { $_ -notin $documented })
            $extra = @($documented | Where-Object { $_ -notin $members })
            if ($missing.Count -gt 0) {
                Add-Finding -Status fail -File $doc -Line $row.Number -Claim 'Tree lists every workspace crate' `
                    -Actual ('missing from tree: ' + ($missing -join ', ')) -FixHint 'Add the crates named in Native/pcai_core/Cargo.toml members.'
            }
            if ($extra.Count -gt 0) {
                Add-Finding -Status fail -File $doc -Line $row.Number -Claim 'Every crate in the tree is a workspace member' `
                    -Actual ('not workspace members: ' + ($extra -join ', ')) -FixHint 'Remove crates that Cargo.toml does not list.'
            }
            $count = [regex]::Match($row.Text, '\((\d+)[- ]crates?')
            if ($count.Success -and [int]$count.Groups[1].Value -ne $members.Count) {
                Add-Finding -Status fail -File $doc -Line $row.Number -Claim ("Tree says $($count.Groups[1].Value) crates") `
                    -Actual "Cargo.toml has $($members.Count) members" -FixHint 'Fix the number or drop it and let the list speak.'
            }
        }
    }
    if ($docsSeen -eq 0) {
        Add-Finding -Status fail -Claim 'CLAUDE.md or README.md shows the Native/pcai_core crate tree' -Actual 'no tree found' `
            -FixHint 'Restore the Native/pcai_core/ tree in the project-structure block.'
    }
    $script:CheckSummary = "$($members.Count) workspace members match the documented trees"
}

function Test-Workflows {
    param($FileSet)
    $files = @($FileSet | Where-Object { $_ -cmatch '^\.github/workflows/[^/]+\.ya?ml$' } | ForEach-Object { $_.Substring('.github/workflows/'.Length) } | Sort-Object)
    $claude = Read-RepoLines 'CLAUDE.md'
    if ($files.Count -eq 0 -or $null -eq $claude) {
        Add-Finding -Status skip -Claim 'workflow files and CLAUDE.md present' -Actual 'one of them is missing'
        return
    }
    $rows = @{}
    for ($i = 0; $i -lt $claude.Count; $i++) {
        $m = [regex]::Match($claude[$i], '^\|\s*`([\w.-]+\.ya?ml)`\s*\|')
        if ($m.Success) { $rows[$m.Groups[1].Value] = $i + 1 }
    }
    foreach ($f in $files) {
        if (-not $rows.ContainsKey($f)) {
            Add-Finding -Status fail -File 'CLAUDE.md' -Claim "CI table lists $f" -Actual 'no row' `
                -FixHint "Add a row for .github/workflows/$f with its trigger and state."
        }
    }
    foreach ($k in ($rows.Keys | Sort-Object)) {
        if ($k -notin $files) {
            Add-Finding -Status fail -File 'CLAUDE.md' -Line $rows[$k] -Claim "CI table row $k" -Actual 'no such workflow file' `
                -FixHint 'Remove the row or restore the workflow.'
        }
    }
    $guide = Read-RepoLines 'CI-CD-GUIDE.md'
    if ($null -ne $guide) {
        $refs = @{}
        for ($i = 0; $i -lt $guide.Count; $i++) {
            foreach ($m in [regex]::Matches($guide[$i], '\.github/workflows/([\w.-]+\.ya?ml)')) {
                $name = $m.Groups[1].Value
                if (-not $refs.ContainsKey($name)) { $refs[$name] = $i + 1 }
            }
        }
        foreach ($f in $files) {
            if (-not $refs.ContainsKey($f)) {
                Add-Finding -Status fail -File 'CI-CD-GUIDE.md' -Claim "Guide describes $f" -Actual 'not mentioned' `
                    -FixHint "Add a section for .github/workflows/$f."
            }
        }
        foreach ($k in ($refs.Keys | Sort-Object)) {
            if ($k -notin $files) {
                Add-Finding -Status fail -File 'CI-CD-GUIDE.md' -Line $refs[$k] -Claim "Guide references $k" -Actual 'no such workflow file' `
                    -FixHint 'Describe a workflow that exists, or remove the section.'
            }
        }
    }
    # README badges and links must point at workflows that exist (a dead badge reads as a permanently missing check).
    $readme = Read-RepoLines 'README.md'
    if ($null -ne $readme) {
        for ($i = 0; $i -lt $readme.Count; $i++) {
            foreach ($m in [regex]::Matches($readme[$i], 'workflows/([\w.-]+\.ya?ml)')) {
                if ($m.Groups[1].Value -notin $files) {
                    Add-Finding -Status fail -File 'README.md' -Line ($i + 1) -Claim "README references workflow $($m.Groups[1].Value)" `
                        -Actual 'no such workflow file' -FixHint 'Point the badge or link at a workflow that exists.'
                }
            }
        }
    }
    $script:CheckSummary = "$($files.Count) workflow files match the CLAUDE.md table and CI-CD-GUIDE.md; README workflow references exist (enabled state not checked offline)"
}

function Test-Endpoints {
    $routeFile = 'Native/pcai_core/pcai_inference/src/http/mod.rs'
    $src = Read-RepoLines $routeFile
    if ($null -eq $src) {
        Add-Finding -Status skip -Claim "$routeFile present" -Actual 'missing'
        return
    }
    $routes = [System.Collections.Generic.List[object]]::new()
    foreach ($m in [regex]::Matches(($src -join "`n"), '\.route\(\s*"(/[^"]*)"\s*,\s*(get|post|put|delete|patch)')) {
        $routes.Add([pscustomobject]@{ Verb = $m.Groups[2].Value.ToUpperInvariant(); Path = $m.Groups[1].Value })
    }
    $other = 'functiongemma|media|router|ollama|vllm|lmstudio|janus'
    $claims = 0
    foreach ($doc in 'CLAUDE.md', 'README.md') {
        $lines = Read-RepoLines $doc
        if ($null -eq $lines) { continue }
        for ($i = 0; $i -lt $lines.Count; $i++) {
            $text = $lines[$i]
            if ($text -imatch $other) { continue }
            foreach ($m in [regex]::Matches($text, '\b(GET|POST|PUT|DELETE|PATCH)\s+https?://[^\s/`]+(/[^\s`)]*)')) {
                $claims++
                $verb = $m.Groups[1].Value
                $path = $m.Groups[2].Value
                $hit = @($routes | Where-Object { $_.Path -ceq $path })
                if ($hit.Count -eq 0) {
                    Add-Finding -Status fail -File $doc -Line ($i + 1) -Claim "$verb $path" -Actual 'no such route' `
                        -FixHint "Routes in $routeFile : $(($routes | ForEach-Object { $_.Verb + ' ' + $_.Path }) -join ', ')"
                } elseif (-not ($hit | Where-Object { $_.Verb -eq $verb })) {
                    Add-Finding -Status fail -File $doc -Line ($i + 1) -Claim "$verb $path" -Actual ('route accepts ' + ($hit[0].Verb)) `
                        -FixHint 'Fix the HTTP verb.'
                }
            }
            foreach ($m in [regex]::Matches($text, '`(/[A-Za-z0-9_/{}\-]+)`\s+endpoint')) {
                $claims++
                $path = $m.Groups[1].Value
                if (-not ($routes | Where-Object { $_.Path -ceq $path })) {
                    Add-Finding -Status fail -File $doc -Line ($i + 1) -Claim "$path endpoint" -Actual 'no such route' `
                        -FixHint "Routes in $routeFile : $(($routes | ForEach-Object { $_.Verb + ' ' + $_.Path }) -join ', ')"
                }
            }
        }
    }
    $script:CheckSummary = "$claims documented endpoint claims resolve to $($routes.Count) routes"
}

function Test-ToolSchema {
    param($FileSet)
    $schemaRel = 'Config/pcai-tools.json'
    $docRel = 'Deploy/rust-functiongemma/TOOLS.md'
    $schemaLines = Read-RepoLines $schemaRel
    $docLines = Read-RepoLines $docRel
    if ($null -eq $schemaLines -or $null -eq $docLines) {
        Add-Finding -Status skip -Claim "$schemaRel and $docRel present" -Actual 'one of them is missing'
        return
    }
    $schema = ($schemaLines -join "`n") | ConvertFrom-Json
    $tools = @($schema.tools)
    $schemaNames = @($tools | ForEach-Object { $_.function.name })
    $docNames = @($docLines | ForEach-Object { if ($_ -match '^###\s+(\S+)\s*$') { $Matches[1] } })
    foreach ($n in $schemaNames) {
        if ($n -notin $docNames) {
            Add-Finding -Status fail -File $docRel -Claim "TOOLS.md documents $n" -Actual 'no heading' `
                -FixHint 'Regenerate with Tools/generate-functiongemma-tool-docs.ps1.'
        }
    }
    foreach ($n in ($docNames | Select-Object -Unique)) {
        if ($n -notin $schemaNames) {
            Add-Finding -Status fail -File $docRel -Claim "TOOLS.md documents $n" -Actual "not in $schemaRel" `
                -FixHint 'Regenerate with Tools/generate-functiongemma-tool-docs.ps1.'
        }
    }
    $external = 0
    foreach ($tool in $tools) {
        $name = $tool.function.name
        $mapProp = $tool.PSObject.Properties['pcai_mapping']
        if (-not $mapProp -or $null -eq $mapProp.Value) { continue }
        $map = $mapProp.Value
        $scriptProp = $map.PSObject.Properties['script']
        $cmdletProp = $map.PSObject.Properties['cmdlet']
        $moduleProp = $map.PSObject.Properties['module']
        if ($scriptProp -and $scriptProp.Value) {
            $rel = ([string]$scriptProp.Value).Replace('\', '/')
            if (-not $FileSet.Contains($rel)) {
                Add-Finding -Status fail -File $schemaRel -Claim "$name maps to script $rel" -Actual 'script file does not exist' `
                    -FixHint 'Fix the pcai_mapping.script path or add the script.'
            }
        } elseif ($cmdletProp -and $cmdletProp.Value) {
            if (-not $moduleProp -or -not $moduleProp.Value) { $external++; continue }
            $cmdlet = [string]$cmdletProp.Value
            $module = [string]$moduleProp.Value
            $sources = @(Get-ModuleSourceFiles -ModuleName $module -FileSet $FileSet)
            if ($sources.Count -eq 0) {
                Add-Finding -Status fail -File $schemaRel -Claim "$name maps to $module\$cmdlet" -Actual "no source for module $module" `
                    -FixHint 'Fix pcai_mapping.module.'
                continue
            }
            $defined = $false
            foreach ($s in $sources) {
                if ((Get-DefinedFunctionNames $s) -contains $cmdlet) { $defined = $true; break }
            }
            if (-not $defined) {
                Add-Finding -Status fail -File $schemaRel -Claim "$name maps to $module\$cmdlet" -Actual 'function is not defined in the module sources' `
                    -FixHint 'Fix pcai_mapping.cmdlet or define the function.'
            }
        } else {
            Add-Finding -Status fail -File $schemaRel -Claim "$name has a pcai_mapping" -Actual 'neither cmdlet nor script' -FixHint 'Add one.'
        }
    }
    $script:CheckSummary = "$($schemaNames.Count) schema tools match TOOLS.md and resolve to cmdlets or scripts ($external external commands not verified)"
}

function Test-RelativeLinks {
    param($FileSet)
    $dirs = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::Ordinal)
    $lower = @{}
    foreach ($f in $FileSet) {
        $lower[$f.ToLowerInvariant()] = $f
        $idx = $f.LastIndexOf('/')
        while ($idx -gt 0) {
            $d = $f.Substring(0, $idx)
            if (-not $dirs.Add($d)) { break }
            $lower[$d.ToLowerInvariant()] = $d
            $idx = $d.LastIndexOf('/')
        }
    }
    $excluded = '^(Reports|worktrees|target|node_modules|\.pcai)/|(^|/)(target|node_modules|vendor)/'
    $vendoredException = 'Native/pcai_core/third_party/ggml-sys/UPDATING.md'
    $docs = @($FileSet | Where-Object { $_ -cmatch '\.md$' } | Sort-Object)
    $checked = 0
    foreach ($doc in $docs) {
        if ($doc -cmatch $excluded) { continue }
        if ($doc -cmatch '(^|/)third_party/' -and $doc -cne $vendoredException) { continue }
        $lines = Read-RepoLines $doc
        $baseDir = if ($doc.Contains('/')) { $doc.Substring(0, $doc.LastIndexOf('/')) } else { '' }
        foreach ($row in (Get-FencedLines $lines)) {
            if ($row.InFence) { continue }
            $text = [regex]::Replace($row.Text, '`[^`]*`', '')
            $targets = @([regex]::Matches($text, '\]\(\s*(<[^>]+>|[^)\s]+)') | ForEach-Object { $_.Groups[1].Value })
            $refDef = [regex]::Match($text, '^\s*\[[^\]]+\]:\s+(<[^>]+>|\S+)')
            if ($refDef.Success) { $targets += $refDef.Groups[1].Value }
            foreach ($raw in $targets) {
                $t = $raw.Trim('<', '>')
                if ($t -match '^[A-Za-z][A-Za-z0-9+.-]*:' -or $t -match '^//' -or $t.StartsWith('#')) { continue }
                $t = ($t -replace '[#?].*$', '')
                if (-not $t) { continue }
                try { $t = [System.Uri]::UnescapeDataString($t) } catch { Write-Verbose "unescape failed for $t" }
                $t = $t.Replace('\', '/')
                $checked++
                $segments = [System.Collections.Generic.List[string]]::new()
                if (-not $t.StartsWith('/') -and $baseDir) { foreach ($s in $baseDir.Split('/')) { $segments.Add($s) } }
                $escaped = $false
                foreach ($s in $t.Split('/')) {
                    if ($s -eq '' -or $s -eq '.') { continue }
                    if ($s -eq '..') {
                        if ($segments.Count -eq 0) { $escaped = $true; break }
                        $segments.RemoveAt($segments.Count - 1)
                    } else { $segments.Add($s) }
                }
                $resolved = $segments -join '/'
                if ($escaped) {
                    Add-Finding -Status fail -File $doc -Line $row.Number -Claim $raw -Actual 'link leaves the repository' -FixHint 'Fix the relative path.'
                } elseif ($resolved -and -not ($FileSet.Contains($resolved) -or $dirs.Contains($resolved))) {
                    $hint = 'Fix the path or restore the target.'
                    if ($lower.ContainsKey($resolved.ToLowerInvariant())) { $hint = 'Case mismatch; actual path is ' + $lower[$resolved.ToLowerInvariant()] }
                    Add-Finding -Status fail -File $doc -Line $row.Number -Claim $raw -Actual "no such path: $resolved" -FixHint $hint
                }
            }
        }
    }
    $script:CheckSummary = "$checked relative links in $($docs.Count) markdown files resolve"
}

function Test-ModuleExports {
    param($FileSet)
    $manifests = @($FileSet | Where-Object { $_ -cmatch '^Modules/.*\.psd1$' } | Sort-Object)
    if ($manifests.Count -eq 0) {
        Add-Finding -Status skip -Claim 'Modules/**/*.psd1 present' -Actual 'none'
        return
    }
    $exports = 0
    foreach ($manifest in $manifests) {
        $tokens = $null
        $errors = $null
        $ast = [System.Management.Automation.Language.Parser]::ParseFile((Get-FullPath $manifest), [ref]$tokens, [ref]$errors)
        $ht = $ast.Find({ param($n) $n -is [System.Management.Automation.Language.HashtableAst] }, $true)
        if (-not $ht) { continue }
        $exported = @()
        $rootModule = ''
        foreach ($kv in $ht.KeyValuePairs) {
            $key = $kv.Item1
            if ($key -isnot [System.Management.Automation.Language.StringConstantExpressionAst]) { continue }
            if ($key.Value -eq 'FunctionsToExport') {
                $exported = @($kv.Item2.FindAll({ param($n) $n -is [System.Management.Automation.Language.StringConstantExpressionAst] }, $true) |
                        ForEach-Object { $_.Value })
            } elseif ($key.Value -eq 'RootModule') {
                $v = $kv.Item2.Find({ param($n) $n -is [System.Management.Automation.Language.StringConstantExpressionAst] }, $true)
                if ($v) { $rootModule = $v.Value }
            }
        }
        $exported = @($exported | Where-Object { $_ -and $_ -ne '*' })
        if ($exported.Count -eq 0) { continue }
        $dir = $manifest.Substring(0, $manifest.LastIndexOf('/'))
        if ($dir -eq 'Modules') {
            $leaf = if ($rootModule) { $rootModule } else { [System.IO.Path]::GetFileNameWithoutExtension($manifest) + '.psm1' }
            $sources = @("Modules/$leaf" | Where-Object { $FileSet.Contains($_) })
        } else {
            $sources = @($FileSet | Where-Object { $_.StartsWith($dir + '/', [System.StringComparison]::Ordinal) -and $_ -cmatch '\.(ps1|psm1)$' })
        }
        $defined = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
        foreach ($s in $sources) { foreach ($n in (Get-DefinedFunctionNames $s)) { [void]$defined.Add($n) } }
        foreach ($name in $exported) {
            $exports++
            if (-not $defined.Contains($name)) {
                Add-Finding -Status fail -File $manifest -Claim "FunctionsToExport lists $name" -Actual 'no function with that name is defined in the module sources' `
                    -FixHint 'Define the function or remove it from FunctionsToExport.'
            }
        }
    }
    $script:CheckSummary = "$exports exported functions across $($manifests.Count) manifests are defined"
}

function Test-RetiredHosts {
    param($FileSet)
    $patterns = @(
        @{ Name = 'asuspro13'; Regex = '\basuspro13\b' },
        @{ Name = 'vigil1'; Regex = '\bvigil1\b' },
        @{ Name = 'spark-NNNN'; Regex = '\bspark-\d+\b' },
        @{ Name = 'DGX Spark'; Regex = '\bDGX\s+Sparks?\b' },
        @{ Name = 'millylaptop1'; Regex = '\bmillylaptop1\b' },
        @{ Name = 'milly'; Regex = '\bmilly\b' },
        @{ Name = 'localhost:8400'; Regex = 'localhost:8400\b' }
    )
    $scoped = @($FileSet | Where-Object { (Test-GlobMatch $_ $RetiredHostScope) -and -not (Test-GlobMatch $_ $RetiredHostAllowlist) } | Sort-Object)
    $hits = 0
    foreach ($file in $scoped) {
        $lines = Read-RepoLines $file
        for ($i = 0; $i -lt $lines.Count; $i++) {
            foreach ($p in $patterns) {
                if ($lines[$i] -imatch $p.Regex) {
                    $hits++
                    Add-Finding -Status $RetiredHostSeverity -File $file -Line ($i + 1) -Claim 'Docs name only hosts that are still in the fleet' `
                        -Actual "mentions retired host: $($p.Name)" -FixHint 'Remove or rewrite the mention; keep history only in Reports/ or CHANGELOG.md.'
                }
            }
        }
    }
    $script:CheckSummary = "$($scoped.Count) scoped files name no retired host"
}

function Test-TrainingData {
    $schemaLines = Read-RepoLines 'Config/pcai-tools.json'
    $dataRel = 'Deploy/rust-functiongemma-train/data/training_data.jsonl'
    $dataPath = Get-FullPath $dataRel
    if ($null -eq $schemaLines -or -not [System.IO.File]::Exists($dataPath)) {
        Add-Finding -Status skip -Claim 'schema and training_data.jsonl present' -Actual 'one of them is missing'
        return
    }
    $names = @((($schemaLines -join "`n") | ConvertFrom-Json).tools | ForEach-Object { $_.function.name })
    $n = 0
    $total = 0
    foreach ($line in [System.IO.File]::ReadLines($dataPath)) {
        $n++
        if (-not $line.Trim()) { continue }
        $total++
        $item = $line | ConvertFrom-Json
        $msgs = @($item.messages)
        if ($msgs.Count -eq 0) { continue }
        $last = $msgs[-1]
        $callsProp = $last.PSObject.Properties['tool_calls']
        if ($callsProp -and $callsProp.Value) {
            $tool = $callsProp.Value[0].function.name
            if ($tool -notin $names) {
                Add-Finding -Status fail -File $dataRel -Line $n -Claim 'training example calls a schema tool' -Actual "unknown tool $tool" `
                    -FixHint 'Regenerate training data or fix the schema.'
            }
        }
    }
    $script:CheckSummary = "$total training examples call only schema tools"
}

# ---------------------------------------------------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------------------------------------------------

$selected = if ($Check) { @($script:AllChecks | Where-Object { $_ -in $Check }) } else { $script:AllChecks }
$fileSet = Get-RepoFileList

foreach ($id in $selected) {
    $script:CurrentCheck = $id
    $script:CheckSummary = ''
    $before = $script:Results.Count
    switch ($id) {
        'build-components' { Test-BuildComponents }
        'rust-crates' { Test-RustCrates }
        'workflows' { Test-Workflows -FileSet $fileSet }
        'endpoints' { Test-Endpoints }
        'tool-schema' { Test-ToolSchema -FileSet $fileSet }
        'relative-links' { Test-RelativeLinks -FileSet $fileSet }
        'module-exports' { Test-ModuleExports -FileSet $fileSet }
        'retired-hosts' { Test-RetiredHosts -FileSet $fileSet }
        'training-data' { Test-TrainingData }
    }
    if ($script:Results.Count -eq $before) {
        Add-Finding -Status pass -Claim $script:CheckSummary -Actual 'documentation matches the repository'
    }
}

# Deterministic order: canonical check order, then file, then line.
$order = @{}
for ($i = 0; $i -lt $script:AllChecks.Count; $i++) { $order[$script:AllChecks[$i]] = $i }
$sorted = @($script:Results | Sort-Object -Property @{ Expression = { $order[$_.id] } }, @{ Expression = { [string]$_.file } }, @{ Expression = { [int]$_.line } },
    @{ Expression = { [string]$_.claim } }, @{ Expression = { [string]$_.actual } })

$summary = [ordered]@{ pass = 0; fail = 0; warn = 0; skip = 0 }
foreach ($r in $sorted) { $summary[$r.status]++ }

$repoHead = $null
if (Test-Path -LiteralPath (Join-Path $RepoRoot '.git')) {
    try {
        $head = & git -C $RepoRoot rev-parse HEAD 2>$null
        if ($LASTEXITCODE -eq 0 -and $head) { $repoHead = ([string]$head).Trim() }
    } catch {
        $repoHead = $null
    }
}

$report = [ordered]@{
    schema_version = 1
    repo_head      = $repoHead
    generated_at   = [DateTime]::UtcNow.ToString('o')
    checks         = $sorted
    summary        = $summary
}

if ($OutputPath) {
    $json = ($report | ConvertTo-Json -Depth 6) -replace "`r`n", "`n"
    $dir = Split-Path -Parent ([System.IO.Path]::GetFullPath($OutputPath))
    if ($dir -and -not (Test-Path -LiteralPath $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
    [System.IO.File]::WriteAllText([System.IO.Path]::GetFullPath($OutputPath), $json + "`n", [System.Text.UTF8Encoding]::new($false))
}

foreach ($r in $sorted) {
    $where = if ($r.file) { if ($r.line) { "$($r.file):$($r.line)" } else { $r.file } } else { '' }
    $color = switch ($r.status) { 'pass' { 'Green' } 'fail' { 'Red' } 'warn' { 'Yellow' } default { 'Gray' } }
    $text = if ($r.status -eq 'pass') { "[pass] $($r.id): $($r.claim)" } else { "[$($r.status)] $($r.id) $where : $($r.claim) -> $($r.actual)" }
    Write-Host $text -ForegroundColor $color
}
Write-Host ("Summary: pass={0} fail={1} warn={2} skip={3}" -f $summary.pass, $summary.fail, $summary.warn, $summary.skip)

if ($summary.fail -gt 0) { exit 1 }
exit 0
