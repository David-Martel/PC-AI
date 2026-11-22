#Requires -Version 7.0
<#
.SYNOPSIS
    Static checks for the PC-AI quality ratchet that PSScriptAnalyzer and ast-grep cannot express.

.DESCRIPTION
    ast-grep cannot parse PowerShell (rules/powershell/*.yml declared `language: bash` and never
    matched a .ps1 file), and the repo's PSScriptAnalyzer settings hide the failure-swallowing
    patterns that matter most for a Windows system-repair tool. This module walks the real
    PowerShell AST and, for C#, runs a comment/string-aware scanner. Each function returns plain
    finding objects: Rule, Line (1-based), Message.

    PowerShell rules (Get-PcaiPsAstFinding):
      empty-catch                       catch block with no statements.
      silent-continue-state-changing    -ErrorAction SilentlyContinue (or -EA / abbreviations) on a
                                        Set-/New-/Remove-/Register-/Mount-/Install-/Enable-/
                                        Disable-/Start-/Stop-/Restart- command.
      native-exe-exit-code-unchecked    native executable invoked with no $LASTEXITCODE / $? read
                                        later in the same script block. HEURISTIC, see below.
      shadowed-builtin-command          function named like a built-in cmdlet or function.
      invoke-expression                 Invoke-Expression / iex (replaces rules/powershell/
                                        no-invoke-expression.yml, which never matched).

    Heuristic limits of native-exe-exit-code-unchecked (it counts, it does not prove a bug):
      * A command is "native" when its static name ends in .exe or is in $script:NativeCommandNames,
        or it is `& $var` where the variable name is such a name or contains "exe". Other `& $var`
        calls, aliases and wrapper functions (Invoke-NativeCommandCapture) are not recognised.
      * "Checked" means any read of $LASTEXITCODE or $? after the end of the call's pipeline inside
        the same enclosing ScriptBlockAst (nested script blocks included). It does not prove the
        read belongs to that call: a later call's check masks an earlier unchecked one.
      * A file that assigns $PSNativeCommandUseErrorActionPreference is treated as checked, because
        PowerShell 7.4+ then throws on a non-zero exit code.
      * A final-statement call whose exit code the caller inspects is still reported.

    C# rules (Get-PcaiCSharpCatchFinding), regex over source with comments and string literals
    blanked out (no Roslyn):
      catch-all-empty                   `catch { }` or `catch (Exception ...) { }` with an empty body.
      catch-all-returns-default         catch-all whose body has no `throw` and returns null, false,
                                        0, "", default, an empty collection or a bare `new T()`.
    A catch clause with a `when (...)` filter is not a catch-all. Raw string literals with more than
    three quotes and interpolated strings that embed quotes can desynchronise the scanner for the
    rest of that file; the effect is an under-count in that file, never a crash.

    Scope classification (Get-PcaiQualityFileScope, Get-PcaiRustTestStartLine) decides prod vs test.
#>

Set-StrictMode -Version Latest

# Bump when a check's meaning changes. The ratchet records this in the baseline and fails closed on a
# MAJOR.MINOR mismatch, so a changed check cannot silently move the counts.
$script:ChecksVersion = '1.0.0'

$script:StateChangingVerb = [regex]::new('^(Set|New|Remove|Register|Mount|Install|Enable|Disable|Start|Stop|Restart)-', 'IgnoreCase')

# -ErrorAction, its -EA alias and the unambiguous prefixes (-ErrorA ... -ErrorActio). -ErrorV is
# -ErrorVariable and is deliberately not matched.
$script:ErrorActionParameter = [regex]::new('^(ea|errora(c(t(i(on?)?)?)?)?)$', 'IgnoreCase')

$script:NativeCommandNames = [System.Collections.Generic.HashSet[string]]::new(
    [string[]]@(
        'git', 'gh', 'cargo', 'rustc', 'rustup', 'rustfmt', 'dotnet', 'node', 'npm', 'npx', 'pnpm', 'yarn',
        'uv', 'python', 'python3', 'pip', 'docker', 'wsl', 'robocopy', 'xcopy', 'msbuild', 'cmake', 'ninja',
        'nvidia-smi', 'nvcc', 'sg', 'ast-grep', 'pwsh', 'powershell', 'reg', 'netsh', 'sc', 'schtasks',
        'icacls', 'takeown', 'bcdedit', 'diskpart', 'chkdsk', 'dism', 'sfc', 'wmic', 'taskkill', 'tasklist',
        'ipconfig', 'ping', 'nslookup', 'ssh', 'scp', 'curl', 'tar', '7z', 'msiexec', 'wevtutil', 'vssadmin',
        'mountvol', 'fsutil', 'defrag', 'powercfg', 'rg', 'fd', 'sccache', 'lefthook', 'bw', 'gpg', 'sudo'
    ),
    [StringComparer]::OrdinalIgnoreCase)

$script:BuiltinCommands = $null

function Get-PcaiBuiltinCommandSet {
    if ($null -eq $script:BuiltinCommands) {
        $set = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
        $path = Join-Path $PSScriptRoot 'Data/PowerShellBuiltinCommands.txt'
        if (-not (Test-Path -LiteralPath $path)) {
            throw "Built-in command list is missing: $path (regenerate with Update-PowerShellBuiltinCommandList.ps1)"
        }
        foreach ($line in [System.IO.File]::ReadAllLines($path)) {
            $trimmed = $line.Trim()
            if ($trimmed.Length -gt 0 -and -not $trimmed.StartsWith('#')) { [void]$set.Add($trimmed) }
        }
        if ($set.Count -lt 100) { throw "Built-in command list looks truncated ($($set.Count) entries): $path" }
        $script:BuiltinCommands = $set
    }
    return , $script:BuiltinCommands
}

function Get-PcaiChecksVersion { return $script:ChecksVersion }

function Get-PcaiQualityFileScope {
    <#
    .SYNOPSIS
        Classifies a repo-relative path as 'test' or 'prod' (case-insensitive, / or \ separators).
    .DESCRIPTION
        Test: any path segment equal to tests, test, benches or examples, or ending in ".Tests";
        file names ending in .Tests.ps1, Tests.cs, Test.cs or _test.rs / _tests.rs. Rust code after a
        `#[cfg(test)] mod` is test code too, but that needs file contents and is handled by
        Get-PcaiRustTestStartLine.
    #>
    [CmdletBinding()]
    [OutputType([string])]
    param([Parameter(Mandatory)][string]$RelativePath)

    $normalized = $RelativePath.Replace('\', '/')
    $segments = $normalized.Split('/')
    for ($i = 0; $i -lt $segments.Length - 1; $i++) {
        $segment = $segments[$i]
        if ($segment -ieq 'tests' -or $segment -ieq 'test' -or $segment -ieq 'benches' -or $segment -ieq 'examples' -or $segment -like '*.Tests') {
            return 'test'
        }
    }
    $leaf = $segments[$segments.Length - 1]
    if ($leaf -match '\.Tests\.ps1$' -or $leaf -match 'Tests?\.cs$' -or $leaf -match '_tests?\.rs$') { return 'test' }
    return 'prod'
}

function Get-PcaiRustTestStartLine {
    <#
    .SYNOPSIS
        Zero-based line of the first `#[cfg(test)]` that is followed (within two lines) by a `mod`, or
        [int]::MaxValue when the file has none.
    .DESCRIPTION
        Heuristic: Rust test modules conventionally sit at the end of the file, so every finding at or
        after that line is classified as test code. A `#[cfg(test)]` on a `use` or `fn` does not start
        a test region. Lines are zero-based to match ast-grep output.
    #>
    [CmdletBinding()]
    [OutputType([int])]
    param([Parameter(Mandatory)][string]$Path)

    try {
        $lines = [System.IO.File]::ReadAllLines($Path)
    } catch {
        throw "Cannot read Rust source for scope classification: $Path ($($_.Exception.Message))"
    }
    for ($i = 0; $i -lt $lines.Length; $i++) {
        if ($lines[$i] -match '^\s*#\[cfg\(test\)\]') {
            $last = [Math]::Min($i + 2, $lines.Length - 1)
            for ($j = $i + 1; $j -le $last; $j++) {
                if ($lines[$j] -match '^\s*(pub(\([a-z]+\))?\s+)?mod\s+\w+') { return $i }
            }
        }
    }
    return [int]::MaxValue
}

function Get-PcaiPsVariableLeafName {
    param([System.Management.Automation.Language.VariableExpressionAst]$Variable)
    $user = $Variable.VariablePath.UserPath
    $colon = $user.LastIndexOf(':')
    if ($colon -ge 0) { return $user.Substring($colon + 1) }
    return $user
}

function Test-PcaiPsNativeCommand {
    param([System.Management.Automation.Language.CommandAst]$Command)

    $name = $Command.GetCommandName()
    if ($name) {
        $leaf = $name.Replace('\', '/')
        $leaf = $leaf.Substring($leaf.LastIndexOf('/') + 1)
        return ($leaf -match '\.exe$') -or $script:NativeCommandNames.Contains($leaf)
    }
    if ($Command.InvocationOperator -eq [System.Management.Automation.Language.TokenKind]::Ampersand -and $Command.CommandElements.Count -gt 0) {
        $first = $Command.CommandElements[0]
        if ($first -is [System.Management.Automation.Language.VariableExpressionAst]) {
            $variable = (Get-PcaiPsVariableLeafName -Variable $first)
            return $script:NativeCommandNames.Contains($variable) -or ($variable -match 'exe')
        }
    }
    return $false
}

function Get-PcaiPsAstFinding {
    <#
    .SYNOPSIS
        Runs the PowerShell AST checks on one file and returns finding objects (Rule, Line, Message).
    .PARAMETER Path
        .ps1 or .psm1 file. A file that does not parse completely is still scanned on the partial AST.
    #>
    [CmdletBinding()]
    [OutputType([pscustomobject])]
    param([Parameter(Mandatory)][string]$Path)

    $tokens = $null
    $errors = $null
    $ast = [System.Management.Automation.Language.Parser]::ParseFile($Path, [ref]$tokens, [ref]$errors)
    if ($null -eq $ast) { throw "Parser returned no AST for $Path" }

    $builtins = Get-PcaiBuiltinCommandSet
    $found = [System.Collections.Generic.List[object]]::new()
    $add = {
        param($rule, $extent, $message)
        $found.Add([pscustomobject]@{ Rule = $rule; Line = [int]$extent.StartLineNumber; Message = $message })
    }

    $interesting = [Func[System.Management.Automation.Language.Ast, bool]] {
        param($node)
        $node -is [System.Management.Automation.Language.CatchClauseAst] -or
        $node -is [System.Management.Automation.Language.CommandAst] -or
        $node -is [System.Management.Automation.Language.FunctionDefinitionAst]
    }
    $nodes = $ast.FindAll($interesting, $true)

    # $PSNativeCommandUseErrorActionPreference makes PowerShell 7.4+ throw on a non-zero exit code.
    $nativeErrorPreference = $false
    $exitCodeReads = $null
    $nativeCalls = [System.Collections.Generic.List[object]]::new()

    foreach ($node in $nodes) {
        if ($node -is [System.Management.Automation.Language.CatchClauseAst]) {
            if ($node.Body.Statements.Count -eq 0) {
                & $add 'empty-catch' $node.Extent 'catch block swallows the error without any statement'
            }
            continue
        }
        if ($node -is [System.Management.Automation.Language.FunctionDefinitionAst]) {
            if ($builtins.Contains($node.Name)) {
                & $add 'shadowed-builtin-command' $node.Extent "function '$($node.Name)' shadows a built-in command"
            }
            continue
        }

        $command = [System.Management.Automation.Language.CommandAst]$node
        $name = $command.GetCommandName()
        if ($name) {
            if ($name -ieq 'Invoke-Expression' -or $name -ieq 'iex') {
                & $add 'invoke-expression' $command.Extent 'Invoke-Expression evaluates a string as code'
            }
            if ($script:StateChangingVerb.IsMatch($name)) {
                for ($i = 1; $i -lt $command.CommandElements.Count; $i++) {
                    $element = $command.CommandElements[$i]
                    if ($element -isnot [System.Management.Automation.Language.CommandParameterAst]) { continue }
                    if (-not $script:ErrorActionParameter.IsMatch($element.ParameterName)) { continue }
                    $valueAst = $element.Argument
                    if ($null -eq $valueAst -and ($i + 1) -lt $command.CommandElements.Count) { $valueAst = $command.CommandElements[$i + 1] }
                    if ($null -ne $valueAst -and $valueAst.Extent.Text.Trim('''', '"') -ieq 'SilentlyContinue') {
                        & $add 'silent-continue-state-changing' $command.Extent "$name suppresses its own errors with -ErrorAction SilentlyContinue"
                        break
                    }
                }
            }
        }
        if (Test-PcaiPsNativeCommand -Command $command) { $nativeCalls.Add($command) }
    }

    if ($nativeCalls.Count -gt 0) {
        $variables = $ast.FindAll({ param($n) $n -is [System.Management.Automation.Language.VariableExpressionAst] }, $true)
        $exitCodeReads = [System.Collections.Generic.List[object]]::new()
        foreach ($variable in $variables) {
            $leaf = Get-PcaiPsVariableLeafName -Variable $variable
            if ($leaf -ieq 'PSNativeCommandUseErrorActionPreference') { $nativeErrorPreference = $true }
            elseif ($leaf -ieq 'LASTEXITCODE' -or $leaf -eq '?') { $exitCodeReads.Add($variable) }
        }
        if (-not $nativeErrorPreference) {
            foreach ($call in $nativeCalls) {
                $scope = $call.Parent
                while ($null -ne $scope -and $scope -isnot [System.Management.Automation.Language.ScriptBlockAst]) { $scope = $scope.Parent }
                $pipeline = $call.Parent
                while ($null -ne $pipeline -and $pipeline -isnot [System.Management.Automation.Language.PipelineAst]) { $pipeline = $pipeline.Parent }
                $afterOffset = if ($null -ne $pipeline) { $pipeline.Extent.EndOffset } else { $call.Extent.EndOffset }
                $checked = $false
                foreach ($read in $exitCodeReads) {
                    if ($read.Extent.StartOffset -lt $afterOffset) { continue }
                    if ($null -ne $scope -and ($read.Extent.StartOffset -lt $scope.Extent.StartOffset -or $read.Extent.EndOffset -gt $scope.Extent.EndOffset)) { continue }
                    $checked = $true
                    break
                }
                if (-not $checked) {
                    $label = if ($call.GetCommandName()) { $call.GetCommandName() } else { $call.CommandElements[0].Extent.Text }
                    & $add 'native-exe-exit-code-unchecked' $call.Extent "native command '$label' has no later `$LASTEXITCODE or `$? check in its script block"
                }
            }
        }
    }

    return $found.ToArray()
}

function ConvertTo-PcaiBlankedCSharp {
    # Replaces comments and string/char literal contents with spaces, keeping offsets and newlines.
    param([string]$Text)
    $chars = $Text.ToCharArray()
    $n = $chars.Length
    $i = 0
    while ($i -lt $n) {
        $c = $chars[$i]
        $next = if ($i + 1 -lt $n) { $chars[$i + 1] } else { [char]0 }
        if ($c -eq '/' -and $next -eq '/') {
            while ($i -lt $n -and $chars[$i] -ne "`n") { $chars[$i] = ' '; $i++ }
        } elseif ($c -eq '/' -and $next -eq '*') {
            $chars[$i] = ' '; $chars[$i + 1] = ' '; $i += 2
            while ($i -lt $n -and -not ($chars[$i] -eq '*' -and $i + 1 -lt $n -and $chars[$i + 1] -eq '/')) {
                if ($chars[$i] -ne "`n" -and $chars[$i] -ne "`r") { $chars[$i] = ' ' }
                $i++
            }
            if ($i -lt $n) { $chars[$i] = ' '; if ($i + 1 -lt $n) { $chars[$i + 1] = ' ' }; $i += 2 }
        } elseif ($c -eq '"' -and $next -eq '"' -and $i + 2 -lt $n -and $chars[$i + 2] -eq '"') {
            # raw string literal: blank through the next triple quote
            $i += 3
            while ($i -lt $n -and -not ($chars[$i] -eq '"' -and $i + 2 -lt $n -and $chars[$i + 1] -eq '"' -and $chars[$i + 2] -eq '"')) {
                if ($chars[$i] -ne "`n" -and $chars[$i] -ne "`r") { $chars[$i] = ' ' }
                $i++
            }
            $i += 3
        } elseif ($c -eq '"') {
            $verbatim = $i -gt 0 -and ($chars[$i - 1] -eq '@' -or ($i -gt 1 -and $chars[$i - 1] -eq '$' -and $chars[$i - 2] -eq '@') -or ($i -gt 1 -and $chars[$i - 1] -eq '@' -and $chars[$i - 2] -eq '$'))
            $i++
            while ($i -lt $n) {
                if ($verbatim) {
                    if ($chars[$i] -eq '"') {
                        if ($i + 1 -lt $n -and $chars[$i + 1] -eq '"') { $chars[$i] = ' '; $chars[$i + 1] = ' '; $i += 2; continue }
                        break
                    }
                } else {
                    if ($chars[$i] -eq '\') { $chars[$i] = ' '; if ($i + 1 -lt $n -and $chars[$i + 1] -ne "`n") { $chars[$i + 1] = ' ' }; $i += 2; continue }
                    if ($chars[$i] -eq '"' -or $chars[$i] -eq "`n") { break }
                }
                if ($chars[$i] -ne "`n" -and $chars[$i] -ne "`r") { $chars[$i] = ' ' }
                $i++
            }
            $i++
        } elseif ($c -eq "'") {
            # char literal: 'a', '\n', 'A'; anything longer is not a char literal (e.g. a generic tick), leave it
            $end = -1
            for ($k = $i + 1; $k -lt [Math]::Min($i + 10, $n); $k++) {
                if ($chars[$k] -eq '\') { $k++; continue }
                if ($chars[$k] -eq "'") { $end = $k; break }
                if ($chars[$k] -eq "`n") { break }
            }
            if ($end -gt 0) { for ($k = $i + 1; $k -lt $end; $k++) { $chars[$k] = ' ' }; $i = $end + 1 } else { $i++ }
        } else {
            $i++
        }
    }
    return [string]::new($chars)
}

function Get-PcaiCSharpCatchFinding {
    <#
    .SYNOPSIS
        Finds catch-all clauses that swallow the exception in one C# file (Rule, Line, Message).
    #>
    [CmdletBinding()]
    [OutputType([pscustomobject])]
    param([Parameter(Mandatory)][string]$Path)

    $text = [System.IO.File]::ReadAllText($Path)
    $blank = ConvertTo-PcaiBlankedCSharp -Text $text
    $found = [System.Collections.Generic.List[object]]::new()
    $catchPattern = [regex]::new('\bcatch\b\s*(\(\s*(?:System\.)?Exception\b[^)]*\))?\s*(?<when>when\s*\([^)]*\)\s*)?\{')
    $defaultReturn = [regex]::new('\breturn\s+(null|false|0|-1|""|string\.Empty|default(\([^)]*\))?|\[\s*\]|Array\.Empty<[^>]+>\(\)|new\s+[\w.<>,\s\[\]]+?\s*(\(\s*\)|\[\s*0?\s*\])?\s*(\{\s*\})?)\s*;')

    foreach ($match in $catchPattern.Matches($blank)) {
        if ($match.Groups['when'].Success) { continue }
        $bodyStart = $match.Index + $match.Length
        $depth = 1
        $k = $bodyStart
        while ($k -lt $blank.Length -and $depth -gt 0) {
            $ch = $blank[$k]
            if ($ch -eq '{') { $depth++ } elseif ($ch -eq '}') { $depth-- }
            $k++
        }
        if ($depth -ne 0) { continue }
        $body = $blank.Substring($bodyStart, $k - 1 - $bodyStart)
        if ($body -match '\bthrow\b') { continue }
        $line = 1 + ($blank.Substring(0, $match.Index).Split("`n").Length - 1)
        if ($body.Trim().Length -eq 0) {
            $found.Add([pscustomobject]@{ Rule = 'catch-all-empty'; Line = $line; Message = 'catch-all with an empty body swallows every exception' })
        } elseif ($defaultReturn.IsMatch($body)) {
            $found.Add([pscustomobject]@{ Rule = 'catch-all-returns-default'; Line = $line; Message = 'catch-all returns a default value without rethrowing' })
        }
    }
    return $found.ToArray()
}

Export-ModuleMember -Function Get-PcaiPsAstFinding, Get-PcaiCSharpCatchFinding, Get-PcaiQualityFileScope, Get-PcaiRustTestStartLine, Get-PcaiChecksVersion, Get-PcaiBuiltinCommandSet
