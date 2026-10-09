#Requires -Version 7.0
BeforeAll {
    $repo = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
    . (Join-Path $repo 'Modules/PC-AI.Acceleration/Private/Initialize-RustTools.ps1')
    . (Join-Path $repo 'Modules/PC-AI.Acceleration/Public/Search-ContentFast.ps1')
    . (Join-Path $repo 'Modules/PC-AI.Acceleration/Public/Search-LogsFast.ps1')
    # Cache/path collaborators are routing seams; these fixtures exercise the
    # maintained search implementations, not cache or native DLL availability.
    function Resolve-PcaiPath { param($Path) $Path }
    function Get-PcaiCacheKey { param($Category, $Parameters) 'fixture-key' }
    function Get-PcaiCachedValue { param($Key, $TtlSeconds) $null }
    function Set-PcaiCachedValue { param($Key, $Value) $Value }
    $script:QualifiedRg = @(Get-Command rg -CommandType Application -ErrorAction Stop)[0].Source
    if (-not [IO.File]::Exists($script:QualifiedRg)) { throw 'Selected ripgrep is not an existing executable.' }
}

Describe 'Acceleration content and log search contracts' -Tag 'Unit', 'Acceleration' {
    BeforeEach {
        $script:SearchFixture = Join-Path $TestDrive ([guid]::NewGuid().ToString('N'))
        $null = New-Item -ItemType Directory -Path $script:SearchFixture
        [IO.File]::WriteAllText((Join-Path $script:SearchFixture 'first.log'), "before`nalpha a.b`nALPHA a+b`naXb`na+b`na.b`nalpha tail`nafter`n")
        [IO.File]::WriteAllText((Join-Path $script:SearchFixture 'second.log'), "alpha second`nother`nalpha third`n")
        [IO.File]::WriteAllText((Join-Path $script:SearchFixture 'other.txt'), "alpha excluded`n")
        Mock Get-RustToolPath { $script:QualifiedRg }
        Mock Search-WithPcaiNativeContent { $null }
        Mock Search-WithPcaiNativeLogs { $null }
    }

    It 'returns exact real ripgrep content positions and file filtering' {
        $result = @(Search-ContentFast -Path $script:SearchFixture -Pattern '^alpha' -FilePattern '*.log' -CaseSensitive)
        $result | Should -HaveCount 4
        @($result.LineNumber | Sort-Object) | Should -Be @(1,2,3,7)
        @($result | Where-Object Tool -ne 'ripgrep') | Should -HaveCount 0
        @($result | Where-Object { $_.Path -like '*.txt' }) | Should -HaveCount 0
    }
    It 'uses case-insensitive matching unless explicitly requested' {
        @(Search-ContentFast -Path $script:SearchFixture -Pattern '^alpha' -FilePattern '*.log') | Should -HaveCount 5
    }
    It 'preserves literal punctuation under the actual fixed-string backend' -ForEach @(
        @{ Literal = 'a.b'; Lines = @(2,6) },
        @{ Literal = 'a+b'; Lines = @(3,5) }
    ) {
        $result = @(Search-ContentFast -Path (Join-Path $script:SearchFixture 'first.log') -LiteralPattern $Literal)
        @($result.LineNumber | Sort-Object) | Should -Be $Lines
    }
    It 'limits the complete content result set across several files' {
        @(Search-ContentFast -Path $script:SearchFixture -Pattern alpha -FilePattern '*.log' -MaxResults 2) | Should -HaveCount 2
    }
    It 'returns each matching file once in FilesOnly mode' {
        $result = @(Search-ContentFast -Path $script:SearchFixture -Pattern alpha -FilePattern '*.log' -FilesOnly)
        $result | Should -HaveCount 2
        @($result.Path | Sort-Object -Unique) | Should -HaveCount 2
    }
    It 'applies the public global bound to FilesOnly results too' {
        @(Search-ContentFast -Path $script:SearchFixture -Pattern alpha -FilePattern '*.log' -FilesOnly -MaxResults 1) | Should -HaveCount 1
    }
    It 'treats an empty successful search as no matches' {
        @(Search-ContentFast -Path $script:SearchFixture -Pattern missing) | Should -HaveCount 0
    }
    It 'reports malformed regular expressions as a failure rather than empty success' {
        { Search-ContentFast -Path $script:SearchFixture -Pattern '[' -ErrorAction Stop -WarningAction SilentlyContinue } | Should -Throw
    }
    It 'does not interpret a leading hyphen pattern as a CLI option' {
        [IO.File]::WriteAllText((Join-Path $script:SearchFixture 'hyphen.log'), "-needle`n")
        $result = @(Search-ContentFast -Path $script:SearchFixture -LiteralPattern '-needle')
        $result | Should -HaveCount 1
        $result[0].Line | Should -BeExactly '-needle'
    }
    It 'includes ignored private fixture content when NoIgnore is selected' {
        [IO.File]::WriteAllText((Join-Path $script:SearchFixture '.ignore'), "hidden.log`n")
        [IO.File]::WriteAllText((Join-Path $script:SearchFixture 'hidden.log'), "needle-hidden`n")
        @(Search-ContentFast -Path $script:SearchFixture -Pattern 'needle-hidden' -NoIgnore) | Should -HaveCount 1
    }
    It 'accepts NoIgnore when only the maintained PowerShell fallback is available' {
        Mock Get-RustToolPath { $null }
        $result = @(Search-ContentFast -Path $script:SearchFixture -LiteralPattern 'a.b' -NoIgnore -ThrottleLimit 1)
        $result | Should -HaveCount 2
        @($result.LineNumber | Sort-Object) | Should -Be @(2,6)
    }
    It 'returns a public log count across matching files with colon-bearing paths' {
        $result = Search-LogsFast -Path $script:SearchFixture -Pattern '^alpha' -Include '*.log' -CaseSensitive -CountOnly
        $result.Count | Should -Be 4
        $result.Tool | Should -BeExactly 'ripgrep'
    }
    It 'honors the documented per-file log match bound' {
        $result = @(Search-LogsFast -Path $script:SearchFixture -Pattern alpha -Include '*.log' -MaxCount 1)
        $result | Should -HaveCount 2
        @($result | Group-Object Path | Where-Object Count -ne 1) | Should -HaveCount 0
    }
    It 'returns zero for a successful count with no matching logs' {
        (Search-LogsFast -Path $script:SearchFixture -Pattern missing -Include '*.log' -CountOnly).Count | Should -Be 0
    }
    It 'reports an actual invalid log regex instead of returning empty success' {
        { Search-LogsFast -Path $script:SearchFixture -Pattern '[' -ErrorAction Stop -WarningAction SilentlyContinue } | Should -Throw
    }
    It 'refuses plausible partial match output when its actual native child exits with an error' {
        $shim = Join-Path $script:SearchFixture 'partial-search.cmd'
        [IO.File]::WriteAllText($shim, "@echo off`r`necho {`"type`":`"match`",`"data`":{`"path`":{`"text`":`"partial.log`"},`"line_number`":1,`"lines`":{`"text`":`"alpha`"},`"submatches`":[]}}`r`nexit /b 2`r`n")
        { Search-WithRipgrepAdvanced -Path $script:SearchFixture -SearchPattern alpha -RgPath $shim -WarningAction SilentlyContinue } | Should -Throw '*exit code 2*'
        { Search-WithRipgrep -Path $script:SearchFixture -Pattern alpha -RgPath $shim -WarningAction SilentlyContinue } | Should -Throw '*exit code 2*'
    }
    It 'refuses plausible count output when its actual native child exits with an error' {
        $shim = Join-Path $script:SearchFixture 'partial-count.cmd'
        [IO.File]::WriteAllText($shim, "@echo off`r`necho 99`r`nexit /b 2`r`n")
        { Search-WithRipgrep -Path $script:SearchFixture -Pattern alpha -CountOnly -RgPath $shim -WarningAction SilentlyContinue } | Should -Throw '*exit code 2*'
    }
    It 'matches and counts actual files through the PowerShell log fallback' {
        Mock Get-RustToolPath { $null }
        $result = @(Search-LogsFast -Path $script:SearchFixture -Pattern '^alpha' -Include '*.log' -CaseSensitive)
        $result | Should -HaveCount 4
        (Search-LogsFast -Path $script:SearchFixture -Pattern '^alpha' -Include '*.log' -CaseSensitive -CountOnly).Count | Should -Be 4
    }
    It 'applies the same per-file log bound through the PowerShell fallback' {
        Mock Get-RustToolPath { $null }
        $result = @(Search-LogsFast -Path $script:SearchFixture -Pattern alpha -Include '*.log' -MaxCount 1)
        $result | Should -HaveCount 2
        (Search-LogsFast -Path $script:SearchFixture -Pattern alpha -Include '*.log' -MaxCount 1 -CountOnly).Count | Should -Be 2
    }
    It 'routes explicit NoIgnore=<Value> safely with synthetic native availability in a fresh process' -ForEach @(
        @{ Value = 'false'; Tool = 'pcai_native'; NativeCalls = 1 },
        @{ Value = 'true'; Tool = 'ripgrep'; NativeCalls = 0 }
    ) {
        # The ephemeral type tests public argument routing only. It never loads
        # a native DLL and cannot persist in the parent test process.
        $childScript = Join-Path $script:SearchFixture 'availability-route.ps1'
        $body = @'
param($Source, $InputFile, $Rg, $NoIgnoreValue)
$ErrorActionPreference='Stop'
Add-Type 'namespace PcaiNative { public static class PcaiCore { public static bool IsAvailable { get { return true; } } } }'
. $Source
function Resolve-PcaiPath { param($Path) $Path }
function Get-PcaiCacheKey { param($Category,$Parameters) 'fixture' }
function Get-PcaiCachedValue { param($Key,$TtlSeconds) $null }
function Set-PcaiCachedValue { param($Key,$Value) $Value }
function Get-RustToolPath { param($ToolName) $Rg }
$script:Calls=0
function Invoke-PcaiNativeContentSearch {
    param($Pattern,$Path,$FilePattern,$MaxResults,$ContextLines)
    $script:Calls++
    [pscustomobject]@{Matches=@([pscustomobject]@{Path=$Path;LineNumber=2;Line='alpha a.b';Before=@();After=@()})}
}
$result=@(Search-ContentFast -Path $InputFile -Pattern '^alpha a[.]b$' -CaseSensitive -NoIgnore:($NoIgnoreValue -eq 'true'))
if($result.Count -ne 1){throw "Routing result count differs: $($result.Count), calls=$script:Calls"}
[ordered]@{Tool=$result[0].Tool;NativeCalls=$script:Calls;Line=$result[0].Line}|ConvertTo-Json -Compress
'@
        [IO.File]::WriteAllText($childScript, $body)
        $source = Join-Path $repo 'Modules/PC-AI.Acceleration/Public/Search-ContentFast.ps1'
        $output = & ([Environment]::ProcessPath) -NoLogo -NoProfile -File $childScript $source (Join-Path $script:SearchFixture 'first.log') $script:QualifiedRg $Value
        $LASTEXITCODE | Should -Be 0
        $result = ($output -join "`n") | ConvertFrom-Json
        $result.Tool | Should -BeExactly $Tool
        $result.NativeCalls | Should -Be $NativeCalls
        $result.Line | Should -BeExactly 'alpha a.b'
    }
}
