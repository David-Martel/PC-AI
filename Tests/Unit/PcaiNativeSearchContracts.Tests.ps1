#Requires -Version 7.0
BeforeAll {
    $repo = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
    . (Join-Path $repo 'Modules/PC-AI.Acceleration/Public/Find-FilesFast.ps1')
    $tokens = $null
    $errors = $null
    $ast = [Management.Automation.Language.Parser]::ParseFile(
        (Join-Path $repo 'Modules/PC-AI.Acceleration/PC-AI.Acceleration.psm1'),
        [ref]$tokens, [ref]$errors)
    if ($errors.Count) { throw 'Acceleration module parse failed.' }
    foreach ($name in @('Resolve-PcaiPath', 'New-PcaiPathItem', 'Get-PcaiRelativeDepth',
            'Get-PcaiCacheKey', 'Get-PcaiCachedValue', 'Set-PcaiCachedValue')) {
        $definitions = @($ast.FindAll({
            param($node)
            $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -ceq $name
        }, $true))
        if ($definitions.Count -ne 1) { throw "Expected one maintained $name definition." }
        . ([scriptblock]::Create($definitions[0].Extent.Text))
    }
    $script:SearchFd = @(Get-Command fd -CommandType Application -ErrorAction Stop)[0].Source
    function Get-RustToolPath { param($ToolName) $script:SearchFd }
    function Test-PcaiNativeAvailable { $true }
    function Invoke-PcaiNativeFileSearch {
        param($Pattern, $Path, $MaxResults)
        $script:UnsupportedNativeCalls++
        # This inert transport exposes the wrapper's discarded filters. It
        # qualifies neither Rust glob matching nor the native walker/ABI.
        [pscustomobject]@{ Files = @(Get-ChildItem -LiteralPath $Path -Recurse -File |
                Where-Object Name -Like $Pattern |
                ForEach-Object { [pscustomobject]@{ Path = $_.FullName } }) }
    }
    function Invoke-PcaiNativeDirectoryManifest {
        $script:UnsupportedNativeCalls++
        throw 'Unsupported native manifest was selected.'
    }
}

Describe 'Public file selection preserves filters with native preference' -Tag 'Unit', 'Acceleration' {
    BeforeEach {
        $script:SearchFixture = Join-Path $TestDrive ([guid]::NewGuid().ToString('N'))
        [void][IO.Directory]::CreateDirectory((Join-Path $script:SearchFixture 'nested'))
        # A real owned Git repository makes the ignore oracle independent of
        # whether the surrounding checkout or test directory is ignored.
        & git init --quiet $script:SearchFixture
        if ($LASTEXITCODE -ne 0) { throw 'Owned Git fixture creation failed.' }
        [IO.File]::WriteAllText((Join-Path $script:SearchFixture '.gitignore'), "ignored.txt`n")
        foreach ($leaf in @('match.txt', 'other.txt', 'match.log', 'report3', 'report3.txt',
                '.hidden.txt', 'ignored.txt', 'nested/match-deep.txt')) {
            [IO.File]::WriteAllText((Join-Path $script:SearchFixture $leaf), 'owned search fixture')
        }
        $script:UnsupportedNativeCalls = 0
        $script:PcaiQueryCache = [pscustomobject]@{
            MaxEntries = 128
            Entries = [Collections.Specialized.OrderedDictionary]::new()
        }
    }

    # Full dispatcher and physical fd run; the native bridge is an inert
    # collaborator. Exact results detect dropped/interpreted filters.
    It 'selects <Case> using the documented intersection' -ForEach @(
        @{ Case = 'pattern and extension'; Arguments = @{ Pattern = '^match'; Extension = @('txt'); Type = 'file'; NoIgnore = $true; MaxDepth = 2 }; Expected = @('match-deep.txt', 'match.txt') },
        @{ Case = 'multiple extensions and regex'; Arguments = @{ Pattern = '^match'; Extension = @('txt', 'log'); Type = 'file'; NoIgnore = $true; MaxDepth = 2 }; Expected = @('match-deep.txt', 'match.log', 'match.txt') },
        @{ Case = 'regex rather than glob'; Arguments = @{ Pattern = 'report[0-9]'; Type = 'file'; NoIgnore = $true; MaxDepth = 1 }; Expected = @('report3', 'report3.txt') },
        @{ Case = 'exclusion with explicit no-ignore'; Arguments = @{ Extension = @('txt'); Exclude = @('match*'); Type = 'file'; NoIgnore = $true; MaxDepth = 2 }; Expected = @('ignored.txt', 'other.txt', 'report3.txt') },
        @{ Case = 'depth restriction'; Arguments = @{ Pattern = '^match'; Extension = @('txt'); Type = 'file'; NoIgnore = $true; MaxDepth = 1 }; Expected = @('match.txt') },
        @{ Case = 'successful no-match'; Arguments = @{ Pattern = '^absent$'; Extension = @('txt'); Type = 'file'; NoIgnore = $true; MaxDepth = 2 }; Expected = @() },
        @{ Case = 'hidden inclusion'; Arguments = @{ Pattern = 'hidden'; Extension = @('txt'); Type = 'file'; NoIgnore = $true; Hidden = $true }; Expected = @('.hidden.txt') },
        @{ Case = 'Git ignore defaults'; Arguments = @{ Extension = @('txt'); Type = 'file' }; Expected = @('match-deep.txt', 'match.txt', 'other.txt', 'report3.txt') }
    ) {
        $actual = @(Find-FilesFast -Path $script:SearchFixture -PreferNative @Arguments)
        (@($actual.Name | Sort-Object) -join '|') | Should -BeExactly (@($Expected | Sort-Object) -join '|')
        $script:UnsupportedNativeCalls | Should -Be 0
        # A second call preserves the complete selection. Empty-result cache
        # hits are not qualified here; unsupported native transports stay excluded.
        $cached = @(Find-FilesFast -Path $script:SearchFixture -PreferNative @Arguments)
        (@($cached.Name | Sort-Object) -join '|') | Should -BeExactly (@($Expected | Sort-Object) -join '|')
        $script:UnsupportedNativeCalls | Should -Be 0
    }

    # Protects upgraded callers from results populated with the old native
    # semantics. The maintained cache implementation is used, not a cache spy.
    It 'does not consume a legacy cached result with discarded pattern filtering' {
        $legacyKey = Get-PcaiCacheKey -Category 'find-files' -Parameters @{
            path = $script:SearchFixture; pattern = '^match'; extension = 'txt'; type = 'file'
            depth = 2; hidden = $false; exclude = ''; fullPath = $false
            noIgnore = $true; preferNative = $true
        }
        $incorrect = Get-Item -LiteralPath (Join-Path $script:SearchFixture 'other.txt')
        $null = Set-PcaiCachedValue -Key $legacyKey -Value @($incorrect)
        $actual = @(Find-FilesFast -Path $script:SearchFixture -Pattern '^match' -Extension txt -Type file -MaxDepth 2 -NoIgnore -PreferNative)
        (@($actual.Name | Sort-Object) -join '|') | Should -BeExactly 'match-deep.txt|match.txt'
        $script:PcaiQueryCache.Entries.Contains($legacyKey) | Should -BeTrue
        $script:UnsupportedNativeCalls | Should -Be 0
    }
}
