param(
    [string]$SourceRoot = ([IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../..')))
)

BeforeAll {
    $script:OriginalProcessPath = $env:PATH
    $script:OriginalDisableNative = $env:PCAI_DISABLE_NATIVE
    $script:OriginalFixtureVariable = $env:PCAI_PATH_LITERAL_FIXTURE
    $env:PCAI_DISABLE_NATIVE = '1'
    $script:FixtureRoot = Join-Path $TestDrive 'path-literal-fixture'
    foreach ($leaf in @('[stage]', 'matcha', 'ordinary', 'with spaces')) {
        [IO.Directory]::CreateDirectory((Join-Path $script:FixtureRoot $leaf)) | Out-Null
    }
    $env:PCAI_PATH_LITERAL_FIXTURE = Join-Path $script:FixtureRoot '[stage]'

    # Load complete maintained functions; suppress only logging/backup writes.
    # Repair's static persistent-environment read is the sole source substitution:
    # the real parser, existence checks, normalization and ShouldProcess execute.
    $definitions = foreach ($entry in @(
        @{ Path = 'Modules/PC-AI.Cleanup/Private/Cleanup-Helpers.ps1'; Name = 'Test-PathExists' },
        @{ Path = 'Modules/PC-AI.Cleanup/Public/Get-PathDuplicates.ps1'; Name = 'Get-PathDuplicates' },
        @{ Path = 'Modules/PC-AI.Cleanup/Public/Repair-MachinePath.ps1'; Name = 'Repair-MachinePath' }
    )) {
        $tokens = $null; $errors = $null
        $ast = [Management.Automation.Language.Parser]::ParseFile((Join-Path $SourceRoot $entry.Path), [ref]$tokens, [ref]$errors)
        if ($errors.Count) { throw 'Maintained PATH source did not parse.' }
        $functions = @($ast.FindAll({ param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst] }, $true) | Where-Object Name -EQ $entry.Name)
        if ($functions.Count -ne 1) { throw 'Maintained PATH function identity is ambiguous.' }
        $definition = $functions[0].Extent.Text
        if ($entry.Name -eq 'Repair-MachinePath') {
            $boundary = "[Environment]::GetEnvironmentVariable('PATH', `$Target)"
            if (($definition.Split([string[]]@($boundary), [StringSplitOptions]::None)).Count -ne 2) {
                throw 'Persistent PATH fixture boundary changed.'
            }
            $definition = $definition.Replace($boundary, '(Get-FixturePathValue)')
        }
        $definition
    }
    $script:FixtureModule = New-Module -Name PcaiPathLiteralFixture -ArgumentList (, $definitions) -ScriptBlock {
        param($definitions)
        $script:FixturePath = ''
        $script:RegistryWrites = 0
        function Write-CleanupLog { param($Message, $Level) }
        function Backup-EnvironmentVariable { param($Name, $Target, $BackupPath) 'owned-backup-boundary' }
        function Test-IsAdministrator { $true }
        function Get-FixturePathValue { $script:FixturePath }
        function Set-ItemProperty { $script:RegistryWrites++; throw 'Unexpected persistent PATH write.' }
        foreach ($definition in $definitions) { . ([scriptblock]::Create($definition)) }
    }
}

AfterAll {
    foreach ($entry in @(
        @{ Name = 'PATH'; Value = $script:OriginalProcessPath },
        @{ Name = 'PCAI_DISABLE_NATIVE'; Value = $script:OriginalDisableNative },
        @{ Name = 'PCAI_PATH_LITERAL_FIXTURE'; Value = $script:OriginalFixtureVariable }
    )) {
        if ($null -eq $entry.Value) {
            Remove-Item -LiteralPath "Env:$($entry.Name)" -ErrorAction SilentlyContinue
        } else {
            [Environment]::SetEnvironmentVariable($entry.Name, $entry.Value, 'Process')
        }
    }
    if ($script:FixtureModule) { Remove-Module $script:FixtureModule -Force -ErrorAction Stop }
}

Describe 'Literal PATH existence' -Tag 'Unit', 'Cleanup', 'Fast', 'Portable' {
    AfterEach {
        $env:PATH = $script:OriginalProcessPath
        (& $script:FixtureModule { $script:RegistryWrites }) | Should -Be 0
    }

    # Protects: literal names and exact absence after percent-variable expansion.
    # Detects: bracket false-negatives and wildcard matches mistaken for existence.
    # Needs: real owned directories; no filesystem or existence-command mock.
    # Breadcrumb: path-literal-repair-actual-r1 baseline/candidate receipts.
    It 'checks the exact helper path <Leaf>' -ForEach @(
        @{ Leaf = '[stage]'; Expected = $true },
        @{ Leaf = 'match[a]'; Expected = $false },
        @{ Leaf = 'match*'; Expected = $false },
        @{ Leaf = 'match?'; Expected = $false },
        @{ Leaf = 'ordinary'; Expected = $true },
        @{ Leaf = 'with spaces'; Expected = $true },
        @{ Leaf = 'missing'; Expected = $false }
    ) {
        $path = Join-Path $script:FixtureRoot $Leaf
        (& $script:FixtureModule { param($path) Test-PathExists -Path $path } $path) | Should -Be $Expected
    }

    # Protects: the existing percent-environment-variable contract.
    # Detects: expansion followed by wildcard interpretation of a literal bracket.
    # Needs: process-only owned variable, restored by AfterAll.
    # Breadcrumb: helper expansion and real bracket directory above.
    It 'expands a percent variable to a literal existing directory' {
        (& $script:FixtureModule { Test-PathExists -Path '%PCAI_PATH_LITERAL_FIXTURE%' }) | Should -BeTrue
    }

    # Protects: analysis reports the literal existing and absent entries accurately.
    # Detects: real bracket directory rejected or absent wildcard accepted.
    # Needs: real public analysis with process PATH only; native dispatch disabled.
    # Breadcrumb: canonical Get-PathDuplicates ProcessPath.AllEntries.
    It 'classifies exact process PATH entries without wildcard expansion' {
        $env:PATH = (@('[stage]', 'match[a]', 'match*', 'ordinary', 'missing') | ForEach-Object { Join-Path $script:FixtureRoot $_ }) -join ';'
        $analysis = (& $script:FixtureModule { Get-PathDuplicates -Target User -IncludeProcess }).ProcessPath
        $analysis.TotalEntries | Should -Be 5
        @($analysis.AllEntries | ForEach-Object Exists) | Should -Be @($true, $false, $false, $true, $false)
        $analysis.NonExistent.Count | Should -Be 3
    }

    # Protects: whitespace trimming and case/trailing-slash duplicate normalization.
    # Detects: literal correction changing duplicate identity or original indexing.
    # Needs: existing ordinary directory and unchanged real analysis body.
    # Breadcrumb: Analyze-PathVariable normalization/duplicate contract.
    It 'preserves normalized duplicate identity and empty-entry accounting' {
        $path = Join-Path $script:FixtureRoot 'ordinary'
        $env:PATH = " $path ;$($path.ToUpperInvariant())/;;"
        $analysis = (& $script:FixtureModule { Get-PathDuplicates -Target User -IncludeProcess }).ProcessPath
        $analysis.TotalEntries | Should -Be 2
        $analysis.UniqueEntries | Should -Be 1
        $analysis.EmptyEntries | Should -Be 2
        $analysis.Duplicates.Count | Should -Be 1
        $analysis.Duplicates[0].DuplicateOf | Should -Be 1
    }

    # Protects: repair retains real bracket paths and removes exact absent paths.
    # Detects: wildcard existence changes the proposed removal/final counts.
    # Needs: complete repair body; only persistent input, log and backup boundaries.
    # Breadcrumb: Repair-MachinePath RemoveNonExistent / ShouldProcess.
    It 'previews literal repair without persistent writes even with Force' {
        $value = (@('[stage]', 'match[a]', 'match*', 'ordinary', 'missing') | ForEach-Object { Join-Path $script:FixtureRoot $_ }) -join ';'
        $result = & $script:FixtureModule { param($value) $script:FixturePath = $value; Repair-MachinePath -Target User -RemoveNonExistent -Force -WhatIf } $value
        $result.OriginalCount | Should -Be 5
        $result.FinalCount | Should -Be 2
        $result.NonExistentRemoved | Should -Be 3
        $result.Success | Should -BeFalse
        @($result.Changes | ForEach-Object OriginalValue) | Should -Not -Contain (Join-Path $script:FixtureRoot '[stage]')
        $env:PATH | Should -BeExactly $script:OriginalProcessPath
    }

    # Protects: ordinary paths and percent-expanded literal paths stay usable.
    # Detects: repair expansion disagreeing with the helper/analysis contract.
    # Needs: same owned real directory, genuine ShouldProcess preview.
    # Breadcrumb: unchanged ExpandEnvironmentVariables before existence.
    It 'keeps an expanded literal entry and reports an ordinary missing entry' {
        $value = '%PCAI_PATH_LITERAL_FIXTURE%;' + (Join-Path $script:FixtureRoot 'missing')
        $result = & $script:FixtureModule { param($value) $script:FixturePath = $value; Repair-MachinePath -RemoveNonExistent -WhatIf } $value
        $result.FinalCount | Should -Be 1
        $result.NonExistentRemoved | Should -Be 1
        $result.Changes[0].OriginalValue | Should -BeExactly (Join-Path $script:FixtureRoot 'missing')
    }

    # Protects: optional removal stays optional; normalization remains independent.
    # Detects: literal fix accidentally deleting entries or changing trailing cleanup.
    # Needs: real existing plus absent literal entries; registry write tripwire.
    # Breadcrumb: Repair-MachinePath NormalizeSlashes and WhatIf.
    It 'preserves absent entries when removal is not requested and normalizes slashes' {
        $path = Join-Path $script:FixtureRoot 'ordinary'
        $value = "$path/;" + (Join-Path $script:FixtureRoot 'match[a]')
        $result = & $script:FixtureModule { param($value) $script:FixturePath = $value; Repair-MachinePath -NormalizeSlashes -WhatIf } $value
        $result.FinalCount | Should -Be 2
        $result.NonExistentRemoved | Should -Be 0
        $result.Changes.Count | Should -Be 1
        $result.Changes[0].NewValue | Should -BeExactly $path
        $env:PATH | Should -BeExactly $script:OriginalProcessPath
    }

    # Protects: repair's duplicate, whitespace and empty-entry behavior.
    # Detects: literal correction changing duplicate normalization or removal counts.
    # Needs: complete repair parser and real bracket directory, with WhatIf.
    # Breadcrumb: Repair-MachinePath seenPaths and empty-entry branches.
    It 'retains one literal path while removing its normalized duplicate and empty entries' {
        $path = Join-Path $script:FixtureRoot '[stage]'
        $value = " $path ;$($path.ToUpperInvariant())/;;"
        $result = & $script:FixtureModule { param($value) $script:FixturePath = $value; Repair-MachinePath -RemoveNonExistent -NormalizeSlashes -WhatIf } $value
        $result.FinalCount | Should -Be 1
        $result.DuplicatesRemoved | Should -Be 1
        $result.EmptyRemoved | Should -Be 2
        $result.NonExistentRemoved | Should -Be 0
        $env:PATH | Should -BeExactly $script:OriginalProcessPath
    }

    # Protects: whitespace-only helper inputs are not considered existing.
    # Detects: existence fix bypassing the existing whitespace guard.
    # Needs: complete maintained helper, no native or filesystem mutations.
    # Breadcrumb: Test-PathExists IsNullOrWhiteSpace guard.
    It 'rejects a whitespace-only helper path' {
        (& $script:FixtureModule { Test-PathExists -Path '   ' }) | Should -BeFalse
    }
}
