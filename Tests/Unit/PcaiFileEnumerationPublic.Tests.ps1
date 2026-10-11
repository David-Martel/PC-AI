#Requires -Version 7.0
<#
Public enumeration contracts use the normally imported canonical module and real,
owned filesystem bytes. Only optional executable/native discovery is suppressed;
enumeration, filtering, query caching, sizing and formatting remain production code.
These controls qualify managed fallback behavior, not native execution or speed.
#>
BeforeAll {
    $script:EnumerationModule = Import-Module (Join-Path $PSScriptRoot '../../Modules/PC-AI.Acceleration/PC-AI.Acceleration.psd1') -Force -PassThru
    $script:FixtureNumber = 0

    function Assert-EnumerationPaths {
        param([object[]]$Rows, [string[]]$Expected)
        (($Rows | ForEach-Object FullName | Sort-Object) -join "`n") |
            Should -Be (($Expected | Sort-Object) -join "`n")
    }

    function New-EnumerationBytes {
        param([string]$Path, [int]$Length)
        [void][IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($Path))
        [IO.File]::WriteAllBytes($Path, [byte[]]::new($Length))
    }
}

Describe 'Canonical public file enumeration on owned bytes' -Tag 'Unit', 'Acceleration' {
    BeforeEach {
        $script:SavedEnumerationEnvironment = @{}
        foreach ($name in @('PCAI_PREFER_NATIVE_SEARCH', 'PCAI_PREFER_RUST_CLI')) {
            $script:SavedEnumerationEnvironment[$name] = [Environment]::GetEnvironmentVariable($name)
        }
        $env:PCAI_PREFER_NATIVE_SEARCH = $null
        $env:PCAI_PREFER_RUST_CLI = '0'
        Mock Get-RustToolPath -ModuleName PC-AI.Acceleration { $null }
        Mock Get-PcaiPerfToolPath -ModuleName PC-AI.Acceleration { $null }
        Mock Test-PcaiNativeAvailable -ModuleName PC-AI.Acceleration { $false }
        Mock Initialize-PcaiNative -ModuleName PC-AI.Acceleration { throw 'Native initialization is outside this managed fixture.' }

        $script:FixtureNumber++
        $script:EnumerationRoot = Join-Path $TestDrive "enumeration-$script:FixtureNumber"
        [void][IO.Directory]::CreateDirectory($script:EnumerationRoot)
        foreach ($entry in @(
            @{ Name = 'alpha.txt'; Bytes = 1024 }
            @{ Name = 'readme.md'; Bytes = 512 }
            @{ Name = 'upper.TXT'; Bytes = 2048 }
            @{ Name = 'docs/notes.txt'; Bytes = 256 }
            @{ Name = 'docs/deep/last.txt'; Bytes = 128 }
            @{ Name = 'keep[1]/skip.txt'; Bytes = 64 }
            @{ Name = 'keep1/retain.txt'; Bytes = 32 }
            @{ Name = '.hidden.txt'; Bytes = 16 }
        )) {
            New-EnumerationBytes -Path (Join-Path $script:EnumerationRoot $entry.Name) -Length $entry.Bytes
        }
        if ($IsWindows) {
            $hiddenPath = Join-Path $script:EnumerationRoot '.hidden.txt'
            [IO.File]::SetAttributes($hiddenPath, [IO.FileAttributes]::Hidden)
        }
    }

    AfterEach {
        foreach ($entry in $script:SavedEnumerationEnvironment.GetEnumerator()) {
            [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value)
        }
    }

    # Protects: usable default fallback. Detects: missing recursion, invented rows or hidden leakage.
    # Needs: ordinary module import, real files, unavailable optional transports. Breadcrumb: default Find-FilesFast.
    It 'enumerates the complete visible tree with the default public options' {
        $rows = @(Find-FilesFast -Path $script:EnumerationRoot)
        Assert-EnumerationPaths $rows @(
            'alpha.txt', 'readme.md', 'upper.TXT', 'docs', 'docs/notes.txt',
            'docs/deep', 'docs/deep/last.txt', 'keep[1]', 'keep[1]/skip.txt', 'keep1', 'keep1/retain.txt'
        ).ForEach({ [IO.Path]::GetFullPath((Join-Path $script:EnumerationRoot $_)) })
        @($rows | Where-Object { -not ($_.GetType() -in @([IO.FileInfo], [IO.DirectoryInfo])) }).Count | Should -Be 0
        Should -Invoke Get-RustToolPath -ModuleName PC-AI.Acceleration -Times 1 -Exactly -ParameterFilter { $ToolName -eq 'fd' }
        Should -Invoke Initialize-PcaiNative -ModuleName PC-AI.Acceleration -Times 0 -Exactly
    }

    # Protects: extension selection. Detects: dot/case handling, accidental directories or wrong extensions.
    # Needs: mixed physical suffixes. Breadcrumb: public Extension/Type parameters.
    It 'accepts <Suffix> while retaining only matching physical files' -ForEach @(
        @{ Suffix = 'txt' }
        @{ Suffix = '.TXT' }
    ) {
        $rows = @(Find-FilesFast -Path $script:EnumerationRoot -Extension $Suffix -Type file)
        Assert-EnumerationPaths $rows @(
            'alpha.txt', 'upper.TXT', 'docs/notes.txt', 'docs/deep/last.txt', 'keep[1]/skip.txt', 'keep1/retain.txt'
        ).ForEach({ [IO.Path]::GetFullPath((Join-Path $script:EnumerationRoot $_)) })
        @($rows | Where-Object PSIsContainer).Count | Should -Be 0
        Should -Invoke Test-PcaiNativeAvailable -ModuleName PC-AI.Acceleration -Times 1 -Exactly
    }

    # Protects: regex semantics and combined restrictions. Detects: glob substitution or ignored extension.
    # Needs: positive and negative names. Breadcrumb: public Pattern plus Extension.
    It 'combines a filename regex with a physical extension restriction' {
        $rows = @(Find-FilesFast -Path $script:EnumerationRoot -Pattern '^(alpha|readme)\.' -Extension txt -Type f)
        Assert-EnumerationPaths $rows @((Join-Path $script:EnumerationRoot 'alpha.txt'))
    }

    # Protects: directory aliases. Detects: file leakage and omitted nested directories.
    # Needs: nested real directories. Breadcrumb: public directory/d selection.
    It 'returns only physical directories through the <DirectoryType> alias' -ForEach @(
        @{ DirectoryType = 'directory' }
        @{ DirectoryType = 'd' }
    ) {
        $rows = @(Find-FilesFast -Path $script:EnumerationRoot -Type $DirectoryType)
        Assert-EnumerationPaths $rows @('docs', 'docs/deep', 'keep[1]', 'keep1').ForEach({
            [IO.Path]::GetFullPath((Join-Path $script:EnumerationRoot $_))
        })
        @($rows | Where-Object { -not $_.PSIsContainer }).Count | Should -Be 0
    }

    # Protects: bounded recursion. Detects: ignored depth with an actual deeper negative file.
    # Needs: three physical levels. Breadcrumb: Get-ChildItem fallback MaxDepth=1.
    It 'keeps root and immediate-child files but excludes a deeper file with MaxDepth one' {
        $rows = @(Find-FilesFast -Path $script:EnumerationRoot -Type file -MaxDepth 1)
        Assert-EnumerationPaths $rows @(
            'alpha.txt', 'readme.md', 'upper.TXT', 'docs/notes.txt', 'keep[1]/skip.txt', 'keep1/retain.txt'
        ).ForEach({ [IO.Path]::GetFullPath((Join-Path $script:EnumerationRoot $_)) })
    }

    # Protects: literal exclusions. Detects: treating brackets as regex or ignoring exclusion.
    # Needs: keep[1] versus keep1 real directories. Breadcrumb: public Exclude literal substring.
    It 'excludes a literal bracket-bearing path without excluding its similar sibling' {
        $rows = @(Find-FilesFast -Path $script:EnumerationRoot -Type file -Exclude 'keep[1]')
        Assert-EnumerationPaths $rows @(
            'alpha.txt', 'readme.md', 'upper.TXT', 'docs/notes.txt', 'docs/deep/last.txt', 'keep1/retain.txt'
        ).ForEach({ [IO.Path]::GetFullPath((Join-Path $script:EnumerationRoot $_)) })
    }

    # Protects: opt-in hidden visibility. Detects: discarded Hidden or accidental inclusion by default.
    # Needs: OS-appropriate hidden file. Breadcrumb: public Hidden/FullPath/NoIgnore options.
    It 'includes the actual hidden file only when explicitly requested' {
        $ordinary = @(Find-FilesFast -Path $script:EnumerationRoot -Type file)
        $inclusive = @(Find-FilesFast -Path $script:EnumerationRoot -Type file -Hidden -FullPath -NoIgnore)
        $ordinary.Count | Should -Be 7
        $inclusive.Count | Should -Be 8
        @($inclusive | Where-Object Name -EQ '.hidden.txt').Count | Should -Be 1
        @($inclusive | Where-Object { -not [IO.Path]::IsPathFullyQualified($_.FullName) }).Count | Should -Be 0
    }

    # Protects: no-result behavior. Detects: null marker rows or unrelated content on a miss.
    # Needs: a populated real tree. Breadcrumb: public Pattern negative.
    It 'returns no rows for a regex with no matching physical name' {
        @(Find-FilesFast -Path $script:EnumerationRoot -Pattern '^absent$' -Type file).Count | Should -Be 0
    }

    # Protects: query cache and parameter isolation. Detects: repeated scans or extension key collision.
    # Needs: a physical mutation between public calls. Breadcrumb: genuine public cache hit.
    It 'reuses a recent identical query while a different extension still sees its own files' {
        $first = @(Find-FilesFast -Path $script:EnumerationRoot -Extension txt -Type file)
        New-EnumerationBytes -Path (Join-Path $script:EnumerationRoot 'new.txt') -Length 1
        $cached = @(Find-FilesFast -Path $script:EnumerationRoot -Extension txt -Type file)
        Assert-EnumerationPaths $cached @($first.FullName)
        $other = @(Find-FilesFast -Path $script:EnumerationRoot -Extension md -Type file)
        Assert-EnumerationPaths $other @((Join-Path $script:EnumerationRoot 'readme.md'))
        Should -Invoke Get-RustToolPath -ModuleName PC-AI.Acceleration -Times 2 -Exactly -ParameterFilter { $ToolName -eq 'fd' }
    }
}

Describe 'Canonical public disk enumeration on owned bytes' -Tag 'Unit', 'Acceleration' {
    BeforeEach {
        Mock Get-RustToolPath -ModuleName PC-AI.Acceleration { $null }
        Mock Get-PcaiPerfToolPath -ModuleName PC-AI.Acceleration { $null }
        $script:FixtureNumber++
        $script:DiskRoot = Join-Path $TestDrive "disk-$script:FixtureNumber"
        [void][IO.Directory]::CreateDirectory((Join-Path $script:DiskRoot 'empty'))
        foreach ($entry in @(
            @{ Name = 'base.bin'; Bytes = 512 }
            @{ Name = 'alpha/a.bin'; Bytes = 1024 }
            @{ Name = 'alpha/b.bin'; Bytes = 512 }
            @{ Name = 'alpha/inner/c.bin'; Bytes = 256 }
            @{ Name = 'zeta/z.bin'; Bytes = 2048 }
        )) {
            New-EnumerationBytes -Path (Join-Path $script:DiskRoot $entry.Name) -Length $entry.Bytes
        }
    }

    # Protects: recursive totals at root-only display depth. Detects: nonrecursive totals or duplicate rows.
    # Needs: five real files totaling 4352 bytes. Breadcrumb: public Depth zero, Top zero, ThrottleLimit one.
    It 'reports the exact recursive root total when display Depth is zero' {
        $rows = @(Get-DiskUsageFast -Path $script:DiskRoot -Depth 0 -ThrottleLimit 1)
        $rows.Count | Should -Be 1
        $rows[0].Path | Should -Be $script:DiskRoot
        $rows[0].SizeBytes | Should -Be 4352
        $rows[0].FileCount | Should -Be 5
        $rows[0].Tool | Should -Be 'PowerShell'
    }

    # Protects: display recursion boundary. Detects: missing immediate directories or a deeper display row.
    # Needs: real nested directories; aggregate totals remain recursive. Breadcrumb: public Depth one.
    It 'displays immediate directories without displaying the deeper inner directory' {
        $rows = @(Get-DiskUsageFast -Path $script:DiskRoot -Depth 1 -ThrottleLimit 1)
        (($rows.Path | Sort-Object) -join "`n") | Should -Be ((@(
            $script:DiskRoot, (Join-Path $script:DiskRoot 'alpha'), (Join-Path $script:DiskRoot 'empty'), (Join-Path $script:DiskRoot 'zeta')
        ) | Sort-Object) -join "`n")
        ($rows | Where-Object Path -EQ (Join-Path $script:DiskRoot 'alpha')).SizeBytes | Should -Be 1792
    }

    # Protects: distinct public sorting contracts. Detects: a shared wrong sort across size/name/count.
    # Needs: exact totals and counts deliberately disagreeing. Breadcrumb: public SortBy with real enumeration.
    It 'orders the real rows by <Order> without changing their totals' -ForEach @(
        @{ Order = 'size'; Names = @('.', 'zeta', 'alpha', 'alpha/inner', 'empty') }
        @{ Order = 'count'; Names = @('.', 'alpha', 'alpha/inner', 'zeta', 'empty') }
        @{ Order = 'name'; Names = @('.', 'alpha', 'alpha/inner', 'empty', 'zeta') }
    ) {
        $rows = @(Get-DiskUsageFast -Path $script:DiskRoot -Depth 2 -SortBy $Order -ThrottleLimit 1)
        $expected = @($Names | ForEach-Object {
            if ($_ -eq '.') { $script:DiskRoot } else { [IO.Path]::GetFullPath((Join-Path $script:DiskRoot $_)) }
        })
        if ($Order -eq 'count') {
            # The two count-one rows are tied; only their membership is constrained.
            $rows[0].Path | Should -Be $expected[0]
            $rows[1].Path | Should -Be $expected[1]
            (($rows[2..3].Path | Sort-Object) -join "`n") | Should -Be (($expected[2..3] | Sort-Object) -join "`n")
            $rows[4].Path | Should -Be $expected[4]
        } else {
            ($rows.Path -join "`n") | Should -Be ($expected -join "`n")
        }
        $rows.Count | Should -Be 5
        ($rows | Where-Object Path -EQ $script:DiskRoot).SizeBytes | Should -Be 4352
        ($rows | Where-Object Path -EQ (Join-Path $script:DiskRoot 'alpha')).FileCount | Should -Be 3
    }

    # Protects: inclusive size threshold and empty output. Detects: > instead of >= or omitted filtering.
    # Needs: a physical 2048-byte directory and larger recursive root. Breadcrumb: public MinSize parsing.
    It 'retains the equality row at two KB and returns no rows at eight KB' {
        $rows = @(Get-DiskUsageFast -Path $script:DiskRoot -Depth 2 -MinSize '2KB' -ThrottleLimit 1)
        $rows.Count | Should -Be 2
        ($rows.Path -join "`n") | Should -Be (@($script:DiskRoot, (Join-Path $script:DiskRoot 'zeta')) -join "`n")
        $rows[1].SizeBytes | Should -Be 2048
        @(Get-DiskUsageFast -Path $script:DiskRoot -Depth 2 -MinSize '8KB' -ThrottleLimit 1).Count | Should -Be 0
    }

    # Protects: empty-directory shape. Detects: dropped zero-size rows, null count or invented content.
    # Needs: an actual empty directory. Breadcrumb: public disk result SizeHuman.
    It 'returns an exact zero-byte zero-file row for a physical empty directory' {
        $rows = @(Get-DiskUsageFast -Path (Join-Path $script:DiskRoot 'empty') -Depth 0 -ThrottleLimit 1)
        $rows.Count | Should -Be 1
        $rows[0].SizeBytes | Should -Be 0
        $rows[0].FileCount | Should -Be 0
        $rows[0].SizeHuman | Should -Be '0 B'
        $rows[0].Tool | Should -Be 'PowerShell'
    }
}
