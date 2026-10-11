#Requires -Version 7.0
# Optional pins bind an isolated qualification to exact source bytes. Ordinary
# repository runs snapshot both current source files and reject mid-run drift.
param(
    [string]$SourcePath = (Join-Path $PSScriptRoot '../../Modules/PC-AI.Acceleration/Public/Find-DuplicatesFast.ps1'),
    [string]$SourceSHA256,
    [string]$HelperPath = (Join-Path $PSScriptRoot '../../Modules/PC-AI.Acceleration/Private/DotNet-Helpers.ps1'),
    [string]$HelperSHA256,
    [string]$EvidenceRoot
)

Describe 'Shipped duplicate scan fallback contracts' -Tag 'Unit','Portable','Acceleration' {
    BeforeAll {
        $ErrorActionPreference = 'Stop'
        if ([string]::IsNullOrWhiteSpace($EvidenceRoot)) {
            $EvidenceRoot = Join-Path $TestDrive 'duplicate-fallback-contracts'
        }
        $script:BoundPins = @(foreach ($pin in @(@{Path=$SourcePath;SHA=$SourceSHA256},@{Path=$HelperPath;SHA=$HelperSHA256})) {
            $observed = (Get-FileHash -LiteralPath $pin.Path -Algorithm SHA256).Hash
            if ($pin.SHA -and $observed -cne $pin.SHA) { throw 'Bound source changed before execution.' }
            @{Path=$pin.Path;SHA=$observed}
        })
        . $SourcePath
        . $HelperPath
        # Only the fd discovery boundary is replaced. Scan, grouping, projection,
        # filesystem enumeration and actual shipped .NET hashing execute verbatim.
        function Get-RustToolPath { param([string]$ToolName) throw 'Unmocked tool discovery forbidden.' }
        function Test-PcaiNativeAvailable { throw 'Native execution forbidden.' }
        function Invoke-PcaiNativeDuplicates { throw 'Native execution forbidden.' }
        $script:FixtureRoot = Join-Path $EvidenceRoot 'fixture[owned]'
        if (Test-Path -LiteralPath $script:FixtureRoot) { throw 'Fixture custody collision.' }
        [void][IO.Directory]::CreateDirectory($script:FixtureRoot)
        [void][IO.Directory]::CreateDirectory((Join-Path $script:FixtureRoot 'nested'))
        $script:FixtureFiles = [ordered]@{
            'keep-a.txt'='AAAA'; 'keep-b.txt'='AAAA'; 'skip-c.txt'='AAAA'
            'same-size-different.txt'='BBBB'; 'unique.txt'='12345'
            'low-a.bin'='x'; 'low-b.bin'='x'; 'nested/deep.txt'='AAAA'
        }
        foreach ($entry in $script:FixtureFiles.GetEnumerator()) {
            [IO.File]::WriteAllBytes((Join-Path $script:FixtureRoot $entry.Key),[Text.Encoding]::UTF8.GetBytes($entry.Value))
        }
        $script:InertFd = Join-Path $EvidenceRoot 'inert-fd-throw.ps1'
        [IO.File]::WriteAllText($script:InertFd,"throw 'Owned inert fd execution failed.'`n",[Text.UTF8Encoding]::new($false))
        $script:ExitFd = Join-Path $EvidenceRoot 'inert-fd-exit.ps1'
        [IO.File]::WriteAllText($script:ExitFd,"param([Parameter(ValueFromRemainingArguments)][object[]]`$Arguments)`nif (`$env:PCAI_DUPLICATE_FD_PATH) { Write-Output `$env:PCAI_DUPLICATE_FD_PATH }`nexit ([int]`$env:PCAI_DUPLICATE_FD_EXIT)`n",[Text.UTF8Encoding]::new($false))
        function Assert-DuplicateGroup {
            param([object[]]$Rows,[int]$ExpectedCount,[long]$ExpectedSize,[string]$Algorithm)
            $Rows.Count | Should -Be 1
            $row=$Rows[0]
            $row.Count | Should -Be $ExpectedCount
            $row.FileSize | Should -Be $ExpectedSize
            $row.WastedBytes | Should -Be ($ExpectedSize*($ExpectedCount-1))
            $row.Algorithm | Should -Be $Algorithm
            $row.Files.Count | Should -Be $ExpectedCount
            $row.Duplicates.Count | Should -Be ($ExpectedCount-1)
            $row.Original | Should -Be @($row.Files | Sort-Object)[0]
            $row.Hash | Should -Be (Get-FileHash -LiteralPath $row.Original -Algorithm $Algorithm).Hash
        }
    }
    BeforeEach {
        Mock Get-RustToolPath { $null }
        Mock Write-Host {}
        Mock Write-Warning {}
    }

    # Protects: Exclude during the ordinary Get-ChildItem duplicate scan.
    # Detects: Excluded same-content file inflating Count and WastedBytes.
    # Needs: Eight tiny owned files, actual shipped scan/hash helper, no fd/native.
    # Breadcrumb: Invoke-PowerShellDuplicateScan Get-ChildItem parameters.
    It 'excludes same-content files from the no-fd public duplicate scan' {
        $rows=@(Find-DuplicatesFast -Path $script:FixtureRoot -Recurse -Include '*.txt' -Exclude 'skip-*' -MinimumSize 4 -MaximumSize 4 -DisableNative -ThrottleLimit 1)
        Assert-DuplicateGroup -Rows $rows -ExpectedCount 3 -ExpectedSize 4 -Algorithm SHA256
        @($rows[0].Files | ForEach-Object { [IO.Path]::GetFileName($_) }) | Should -Not -Contain 'skip-c.txt'
    }

    It 'applies Include and Exclude to a nonrecursive no-fd root' {
        $rows=@(Find-DuplicatesFast -Path $script:FixtureRoot -Include '*.txt' -Exclude 'skip-*' -MinimumSize 4 -MaximumSize 4 -DisableNative -ThrottleLimit 1)
        Assert-DuplicateGroup -Rows $rows -ExpectedCount 2 -ExpectedSize 4 -Algorithm SHA256
        @($rows[0].Files | ForEach-Object { [IO.Path]::GetFileName($_) }) | Should -Not -Contain 'deep.txt'
    }

    # Protects: The caller's nonrecursive scope after fd throws.
    # Detects: Nested recording admitted by an unconditional Recurse fallback.
    # Needs: Real fixture enumeration and an owned inert throwing fd script.
    # Breadcrumb: Find-WithFdForDuplicates catch Get-ChildItem route.
    It 'keeps nested files out of the fd-error fallback when Recurse is false' {
        $files=@(Find-WithFdForDuplicates -Path $script:FixtureRoot -Recurse:$false -FdPath $script:InertFd)
        $files.Count | Should -Be 7
        @($files.Name) | Should -Not -Contain 'deep.txt'
        Should -Invoke Write-Warning -Exactly -Times 1
    }

    # Protects: Both Include and Exclude after an fd execution exception.
    # Detects: Catch route dropping either filter while preserving recursion.
    # Needs: Actual Get-ChildItem over eight fixture files and inert fd failure.
    # Breadcrumb: Find-WithFdForDuplicates fallback parameter forwarding.
    It 'retains Include and Exclude together through the recursive fd-error fallback' {
        $files=@(Find-WithFdForDuplicates -Path $script:FixtureRoot -Recurse -Include '*.txt' -Exclude 'skip-*' -FdPath $script:InertFd)
        $files.Count | Should -Be 5
        @($files.Name) | Should -Contain 'deep.txt'
        @($files.Name) | Should -Not -Contain 'skip-c.txt'
        @($files.Extension | Select-Object -Unique) | Should -Be @('.txt')
    }

    It 'discards partial fd output after a nonzero exit and preserves fallback scope' {
        $savedPath=$env:PCAI_DUPLICATE_FD_PATH; $savedExit=$env:PCAI_DUPLICATE_FD_EXIT
        try {
            $env:PCAI_DUPLICATE_FD_PATH=Join-Path $script:FixtureRoot 'nested/deep.txt'
            $env:PCAI_DUPLICATE_FD_EXIT='7'
            $files=@(Find-WithFdForDuplicates -Path $script:FixtureRoot -Recurse:$false -Include '*.txt' -Exclude 'skip-*' -FdPath $script:ExitFd)
            $files.Count | Should -Be 4
            @($files.Name) | Should -Not -Contain 'deep.txt'
            @($files.Name) | Should -Not -Contain 'skip-c.txt'
            Should -Invoke Write-Warning -Exactly -Times 1
        } finally { $env:PCAI_DUPLICATE_FD_PATH=$savedPath; $env:PCAI_DUPLICATE_FD_EXIT=$savedExit }
    }

    It 'preserves successful fd output without invoking fallback' {
        $savedPath=$env:PCAI_DUPLICATE_FD_PATH; $savedExit=$env:PCAI_DUPLICATE_FD_EXIT
        try {
            $env:PCAI_DUPLICATE_FD_PATH=Join-Path $script:FixtureRoot 'keep-a.txt'
            $env:PCAI_DUPLICATE_FD_EXIT='0'
            $files=@(Find-WithFdForDuplicates -Path $script:FixtureRoot -FdPath $script:ExitFd)
            $files.Count | Should -Be 1
            $files[0].Name | Should -Be 'keep-a.txt'
            Should -Invoke Write-Warning -Exactly -Times 0
        } finally { $env:PCAI_DUPLICATE_FD_PATH=$savedPath; $env:PCAI_DUPLICATE_FD_EXIT=$savedExit }
    }

    It 'accepts a successful empty fd result rather than scanning the fallback tree' {
        $savedPath=$env:PCAI_DUPLICATE_FD_PATH; $savedExit=$env:PCAI_DUPLICATE_FD_EXIT
        try {
            $env:PCAI_DUPLICATE_FD_PATH=$null
            $env:PCAI_DUPLICATE_FD_EXIT='0'
            @(Find-WithFdForDuplicates -Path $script:FixtureRoot -FdPath $script:ExitFd).Count | Should -Be 0
            Should -Invoke Write-Warning -Exactly -Times 0
        } finally { $env:PCAI_DUPLICATE_FD_PATH=$savedPath; $env:PCAI_DUPLICATE_FD_EXIT=$savedExit }
    }

    # Protects: Public duplicate results after fd fallback with bounded sizes.
    # Detects: Dropped exclusion reaching real grouping and savings projection.
    # Needs: Actual public scan/hash helper; only discovery points at inert fd.
    # Breadcrumb: Find-DuplicatesFast -> Find-WithFdForDuplicates -> scan.
    It 'preserves filtered duplicate groups when the discovered fd throws' {
        Mock Get-RustToolPath { $script:InertFd }
        $rows=@(Find-DuplicatesFast -Path $script:FixtureRoot -Recurse -Include '*.txt' -Exclude 'skip-*' -MinimumSize 4 -MaximumSize 4 -DisableNative -ThrottleLimit 1)
        Assert-DuplicateGroup -Rows $rows -ExpectedCount 3 -ExpectedSize 4 -Algorithm SHA256
        Should -Invoke Get-RustToolPath -Exactly -Times 1
        Should -Invoke Write-Warning -Exactly -Times 1
    }

    # Protects: Genuine byte-equal grouping for each documented algorithm.
    # Detects: Size-only grouping, fabricated hashes or wrong wasted-byte math.
    # Needs: Actual shipped hash helper on four equal files plus unequal control.
    # Breadcrumb: Invoke-PowerShellDuplicateScan SHA256/SHA1/MD5 projections.
    It 'computes genuine equal-content groups with <Algorithm>' -TestCases @(@{Algorithm='SHA256'},@{Algorithm='SHA1'},@{Algorithm='MD5'}) {
        param($Algorithm)
        $rows=@(Find-DuplicatesFast -Path $script:FixtureRoot -Recurse -MinimumSize 4 -MaximumSize 4 -Algorithm $Algorithm -DisableNative -ThrottleLimit 1)
        Assert-DuplicateGroup -Rows $rows -ExpectedCount 4 -ExpectedSize 4 -Algorithm $Algorithm
        @($rows[0].Files | ForEach-Object { [IO.Path]::GetFileName($_) }) | Should -Not -Contain 'same-size-different.txt'
    }

    # Protects: Inclusive minimum and maximum byte filters.
    # Detects: Wrong boundary admission or large duplicates reaching small scan.
    # Needs: Actual one-byte fixture pair and shipped scan/hash helper.
    # Breadcrumb: Invoke-PowerShellDuplicateScan size filter and wasted bytes.
    It 'keeps an exact one-byte duplicate pair at both size boundaries' {
        $rows=@(Find-DuplicatesFast -Path $script:FixtureRoot -Recurse -MinimumSize 1 -MaximumSize 1 -DisableNative -ThrottleLimit 1)
        Assert-DuplicateGroup -Rows $rows -ExpectedCount 2 -ExpectedSize 1 -Algorithm SHA256
    }

    # Protects: Hash equality rather than size equality as duplicate evidence.
    # Detects: Distinct four-byte contents wrongly published as duplicates.
    # Needs: Actual two same-size files with different contents; no native metrics.
    # Breadcrumb: Invoke-PowerShellDuplicateScan hash grouping negative control.
    It 'does not group same-sized files with different bytes' {
        $rows=@(Find-DuplicatesFast -Path $script:FixtureRoot -Recurse -Include 'keep-a.txt','same-size-different.txt' -MinimumSize 4 -MaximumSize 4 -DisableNative -ThrottleLimit 1)
        $rows.Count | Should -Be 0
    }

    # Protects: Genuine empty duplicate result for a unique-size candidate.
    # Detects: Phantom group or hash work despite no same-size candidates.
    # Needs: Existing five-byte unique fixture and a forbidden hash boundary.
    # Breadcrumb: Invoke-PowerShellDuplicateScan early no-candidates branch.
    It 'returns no duplicate group for a single admitted unique-size file' {
        Mock Invoke-ParallelFileHash { throw 'Unexpected hashing after unique-size admission.' }
        $rows=@(Find-DuplicatesFast -Path $script:FixtureRoot -Recurse -MinimumSize 5 -MaximumSize 5 -DisableNative -ThrottleLimit 1)
        $rows.Count | Should -Be 0
        Should -Invoke Invoke-ParallelFileHash -Exactly -Times 0
    }

    # Protects: Genuine empty duplicate result when size selection admits nothing.
    # Detects: Unbounded hash work or stale cached groups after empty enumeration.
    # Needs: Tiny fixture tree and forbidden hashing boundary, no real tree scans.
    # Breadcrumb: Invoke-PowerShellDuplicateScan early zero-file branch.
    It 'returns no duplicate group when no file meets the requested size' {
        Mock Invoke-ParallelFileHash { throw 'Unexpected hashing after zero-file admission.' }
        $rows=@(Find-DuplicatesFast -Path $script:FixtureRoot -Recurse -MinimumSize 6 -MaximumSize 6 -DisableNative -ThrottleLimit 1)
        $rows.Count | Should -Be 0
        Should -Invoke Invoke-ParallelFileHash -Exactly -Times 0
    }

    AfterAll {
        foreach ($pin in $script:BoundPins) {
            if ((Get-FileHash -LiteralPath $pin.Path -Algorithm SHA256).Hash -cne $pin.SHA) { throw 'Bound source changed during execution.' }
        }
        $files=@(Get-ChildItem -LiteralPath $script:FixtureRoot -File -Recurse)
        if ($files.Count -ne 8 -or ($files | Measure-Object -Property Length -Sum).Sum -ne 27) { throw 'Fixture identity/size changed.' }
    }
}
