# Real private files exercise the full public wrapper; only native availability
# is replaced. Zero/single pipelines and post-limit selection must stay usable
# under strict callers without changing result selection or provenance.
BeforeAll {
    $global:PCAI_StrictSearchOutcomesR1 = [Collections.Generic.List[object]]::new()
    . (Join-Path $PSScriptRoot '../../Modules/PC-AI.LLM/Public/Invoke-NativeSearch.ps1')
    function Test-PcaiNativeAvailable { throw 'Unexpected native availability call' }
    function Invoke-PcaiNativeFileSearch { param($Pattern, $Path, $MaxResults) throw 'Unexpected native file search' }
    function Invoke-PcaiNativeContentSearch { param($Pattern, $Path, $FilePattern, $MaxResults, $ContextLines) throw 'Unexpected native content search' }
}

Describe 'Actual managed search collection cardinality' -Tag 'Unit', 'LLM', 'Portable' {
    BeforeEach {
        Set-StrictMode -Version Latest
        Mock Get-Module { [pscustomobject]@{ Name = 'PC-AI.Acceleration' } } -ParameterFilter { $Name -eq 'PC-AI.Acceleration' }
        Mock Test-PcaiNativeAvailable { $false }
        Mock Import-Module { throw 'Unexpected acceleration import' }
        Mock Invoke-PcaiNativeFileSearch { throw 'Unexpected native file search' }
        Mock Invoke-PcaiNativeContentSearch { throw 'Unexpected native content search' }
    }

    # Protects real file selection/counts. Detects scalar discovery and scalar
    # Select-Object output. Needs zero, one, many and many reduced to one.
    It 'retains file results for <Case>' -ForEach @(
        @{ Case = 'zero'; Count = 0; Limit = 0; Expected = 0 }
        @{ Case = 'one'; Count = 1; Limit = 0; Expected = 1 }
        @{ Case = 'many'; Count = 3; Limit = 0; Expected = 3 }
        @{ Case = 'post-limit-one'; Count = 3; Limit = 1; Expected = 1 }
    ) {
        $directory = Join-Path $TestDrive ('files-' + $Case)
        [void][IO.Directory]::CreateDirectory($directory)
        [IO.File]::WriteAllText((Join-Path $directory 'excluded.log'), 'excluded')
        $expectedPaths = @(for ($i = 0; $i -lt $Count; $i++) {
            $path = Join-Path $directory ("match-$i.txt")
            [IO.File]::WriteAllText($path, 'data')
            $path
        })
        $before = @(Get-ChildItem -LiteralPath $directory -File | Sort-Object Name | ForEach-Object { "{0}|{1}|{2}" -f $_.Name, $_.Length, (Get-FileHash -LiteralPath $_.FullName).Hash })
        $result = Invoke-NativeSearch -Operation Files -Path $directory -Pattern '*.txt' -MaxResults $Limit -ErrorAction Stop
        $after = @(Get-ChildItem -LiteralPath $directory -File | Sort-Object Name | ForEach-Object { "{0}|{1}|{2}" -f $_.Name, $_.Length, (Get-FileHash -LiteralPath $_.FullName).Hash })
        $global:PCAI_StrictSearchOutcomesR1.Add([pscustomobject]@{ Operation = 'Files'; Case = $Case; Directory = $directory; InputBefore = $before; InputAfter = $after; Result = $result })
        ($after -join "`n") | Should -BeExactly ($before -join "`n")
        $result.Status | Should -Be 'Success'
        $result.Engine | Should -Be 'PowerShell'
        $result.FilesScanned | Should -Be $Expected
        $result.FilesMatched | Should -Be $Expected
        $result.TotalSize | Should -Be (4 * $Expected)
        $result.SearchPath | Should -Be (Get-Item -LiteralPath $directory).FullName
        $result.TotalDurationMs | Should -BeGreaterOrEqual 0
        if ($Expected -eq 0) { $result.Files | Should -BeNullOrEmpty }
        else {
            @($result.Files) | Should -HaveCount $Expected
            foreach ($file in $result.Files) {
                $expectedPaths | Should -Contain $file.Path
                $file.Size | Should -Be 4
            }
        }
        Should -Invoke Invoke-PcaiNativeFileSearch -Exactly -Times 0
    }

    # Protects real content matching and filtering. Detects one-file discovery
    # while preserving the independent match limit and exact returned bytes.
    It 'retains content results for <Case>' -ForEach @(
        @{ Case = 'zero'; Count = 0; Limit = 0; Expected = 0; MatchedFiles = 0 }
        @{ Case = 'one'; Count = 1; Limit = 0; Expected = 2; MatchedFiles = 1 }
        @{ Case = 'many'; Count = 3; Limit = 0; Expected = 6; MatchedFiles = 3 }
        @{ Case = 'post-limit-one'; Count = 3; Limit = 1; Expected = 1; MatchedFiles = 1 }
    ) {
        $directory = Join-Path $TestDrive ('content-' + $Case)
        [void][IO.Directory]::CreateDirectory($directory)
        [IO.File]::WriteAllText((Join-Path $directory 'excluded.txt'), 'error excluded')
        $expectedPaths = @(for ($i = 0; $i -lt $Count; $i++) {
            $path = Join-Path $directory ("match-$i.log")
            [IO.File]::WriteAllLines($path, @('safe', 'error one', 'error two'))
            $path
        })
        $before = @(Get-ChildItem -LiteralPath $directory -File | Sort-Object Name | ForEach-Object { "{0}|{1}|{2}" -f $_.Name, $_.Length, (Get-FileHash -LiteralPath $_.FullName).Hash })
        $result = Invoke-NativeSearch -Operation Content -Path $directory -Pattern 'error' -FilePattern '*.log' -MaxResults $Limit -ErrorAction Stop
        $after = @(Get-ChildItem -LiteralPath $directory -File | Sort-Object Name | ForEach-Object { "{0}|{1}|{2}" -f $_.Name, $_.Length, (Get-FileHash -LiteralPath $_.FullName).Hash })
        $global:PCAI_StrictSearchOutcomesR1.Add([pscustomobject]@{ Operation = 'Content'; Case = $Case; Directory = $directory; InputBefore = $before; InputAfter = $after; Result = $result })
        ($after -join "`n") | Should -BeExactly ($before -join "`n")
        $result.Status | Should -Be 'Success'
        $result.Engine | Should -Be 'PowerShell'
        $result.FilesScanned | Should -Be $Count
        $result.FilesMatched | Should -Be $MatchedFiles
        $result.TotalMatches | Should -Be $Expected
        $result.SearchPath | Should -Be (Get-Item -LiteralPath $directory).FullName
        @($result.Matches) | Should -HaveCount $Expected
        foreach ($match in $result.Matches) {
            $expectedPaths | Should -Contain $match.Path
            $match.LineNumber | Should -BeIn @(2, 3)
            $match.Line | Should -BeIn @('error one', 'error two')
        }
        if ($Limit -eq 1) {
            $result.Matches[0].LineNumber | Should -Be 2
            $result.Matches[0].Line | Should -Be 'error one'
        }
        Should -Invoke Invoke-PcaiNativeContentSearch -Exactly -Times 0
    }
}
