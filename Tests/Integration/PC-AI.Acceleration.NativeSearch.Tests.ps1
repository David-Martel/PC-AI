#Requires -Version 7.0
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }

BeforeAll {
    $script:ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    $script:AccelerationModulePath = Join-Path $script:ProjectRoot 'Modules\PC-AI.Acceleration\PC-AI.Acceleration.psd1'
    Import-Module $script:AccelerationModulePath -Force -ErrorAction Stop

    $script:FixtureRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("pcai-native-search-" + [guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $script:FixtureRoot -Force | Out-Null
    New-Item -ItemType Directory -Path (Join-Path $script:FixtureRoot 'subdir') -Force | Out-Null

    Set-Content -Path (Join-Path $script:FixtureRoot 'file1.txt') -Value "Hello world`nThis is a test file`nHello again" -Encoding UTF8
    Set-Content -Path (Join-Path $script:FixtureRoot 'file2.log') -Value "Warning: ignore this`nError: check compact path" -Encoding UTF8
    Set-Content -Path (Join-Path $script:FixtureRoot 'subdir\nested.txt') -Value "Hello from nested directory" -Encoding UTF8
}

AfterAll {
    Remove-Item -Path $script:FixtureRoot -Recurse -Force -ErrorAction SilentlyContinue
}

Describe 'PC-AI.Acceleration native search integration' -Tag 'Integration', 'Acceleration', 'NativeSearch' {
    BeforeAll {
        $script:NativeSearchAvailable = Test-PcaiNativeAvailable
    }

    It 'returns compact native file-search results through the C# bridge' {
        if (-not $script:NativeSearchAvailable) {
            Set-ItResult -Skipped -Because 'PCAI native search is unavailable on this machine'
        }

        $result = Invoke-PcaiNativeFileSearch -Pattern '*.txt' -Path $script:FixtureRoot -MaxResults 10

        $result | Should -Not -BeNullOrEmpty
        $result.Status | Should -Be 'Success'
        $result.FilesMatched | Should -BeGreaterOrEqual 2
        @($result.Files).Count | Should -BeGreaterOrEqual 2
        @($result.Files | Select-Object -ExpandProperty Path) | Should -Contain (Join-Path $script:FixtureRoot 'file1.txt')
        @($result.Files | Select-Object -ExpandProperty Path) | Should -Contain (Join-Path $script:FixtureRoot 'subdir\nested.txt')
    }

    It 'returns native content-search matches through Search-ContentFast without JSON parsing failures' {
        if (-not $script:NativeSearchAvailable) {
            Set-ItResult -Skipped -Because 'PCAI native search is unavailable on this machine'
        }

        $results = @(Search-ContentFast -Path $script:FixtureRoot -Pattern 'Hello' -FilePattern '*.txt' -MaxResults 10)

        $results.Count | Should -BeGreaterOrEqual 2
        ($results | Select-Object -ExpandProperty Tool -Unique) | Should -Be @('pcai_native')
        @($results | Select-Object -ExpandProperty Path) | Should -Contain (Join-Path $script:FixtureRoot 'file1.txt')
        @($results | Select-Object -ExpandProperty Path) | Should -Contain (Join-Path $script:FixtureRoot 'subdir\nested.txt')
        @($results | Select-Object -ExpandProperty Line) | Should -Contain 'Hello world'
        @($results | Select-Object -ExpandProperty Line) | Should -Contain 'Hello again'
    }

    It 'matches the Select-String hit set for the same fixture and pattern' {
        if (-not $script:NativeSearchAvailable) {
            Set-ItResult -Skipped -Because 'PCAI native search is unavailable on this machine'
        }

        $native = @(Search-ContentFast -Path $script:FixtureRoot -Pattern 'Hello' -FilePattern '*.txt' -MaxResults 10) |
            ForEach-Object { '{0}|{1}|{2}' -f $_.Path, $_.LineNumber, $_.Line } |
            Sort-Object

        $baseline = Get-ChildItem -Path $script:FixtureRoot -Recurse -File -Filter '*.txt' |
            Select-String -Pattern 'Hello' |
            ForEach-Object { '{0}|{1}|{2}' -f $_.Path, $_.LineNumber, $_.Line } |
            Sort-Object

        $native | Should -Be $baseline
    }
}
