#Requires -Version 7.0
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }

# Jules PRs are opened under the repository owner's account; the only stable
# marker is a commit authored by google-labs-jules[bot]. `--author jules[bot]`
# never matched a single PR, so the dashboard always reported none.
BeforeAll {
    $script:ScriptPath = Join-Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) 'Tools' 'Get-JulesPRStatus.ps1'

    function New-PrFixture([int]$Number, [string[]]$CommitLogins) {
        [PSCustomObject]@{
            number = $Number; title = "PR $Number"; state = 'OPEN'; mergeable = 'MERGEABLE'
            statusCheckRollup = @([PSCustomObject]@{ conclusion = 'SUCCESS'; status = 'COMPLETED' })
            changedFiles = 1; headRefName = "branch-$Number"; createdAt = '2026-09-26T00:00:00Z'
            url = "https://example.test/pull/$Number"
            commits = @($CommitLogins | ForEach-Object { [PSCustomObject]@{ authors = @([PSCustomObject]@{ login = $_ }) } })
        }
    }
}

Describe 'Get-JulesPRStatus' -Tag 'Unit', 'Portable' {
    AfterEach { Remove-Item function:global:gh -ErrorAction SilentlyContinue }

    It 'selects PRs by Jules commit authorship, not PR author' {
        $global:JulesPrFixture = @(
            (New-PrFixture 1 @('David-Martel', 'google-labs-jules[bot]')),
            (New-PrFixture 2 @('David-Martel')),
            (New-PrFixture 3 @('dependabot[bot]'))
        ) | ConvertTo-Json -Depth 6
        function global:gh { $global:LASTEXITCODE = 0; $global:JulesPrFixture }
        $out = & $script:ScriptPath -Format Json | ConvertFrom-Json -NoEnumerate
        @($out).Count | Should -Be 1
        $out[0].PR | Should -Be 1
    }

    It 'does not pass the never-matching --author filter to gh' {
        (Get-Content -Raw $script:ScriptPath) | Should -Not -Match "--author 'jules\[bot\]'"
    }

    It 'emits [] in Json mode when nothing matches' {
        $global:JulesPrFixture = @((New-PrFixture 2 @('David-Martel'))) | ConvertTo-Json -Depth 6 -AsArray
        function global:gh { $global:LASTEXITCODE = 0; $global:JulesPrFixture }
        (& $script:ScriptPath -Format Json | Out-String).Trim() | Should -Be '[]'
    }

    It 'keeps a single match as a JSON array' {
        $global:JulesPrFixture = @((New-PrFixture 7 @('google-labs-jules[bot]'))) | ConvertTo-Json -Depth 6 -AsArray
        function global:gh { $global:LASTEXITCODE = 0; $global:JulesPrFixture }
        (& $script:ScriptPath -Format Json | Out-String).Trim() | Should -Match '^\['
    }
}

Describe 'Get-JulesPRStatus limits' -Tag 'Unit', 'Portable' {
    It 'caps -Limit at 40 to stay under the 500,000-node GitHub GraphQL limit' {
        $range = (Get-Command $script:ScriptPath).Parameters['Limit'].Attributes |
            Where-Object { $_ -is [System.Management.Automation.ValidateRangeAttribute] }
        $range.MaxRange | Should -Be 40
    }
}
