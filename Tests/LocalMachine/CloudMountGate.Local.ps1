#Requires -Version 7.3
#Requires -Modules Pester
# Explicit local qualification; excluded from ordinary *.Tests.ps1 discovery.
# Test IDs/assertions are preserved from the original r5 165-test receipt.
BeforeAll {
    $script:Tool = Join-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) 'Tools/SystemScripts/Networking/Start-PcaiCloudMount.ps1'
    . $script:Tool
}
Describe 'Bounded actual Storage worker' {
    It 'executes the real worker and shared helper against this attached cache' {
        $state = Get-PcaiCloudMountStorage -ImagePath 'T:\vm\cloud-cache-disk.vhdx' -BudgetMilliseconds 30000
        $state.State | Should -Be Healthy
        $state.CacheGuid | Should -Be 'c3e0daf0-3678-4842-8d9c-66aa3619b11a'
        $state.BackingGuid | Should -Be 'b9f25542-491b-4718-8556-4cbe67e599b8'
        [uint64]$state.CacheFree | Should -BeGreaterThan 100GB
        [uint64]$state.BackingFree | Should -BeGreaterThan 140GB
    }
    It 'bounds an owned worker that cannot complete inside a one millisecond budget' {
        $timer = [Diagnostics.Stopwatch]::StartNew()
        $state = Get-PcaiCloudMountStorage -ImagePath 'T:\vm\cloud-cache-disk.vhdx' -BudgetMilliseconds 1
        $state.State | Should -Be Pending
        $state.Reason | Should -Be 'Storage probe deadline expired.'
        $timer.Elapsed.TotalSeconds | Should -BeLessThan 5
    }
}
