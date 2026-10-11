#Requires -Version 7.0
BeforeAll {
    $script:RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    $PSModuleAutoLoadingPreference = 'None'
    . (Join-Path $script:RepoRoot 'Modules/PC-AI.LLM/Private/LLM-Helpers.ps1')
    function Get-PcaiResourceMetrics { throw 'Mock required' }
    $script:PromptRoot = Join-Path $TestDrive 'prompts'
    [IO.Directory]::CreateDirectory($script:PromptRoot) | Out-Null
    [IO.File]::WriteAllText((Join-Path $script:PromptRoot 'CHAT.md'), '[SYSTEM_RESOURCE_STATUS]')
}
Describe 'Telemetry prompt unknown measurements' {
    It 'preserves unknown CPU, memory, GPU and uptime rather than fabricated zero' {
        Mock Get-PcaiResourceMetrics { [pscustomobject]@{ CpuUsage=$null; MemoryUsage=$null; TotalMemory=$null; GpuUsage=$null; Uptime=$null } }
        $prompt = Get-EnrichedSystemPrompt -Mode chat -ProjectRoot $script:PromptRoot
        foreach ($label in @('CPU Usage', 'Memory Usage', 'GPU Usage', 'System Uptime')) {
            $prompt | Should -Match ([regex]::Escape($label) + ': n/a')
        }
        $prompt | Should -Not -Match '0%|0 GB'
    }
    It 'retains legitimately measured zero CPU and GPU readings' {
        Mock Get-PcaiResourceMetrics { [pscustomobject]@{ CpuUsage=0; MemoryUsage=0; TotalMemory=1GB; GpuUsage=0; Uptime=0 } }
        $prompt = Get-EnrichedSystemPrompt -Mode chat -ProjectRoot $script:PromptRoot
        $prompt | Should -Match 'CPU Usage: 0%'
        $prompt | Should -Match 'GPU Usage: 0%'
        $prompt | Should -Match 'Memory Usage: 0 GB / 1 GB'
    }
}
