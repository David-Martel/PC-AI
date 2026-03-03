#Requires -Version 5.1

BeforeAll {
    $resolveRootHelper = Join-Path $PSScriptRoot '..\Helpers\Resolve-TestRepoRoot.ps1'
    if (Test-Path $resolveRootHelper) {
        . $resolveRootHelper
        $PcaiRoot = Resolve-TestRepoRoot
    } else {
        $PcaiRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
    }

    # Ensure Acceleration module is imported to load native types
    $AccelPath = Join-Path $PcaiRoot 'Modules\\PC-AI.Acceleration\\PC-AI.Acceleration.psd1'
    if (Test-Path $AccelPath) {
        Import-Module $AccelPath -Force -ErrorAction Stop
    }

    $ModulePath = Join-Path $PcaiRoot 'Modules\\PC-AI.LLM\\PC-AI.LLM.psd1'
    Import-Module $ModulePath -Force -ErrorAction Stop
    $script:LlmModule = Get-Module PC-AI.LLM
}

Describe "PC-AI Prompt Enrichment" {
    It "Should load and enrich CHAT.md with telemetry" {
        $prompt = & $script:LlmModule { Get-EnrichedSystemPrompt -Mode 'chat' }
        $prompt | Should -Not -BeNullOrEmpty
        # Placeholder should have been replaced
        $prompt | Should -Not -Match '\[SYSTEM_RESOURCE_STATUS\]' # Escaped for regex
        $prompt | Should -Match 'CPU Usage:|Memory:'
    }

    It "Should load and enrich DIAGNOSE.md with logic and telemetry" {
        $prompt = & $script:LlmModule { Get-EnrichedSystemPrompt -Mode 'diagnose' }
        $prompt | Should -Not -BeNullOrEmpty
        $prompt | Should -Not -Match '\[SYSTEM_RESOURCE_STATUS\]'
        # The logic injection adds ## REASONING FRAMEWORK
        # If it's missing, maybe ProjectRoot in the function needs to be explicitly passed
        $prompt | Should -Match 'REASONING FRAMEWORK|TOOL INTERPRETATION HINTS'
    }

    It "Should correctly call a tool by name using the mapping" {
        $Tools = @(
            @{
                function = @{
                    name = "test_tool"
                    description = "A test tool"
                }
                pcai_mapping = @{
                    cmdlet = "Get-Date"
                    module = $null
                    params = @{
                        Format = "yyyy"
                    }
                }
            } | ConvertTo-Json | ConvertFrom-Json
        )

        $result = & $script:LlmModule {
            param($ToolsArg, $ModuleRootArg)
            Invoke-ToolByName -Name "test_tool" -Args @{} -Tools $ToolsArg -ModuleRoot $ModuleRootArg
        } $Tools $PcaiRoot

        $expected = (Get-Date -Format "yyyy") | ConvertTo-Json -Compress
        $result | Should -Be $expected
    }

    It "Should handle missing tools gracefully when collection is empty" {
        $result = & $script:LlmModule {
            param($ModuleRootArg)
            Invoke-ToolByName -Name "missing_tool" -Args @{} -Tools @() -ModuleRoot $ModuleRootArg
        } $PcaiRoot
        $result | Should -Match "Unhandled tool"
        $result | Should -Match "no tools provided"
    }
}
