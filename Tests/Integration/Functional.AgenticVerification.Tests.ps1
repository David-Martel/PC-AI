. (Join-Path $PSScriptRoot '..\Helpers\Resolve-TestRepoRoot.ps1')

Describe "PC-AI Agentic Verification (Phase 6)" {
    BeforeAll {
        $script:PcaiRoot = Resolve-TestRepoRoot -StartPath $PSScriptRoot
        $script:ModulePath = Join-Path $script:PcaiRoot "Modules\PC-AI.LLM\PC-AI.LLM.psd1"

        # Load dependencies first
        Import-Module (Join-Path $script:PcaiRoot "Modules\PC-AI.Acceleration\PC-AI.Acceleration.psm1") -Force
        Import-Module (Join-Path $script:PcaiRoot "Modules\PC-AI.Virtualization\PC-AI.Virtualization.psd1") -Force
        Import-Module $script:ModulePath -Force
    }

    Context "Diagnostic Triage" {
        It "Should invoke Triage mode when report is oversized" {
            $script:capturedMessages = $null

            Mock Invoke-FunctionGemmaReAct {
                return [PSCustomObject]@{
                    ToolResults = @(
                        [PSCustomObject]@{
                            Tool = "Get-UsbDeviceList"
                            Result = [PSCustomObject]@{
                                Devices = @(
                                    [PSCustomObject]@{
                                        DeviceID = "USB\\VID_1234&PID_5678\\GARY_BROKEN"
                                        Status = "Error"
                                    }
                                )
                            }
                        }
                    )
                }
            } -ModuleName PC-AI.LLM

            Mock Invoke-LLMChatWithFallback {
                param([object[]]$Messages)
                $script:capturedMessages = $Messages
                return @{
                    message = @{
                        content = '{"summary":"triage needed","issues":[{"id":"USB-1","severity":"High"}]}'
                    }
                }
            } -ModuleName PC-AI.LLM

            $largeReport = [PSCustomObject]@{
                ReportId = "TEST-123"
                RawData = "A" * 100000 # Large enough to trigger triage if mocked/config'd
                Sections = @("USB", "Network", "Performance")
            }

            $result = Invoke-PCDiagnosis -ReportText ($largeReport | ConvertTo-Json -Depth 4) -UseRouter -RouterExecuteTools -TimeoutSeconds 30

            $result.JsonValid | Should -Be $true
            $result.AnalysisJson | Should -Not -BeNullOrEmpty
            $result.AnalysisJson.summary | Should -Be "triage needed"

            $script:capturedMessages | Should -Not -BeNullOrEmpty
            $script:capturedMessages[1].content | Should -Match "\[TOOL_RESULTS\]"

            Should -Invoke Invoke-FunctionGemmaReAct -ModuleName PC-AI.LLM
            Should -Invoke Invoke-LLMChatWithFallback -ModuleName PC-AI.LLM
        }
    }
}
