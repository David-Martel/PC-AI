<#
.SYNOPSIS
    Integration tests for FunctionGemma router execution with tool calls.
#>

BeforeAll {
    $script:PcaiRoot = Join-Path $PSScriptRoot '..\..'
    $script:ModulePath = Join-Path $script:PcaiRoot 'Modules\PC-AI.LLM\PC-AI.LLM.psd1'
    $script:VirtPath = Join-Path $script:PcaiRoot 'Modules\PC-AI.Virtualization\PC-AI.Virtualization.psd1'
    $script:ToolsPath = Join-Path $script:PcaiRoot 'Config\pcai-tools.json'

    Import-Module $script:VirtPath -Force -ErrorAction Stop
    Import-Module $script:ModulePath -Force -ErrorAction Stop

    if (-not ([System.Management.Automation.PSTypeName]'PcaiNative.PcaiCore').Type) {
        Add-Type -TypeDefinition @"
namespace PcaiNative {
    public static class PcaiCore { public static bool IsAvailable => false; }
}
"@
    }
}

Describe "Invoke-FunctionGemmaReAct tool execution" -Tag 'Integration', 'Router', 'FunctionGemma' {
    BeforeEach {
        Mock Invoke-RestMethod {
            param($Uri)
            if ($Uri -match '/v1/chat/completions') {
                return [PSCustomObject]@{
                    choices = @(
                        [PSCustomObject]@{
                            message = [PSCustomObject]@{
                                tool_calls = @(
                                    [PSCustomObject]@{
                                        id = 'call_1'
                                        function = [PSCustomObject]@{
                                            name = 'pcai_get_docker_status'
                                            arguments = [hashtable]@{}
                                        }
                                    }
                                )
                            }
                        }
                    )
                }
            }
            return @{}
        } -ModuleName PC-AI.LLM

        Mock Get-DockerStatus {
            [PSCustomObject]@{ Installed = $true; Running = $true; Version = '27.0.0'; Backend = 'WSL2'; Severity = 'OK' }
        } -ModuleName PC-AI.Virtualization
    }

    It "Should execute docker status tool call from FunctionGemma" {
        # Skip when FunctionGemma router is not running
        $fgUp = try { $null = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/health' -TimeoutSec 2; $true } catch { $false }
        if (-not $fgUp) { Set-ItResult -Skipped -Because 'FunctionGemma router not running at localhost:8000'; return }

        $result = Invoke-FunctionGemmaReAct -Prompt "Check Docker status" -ExecuteTools -ToolsPath $script:ToolsPath -MaxToolCalls 1

        $result.ToolCalls.Count | Should -Be 1
        $result.ToolCalls[0].function.name | Should -Be 'pcai_get_docker_status'
        $result.ToolResults.Count | Should -Be 1
        $result.ToolResults[0].name | Should -Be 'pcai_get_docker_status'
        $result.ToolResults[0].result | Should -Match '"Running"'
    }
}

AfterAll {
    Remove-Module PC-AI.LLM -Force -ErrorAction SilentlyContinue
    Remove-Module PC-AI.Virtualization -Force -ErrorAction SilentlyContinue
}
