<#
.SYNOPSIS
    Unit tests for the TUI launcher wrapper.
#>

BeforeAll {
    $ModulePath = Join-Path $PSScriptRoot '..\..\Modules\PC-AI.LLM\PC-AI.LLM.psd1'
    Import-Module $ModulePath -Force -ErrorAction Stop
    $script:PrimaryTuiCandidate = 'C:\__pcai_mock\PcaiChatTui.exe'
}

Describe "Invoke-LLMChatTui" -Tag 'Unit', 'LLM', 'TUI' {
    AfterEach {
        Remove-Item -Path "Function:global:$script:PrimaryTuiCandidate" -ErrorAction SilentlyContinue
        Remove-Item Env:PCAI_TUI_EXE -ErrorAction SilentlyContinue
        $global:CapturedTuiArgs = $null
    }

    It "Should throw when the TUI executable is missing" {
        Mock Test-Path { $false } -ModuleName PC-AI.LLM
        { Invoke-LLMChatTui -ErrorAction Stop } | Should -Throw -ExpectedMessage '*PcaiChatTui.exe not found*'
    }

    It "Should invoke the first discovered TUI candidate with forwarded arguments" {
        $global:CapturedTuiArgs = $null
        $env:PCAI_TUI_EXE = $script:PrimaryTuiCandidate
        Set-Item -Path "Function:global:$script:PrimaryTuiCandidate" -Value {
            $global:CapturedTuiArgs = @($args)
            return 'invoked'
        }

        Mock Test-Path -ModuleName PC-AI.LLM -MockWith {
            param([string]$Path)
            return $Path -eq $script:PrimaryTuiCandidate
        }

        { Invoke-LLMChatTui -Arguments @('--provider', 'pcai-inference') -ProjectRoot $PSScriptRoot -ErrorAction Stop } | Should -Not -Throw
        $global:CapturedTuiArgs | Should -Be @('--provider', 'pcai-inference')
        Assert-MockCalled Test-Path -ModuleName PC-AI.LLM -Times 1 -ParameterFilter { $Path -eq $script:PrimaryTuiCandidate }
    }
}

AfterAll {
    Remove-Module PC-AI.LLM -Force -ErrorAction SilentlyContinue
}
