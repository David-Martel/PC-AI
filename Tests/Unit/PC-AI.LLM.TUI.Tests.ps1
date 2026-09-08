<#
.SYNOPSIS
    Unit tests for the TUI launcher wrapper.
#>

BeforeAll {
    $ModulePath = Join-Path $PSScriptRoot '..\..\Modules\PC-AI.LLM\PC-AI.LLM.psd1'
    Import-Module $ModulePath -Force -ErrorAction Stop
    $script:PrimaryTuiCandidate = 'C:\__pcai_mock\PcaiChatTui.exe'
}

Describe "Invoke-LLMChatTui" -Tag 'Unit', 'LLM', 'TUI', 'Portable' {
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

        # This test could never pass. It pointed at C:\__pcai_mock\PcaiChatTui.exe
        # and set $env:PCAI_TUI_EXE, but Invoke-LLMChatTui has no such override --
        # it builds a fixed candidate list from -ProjectRoot and takes
        # `Select-Object -First 1`. A path outside that list is never selected, so
        # $exe stayed $null and the function always threw "not found".
        #
        # Target the FIRST real candidate instead, which is what the test name
        # claims to check. This still fails if the discovery order changes: `& $exe`
        # would then resolve a name no function is defined under.
        $root = Join-Path $env:TEMP 'pcai_tui_test_root'
        $script:PrimaryTuiCandidate = Join-Path $root '.pcai\build\artifacts\pcai-chattui\PcaiChatTui.exe'

        Set-Item -Path "Function:global:$script:PrimaryTuiCandidate" -Value {
            $global:CapturedTuiArgs = @($args)
            return 'invoked'
        }

        # Deliberately unfiltered. A -ParameterFilter referencing a test-scope
        # variable is evaluated in the MODULE's script scope, where it is $null --
        # every comparison would be false and Test-Path would answer $false for
        # everything. With all candidates "present", First-1 still proves ordering.
        Mock Test-Path -ModuleName PC-AI.LLM -MockWith { $true }

        { Invoke-LLMChatTui -Arguments @('--provider', 'pcai-inference') -ProjectRoot $root -ErrorAction Stop } | Should -Not -Throw
        $global:CapturedTuiArgs | Should -Be @('--provider', 'pcai-inference')
        Assert-MockCalled Test-Path -ModuleName PC-AI.LLM -Times 1 -Scope It
    }
}

AfterAll {
    Remove-Module PC-AI.LLM -Force -ErrorAction SilentlyContinue
}
