#Requires -Version 7.0
#Requires -Modules Pester
<#
.SYNOPSIS
    Pester tests for Optimize-PathCompression.
.DESCRIPTION
    Hermetic tests that operate on a test registry key under HKCU\Software\PC-AI-Test\Environment
    rather than touching the real PATH. The function under test targets the real registry paths,
    so these tests exercise the pure functions (normalization, substitution, CUDA detection,
    agent-ephemeral detection) via dot-sourcing the script file directly. For end-to-end
    registry behavior, use integration tests.
#>

BeforeAll {
    $modulePath = Join-Path $PSScriptRoot '..\..\..\Modules\PC-AI.Cleanup\PC-AI.Cleanup.psd1'
    Import-Module $modulePath -Force -ErrorAction Stop

    # Dot-source the script so we can call the nested helper functions directly
    $scriptPath = Join-Path $PSScriptRoot '..\..\..\Modules\PC-AI.Cleanup\Public\Optimize-PathCompression.ps1'

    # Parse the file and extract the helper function bodies for direct testing
    # (they're defined inside the begin{} block and not exposed)
    $src = Get-Content $scriptPath -Raw
    $ast = [System.Management.Automation.Language.Parser]::ParseInput($src, [ref]$null, [ref]$null)
    $helpers = $ast.FindAll({
        param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
        $n.Name -in @('Normalize-Path', 'Test-IsUserPath', 'Test-IsAgentEphemeral', 'Test-IsCudaVersionPath', 'Get-CudaVersionFromPath', 'Invoke-Substitute')
    }, $true)
    foreach ($h in $helpers) {
        # Dot-source a scriptblock built from the extracted function text so the
        # helper definition lands in the current scope. Preferred over
        # Invoke-Expression (flagged by PSAvoidUsingInvokeExpression) and
        # functionally equivalent for function-definition text.
        . ([scriptblock]::Create($h.Extent.Text))
    }
}

Describe 'Optimize-PathCompression module loading' {
    It 'exports Optimize-PathCompression from PC-AI.Cleanup' {
        Get-Command Optimize-PathCompression -Module PC-AI.Cleanup | Should -Not -BeNullOrEmpty
    }

    It 'supports -WhatIf' {
        (Get-Command Optimize-PathCompression).Parameters.ContainsKey('WhatIf') | Should -BeTrue
    }

    It 'declares Target parameter with correct values' {
        $p = (Get-Command Optimize-PathCompression).Parameters['Target']
        $p.Attributes.ValidValues | Should -Contain 'User'
        $p.Attributes.ValidValues | Should -Contain 'Machine'
        $p.Attributes.ValidValues | Should -Contain 'Both'
    }
}

Describe 'Path normalization' {
    It 'strips trailing backslash' {
        Normalize-Path 'C:\Program Files\Git\' | Should -Be 'C:\Program Files\Git'
    }
    It 'strips trailing forward slash' {
        Normalize-Path 'C:/Program Files/Git/' | Should -Be 'C:\Program Files\Git'
    }
    It 'converts forward slashes to backslashes' {
        Normalize-Path 'C:/foo/bar' | Should -Be 'C:\foo\bar'
    }
    It 'strips leading/trailing whitespace' {
        Normalize-Path '  C:\Tools  ' | Should -Be 'C:\Tools'
    }
}

Describe 'Agent-ephemeral detection' {
    It 'detects Claude Code agent-home paths with UUID' {
        Test-IsAgentEphemeral 'C:\Users\david\AppData\Local\ClaudeCode\agent-homes\762a5aef-e846-41bf-811f-29285e5356a2\home\Bin' | Should -BeTrue
    }
    It 'ignores regular Claude Code paths' {
        Test-IsAgentEphemeral 'C:\Users\david\AppData\Local\ClaudeCode\cli\bin' | Should -BeFalse
    }
    It 'ignores unrelated paths' {
        Test-IsAgentEphemeral 'C:\Program Files\Git\cmd' | Should -BeFalse
    }
}

Describe 'User-path classification' {
    It 'classifies paths under C:\Users\<name>\ as user paths' {
        Test-IsUserPath 'C:\Users\david\.cargo\bin' | Should -BeTrue
    }
    It 'classifies %USERPROFILE%-prefixed paths as user paths' {
        Test-IsUserPath '%USERPROFILE%\tools' | Should -BeTrue
    }
    It 'classifies C:\Program Files as NOT a user path' {
        Test-IsUserPath 'C:\Program Files\Git\cmd' | Should -BeFalse
    }
    It 'classifies C:\Windows as NOT a user path' {
        Test-IsUserPath 'C:\Windows\System32' | Should -BeFalse
    }
}

Describe 'CUDA version path detection' {
    It 'detects CUDA v13.1 bin path' {
        Test-IsCudaVersionPath 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin' | Should -BeTrue
    }
    It 'extracts version 13.1' {
        Get-CudaVersionFromPath 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin' | Should -Be '13.1'
    }
    It 'extracts version 12.9' {
        Get-CudaVersionFromPath 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\libnvvp' | Should -Be '12.9'
    }
    It 'does not match cuDNN paths' {
        Test-IsCudaVersionPath 'C:\Program Files\NVIDIA\CUDNN\v9.8\bin' | Should -BeFalse
    }
}

Describe 'Variable substitution' {
    It 'substitutes longest prefix first' {
        $subs = @(
            [PSCustomObject]@{ Token = '%PROGRAMFILES%'; Literal = 'C:\Program Files' }
            [PSCustomObject]@{ Token = '%PROGRAMFILES(X86)%'; Literal = 'C:\Program Files (x86)' }
        ) | Sort-Object { $_.Literal.Length } -Descending

        Invoke-Substitute -Entry 'C:\Program Files (x86)\Nmap' -Substitutions $subs |
            Should -Be '%PROGRAMFILES(X86)%\Nmap'
        Invoke-Substitute -Entry 'C:\Program Files\Git\cmd' -Substitutions $subs |
            Should -Be '%PROGRAMFILES%\Git\cmd'
    }
    It 'leaves non-matching paths unchanged' {
        $subs = @([PSCustomObject]@{ Token = '%PROGRAMFILES%'; Literal = 'C:\Program Files' })
        Invoke-Substitute -Entry 'D:\Tools' -Substitutions $subs | Should -Be 'D:\Tools'
    }
    It 'handles CUDA_PATH substitution' {
        $subs = @(
            [PSCustomObject]@{ Token = '%CUDA_PATH%'; Literal = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1' }
            [PSCustomObject]@{ Token = '%PROGRAMFILES%'; Literal = 'C:\Program Files' }
        ) | Sort-Object { $_.Literal.Length } -Descending
        Invoke-Substitute -Entry 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin' -Substitutions $subs |
            Should -Be '%CUDA_PATH%\bin'
    }
}

Describe 'Function declared with ShouldProcess semantics' {
    It 'uses CmdletBinding(SupportsShouldProcess)' {
        $cmd = Get-Command Optimize-PathCompression
        $cmd.CmdletBinding | Should -BeTrue
        # All ShouldProcess cmdlets get WhatIf and Confirm parameters automatically
        $cmd.Parameters.ContainsKey('WhatIf') | Should -BeTrue
        $cmd.Parameters.ContainsKey('Confirm') | Should -BeTrue
    }
}
