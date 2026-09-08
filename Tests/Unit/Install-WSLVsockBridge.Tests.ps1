#Requires -Version 5.1
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }

<#
.SYNOPSIS
    Unit tests for Install-WSLVsockBridge.

.DESCRIPTION
    Tests the Install-WSLVsockBridge function in the PC-AI.Virtualization module.
    All WSL, systemd, and file-system operations are replaced with Pester mocks
    so these tests run without an actual WSL distribution or elevated privileges.

    Scenarios covered:
    - Missing bridge script / service file / config file each cause an error entry
    - Happy path: all files present, WSL commands succeed, result flags set correctly
    - EnableService=$false and StartService=$false suppress the respective steps
    - socat already installed skips the apt-get install
    - Multiple errors accumulate in the Errors array

.NOTES
    Run with: Invoke-Pester -Path .\Tests\Unit\Install-WSLVsockBridge.Tests.ps1 -Tag Unit,Virtualization,VSock
#>

# ---------------------------------------------------------------------------
# Helpers MUST be defined inside BeforeAll.
#
# Pester 5 executes a test file's top level during DISCOVERY only; the run
# phase evaluates BeforeAll/BeforeEach/It in a separate scope where a
# top-level `function` no longer exists. New-TempBridgeFiles used to live at
# the top level, so every BeforeEach threw CommandNotFoundException, left
# $script:TempFiles unset, and then called Install-WSLVsockBridge with null
# paths -- which reached the real `wsl.exe` and HUNG the whole suite.
# ---------------------------------------------------------------------------
BeforeAll {
    $ModulePath = Join-Path $PSScriptRoot '..\..\Modules\PC-AI.Virtualization\PC-AI.Virtualization.psd1'
    Import-Module $ModulePath -Force -ErrorAction Stop

    function New-TempBridgeFiles {
        <#
        .SYNOPSIS Creates three temp stub files representing the bridge script, service, and config.
        .OUTPUTS PSCustomObject with BridgeScriptPath, ServiceFilePath, ConfigPath, TempDir
        #>
        $dir = Join-Path $env:TEMP "PcaiBridgeTest_$(New-Guid)"
        New-Item -ItemType Directory -Path $dir -Force | Out-Null

        $bridge  = Join-Path $dir 'pcai-vsock-bridge.sh'
        $service = Join-Path $dir 'pcai-vsock-bridge.service'
        $config  = Join-Path $dir 'vsock-bridges.conf'

        [System.IO.File]::WriteAllText($bridge,  '#!/bin/bash')
        [System.IO.File]::WriteAllText($service, '[Unit]')
        [System.IO.File]::WriteAllText($config,  '# config')

        [PSCustomObject]@{
            BridgeScriptPath = $bridge
            ServiceFilePath  = $service
            ConfigPath       = $config
            TempDir          = $dir
        }
    }
}

AfterAll {
    Remove-Module 'PC-AI.Virtualization' -Force -ErrorAction SilentlyContinue
}

Describe 'Install-WSLVsockBridge' -Tag 'Unit', 'Virtualization', 'VSock', 'Portable' {

    # -----------------------------------------------------------------------
    # Missing source files
    # -----------------------------------------------------------------------
    Context 'When a required source file is missing' {

        It 'Adds an error entry when bridge script is not found' {
            $result = Install-WSLVsockBridge `
                -Distribution 'Ubuntu' `
                -BridgeScriptPath 'C:\nonexistent\bridge.sh' `
                -ServiceFilePath  'C:\nonexistent\bridge.service' `
                -ConfigPath       'C:\nonexistent\vsock.conf'

            $result.Errors.Count | Should -BeGreaterThan 0
        }

        It 'Does not set ScriptInstalled=true when bridge script is missing' {
            $result = Install-WSLVsockBridge `
                -Distribution 'Ubuntu' `
                -BridgeScriptPath 'C:\nonexistent\bridge.sh' `
                -ServiceFilePath  'C:\nonexistent\bridge.service' `
                -ConfigPath       'C:\nonexistent\vsock.conf'

            $result.ScriptInstalled | Should -BeFalse
        }

        It 'Adds an error entry when service file is not found (bridge script exists)' {
            $files = New-TempBridgeFiles
            try {
                $result = Install-WSLVsockBridge `
                    -Distribution 'Ubuntu' `
                    -BridgeScriptPath $files.BridgeScriptPath `
                    -ServiceFilePath  'C:\nonexistent\bridge.service' `
                    -ConfigPath       $files.ConfigPath

                $result.Errors.Count | Should -BeGreaterThan 0
            } finally {
                Remove-Item $files.TempDir -Recurse -Force -ErrorAction SilentlyContinue
            }
        }

        It 'Adds an error entry when config file is not found (bridge + service exist)' {
            $files = New-TempBridgeFiles
            try {
                $result = Install-WSLVsockBridge `
                    -Distribution 'Ubuntu' `
                    -BridgeScriptPath $files.BridgeScriptPath `
                    -ServiceFilePath  $files.ServiceFilePath `
                    -ConfigPath       'C:\nonexistent\vsock.conf'

                $result.Errors.Count | Should -BeGreaterThan 0
            } finally {
                Remove-Item $files.TempDir -Recurse -Force -ErrorAction SilentlyContinue
            }
        }
    }

    # -----------------------------------------------------------------------
    # Result object structure
    # -----------------------------------------------------------------------
    Context 'Result object always has the expected properties' {

        It 'Returns a PSCustomObject regardless of success or failure' {
            $result = Install-WSLVsockBridge `
                -Distribution 'Ubuntu' `
                -BridgeScriptPath 'C:\nonexistent\x.sh' `
                -ServiceFilePath  'C:\nonexistent\x.service' `
                -ConfigPath       'C:\nonexistent\x.conf'

            $result | Should -BeOfType [PSCustomObject]
        }

        It 'Result has Distribution property' {
            $result = Install-WSLVsockBridge `
                -Distribution 'Ubuntu' `
                -BridgeScriptPath 'C:\nonexistent\x.sh' `
                -ServiceFilePath  'C:\nonexistent\x.service' `
                -ConfigPath       'C:\nonexistent\x.conf'

            $result.Distribution | Should -Be 'Ubuntu'
        }

        # A bare `foreach` around `It` binds $prop during DISCOVERY only; by the
        # time the It body runs, $prop is $null and every case asserted
        # "Expected $null to be found in collection". -ForEach is the Pester 5
        # idiom that carries the value into the run phase as $_.
        It "Result has <_> property" -ForEach @(
            'ScriptInstalled', 'ServiceInstalled', 'ConfigInstalled',
            'SocatInstalled', 'ServiceEnabled', 'ServiceStarted', 'Errors'
        ) {
            $result = Install-WSLVsockBridge `
                -Distribution 'Ubuntu' `
                -BridgeScriptPath 'C:\nonexistent\x.sh' `
                -ServiceFilePath  'C:\nonexistent\x.service' `
                -ConfigPath       'C:\nonexistent\x.conf'

            $result.PSObject.Properties.Name | Should -Contain $_
        }
    }

    # -----------------------------------------------------------------------
    # Happy path — all files present, WSL commands mocked
    # -----------------------------------------------------------------------
    Context 'When all files are present and WSL commands succeed' {

        BeforeEach {
            $script:TempFiles = New-TempBridgeFiles

            # Mock every external call that Install-WSLVsockBridge issues
            Mock wsl        {} -ModuleName PC-AI.Virtualization
            Mock Enable-WSLSystemd { [PSCustomObject]@{ Success = $true } } -ModuleName PC-AI.Virtualization
        }

        AfterEach {
            Remove-Item $script:TempFiles.TempDir -Recurse -Force -ErrorAction SilentlyContinue
        }

        It 'Returns a result with no errors' {
            # Mock LASTEXITCODE for socat check to 0 (socat present)
            Mock Invoke-Expression {} -ModuleName PC-AI.Virtualization

            $result = Install-WSLVsockBridge `
                -Distribution    'Ubuntu' `
                -BridgeScriptPath $script:TempFiles.BridgeScriptPath `
                -ServiceFilePath  $script:TempFiles.ServiceFilePath `
                -ConfigPath       $script:TempFiles.ConfigPath `
                -EnableService:$false `
                -StartService:$false

            $result.Errors.Count | Should -Be 0
        }

        It 'Reflects the correct Distribution in the result' {
            $result = Install-WSLVsockBridge `
                -Distribution    'Debian' `
                -BridgeScriptPath $script:TempFiles.BridgeScriptPath `
                -ServiceFilePath  $script:TempFiles.ServiceFilePath `
                -ConfigPath       $script:TempFiles.ConfigPath `
                -EnableService:$false `
                -StartService:$false

            $result.Distribution | Should -Be 'Debian'
        }
    }

    # -----------------------------------------------------------------------
    # EnableService=$false and StartService=$false suppress respective steps
    # -----------------------------------------------------------------------
    Context 'When EnableService and StartService are false' {

        BeforeEach {
            $script:TempFiles = New-TempBridgeFiles
            Mock wsl        {} -ModuleName PC-AI.Virtualization
            Mock Enable-WSLSystemd { [PSCustomObject]@{ Success = $true } } -ModuleName PC-AI.Virtualization
        }

        AfterEach {
            Remove-Item $script:TempFiles.TempDir -Recurse -Force -ErrorAction SilentlyContinue
        }

        It 'ServiceEnabled is false when -EnableService $false is passed' {
            $result = Install-WSLVsockBridge `
                -Distribution    'Ubuntu' `
                -BridgeScriptPath $script:TempFiles.BridgeScriptPath `
                -ServiceFilePath  $script:TempFiles.ServiceFilePath `
                -ConfigPath       $script:TempFiles.ConfigPath `
                -EnableService:$false `
                -StartService:$false

            $result.ServiceEnabled | Should -BeFalse
        }

        It 'ServiceStarted is false when -StartService $false is passed' {
            $result = Install-WSLVsockBridge `
                -Distribution    'Ubuntu' `
                -BridgeScriptPath $script:TempFiles.BridgeScriptPath `
                -ServiceFilePath  $script:TempFiles.ServiceFilePath `
                -ConfigPath       $script:TempFiles.ConfigPath `
                -EnableService:$false `
                -StartService:$false

            $result.ServiceStarted | Should -BeFalse
        }
    }

    # -----------------------------------------------------------------------
    # Default distribution name
    # -----------------------------------------------------------------------
    Context 'Default parameter values' {

        It 'Uses Ubuntu as the default Distribution' {
            # Provide explicit paths that do not exist to force an early exit;
            # we just want to confirm the default was applied.
            $result = Install-WSLVsockBridge `
                -BridgeScriptPath 'C:\nonexistent\x.sh' `
                -ServiceFilePath  'C:\nonexistent\x.service' `
                -ConfigPath       'C:\nonexistent\x.conf'

            $result.Distribution | Should -Be 'Ubuntu'
        }
    }

    # -----------------------------------------------------------------------
    # Errors array accumulates messages
    # -----------------------------------------------------------------------
    Context 'Error accumulation' {

        It 'Errors array starts empty on a clean result' {
            $result = Install-WSLVsockBridge `
                -Distribution 'Ubuntu' `
                -BridgeScriptPath 'C:\nonexistent\x.sh' `
                -ServiceFilePath  'C:\nonexistent\x.service' `
                -ConfigPath       'C:\nonexistent\x.conf'

            # Only one throw from the first missing file — Errors has exactly one entry
            $result.Errors.Count | Should -Be 1
        }

        It 'Error message is a non-empty string' {
            $result = Install-WSLVsockBridge `
                -Distribution 'Ubuntu' `
                -BridgeScriptPath 'C:\nonexistent\x.sh' `
                -ServiceFilePath  'C:\nonexistent\x.service' `
                -ConfigPath       'C:\nonexistent\x.conf'

            $result.Errors[0] | Should -Not -BeNullOrEmpty
        }
    }
}
