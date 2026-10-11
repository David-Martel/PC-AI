#Requires -Version 7.0
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }

BeforeAll {
    $shim = Join-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) 'Config/PowerShell/Profile/Microsoft.PowerShell_profile.ps1'
    $keys = @('POWERSHELL_PROFILE_ROOT', 'POWERSHELL_MODULES_PATH', 'LOCALAPPDATA', 'PROFILEUTILITIES_PCAI_AUTO_PATH_OPTIMIZE', 'PS_SKIP_PROFILE_ACCELERATOR', 'PS_SKIP_PSMODULEPATH_OPTIMIZE')
}
Describe 'Profile shim honors the current consumer environment' {
    BeforeEach {
        $saved = @{}
        foreach ($key in $keys) { $saved[$key] = [Environment]::GetEnvironmentVariable($key, 'Process') }
        $canonical = Join-Path $TestDrive 'canonical'
        [void](New-Item -ItemType Directory -Path $canonical -Force)
        Set-Content -LiteralPath (Join-Path $canonical 'Microsoft.PowerShell_profile.ps1') -Value '$script:ShimFixtureLoaded = $true'
        $env:POWERSHELL_PROFILE_ROOT = $canonical
        $script:ShimFixtureLoaded = $false
    }
    AfterEach {
        foreach ($key in $keys) { [Environment]::SetEnvironmentVariable($key, $saved[$key], 'Process') }
    }
    It 'loads a consumer-selected profile and retains an explicit module root' {
        $env:POWERSHELL_MODULES_PATH = Join-Path $TestDrive 'consumer-modules'
        . $shim
        $script:ShimFixtureLoaded | Should -BeTrue
        $env:POWERSHELL_MODULES_PATH | Should -BeExactly (Join-Path $TestDrive 'consumer-modules')
    }
    It 'selects the current users local module root when no override is provided' {
        $env:POWERSHELL_MODULES_PATH = $null
        $env:LOCALAPPDATA = Join-Path $TestDrive 'consumer-local'
        . $shim
        $env:POWERSHELL_MODULES_PATH | Should -BeExactly (Join-Path $env:LOCALAPPDATA 'PowerShell/Modules')
    }
    It 'has a per-user module fallback when LOCALAPPDATA is absent' {
        $env:POWERSHELL_MODULES_PATH = $null
        $env:LOCALAPPDATA = $null
        . $shim
        $env:POWERSHELL_MODULES_PATH | Should -BeExactly (Join-Path $HOME '.local/share/powershell/Modules')
    }
    It 'refuses a self-referencing canonical root without recursion' {
        $env:POWERSHELL_PROFILE_ROOT = Split-Path -Parent $shim
        Mock Write-Warning {}
        . $shim
        Should -Invoke Write-Warning -Times 1 -ParameterFilter { $Message -match 'resolves to this shim' }
        $script:ShimFixtureLoaded | Should -BeFalse
    }
}
