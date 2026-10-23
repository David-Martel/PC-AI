#Requires -Version 7.0
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }

Describe 'Media availability after a real native operation error' -Tag 'NativeIntegration' {
    It 'keeps a callable DLL available after a negative last error' -Skip:([string]::IsNullOrWhiteSpace($env:PCAI_TEST_MEDIA_DLL)) {
        $repoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
        $managed = Join-Path $repoRoot 'Native/PcaiNative/bin/Release/net8.0/win-x64/PcaiNative.dll'
        Test-Path -LiteralPath $managed | Should -BeTrue
        Test-Path -LiteralPath $env:PCAI_TEST_MEDIA_DLL | Should -BeTrue
        $child = Join-Path $TestDrive 'media-error-probe.ps1'
        @'
param([string]$Managed, [string]$Media)
$ErrorActionPreference = 'Stop'
$env:PCAI_NATIVE_BUNDLE_ROOT = Split-Path -Parent $Media
Add-Type -LiteralPath $Managed
if (-not [PcaiNative.MediaModule]::IsAvailable) { throw 'Fresh explicit media DLL is unavailable.' }
$result = [PcaiNative.MediaModule]::pcai_media_load_model('pcai-fixture-does-not-exist', 0)
$errorCode = [PcaiNative.MediaModule]::pcai_media_last_error_code()
if ($result -ge 0 -or $errorCode -ge 0) { throw 'Fixture did not produce a real native error.' }
if (-not [PcaiNative.MediaModule]::IsAvailable) { throw 'Operation error incorrectly changed DLL availability.' }
[pscustomobject]@{ OperationStatus = $result; LastErrorCode = $errorCode; AvailableAfterError = $true } | ConvertTo-Json -Compress
'@ | Set-Content -LiteralPath $child -Encoding utf8
        $output = & pwsh -NoLogo -NoProfile -File $child -Managed $managed -Media $env:PCAI_TEST_MEDIA_DLL 2>&1
        $LASTEXITCODE | Should -Be 0 -Because ($output -join "`n")
        $probe = ($output -join "`n") | ConvertFrom-Json
        $probe.AvailableAfterError | Should -BeTrue
        $probe.LastErrorCode | Should -BeLessThan 0
    }
}
