# pc_accel shim: keep the cloud profile path tiny and load the real profile locally.

$localProfileRoot = if ($env:POWERSHELL_PROFILE_ROOT) {
    $env:POWERSHELL_PROFILE_ROOT
} else {
    Join-Path $HOME '.config\powershell'
}
$localModulesRoot = if ($env:POWERSHELL_MODULES_PATH) {
    $env:POWERSHELL_MODULES_PATH
} elseif ($env:LOCALAPPDATA) {
    Join-Path $env:LOCALAPPDATA 'PowerShell\Modules'
} else {
    Join-Path $HOME '.local/share/powershell/Modules'
}
$localProfile = Join-Path $localProfileRoot 'Microsoft.PowerShell_profile.ps1'

$env:POWERSHELL_PROFILE_ROOT = $localProfileRoot
$env:POWERSHELL_MODULES_PATH = $localModulesRoot

if (-not $env:PROFILEUTILITIES_PCAI_AUTO_PATH_OPTIMIZE) {
    $env:PROFILEUTILITIES_PCAI_AUTO_PATH_OPTIMIZE = '0'
}
if (-not $env:PS_SKIP_PROFILE_ACCELERATOR) {
    $env:PS_SKIP_PROFILE_ACCELERATOR = '1'
}
if (-not $env:PS_SKIP_PSMODULEPATH_OPTIMIZE) {
    $env:PS_SKIP_PSMODULEPATH_OPTIMIZE = '1'
}

if (Test-Path -LiteralPath $localProfile) {
    if ([string]::Equals([IO.Path]::GetFullPath($localProfile),
            [IO.Path]::GetFullPath($PSCommandPath), [StringComparison]::OrdinalIgnoreCase)) {
        Write-Warning 'Canonical PowerShell profile resolves to this shim; correct POWERSHELL_PROFILE_ROOT.'
        return
    }
    . $localProfile
    return
}

Write-Warning "Canonical local PowerShell profile not found: $localProfile"
