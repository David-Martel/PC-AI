#Requires -Version 7.0
[CmdletBinding()]
param()
$ErrorActionPreference='Stop'
$helper=(Resolve-Path (Join-Path $PSScriptRoot '../../Tools/Test-PcaiModuleManifests.ps1')).Path
$root=Join-Path ([IO.Path]::GetTempPath()) ('pcai-manifest-'+[guid]::NewGuid().ToString('N'))
$null=New-Item -ItemType Directory -Path $root
try{
    Set-Content -LiteralPath (Join-Path $root 'Fixture.psm1') -Value ''
    $manifest=Join-Path $root 'Fixture.psd1'
    Set-Content -LiteralPath $manifest -Value "@{ModuleVersion='1.0.0';RootModule='Fixture.psm1'}"
    # Selector valid-manifest protects real validation through -File across shells.
    # Detects quoting/path expansion errors. Needs pwsh7, owned temporary fixture.
    # Breadcrumb: Tools/Test-PcaiModuleManifests.ps1 and lefthook ps-manifest-check.
    $pwsh=(Get-Process -Id $PID).Path
    $output=& $pwsh -NoLogo -NoProfile -File $helper -ModulesPath $root 2>&1
    if($LASTEXITCODE -ne 0 -or "$output" -notmatch '1 module manifests validated'){throw "Valid manifest rejected: $output"}
    Write-Output 'PASS: valid fixture manifest admitted through file entrypoint'
    # Selector broken-manifest protects rejection, not unconditional hook success.
    # Detects missing root modules. Needs no installed modules/hardware.
    # Breadcrumb: Test-ModuleManifest invocation in the production helper.
    Set-Content -LiteralPath $manifest -Value "@{ModuleVersion='1.0.0';RootModule='DeliberatelyMissing.psm1'}"
    $output=& $pwsh -NoLogo -NoProfile -File $helper -ModulesPath $root 2>&1
    if($LASTEXITCODE -eq 0 -or "$output" -notmatch 'DeliberatelyMissing'){throw 'Broken manifest was not specifically rejected'}
    Write-Output 'PASS: broken fixture manifest rejected'
}finally{Remove-Item -LiteralPath $root -Recurse -Force}

# The last child intentionally fails. Report this CLI's validated success rather
# than leaking that expected failure to GitHub's LASTEXITCODE wrapper. Any failed
# assertion/cleanup throws before this line and remains nonzero.
exit 0
