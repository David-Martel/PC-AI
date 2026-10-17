#Requires -Version 7.0
[CmdletBinding()]
param([string]$ModulesPath=(Join-Path $PSScriptRoot '../Modules'))
$ErrorActionPreference='Stop'
$manifests=@(Get-ChildItem -LiteralPath $ModulesPath -Recurse -Filter '*.psd1' -File -ErrorAction Stop)
if($manifests.Count -eq 0){throw 'No module manifests found'}
foreach($manifest in $manifests){
    Test-ModuleManifest -Path $manifest.FullName -ErrorAction Stop | Out-Null
}
Write-Output "PASS: $($manifests.Count) module manifests validated"
