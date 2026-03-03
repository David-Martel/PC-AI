# Check PcaiNative.dll types
$repoRoot = if ($env:PCAI_ROOT) {
    $env:PCAI_ROOT
} else {
    Split-Path -Parent $PSScriptRoot
}
$sourceDll = Join-Path $repoRoot 'Native\PcaiNative\bin\Release\net8.0\win-x64\PcaiNative.dll'
$targetDir = Join-Path $repoRoot 'bin'
$targetDll = Join-Path $targetDir 'PcaiNative.dll'

Copy-Item $sourceDll $targetDir -Force
$assembly = [System.Reflection.Assembly]::LoadFrom($targetDll)
Write-Host "Public types in PcaiNative.dll:"
$assembly.GetTypes() | Where-Object { $_.IsPublic } | ForEach-Object { Write-Host "  - $($_.FullName)" }