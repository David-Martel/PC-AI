# PC-AI Common Module
. "$PSScriptRoot\Module-Common.ps1"

$publicPath = Join-Path $PSScriptRoot 'Public'
if (Test-Path $publicPath) {
    Get-ChildItem -Path $publicPath -Filter '*.ps1' -File -ErrorAction SilentlyContinue | ForEach-Object {
        . $_.FullName
    }
}

Export-ModuleMember -Function @(
    'Write-Success',
    'Write-Warning',
    'Write-Error',
    'Write-Info',
    'Write-Header',
    'Write-SubHeader',
    'Write-Bullet',
    'Initialize-PcaiNative',
    'Resolve-PcaiRepoRoot',
    'Get-PcaiRuntimeConfig',
    'Get-ScriptMetadata'
)
