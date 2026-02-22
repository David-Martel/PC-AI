function Get-PcaiProjectRoot {
    $moduleRoot = Split-Path -Parent $PSScriptRoot
    return Split-Path -Parent $moduleRoot
}
