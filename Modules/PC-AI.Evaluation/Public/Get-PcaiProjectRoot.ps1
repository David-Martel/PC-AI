function Get-PcaiProjectRoot {
    [CmdletBinding()]
    param()
    $moduleRoot = Split-Path -Parent $PSScriptRoot
    return Split-Path -Parent $moduleRoot
}
