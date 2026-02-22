function Get-PCCommandList {
    [CmdletBinding()]
    param([string]$ProjectRoot)

    $map = Get-PCCommandMap -ProjectRoot $ProjectRoot
    return $map.Keys | Sort-Object
}
