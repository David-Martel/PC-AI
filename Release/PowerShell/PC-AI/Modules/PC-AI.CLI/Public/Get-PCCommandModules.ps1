function Get-PCCommandModules {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$CommandName,
        [string]$ProjectRoot
    )

    $map = Get-PCCommandMap -ProjectRoot $ProjectRoot
    return $map[$CommandName]
}
