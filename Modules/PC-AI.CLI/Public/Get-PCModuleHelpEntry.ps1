function Get-PCModuleHelpEntry {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Name,
        [string[]]$Modules,
        [string]$ProjectRoot
    )

    $entries = Get-PCModuleHelpIndex -Modules $Modules -ProjectRoot $ProjectRoot
    return $entries | Where-Object { $_.Name -eq $Name -or $_.Module -eq $Name }
}
