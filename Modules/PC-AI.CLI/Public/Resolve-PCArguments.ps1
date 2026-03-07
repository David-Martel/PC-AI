function Resolve-PCArguments {
    [CmdletBinding()]
    param(
        [string[]]$InputArgs,
        [hashtable]$Defaults = @{}
    )

    return ConvertTo-PCArgumentMap -InputArgs $InputArgs -Defaults $Defaults
}
