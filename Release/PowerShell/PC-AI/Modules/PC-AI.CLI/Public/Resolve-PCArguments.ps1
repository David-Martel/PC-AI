function Resolve-PCArguments {
    [CmdletBinding()]
    param(
        [string[]]$InputArgs,
        [hashtable]$Defaults = @{}
    )

    return Parse-PCArguments -InputArgs $InputArgs -Defaults $Defaults
}
