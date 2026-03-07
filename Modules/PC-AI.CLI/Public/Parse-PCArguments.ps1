function ConvertTo-PCArgumentMap {
    [CmdletBinding()]
    param(
        [string[]]$InputArgs,
        [hashtable]$Defaults = @{}
    )

    $parsed = @{
        SubCommand = $null
        Flags = @{}
        Values = @{}
        Positional = @()
    }

    foreach ($key in $Defaults.Keys) {
        $parsed.Values[$key] = $Defaults[$key]
    }

    $i = 0
    while ($i -lt $InputArgs.Count) {
        $arg = $InputArgs[$i]

        if ($arg -match '^--(.+)=(.+)$') {
            $parsed.Values[$Matches[1]] = $Matches[2]
        } elseif ($arg -match '^--(.+)$') {
            $key = $Matches[1]
            if ($i + 1 -lt $InputArgs.Count -and $InputArgs[$i + 1] -notmatch '^-') {
                $parsed.Values[$key] = $InputArgs[$i + 1]
                $i++
            } else {
                $parsed.Flags[$key] = $true
            }
        } elseif ($arg -match '^-([a-zA-Z])$') {
            $parsed.Flags[$Matches[1]] = $true
        } elseif ($null -eq $parsed.SubCommand -and $arg -notmatch '^-') {
            $parsed.SubCommand = $arg
        } else {
            $parsed.Positional += $arg
        }

        $i++
    }

    return $parsed
}
