#Requires -Version 5.1
<#
.SYNOPSIS
    Stops verified proxies and retains unresolved process custody.
#>
function Stop-HVSockProxy {
    [CmdletBinding(SupportsShouldProcess)]
    [OutputType([PSCustomObject])]
    param(
        [string]$StatePath = "$env:ProgramData\PC_AI\hvsock-proxy\state.json",
        [ValidateRange(0, 2147483647)][int]$WaitForExitMilliseconds = 5000
    )
    $StatePath = Resolve-HVSockStatePath -Path $StatePath
    if (-not (Test-Path -LiteralPath $StatePath -PathType Leaf)) {
        return [pscustomobject]@{ Stopped = 0; Message = 'No state file found.'; StatePath = $StatePath; Unresolved = @() }
    }
    $snapshot = Get-HVSockStateSnapshot -Path $StatePath
    $state = [Text.Encoding]::UTF8.GetString($snapshot.Bytes).TrimStart([char]0xfeff) | ConvertFrom-Json -ErrorAction Stop
    $unresolved = [Collections.Generic.List[object]]::new()
    $failures = [Collections.Generic.List[object]]::new()
    $stopped = 0
    foreach ($entry in $state) {
        $owned = Get-HVSockOwnedProcess -Entry $entry
        try {
            if (-not $owned.Verified) { throw $owned.Reason }
            if (-not $PSCmdlet.ShouldProcess("$($entry.Name) PID $($entry.Pid)", 'Stop verified proxy')) {
                $unresolved.Add($entry); continue
            }
            Stop-Process -InputObject $owned.Process -Force -ErrorAction Stop
            if (-not $owned.Process.WaitForExit($WaitForExitMilliseconds)) { throw 'Proxy exit wait expired; custody retained.' }
            if (-not $owned.Process.HasExited) { throw 'Proxy exit could not be confirmed.' }
            $stopped++
        } catch {
            $unresolved.Add($entry)
            $failures.Add([pscustomobject]@{ Pid = $entry.Pid; Error = $_.Exception.Message })
        } finally { if ($owned.Process) { $owned.Process.Dispose() } }
    }
    $previous = $null
    if ($stopped -gt 0 -and $PSCmdlet.ShouldProcess($StatePath, 'Publish confirmed proxy exits')) {
        $previous = Set-HVSockCustodyState -Path $StatePath -Expected $snapshot -Entries @($unresolved.ToArray())
    }
    return [pscustomobject]@{ Stopped = $stopped; StatePath = $StatePath; Unresolved = @($unresolved.ToArray()); Errors = @($failures.ToArray()); PreviousStatePath = $previous }
}
