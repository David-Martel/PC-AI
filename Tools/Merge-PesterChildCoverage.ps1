function Get-PesterCoveragePathKey {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$Path,
        [bool]$CaseSensitive=([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT)
    )
    $fullPath=[IO.Path]::GetFullPath($Path)
    if ($CaseSensitive) { return $fullPath }
    return $fullPath.ToUpperInvariant()
}

function Get-PesterCoverageCommandKey {
    [CmdletBinding()]
    param([Parameter(Mandatory)]$Command, [string]$PathKey)
    $path = if ($PSBoundParameters.ContainsKey('PathKey')) { $PathKey } else { Get-PesterCoveragePathKey -Path $Command.File }
    $span = foreach ($name in @('StartLine','StartColumn','EndLine','EndColumn')) {
        $number = 0
        if (-not [int]::TryParse([string]$Command.$name, [ref]$number) -or $number -lt 1) { throw "Invalid command extent: $name" }
        $number
    }
    if ($span[2] -lt $span[0] -or ($span[2] -eq $span[0] -and $span[3] -lt $span[1])) { throw 'Invalid command extent order.' }
    if ($Command.Command -isnot [string] -or [string]::IsNullOrWhiteSpace($Command.Command)) { throw 'Missing command text.' }
    # Full extent and verbatim text distinguish multiple commands on one line.
    ConvertTo-Json -InputObject @($path, $span[0], $span[1], $span[2], $span[3], $Command.Command) -Compress
}

function New-PesterCoverageSnapshot {
    [CmdletBinding()]
    param([Parameter(Mandatory)]$Coverage, [Parameter(Mandatory)][object[]]$SourceBindings)
    $sources = @($SourceBindings | ForEach-Object {
        $path = [IO.Path]::GetFullPath($_.Path)
        $hash = (Get-FileHash -LiteralPath $path -Algorithm SHA256 -ErrorAction Stop).Hash
        if ($hash -cne $_.SHA) { throw 'Coverage source changed before snapshot.' }
        [pscustomobject]@{ Path=$path; SHA256=$hash }
    })
    $records = @(foreach ($executed in @($false,$true)) {
        $commands = if ($executed) { @($Coverage.CommandsExecuted) } else { @($Coverage.CommandsMissed) }
        foreach ($command in $commands) {
            if ($null -eq $command) { throw 'Missing coverage command inventory.' }
            [pscustomobject]@{ File=$command.File; StartLine=$command.StartLine; StartColumn=$command.StartColumn; EndLine=$command.EndLine; EndColumn=$command.EndColumn; Command=$command.Command; Executed=$executed }
        }
    })
    $snapshot = [pscustomobject]@{ SchemaVersion=1; Sources=$sources; Commands=$records; Analyzed=$Coverage.CommandsAnalyzedCount; Executed=$Coverage.CommandsExecutedCount }
    $null = Get-PesterChildCoverageMap -Snapshot $snapshot
    return $snapshot
}

function Get-PesterChildCoverageMap {
    [CmdletBinding()]
    param([Parameter(Mandatory)]$Snapshot)
    if ($Snapshot.SchemaVersion -ne 1) { throw 'Unsupported child coverage schema.' }
    $sources = [Collections.Generic.Dictionary[string,string]]::new([StringComparer]::Ordinal)
    foreach ($source in @($Snapshot.Sources)) {
        $path = [IO.Path]::GetFullPath($source.Path)
        $key=Get-PesterCoveragePathKey $path
        if ($source.SHA256 -cnotmatch '^[A-F0-9]{64}$' -or $sources.ContainsKey($key)) { throw 'Invalid or duplicate coverage source binding.' }
        if ((Get-FileHash -LiteralPath $path -Algorithm SHA256 -ErrorAction Stop).Hash -cne $source.SHA256) { throw 'Child coverage source identity mismatch.' }
        $sources.Add($key,$source.SHA256)
    }
    if ($sources.Count -eq 0) { throw 'Missing child coverage source bindings.' }
    $map = [Collections.Generic.Dictionary[string,object]]::new([StringComparer]::Ordinal)
    $fileCounts = [Collections.Generic.Dictionary[string,int]]::new([StringComparer]::Ordinal)
    $pathKeys = [Collections.Generic.Dictionary[string,string]]::new([StringComparer]::Ordinal)
    $executed = 0
    foreach ($command in @($Snapshot.Commands)) {
        $fileKey=$null
        if (-not $pathKeys.TryGetValue([string]$command.File,[ref]$fileKey)) {
            $fileKey=Get-PesterCoveragePathKey $command.File
            $pathKeys.Add([string]$command.File,$fileKey)
        }
        if ($command.Executed -isnot [bool] -or -not $sources.ContainsKey($fileKey)) { throw 'Invalid or unbound child command.' }
        $key = Get-PesterCoverageCommandKey -Command $command -PathKey $fileKey
        if ($map.ContainsKey($key)) { throw 'Duplicate coverage command identity in inventory.' }
        $map.Add($key,$command)
        if (-not $fileCounts.ContainsKey($fileKey)) { $fileCounts.Add($fileKey,0) }
        $fileCounts[$fileKey]++
        if ($command.Executed) { $executed++ }
    }
    if ($map.Count -le 0 -or $Snapshot.Analyzed -ne $map.Count -or $Snapshot.Executed -ne $executed) { throw 'Coverage inventory/count reconciliation failed.' }
    $map | Add-Member -NotePropertyName SourceCommandCounts -NotePropertyValue $fileCounts
    return ,$map
}

function Merge-PesterChildCoverage {
    [CmdletBinding()]
    param([Parameter(Mandatory)]$Result)
    $attachments = @(foreach ($test in @($Result.Tests)) {
        if ($null -ne $test -and $null -ne $test.PSObject.Properties['PcaiChildCoverage']) {
            if ($test.Result -cne 'Passed') { throw 'Unpassed parent test cannot admit child coverage.' }
            $test.PcaiChildCoverage
        }
    })
    if ($attachments.Count -eq 0) { return $Result.CodeCoverage }
    $coverage = $Result.CodeCoverage
    $parent = [Collections.Generic.Dictionary[string,object]]::new([StringComparer]::Ordinal)
    $parentFileCounts = [Collections.Generic.Dictionary[string,int]]::new([StringComparer]::Ordinal)
    $pathKeys = [Collections.Generic.Dictionary[string,string]]::new([StringComparer]::Ordinal)
    $hits = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    foreach ($executed in @($false,$true)) {
        $commands = if ($executed) { @($coverage.CommandsExecuted) } else { @($coverage.CommandsMissed) }
        foreach ($command in $commands) {
            $fileKey=$null
            if (-not $pathKeys.TryGetValue([string]$command.File,[ref]$fileKey)) {
                $fileKey=Get-PesterCoveragePathKey $command.File
                $pathKeys.Add([string]$command.File,$fileKey)
            }
            $key = Get-PesterCoverageCommandKey -Command $command -PathKey $fileKey
            if ($parent.ContainsKey($key)) { throw 'Duplicate parent coverage command identity.' }
            $parent.Add($key,$command)
            if (-not $parentFileCounts.ContainsKey($fileKey)) { $parentFileCounts.Add($fileKey,0) }
            $parentFileCounts[$fileKey]++
            if ($executed) { [void]$hits.Add($key) }
        }
    }
    if ($parent.Count -ne $coverage.CommandsAnalyzedCount -or $hits.Count -ne $coverage.CommandsExecutedCount -or $parent.Count -le 0) { throw 'Parent coverage inventory/count reconciliation failed.' }
    $originalExecuted = $hits.Count
    foreach ($attachment in $attachments) {
        $child = Get-PesterChildCoverageMap $attachment.Child
        $parentSources = [Collections.Generic.Dictionary[string,string]]::new([StringComparer]::Ordinal)
        foreach ($source in @($attachment.ParentSources)) {
            $path = Get-PesterCoveragePathKey $source.Path
            if ($parentSources.ContainsKey($path)) { throw 'Duplicate parent source binding.' }
            $parentSources.Add($path,$source.SHA)
        }
        foreach ($source in @($attachment.Child.Sources)) {
            $path = Get-PesterCoveragePathKey $source.Path
            if (-not $parentSources.ContainsKey($path) -or $parentSources[$path] -cne $source.SHA256) { throw 'Parent/child coverage source identity mismatch.' }
            if (-not $parentFileCounts.ContainsKey($path) -or $parentFileCounts[$path] -ne $child.SourceCommandCounts[$path]) { throw 'Parent/child coverage command inventory mismatch.' }
        }
        foreach ($entry in $child.GetEnumerator()) {
            if (-not $parent.ContainsKey($entry.Key)) { throw 'Child coverage command absent from parent denominator.' }
            if ($entry.Value.Executed) { [void]$hits.Add($entry.Key) }
        }
    }
    $merged = [pscustomobject]@{ CoveragePercent=100.0*$hits.Count/$parent.Count; CommandsAnalyzedCount=$parent.Count; CommandsExecutedCount=$hits.Count; CommandsMissedCount=$parent.Count-$hits.Count; AddedChildCommands=$hits.Count-$originalExecuted; ChildReceipts=$attachments.Count; Scope='Deduplicated source-bound command union; raw parent and child reports remain separate.' }
    $Result | Add-Member -NotePropertyName PcaiMergedCoverage -NotePropertyValue $merged -Force
    return $merged
}
