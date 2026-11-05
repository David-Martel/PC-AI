function Get-PcaiProcessCategories {
    <#
    .SYNOPSIS
        Groups process snapshots using the shared ordered native taxonomy.
    .DESCRIPTION
        The five categories are heuristic name groups. System_Services includes
        all unmatched names, without an inference about role or resource waste.
        Native virtual memory is not reported as measured private bytes.
    .OUTPUTS
        Category rows, or a JSON array with AsJson.
    #>
    [CmdletBinding()]
    param([switch]$AsJson)
    function Test-CategoryNumber($Value) {
        if ($null -eq $Value -or $Value -is [bool] -or $Value -is [string] -or $Value -isnot [ValueType]) { return $false }
        try { $number=[double]$Value } catch { return $false }
        return -not [double]::IsNaN($number) -and -not [double]::IsInfinity($number) -and $number -ge 0
    }
    $groups=[ordered]@{
        llm_agents=@('claude','codex','ollama','copilot','pcai','llama')
        browsers=@('chrome','brave','msedge','firefox')
        terminals=@('conhost','cmd','powershell','pwsh','wezterm','windowsterminal')
        build_tools=@('rust-analyzer','cargo','node','dotnet','msbuild','cl')
        system_services=@()
    }
    $labels=@{llm_agents='LLM_Agents';browsers='Browsers';terminals='Terminals';build_tools='Build_Tools';system_services='System_Services'}
    Import-Module PC-AI.Common -ErrorAction SilentlyContinue
    $nativeAvailable=$false
    try { $nativeAvailable=Initialize-PcaiNative } catch { Write-Verbose 'Native initialization unavailable.' }
    if ($nativeAvailable) {
        try {
            $json=[PcaiNative.OptimizerModule]::GetProcessCategoriesJson()
            if ([string]::IsNullOrWhiteSpace($json) -or -not $json.TrimStart().StartsWith('{')) { throw 'Native categories object required.' }
            $raw=$json|ConvertFrom-Json -ErrorAction Stop
            if ($raw.status -cne 'Success' -or $raw.categories -isnot [pscustomobject]) { throw 'Native categories schema/status is invalid.' }
            $rows=@(foreach($property in $raw.categories.PSObject.Properties) {
                if (-not $groups.Contains($property.Name)) { throw 'Unknown native category.' }
                foreach ($name in @('count','working_set_mb','private_mb','handle_count')) {
                    if (-not (Test-CategoryNumber $property.Value.$name)) { throw 'Native category metric is invalid.' }
                }
                if ($property.Value.count -ne [math]::Truncate($property.Value.count)) { throw 'Native category count must be integral.' }
                [pscustomobject]@{Category=$labels[$property.Name];ProcessCount=$property.Value.count;WorkingSetMB=$property.Value.working_set_mb;PrivateMB=$null;HandleCount=$null;TopProcess=$null;Source='PcaiNative.OptimizerModule';Taxonomy='NativeOrderedNameHeuristicV1';MeasurementStatus=[ordered]@{WorkingSetMB='MeasuredSnapshot';PrivateMB='UnavailableNativeVirtualSpace';HandleCount='UnavailableNativePartialQueries';TopProcess='Unavailable'}}
            })
            $rows=@($rows|Sort-Object Category)
            if ($AsJson) { return ConvertTo-Json -InputObject @($rows) -Depth 5 }
            return $rows
        } catch { Write-Verbose 'Native category acquisition or schema unavailable; using fallback.' }
    }
    $buckets=@{}; foreach ($name in $groups.Keys) { $buckets[$name]=[Collections.Generic.List[object]]::new() }
    foreach ($process in @(Get-Process)) {
        if ([string]::IsNullOrWhiteSpace($process.ProcessName)) { throw [IO.InvalidDataException]::new('Process name is unavailable.') }
        foreach ($field in @('WorkingSet64','PrivateMemorySize64','HandleCount')) {
            if (-not (Test-CategoryNumber $process.$field)) { throw [IO.InvalidDataException]::new('Process metric is invalid.') }
        }
        $name=$process.ProcessName.ToLowerInvariant(); $selected='system_services'
        foreach ($category in $groups.Keys) {
            foreach ($pattern in $groups[$category]) {
                if ($name.Contains($pattern)) { $selected=$category; break }
            }
            if ($selected -ne 'system_services') { break }
        }
        $buckets[$selected].Add($process)
    }
    $rows=@(foreach ($category in $groups.Keys) {
        $items=@($buckets[$category].ToArray())
        $working=0.0; $private=0.0; $handles=0.0
        foreach ($item in $items) { $working+=[double]$item.WorkingSet64; $private+=[double]$item.PrivateMemorySize64; $handles+=[double]$item.HandleCount }
        [pscustomobject]@{Category=$labels[$category];ProcessCount=$items.Count;WorkingSetMB=$working/1MB;PrivateMB=$private/1MB;HandleCount=$handles;TopProcess=if($items.Count){($items|Sort-Object PrivateMemorySize64 -Descending|Select-Object -First 1).ProcessName}else{$null};Source='PowerShell-Fallback';Taxonomy='NativeOrderedNameHeuristicV1';MeasurementStatus=[ordered]@{WorkingSetMB='MeasuredSnapshot';PrivateMB='MeasuredPrivateBytes';HandleCount='MeasuredSnapshot';TopProcess=if($items.Count){'MeasuredSnapshot'}else{'Unavailable'}}}
    })
    $rows=@($rows|Sort-Object Category)
    if ($AsJson) { return ConvertTo-Json -InputObject @($rows) -Depth 5 }
    return $rows
}
