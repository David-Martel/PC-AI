#Requires -Version 7.0
<#
.SYNOPSIS
    Batch-dispatch Jules review sessions for PC_AI modules.
.DESCRIPTION
    Reads Config/jules-review-prompts.json, filters by scope/module names/git diff,
    sorts by priority, and dispatches up to -MaxSessions reviews via Invoke-JulesSession.ps1.
.EXAMPLE
    .\Tools\Invoke-JulesBatchReview.ps1 -Scope RustCrates -DryRun
.EXAMPLE
    .\Tools\Invoke-JulesBatchReview.ps1 -Modules pcai_inference,pcai_media
.EXAMPLE
    .\Tools\Invoke-JulesBatchReview.ps1 -ChangedSinceTag v0.2.0 -MaxSessions 8
#>
[CmdletBinding(DefaultParameterSetName = 'ByScope')]
param(
    [Parameter(Mandatory, ParameterSetName = 'ByScope')]
    [ValidateSet('RustCrates', 'PowerShellModules', 'CSharp', 'All')]
    [string]  $Scope,

    [Parameter(Mandatory, ParameterSetName = 'ByModule')]
    [string[]] $Modules,

    [Parameter(Mandatory, ParameterSetName = 'ByDiff')]
    [string]  $ChangedSinceTag,

    [int]    $MaxSessions = 5,
    [switch] $RequirePlanApproval = $true,
    [switch] $DryRun,
    [ValidateSet('Table', 'Json')]
    [string] $Format = 'Table'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot   = Split-Path $PSScriptRoot -Parent
$config     = Get-Content (Join-Path $repoRoot 'Config/jules-review-prompts.json') -Raw | ConvertFrom-Json
$sessionDir = Join-Path $repoRoot '.pcai/jules/sessions'
$sessionPs1 = Join-Path $PSScriptRoot 'Invoke-JulesSession.ps1'

$priorityRank = @{ high = 1; medium = 2; low = 3 }
$scopeTagMap  = @{ RustCrates = 'rust'; PowerShellModules = 'powershell'; CSharp = 'csharp'; All = $null }

# Filter
$allMods = $config.modules.PSObject.Properties
$queue = switch ($PSCmdlet.ParameterSetName) {
    'ByScope' {
        $tag = $scopeTagMap[$Scope]
        $allMods | Where-Object { $null -eq $tag -or $_.Value.tags -contains $tag }
    }
    'ByModule' {
        $allMods | Where-Object { $_.Name -in $Modules }
    }
    'ByDiff' {
        $changed = git -C $repoRoot diff --name-only "${ChangedSinceTag}...HEAD" 2>$null
        $allMods | Where-Object { $_.Value.files | Where-Object { $_ -in $changed } }
    }
}

$queue = $queue | Sort-Object { $priorityRank[$_.Value.priority] ?? 99 } | Select-Object -First $MaxSessions

if (-not $queue) { Write-Warning 'No modules matched the filter.'; return }

$null = New-Item $sessionDir -ItemType Directory -Force
$results = [System.Collections.Generic.List[pscustomobject]]::new()

foreach ($entry in $queue) {
    $mod = $entry.Value
    $r   = [pscustomobject]@{
        Module    = $entry.Name
        Priority  = $mod.priority
        Tags      = $mod.tags -join ', '
        Status    = $null
        SessionId = $null
        Error     = $null
    }

    if ($DryRun) {
        $r.Status = 'DryRun'
    } else {
        try {
            $params = @{ Action = 'New'; Prompt = $mod.prompt; AutomationMode = 'AutoCreatePR'; Format = 'Json' }
            if ($RequirePlanApproval) { $params['RequirePlanApproval'] = $true }
            $raw = & $sessionPs1 @params 2>&1
            $rawStr = ($raw | Out-String).Trim()
            $parsed = $rawStr | ConvertFrom-Json -ErrorAction Stop
            $sid = if ($parsed.name) { ($parsed.name -split '/')[-1] } elseif ($parsed.sessionId) { $parsed.sessionId } else { $parsed.id }
            $r.SessionId = $sid
            $r.Status    = 'Dispatched'
        } catch {
            $r.Status = 'Failed'
            $errMsg = if ($_.Exception) { $_.Exception.Message } else { "$_" }
            $r.Error  = $errMsg
            Write-Warning "Jules dispatch failed for '$($entry.Name)': $errMsg"
        }
    }

    $results.Add($r)
}

$batchFile = Join-Path $sessionDir "batch-$(Get-Date -Format 'yyyyMMdd_HHmmss').json"
$results | ConvertTo-Json -Depth 4 | Set-Content $batchFile -Encoding utf8
Write-Verbose "Batch results saved: $batchFile"

if ($Format -eq 'Json') { $results | ConvertTo-Json -Depth 4 } else { $results | Format-Table -AutoSize }
