#Requires -Version 7.0
<#
.SYNOPSIS
    LLM orchestration layer for Jules integration — signal-driven dispatch and plan review.

.DESCRIPTION
    Two capabilities:
      AnalyzeAndDispatch  Gather repo signals (clippy, git diff, TODOs, issues), build an
                          analysis JSON, and (unless -DryRun) create Jules sessions ordered
                          by signal severity.
      ReviewPlans         Review all sessions currently AwaitingPlanApproval.
      ReviewPlan          Review a single session by -SessionId, with optional feedback loop.

.PARAMETER Action
    AnalyzeAndDispatch | ReviewPlans | ReviewPlan

.PARAMETER SessionId
    Required for ReviewPlan.

.PARAMETER MaxSessions
    Cap on the number of Jules sessions created by AnalyzeAndDispatch. Default: 5.

.PARAMETER DryRun
    Print what would be dispatched without creating sessions.

.PARAMETER Format
    Output format: Table | Json | Report. Default: Json.

.PARAMETER MaxFeedbackRounds
    Maximum plan-feedback iterations before giving up. Default: 3.

.PARAMETER PollIntervalSeconds
    Seconds between status polls during feedback loop. Default: 30.

.PARAMETER PollTimeoutMinutes
    Maximum minutes to wait for Jules to revise a plan. Default: 10.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [ValidateSet('AnalyzeAndDispatch', 'ReviewPlans', 'ReviewPlan')]
    [string]$Action,

    [string]$SessionId,
    [int]$MaxSessions           = 5,
    [switch]$DryRun,
    [ValidateSet('Table', 'Json', 'Report')][string]$Format = 'Json',
    [int]$MaxFeedbackRounds     = 3,
    [int]$PollIntervalSeconds   = 30,
    [int]$PollTimeoutMinutes    = 10
)

$ErrorActionPreference = 'Stop'
# StrictMode removed — causes .Count issues with null/empty arrays from CLI output

$script:RepoRoot    = Split-Path -Parent $PSScriptRoot
$script:JulesScript = Join-Path $PSScriptRoot 'Invoke-JulesSession.ps1'
$script:AnalysisDir = Join-Path $script:RepoRoot '.pcai\jules'
$script:PromptsJson = Join-Path $script:RepoRoot 'Config\jules-review-prompts.json'

# ---------------------------------------------------------------------------
# Helper: invoke Invoke-JulesSession.ps1 as a subprocess, return parsed JSON
# ---------------------------------------------------------------------------
function Invoke-Jules {
    [OutputType([object])]
    param([hashtable]$Params)

    $argList = @()
    foreach ($kv in $Params.GetEnumerator()) {
        if ($kv.Value -is [switch] -or $kv.Value -is [bool]) {
            if ($kv.Value) { $argList += "-$($kv.Key)" }
        } else {
            $argList += "-$($kv.Key)", $kv.Value
        }
    }
    $argList += '-Format', 'Json'

    $raw = & pwsh -NoLogo -NonInteractive -File $script:JulesScript @argList 2>$null
    if (-not $raw) { return $null }
    try { return $raw | ConvertFrom-Json -Depth 20 } catch { return $raw }
}

# ---------------------------------------------------------------------------
# Internal: load jules-review-prompts.json module map
# ---------------------------------------------------------------------------
function Get-ModuleMap {
    [OutputType([hashtable])]
    param()
    if (-not (Test-Path -LiteralPath $script:PromptsJson)) { return @{} }
    $cfg = Get-Content -LiteralPath $script:PromptsJson -Raw | ConvertFrom-Json -Depth 10
    $map = @{}
    foreach ($prop in $cfg.modules.PSObject.Properties) {
        $map[$prop.Name] = $prop.Value
    }
    return $map
}

# ---------------------------------------------------------------------------
# Internal: Invoke-PlanReview
#   Returns [PSCustomObject]{ SessionId, Module, Recommendation, Message, Checks }
# ---------------------------------------------------------------------------
function Invoke-PlanReview {
    [OutputType([PSCustomObject])]
    param(
        [string]$ReviewSessionId,
        [object[]]$Steps,        # array of {title, description}
        [string]$ModuleName,
        [string[]]$ScopePaths
    )

    $checks = [ordered]@{}

    # -- scope_check --
    $outOfScope = @($Steps | Where-Object {
        $title = $_.title
        $inScope = $false
        foreach ($p in $ScopePaths) {
            $p2 = $p.TrimEnd('/')
            if ($title -like "*$p2*" -or $title -like "*$(Split-Path $p2 -Leaf)*") {
                $inScope = $true; break
            }
        }
        -not $inScope
    }).Count

    $checks['scope_check'] = if ($outOfScope -eq 0) { 'pass' }
                             elseif ($outOfScope -le 2) { 'warn' }
                             else { 'fail' }

    # -- test_coverage --
    $hasTest = ($Steps | Where-Object { $_.title -imatch 'test' }).Count -gt 0
    $removesTest = ($Steps | Where-Object { $_.title -imatch 'remov' -and $_.title -imatch 'test' }).Count -gt 0

    $checks['test_coverage'] = if ($removesTest) { 'fail' }
                               elseif ($hasTest) { 'pass' }
                               else { 'warn' }

    # -- conventions --
    $badUnwrap = ($Steps | Where-Object {
        $desc = $_.description
        ($desc -imatch 'unwrap') -and ($desc -notmatch '(?i)expect')
    }).Count -gt 0
    $unsafeNoDoc = ($Steps | Where-Object {
        $desc = $_.description
        ($desc -imatch 'unsafe') -and ($desc -notmatch '(?i)document')
    }).Count -gt 0

    $checks['conventions'] = if ($unsafeNoDoc) { 'fail' }
                             elseif ($badUnwrap) { 'warn' }
                             else { 'pass' }

    # -- recommendation --
    $failCount = ($checks.Values | Where-Object { $_ -eq 'fail' }).Count
    $warnCount = ($checks.Values | Where-Object { $_ -eq 'warn' }).Count

    $recommendation = if ($failCount -gt 2) { 'reject' }
                      elseif ($failCount -gt 0) { 'feedback' }
                      elseif ($warnCount -gt 0) { 'feedback' }
                      else { 'approve' }

    $msg = switch ($recommendation) {
        'approve'  { 'Plan meets all conventions.' }
        'feedback' {
            $issues = @()
            if ($checks['scope_check'] -ne 'pass') { $issues += "scope: $outOfScope step(s) reference out-of-scope files" }
            if ($checks['test_coverage'] -eq 'warn') { $issues += 'test_coverage: no test steps found — add tests or confirm this is intentional' }
            if ($checks['conventions'] -eq 'warn') { $issues += 'conventions: use .expect("<reason>") instead of .unwrap() per M-PANIC-ON-BUG' }
            if ($checks['conventions'] -eq 'fail') { $issues += 'conventions: unsafe blocks must include a SAFETY doc comment per M-UNSAFE' }
            "Please revise the plan: " + ($issues -join '; ')
        }
        'reject'   { "Plan rejected ($failCount critical failures). Restart session with a narrower scope." }
    }

    return [PSCustomObject]@{
        SessionId      = $ReviewSessionId
        Module         = $ModuleName
        Recommendation = $recommendation
        Message        = $msg
        Checks         = [PSCustomObject]$checks
    }
}

# ---------------------------------------------------------------------------
# Action: ReviewPlan (single session, with optional feedback loop)
# ---------------------------------------------------------------------------
function Invoke-ReviewPlanAction {
    param([string]$TargetSessionId)

    $moduleMap = Get-ModuleMap

    # Fetch plan
    $plan = Invoke-Jules @{ Action = 'GetPlan'; SessionId = $TargetSessionId }
    if (-not $plan) { Write-Error "No plan returned for session $TargetSessionId"; return }

    # Resolve steps array regardless of API shape
    $steps = if ($plan.steps) { $plan.steps }
             elseif ($plan.plan.steps) { $plan.plan.steps }
             else { @() }

    # Detect module from session title / session metadata
    $moduleName = 'unknown'
    $scopePaths = @()
    $statusObj  = Invoke-Jules @{ Action = 'Status'; SessionId = $TargetSessionId }
    $sessionTitle = if ($statusObj.title) { $statusObj.title } else { '' }
    foreach ($kv in $moduleMap.GetEnumerator()) {
        if ($sessionTitle -imatch [regex]::Escape($kv.Key)) {
            $moduleName = $kv.Key
            $scopePaths = $kv.Value.files
            break
        }
    }

    $round = 0
    do {
        $result = Invoke-PlanReview -ReviewSessionId $TargetSessionId `
                                    -Steps $steps `
                                    -ModuleName $moduleName `
                                    -ScopePaths $scopePaths

        if ($Format -eq 'Table') {
            $result | Format-List
        } elseif ($Format -eq 'Report') {
            "Session : $TargetSessionId"
            "Module  : $moduleName"
            "Result  : $($result.Recommendation.ToUpper())"
            "Message : $($result.Message)"
        } else {
            $result | ConvertTo-Json -Depth 5
        }

        if ($result.Recommendation -eq 'approve') {
            Invoke-Jules @{ Action = 'Approve'; SessionId = $TargetSessionId } | Out-Null
            break
        }
        if ($result.Recommendation -eq 'reject') { break }

        # feedback — send message, poll for plan revision
        $round++
        if ($round -gt $MaxFeedbackRounds) {
            Write-Warning "Max feedback rounds ($MaxFeedbackRounds) reached for $TargetSessionId"
            break
        }

        Write-Verbose "Sending feedback (round $round): $($result.Message)"
        Invoke-Jules @{ Action = 'Message'; SessionId = $TargetSessionId; Prompt = $result.Message } | Out-Null

        # Poll until state leaves AWAITING_PLAN_APPROVAL or timeout
        $deadline = [datetime]::UtcNow.AddMinutes($PollTimeoutMinutes)
        $revised  = $false
        while ([datetime]::UtcNow -lt $deadline) {
            Start-Sleep -Seconds $PollIntervalSeconds
            $st = Invoke-Jules @{ Action = 'Status'; SessionId = $TargetSessionId }
            if ($st.state -ine 'AWAITING_PLAN_APPROVAL') { $revised = $true; break }
        }
        if (-not $revised) {
            Write-Warning "Timed out waiting for plan revision on $TargetSessionId"
            break
        }

        # Re-fetch the revised plan
        $plan  = Invoke-Jules @{ Action = 'GetPlan'; SessionId = $TargetSessionId }
        $steps = if ($plan.steps) { $plan.steps } elseif ($plan.plan.steps) { $plan.plan.steps } else { @() }

    } while ($true)
}

# ---------------------------------------------------------------------------
# Action: ReviewPlans (all pending)
# ---------------------------------------------------------------------------
function Invoke-ReviewPlansAction {
    $sessions = Invoke-Jules @{ Action = 'List'; State = 'AwaitingPlanApproval' }
    if (-not $sessions) { Write-Verbose 'No sessions awaiting plan approval.'; return }
    $list = if ($sessions -is [array]) { $sessions } else { @($sessions) }
    foreach ($s in $list) {
        Invoke-ReviewPlanAction -TargetSessionId $s.name.Split('/')[-1]
    }
}

# ---------------------------------------------------------------------------
# Action: AnalyzeAndDispatch
# ---------------------------------------------------------------------------
function Invoke-AnalyzeAndDispatchAction {
    $moduleMap = Get-ModuleMap

    # 1. Clippy violations
    Write-Verbose 'Running cargo clippy…'
    $clippyOut    = cargo clippy --manifest-path "$($script:RepoRoot)\Native\pcai_core\Cargo.toml" `
                        --all-targets --message-format=json 2>$null
    $clippyCount  = ([string[]]$clippyOut | Where-Object { $_ -like '*"level":"warning"*' }).Count

    # 2. Recently changed files and module mapping
    Write-Verbose 'Checking git diff…'
    $changedFiles = git -C $script:RepoRoot diff --name-only HEAD~20 2>$null
    $changedModules = [System.Collections.Generic.HashSet[string]]::new()
    foreach ($file in $changedFiles) {
        foreach ($kv in $moduleMap.GetEnumerator()) {
            foreach ($scopePath in $kv.Value.files) {
                $prefix = $scopePath.TrimEnd('/')
                if ($file -like "$prefix*" -or $file -like "*$(Split-Path $prefix -Leaf)*") {
                    [void]$changedModules.Add($kv.Key)
                }
            }
        }
    }

    # 3. TODO/FIXME/HACK markers
    Write-Verbose 'Counting TODO markers…'
    $todoOut   = git -C $script:RepoRoot grep -c 'TODO\|FIXME\|HACK' -- 'Native/' 'Modules/' 2>$null
    $todoCount = ([string[]]$todoOut | Measure-Object -Sum {
        if ($_ -match ':(\d+)$') { [int]$Matches[1] } else { 0 }
    }).Sum

    # 4. Open issues
    Write-Verbose 'Fetching open issues…'
    $issueJson  = gh issue list --label bug --label enhancement --json number,title --limit 10 2>$null
    $openIssues = if ($issueJson) { $issueJson | ConvertFrom-Json -Depth 5 } else { @() }

    $signals = [PSCustomObject]@{
        clippy_violations     = $clippyCount
        recently_changed_files= @($changedFiles)
        changed_modules       = @($changedModules)
        todo_markers          = [int]$todoCount
        open_issues           = @($openIssues)
    }

    # Build recommended sessions — sorted: clippy > recently changed > config priority
    $priorityOrder = @{ high = 0; medium = 1; low = 2 }
    $recommended = $moduleMap.GetEnumerator() | Sort-Object {
        $m = $_.Value
        $score  = $priorityOrder[$m.priority] * 100
        # recently changed bumps score down (higher priority)
        if ($changedModules.Contains($_.Key)) { $score -= 50 }
        # clippy penalty only for Rust modules
        if ($clippyCount -gt 0 -and 'rust' -in $m.tags) { $score -= 20 }
        $score
    } | Select-Object -First $MaxSessions

    $analysis = [PSCustomObject]@{
        timestamp             = [datetime]::UtcNow.ToString('o')
        signals               = $signals
        recommended_sessions  = @($recommended | ForEach-Object {
            [PSCustomObject]@{ module = $_.Key; priority = $_.Value.priority; prompt = $_.Value.prompt }
        })
    }

    # Persist analysis
    $null = [System.IO.Directory]::CreateDirectory($script:AnalysisDir)
    $stamp   = [datetime]::UtcNow.ToString('yyyyMMdd_HHmmss')
    $outFile = Join-Path $script:AnalysisDir "analysis-$stamp.json"
    $analysis | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $outFile -Encoding utf8

    if ($DryRun) {
        Write-Verbose "[DryRun] Would create $($recommended.Count) Jules session(s). Analysis: $outFile"
        if ($Format -eq 'Json') { $analysis | ConvertTo-Json -Depth 10 }
        else { $analysis | Format-List }
        return
    }

    # Dispatch sessions
    foreach ($session in $analysis.recommended_sessions) {
        Write-Verbose "Creating Jules session for module: $($session.module)"
        Invoke-Jules @{
            Action             = 'New'
            Prompt             = $session.prompt
            Title              = "Auto-review: $($session.module)"
            RequirePlanApproval= $true
        } | Out-Null
    }

    if ($Format -eq 'Json') { $analysis | ConvertTo-Json -Depth 10 }
    else { $analysis | Format-List }
}

# ---------------------------------------------------------------------------
# __test_load__ — expose Invoke-PlanReview in global scope for Pester
# ---------------------------------------------------------------------------
if ($Action -eq '__test_load__') {
    Set-Item -Path 'function:global:Invoke-PlanReview' -Value (Get-Item 'function:Invoke-PlanReview').ScriptBlock
    Set-Item -Path 'function:global:Get-ModuleMap'    -Value (Get-Item 'function:Get-ModuleMap').ScriptBlock
    return
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
switch ($Action) {
    'AnalyzeAndDispatch' { Invoke-AnalyzeAndDispatchAction }
    'ReviewPlans'        { Invoke-ReviewPlansAction }
    'ReviewPlan'         {
        if (-not $SessionId) { throw '-SessionId is required for ReviewPlan' }
        Invoke-ReviewPlanAction -TargetSessionId $SessionId
    }
}
