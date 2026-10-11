#Requires -Version 7.0
<#
.SYNOPSIS
    PR triage dashboard for pull requests Jules contributed to, with CI status.
.PARAMETER State
    Lifecycle filter: open (default), closed, merged, all.
.PARAMETER Format
    Table (default) or Json.
.EXAMPLE
    .\Get-JulesPRStatus.ps1 -State all -Format Json
#>
[CmdletBinding()]
param(
    [ValidateSet('open', 'closed', 'merged', 'all')]
    [string]$State = 'open',

    [ValidateSet('Table', 'Json')]
    [string]$Format = 'Table',

    # Most recent PRs fetched before the Jules filter is applied. Capped at 40:
    # `gh pr list --json commits` asks for commits(100) x authors(100) per PR,
    # plus ~102 status-check nodes, about 10,200 nodes per PR, and GitHub rejects
    # queries over 500,000 nodes (50 PRs = 510,100). 40 leaves headroom.
    [ValidateRange(1, 40)]
    [int]$Limit = 40
)

$ErrorActionPreference = 'Stop'

# 1. Verify gh is available
if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    throw 'gh CLI is not on PATH. Install from https://cli.github.com/ and authenticate with `gh auth login`.'
}

# 2. Fetch PRs and keep those Jules contributed to. Jules opens PRs under the
#    repository owner's account, so the PR author is never jules; the stable
#    marker is a commit authored by google-labs-jules[bot].
$julesLogin = 'google-labs-jules[bot]'
$fields = 'number,title,state,mergeable,statusCheckRollup,changedFiles,headRefName,createdAt,url,commits'
$raw = gh pr list --state $State --limit $Limit --json $fields 2>&1
if ($LASTEXITCODE -ne 0) {
    throw "gh pr list failed: $raw"
}

$parsed = ($raw | Out-String) | ConvertFrom-Json -ErrorAction SilentlyContinue
[array]$prs = @($parsed | Where-Object {
        $_ -and @($_.commits | ForEach-Object { $_.authors.login }) -contains $julesLogin
    })

# 3. Handle empty result
if ($prs.Count -eq 0) {
    if ($Format -eq 'Json') { return '[]' }
    Write-Output "No Jules PRs found (state=$State)."
    return
}

# 4. Derive CI status from statusCheckRollup
function Get-CiStatus([object]$rollup) {
    if (-not $rollup -or $rollup.Count -eq 0) { return 'NONE' }
    $conclusions = $rollup | ForEach-Object { $_.conclusion ?? $_.status }
    if ($conclusions -contains 'FAILURE' -or $conclusions -contains 'TIMED_OUT' -or $conclusions -contains 'CANCELLED') { return 'FAILED' }
    if ($conclusions -contains 'IN_PROGRESS' -or $conclusions -contains 'QUEUED' -or $conclusions -contains 'WAITING' -or $conclusions -contains 'PENDING') { return 'PENDING' }
    if ($conclusions | Where-Object { $_ -in @('SUCCESS', 'NEUTRAL', 'SKIPPED') }) { return 'PASSED' }
    return 'NONE'
}

# 5. Map to output objects
$results = $prs | ForEach-Object {
    [PSCustomObject]@{
        PR        = $_.number
        Title     = if ($_.title.Length -gt 60) { $_.title.Substring(0, 57) + '...' } else { $_.title }
        State     = $_.state
        Mergeable = $_.mergeable
        CI        = Get-CiStatus $_.statusCheckRollup
        Files     = $_.changedFiles
        Branch    = $_.headRefName
        Created   = ([datetime]$_.createdAt).ToString('yyyy-MM-dd')
        URL       = $_.url
    }
}

# 6. Output
if ($Format -eq 'Json') {
    # -InputObject keeps a single PR wrapped in an array.
    ConvertTo-Json -InputObject @($results) -Depth 3
} else {
    $results | Format-Table PR, Title, State, Mergeable, CI, Files, Branch, Created -AutoSize
    Write-Output "  URL lookup: pipe with -Format Json or access .URL property directly."
}
