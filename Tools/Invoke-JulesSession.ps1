#Requires -Version 7.0
<#
.SYNOPSIS
    Jules REST API (v1alpha) and CLI wrapper for the PC_AI repository.

.DESCRIPTION
    Full wrapper around every Jules API endpoint and CLI command. Covers all session
    lifecycle operations (create, list, status, approve, message, delete), activity
    inspection (list, get), compound artifact extraction (plan, patch, bash output,
    media), source discovery, batch operations, and Jules CLI pass-throughs.

    Authentication for REST calls uses the X-Goog-Api-Key header populated from
    $env:JULES_API_KEY or a .env file in the repository root.

.PARAMETER Action
    Operation to perform. One of:
        REST actions  : List, New, Status, Approve, Message, Delete,
                        ListActivities, GetActivity, GetPlan, GetPatch,
                        GetBashOutput, GetMedia, ListSources, GetSource
        CLI actions   : Version, ListRepos, ListSessionsCli, NewCli, Pull
        Batch actions : Batch, BatchCli
        Test helper   : __test_load__

.PARAMETER SessionId
    Jules session identifier (required for: Status, Approve, Message, Delete,
    ListActivities, GetActivity, GetPlan, GetPatch, GetBashOutput, GetMedia, Pull).

.PARAMETER ActivityId
    Activity identifier within a session (required for: GetActivity).

.PARAMETER SourceId
    Source identifier (required for: GetSource). Example: sources/github/David-Martel/PC-AI

.PARAMETER Prompt
    Natural-language task description (required for: New, NewCli, Message).

.PARAMETER Prompts
    Array of prompts for batch operations (required for: Batch, BatchCli).

.PARAMETER Branch
    Git branch to target. Default: main.

.PARAMETER Source
    Jules source resource name. Default: sources/github/David-Martel/PC-AI

.PARAMETER Title
    Optional human-readable title for a new session.

.PARAMETER RequirePlanApproval
    When set, the new session will pause and await plan approval before executing.

.PARAMETER AutomationMode
    Post-completion automation. 'None' (default) or 'AutoCreatePR'.

.PARAMETER Format
    Output format for list/table actions. 'Json' (default) or 'Table'.

.PARAMETER State
    Filter sessions by state when listing (optional).

.PARAMETER PageSize
    Maximum number of records returned by list operations. Default: 30.

.PARAMETER Filter
    API-side filter expression for list operations.

.PARAMETER All
    When specified with List, retrieves all pages of results.

.PARAMETER Repo
    Local repository path or identifier for CLI operations. Default: '.'

.PARAMETER Parallel
    Number of parallel sessions to create via NewCli. Default: 1.

.OUTPUTS
    PSCustomObject  — for single-item actions (Status, New, GetActivity, etc.)
    PSCustomObject[]— for list actions (List, ListActivities, ListSources, etc.)
    String          — for Version and CLI pass-throughs
    [void]          — for Delete, Pull

.EXAMPLE
    .\Invoke-JulesSession.ps1 -Action List -Format Table
    Lists all sessions and renders a human-readable table.

.EXAMPLE
    .\Invoke-JulesSession.ps1 -Action New -Prompt 'Add unit tests for Get-DeviceErrors'
    Creates a new Jules session with the given prompt and returns the session object.

.EXAMPLE
    .\Invoke-JulesSession.ps1 -Action Status -SessionId 'abc123'
    Returns the current status of session abc123.

.EXAMPLE
    .\Invoke-JulesSession.ps1 -Action Approve -SessionId 'abc123'
    Approves the generated plan so Jules proceeds with execution.

.EXAMPLE
    .\Invoke-JulesSession.ps1 -Action GetPatch -SessionId 'abc123'
    Extracts changeSet artifacts and writes .patch files to .pcai/jules/patches/.

.EXAMPLE
    .\Invoke-JulesSession.ps1 -Action Batch -Prompts 'Fix lint errors','Add README section'
    Creates two sessions in sequence and records results to .pcai/jules/sessions/.

.EXAMPLE
    .\Invoke-JulesSession.ps1 -Action Version
    Runs 'jules version' and returns the version string.
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [ValidateSet('List','New','Status','Approve','Message','Delete',
                 'ListActivities','GetActivity','GetPlan','GetPatch','GetBashOutput','GetMedia',
                 'ListSources','GetSource',
                 'Pull','ListRepos','ListSessionsCli','NewCli','BatchCli',
                 'Batch','Version','__test_load__')]
    [string]$Action,

    [string]$SessionId,
    [string]$ActivityId,
    [string]$SourceId,
    [string]$Prompt,
    [string[]]$Prompts,
    [string]$Branch = 'main',
    [string]$Source = 'sources/github/David-Martel/PC-AI',
    [string]$Title,
    [switch]$RequirePlanApproval,
    [ValidateSet('None','AutoCreatePR')][string]$AutomationMode = 'None',
    [ValidateSet('Table','Json')][string]$Format = 'Json',
    [ValidateSet('Completed','InProgress','Failed','AwaitingPlanApproval',
                 'AwaitingUserFeedback','Queued','Planning','Paused')]
    [string]$State,
    [int]$PageSize = 30,
    [string]$Filter,
    [switch]$All,
    [string]$Repo = '.',
    [int]$Parallel = 1
)

$ErrorActionPreference = 'Stop'

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
$script:JulesBaseUrl  = 'https://jules.googleapis.com/v1alpha'
$script:RepoRoot      = Split-Path -Parent $PSScriptRoot
$script:PatchDir      = Join-Path $script:RepoRoot '.pcai\jules\patches'
$script:SessionLogDir = Join-Path $script:RepoRoot '.pcai\jules\sessions'

# ---------------------------------------------------------------------------
# Helper: Get-JulesApiKey
#   Reads the Jules API key from (in order):
#   1. $env:JULES_API_KEY
#   2. ~/.machine/*.json files (machine-local secrets store)
#   3. .env file in the repository root
#   4. Bitwarden CLI (bw get notes "JULES_API_KEY")
#   Returns the key string or $null.
# ---------------------------------------------------------------------------
function Get-JulesApiKey {
    [CmdletBinding()]
    [OutputType([string])]
    param()

    # 1. Environment variable (fastest)
    if ($env:JULES_API_KEY) {
        return $env:JULES_API_KEY
    }

    # 2. ~/.machine/*.json files
    $machineDir = Join-Path $HOME '.machine'
    if (Test-Path -LiteralPath $machineDir) {
        foreach ($f in Get-ChildItem -LiteralPath $machineDir -Filter '*.json' -File) {
            try {
                $data = Get-Content -LiteralPath $f.FullName -Raw | ConvertFrom-Json -Depth 5
                if ($data.JULES_API_KEY) { return $data.JULES_API_KEY }
                if ($data.jules_api_key) { return $data.jules_api_key }
            } catch { <# skip malformed files #> }
        }
    }

    # 3. .env file in repository root
    $envFile = Join-Path $script:RepoRoot '.env'
    if (Test-Path -LiteralPath $envFile) {
        foreach ($line in [System.IO.File]::ReadAllLines($envFile)) {
            if ($line.Trim() -match '^JULES_API_KEY\s*=\s*(.+)$') {
                $val = $Matches[1].Trim().Trim('"').Trim("'")
                if ($val) { return $val }
            }
        }
    }

    # 4. Bitwarden CLI (bw)
    if (Get-Command 'bw' -ErrorAction SilentlyContinue) {
        try {
            $bwResult = bw get notes 'JULES_API_KEY' 2>$null
            if ($bwResult -and $bwResult.Trim()) { return $bwResult.Trim() }
        } catch { <# bw not unlocked or item not found #> }
    }

    return $null
}

# ---------------------------------------------------------------------------
# Helper: Get-JulesApiUrl
#   Constructs a Jules REST API URL.
#
#   Patterns:
#     -Endpoint 'sessions'                               → .../sessions
#     -Endpoint 'sessions' -Id 'abc'                    → .../sessions/abc
#     -Endpoint 'sessions' -Id 'abc' -Sub 'activities'  → .../sessions/abc/activities
#     -Endpoint 'sessions' -Id 'abc' -Action 'approvePlan'
#                                                        → .../sessions/abc:approvePlan
# ---------------------------------------------------------------------------
function Get-JulesApiUrl {
    [CmdletBinding()]
    [OutputType([string])]
    param(
        [Parameter(Mandatory)]
        [string]$Endpoint,

        [string]$Id,
        [string]$Sub,
        [string]$UrlAction
    )

    $url = "$script:JulesBaseUrl/$Endpoint"

    if ($Id) {
        if ($UrlAction) {
            # Colon-separated action variant: .../endpoint/id:action
            $url = "$url/$($Id):$UrlAction"
        } elseif ($Sub) {
            # Sub-resource variant: .../endpoint/id/sub
            $url = "$url/$Id/$Sub"
        } else {
            # Plain id: .../endpoint/id
            $url = "$url/$Id"
        }
    }

    return $url
}

# ---------------------------------------------------------------------------
# Helper: New-JulesSessionBody
#   Builds the hashtable sent as the POST body when creating a new session.
# ---------------------------------------------------------------------------
function New-JulesSessionBody {
    [CmdletBinding()]
    [OutputType([hashtable])]
    param(
        [Parameter(Mandatory)]
        [string]$PromptText,

        [Parameter(Mandatory)]
        [string]$SourceName,

        [Parameter(Mandatory)]
        [string]$BranchName,

        [string]$SessionTitle,
        [switch]$PlanApproval,
        [string]$Automation = 'None'
    )

    $body = @{
        prompt = $PromptText
        source = $SourceName
        branch = $BranchName
    }

    if ($SessionTitle) {
        $body['title'] = $SessionTitle
    }

    if ($PlanApproval) {
        $body['requirePlanApproval'] = $true
    }

    # Map PowerShell-friendly enum value to API string
    if ($Automation -eq 'AutoCreatePR') {
        $body['automationMode'] = 'AUTO_CREATE_PR'
    }

    return $body
}

# ---------------------------------------------------------------------------
# Helper: Invoke-JulesApi
#   Wraps Invoke-RestMethod with Jules authentication, retry logic on 429,
#   and descriptive error translation for common HTTP status codes.
# ---------------------------------------------------------------------------
function Invoke-JulesApi {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Url,

        [ValidateSet('GET','POST','DELETE')]
        [string]$Method = 'GET',

        [hashtable]$Body,

        [Parameter(Mandatory)]
        [string]$ApiKey
    )

    $headers = @{
        'X-Goog-Api-Key' = $ApiKey
        'Content-Type'   = 'application/json'
        'Accept'         = 'application/json'
    }

    $invokeParams = @{
        Uri     = $Url
        Method  = $Method
        Headers = $headers
    }

    if ($Body -and $Method -ne 'GET') {
        $invokeParams['Body'] = ($Body | ConvertTo-Json -Depth 10 -Compress)
    }

    # Exponential backoff on 429 — 3 attempts with 2s, 4s, 8s delays
    $maxAttempts = 3
    $delaySeconds = 2

    for ($attempt = 1; $attempt -le $maxAttempts; $attempt++) {
        try {
            $response = Invoke-RestMethod @invokeParams
            return $response
        }
        catch {
            $statusCode = $null

            # Extract HTTP status code — differs between PS 5.1 and PS 7+
            if ($_.Exception -is [System.Net.WebException]) {
                $webEx = [System.Net.WebException]$_.Exception
                if ($webEx.Response -is [System.Net.HttpWebResponse]) {
                    $statusCode = [int]([System.Net.HttpWebResponse]$webEx.Response).StatusCode
                }
            } elseif ($_.Exception.PSObject.Properties['Response'] -and
                      $_.Exception.Response -ne $null) {
                # PS 7+ HttpRequestException
                $statusCode = [int]$_.Exception.Response.StatusCode
            }

            # Retry on 429 (rate limit)
            if ($statusCode -eq 429 -and $attempt -lt $maxAttempts) {
                $sleepMs = $delaySeconds * 1000
                Write-Verbose "Jules API rate-limited (429). Retrying in ${delaySeconds}s (attempt $attempt/$maxAttempts)..."
                Start-Sleep -Milliseconds $sleepMs
                $delaySeconds *= 2
                continue
            }

            # Translate common HTTP errors to descriptive messages
            $message = switch ($statusCode) {
                401 { "Jules API authentication failed. Verify that JULES_API_KEY is set and valid." }
                403 { "Jules API access denied (403). Ensure the repository is connected in Jules and your key has permission." }
                404 { "Jules resource not found (404): $Url" }
                500 { "Jules server error (500). The Jules service may be temporarily unavailable." }
                $null { "Jules API request failed: $($_.Exception.Message)" }
                default { "Jules API error ($statusCode): $($_.Exception.Message)" }
            }

            throw [System.Exception]$message
        }
    }
}

# ---------------------------------------------------------------------------
# Helper: Invoke-JulesCli
#   Wraps the 'jules' CLI binary. Verifies it is on PATH, executes the
#   supplied arguments, and returns stdout as a string.
# ---------------------------------------------------------------------------
function Invoke-JulesCli {
    [CmdletBinding()]
    [OutputType([string])]
    param(
        [Parameter(Mandatory)]
        [string[]]$Arguments
    )

    if (-not (Get-Command 'jules' -ErrorAction SilentlyContinue)) {
        throw "The 'jules' CLI is not available on PATH. Install Jules CLI and ensure it is in your PATH."
    }

    $output = & jules @Arguments 2>&1
    $joined = $output -join [System.Environment]::NewLine
    if ($LASTEXITCODE -ne 0) {
        throw "jules CLI exited with code ${LASTEXITCODE}: $joined"
    }

    return $joined
}

# ---------------------------------------------------------------------------
# Helper: Format-JulesSessionTable
#   Converts raw session API objects into table-friendly PSCustomObjects.
#   Extracts SessionId from the 'name' field (last path segment), Created
#   from createTime, and PRUrl from the first pullRequest output if present.
# ---------------------------------------------------------------------------
function Format-JulesSessionTable {
    [CmdletBinding()]
    [OutputType([PSCustomObject[]])]
    param(
        [Parameter(Mandatory, ValueFromPipeline)]
        [AllowNull()]
        [object[]]$Sessions
    )

    process {
        foreach ($session in $Sessions) {
            if ($null -eq $session) { continue }

            # Extract the final path segment from the 'name' field
            $sessionId = ''
            if ($session.name) {
                $sessionId = $session.name -replace '^.*/', ''
            }

            # Pull request URL from the first output entry, if available
            $prUrl = ''
            if ($session.PSObject.Properties['outputs'] -and $session.outputs) {
                $first = @($session.outputs)[0]
                if ($first -and $first.PSObject.Properties['pullRequest'] -and
                    $first.pullRequest -and $first.pullRequest.PSObject.Properties['url']) {
                    $prUrl = [string]$first.pullRequest.url
                }
            }

            [PSCustomObject]@{
                SessionId = $sessionId
                Title     = if ($session.PSObject.Properties['title']) { [string]$session.title } else { '' }
                State     = if ($session.PSObject.Properties['state']) { [string]$session.state } else { '' }
                Created   = if ($session.PSObject.Properties['createTime']) { [string]$session.createTime } else { '' }
                PRUrl     = $prUrl
            }
        }
    }
}

# ---------------------------------------------------------------------------
# Helper: Assert-SessionId / Assert-Prompt / Assert-ActivityId
#   Lightweight validators that throw a descriptive error when a required
#   parameter is missing for the given action.
# ---------------------------------------------------------------------------
function Assert-JulesParam {
    param(
        [string]$Value,
        [string]$ParamName,
        [string]$ForAction
    )
    if (-not $Value) {
        throw "-$ParamName is required for the '$ForAction' action."
    }
}

# ---------------------------------------------------------------------------
# Helper: Ensure-JulesDirectory
#   Creates a directory path if it does not already exist.
# ---------------------------------------------------------------------------
function Ensure-JulesDirectory {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
    }
}

# ---------------------------------------------------------------------------
# Helper: Get-RequiredApiKey
#   Retrieves the API key and throws a descriptive error if missing.
# ---------------------------------------------------------------------------
function Get-RequiredApiKey {
    param([string]$ForAction)
    $key = Get-JulesApiKey
    if (-not $key) {
        throw "JULES_API_KEY is not set. Export it as an environment variable or add JULES_API_KEY=<key> to the .env file in the repository root. (Required for '$ForAction' action.)"
    }
    return $key
}

# ---------------------------------------------------------------------------
# __test_load__ — expose internal functions in the global scope so that
# Pester tests dot-sourcing this script with -Action __test_load__ can
# call them directly.
# ---------------------------------------------------------------------------
if ($Action -eq '__test_load__') {
    # Functions are already defined in this script scope; promote them to the
    # global scope so tests loaded via . .\Invoke-JulesSession.ps1 -Action __test_load__
    # can call them without further ceremony.
    $functionsToExport = @(
        'Get-JulesApiKey',
        'Get-JulesApiUrl',
        'New-JulesSessionBody',
        'Invoke-JulesApi',
        'Invoke-JulesCli',
        'Format-JulesSessionTable',
        'Get-RequiredApiKey',
        'Ensure-JulesDirectory',
        'Assert-JulesParam'
    )
    foreach ($fnName in $functionsToExport) {
        $fn = Get-Item "function:$fnName" -ErrorAction SilentlyContinue
        if ($fn) {
            Set-Item -Path "function:global:$fnName" -Value $fn.ScriptBlock
        }
    }

    # Also export script-scope constants so tests can verify URLs etc.
    $global:JulesBaseUrl  = $script:JulesBaseUrl
    $global:JulesRepoRoot = $script:RepoRoot

    return
}

# ===========================================================================
# ACTION DISPATCH
# ===========================================================================
switch ($Action) {

    # -----------------------------------------------------------------------
    # Source discovery
    # -----------------------------------------------------------------------
    'ListSources' {
        $apiKey = Get-RequiredApiKey -ForAction $Action
        $url    = Get-JulesApiUrl -Endpoint 'sources'
        $result = Invoke-JulesApi -Url $url -Method 'GET' -ApiKey $apiKey
        if ($Format -eq 'Table') {
            $result.sources | Format-Table -AutoSize
        } else {
            $result.sources | ConvertTo-Json -Depth 10
        }
    }

    'GetSource' {
        $apiKey = Get-RequiredApiKey -ForAction $Action
        Assert-JulesParam -Value $SourceId -ParamName 'SourceId' -ForAction $Action
        $url    = Get-JulesApiUrl -Endpoint 'sources' -Id $SourceId
        $result = Invoke-JulesApi -Url $url -Method 'GET' -ApiKey $apiKey
        if ($Format -eq 'Table') {
            $result | Format-List
        } else {
            $result | ConvertTo-Json -Depth 10
        }
    }

    # -----------------------------------------------------------------------
    # Session list
    # -----------------------------------------------------------------------
    'List' {
        $apiKey = Get-RequiredApiKey -ForAction $Action

        $allSessions = [System.Collections.Generic.List[object]]::new()
        $pageToken   = $null

        do {
            $queryParts = @("pageSize=$PageSize")
            if ($Filter)    { $queryParts += "filter=$([Uri]::EscapeDataString($Filter))" }
            if ($State)     { $queryParts += "filter=state=$State" }
            if ($pageToken) { $queryParts += "pageToken=$([Uri]::EscapeDataString($pageToken))" }

            $qs  = if ($queryParts.Count -gt 0) { '?' + ($queryParts -join '&') } else { '' }
            $url = (Get-JulesApiUrl -Endpoint 'sessions') + $qs

            $result = Invoke-JulesApi -Url $url -Method 'GET' -ApiKey $apiKey

            if ($result.PSObject.Properties['sessions'] -and $result.sessions) {
                foreach ($s in $result.sessions) { $allSessions.Add($s) }
            }

            $pageToken = if ($result.PSObject.Properties['nextPageToken']) {
                $result.nextPageToken
            } else { $null }

        } while ($All -and $pageToken)

        $output = @($allSessions)

        if ($Format -eq 'Table') {
            Format-JulesSessionTable -Sessions $output | Format-Table -AutoSize
        } else {
            $output | ConvertTo-Json -Depth 10
        }
    }

    # -----------------------------------------------------------------------
    # Create session
    # -----------------------------------------------------------------------
    'New' {
        $apiKey = Get-RequiredApiKey -ForAction $Action
        Assert-JulesParam -Value $Prompt -ParamName 'Prompt' -ForAction $Action

        $body   = New-JulesSessionBody -PromptText $Prompt -SourceName $Source `
                      -BranchName $Branch -SessionTitle $Title `
                      -PlanApproval:$RequirePlanApproval -Automation $AutomationMode
        $url    = Get-JulesApiUrl -Endpoint 'sessions'
        $result = Invoke-JulesApi -Url $url -Method 'POST' -Body $body -ApiKey $apiKey
        $result
    }

    # -----------------------------------------------------------------------
    # Session status
    # -----------------------------------------------------------------------
    'Status' {
        $apiKey = Get-RequiredApiKey -ForAction $Action
        Assert-JulesParam -Value $SessionId -ParamName 'SessionId' -ForAction $Action

        $url    = Get-JulesApiUrl -Endpoint 'sessions' -Id $SessionId
        $result = Invoke-JulesApi -Url $url -Method 'GET' -ApiKey $apiKey
        $result
    }

    # -----------------------------------------------------------------------
    # Delete session
    # -----------------------------------------------------------------------
    'Delete' {
        $apiKey = Get-RequiredApiKey -ForAction $Action
        Assert-JulesParam -Value $SessionId -ParamName 'SessionId' -ForAction $Action

        $url = Get-JulesApiUrl -Endpoint 'sessions' -Id $SessionId
        Invoke-JulesApi -Url $url -Method 'DELETE' -ApiKey $apiKey | Out-Null
        Write-Verbose "Session '$SessionId' deleted."
    }

    # -----------------------------------------------------------------------
    # Approve plan
    # -----------------------------------------------------------------------
    'Approve' {
        $apiKey = Get-RequiredApiKey -ForAction $Action
        Assert-JulesParam -Value $SessionId -ParamName 'SessionId' -ForAction $Action

        $url    = Get-JulesApiUrl -Endpoint 'sessions' -Id $SessionId -UrlAction 'approvePlan'
        $result = Invoke-JulesApi -Url $url -Method 'POST' -ApiKey $apiKey
        $result
    }

    # -----------------------------------------------------------------------
    # Send message to session
    # -----------------------------------------------------------------------
    'Message' {
        $apiKey = Get-RequiredApiKey -ForAction $Action
        Assert-JulesParam -Value $SessionId -ParamName 'SessionId' -ForAction $Action
        Assert-JulesParam -Value $Prompt    -ParamName 'Prompt'    -ForAction $Action

        $body   = @{ message = $Prompt }
        $url    = Get-JulesApiUrl -Endpoint 'sessions' -Id $SessionId -UrlAction 'sendMessage'
        $result = Invoke-JulesApi -Url $url -Method 'POST' -Body $body -ApiKey $apiKey
        $result
    }

    # -----------------------------------------------------------------------
    # List activities for a session
    # -----------------------------------------------------------------------
    'ListActivities' {
        $apiKey = Get-RequiredApiKey -ForAction $Action
        Assert-JulesParam -Value $SessionId -ParamName 'SessionId' -ForAction $Action

        $url    = Get-JulesApiUrl -Endpoint 'sessions' -Id $SessionId -Sub 'activities'
        $result = Invoke-JulesApi -Url $url -Method 'GET' -ApiKey $apiKey

        $activities = if ($result.PSObject.Properties['activities']) { $result.activities } else { @() }

        if ($Format -eq 'Table') {
            $activities | Format-Table -AutoSize
        } else {
            $activities | ConvertTo-Json -Depth 10
        }
    }

    # -----------------------------------------------------------------------
    # Get a specific activity
    # -----------------------------------------------------------------------
    'GetActivity' {
        $apiKey = Get-RequiredApiKey -ForAction $Action
        Assert-JulesParam -Value $SessionId  -ParamName 'SessionId'  -ForAction $Action
        Assert-JulesParam -Value $ActivityId -ParamName 'ActivityId' -ForAction $Action

        $url    = Get-JulesApiUrl -Endpoint 'sessions' -Id $SessionId -Sub "activities/$ActivityId"
        $result = Invoke-JulesApi -Url $url -Method 'GET' -ApiKey $apiKey
        $result
    }

    # -----------------------------------------------------------------------
    # Compound: extract plan from activities
    # -----------------------------------------------------------------------
    'GetPlan' {
        $apiKey = Get-RequiredApiKey -ForAction $Action
        Assert-JulesParam -Value $SessionId -ParamName 'SessionId' -ForAction $Action

        $url    = Get-JulesApiUrl -Endpoint 'sessions' -Id $SessionId -Sub 'activities'
        $result = Invoke-JulesApi -Url $url -Method 'GET' -ApiKey $apiKey

        $activities = if ($result.PSObject.Properties['activities']) { $result.activities } else { @() }

        # Find the planGenerated event
        $planActivity = $activities | Where-Object {
            $_.PSObject.Properties['eventType'] -and $_.eventType -eq 'planGenerated'
        } | Select-Object -Last 1

        if (-not $planActivity) {
            $planActivity = $activities | Where-Object {
                $_.PSObject.Properties['type'] -and $_.type -eq 'planGenerated'
            } | Select-Object -Last 1
        }

        if (-not $planActivity) {
            Write-Warning "No planGenerated activity found for session '$SessionId'."
            return @()
        }

        # Extract steps — structure may vary by API version
        $steps = @()
        if ($planActivity.PSObject.Properties['plan'] -and $planActivity.plan) {
            $steps = if ($planActivity.plan.PSObject.Properties['steps']) {
                $planActivity.plan.steps
            } else { @($planActivity.plan) }
        } elseif ($planActivity.PSObject.Properties['steps']) {
            $steps = $planActivity.steps
        } else {
            $steps = @($planActivity)
        }

        $steps
    }

    # -----------------------------------------------------------------------
    # Compound: extract patch files from changeSet artifacts
    # -----------------------------------------------------------------------
    'GetPatch' {
        $apiKey = Get-RequiredApiKey -ForAction $Action
        Assert-JulesParam -Value $SessionId -ParamName 'SessionId' -ForAction $Action

        $url    = Get-JulesApiUrl -Endpoint 'sessions' -Id $SessionId -Sub 'activities'
        $result = Invoke-JulesApi -Url $url -Method 'GET' -ApiKey $apiKey

        $activities = if ($result.PSObject.Properties['activities']) { $result.activities } else { @() }

        # Collect all changeSet artifacts
        $changeSets = $activities | Where-Object {
            ($_.PSObject.Properties['artifactType'] -and $_.artifactType -eq 'changeSet') -or
            ($_.PSObject.Properties['type']         -and $_.type         -eq 'changeSet')
        }

        if (-not $changeSets) {
            # Also check inside nested artifacts property
            $changeSets = $activities | ForEach-Object {
                if ($_.PSObject.Properties['artifacts']) {
                    $_.artifacts | Where-Object {
                        ($_.PSObject.Properties['type'] -and $_.type -eq 'changeSet') -or
                        ($_.PSObject.Properties['artifactType'] -and $_.artifactType -eq 'changeSet')
                    }
                }
            }
        }

        if (-not $changeSets) {
            Write-Warning "No changeSet artifacts found for session '$SessionId'."
            return @()
        }

        Ensure-JulesDirectory -Path $script:PatchDir

        $writtenFiles = [System.Collections.Generic.List[string]]::new()
        $index = 0

        foreach ($cs in $changeSets) {
            $index++
            $patchContent = ''

            if ($cs.PSObject.Properties['content']) {
                $patchContent = [string]$cs.content
            } elseif ($cs.PSObject.Properties['patch']) {
                $patchContent = [string]$cs.patch
            } elseif ($cs.PSObject.Properties['diff']) {
                $patchContent = [string]$cs.diff
            }

            $patchName = if ($cs.PSObject.Properties['filename'] -and $cs.filename) {
                $cs.filename -replace '[\\/:*?"<>|]', '_'
            } else {
                "$($SessionId -replace '[\\/:*?"<>|]', '_')_$('{0:D3}' -f $index).patch"
            }

            $patchPath = Join-Path $script:PatchDir $patchName
            [System.IO.File]::WriteAllText($patchPath, $patchContent, [System.Text.Encoding]::UTF8)
            $writtenFiles.Add($patchPath)
            Write-Verbose "Wrote patch: $patchPath"
        }

        @($writtenFiles)
    }

    # -----------------------------------------------------------------------
    # Compound: extract bash output artifacts
    # -----------------------------------------------------------------------
    'GetBashOutput' {
        $apiKey = Get-RequiredApiKey -ForAction $Action
        Assert-JulesParam -Value $SessionId -ParamName 'SessionId' -ForAction $Action

        $url    = Get-JulesApiUrl -Endpoint 'sessions' -Id $SessionId -Sub 'activities'
        $result = Invoke-JulesApi -Url $url -Method 'GET' -ApiKey $apiKey

        $activities = if ($result.PSObject.Properties['activities']) { $result.activities } else { @() }

        $bashOutputs = [System.Collections.Generic.List[PSCustomObject]]::new()

        foreach ($activity in $activities) {
            $isBashtType = ($activity.PSObject.Properties['artifactType'] -and $activity.artifactType -eq 'bashOutput') -or
                           ($activity.PSObject.Properties['type']         -and $activity.type         -eq 'bashOutput')

            if ($isBashtType) {
                $bashOutputs.Add([PSCustomObject]@{
                    Command  = if ($activity.PSObject.Properties['command'])  { [string]$activity.command  } else { '' }
                    Output   = if ($activity.PSObject.Properties['output'])   { [string]$activity.output   } else { '' }
                    ExitCode = if ($activity.PSObject.Properties['exitCode']) { [int]$activity.exitCode    } else { 0  }
                })
                continue
            }

            # Also check nested artifacts
            if ($activity.PSObject.Properties['artifacts']) {
                foreach ($artifact in $activity.artifacts) {
                    $isBash = ($artifact.PSObject.Properties['type'] -and $artifact.type -eq 'bashOutput') -or
                              ($artifact.PSObject.Properties['artifactType'] -and $artifact.artifactType -eq 'bashOutput')
                    if ($isBash) {
                        $bashOutputs.Add([PSCustomObject]@{
                            Command  = if ($artifact.PSObject.Properties['command'])  { [string]$artifact.command  } else { '' }
                            Output   = if ($artifact.PSObject.Properties['output'])   { [string]$artifact.output   } else { '' }
                            ExitCode = if ($artifact.PSObject.Properties['exitCode']) { [int]$artifact.exitCode    } else { 0  }
                        })
                    }
                }
            }
        }

        @($bashOutputs)
    }

    # -----------------------------------------------------------------------
    # Compound: extract media artifacts
    # -----------------------------------------------------------------------
    'GetMedia' {
        $apiKey = Get-RequiredApiKey -ForAction $Action
        Assert-JulesParam -Value $SessionId -ParamName 'SessionId' -ForAction $Action

        $url    = Get-JulesApiUrl -Endpoint 'sessions' -Id $SessionId -Sub 'activities'
        $result = Invoke-JulesApi -Url $url -Method 'GET' -ApiKey $apiKey

        $activities = if ($result.PSObject.Properties['activities']) { $result.activities } else { @() }

        $mediaItems = [System.Collections.Generic.List[PSCustomObject]]::new()

        foreach ($activity in $activities) {
            $checkItems = @($activity)
            if ($activity.PSObject.Properties['artifacts']) {
                $checkItems = @($activity.artifacts)
            }

            foreach ($item in $checkItems) {
                $isMedia = ($item.PSObject.Properties['artifactType'] -and $item.artifactType -eq 'media') -or
                           ($item.PSObject.Properties['type']         -and $item.type         -eq 'media')
                if ($isMedia) {
                    $mediaItems.Add([PSCustomObject]@{
                        MimeType  = if ($item.PSObject.Properties['mimeType'])  { [string]$item.mimeType  } else { '' }
                        Base64    = if ($item.PSObject.Properties['base64'])    { [string]$item.base64    } else {
                                    if ($item.PSObject.Properties['data'])      { [string]$item.data      } else { '' } }
                        Filename  = if ($item.PSObject.Properties['filename'])  { [string]$item.filename  } else { '' }
                    })
                }
            }
        }

        @($mediaItems)
    }

    # -----------------------------------------------------------------------
    # Batch: create multiple sessions via REST API
    # -----------------------------------------------------------------------
    'Batch' {
        $apiKey = Get-RequiredApiKey -ForAction $Action
        if (-not $Prompts -or $Prompts.Count -eq 0) {
            throw "-Prompts is required for the 'Batch' action (pass one or more prompt strings)."
        }

        Ensure-JulesDirectory -Path $script:SessionLogDir

        $timestamp  = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ')
        $logPath    = Join-Path $script:SessionLogDir "$timestamp.json"
        $results    = [System.Collections.Generic.List[PSCustomObject]]::new()

        foreach ($promptText in $Prompts) {
            Write-Verbose "Batch: creating session for prompt: $promptText"
            try {
                $body   = New-JulesSessionBody -PromptText $promptText -SourceName $Source `
                              -BranchName $Branch -SessionTitle $Title `
                              -PlanApproval:$RequirePlanApproval -Automation $AutomationMode
                $url    = Get-JulesApiUrl -Endpoint 'sessions'
                $result = Invoke-JulesApi -Url $url -Method 'POST' -Body $body -ApiKey $apiKey

                $results.Add([PSCustomObject]@{
                    Prompt    = $promptText
                    SessionId = ($result.name -replace '^.*/', '')
                    Name      = $result.name
                    State     = if ($result.PSObject.Properties['state']) { $result.state } else { '' }
                    Error     = $null
                    Raw       = $result
                })
            }
            catch {
                $results.Add([PSCustomObject]@{
                    Prompt    = $promptText
                    SessionId = $null
                    Name      = $null
                    State     = 'Error'
                    Error     = $_.Exception.Message
                    Raw       = $null
                })
                Write-Warning "Batch session failed for prompt '$promptText': $($_.Exception.Message)"
            }
        }

        $output = @($results)
        $output | ConvertTo-Json -Depth 10 | Set-Content -Path $logPath -Encoding UTF8
        Write-Verbose "Batch results written to: $logPath"

        $output
    }

    # -----------------------------------------------------------------------
    # CLI: jules version
    # -----------------------------------------------------------------------
    'Version' {
        Invoke-JulesCli -Arguments @('version')
    }

    # -----------------------------------------------------------------------
    # CLI: list connected repositories
    # -----------------------------------------------------------------------
    'ListRepos' {
        Invoke-JulesCli -Arguments @('remote', 'list', '--repo')
    }

    # -----------------------------------------------------------------------
    # CLI: list sessions via CLI
    # -----------------------------------------------------------------------
    'ListSessionsCli' {
        Invoke-JulesCli -Arguments @('remote', 'list', '--session')
    }

    # -----------------------------------------------------------------------
    # CLI: create a new session via CLI
    # -----------------------------------------------------------------------
    'NewCli' {
        Assert-JulesParam -Value $Prompt -ParamName 'Prompt' -ForAction $Action

        $cliArgs = @('remote', 'new', '--repo', $Repo, '--session', $Prompt)

        if ($Parallel -gt 1) {
            $cliArgs += @('--parallel', [string]$Parallel)
        }

        Invoke-JulesCli -Arguments $cliArgs
    }

    # -----------------------------------------------------------------------
    # CLI: pull session results
    # -----------------------------------------------------------------------
    'Pull' {
        Assert-JulesParam -Value $SessionId -ParamName 'SessionId' -ForAction $Action
        Invoke-JulesCli -Arguments @('remote', 'pull', '--session', $SessionId)
    }

    # -----------------------------------------------------------------------
    # CLI batch: create multiple sessions via CLI
    # -----------------------------------------------------------------------
    'BatchCli' {
        if (-not $Prompts -or $Prompts.Count -eq 0) {
            throw "-Prompts is required for the 'BatchCli' action."
        }

        Ensure-JulesDirectory -Path $script:SessionLogDir

        $timestamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ')
        $logPath   = Join-Path $script:SessionLogDir "cli_$timestamp.json"
        $results   = [System.Collections.Generic.List[PSCustomObject]]::new()

        foreach ($promptText in $Prompts) {
            Write-Verbose "BatchCli: creating session for prompt: $promptText"
            try {
                $cliArgs = @('remote', 'new', '--repo', $Repo, '--session', $promptText)
                if ($Parallel -gt 1) {
                    $cliArgs += @('--parallel', [string]$Parallel)
                }
                $output = Invoke-JulesCli -Arguments $cliArgs

                $results.Add([PSCustomObject]@{
                    Prompt = $promptText
                    Output = $output
                    Error  = $null
                })
            }
            catch {
                $results.Add([PSCustomObject]@{
                    Prompt = $promptText
                    Output = $null
                    Error  = $_.Exception.Message
                })
                Write-Warning "BatchCli session failed for prompt '$promptText': $($_.Exception.Message)"
            }
        }

        $output = @($results)
        $output | ConvertTo-Json -Depth 10 | Set-Content -Path $logPath -Encoding UTF8
        Write-Verbose "BatchCli results written to: $logPath"

        $output
    }
}
