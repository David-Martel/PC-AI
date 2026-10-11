BeforeAll {
    $script:source = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
    $script:base = Join-Path $TestDrive 'GitOpsFixtures'
    $null = New-Item -ItemType Directory -Path $script:base -Force
    $script:child = Join-Path $script:base 'fixture-child.ps1'
    $childScript = @'
[CmdletBinding()]
param([string]$SourceRoot, [string]$RepoRoot, [string]$OutputDirectory, [string]$ScenarioPath,
    [string]$CallsPath, [string]$Script = 'Invoke-GitOpsMonitors.ps1', [string]$BusUrl)
$ErrorActionPreference = 'Stop'
$global:fixtureScenario = Get-Content -LiteralPath $ScenarioPath -Raw | ConvertFrom-Json -AsHashtable
$global:fixtureCalls = $CallsPath
$global:fixtureRoot = $RepoRoot
$env:AGENT_BUS_SERVER_CANDIDATES = '[{"url":"https://fixture.example","role":"cloud","token_env":"FIXTURE_SECRET"}]'
$env:FIXTURE_SECRET = 'fixture-secret-DO-NOT-LOG'
function global:Invoke-VigilBoundedProcess {
    param([string]$FilePath, [string[]]$Arguments, [string]$WorkingDirectory, [int]$TimeoutSeconds)
    $operation = if ($FilePath -eq 'gh') {
        if ($Arguments[0] -eq 'repo') { 'repository' }
        elseif ($Arguments[0] -eq 'run') { 'runs' }
        elseif ($Arguments[1] -match 'actions/workflows') { 'workflows' }
        elseif ($Arguments[1] -match 'billing') { 'billing' }
        elseif ($Arguments[1] -match '^users/') { 'billing-owner' }
        elseif ($Arguments[1] -match 'dependabot') { 'dependabot' }
        elseif ($Arguments[1] -match 'code-scanning') { 'code-scanning' }
        elseif ($Arguments[1] -match '/reviews') { 'reviews' }
        else { 'pulls' }
    } else { 'send' }
    if ($WorkingDirectory -ne $global:fixtureRoot -or $TimeoutSeconds -ne 2) { throw 'Unexpected request directory or timeout.' }
    [IO.File]::AppendAllText($global:fixtureCalls, (([pscustomobject]@{
        pid = $PID; operation = $operation; file = $FilePath; arguments = $Arguments
        candidate_config_preserved = ($env:AGENT_BUS_SERVER_CANDIDATES -match 'fixture.example')
    } | ConvertTo-Json -Depth 5 -Compress) + [Environment]::NewLine))
    if ($operation -eq 'runs' -and $global:fixtureScenario.ContainsKey('hold_seconds')) { Start-Sleep -Seconds $global:fixtureScenario.hold_seconds }
    $defaults = @{ repository = '{"nameWithOwner":"fixture/repo"}'; runs = '[]'; workflows = '{"workflows":[]}'; 'billing-owner' = '{"login":"fixture","type":"User"}'; billing = '{"usageItems":[]}'; dependabot = '[[]]'; 'code-scanning' = '[]'; pulls = '[]'; reviews = '[]'; send = '{"ok":true}' }
    $selected = if ($global:fixtureScenario.ContainsKey($operation)) { $global:fixtureScenario[$operation] } elseif ($operation -ne 'send' -and $global:fixtureScenario.ContainsKey('all_gh')) { $global:fixtureScenario.all_gh } else { @{ ExitCode = 0; Stdout = $defaults[$operation]; Stderr = '' } }
    if ($selected.Throw) { throw 'fixture-secret-DO-NOT-LOG C:/private/credential-file' }
    [pscustomobject]@{ ExitCode = $selected.ExitCode; Stdout = $selected.Stdout; Stderr = $selected.Stderr }
}
function global:Invoke-CimMethod {
    param([string]$ClassName, [string]$MethodName, [hashtable]$Arguments, [string]$ErrorAction)
    if ($ClassName -ne 'Win32_Process' -or $MethodName -ne 'Create') { throw 'Unexpected native launch method.' }
    # Project only known fixture presence/flags; never serialize the environment block.
    $startup = $Arguments.ProcessStartupInformation
    $variables = @($startup.EnvironmentVariables)
    [IO.File]::AppendAllText($global:fixtureCalls, (([pscustomobject]@{
        operation = 'launch'; arguments = @{ CurrentDirectory = $Arguments.CurrentDirectory; CommandLine = $Arguments.CommandLine }
        create_flags = $startup.CreateFlags; show_window = $startup.ShowWindow
        fixture_secret_preserved = ($variables -contains 'FIXTURE_SECRET=fixture-secret-DO-NOT-LOG')
        candidate_config_preserved = (@($variables | Where-Object { $_ -like 'AGENT_BUS_SERVER_CANDIDATES=*fixture.example*' }).Count -eq 1)
    } | ConvertTo-Json -Compress) + [Environment]::NewLine))
    if ($global:fixtureScenario.launch_null) { return $null }
    if ($global:fixtureScenario.launch_throw) { throw 'fixture-secret-DO-NOT-LOG C:/private/credential-file' }
    [pscustomobject]@{ ReturnValue = if ($global:fixtureScenario.ContainsKey('launch_code')) { $global:fixtureScenario.launch_code } else { 0 }; ProcessId = 12345 }
}
$parameters = @{ Repo = 'fixture/repo'; RepoRoot = $RepoRoot; OutputDirectory = $OutputDirectory; RequestTimeoutSeconds = 2; Quiet = $true }
if ($Script -eq 'Invoke-GitOpsMonitors.ps1') { $parameters.Remove('Quiet'); $parameters.NoDelay = $true }
if ($BusUrl) { $parameters.BusUrl = $BusUrl }
if ($Script -eq 'Start-GitOpsMonitors.ps1') { $parameters = @{} }
& (Join-Path $SourceRoot "Tools/GitOps/$Script") @parameters
exit $LASTEXITCODE
'@
    [IO.File]::WriteAllText($script:child, $childScript, [Text.UTF8Encoding]::new($false))
    function Start-Fixture {
        param([hashtable]$Scenario = @{}, [string]$Script = 'Invoke-GitOpsMonitors.ps1', [string]$Root, [string]$Calls, [string]$BusUrl, [string]$SourceRoot = $script:source)
        $case = Join-Path $script:base ('case-' + [guid]::NewGuid().ToString('N'))
        $null = New-Item -ItemType Directory -Path $case -Force
        if (-not $Root) { $Root = $case }
        if (-not $Calls) { $Calls = Join-Path $case 'calls.jsonl' }
        $scenarioPath = Join-Path $case 'scenario.json'
        $Scenario | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $scenarioPath
        $info = [Diagnostics.ProcessStartInfo]::new((Join-Path $PSHOME $(if ($IsWindows) { 'pwsh.exe' } else { 'pwsh' })))
        $info.UseShellExecute = $false
        $info.CreateNoWindow = $true
        $info.RedirectStandardOutput = $true
        $info.RedirectStandardError = $true
        $info.WorkingDirectory = $script:base
        foreach ($argument in @('-NoLogo', '-NoProfile', '-File', $script:child, '-SourceRoot', $SourceRoot, '-RepoRoot', $Root, '-OutputDirectory', $case, '-ScenarioPath', $scenarioPath, '-CallsPath', $Calls, '-Script', $Script)) { $info.ArgumentList.Add($argument) }
        if ($BusUrl) { $info.ArgumentList.Add('-BusUrl'); $info.ArgumentList.Add($BusUrl) }
        $process = [Diagnostics.Process]::new()
        $process.StartInfo = $info
        $null = $process.Start()
        [pscustomobject]@{ Process = $process; Stdout = $process.StandardOutput.ReadToEndAsync(); Stderr = $process.StandardError.ReadToEndAsync(); Directory = $case; Root = $Root; CallsPath = $Calls }
    }
    function Complete-Fixture {
        param($Fixture)
        try {
            if (-not $Fixture.Process.WaitForExit(20000)) { $Fixture.Process.Kill($true); throw 'Fixture exceeded20s.' }
            $calls = if (Test-Path $Fixture.CallsPath) { @(Get-Content $Fixture.CallsPath | ForEach-Object { $_ | ConvertFrom-Json }) } else { @() }
            $report = Get-ChildItem $Fixture.Directory -Filter '*.json' | Where-Object Name -NE 'scenario.json' | Sort-Object Name | ForEach-Object { [pscustomobject]@{ Name = $_.Name; Value = Get-Content $_.FullName -Raw | ConvertFrom-Json } }
            [pscustomobject]@{ ExitCode = $Fixture.Process.ExitCode; Stdout = $Fixture.Stdout.GetAwaiter().GetResult(); Stderr = $Fixture.Stderr.GetAwaiter().GetResult(); Calls = $calls; Reports = @($report); Directory = $Fixture.Directory }
        }
        finally { $Fixture.Process.Dispose() }
    }
    function Get-Combined($Result) { ($Result.Reports | Where-Object Name -Like 'monitors-latest-*').Value }
    function Wait-Owner($Fixture) {
        $watch = [Diagnostics.Stopwatch]::StartNew()
        while (-not (Test-Path $Fixture.CallsPath)) { if ($watch.Elapsed.TotalSeconds -gt 8) { throw 'Owner did not enter actual worker.' }; Start-Sleep -Milliseconds 50 }
    }
    $script:findings = @{ runs = @{ ExitCode = 0; Stdout = '[{"databaseId":42,"status":"completed","conclusion":"failure","workflowName":"fixture-ci"}]'; Stderr = '' }; dependabot = @{ ExitCode = 0; Stdout = '[[{"security_advisory":{"severity":"high","summary":"fixture advisory"},"html_url":"https://fixture.example/advisory"}]]'; Stderr = '' } }
}
Describe 'Actual GitOps scripts with external requests stubbed' {
    # Protects: user-owned repositories use the supported usage API.
    # Detects: retired endpoint and obsolete allowance schema poisoning workflow health.
    # Needs: the complete monitor with bounded API collaborator responses.
    # Breadcrumb: GitHub billing endpoint retirement, 2025-09-26.
    It 'uses current-month user billing usage and leaves quota unknown' -Tag BillingMigration {
        $r = Complete-Fixture (Start-Fixture -Script Watch-WorkflowHealth.ps1)
        $w = ($r.Reports | Where-Object Name -Like 'workflow-health-*').Value
        $now = [datetime]::UtcNow
        $r.ExitCode | Should -Be 0; $w.inspection_status | Should -Be Available
        $w.workflow_inspection_status | Should -Be Available; $w.billing.quota_status | Should -Be Unknown
        ($r.Calls | Where-Object operation -EQ billing).arguments[1] | Should -Be ("users/fixture/settings/billing/usage?year=$($now.Year)&month=$($now.Month)")
        @($r.Calls | Where-Object operation -EQ billing-owner).Count | Should -Be 1
        ($w.billing | ConvertTo-Json) | Should -Not -Match 'included|used|quantity|Amount'
    }
    # Protects: billing is queried for the repository's organization rather than the token user.
    # Detects: organization ownership incorrectly routed to the user endpoint.
    # Needs: a real monitor invocation with an Organization owner response.
    # Breadcrumb: GET /organizations/{org}/settings/billing/usage.
    It 'routes organization ownership and validates an Actions usage item' -Tag BillingMigration {
        $date = [datetime]::UtcNow.ToString('yyyy-MM-dd')
        $usage = @{ usageItems = @(@{ date = $date; product = 'Actions'; sku = 'Actions Linux'; quantity = 1000000; unitType = 'minutes'; pricePerUnit = 0.008; grossAmount = 8000; discountAmount = 0; netAmount = 8000 }) } | ConvertTo-Json -Depth 5 -Compress
        $r = Complete-Fixture (Start-Fixture -Script Watch-WorkflowHealth.ps1 -Scenario @{ 'billing-owner' = @{ExitCode=0;Stdout='{"login":"fixture","type":"Organization"}';Stderr=''}; billing = @{ExitCode=0;Stdout=$usage;Stderr=''} })
        $w = ($r.Reports | Where-Object Name -Like 'workflow-health-*').Value
        $r.ExitCode | Should -Be 0; $w.problem_count | Should -Be 0; $w.billing.quota_status | Should -Be Unknown
        ($r.Calls | Where-Object operation -EQ billing).arguments[1] | Should -Match '^organizations/fixture/settings/billing/usage\?year=\d{4}&month=\d{1,2}$'
        ($r.Stdout + ($w | ConvertTo-Json -Depth 8)) | Should -Not -Match '8000|1000000'
    }
    # Protects: successful HTTP responses cannot substitute malformed billing metadata.
    # Detects: missing, null or non-array usageItems and obsolete minute counters.
    # Needs: complete monitor execution for each literal response.
    # Breadcrumb: enhanced billing usageItems array contract.
    It 'refuses missing null and object usage arrays without hiding available workflows' -Tag BillingMigration {
        foreach ($json in @('{}','{"usageItems":null}','{"usageItems":{}}','{"total_minutes_used":0,"included_minutes":100}')) {
            $r = Complete-Fixture (Start-Fixture -Script Watch-WorkflowHealth.ps1 -Scenario @{billing=@{ExitCode=0;Stdout=$json;Stderr=''}})
            $w = ($r.Reports | Where-Object Name -Like 'workflow-health-*').Value
            $r.ExitCode | Should -Be 2; $w.workflow_inspection_status | Should -Be Available
            ($w.inspections | Where-Object operation -EQ billing).reason | Should -Be 'Billing usage metadata is incomplete or invalid.'
        }
    }
    # Protects: advertised usage values retain their scalar numeric and date contracts.
    # Detects: array enums, string numbers, missing amounts and out-of-period dates.
    # Needs: inert current-month usage items returned to the actual monitor.
    # Breadcrumb: billing usage response schema and explicit UTC month filter.
    It 'refuses malformed typed usage items and records no private values' -Tag BillingMigration {
        $valid = @{ date=[datetime]::UtcNow.ToString('yyyy-MM-dd');product='Actions';sku='Actions Linux';quantity=1;unitType='minutes';pricePerUnit=0.008;grossAmount=0.008;discountAmount=0;netAmount=0.008 }
        foreach($change in @(@{product=@('Actions')},@{quantity='1'},@{netAmount=$null},@{date='2001-01-01'},@{date='bad'},@{unitType=@('minutes')})) {
            $item=$valid.Clone(); foreach($key in $change.Keys){$item[$key]=$change[$key]}
            $json=@{usageItems=@($item)} | ConvertTo-Json -Depth 6 -Compress
            $r=Complete-Fixture (Start-Fixture -Script Watch-WorkflowHealth.ps1 -Scenario @{billing=@{ExitCode=0;Stdout=$json;Stderr=''}})
            $w=($r.Reports | Where-Object Name -Like 'workflow-health-*').Value
            $r.ExitCode | Should -Be 2; ($w.inspections | Where-Object operation -EQ billing).status | Should -Be Unavailable
            $w.workflow_inspection_status | Should -Be Available
        }
    }
    # Protects: an optional returned timePeriod must agree with the requested month.
    # Detects: stale or incorrectly typed report period despite valid empty usageItems.
    # Needs: real monitor invocations with valid and invalid period objects.
    # Breadcrumb: current UTC year/month request validation.
    It 'accepts matching timePeriod and refuses stale or malformed period metadata' -Tag BillingMigration {
        $now=[datetime]::UtcNow
        foreach($period in @(@{year=$now.Year;month=$now.Month},@{year=2001;month=1},@{year="$($now.Year)";month=$now.Month},@{year=$now.Year})) {
            $json=@{usageItems=@();timePeriod=$period} | ConvertTo-Json -Depth 5 -Compress
            $r=Complete-Fixture (Start-Fixture -Script Watch-WorkflowHealth.ps1 -Scenario @{billing=@{ExitCode=0;Stdout=$json;Stderr=''}})
            $expected=if($period.year -is [int] -and $period.year -eq $now.Year -and $period.month -eq $now.Month){0}else{2}
            $r.ExitCode | Should -Be $expected
        }
    }
    # Protects: unknown account ownership cannot direct a billing request.
    # Detects: mismatched login, unsupported account type and array type coercion.
    # Needs: bounded owner discovery stubs and actual monitor control flow.
    # Breadcrumb: users owner type determines users versus organizations route.
    It 'refuses malformed billing ownership before calling usage' -Tag BillingMigration {
        foreach($json in @('{}','{"login":"other","type":"User"}','{"login":"fixture","type":"Bot"}','{"login":"fixture","type":["User"]}')) {
            $r=Complete-Fixture (Start-Fixture -Script Watch-WorkflowHealth.ps1 -Scenario @{'billing-owner'=@{ExitCode=0;Stdout=$json;Stderr=''}})
            $w=($r.Reports | Where-Object Name -Like 'workflow-health-*').Value
            $r.ExitCode | Should -Be 2; @($r.Calls | Where-Object operation -EQ billing).Count | Should -Be 0
            ($w.inspections | Where-Object operation -EQ billing-owner).status | Should -Be Unavailable
        }
    }
    # Protects: billing permission failures remain visible and sanitized.
    # Detects: valid-looking stdout from exit403 being accepted or leaking stderr.
    # Needs: complete monitor plus nonzero API collaborator result.
    # Breadcrumb: user Plan/read and organization Administration/read permission gates.
    It 'retains billing access failures while preserving workflow metadata availability' -Tag BillingMigration {
        $r=Complete-Fixture (Start-Fixture -Script Watch-WorkflowHealth.ps1 -Scenario @{billing=@{ExitCode=403;Stdout='{"usageItems":[]}';Stderr='fixture-secret-DO-NOT-LOG'}})
        $w=($r.Reports | Where-Object Name -Like 'workflow-health-*').Value
        $r.ExitCode | Should -Be 2; $w.workflow_inspection_status | Should -Be Available
        ($w.inspections | Where-Object operation -EQ billing).exit_code | Should -Be 403
        $w.billing.quota_status | Should -Be Unknown
        ($r.Stdout+$r.Stderr+($w | ConvertTo-Json -Depth 8)) | Should -Not -Match 'fixture-secret'
    }
    # Protects: workflow failure detection survives the supported billing migration.
    # Detects: a large usage quantity becoming guessed exhaustion or losing run failures.
    # Needs: actual workflow failure metadata with current billing schema.
    # Breadcrumb: usage quantities provide no included Actions allowance.
    It 'retains workflow findings with valid usage and unknown allowance' -Tag BillingMigration {
        $r=Complete-Fixture (Start-Fixture -Script Watch-WorkflowHealth.ps1 -Scenario $script:findings)
        $w=($r.Reports | Where-Object Name -Like 'workflow-health-*').Value
        $r.ExitCode | Should -Be 1; $w.problem_count | Should -Be 1
        $w.problems[0] | Should -Match 'failure: fixture-ci'; $w.billing.quota_status | Should -Be Unknown
    }
    It 'accepts inspected empty arrays without phantom Dependabot findings' {
        $r = Complete-Fixture (Start-Fixture); $c = Get-Combined $r
        $r.ExitCode | Should -Be 0; $c.inspection_status | Should -Be Available; $c.upstream_items | Should -Be 0; $c.workflow_problems | Should -Be 0
        $c.publication.status | Should -Be NotAttempted; $r.Calls.Count | Should -Be 7
    }
    It 'does not classify missing authentication with empty stdout as healthy' {
        $r = Complete-Fixture (Start-Fixture -Scenario @{ all_gh = @{ ExitCode = 4; Stdout = ''; Stderr = 'fixture-secret-DO-NOT-LOG C:/private/credential-file' } }); $c = Get-Combined $r
        $r.ExitCode | Should -Be 2; $c.inspection_status | Should -Be Unavailable; $c.upstream_items | Should -Be 0
        ($c | ConvertTo-Json -Depth 12) | Should -Not -Match 'fixture-secret|C:/private'; $r.Calls.Count | Should -Be 7
    }
    It 'rejects valid-looking error JSON from nonzero gh exit' {
        $r = Complete-Fixture (Start-Fixture -Scenario @{ all_gh = @{ ExitCode = 1; Stdout = '{"message":"Bad credentials"}'; Stderr = '' } })
        $r.ExitCode | Should -Be 2; (Get-Combined $r).inspection_status | Should -Be Unavailable
    }
    It 'rejects malformed JSON from a successful request' {
        $r = Complete-Fixture (Start-Fixture -Scenario @{ runs = @{ ExitCode = 0; Stdout = '{malformed'; Stderr = '' } }); $c = Get-Combined $r
        $r.ExitCode | Should -Be 2; $c.detail.workflow.inspections[0].reason | Should -Be 'GitHub inspection returned missing or invalid JSON.'
    }
    It 'recognizes only explicit disabled code scanning and preserves its unsupported status' {
        $r = Complete-Fixture (Start-Fixture -Scenario @{ 'code-scanning' = @{ ExitCode = 1; Stdout = ''; Stderr = 'HTTP 404: Code scanning is not enabled' } }); $c = Get-Combined $r
        $r.ExitCode | Should -Be 0; ($c.detail.upstream.inspections | Where-Object operation -EQ code-scanning).status | Should -Be Unsupported
        $r2 = Complete-Fixture (Start-Fixture -Scenario @{ 'code-scanning' = @{ ExitCode = 1; Stdout = ''; Stderr = 'HTTP 404: Not Found' } })
        $r2.ExitCode | Should -Be 2
    }
    It 'preserves actual child exit1 findings and publishes them' {
        $r = Complete-Fixture (Start-Fixture -Scenario $script:findings); $c = Get-Combined $r
        $r.ExitCode | Should -Be 1; $c.workflow_problems | Should -Be 1; $c.upstream_items | Should -Be 1; $c.publication.status | Should -Be Published
    }
    It 'marks incomplete run schema unavailable rather than aggregating zero healthy' {
        foreach ($response in @('[{}]', '[{"status":"completed","workflowName":"ci"}]', '[{"status":"completed","workflowName":"ci","conclusion":null}]', '[{"status":"completed","workflowName":"ci","conclusion":""}]', '[{"status":"completed","workflowName":"ci","conclusion":" "}]')) {
            $r = Complete-Fixture (Start-Fixture -Scenario @{ runs = @{ ExitCode = 0; Stdout = $response; Stderr = '' } }); $c = Get-Combined $r
            $r.ExitCode | Should -Be 2; $c.detail.workflow.inspections[0].reason | Should -Be 'Run metadata is incomplete.'
        }
        foreach ($status in @('queued', 'in_progress')) {
            $response = '[{"status":"' + $status + '","workflowName":"ci","conclusion":null}]'
            $r = Complete-Fixture (Start-Fixture -Scenario @{ runs = @{ ExitCode = 0; Stdout = $response; Stderr = '' } })
            $r.ExitCode | Should -Be 0; (Get-Combined $r).inspection_status | Should -Be Available
        }
    }
    It 'uses canonical client send with all and preserves candidate configuration' {
        $r = Complete-Fixture (Start-Fixture -Scenario $script:findings); $send = $r.Calls | Where-Object operation -EQ send
        $expectedClient = if ($IsWindows) { Join-Path $HOME 'bin/agent-bus.exe' } else { Join-Path $HOME '.local/bin/agent-bus' }
        if (-not (Test-Path -LiteralPath $expectedClient)) { $application = Get-Command agent-bus -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1; if ($application) { $expectedClient = $application.Source } }
        $send.file | Should -Be $expectedClient
        $hasher = [Security.Cryptography.SHA256]::Create()
        $keyPath = if ($IsWindows) { $r.Directory.ToLowerInvariant() } else { $r.Directory }
        try { $key = [BitConverter]::ToString($hasher.ComputeHash([Text.Encoding]::UTF8.GetBytes($keyPath))).Replace('-', '') } finally { $hasher.Dispose() }
        $sender = 'gitops-monitor-{0}-{1}' -f [Environment]::MachineName.ToLowerInvariant(), $key.Substring(0, 12)
        ($send.arguments -join '|') | Should -Be ('send|--from-agent|' + $sender + '|--to-agent|all|--topic|status|--body|GitOps monitor [fixture/repo]: inspection=Available, workflows=1, upstream=1. See the checkout GitOps report.|--tag|repo:repo|--encoding|compact')
        $send.candidate_config_preserved | Should -BeTrue
    }
    It 'retains sanitized delivery failures for missing auth401503 and timeout' {
        foreach ($code in @(4, 401, 503, -1)) {
            $scenario = $script:findings.Clone(); $scenario.send = @{ ExitCode = $code; Stdout = 'fixture-secret-DO-NOT-LOG'; Stderr = 'C:/private/credential-file' }
            $r = Complete-Fixture (Start-Fixture -Scenario $scenario); $c = Get-Combined $r
            $r.ExitCode | Should -Be 2; $c.publication.status | Should -Be Failed; $c.publication.exit_code | Should -Be $code
            ($c | ConvertTo-Json -Depth 12) | Should -Not -Match 'fixture-secret|C:/private'
        }
    }
    It 'rejects obsolete arbitrary BusUrl before any request or credential use' {
        $r = Complete-Fixture (Start-Fixture -BusUrl 'https://unconfigured.example/messages'); $c = Get-Combined $r
        $r.ExitCode | Should -Be 2; $r.Calls.Count | Should -Be 0; $c.configuration_error | Should -Match 'BusUrl is unsupported'; $c.publication.status | Should -Be NotAttempted
    }
    It 'checks WMI result and keeps launch rejection visible without blocking the hook' -Skip:(-not $IsWindows) {
        foreach ($scenario in @(@{}, @{launch_code = 2 }, @{launch_null = $true }, @{launch_throw = $true })) {
            $r = Complete-Fixture (Start-Fixture -Scenario $scenario -Script Start-GitOpsMonitors.ps1)
            $r.ExitCode | Should -Be 0; $r.Calls.Count | Should -Be 1
            $r.Calls[0].arguments.CurrentDirectory | Should -Be $script:source
            $r.Calls[0].create_flags | Should -Be 16778240
            $r.Calls[0].show_window | Should -Be 0
            $r.Calls[0].fixture_secret_preserved | Should -BeTrue
            $r.Calls[0].candidate_config_preserved | Should -BeTrue
            $r.Calls[0].arguments.CommandLine | Should -Not -Match 'fixture-secret|AGENT_BUS_SERVER|FIXTURE_SECRET'
            if ($scenario.Count) { $r.Stdout | Should -Match 'launch.*(rejected|failed)' } else { $r.Stdout | Should -Not -Match 'launch.*(rejected|failed)' }
            $r.Stdout | Should -Not -Match 'fixture-secret|C:/private'
        }
    }
    It 'anchors a launcher under a checkout with spaces regardless of caller cwd' -Skip:(-not $IsWindows) {
        $root = Join-Path $script:base ('checkout with spaces ' + [guid]::NewGuid().ToString('N')); $null = New-Item -ItemType Directory (Join-Path $root 'Tools/GitOps') -Force
        Copy-Item -LiteralPath (Join-Path $script:source 'Tools/GitOps/Start-GitOpsMonitors.ps1') -Destination (Join-Path $root 'Tools/GitOps/Start-GitOpsMonitors.ps1')
        $r = Complete-Fixture (Start-Fixture -Script Start-GitOpsMonitors.ps1 -SourceRoot $root)
        $r.Calls[0].arguments.CurrentDirectory | Should -Be $root
        $r.Calls[0].arguments.CommandLine | Should -Be ('"{0}" -NoProfile -WindowStyle Hidden -File "{1}" -RepoRoot "{2}"' -f (Join-Path $PSHOME 'pwsh.exe'), (Join-Path $root 'Tools/GitOps/Invoke-GitOpsMonitors.ps1'), $root)
    }
    It 'coalesces overlapping actual workers for the same canonical checkout' {
        $owner = Start-Fixture -Scenario @{hold_seconds = 3 }; Wait-Owner $owner
        $duplicate = Complete-Fixture (Start-Fixture -Root $owner.Root -Calls $owner.CallsPath)
        $duplicate.ExitCode | Should -Be 0; $duplicate.Stdout | Should -Match 'existing worker owns this checkout'
        $r = Complete-Fixture $owner; $r.ExitCode | Should -Be 0; $r.Calls.Count | Should -Be 7
        @($r.Calls.pid | Select-Object -Unique).Count | Should -Be 1
    }
    It 'keeps independently scoped worktrees concurrent without suppressing their requests' {
        $scenario = $script:findings.Clone(); $scenario.hold_seconds = 3
        $owner = Start-Fixture -Scenario $scenario; Wait-Owner $owner
        $other = Complete-Fixture (Start-Fixture -Scenario $script:findings); $r = Complete-Fixture $owner
        $r.ExitCode | Should -Be 1; $other.ExitCode | Should -Be 1; $r.Calls.Count | Should -Be 8; $other.Calls.Count | Should -Be 8
        ($r.Calls | Where-Object operation -EQ send).arguments[2] | Should -Not -Be ($other.Calls | Where-Object operation -EQ send).arguments[2]
    }
    It 'releases the lease after failures and atomically replaces readable reports' {
        $failed = Start-Fixture -Scenario @{runs = @{Throw = $true } }; $root = $failed.Root; $r = Complete-Fixture $failed
        $r.ExitCode | Should -Be 2; (Get-Combined $r).inspection_status | Should -Be Unavailable
        $next = Complete-Fixture (Start-Fixture -Root $root); $next.ExitCode | Should -Be 0; (Get-Combined $next).inspection_status | Should -Be Available
        @(Get-ChildItem $r.Directory, $next.Directory -Filter '.gitops-*.tmp').Count | Should -Be 0
        ($r.Stdout + $r.Stderr + (Get-Combined $r | ConvertTo-Json -Depth 12)) | Should -Not -Match 'fixture-secret|C:/private'
    }
}
