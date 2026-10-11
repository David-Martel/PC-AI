#Requires -Version 7.0
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }

BeforeAll {
    $script:ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    $script:ScriptPath  = Join-Path $script:ProjectRoot 'Tools' 'Invoke-JulesSession.ps1'
    . $script:ScriptPath -Action '__test_load__'
}

Describe 'Get-JulesApiUrl' -Tag 'Unit', 'Portable' {
    It 'sessions base'        { Get-JulesApiUrl -Endpoint sessions | Should -Be 'https://jules.googleapis.com/v1alpha/sessions' }
    It 'session by id'        { Get-JulesApiUrl -Endpoint sessions -Id s1 | Should -Be 'https://jules.googleapis.com/v1alpha/sessions/s1' }
    It 'activities sub'       { Get-JulesApiUrl -Endpoint sessions -Id s1 -Sub activities | Should -Be 'https://jules.googleapis.com/v1alpha/sessions/s1/activities' }
    It 'approvePlan action'   { Get-JulesApiUrl -Endpoint sessions -Id s1 -UrlAction approvePlan | Should -Be 'https://jules.googleapis.com/v1alpha/sessions/s1:approvePlan' }
    It 'sendMessage action'   { Get-JulesApiUrl -Endpoint sessions -Id s1 -UrlAction sendMessage | Should -Be 'https://jules.googleapis.com/v1alpha/sessions/s1:sendMessage' }
    It 'sources base'         { Get-JulesApiUrl -Endpoint sources | Should -Be 'https://jules.googleapis.com/v1alpha/sources' }
    It 'source by id'         { Get-JulesApiUrl -Endpoint sources -Id src1 | Should -Be 'https://jules.googleapis.com/v1alpha/sources/src1' }
}

Describe 'New-JulesSessionBody' -Tag 'Unit', 'Portable' {
    It 'builds minimal body' {
        $b = New-JulesSessionBody -PromptText 'Fix bug' -SourceName 'sources/github/o/r' -BranchName main
        $b.prompt | Should -Be 'Fix bug'
        $b.source | Should -Be 'sources/github/o/r'
        $b.branch | Should -Be 'main'
    }
    It 'includes plan approval' {
        $b = New-JulesSessionBody -PromptText 'x' -SourceName 's' -BranchName main -PlanApproval
        $b.requirePlanApproval | Should -BeTrue
    }
    It 'maps AutoCreatePR' {
        $b = New-JulesSessionBody -PromptText 'x' -SourceName 's' -BranchName main -Automation AutoCreatePR
        $b.automationMode | Should -Be 'AUTO_CREATE_PR'
    }
    It 'includes title' {
        $b = New-JulesSessionBody -PromptText 'x' -SourceName 's' -BranchName main -SessionTitle 'My Title'
        $b.title | Should -Be 'My Title'
    }
}

Describe 'Get-JulesApiKey' -Tag 'Unit', 'Portable' {
    It 'returns env var when set' {
        $saved = $env:JULES_API_KEY
        try {
            $env:JULES_API_KEY = 'test-key-abc'
            Get-JulesApiKey | Should -Be 'test-key-abc'
        } finally { $env:JULES_API_KEY = $saved }
    }
    It 'returns null when nothing available' {
        $saved = $env:JULES_API_KEY
        try {
            Remove-Item env:JULES_API_KEY -ErrorAction SilentlyContinue
            Get-JulesApiKey | Should -BeNullOrEmpty
        } finally { if ($saved) { $env:JULES_API_KEY = $saved } }
    }
}

Describe 'Get-RequiredApiKey' -Tag 'Unit', 'Portable' {
    It 'throws when key is missing' {
        $saved = $env:JULES_API_KEY
        try {
            Remove-Item env:JULES_API_KEY -ErrorAction SilentlyContinue
            { Get-RequiredApiKey -ForAction Test } | Should -Throw '*JULES_API_KEY*'
        } finally { if ($saved) { $env:JULES_API_KEY = $saved } }
    }
}

Describe 'Format-JulesSessionTable' -Tag 'Unit', 'Portable' {
    It 'extracts session fields' {
        $session = [PSCustomObject]@{
            name       = 'sessions/abc'
            title      = 'Test'
            state      = 'COMPLETED'
            createTime = '2026-03-27T00:00:00Z'
            outputs    = @([PSCustomObject]@{ pullRequest = [PSCustomObject]@{ url = 'https://github.com/pr/1' } })
        }
        $r = Format-JulesSessionTable -Sessions @($session)
        $r[0].SessionId | Should -Be 'abc'
        $r[0].State | Should -Be 'COMPLETED'
        $r[0].PRUrl | Should -Be 'https://github.com/pr/1'
    }
}

Describe 'Parameter validation' -Tag 'Unit', 'Portable' {
    It 'rejects missing Action'    { { & $script:ScriptPath } | Should -Throw }
    It 'rejects New without Prompt' { { & $script:ScriptPath -Action New } | Should -Throw '*Prompt*' }
    It 'rejects Status without SessionId' { { & $script:ScriptPath -Action Status } | Should -Throw '*SessionId*' }
}

# The Jules API rejects every server-side state filter with HTTP 400
# (`filter=state=COMPLETED`, `state = "COMPLETED"`, `state:COMPLETED` all
# verified 2026-09-27), and reports states as UPPER_SNAKE (`IN_PROGRESS`)
# while -State takes PascalCase (`InProgress`). -State is applied client-side.
Describe 'ConvertTo-JulesStateEnum' -Tag 'Unit', 'Portable' {
    It 'maps <State> to <Expected>' -TestCases @(
        @{ State = 'Completed';            Expected = 'COMPLETED' }
        @{ State = 'InProgress';           Expected = 'IN_PROGRESS' }
        @{ State = 'Failed';               Expected = 'FAILED' }
        @{ State = 'AwaitingPlanApproval'; Expected = 'AWAITING_PLAN_APPROVAL' }
        @{ State = 'AwaitingUserFeedback'; Expected = 'AWAITING_USER_FEEDBACK' }
        @{ State = 'Queued';               Expected = 'QUEUED' }
        @{ State = 'Planning';             Expected = 'PLANNING' }
        @{ State = 'Paused';               Expected = 'PAUSED' }
    ) {
        ConvertTo-JulesStateEnum -State $State | Should -Be $Expected
    }

    It 'covers every value the -State parameter accepts' {
        $cmd = Get-Command $script:ScriptPath
        $valid = ($cmd.Parameters['State'].Attributes |
            Where-Object { $_ -is [System.Management.Automation.ValidateSetAttribute] }).ValidValues
        foreach ($v in $valid) { ConvertTo-JulesStateEnum -State $v | Should -Match '^[A-Z]+(_[A-Z]+)*$' }
    }
}

Describe 'List action state filtering' -Tag 'Unit', 'Portable' {
    It 'does not send a state filter to the API' {
        $source = Get-Content -Raw $script:ScriptPath
        $source | Should -Not -Match 'filter=state='
    }
}

Describe 'List action JSON output' -Tag 'Unit', 'Portable' {
    It 'serializes with -InputObject so an empty result is [] rather than nothing' {
        $source = Get-Content -Raw $script:ScriptPath
        $source | Should -Match 'ConvertTo-Json -InputObject \$output'
    }
}

# Key lookup must never block on a locked Bitwarden vault, and must not read
# ~/.machine/*.json: that folder is writable by sandbox accounts, and no file
# in it has ever held the key (audited 2026-09-27).
Describe 'Get-JulesApiKey key sources' -Tag 'Unit', 'Portable' {
    BeforeEach {
        $script:savedKey = $env:JULES_API_KEY
        Remove-Item env:JULES_API_KEY -ErrorAction SilentlyContinue
        $global:JulesBwCalls = [System.Collections.Generic.List[object]]::new()
        function global:bw { $global:JulesBwCalls.Add(@($args)); $global:LASTEXITCODE = 1 }
    }
    AfterEach {
        Remove-Item function:global:bw -ErrorAction SilentlyContinue
        if ($script:savedKey) { $env:JULES_API_KEY = $script:savedKey }
    }

    It 'calls bw with --nointeraction so a locked vault fails fast' {
        Get-JulesApiKey | Out-Null
        $global:JulesBwCalls.Count | Should -Be 1
        $global:JulesBwCalls[0] | Should -Contain '--nointeraction'
    }
    It 'returns the value bw prints' {
        function global:bw { $global:LASTEXITCODE = 0; 'key-from-bw' }
        Get-JulesApiKey | Should -Be 'key-from-bw'
    }
    It 'does not read ~/.machine JSON files' {
        (Get-Command Get-JulesApiKey).Definition | Should -Not -Match '\.machine'
    }
    It 'names every supported key source when the key is missing' {
        { Get-RequiredApiKey -ForAction 'List' } | Should -Throw '*JULES_API_KEY*.env*Bitwarden*'
    }
}

Describe 'Invoke-JulesBatchReview parameters' -Tag 'Unit', 'Portable' {
    BeforeAll { $script:BatchPath = Join-Path $script:ProjectRoot 'Tools' 'Invoke-JulesBatchReview.ps1' }
    It 'has no switch parameter that defaults to true' {
        @(Invoke-ScriptAnalyzer -Path $script:BatchPath -IncludeRule PSAvoidDefaultValueSwitchParameter).Count | Should -Be 0
    }
    It 'requires plan approval unless -SkipPlanApproval is given' {
        $params = (Get-Command $script:BatchPath).Parameters
        $params.ContainsKey('SkipPlanApproval') | Should -BeTrue
        $params.ContainsKey('RequirePlanApproval') | Should -BeFalse
    }
}
