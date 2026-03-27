#Requires -Version 7.0
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }

BeforeAll {
    $script:ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    $script:ScriptPath  = Join-Path $script:ProjectRoot 'Tools' 'Invoke-JulesSession.ps1'
    . $script:ScriptPath -Action '__test_load__'
}

Describe 'Get-JulesApiUrl' {
    It 'sessions base'        { Get-JulesApiUrl -Endpoint sessions | Should -Be 'https://jules.googleapis.com/v1alpha/sessions' }
    It 'session by id'        { Get-JulesApiUrl -Endpoint sessions -Id s1 | Should -Be 'https://jules.googleapis.com/v1alpha/sessions/s1' }
    It 'activities sub'       { Get-JulesApiUrl -Endpoint sessions -Id s1 -Sub activities | Should -Be 'https://jules.googleapis.com/v1alpha/sessions/s1/activities' }
    It 'approvePlan action'   { Get-JulesApiUrl -Endpoint sessions -Id s1 -UrlAction approvePlan | Should -Be 'https://jules.googleapis.com/v1alpha/sessions/s1:approvePlan' }
    It 'sendMessage action'   { Get-JulesApiUrl -Endpoint sessions -Id s1 -UrlAction sendMessage | Should -Be 'https://jules.googleapis.com/v1alpha/sessions/s1:sendMessage' }
    It 'sources base'         { Get-JulesApiUrl -Endpoint sources | Should -Be 'https://jules.googleapis.com/v1alpha/sources' }
    It 'source by id'         { Get-JulesApiUrl -Endpoint sources -Id src1 | Should -Be 'https://jules.googleapis.com/v1alpha/sources/src1' }
}

Describe 'New-JulesSessionBody' {
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

Describe 'Get-JulesApiKey' {
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

Describe 'Get-RequiredApiKey' {
    It 'throws when key is missing' {
        $saved = $env:JULES_API_KEY
        try {
            Remove-Item env:JULES_API_KEY -ErrorAction SilentlyContinue
            { Get-RequiredApiKey -ForAction Test } | Should -Throw '*JULES_API_KEY*'
        } finally { if ($saved) { $env:JULES_API_KEY = $saved } }
    }
}

Describe 'Format-JulesSessionTable' {
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

Describe 'Parameter validation' {
    It 'rejects missing Action'    { { & $script:ScriptPath } | Should -Throw }
    It 'rejects New without Prompt' { { & $script:ScriptPath -Action New } | Should -Throw '*Prompt*' }
    It 'rejects Status without SessionId' { { & $script:ScriptPath -Action Status } | Should -Throw '*SessionId*' }
}
