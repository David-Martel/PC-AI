#Requires -Version 7.0

BeforeDiscovery {
    $workflowDir = Join-Path $PSScriptRoot '../../.github/workflows'
    $script:PullRequestWorkflows = foreach ($file in Get-ChildItem -LiteralPath $workflowDir -Filter '*.yml') {
        $text = Get-Content -LiteralPath $file.FullName -Raw
        $on = [regex]::Match($text, '(?ms)^on:\s*\r?\n(?<body>.*?)(?=^\S)').Groups['body'].Value
        if ($on -match '(?m)^  pull_request(_target)?:') {
            @{ Name = $file.Name; Text = $text }
        }
    }
}

Describe 'Pull-request workflows cancel superseded runs' -Tag 'Unit', 'Portable' {
    # Protects: A rebased or re-pushed pull request (Dependabot rebases its branches on every main change) cancels
    # the run it superseded instead of letting both occupy the self-hosted pc-ai runners.
    # Detects: A pull_request workflow with no top-level concurrency group, or one that never cancels. 37 Dependabot
    # runs between 2026-11-10 and 2026-11-12 came from only 7 branches, and seven PR workflows had no group.
    # Needs: PowerShell 7 and Pester 5; reads the workflow files only.
    # Breadcrumb: .github/workflows/*.yml; runner-audit report C:/codedev/.workspace-index-state/runner-audit.md (T4).
    It '<Name> has a concurrency group that cancels superseded pull-request runs' -ForEach $script:PullRequestWorkflows {
        $block = [regex]::Match($Text, '(?ms)^concurrency:\s*\r?\n(?<body>(?:^  .*\r?\n?)+)').Groups['body'].Value
        $block | Should -Not -BeNullOrEmpty -Because "$Name runs on pull_request and needs a top-level concurrency group"
        $block | Should -Match '(?m)^  group: \S+'
        $block | Should -Match '(?m)^  cancel-in-progress: (true|\$\{\{ github\.event_name == ''pull_request'' \}\})\s*$'
    }
}
