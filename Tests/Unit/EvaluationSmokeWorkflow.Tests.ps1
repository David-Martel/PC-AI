#Requires -Version 7.0

BeforeAll {
    $script:WorkflowText = Get-Content -LiteralPath (Join-Path $PSScriptRoot '../../.github/workflows/evaluation-smoke.yml') -Raw
    function Get-SmokeStep {
        param([string]$Name)
        $pattern = '(?ms)^      - name: ' + [regex]::Escape($Name) + '\r?\n(?<step>.*?)(?=^      - |\z)'
        $step = [regex]::Match($script:WorkflowText, $pattern).Groups['step'].Value
        $run = [regex]::Match($step, '(?ms)^        run: \|\r?\n(?<body>.*)$').Groups['body'].Value
        if (-not $run) { throw "Missing workflow run block: $Name" }
        [scriptblock]::Create(($run -replace '(?m)^          ', ''))
    }
}

Describe 'Evaluation smoke workflow lifecycle' -Tag 'Unit', 'Portable' {
    # Protects: The always-run cleanup stops the recorded mock without assigning PowerShell's read-only PID.
    # Detects: The previous $pid assignment throws before Stop-Process can run.
    # Needs: PowerShell 7, Pester 5 and TestDrive; Stop-Process is mocked, no process is killed.
    # Breadcrumb: .github/workflows/evaluation-smoke.yml Stop Mock Server.
    It 'stops the recorded mock process through the actual cleanup block' {
        $priorTemp = $env:RUNNER_TEMP
        try {
            $env:RUNNER_TEMP = $TestDrive
            Set-Content -LiteralPath (Join-Path $TestDrive 'pcai_eval_server.pid') -Value '2147483000'
            Mock Stop-Process {}
            & (Get-SmokeStep 'Stop Mock Server')
            Should -Invoke Stop-Process -Exactly -Times 1 -ParameterFilter { $Id -eq 2147483000 -and $Force }
        }
        finally { $env:RUNNER_TEMP = $priorTemp }
    }

    # Protects: Dispatch values arrive as literal arguments instead of executable PowerShell source.
    # Detects: Direct expression interpolation allows quotes and statements to enter the run script.
    # Needs: PowerShell 7 and Pester 5; a local pwsh function captures argv without launching evaluation.
    # Breadcrumb: .github/workflows/evaluation-smoke.yml Run Evaluation Smoke env/run boundary.
    It 'passes dispatch input through environment values as literal evaluation arguments' {
        $names = @('PCAI_SMOKE_BASE_URL', 'PCAI_SMOKE_DATASET', 'PCAI_SMOKE_MAX_CASES')
        $prior = @{}
        foreach ($name in $names) { $prior[$name] = [Environment]::GetEnvironmentVariable($name) }
        try {
            $env:PCAI_SMOKE_BASE_URL = 'http://localhost:9911'
            $env:PCAI_SMOKE_DATASET = 'diagnostic"; throw "input executed'
            $env:PCAI_SMOKE_MAX_CASES = '3'
            $script:EvaluationArguments = @()
            function pwsh { $script:EvaluationArguments = @($args) }
            $run = Get-SmokeStep 'Run Evaluation Smoke'
            $run.ToString() | Should -Not -Match '\$\{\{'
            & $run
            $script:EvaluationArguments | Should -Contain $env:PCAI_SMOKE_DATASET
            $script:EvaluationArguments | Should -Contain $env:PCAI_SMOKE_BASE_URL
            $script:EvaluationArguments | Should -Contain 3
            $script:WorkflowText | Should -Match 'PCAI_SMOKE_DATASET: \$\{\{ inputs.dataset \}\}'
        }
        finally {
            foreach ($name in $names) { [Environment]::SetEnvironmentVariable($name, $prior[$name]) }
        }
    }

    # Protects: The Windows mock helper starts without displaying a console window.
    # Detects: A missing WindowStyle Hidden parameter on the workflow's actual Start-Process command.
    # Needs: PowerShell AST parsing and Pester 5; no Python installation or server is required.
    # Breadcrumb: .github/workflows/evaluation-smoke.yml Start Mock LLM Server.
    It 'starts the Windows mock helper with a hidden window' {
        $run = Get-SmokeStep 'Start Mock LLM Server'
        $commands = @($run.Ast.FindAll({ param($node)
                    $node -is [Management.Automation.Language.CommandAst] -and $node.GetCommandName() -eq 'Start-Process'
                }, $true))
        $commands.Count | Should -Be 1
        $commands[0].Extent.Text | Should -Match '-WindowStyle\s+Hidden\b'
    }
}
