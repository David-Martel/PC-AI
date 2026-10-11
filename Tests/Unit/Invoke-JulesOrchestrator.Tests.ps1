#Requires -Version 7.0
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }

# Invoke-PlanReview is a keyword pre-screen over plan step titles and
# descriptions, not a review. It used to approve Jules plans on its own, and
# with AutoCreatePR that let a plan reach a PR with nobody reading it. Approval
# now needs an explicit -AutoApprove.
BeforeAll {
    $script:ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    $script:ScriptPath  = Join-Path $script:ProjectRoot 'Tools' 'Invoke-JulesOrchestrator.ps1'
    . $script:ScriptPath -Action '__test_load__'
}

Describe 'Invoke-JulesOrchestrator test hook' -Tag 'Unit', 'Portable' {
    It 'exposes Invoke-PlanReview through -Action __test_load__' {
        Get-Command Invoke-PlanReview -ErrorAction SilentlyContinue | Should -Not -BeNullOrEmpty
    }
}

Describe 'Invoke-PlanReview pre-screen' -Tag 'Unit', 'Portable' {
    It 'recommends approve for in-scope steps that include tests' {
        $steps = @(
            [PSCustomObject]@{ title = 'Refactor pcai_inference error paths'; description = 'use expect with context' },
            [PSCustomObject]@{ title = 'Add pcai_inference tests'; description = 'unit tests' }
        )
        $r = Invoke-PlanReview -ReviewSessionId 's1' -Steps $steps -ModuleName 'pcai_inference' -ScopePaths @('Native/pcai_core/pcai_inference/')
        $r.Recommendation | Should -Be 'approve'
    }
    It 'asks for changes when a step removes tests' {
        $steps = @([PSCustomObject]@{ title = 'Remove flaky pcai_inference tests'; description = '' })
        $r = Invoke-PlanReview -ReviewSessionId 's2' -Steps $steps -ModuleName 'pcai_inference' -ScopePaths @('Native/pcai_core/pcai_inference/')
        $r.Recommendation | Should -Not -Be 'approve'
    }
}

Describe 'Plan approval gating' -Tag 'Unit', 'Portable' {
    BeforeAll {
        $script:Ast = [System.Management.Automation.Language.Parser]::ParseFile($script:ScriptPath, [ref]$null, [ref]$null)
    }

    It 'has an -AutoApprove switch that is off by default' {
        $p = (Get-Command $script:ScriptPath).Parameters['AutoApprove']
        $p | Should -Not -BeNullOrEmpty
        $p.SwitchParameter | Should -BeTrue
        @(Invoke-ScriptAnalyzer -Path $script:ScriptPath -IncludeRule PSAvoidDefaultValueSwitchParameter).Count | Should -Be 0
    }

    It 'only calls the Approve action inside a branch conditioned on $AutoApprove' {
        $approveCalls = $script:Ast.FindAll({
                param($n)
                $n -is [System.Management.Automation.Language.HashtableAst] -and
                ($n.KeyValuePairs | Where-Object {
                        $_.Item1.Extent.Text -eq 'Action' -and $_.Item2.Extent.Text -match "^'Approve'$"
                    })
            }, $true)
        @($approveCalls).Count | Should -BeGreaterThan 0
        foreach ($call in $approveCalls) {
            $guarded = $false
            $node = $call.Parent
            while ($node) {
                if ($node -is [System.Management.Automation.Language.IfStatementAst] -and
                    ($node.Clauses | Where-Object { $_.Item1.Extent.Text -match '\$AutoApprove' })) {
                    $guarded = $true; break
                }
                $node = $node.Parent
            }
            $guarded | Should -BeTrue -Because "Approve at line $($call.Extent.StartLineNumber) must require -AutoApprove"
        }
    }
}

Describe 'jules-review.yml hardening' -Tag 'Unit', 'Portable' {
    BeforeAll { $script:Wf = Get-Content -Raw (Join-Path $script:ProjectRoot '.github' 'workflows' 'jules-review.yml') }

    It 'grants GITHUB_TOKEN contents: read only' {
        $block = [regex]::Match($script:Wf, '(?m)^permissions:\r?\n((?:[ \t]+.+\r?\n)+)').Groups[1].Value
        ($block -split '\r?\n' | Where-Object { $_.Trim() }) | ForEach-Object { $_.Trim() } | Should -Be @('contents: read')
    }
    It 'lets only the repository owner start the issue job' {
        $script:Wf | Should -Match "github\.event\.label\.name == 'jules' && github\.event\.sender\.login == github\.repository_owner"
    }
    It 'fences the issue title and body as untrusted data' {
        $script:Wf | Should -Match '(?s)<<<ISSUE_BODY\s+\$\{\{ github\.event\.issue\.body \}\}\s+ISSUE_BODY'
        $script:Wf | Should -Match '(?s)<<<ISSUE_TITLE\s+\$\{\{ github\.event\.issue\.title \}\}\s+ISSUE_TITLE'
    }
}
