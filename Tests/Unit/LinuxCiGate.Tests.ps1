#Requires -Version 7.0
# Why this file exists: `ci-gate` in .github/workflows/linux-ci.yml is the single required check for the Linux CI
# workflow, and the merge autopilot waits on it. It runs on a hosted runner, which is safe only while it executes no
# repository code. These tests keep that property and keep the gate covering every other job.
# See CI-CD-GUIDE.md ("Linux self-hosted gate") and the comment above the job.

BeforeAll {
    $script:WorkflowText = Get-Content -LiteralPath (Join-Path $PSScriptRoot '../../.github/workflows/linux-ci.yml') -Raw
    $jobsText = [regex]::Match($script:WorkflowText, '(?ms)^jobs:\r?\n(?<body>.*)\z').Groups['body'].Value
    $script:JobNames = @([regex]::Matches($jobsText, '(?m)^  (?<name>[A-Za-z0-9_-]+):\s*$') | ForEach-Object { $_.Groups['name'].Value })
    $script:GateText = [regex]::Match($jobsText, '(?ms)^  ci-gate:\s*\r?\n(?<body>.*?)(?=^  [A-Za-z0-9_-]+:\s*$|\z)').Groups['body'].Value
}

Describe 'Linux CI aggregate gate' -Tag 'Unit', 'Portable' {
    # Protects: the gate's hosted runner never executes repository code or sees a secret.
    # Detects: a checkout, an action (`uses:`), a local script or a secret reference added to ci-gate. Any of these
    # would run pull-request code or credentials outside the trust boundary the self-hosted jobs keep.
    # Needs: the workflow text only. Breadcrumb: .github/workflows/linux-ci.yml job ci-gate.
    It 'runs on a hosted runner with no checkout, action, local script or secret' {
        $script:GateText | Should -Not -BeNullOrEmpty
        $script:GateText | Should -Match '(?m)^    runs-on: ubuntu-latest\s*$'
        $script:GateText | Should -Not -Match '(?m)^\s*-?\s*uses:'
        $script:GateText | Should -Not -Match 'secrets\.'
        $script:GateText | Should -Not -Match '\./|\.ps1|\.sh\b'
    }

    # Protects: one required check covers the whole workflow.
    # Detects: a job added to the workflow but not to ci-gate's needs, which would let it fail, be cancelled or be
    # skipped while the required check still passes.
    # Needs: the workflow text only. Breadcrumb: .github/workflows/linux-ci.yml job ci-gate `needs`.
    It 'needs every other job in the workflow' {
        $needsLine = [regex]::Match($script:GateText, '(?m)^    needs: \[(?<list>[^\]]*)\]').Groups['list'].Value
        $needs = @($needsLine -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ })
        $others = @($script:JobNames | Where-Object { $_ -ne 'ci-gate' })
        $others.Count | Should -BeGreaterThan 3 -Because 'the job list must be parsed, not empty'
        ($needs | Sort-Object) | Should -Be ($others | Sort-Object)
    }

    # Protects: failure, cancellation and skipping of a needed job fail the gate instead of passing it.
    # Detects: removal of always() (a skipped gate reports success) or of the non-success check.
    # Needs: the workflow text only. Breadcrumb: .github/workflows/linux-ci.yml job ci-gate.
    It 'runs always and rejects every non-success result' {
        $script:GateText | Should -Match '(?m)^    if: always\(\)'
        $script:GateText | Should -Match 'job\["result"\] != "success"'
    }
}
