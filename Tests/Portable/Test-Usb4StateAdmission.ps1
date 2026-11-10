#Requires -Version 7.0
<#
.SYNOPSIS
Refuse damaged persisted USB4 custody before observing or changing networking.
.DESCRIPTION
Exercises the controller's actual state-read assignment with real fixture files.
Windows also runs its real DryRun entrypoint in a bounded child. No live policy,
adapter, task or installed journal is used. See docs/usb4/peer-networking.md.
#>
[CmdletBinding()]
param([string]$SourcePath = (Join-Path $PSScriptRoot '../../Tools/SystemScripts/Networking/Invoke-Usb4PeerNetwork.ps1'))
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$SourcePath = (Resolve-Path -LiteralPath $SourcePath).Path
$tokens = $null; $parseErrors = $null
$ast = [Management.Automation.Language.Parser]::ParseFile($SourcePath, [ref]$tokens, [ref]$parseErrors)
if ($parseErrors.Count) { throw 'Controller source failed to parse.' }
. $SourcePath
$readAssignments = @($ast.FindAll({ param($node)
    $node -is [Management.Automation.Language.AssignmentStatementAst] -and
    $node.Left.Extent.Text -ceq '$previous' -and $node.Right.Extent.Text.Contains('$StatePath')
}, $true))
if ($readAssignments.Count -ne 1) { throw 'Expected one actual entrypoint state-read assignment.' }
$readState = [scriptblock]::Create($readAssignments[0].Extent.Text + '; ,$previous')
$workspace = Join-Path ([IO.Path]::GetTempPath()) ('pcai-state-admission-' + [guid]::NewGuid().ToString('N'))
[void][IO.Directory]::CreateDirectory($workspace)
$StatePath = Join-Path $workspace 'state.json'
$script:checks = 0
function Assert-StateCondition([bool]$Condition, [string]$Message) {
    if (-not $Condition) { throw $Message }
    $script:checks++
}
try {
    # Protects missing-file bootstrap; detects treating a nonexistent journal as
    # corrupt. Needs pwsh and owned temp files. Breadcrumb: controller state read.
    $state = & $readState
    Assert-StateCondition ($state -is [Collections.IDictionary] -and $state.Count -eq 0) 'Missing state did not retain bootstrap behavior.'
    Write-Output 'PASS: missing state permits empty bootstrap'

    # Protects existing custody from silent reset. Detects zero bytes, NULs,
    # whitespace, null, scalars, arrays and malformed JSON being
    # accepted. Needs real files only. Breadcrumb: actual entrypoint state read.
    $badInputs = [ordered]@{
        'zero-byte' = ''; 'zero-filled' = ([string][char]0) * 64; 'whitespace' = " `r`n`t"
        'null' = 'null'; 'number' = '1'; 'string' = '"state"'
        'array' = '[{"schemaVersion":1}]'; 'empty-array' = '[]'; 'broken-json' = '{'
        'null-pending' = '{"schemaVersion":1,"pendingTransaction":null}'
        'array-pending' = '{"schemaVersion":1,"pendingTransaction":[]}'
    }
    foreach ($case in $badInputs.GetEnumerator()) {
        [IO.File]::WriteAllText($StatePath, $case.Value, [Text.UTF8Encoding]::new($false))
        $before = (Get-FileHash -LiteralPath $StatePath -Algorithm SHA256).Hash
        $caught = $null
        try { $null = & $readState } catch { $caught = $_ }
        Assert-StateCondition ($null -ne $caught) "Persisted $($case.Key) was accepted instead of refusing unknown custody."
        Assert-StateCondition ($caught.Exception.Message -match 'USB4 state admission failed') "Persisted $($case.Key) did not produce an explicit state-admission error."
        Assert-StateCondition ((Get-FileHash -LiteralPath $StatePath -Algorithm SHA256).Hash -ceq $before) "Refusal changed $($case.Key) evidence."
    }
    Write-Output 'PASS: malformed persisted custody is refused without rewriting bytes'

    # Protects successful rollback from missing-file bootstrap: the controller
    # publishes {} after restoring that preimage. Detects stranding the next run.
    # Needs real JSON only. Breadcrumb: Invoke-Usb4NetworkReconcile rollback.
    [IO.File]::WriteAllText($StatePath, '{}', [Text.UTF8Encoding]::new($false))
    $state = & $readState
    Assert-StateCondition ($state -is [Collections.IDictionary] -and $state.Count -eq 0) 'Successful rollback empty-object state was refused.'
    Assert-StateCondition ([IO.File]::ReadAllText($StatePath) -ceq '{}') 'Reading rollback state changed its bytes.'
    Write-Output 'PASS: successful rollback empty-object state remains readable'

    # Protects readable legacy and pending records. Detects dropping unknown fields
    # or pending rollback evidence. Needs real JSON; same controller breadcrumb.
    $valid = '{"schemaVersion":1,"hostName":"fixture","unknownFutureField":"preserve","pendingTransaction":{"phase":"Rollback","attemptId":"original-attempt"}}'
    [IO.File]::WriteAllText($StatePath, $valid, [Text.UTF8Encoding]::new($false))
    $before = (Get-FileHash -LiteralPath $StatePath -Algorithm SHA256).Hash
    $state = & $readState
    Assert-StateCondition ($state.pendingTransaction.attemptId -ceq 'original-attempt') 'Pending original attempt was lost.'
    Assert-StateCondition ($state.unknownFutureField -ceq 'preserve') 'Unknown existing state field was lost.'
    Assert-StateCondition ((Get-FileHash -LiteralPath $StatePath -Algorithm SHA256).Hash -ceq $before) 'Reading valid custody rewrote its evidence.'
    Write-Output 'PASS: valid pending custody and unknown fields are preserved'

    # Protects the real runtime ordering and failure phase. Detects validation
    # being unwired or delayed until InitialObservation. Needs Windows + pwsh;
    # Linux is explicitly NOT_RUN. Breadcrumb: controller main try/catch.
    if ($IsWindows) {
        $policyPath = Join-Path $workspace 'policy.json'
        $reportPath = Join-Path $workspace 'report.json'
        [IO.File]::WriteAllText($policyPath, '{}', [Text.UTF8Encoding]::new($false))
        [IO.File]::WriteAllBytes($StatePath, [byte[]]::new(64))
        $before = (Get-FileHash -LiteralPath $StatePath -Algorithm SHA256).Hash
        $start = [Diagnostics.ProcessStartInfo]::new((Join-Path $PSHOME 'pwsh.exe'))
        $start.UseShellExecute = $false; $start.CreateNoWindow = $true
        $start.RedirectStandardOutput = $true; $start.RedirectStandardError = $true
        foreach ($argument in @('-NoLogo', '-NoProfile', '-NonInteractive', '-File', $SourcePath,
            '-PolicyPath', $policyPath, '-StatePath', $StatePath, '-ReportPath', $reportPath, '-DryRun')) {
            $start.ArgumentList.Add($argument)
        }
        $child = [Diagnostics.Process]::Start($start)
        try {
            $stdout = $child.StandardOutput.ReadToEndAsync(); $stderr = $child.StandardError.ReadToEndAsync()
            if (-not $child.WaitForExit(15000)) { $child.Kill($true); $child.WaitForExit(); throw 'Owned DryRun child exceeded its deadline.' }
            $result = $stdout.GetAwaiter().GetResult() | ConvertFrom-Json -AsHashtable
            Assert-StateCondition ($child.ExitCode -eq 1) 'Corrupt state entrypoint did not fail.'
            Assert-StateCondition ($result.phase -ceq 'ReadPolicyAndState') 'Corrupt state reached OS observation.'
            Assert-StateCondition (-not $result.networkReady -and ($result.errors -join ' ') -match 'USB4 state admission failed') 'Corrupt state lost explicit refusal/readiness.'
            Assert-StateCondition (-not (Test-Path -LiteralPath $reportPath)) 'DryRun wrote a report.'
            Assert-StateCondition ((Get-FileHash -LiteralPath $StatePath -Algorithm SHA256).Hash -ceq $before) 'DryRun changed corrupt journal bytes.'
            Assert-StateCondition ([string]::IsNullOrWhiteSpace($stderr.GetAwaiter().GetResult())) 'DryRun emitted unexpected stderr.'
        }
        finally { $child.Dispose() }
        Write-Output 'PASS: real Windows entrypoint refuses corruption before observation'
    }
    else { Write-Output 'NOT_RUN: real Windows entrypoint requires Windows' }
    Write-Output "PASS: $script:checks state-admission assertions"
}
finally { [IO.Directory]::Delete($workspace, $true) }
