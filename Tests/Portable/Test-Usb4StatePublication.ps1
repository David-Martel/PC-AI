#Requires -Version 7.0
<# .SYNOPSIS
Detect custody loss during real-file controller state publication.
.DESCRIPTION
Uses the actual main StateWriter callback and owned temporary files. No installed
state, network provider, task or live policy is used. The file-lock control is
Windows-only; durable API calls do not qualify hardware power-loss behavior.
#>
[CmdletBinding()]
param([string]$SourcePath = (Join-Path $PSScriptRoot '../../Tools/SystemScripts/Networking/Invoke-Usb4PeerNetwork.ps1'))
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. $SourcePath
$tokens = $null; $parseErrors = $null
$ast = [Management.Automation.Language.Parser]::ParseFile((Resolve-Path $SourcePath).Path, [ref]$tokens, [ref]$parseErrors)
if ($parseErrors.Count) { throw 'Controller parse failure.' }
$publish = @($ast.FindAll({ param($n)
    $n -is [Management.Automation.Language.CommandAst] -and $n.GetCommandName() -eq 'Publish-Usb4NetworkResult'
}, $true))
if ($publish.Count -ne 1) { throw 'Expected one actual main publication call.' }
$stateWriter = $null
for ($i = 0; $i -lt $publish[0].CommandElements.Count - 1; $i++) {
    $element = $publish[0].CommandElements[$i]
    if ($element -is [Management.Automation.Language.CommandParameterAst] -and $element.ParameterName -eq 'StateWriter') {
        $stateWriter = $publish[0].CommandElements[$i + 1].ScriptBlock.GetScriptBlock()
    }
}
if ($null -eq $stateWriter) { throw 'Actual main StateWriter is unavailable.' }
$workspace = Join-Path ([IO.Path]::GetTempPath()) ('pcai-state-publication-' + [guid]::NewGuid().ToString('N'))
[void][IO.Directory]::CreateDirectory($workspace)
$StatePath = Join-Path $workspace 'state.json'
$script:checks = 0
function Assert-Publication([bool]$Condition, [string]$Message) {
    if (-not $Condition) { throw $Message }
    $script:checks++
}
function Invoke-ActualStatePublication([object]$Value, [switch]$DryRun) {
    $result = @{ state = $Value; errors = @(); networkReady = $true; exitCode = 0 }
    Publish-Usb4NetworkResult -Result $result -StateWriter $stateWriter -ReportWriter { param($Report) } -Apply:(-not $DryRun)
}
function Get-FixtureFiles {
    @((Get-ChildItem -LiteralPath $workspace -File -Recurse | ForEach-Object {
        $_.FullName + ':' + (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash
    }) | Sort-Object)
}
try {
    # Protects interrupted custody from replacement by an in-memory result.
    # Detects the old real writer overwriting NUL/invalid state and claiming success.
    # Needs real temp files; breadcrumb: actual main StateWriter callback.
    foreach ($bad in @((([string][char]0) * 64), 'null', '[]', '{')) {
        [IO.File]::WriteAllText($StatePath, $bad, [Text.UTF8Encoding]::new($false))
        $before = (Get-FileHash $StatePath -Algorithm SHA256).Hash
        $result = Invoke-ActualStatePublication @{ schemaVersion = 1; generation = 2 }
        Assert-Publication ($result.exitCode -eq 1 -and -not $result.networkReady) 'Actual StateWriter overwrote unreadable custody and reported success.'
        Assert-Publication ((Get-FileHash $StatePath -Algorithm SHA256).Hash -ceq $before) 'Refusal changed corrupt predecessor bytes.'
    }
    Write-Output 'PASS: corrupt predecessor remains untouched and publication fails'

    # Protects exact recovery bytes, including formatting/BOM differences.
    # Detects missing, reserialized or corrupted immutable content-bound history.
    # Needs real files; breadcrumb: Write-Usb4NetworkState and byte publication.
    $original = "{`r`n  `"schemaVersion`": 1, `"generation`": 1`r`n}`r`n"
    [IO.File]::WriteAllText($StatePath, $original, [Text.UTF8Encoding]::new($false))
    $before = (Get-FileHash $StatePath -Algorithm SHA256).Hash
    $value = [ordered]@{ schemaVersion = 1; generation = 2 }
    $result = Invoke-ActualStatePublication $value
    $history = Join-Path ($StatePath + '.history') ($before.ToLowerInvariant() + '.json')
    Assert-Publication ($result.exitCode -eq 0 -and (Test-Path -LiteralPath $history)) 'Publication omitted the exact predecessor history.'
    Assert-Publication ((Get-FileHash $history -Algorithm SHA256).Hash -ceq $before) 'History does not preserve exact original bytes.'
    Assert-Publication ((Get-Content $StatePath -Raw | ConvertFrom-Json).generation -eq 2) 'Published state readback differs.'
    $stamp = (Get-Item $StatePath).LastWriteTimeUtc
    $files = @(Get-FixtureFiles)
    $result = Invoke-ActualStatePublication $value
    Assert-Publication ($result.exitCode -eq 0 -and (Get-Item $StatePath).LastWriteTimeUtc -eq $stamp) 'Identical state was unnecessarily rewritten.'
    Assert-Publication (@(Compare-Object $files @(Get-FixtureFiles)).Count -eq 0) 'Identical state created extra history/staging files.'
    Write-Output 'PASS: exact predecessor history and idempotent publication'

    # Protects foreign/interrupted staging and failed publication rollback.
    # Detects fixed .pending clobbering or a locked target being declared replaced.
    # Needs temp files; the real target-sharing lock specifically requires Windows.
    $foreign = $StatePath + '.pending'
    [IO.File]::WriteAllText($foreign, 'foreign interrupted staging')
    $result = Invoke-ActualStatePublication ([ordered]@{ schemaVersion = 1; generation = 3 })
    Assert-Publication ($result.exitCode -eq 0 -and [IO.File]::ReadAllText($foreign) -ceq 'foreign interrupted staging') 'Publication clobbered foreign fixed staging.'
    if ($IsWindows) {
        $before = (Get-FileHash $StatePath -Algorithm SHA256).Hash
        $lease = [IO.File]::Open($StatePath, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::Read)
        try { $result = Invoke-ActualStatePublication ([ordered]@{ schemaVersion = 1; generation = 4 }) }
        finally { $lease.Dispose() }
        Assert-Publication ($result.exitCode -eq 1 -and (Get-FileHash $StatePath -Algorithm SHA256).Hash -ceq $before) 'Locked publication lost its prior target or claimed success.'
        Assert-Publication (@(Get-ChildItem $workspace -File -Filter '*.pending-*').Count -eq 0) 'Owned failed staging was not cleaned.'
    }
    else { Write-Output 'NOT_RUN: Windows target-sharing failure requires Windows' }
    Write-Output 'PASS: foreign staging and locked predecessor are preserved'

    # Protects no-write plans and bounded allocation. Detects history/file writes
    # in DryRun or acceptance of oversized state. Needs pwsh + temp filesystem.
    # Breadcrumb: Publish-Usb4NetworkResult / Write-Usb4NetworkState byte bound.
    $files = @(Get-FixtureFiles)
    $null = Invoke-ActualStatePublication @{ schemaVersion = 1; generation = 9 } -DryRun
    Assert-Publication (@(Compare-Object $files @(Get-FixtureFiles)).Count -eq 0) 'DryRun wrote state or history.'
    $result = Invoke-ActualStatePublication @{ schemaVersion = 1; data = ('a' * (4MB + 1)) }
    Assert-Publication ($result.exitCode -eq 1 -and @(Compare-Object $files @(Get-FixtureFiles)).Count -eq 0) 'Oversized state was accepted or wrote files before refusal.'
    foreach ($invalid in @('scalar-state', 12, @(@{ generation = 4 }), @{ pendingTransaction = $null })) {
        $result = Invoke-ActualStatePublication -Value $invalid
        Assert-Publication ($result.exitCode -eq 1 -and @(Compare-Object $files @(Get-FixtureFiles)).Count -eq 0) 'Invalid new state was accepted or wrote history before refusal.'
    }
    Write-Output 'PASS: DryRun and oversized state never write'

    # Protects an existing history entry against substitution under a correct
    # filename. Detects adoption/overwrite of wrong bytes. Needs owned real files.
    # Breadcrumb: Write-Usb4NetworkState content-addressed history collision guard.
    $StatePath = Join-Path $workspace 'collision.json'
    [IO.File]::WriteAllText($StatePath, '{"schemaVersion":1,"generation":1}')
    $before = (Get-FileHash $StatePath -Algorithm SHA256).Hash
    $history = Join-Path ($StatePath + '.history') ($before.ToLowerInvariant() + '.json')
    [void][IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($history))
    [IO.File]::WriteAllText($history, '{"foreign":"keep"}')
    $files = @(Get-FixtureFiles)
    $result = Invoke-ActualStatePublication @{ schemaVersion = 1; generation = 2 }
    Assert-Publication ($result.exitCode -eq 1 -and @(Compare-Object $files @(Get-FixtureFiles)).Count -eq 0) 'History collision was adopted or overwritten.'
    Write-Output 'PASS: content-bound history collision fails closed'
    Write-Output "PASS: $script:checks real-file state publication assertions"
}
finally {
    $absolute = [IO.Path]::GetFullPath($workspace)
    $temporary = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
    if (-not $absolute.StartsWith($temporary, [StringComparison]::OrdinalIgnoreCase) -or
        [IO.Path]::GetFileName($absolute) -notmatch '^pcai-state-publication-[a-f0-9]{32}$') { throw 'Refusing cleanup outside the owned publication fixture.' }
    [IO.Directory]::Delete($absolute, $true)
}
