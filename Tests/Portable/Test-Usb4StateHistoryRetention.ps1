#Requires -Version 7.0
<# .SYNOPSIS
Detect recurring failed transactions exhausting controller recovery storage.
.DESCRIPTION
Uses actual reconciliation/checkpoint publication with the existing in-memory
network provider and owned temporary files. No installed state or live network.
#>
[CmdletBinding()]
param([string]$SourcePath = (Join-Path $PSScriptRoot '../../Tools/SystemScripts/Networking/Invoke-Usb4PeerNetwork.ps1'))
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. $SourcePath
# Reuse the already maintained network provider without executing its test suite.
$tokens = $null; $errors = $null
$fixture = [Management.Automation.Language.Parser]::ParseFile((Join-Path $PSScriptRoot 'Test-Usb4PeerNetwork.ps1'), [ref]$tokens, [ref]$errors)
if ($errors.Count) { throw 'Network fixture parse failure.' }
foreach ($name in @('Initialize-Fixture', 'Add-FixtureAddress', 'Get-FixtureSnapshot', 'Invoke-FixtureOperation', 'Save-FixtureCheckpoint')) {
    $definition = @($fixture.FindAll({ param($n) $n -is [Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -ceq $name }, $true))
    if ($definition.Count -ne 1) { throw "Expected one maintained fixture function: $name" }
    . ([scriptblock]::Create($definition[0].Extent.Text))
}
$workspace = Join-Path ([IO.Path]::GetTempPath()) ('pcai-state-retention-' + [guid]::NewGuid().ToString('N'))
[void][IO.Directory]::CreateDirectory($workspace)
$script:checks = 0
function Assert-Retention([bool]$Condition, [string]$Message) {
    if (-not $Condition) { throw $Message }
    $script:checks++
}
function Save-RetentionCheckpoint([hashtable]$State) {
    Save-FixtureCheckpoint $State
    Write-Usb4NetworkState -Path $script:retentionState -Value $State
    $script:retentionWrites++
}
try {
    # Protects recurring rollback custody and disk capacity. Detects unrestricted
    # per-minute checkpoint growth. Needs real files + in-memory network provider.
    # Breadcrumb: Invoke-Usb4NetworkReconcile / Write-Usb4NetworkState.
    $script:retentionState = Join-Path $workspace 'failed-attempts.json'
    $script:retentionWrites = 0
    Write-Usb4NetworkState -Path $script:retentionState -Value @{}
    for ($attempt = 0; $attempt -lt 16; $attempt++) {
        Initialize-Fixture Consumer
        $script:failOperation = 'AddRoute'; $script:failAfterMutation = $true
        $before = Get-Usb4NetworkFingerprint $script:model
        $result = Invoke-Usb4NetworkReconcile -Policy $script:policy -Snapshot (Get-FixtureSnapshot) -PreviousState @{} -Operation ${function:Invoke-FixtureOperation} -StateCheckpoint ${function:Save-RetentionCheckpoint} -ReportCheckpoint { param($Report) } -Apply
        Assert-Retention ($result.exitCode -eq 1 -and $result.rollbackErrors.Count -eq 0 -and (Get-Usb4NetworkFingerprint $script:model) -ceq $before) 'A repeated injected failure did not restore the exact network preimage.'
        Assert-Retention ((Read-Usb4NetworkState -Path $script:retentionState).Count -eq 0) 'Successful rollback lost its restored state custody.'
        $files = @(Get-ChildItem -LiteralPath ($script:retentionState + '.history') -File)
        Assert-Retention ($files.Count -le 128 -and ($files | Measure-Object Length -Sum).Sum -le 32MB) 'Repeated failed attempts exceeded the history budget.'
    }
    Assert-Retention ($script:retentionWrites -gt 128 -and $files.Count -lt $script:retentionWrites) 'Repeated-failure control did not actually cross the retention boundary.'
    if ($IsWindows) {
        . (Join-Path $PSScriptRoot '../../Tools/SystemScripts/Networking/Install-Usb4PeerNetwork.ps1')
        Assert-Usb4StateHistory -Path ($script:retentionState + '.history') -Operations (New-Usb4InstallerOperations)
        Assert-Retention $true 'Bounded controller history did not pass actual installer admission.'
    }
    Write-Output 'PASS: actual repeated failed transactions remain inside history budget'

    # Protects the byte budget and immediate predecessor despite large snapshots.
    # Detects a count-only cap and deletion of current recovery bytes. Needs temp
    # files; breadcrumb: Limit-Usb4StateHistory reserve-before-stage.
    $large = Join-Path $workspace 'large.json'
    for ($generation = 0; $generation -lt 19; $generation++) {
        $prior = if ([IO.File]::Exists($large)) { Get-Usb4BytesHash (Read-Usb4NetworkBytes $large) } else { $null }
        Write-Usb4NetworkState -Path $large -Value @{ generation = $generation; snapshot = ('x' * 2MB) }
        if ($prior) {
            $entries = @(Get-ChildItem -LiteralPath ($large + '.history') -File)
            Assert-Retention (($entries | Measure-Object Length -Sum).Sum -le 32MB -and $entries.Count -lt 19) 'Large history exceeded its byte budget.'
            $preserved = Join-Path ($large + '.history') ($prior.ToLowerInvariant() + '.json')
            Assert-Retention ([IO.File]::Exists($preserved) -and (Get-Usb4BytesHash (Read-Usb4NetworkBytes $preserved)) -ceq $prior) 'Retention lost the exact immediate predecessor.'
        }
    }
    Write-Output 'PASS: byte budget preserves exact immediate predecessor'

    # Protects foreign history, invalid identities and no-write operation. Detects
    # pruning before complete admission. Needs temp files, no external service.
    # Breadcrumb: Get-Usb4StateHistoryEntry / Limit-Usb4StateHistory.
    $foreign = Join-Path ($large + '.history') 'foreign.json'
    [IO.File]::WriteAllText($foreign, '{"foreign":"keep"}')
    $before = @((Get-ChildItem -LiteralPath $workspace -File -Recurse | ForEach-Object { $_.FullName + ':' + (Get-FileHash -LiteralPath $_.FullName).Hash }) | Sort-Object)
    $refused = $false
    try { Write-Usb4NetworkState -Path $large -Value @{ generation = 99; snapshot = ('x' * 2MB) } } catch { $refused = $true }
    $after = @((Get-ChildItem -LiteralPath $workspace -File -Recurse | ForEach-Object { $_.FullName + ':' + (Get-FileHash -LiteralPath $_.FullName).Hash }) | Sort-Object)
    Assert-Retention ($refused -and @(Compare-Object $before $after).Count -eq 0) 'Foreign history caused deletion or state publication before refusal.'
    $result = @{ state = @{ generation = 100 }; errors = @(); networkReady = $true; exitCode = 0 }
    $null = Publish-Usb4NetworkResult -Result $result -StateWriter { param($State) Write-Usb4NetworkState -Path $large -Value $State } -ReportWriter { param($Report) }
    $after = @((Get-ChildItem -LiteralPath $workspace -File -Recurse | ForEach-Object { $_.FullName + ':' + (Get-FileHash -LiteralPath $_.FullName).Hash }) | Sort-Object)
    Assert-Retention (@(Compare-Object $before $after).Count -eq 0) 'No-Apply publication pruned history.'
    # Protects oversized legacy stores from automatic deletion and bounds installer
    # inspection. Detects unbounded enumeration/adoption. Needs owned real files;
    # breadcrumb: both history budget admission functions.
    $overBudget = $script:retentionState + '.history'
    # A fresh current generation has not itself been archived yet, so the next
    # write genuinely needs another history slot even if an older {} is retained.
    Write-Usb4NetworkState -Path $script:retentionState -Value @{ retentionControl = 'current' }
    $files = @(Get-ChildItem -LiteralPath $overBudget -File)
    for ($index = $files.Count; $index -lt 128; $index++) {
        $extraBytes = [Text.Encoding]::UTF8.GetBytes(('{"extra":' + $index + '}'))
        $extra = Join-Path $overBudget ((Get-Usb4BytesHash $extraBytes).ToLowerInvariant() + '.json')
        Write-Usb4NetworkBytes -Path $extra -Bytes $extraBytes -CreateOnly
    }
    $victim = @(Get-ChildItem -LiteralPath $overBudget -File)[0].FullName
    $victimBytes = Read-Usb4NetworkBytes $victim
    Write-Usb4NetworkBytes -Path $victim -Bytes ([Text.Encoding]::UTF8.GetBytes('{"substituted":true}'))
    $before = @((Get-ChildItem -LiteralPath $overBudget -File | ForEach-Object { $_.Name + ':' + (Get-FileHash -LiteralPath $_.FullName).Hash }) | Sort-Object)
    $refused = $false
    try { Write-Usb4NetworkState -Path $script:retentionState -Value @{ newer = 1 } } catch { $refused = $true }
    $after = @((Get-ChildItem -LiteralPath $overBudget -File | ForEach-Object { $_.Name + ':' + (Get-FileHash -LiteralPath $_.FullName).Hash }) | Sort-Object)
    Assert-Retention ($refused -and @(Compare-Object $before $after).Count -eq 0) "Pruning failed substituted-generation refusal (refused=$refused before=$($before.Count) after=$($after.Count) differences=$(@(Compare-Object $before $after).Count))."
    Write-Usb4NetworkBytes -Path $victim -Bytes $victimBytes
    if ($IsWindows) {
        $originalAcl = Get-Acl -LiteralPath $victim
        $foreignAcl = Get-Acl -LiteralPath $victim
        $foreignAcl.AddAccessRule([Security.AccessControl.FileSystemAccessRule]::new([Security.Principal.SecurityIdentifier]::new('S-1-5-32-545'), 'Read', 'Allow'))
        Set-Acl -LiteralPath $victim -AclObject $foreignAcl
        $priorAcl = (Get-Acl -LiteralPath $victim).Sddl
        $expected = @((Get-ChildItem -LiteralPath $overBudget -File | ForEach-Object { $_.Name + ':' + (Get-FileHash -LiteralPath $_.FullName).Hash }) | Sort-Object)
        $refused = $false
        try { Write-Usb4NetworkState -Path $script:retentionState -Value @{ newer = 1 } } catch { $refused = $true }
        $after = @((Get-ChildItem -LiteralPath $overBudget -File | ForEach-Object { $_.Name + ':' + (Get-FileHash -LiteralPath $_.FullName).Hash }) | Sort-Object)
        Assert-Retention ($refused -and (Get-Acl -LiteralPath $victim).Sddl -ceq $priorAcl -and @(Compare-Object $after $expected).Count -eq 0) 'Pruning changed a foreign-grant generation.'
        Set-Acl -LiteralPath $victim -AclObject $originalAcl
    }
    $extraBytes = [Text.Encoding]::UTF8.GetBytes('{"extra":"over-budget"}')
    $extra = Join-Path $overBudget ((Get-Usb4BytesHash $extraBytes).ToLowerInvariant() + '.json')
    Write-Usb4NetworkBytes -Path $extra -Bytes $extraBytes -CreateOnly
    $before = @((Get-ChildItem -LiteralPath $overBudget -File | ForEach-Object { $_.Name + ':' + (Get-FileHash -LiteralPath $_.FullName).Hash }) | Sort-Object)
    $refused = $false
    try { Write-Usb4NetworkState -Path $script:retentionState -Value @{ newer = 1 } } catch { $refused = $true }
    $after = @((Get-ChildItem -LiteralPath $overBudget -File | ForEach-Object { $_.Name + ':' + (Get-FileHash -LiteralPath $_.FullName).Hash }) | Sort-Object)
    Assert-Retention ($refused -and @(Compare-Object $before $after).Count -eq 0) 'Oversized legacy store was pruned automatically.'
    if ($IsWindows) {
        $refused = $false
        try { Assert-Usb4StateHistory -Path $overBudget -Operations (New-Usb4InstallerOperations) } catch { $refused = $true }
        Assert-Retention $refused 'Installer adopted an oversized legacy store.'
    }
    Write-Output "PASS: $script:checks history retention assertions"
}
finally {
    $absolute = [IO.Path]::GetFullPath($workspace)
    $temporary = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
    if (-not $absolute.StartsWith($temporary, [StringComparison]::OrdinalIgnoreCase) -or [IO.Path]::GetFileName($absolute) -notmatch '^pcai-state-retention-[a-f0-9]{32}$') { throw 'Refusing cleanup outside the owned retention fixture.' }
    [IO.Directory]::Delete($absolute, $true)
}
