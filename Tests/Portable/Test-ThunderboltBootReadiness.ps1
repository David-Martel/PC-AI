#Requires -Version 7.0
[CmdletBinding()]
param([string]$SourcePath = (Join-Path $PSScriptRoot '../../Tools/SystemScripts/Networking/Configure-ThunderboltSwitchRoutes.ps1'))
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $SourcePath)) { throw 'Boot readiness policy is missing; legacy wrapper masks V&V exit.' }
. $SourcePath
function Assert-State {
    param([hashtable]$Snapshot, [hashtable]$Evidence, [int]$ChildExit, [string]$State, [int]$Code)
    $result = Get-ThunderboltBootReadiness -Snapshot $Snapshot -Evidence $Evidence -VnvExitCode $ChildExit
    if ($result.State -cne $State -or $result.ExitCode -ne $Code -or $result.NetworkReady) {
        throw "Expected $State/$Code without network readiness; got $($result | ConvertTo-Json -Compress)"
    }
}
$up = @{ ComputerName = 'fixture'; OsFamily = 'windows'; Adapters = @(@{ OperationalStatus = 'Up' }) }
$pass = @{ schema = 'vigil.hil-device-evidence/v1'; host = 'fixture'; overall_status = 'PASS'; custody_valid = $true;
    requirement_gates = @(@{ technicalVerdict = 'PASS' }, @{ technicalVerdict = 'SKIP_NOT_APPLICABLE' }) }
# Selector: TechnicalFailurePropagates. Protects failed canary/routes; detects the
# historical discarded child failure. Needs fixtures only. Breadcrumb: Get-ThunderboltBootReadiness.
$fail = $pass.Clone(); $fail.overall_status = 'FAIL'; $fail.requirement_gates = @(@{ technicalVerdict = 'FAIL' })
Assert-State $up $fail 1 'Failed' 1
Write-Output 'PASS: TechnicalFailurePropagates'
# Selector: AbsentAndDisconnectedAreUnavailable. Protects roaming laptops; detects
# an absent/down adapter becoming ready. Needs no hardware. Breadcrumb: same policy.
$absent = $up.Clone(); $absent.Adapters = @()
Assert-State $absent $fail 1 'Absent' 2
$down = $up.Clone(); $down.Adapters = @(@{ OperationalStatus = 'Down' })
Assert-State $down $fail 1 'Degraded' 2
Write-Output 'PASS: AbsentAndDisconnectedAreUnavailable'
# Selector: IntakeAndChildErrorsRemainDistinct. Protects technical vs acceptance
# evidence; detects ignoring unexplained native failure. Needs fixture PASS gates.
$pending = $pass.Clone(); $pending.overall_status = 'HOLD_PENDING_TPM_INTAKE'; $pending.custody_valid = $false
Assert-State $up $pending 1 'PendingAcceptance' 3
Assert-State $up $pass 1 'Failed' 1
Assert-State $up $pass 0 'Configured' 0
Write-Output 'PASS: IntakeAndChildErrorsRemainDistinct'
# Selector: MalformedAndWrongHostRefuseSuccess. Protects receipt binding; detects
# stale/wrong-host or incomplete JSON being treated as proof. Needs fixture data.
$wrong = $pass.Clone(); $wrong.host = 'other'
Assert-State $up $wrong 0 'Failed' 1
Assert-State $up @{} 0 'Failed' 1
$empty = $pass.Clone(); $empty.requirement_gates = @()
Assert-State $up $empty 0 'Failed' 1
Write-Output 'PASS: MalformedAndWrongHostRefuseSuccess'
# Selector: UnsupportedIsExplicit. Protects platform boundaries; detects an
# unimplemented platform masquerading as undocked. Needs no host APIs.
$linux = $up.Clone(); $linux.OsFamily = 'linux'
Assert-State $linux $pass 0 'Unsupported' 4
Write-Output 'PASS: UnsupportedIsExplicit'
# Selector: OnlyReadOnlyNativeCommands. Protects multihomed route custody; detects
# reintroducing discovery/banner-driven registration or any automatic Apply.
# Needs pwsh AST only. Breadcrumb: Invoke-ThunderboltBootObservation command list.
$tokens = $null; $parseErrors = $null
$ast = [Management.Automation.Language.Parser]::ParseFile((Resolve-Path $SourcePath).Path, [ref]$tokens, [ref]$parseErrors)
if ($parseErrors.Count) { throw 'Boot policy parse failed.' }
$commands = @($ast.FindAll({ param($node) $node -is [Management.Automation.Language.CommandAst] }, $true))
if (@($commands | Where-Object { $_.GetCommandName() -match '^(Set-Net|New-Net|Remove-Net|netsh)' }).Count) { throw 'Boot observation contains network mutation.' }
$runner = @($ast.FindAll({param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Invoke-ThunderboltBootObservation'}, $true))
if ($runner.Count -ne 1 -or $runner[0].Extent.Text -match "'discover'|'register'|'--apply'") { throw 'Unsafe enrollment returned to boot observation.' }
Write-Output 'PASS: OnlyReadOnlyNativeCommands'
# Selector: DryRunWritesNothing. Protects dry-run semantics including receipts;
# detects help/planning writing logs or invoking mutating commands. Needs fixture runner.
$script:nativeCalls = @()
function Invoke-VigilBoundedProcess {
    param($FilePath, $Arguments, $TimeoutSeconds)
    $script:nativeCalls += ,$Arguments
    $value = if ($Arguments[0] -eq 'status') { $up } else { $pass }
    [pscustomobject]@{ ExitCode = 0; Stdout = ($value | ConvertTo-Json -Depth 10); Stderr = '' }
}
$destination = Join-Path ([IO.Path]::GetTempPath()) ('pcai-dryrun-' + [guid]::NewGuid().ToString('N'))
$result = Invoke-ThunderboltBootObservation -TbExecutable 'fixture.exe' -RegistryPath 'fixture.json' -EvidenceDirectory $destination -DryRun
if ((Test-Path $destination) -or $script:nativeCalls.Count -ne 2 -or $result.ExitCode -ne 0) { throw 'DryRun wrote evidence or skipped actual read-only observation.' }
Write-Output 'PASS: DryRunWritesNothing'
