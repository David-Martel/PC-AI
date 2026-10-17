#Requires -Version 7.0
[CmdletBinding()]
param([string]$SourcePath=(Join-Path $PSScriptRoot '../../Modules/PC-AI.Drivers/Public/Invoke-ThunderboltPeerLifecycle.ps1'))
$ErrorActionPreference='Stop'
$tokens=$null;$errors=$null
$ast=[Management.Automation.Language.Parser]::ParseFile((Resolve-Path -LiteralPath $SourcePath).Path,[ref]$tokens,[ref]$errors)
if($errors.Count){throw 'Production lifecycle source does not parse'}
$loops=@($ast.FindAll({param($n) $n -is [Management.Automation.Language.ForEachStatementAst] -and $n.Variable.VariablePath.UserPath -eq 'route' -and $n.Extent.Text.Contains('WithdrawStaleOwnedRoute')},$true))
if($loops.Count -ne 1){throw "Expected one production withdrawal loop; found $($loops.Count)"}
$script:productionLoop=[scriptblock]::Create($loops[0].Extent.Text)
function Invoke-OwnershipFixture {
    [CmdletBinding(SupportsShouldProcess)]
    param([int[]]$EvaluatedInterfaces,[string]$Prefix='fd00:7462:7032:d700::99/128',[int]$RouteInterface=999)
    $netRoutes=@([pscustomobject]@{DestinationPrefix=$Prefix;InterfaceIndex=$RouteInterface})
    $evaluatedUsb4IfIndices=@($EvaluatedInterfaces);$prefixToPeer=@{};$preservedEndpoints=@{};$activeReadyPeers=@{};$linkReports=@();$canonicalLocalHost='fixture'
    $Apply=$true;$WithdrawStaleRoutes=$true;$routeActions=[Collections.Generic.List[object]]::new();$script:deleteCalls=@()
    function ConvertTo-NormalizedPrefix {param($Prefix) $Prefix.ToLowerInvariant()}
    # No host API is invoked: these doubles record planner intent and route readback.
    function Invoke-ScopedRouteDelete {param($DestinationPrefix,$InterfaceIndex) $script:deleteCalls+=@([pscustomobject]@{Prefix=$DestinationPrefix;InterfaceIndex=$InterfaceIndex})}
    function Get-SafeCimRows {param($ClassName,[switch]$Required) @()}
    . $script:productionLoop
    [pscustomobject]@{Deletes=@($script:deleteCalls);Actions=@($routeActions)}
}
# Protects fail-closed custody when hardware disappears; detects an empty inventory
# broadening withdrawal to foreign interfaces. Needs pwsh7 only, mocked host APIs.
# Breadcrumb: Invoke-ThunderboltPeerLifecycle.ps1 route ownership/withdrawal loop.
$r=Invoke-OwnershipFixture -EvaluatedInterfaces @() -Confirm:$false
if($r.Deletes.Count -ne 0 -or $r.Actions[0].Action -ne 'PreserveForeignInterfaceRoute'){throw 'Empty USB4 inventory admitted route withdrawal'}
Write-Output 'PASS: empty USB4 inventory preserves foreign routes'
# Protects unrelated adapters; detects withdrawal based solely on a PC-AI prefix.
# Needs pwsh7 only; breadcrumb: production evaluatedUsb4IfIndices membership guard.
$r=Invoke-OwnershipFixture -EvaluatedInterfaces @(44) -Confirm:$false
if($r.Deletes.Count -ne 0 -or $r.Actions[0].Action -ne 'PreserveForeignInterfaceRoute'){throw 'Foreign interface admitted route withdrawal'}
Write-Output 'PASS: nonmember interface preserves foreign routes'
# Protects legitimate cleanup; detects an unconditional no-op pretending to fix custody.
# Needs mocked deletion/readback; breadcrumb: production WithdrawStaleOwnedRoute path.
$r=Invoke-OwnershipFixture -EvaluatedInterfaces @(999) -Confirm:$false
if($r.Deletes.Count -ne 1 -or $r.Deletes[0].InterfaceIndex -ne 999 -or $r.Deletes[0].Prefix -ne 'fd00:7462:7032:d700::99/128' -or -not $r.Actions[0].Executed){throw 'Owned interface withdrawal failed'}
Write-Output 'PASS: positively inventoried interface permits scoped withdrawal'
# Protects operator WhatIf; detects mutation bypassing ShouldProcess.
# Needs pwsh7 ShouldProcess; breadcrumb: production withdrawal ShouldProcess guard.
$r=Invoke-OwnershipFixture -EvaluatedInterfaces @(999) -WhatIf
if($r.Deletes.Count -ne 0 -or $r.Actions[0].Executed){throw 'WhatIf admitted route withdrawal'}
Write-Output 'PASS: WhatIf preserves an owned route'

# Protects default connectivity; detects withdrawal bypassing the default-route exclusion.
# Needs mocked host APIs; breadcrumb: production default prefix exclusion.
$r=Invoke-OwnershipFixture -EvaluatedInterfaces @(999) -Prefix '::/0' -Confirm:$false
if($r.Deletes.Count -ne 0 -or $r.Actions.Count -ne 0){throw 'Default route was considered for withdrawal'}
Write-Output 'PASS: default route remains excluded'
# Protects unrelated route space; detects overbroad prefix ownership classification.
# Needs mocked host APIs; breadcrumb: production PC-AI prefix range exclusion.
$r=Invoke-OwnershipFixture -EvaluatedInterfaces @(999) -Prefix 'fd99::99/128' -Confirm:$false
if($r.Deletes.Count -ne 0 -or $r.Actions.Count -ne 0){throw 'Non-PC-AI route was considered for withdrawal'}
Write-Output 'PASS: non-PC-AI route remains excluded'
