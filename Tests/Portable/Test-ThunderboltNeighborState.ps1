#Requires -Version 7.0
[CmdletBinding()]
param(
    [string]$SourcePath = (Join-Path $PSScriptRoot '../../Modules/PC-AI.Drivers/Public/Get-ThunderboltNetworkStatus.ps1')
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Load only the real nested resolver. Parsing the source cannot enumerate devices,
# load the drivers module, probe peers, or change the enrolled network.
$tokens = $null
$parseErrors = $null
$ast = [Management.Automation.Language.Parser]::ParseFile($SourcePath, [ref]$tokens, [ref]$parseErrors)
if ($parseErrors.Count -ne 0) { throw ($parseErrors | Out-String) }
$resolvers = @($ast.FindAll({
            param($node)
            $node -is [Management.Automation.Language.FunctionDefinitionAst] -and
            $node.Name -eq 'Resolve-NeighborState'
        }, $true))
if ($resolvers.Count -ne 1) { throw 'Expected exactly one Resolve-NeighborState source function.' }
. ([scriptblock]::Create($resolvers[0].Extent.Text))

$results = [Collections.Generic.List[object]]::new()
function Assert-NeighborLabel {
    param([string]$Selector, [object]$InputValue, [string]$Expected)
    $actual = Resolve-NeighborState -State $InputValue
    $results.Add([pscustomobject]@{
            selector = $Selector
            input    = $InputValue
            expected = $Expected
            actual   = $actual
            pass     = $actual -ceq $Expected
        })
}

# Selector group: NumericNeighborStatesMatchWindows. Protects accurate Windows
# reachability reporting; detects Probe/Reachable and Delay/Stale label inversions.
# Needs only the source AST and PowerShell; no module, hardware, CIM or network.
# Breadcrumb: Modules/PC-AI.Drivers/Public/Get-ThunderboltNetworkStatus.ps1:
# Resolve-NeighborState. All seven byte/integer/numeric-string cases are named below.
# Reference: https://learn.microsoft.com/en-us/windows/win32/fwp/wmi/nettcpipprov/msft-netneighbor
$labels = @('Unreachable', 'Incomplete', 'Probe', 'Delay', 'Stale', 'Reachable', 'Permanent')
for ($value = 0; $value -lt $labels.Count; $value++) {
    Assert-NeighborLabel "NumericNeighborStatesMatchWindows.byte.$value" ([byte]$value) $labels[$value]
    Assert-NeighborLabel "NumericNeighborStatesMatchWindows.integer.$value" $value $labels[$value]
    Assert-NeighborLabel "NumericNeighborStatesMatchWindows.string.$value" ([string]$value) $labels[$value]
}

# Selector group: TextAndUnknownNeighborStatesRemainStable. Protects named-state
# compatibility and visibility of future provider values; detects unintended
# normalization, invented reachability or rejection of null/blank/unknown input.
# Needs only source AST and values below. Breadcrumb: same resolver as above.
foreach ($label in $labels) {
    Assert-NeighborLabel "TextAndUnknownNeighborStatesRemainStable.named.$label" $label $label
}
Assert-NeighborLabel 'TextAndUnknownNeighborStatesRemainStable.null' $null 'Unknown'
Assert-NeighborLabel 'TextAndUnknownNeighborStatesRemainStable.empty' '' 'Unknown'
Assert-NeighborLabel 'TextAndUnknownNeighborStatesRemainStable.whitespace' '   ' 'Unknown'
Assert-NeighborLabel 'TextAndUnknownNeighborStatesRemainStable.unknown-label' 'Unknown' 'Unknown'
Assert-NeighborLabel 'TextAndUnknownNeighborStatesRemainStable.future-label' 'FutureProviderState' 'FutureProviderState'
Assert-NeighborLabel 'TextAndUnknownNeighborStatesRemainStable.future-number' ([byte]7) '7'

$failed = @($results | Where-Object { -not $_.pass })
[pscustomobject]@{
    sourcePath   = (Resolve-Path -LiteralPath $SourcePath).Path
    sourceSha256 = (Get-FileHash -LiteralPath $SourcePath -Algorithm SHA256).Hash
    testSha256   = (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash
    caseCount    = $results.Count
    failureCount = $failed.Count
    results      = @($results.ToArray())
} | ConvertTo-Json -Depth 5
if ($failed.Count -gt 0) { throw "Neighbor-state regression: $($failed.Count) of $($results.Count) cases failed." }