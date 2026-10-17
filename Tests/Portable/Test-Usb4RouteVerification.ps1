#Requires -Version 7.0
[CmdletBinding()]
param([string]$SourcePath=(Join-Path $PSScriptRoot '../../Modules/PC-AI.Drivers/Public/Invoke-ThunderboltPeerLifecycle.ps1'))
$ErrorActionPreference='Stop'
$tokens=$null;$errors=$null
$ast=[Management.Automation.Language.Parser]::ParseFile((Resolve-Path $SourcePath).Path,[ref]$tokens,[ref]$errors)
if($errors.Count){throw 'Lifecycle parse failure'}
$definition=$ast.Find({param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Get-SafeCimRows'},$true)
. ([scriptblock]::Create($definition.Extent.Text))
function Get-CimInstance {
    [CmdletBinding()]
    param([string]$Namespace,[string]$ClassName)
    switch($script:QueryMode) {
        'error' {throw 'intentional CIM query failure'}
        'present' {[pscustomobject]@{InterfaceIndex=22;DestinationPrefix='fd00::1/128'}}
        'empty' {return @()}
    }
}
# Protects route postconditions: failure must not mean absence. Detects the old
# catch-return-empty behavior. Needs only pwsh; calls the actual extracted helper.
$script:QueryMode='error';$caught=$false
try {$null=Get-SafeCimRows -ClassName MSFT_NetRoute -Required} catch {
    if ($_.Exception.Message -notmatch 'intentional CIM query failure') { throw }
    $caught=$true
}
if(-not $caught){throw 'Required query failure was accepted as absence'}
Write-Output 'PASS: required route query failure propagates'
# Protects optional inventory compatibility; detects unintentional global change
# in best-effort discovery. Needs no CIM server; same helper breadcrumb.
if(@(Get-SafeCimRows -ClassName MSFT_NetRoute).Count -ne 0){throw 'Optional query fallback changed'}
Write-Output 'PASS: optional inventory query remains best effort'
# Protects real empty/present results; detects rejecting all queries or fabricating
# absence. Needs deterministic CIM fixture; same helper breadcrumb.
$script:QueryMode='empty'
if(@(Get-SafeCimRows -ClassName MSFT_NetRoute -Required).Count -ne 0){throw 'Valid empty query rejected'}
$script:QueryMode='present'
$rows=@(Get-SafeCimRows -ClassName MSFT_NetRoute -Required)
if($rows.Count -ne 1 -or $rows[0].InterfaceIndex -ne 22){throw 'Present route lost'}
Write-Output 'PASS: required query preserves valid empty and present results'
# Protects strict helper use at mutation verification; detects unwiring the required
# query from the route postcondition. Needs source AST only; no route mutation.
$strictCalls=@($ast.FindAll({param($node)
    $node -is [Management.Automation.Language.CommandAst] -and
    $node.GetCommandName() -eq 'Get-SafeCimRows' -and
    $node.Extent.Text.Contains('MSFT_NetRoute') -and
    @($node.CommandElements | Where-Object {$_ -is [Management.Automation.Language.CommandParameterAst] -and $_.ParameterName -eq 'Required'}).Count -gt 0
},$true))
if($strictCalls.Count -eq 0){throw 'Route postcondition does not require a successful query'}
Write-Output 'PASS: route postcondition uses required query'
