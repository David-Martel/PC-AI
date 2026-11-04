#Requires -Version 7.0
param([string]$SourcePath = (Join-Path $PSScriptRoot '../../Tools/SystemScripts/Networking/Invoke-ThunderboltRcloneMount.ps1'))
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
if (-not (Test-Path $SourcePath)) { throw 'USB4 mount gate is missing; legacy startup mount can follow the default LAN route.' }
. $SourcePath
$adapter = @{ InterfaceDescription = 'USB4(TM) P2P Network Adapter'; Status = $null; InterfaceOperationalStatus = 1; AdminStatus = 1; ifIndex = 28 }
$addresses = @(@{ IPAddress = '172.31.252.134'; InterfaceIndex = 28; AddressState = 'Preferred' })
$route = @(@{ InterfaceIndex = 28; DestinationPrefix = '172.31.252.132/30'; NextHop = '0.0.0.0' })
# Selector: OwnSourceAndOnLinkRouteAdmitMount. Protects useful USB4 mounts;
# detects an unconditional deny gate. Needs fixtures only. Breadcrumb: Test-ThunderboltMountPath.
if (-not (Test-ThunderboltMountPath $adapter $addresses $route '172.31.252.134')) { throw 'Valid USB4 source/route rejected.' }
Write-Output 'PASS: OwnSourceAndOnLinkRouteAdmitMount'
# Selector: FallbackAndWrongSourceRefuseMount. Protects roaming multihomed hosts;
# detects missing/foreign sources and default-LAN fallback. Needs no host APIs.
# Breadcrumb: Test-ThunderboltMountPath interface ownership and on-link checks.
$wrong = @(@{ InterfaceIndex = 31; DestinationPrefix = '0.0.0.0/0'; NextHop = '192.168.50.1' })
if (Test-ThunderboltMountPath $adapter $addresses $wrong '172.31.252.134') { throw 'Default LAN fallback admitted.' }
if (Test-ThunderboltMountPath $adapter @() $route '172.31.252.134') { throw 'Absent source admitted.' }
$foreign = @(@{ IPAddress = '172.31.252.134'; InterfaceIndex = 31; AddressState = 'Preferred' })
if (Test-ThunderboltMountPath $adapter $foreign $route '172.31.252.134') { throw 'Foreign source admitted.' }
Write-Output 'PASS: FallbackAndWrongSourceRefuseMount'
# Selector: AbsentOrNonUsb4RefuseMount. Protects interface custody; detects an
# alias alone admitting a wired/WiFi adapter or missing hardware. Needs fixtures.
# Breadcrumb: Test-ThunderboltMountPath actual driver description/status checks.
$down = $adapter.Clone(); $down.InterfaceOperationalStatus = 2; $down.Status = 'Up'
if (Test-ThunderboltMountPath $down $addresses $route '172.31.252.134') { throw 'Disconnected adapter admitted.' }
$lan = $adapter.Clone(); $lan.InterfaceDescription = 'Wired Ethernet'
if (Test-ThunderboltMountPath $lan $addresses $route '172.31.252.134') { throw 'Unrelated Ethernet adapter admitted.' }
$disabled = $adapter.Clone(); $disabled.AdminStatus = 2
if (Test-ThunderboltMountPath $disabled $addresses $route '172.31.252.134') { throw 'Administratively disabled USB4 adapter admitted.' }
Write-Output 'PASS: AbsentOrNonUsb4RefuseMount'
# Selector: ExternalSshAndTofuAreDisabled. Protects pin/source binding if a user's
# backend/environment later drifts; detects external SSH or first-use trust
# bypassing the native transport. Needs fixtures only. Breadcrumb: transport args.
if (-not (Get-Command Get-ThunderboltMountTransportArguments -ErrorAction SilentlyContinue)) { throw 'External SSH override is missing.' }
$transport = @(Get-ThunderboltMountTransportArguments '172.31.252.133' '172.31.252.134' 'fixture-known-hosts')
$sshIndex = [Array]::IndexOf($transport, '--sftp-ssh')
if ($sshIndex -lt 0 -or $transport[$sshIndex + 1] -cne '' -or
    '--sftp-pin-host-key=false' -notin $transport -or
    $transport[[Array]::IndexOf($transport, '--bind') + 1] -ne '172.31.252.134' -or
    $transport[[Array]::IndexOf($transport, '--sftp-known-hosts-file') + 1] -ne 'fixture-known-hosts') {
    throw 'Native pinned/source-bound transport can be bypassed by configuration.'
}
Write-Output 'PASS: ExternalSshAndTofuAreDisabled'
# Selector: MalformedKnownHostContentRefusesReadiness. Protects honest DryRun
# readiness; detects a nonempty deployment receipt such as literal 1 being
# admitted as a host pin. Needs fixtures only, no files or host APIs.
# Breadcrumb: Test-ThunderboltKnownHostContent and the pre-DryRun pin gate.
if (-not (Get-Command Test-ThunderboltKnownHostContent -ErrorAction SilentlyContinue)) {
    throw 'Host-pin content preflight is missing; literal 1 can be reported ready.'
}
foreach ($bad in @("1`n", '', "# comment only`n", 'peer ssh-ed25519 not-base64!', 'peer ssh-ed25519 AAAA')) {
    if (Test-ThunderboltKnownHostContent -Content $bad) { throw 'Malformed or empty host-pin content admitted.' }
}
Write-Output 'PASS: MalformedKnownHostContentRefusesReadiness'
# Selector: ValidKnownHostContentPreservesAdmission. Protects useful pinned
# mounts; detects rejecting valid OpenSSH records, comments and trailing notes.
# Needs a public wire-format fixture only; this is not remote identity approval.
# Breadcrumb: Test-ThunderboltKnownHostContent record framing.
$keyType = [Text.Encoding]::ASCII.GetBytes('ssh-ed25519')
$publicBlob = [byte[]](@(0, 0, 0, $keyType.Length) + $keyType + @(0, 0, 0, 32) + (1..32))
$publicKey = [Convert]::ToBase64String($publicBlob)
$validRecord = "peer,192.0.2.1 ssh-ed25519 $publicKey fixture-public-key"
if (-not (Test-ThunderboltKnownHostContent -Content "# fixture`n`n$validRecord`n")) {
    throw 'Valid public host-pin record rejected.'
}
if (Test-ThunderboltKnownHostContent -Content "$validRecord`n1`n") {
    throw 'A valid record hid a malformed second line.'
}
if (Test-ThunderboltKnownHostContent -Content "peer ssh-rsa $publicKey") {
    throw 'Outer key type disagreed with the encoded public key.'
}
if (Test-ThunderboltKnownHostContent -Content "|bad-hash ssh-ed25519 $publicKey") {
    throw 'Malformed hashed host pattern admitted.'
}
$hashedPart = [Convert]::ToBase64String([byte[]](1..20))
if (-not (Test-ThunderboltKnownHostContent -Content "|1|$hashedPart|$hashedPart ssh-ed25519 $publicKey")) {
    throw 'Valid hashed host-pin record rejected.'
}
if (-not (Test-ThunderboltKnownHostContent -Content "@cert-authority *.example ssh-ed25519 $publicKey")) {
    throw 'Valid CA public-key record rejected.'
}
if (Test-ThunderboltKnownHostContent -Content "@revoked peer ssh-ed25519 $publicKey") {
    throw 'An all-revoked file established a usable pin.'
}
if (Test-ThunderboltKnownHostContent -Content "@unknown peer ssh-ed25519 $publicKey") {
    throw 'Unknown record marker admitted.'
}
$truncatedKey = [Convert]::ToBase64String([byte[]]$publicBlob[0..($publicBlob.Length - 2)])
if (Test-ThunderboltKnownHostContent -Content "peer ssh-ed25519 $truncatedKey") {
    throw 'Truncated public-key wire field admitted.'
}
Write-Output 'PASS: ValidKnownHostContentPreservesAdmission'
# Selector: PinPreflightPrecedesReadinessAndTransport. Protects no-write DryRun
# and transport admission; detects moving pin validation behind either action.
# Needs source AST only. Breadcrumb: top-level pin gate, DryRun and process Start.
$parseTokens = $null; $parseErrors = $null
$sourceAst = [Management.Automation.Language.Parser]::ParseFile($SourcePath, [ref]$parseTokens, [ref]$parseErrors)
if ($parseErrors.Count -ne 0) { throw 'Mount entrypoint contains parse errors.' }
$pinCalls = @($sourceAst.FindAll({
    param($node)
    $node -is [Management.Automation.Language.CommandAst] -and $node.GetCommandName() -eq 'Test-ThunderboltKnownHostContent'
}, $true))
$dryRunBlocks = @($sourceAst.FindAll({
    param($node)
    $node -is [Management.Automation.Language.IfStatementAst] -and $node.Clauses[0].Item1.Extent.Text -eq '$DryRun'
}, $true))
$startCalls = @($sourceAst.FindAll({
    param($node)
    $node -is [Management.Automation.Language.InvokeMemberExpressionAst] -and
        $node.Expression.Extent.Text -eq '$process' -and $node.Member.Extent.Text -eq 'Start'
}, $true))
if ($pinCalls.Count -ne 1 -or $dryRunBlocks.Count -ne 1 -or $startCalls.Count -ne 1 -or
    $pinCalls[0].Extent.StartOffset -ge $dryRunBlocks[0].Extent.StartOffset -or
    $pinCalls[0].Extent.StartOffset -ge $startCalls[0].Extent.StartOffset) {
    throw 'Host-pin content can be bypassed by DryRun or transport startup.'
}
Write-Output 'PASS: PinPreflightPrecedesReadinessAndTransport'
