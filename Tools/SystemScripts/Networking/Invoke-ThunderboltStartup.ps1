#Requires -Version 7.0
<#
.SYNOPSIS
Run enrolled address recovery and native observation with separate failure records.
.DESCRIPTION
The existing Windows task invokes this entrypoint using explicit machine paths.
Controller failure never skips native observation. Both process outcomes remain
in the attempt receipt; receipt publication failure returns nonzero as well.
See docs/usb4/windows-boot.md. DryRun performs both observations without writing.
#>
[CmdletBinding()]
param(
    [string]$PolicyPath=(Join-Path ([Environment]::GetFolderPath('CommonApplicationData')) 'pcai-usb4/windows-partners.json'),
    [string]$StatePath=(Join-Path ([Environment]::GetFolderPath('CommonApplicationData')) 'pcai-usb4/windows-owned-edges.json'),
    [string]$TbExecutable=(Join-Path $HOME 'bin/pcai-tb.exe'),
    [string]$RegistryPath=(Join-Path $HOME '.machine/thunderbolt-peers.json'),
    [string]$EvidenceDirectory=(Join-Path $HOME '.machine/logs/usb4-boot'),
    [switch]$DryRun,
    [Alias('h','help')][switch]$ShowHelp
)
Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'

function Assert-Usb4ProcessOutcome {
    <# .SYNOPSIS
    Require one native outcome with a real integral status and captured streams.
    #>
    param([object]$Outcome)
    if ($null -eq $Outcome -or $Outcome -is [Array]) { throw 'Component must return exactly one native process outcome.' }
    $values=@{}
    foreach($name in @('ExitCode','Stdout','Stderr')) {
        if ($Outcome -is [Collections.IDictionary]) {
            if (-not $Outcome.Contains($name)) { throw "Component outcome lacks $name." }
            $values[$name]=$Outcome[$name]
        } else {
            $property=$Outcome.PSObject.Properties[$name]
            if (-not $property) { throw "Component outcome lacks $name." }
            $values[$name]=$property.Value
        }
    }
    $code=$values.ExitCode
    if ($null -eq $code -or [Type]::GetTypeCode($code.GetType()).ToString() -notin
        @('SByte','Byte','Int16','UInt16','Int32','UInt32','Int64','UInt64') -or
        [decimal]$code -lt [int]::MinValue -or [decimal]$code -gt [int]::MaxValue -or
        $values.Stdout -isnot [string] -or $values.Stderr -isnot [string]) { throw 'Component outcome has invalid status or streams.' }
}

function Invoke-Usb4StartupOperations {
    <# .SYNOPSIS
    Preserve independent component failures while always attempting observation.
    #>
    param([scriptblock]$ControllerOperation,[scriptblock]$ObserverOperation,[scriptblock]$PublishOperation)
    try { $controller=& $ControllerOperation; Assert-Usb4ProcessOutcome $controller }
    catch { $controller=@{ExitCode=1;Stdout='';Stderr=$_.Exception.Message;State='ControllerException'} }
    try { $observer=& $ObserverOperation; Assert-Usb4ProcessOutcome $observer }
    catch { $observer=@{ExitCode=1;Stdout='';Stderr=$_.Exception.Message;State='ObserverException'} }
    $code=[int]$observer.ExitCode
    if ($code -eq 0 -and $controller.ExitCode -ne 0) { $code=[int]$controller.ExitCode }
    $record=@{schema=1;observed_at=[DateTimeOffset]::UtcNow.ToString('o');controller=$controller;observer=$observer;
        scope='independent-startup-components';ExitCode=$code;publication_error=$null}
    try { & $PublishOperation $record | Out-Null }
    catch { $record.publication_error=$_.Exception.Message;$record.ExitCode=1 }
    $record
}

if ($MyInvocation.InvocationName -eq '.') { return }
if ($ShowHelp) { Get-Help $PSCommandPath -Detailed;return }
if (-not $IsWindows) { throw 'USB4 startup deployment requires Windows.' }
. (Join-Path $PSScriptRoot 'Invoke-VigilBoundedProcess.ps1')
$pwsh=Join-Path $PSHOME 'pwsh.exe'
$record=Invoke-Usb4StartupOperations -ControllerOperation {
    $arguments=@('-NoLogo','-NoProfile','-NonInteractive','-WindowStyle','Hidden','-File',
        (Join-Path $PSScriptRoot 'Invoke-ThunderboltEnrolledPeer.ps1'),'-PolicyPath',$PolicyPath,'-StatePath',$StatePath)
    $arguments+= $(if($DryRun){'-DryRun'}else{'-Apply'})
    Invoke-VigilBoundedProcess -FilePath $pwsh -Arguments $arguments -TimeoutSeconds 45
} -ObserverOperation {
    $arguments=@('-NoLogo','-NoProfile','-NonInteractive','-WindowStyle','Hidden','-File',
        (Join-Path $PSScriptRoot 'Configure-ThunderboltSwitchRoutes.ps1'),'-TbExecutable',$TbExecutable,
        '-RegistryPath',$RegistryPath,'-EvidenceDirectory',$EvidenceDirectory)
    if ($DryRun) {$arguments+='-DryRun'}
    Invoke-VigilBoundedProcess -FilePath $pwsh -Arguments $arguments -TimeoutSeconds 45
} -PublishOperation {
    param($Attempt)
    if (-not $DryRun) {
        $path=Join-Path $EvidenceDirectory 'enrolled-last-attempt.json'
        $staging=$path+'.pending'
        [IO.File]::WriteAllText($staging,($Attempt|ConvertTo-Json -Depth 20),[Text.UTF8Encoding]::new($false))
        [IO.File]::Move($staging,$path,$true)
    }
}
$record|ConvertTo-Json -Depth 20
exit $record.ExitCode
