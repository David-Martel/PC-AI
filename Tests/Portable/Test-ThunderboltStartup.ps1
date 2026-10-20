#Requires -Version 7.0
param([string]$SourcePath=(Join-Path $PSScriptRoot '../../Tools/SystemScripts/Networking/Invoke-ThunderboltStartup.ps1'))
Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'
. $SourcePath
# Selector: StartupComponentsRetainFailures. Protects independent technical and
# enrolled readiness; detects success masking. Needs operation fixtures only.
# Breadcrumb: Invoke-Usb4StartupOperations exit selection and component records.
$ok={@{ExitCode=0;Stdout='ok';Stderr=''}}
$technical={@{ExitCode=1;Stdout='native failure';Stderr=''}}
$unavailable={@{ExitCode=2;Stdout='absent';Stderr=''}}
$script:published=0
$publish={param($Attempt);$script:published++}
$r=Invoke-Usb4StartupOperations $ok $technical $publish
if ($r.ExitCode-ne1-or$r.controller.ExitCode-ne0-or$r.observer.Stdout-ne'native failure'){throw 'Native failure was masked.'}
$r=Invoke-Usb4StartupOperations $unavailable $ok $publish
if ($r.ExitCode-ne2-or$r.controller.ExitCode-ne2){throw 'Controller unavailability was masked.'}
$r=Invoke-Usb4StartupOperations $ok $ok $publish
if ($r.ExitCode-ne0-or$published-ne3){throw 'Positive startup control failed.'}
Write-Output 'PASS: StartupComponentsRetainFailures'
# Selector: ControllerAndPublicationErrorsStillObserve. Protects technical
# observation after helper, launch or filesystem errors. Detects skipped observer
# and false green. Needs scriptblock exceptions; breadcrumb: orchestration catches.
$script:observed=0
$observe={$script:observed++;@{ExitCode=0;Stdout='observed';Stderr=''}}
$r=Invoke-Usb4StartupOperations {throw 'launch refused'} $observe $publish
if ($observed-ne1-or$r.ExitCode-ne1-or$r.controller.Stderr-ne'launch refused'){throw 'Controller exception skipped observation.'}
$r=Invoke-Usb4StartupOperations $ok $observe {param($Attempt);throw 'receipt write failed'}
if ($observed-ne2-or$r.ExitCode-ne1-or$r.observer.Stdout-ne'observed'-or$r.publication_error-ne'receipt write failed'){throw 'Publication failure skipped or masked observation.'}
$r=Invoke-Usb4StartupOperations $ok {throw 'observer launch failed'} $publish
if ($r.ExitCode-ne1-or$r.observer.Stderr-ne'observer launch failed'){throw 'Observer exception was masked.'}
Write-Output 'PASS: ControllerAndPublicationErrorsStillObserve'
# Selector: InvalidComponentStatusCannotPass. Protects truthful scheduler status;
# detects null-to-zero coercion, missing fields, strings, fractional/out-of-range
# statuses and multiple process outputs. Needs malformed outcome fixtures only.
# Breadcrumb: Assert-Usb4ProcessOutcome on both independent operations.
$invalid=@(@{ExitCode=$null;Stdout='';Stderr=''},@{ExitCode='0';Stdout='';Stderr=''},
    @{ExitCode=0.5;Stdout='';Stderr=''},@{ExitCode=[long]4294967296;Stdout='';Stderr=''},
    @{Stdout='';Stderr=''},@{ExitCode=0},@(@{ExitCode=0;Stdout='';Stderr=''},@{ExitCode=0;Stdout='';Stderr=''}))
foreach($outcome in $invalid) {
    $script:invalidOutcome=$outcome;$before=$script:observed
    $r=Invoke-Usb4StartupOperations {$script:invalidOutcome} $observe $publish
    if ($r.ExitCode-ne1-or$observed-ne($before+1)-or$r.controller.State-ne'ControllerException'){throw 'Invalid controller outcome passed or skipped observer.'}
    $r=Invoke-Usb4StartupOperations $ok {$script:invalidOutcome} $publish
    if ($r.ExitCode-ne1-or$r.observer.State-ne'ObserverException'){throw 'Invalid observer outcome passed.'}
}
$r=Invoke-Usb4StartupOperations {$null} $observe $publish
if ($r.ExitCode-ne1){throw 'Null controller outcome passed.'}
$r=Invoke-Usb4StartupOperations $ok {$null} $publish
if ($r.ExitCode-ne1){throw 'Null observer outcome passed.'}
Write-Output 'PASS: InvalidComponentStatusCannotPass'
