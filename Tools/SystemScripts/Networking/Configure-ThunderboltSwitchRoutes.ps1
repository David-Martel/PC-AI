#Requires -Version 7.0
<#
.SYNOPSIS
Observe Windows USB4 startup state without trusting discovery hints for routing.
.DESCRIPTION
Runs bounded, read-only native status and V&V commands. Keeps the original native
receipt separate from the boot outcome. Absence/down is unavailable (exit 2),
technical/collection failure is exit 1, pending acceptance is exit 3 and unsupported
platform is exit 4. Exit 0 means configured technical checks only; NetworkReady is
always false because this observer does not authenticate or qualify a data edge.
No addresses, routes, metrics, peer registry or enrollment are changed. Banner and
NTLM discovery cannot authorize route installation. See docs/usb4/windows-boot.md.
.PARAMETER DryRun
Perform read-only observation and emit JSON without creating files or directories.
.EXAMPLE
./Configure-ThunderboltSwitchRoutes.ps1 -DryRun
#>
[CmdletBinding()]
param(
    [string]$TbExecutable = (Join-Path $HOME 'bin/pcai-tb.exe'),
    [string]$RegistryPath = (Join-Path $HOME '.machine/thunderbolt-peers.json'),
    [string]$EvidenceDirectory = (Join-Path $HOME '.machine/logs/usb4-boot'),
    [ValidateRange(1, 120)][int]$TimeoutSeconds = 20,
    [switch]$DryRun,
    [Alias('h', 'help')][switch]$ShowHelp
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Get-ThunderboltBootReadiness {
    <# .SYNOPSIS
    Classify one fresh native observation without granting data-path readiness.
    #>
    [CmdletBinding()]
    param([hashtable]$Snapshot, [hashtable]$Evidence, [int]$VnvExitCode)
    $state = 'Failed'; $code = 1
    $reason = 'Missing, malformed or mismatched native receipt.'
    if ($Snapshot.ContainsKey('OsFamily') -and $Snapshot['OsFamily'] -ne 'windows') {
        $state = 'Unsupported'; $code = 4; $reason = 'Windows startup policy requires a Windows inventory.'
    } elseif (-not $Snapshot.ContainsKey('Adapters') -or -not $Snapshot['ComputerName'] -or
        $Snapshot['ComputerName'] -eq 'unknown-host') {
        $reason = 'Native inventory is incomplete.'
    } elseif (@($Snapshot.Adapters).Count -eq 0) {
        $state = 'Absent'; $code = 2; $reason = 'No present USB4 peer adapter; native V&V verdict retained separately.'
    } elseif (@($Snapshot.Adapters | Where-Object { $_.OperationalStatus -eq 'Up' }).Count -eq 0) {
        $state = 'Degraded'; $code = 2; $reason = 'Present USB4 peer adapters have no operational link.'
    } elseif ($Evidence['schema'] -eq 'vigil.hil-device-evidence/v1' -and
        $Evidence['host'] -eq $Snapshot.ComputerName -and $Evidence.ContainsKey('requirement_gates') -and
        @($Evidence.requirement_gates).Count -gt 0) {
        $verdicts = @($Evidence.requirement_gates | ForEach-Object { $_['technicalVerdict'] })
        if (@($verdicts | Where-Object { $_ -notin @('PASS', 'SKIP_NOT_APPLICABLE') }).Count -gt 0 -or
            @($verdicts | Where-Object { $_ -eq 'PASS' }).Count -eq 0) {
            $reason = 'Native technical checks failed or are incomplete.'
        } elseif ($Evidence['custody_valid'] -eq $false -and $Evidence['overall_status'] -match 'PENDING|HOLD') {
            $state = 'PendingAcceptance'; $code = 3; $reason = 'Technical checks passed; acceptance remains pending.'
        } elseif ($VnvExitCode -eq 0 -and $Evidence['custody_valid'] -eq $true -and $Evidence['overall_status'] -eq 'PASS') {
            $state = 'Configured'; $code = 0; $reason = 'Technical checks passed; authenticated data-edge qualification is separate.'
        } else { $reason = 'Native exit and receipt do not establish technical success.' }
    }
    [pscustomobject]@{ State = $state; ExitCode = $code; NetworkReady = $false; Reason = $reason; VnvExitCode = $VnvExitCode }
}

function Invoke-ThunderboltBootObservation {
    <# .SYNOPSIS
    Collect bounded native receipts and publish an atomic observation summary.
    #>
    [CmdletBinding()]
    param([string]$TbExecutable, [string]$RegistryPath, [string]$EvidenceDirectory,
        [ValidateRange(1, 120)][int]$TimeoutSeconds = 20, [switch]$DryRun)
    $observed = [DateTimeOffset]::UtcNow
    $status = $null; $vnv = $null
    try {
        $status = Invoke-VigilBoundedProcess -FilePath $TbExecutable -Arguments @('status', '--include-disconnected', '--compact') -TimeoutSeconds $TimeoutSeconds
        if ($status.ExitCode -ne 0) { throw "Native status failed with exit $($status.ExitCode): $($status.Stderr)" }
        $snapshot = $status.Stdout | ConvertFrom-Json -AsHashtable
        $vnv = Invoke-VigilBoundedProcess -FilePath $TbExecutable -Arguments @('vnv', '--registry', $RegistryPath, '--compact') -TimeoutSeconds $TimeoutSeconds
        $evidence = $vnv.Stdout | ConvertFrom-Json -AsHashtable
        $outcome = Get-ThunderboltBootReadiness -Snapshot $snapshot -Evidence $evidence -VnvExitCode $vnv.ExitCode
    } catch {
        $outcome = [pscustomobject]@{ State = 'Failed'; ExitCode = 1; NetworkReady = $false; Reason = $_.Exception.Message;
            VnvExitCode = $(if ($vnv) { $vnv.ExitCode } else { $null }) }
    }
    $receipt = [ordered]@{ schema = 'pcai.usb4-boot-observation/v1'; observed_at = $observed.ToString('o');
        expires_at = $observed.AddSeconds(90).ToString('o'); result = $outcome; status = $status; vnv = $vnv }
    if (-not $DryRun) {
        # Raw stdout/stderr and exit codes remain unmodified within this receipt.
        # Atomic replacement means consumers never read a truncated JSON document.
        $null = [IO.Directory]::CreateDirectory($EvidenceDirectory)
        $destination = Join-Path $EvidenceDirectory 'current.json'
        $staging = Join-Path $EvidenceDirectory ('pending-' + [guid]::NewGuid().ToString('N') + '.json')
        try {
            [IO.File]::WriteAllText($staging, ($receipt | ConvertTo-Json -Depth 30), [Text.UTF8Encoding]::new($false))
            [IO.File]::Move($staging, $destination, $true)
        } finally {
            if ([IO.File]::Exists($staging)) { [IO.File]::Delete($staging) }
        }
    }
    return $outcome
}

if ($MyInvocation.InvocationName -eq '.') { return }
if ($ShowHelp) { Get-Help $PSCommandPath -Detailed; return }
if (-not $IsWindows) {
    [pscustomobject]@{ State = 'Unsupported'; ExitCode = 4; NetworkReady = $false } | ConvertTo-Json
    exit 4
}
. (Join-Path $PSScriptRoot 'Invoke-VigilBoundedProcess.ps1')
# Native identity currently reads COMPUTERNAME/HOSTNAME. Scheduler/harness shells
# can omit those variables; derive this child-only context from the OS, never DNS.
if ([string]::IsNullOrWhiteSpace($env:COMPUTERNAME)) { $env:COMPUTERNAME = [Environment]::MachineName }
$result = Invoke-ThunderboltBootObservation -TbExecutable $TbExecutable -RegistryPath $RegistryPath `
    -EvidenceDirectory $EvidenceDirectory -TimeoutSeconds $TimeoutSeconds -DryRun:$DryRun
$result | ConvertTo-Json -Depth 8
exit $result.ExitCode
