#Requires -Version 7.0
# Definitions only: this file is lazy-loaded at the first external capture.
function Initialize-PcaiProcessCaptureRunner {
    [CmdletBinding()]
    param()
    if (-not $IsWindows) { throw 'External cache capture requires Windows owned-job containment; local cache remains available.' }
    $captureLoaderGate = [string]::Intern('Pcai.Common.ProcessCaptureV1.Loader')
    [Threading.Monitor]::Enter($captureLoaderGate)
    try {
        $captureSourcePath = Join-Path $PSScriptRoot 'PcaiProcessCapture.cs'
        $captureSourceHash = (Get-FileHash -LiteralPath $captureSourcePath -Algorithm SHA256 -ErrorAction Stop).Hash
        $captureSourceText = [IO.File]::ReadAllText($captureSourcePath, [Text.Encoding]::UTF8)
        $captureTextBytes = [Text.Encoding]::UTF8.GetBytes($captureSourceText)
        $captureHasher = [Security.Cryptography.SHA256]::Create()
        try { $captureTextHash = [BitConverter]::ToString($captureHasher.ComputeHash($captureTextBytes)).Replace('-', '') }
        finally { $captureHasher.Dispose() }
        if ($captureTextHash -ne $captureSourceHash) { throw 'Capture source changed or was not BOM-free UTF8.' }
        if (-not ('Pcai.Common.ProcessCaptureV1' -as [type])) { Add-Type -TypeDefinition $captureSourceText -ErrorAction Stop }
        if ([Pcai.Common.ProcessCaptureV1]::ContractVersion -ne 1) { throw 'A different capture contract is loaded; use a fresh process.' }
        [Pcai.Common.ProcessCaptureV1]::BindSource($captureSourceHash)
        if ([Pcai.Common.ProcessCaptureV1]::SourceBinding -ne $captureSourceHash) { throw 'Loaded capture source binding mismatch.' }
    }
    finally { [Threading.Monitor]::Exit($captureLoaderGate) }
}

function Get-PcaiProcessCaptureCustody {
    <# .SYNOPSIS
    Inspect original capture custody without initializing or starting a child.
    #>
    [CmdletBinding()]
    param()
    if (-not ('Pcai.Common.ProcessCaptureV1' -as [type])) { return [pscustomobject]@{ Active = $null; Pending = @() } }
    [pscustomobject]@{ Active = [Pcai.Common.ProcessCaptureV1]::ActiveCustody(); Pending = @([Pcai.Common.ProcessCaptureV1]::PendingCustody()) }
}

function Repair-PcaiProcessCaptureCustody {
    <# .SYNOPSIS
    Retry closure using only retained original job, process and reader objects.
    #>
    [CmdletBinding(SupportsShouldProcess)]
    param([Parameter(Mandatory)][string]$CustodyId)
    if (-not ('Pcai.Common.ProcessCaptureV1' -as [type])) { throw 'No capture runner is loaded.' }
    if ($PSCmdlet.ShouldProcess($CustodyId, 'Retry original capture custody for at most 1000ms of cooperative cleanup')) {
        [Pcai.Common.ProcessCaptureV1]::RetryOwnedCustody($CustodyId)
    }
}
