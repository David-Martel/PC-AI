#Requires -Version 7.0
<#
.SYNOPSIS
    Returns a unified hardware report combining WMI and native diagnostics

.DESCRIPTION
    Aggregates high-fidelity native diagnostics (USB, Network, Process)
    with standard WMI configuration data to create a comprehensive
    system profile. This report is optimized for LLM "Search Pin" ingestion.

.EXAMPLE
    Get-UnifiedHardwareReportJson
    Returns the full structured report as a PSCustomObject

.OUTPUTS
    PSCustomObject
#>
function Get-UnifiedHardwareReportJson {
    [CmdletBinding()]
    [OutputType([PSCustomObject])]
    param(
        [Parameter()]
        [ValidateSet("Basic", "Normal", "Full")]
        [string]$Verbosity = "Normal"
    )

    # Ensure native DLLs are loaded
    if (-not (Test-PcaiNativeAvailable)) {
        Write-Warning "PCAI Native not available. Falling back to basic WMI report."
        return [PSCustomObject]@{
            Error = "Native diagnostics unavailable"
            Timestamp = (Get-Date).ToString("o")
        }
    }

    try {
        $json = Invoke-PcaiNativeUnifiedHardwareReport -Verbosity $Verbosity
        if ($json) {
            $report = ($json | ConvertFrom-Json)

            # Add native-calculated token estimate for chunking control
            $tokenCount = Invoke-PcaiNativeEstimateTokens -Text $json
            Add-Member -InputObject $report -MemberType NoteProperty -Name "TokenEstimate" -Value $tokenCount

            return $report
        }
    }
    catch {
        Write-Error "Failed to retrieve unified hardware report: $_"
    }

    return $null
}
