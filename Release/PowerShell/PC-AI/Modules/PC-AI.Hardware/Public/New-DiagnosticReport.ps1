#Requires -Version 5.1
<#
.SYNOPSIS
    Generates a comprehensive hardware diagnostic report

.DESCRIPTION
    Combines all hardware diagnostics into a single report file.
    This is the main entry point for full hardware analysis.

.PARAMETER OutputPath
    Path for the output report file

.PARAMETER Format
    Output format: 'txt' (default), 'json', or 'object'

.PARAMETER IncludeRecommendations
    Include basic recommendations based on findings

.EXAMPLE
    New-DiagnosticReport
    Creates a report on the Desktop

.EXAMPLE
    New-DiagnosticReport -OutputPath "C:\Reports\diagnostic.txt" -IncludeRecommendations
    Creates a detailed report with recommendations

.OUTPUTS
    String (path to report) or PSCustomObject[] if -Format 'object'
#>
function New-DiagnosticReport {
    [CmdletBinding()]
    param(
        [Parameter()]
        [string]$OutputPath = (Join-Path $env:USERPROFILE "Desktop\Hardware-Diagnostics-Report.txt"),

        [Parameter()]
        [ValidateSet('txt', 'json', 'object')]
        [string]$Format = 'txt',

        [Parameter()]
        [switch]$IncludeRecommendations
    )

    $reportData = [ordered]@{
        Timestamp      = Get-Date
        DeviceErrors   = @()
        DiskHealth     = @()
        SystemEvents   = @()
        UsbStatus      = @()
        NetworkAdapters = @()
        Summary        = @{
            CriticalCount = 0
            WarningCount  = 0
            ErrorCount    = 0
        }
    }

    Write-Verbose "Collecting device errors..."
    try {
        $reportData.DeviceErrors = @(Get-DeviceErrors -ErrorAction Stop)
    }
    catch {
        Write-Warning "Failed to collect device errors: $_"
        $reportData.DeviceErrors = @()
    }

    Write-Verbose "Collecting disk health..."
    try {
        $reportData.DiskHealth = @(Get-DiskHealth -ErrorAction Stop)
    }
    catch {
        Write-Warning "Failed to collect disk health: $_"
        $reportData.DiskHealth = @()
    }

    Write-Verbose "Collecting system events..."
    try {
        $reportData.SystemEvents = @(Get-SystemEvents -ErrorAction Stop)
    }
    catch {
        Write-Warning "Failed to collect system events: $_"
        $reportData.SystemEvents = @()
    }

    Write-Verbose "Collecting USB status..."
    try {
        $reportData.UsbStatus = @(Get-UsbStatus -ErrorAction Stop)
    }
    catch {
        Write-Warning "Failed to collect USB status: $_"
        $reportData.UsbStatus = @()
    }

    Write-Verbose "Collecting network adapters..."
    try {
        $reportData.NetworkAdapters = @(Get-NetworkAdapters -ErrorAction Stop)
    }
    catch {
        Write-Warning "Failed to collect network adapters: $_"
        $reportData.NetworkAdapters = @()
    }

    # Calculate summary counts
    $allItems = @(
        $reportData.DeviceErrors
        $reportData.DiskHealth
        $reportData.SystemEvents
        $reportData.UsbStatus
    )

    foreach ($item in $allItems) {
        if ($item.Severity -eq 'Critical') { $reportData.Summary.CriticalCount++ }
        if ($item.Severity -eq 'Warning') { $reportData.Summary.WarningCount++ }
        if ($item.Severity -eq 'Error') { $reportData.Summary.ErrorCount++ }
    }

    # Return based on format
    if ($Format -eq 'object') {
        return [PSCustomObject]$reportData
    }

    if ($Format -eq 'json') {
        $json = $reportData | ConvertTo-Json -Depth 10
        [System.IO.File]::WriteAllText($OutputPath, $json, [System.Text.Encoding]::UTF8)
        Write-Host "JSON report created: $OutputPath" -ForegroundColor Green
        return $OutputPath
    }

    # Default: txt format
    $report = @()
    $report += "=== Hardware & Device Diagnostics Report ($($reportData.Timestamp)) ==="
    $report += ""

    # Summary
    $report += "== Summary =="
    $report += "Critical Issues: $($reportData.Summary.CriticalCount)"
    $report += "Warnings: $($reportData.Summary.WarningCount)"
    $report += "Errors: $($reportData.Summary.ErrorCount)"
    $report += ""

    # Section 1: Device Errors
    $report += ConvertTo-ReportSection -Title "1. Devices with Errors in Device Manager" `
        -Data $reportData.DeviceErrors `
        -EmptyMessage "No devices reporting ConfigManagerErrorCode <> 0 (no obvious Device Manager errors)."

    # Section 2: Disk Health
    $report += ConvertTo-ReportSection -Title "2. Disk SMART Overall Status" `
        -Data ($reportData.DiskHealth | Select-Object Model, Status, SizeGB, InterfaceType, Severity) `
        -EmptyMessage "No disk information available."

    # Section 3: System Events
    $report += ConvertTo-ReportSection -Title "3. Recent System Errors/Warnings (disk / storage / USB) - last 3 days" `
        -Data ($reportData.SystemEvents | Select-Object TimeCreated, ProviderName, Id, Level, Message) `
        -EmptyMessage "No disk/USB-related critical/error/warning events found in the last 3 days."

    # Section 4: USB Status
    $report += ConvertTo-ReportSection -Title "4. USB Controllers and USB Devices Status" `
        -Data ($reportData.UsbStatus | Select-Object Name, PNPClass, Status, ErrorCode, Severity) `
        -EmptyMessage "No USB devices found."

    # Section 5: Network Adapters
    $report += ConvertTo-ReportSection -Title "5. Physical Network Adapters Status" `
        -Data ($reportData.NetworkAdapters | Select-Object Name, NetEnabled, Status, MACAddress, SpeedMbps) `
        -EmptyMessage "No physical network adapters found."

    # Section 6: Hints
    $report += "== 6. How to Read This Report (Quick Hints) =="
    $report += "- Section 1: Any devices with non-zero ConfigManagerErrorCode may have driver/hardware issues."
    $report += "- Section 2: Disk status should ideally show 'OK' for all drives."
    $report += "- Section 3: Repeated disk/USB errors can indicate unstable hardware, cabling, or failing drives."
    $report += "- Section 4: USB devices with non-zero ConfigManagerErrorCode or non-OK status are suspect."
    $report += "- Section 5: Network adapters that are not NetEnabled but should be, or show unusual Status, may be misconfigured or failing."
    $report += ""

    if ($IncludeRecommendations -and ($reportData.Summary.CriticalCount -gt 0 -or $reportData.Summary.WarningCount -gt 0)) {
        $report += "== 7. Recommendations =="

        # Critical disk issues
        $criticalDisks = $reportData.DiskHealth | Where-Object { $_.Severity -eq 'Critical' }
        if ($criticalDisks) {
            $report += "CRITICAL: One or more disks may be failing!"
            $report += "  - Back up important data immediately"
            $report += "  - Do not perform heavy disk operations"
            $report += "  - Plan for disk replacement"
            $report += ""
        }

        # Device errors
        if ($reportData.DeviceErrors.Count -gt 0) {
            $report += "DEVICE ISSUES:"
            $report += "  - Update drivers from device/motherboard manufacturer"
            $report += "  - Check Device Manager for disabled devices"
            $report += "  - Try uninstalling and reinstalling problematic drivers"
            $report += ""
        }

        # USB issues
        $usbErrors = $reportData.UsbStatus | Where-Object { $_.ErrorCode -ne 0 }
        if ($usbErrors) {
            $report += "USB ISSUES:"
            $report += "  - Try different USB ports"
            $report += "  - Test with different cables"
            $report += "  - Update USB/chipset drivers"
            $report += "  - Check USB power management settings"
            $report += ""
        }
    }

    $report += "=== End of Report ==="

    $reportText = $report -join "`r`n"
    [System.IO.File]::WriteAllText($OutputPath, $reportText, [System.Text.Encoding]::UTF8)

    Write-Host ""
    Write-Host "Hardware diagnostics report created:" -ForegroundColor Cyan
    Write-Host "  $OutputPath" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "You can share the relevant sections or the whole file for interpretation." -ForegroundColor Cyan

    return $OutputPath
}
