#Requires -Version 5.1
function Optimize-Disks {
    <#
    .SYNOPSIS
        Optimizes disks using TRIM for SSDs or defragmentation for HDDs.

    .DESCRIPTION
        Automatically detects drive types and applies the appropriate optimization:
        - SSDs: Runs TRIM/Retrim operation
        - HDDs: Runs defragmentation

        Requires administrator privileges to execute.

    .PARAMETER DriveLetter
        Optional. One or more drive letters to optimize (e.g., 'C', 'D').
        If not specified, all fixed drives are optimized.

    .PARAMETER Force
        Bypasses confirmation prompts and optimizes all specified drives.

    .PARAMETER Priority
        Sets the optimization priority. Lower priority reduces system impact.
        Valid values: Low, Normal. Default is Normal.

    .PARAMETER AnalyzeOnly
        Only analyzes drives without performing optimization.
        Shows current fragmentation level and recommendations.

    .PARAMETER WhatIf
        Shows what would happen if the cmdlet runs without actually performing the operation.

    .PARAMETER Confirm
        Prompts for confirmation before running the cmdlet.

    .EXAMPLE
        Optimize-Disks
        Optimizes all fixed drives, prompting for confirmation.

    .EXAMPLE
        Optimize-Disks -DriveLetter C -Force
        Optimizes drive C: without confirmation prompt.

    .EXAMPLE
        Optimize-Disks -AnalyzeOnly
        Analyzes all drives and shows fragmentation/optimization status.

    .EXAMPLE
        Optimize-Disks -WhatIf
        Shows what optimization would be performed without executing.

    .OUTPUTS
        PSCustomObject with properties:
        - DriveLetter: The drive letter
        - MediaType: SSD or HDD
        - Operation: TRIM or Defragment
        - Status: Success, Failed, or Skipped
        - Duration: Time taken for optimization
        - Message: Additional details

    .NOTES
        Author: PC_AI Project
        Version: 1.0.0

        IMPORTANT: Requires administrator privileges.
        SSDs should not be defragmented as it provides no benefit and can reduce lifespan.
        This function automatically detects drive type to prevent incorrect optimization.
    #>
    [CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = 'High')]
    [OutputType([PSCustomObject[]])]
    param(
        [Parameter(Position = 0, ValueFromPipeline = $true, ValueFromPipelineByPropertyName = $true)]
        [ValidatePattern('^[A-Za-z]$')]
        [string[]]$DriveLetter,

        [Parameter()]
        [switch]$Force,

        [Parameter()]
        [ValidateSet('Low', 'Normal')]
        [string]$Priority = 'Normal',

        [Parameter()]
        [switch]$AnalyzeOnly
    )

    begin {
        # Verify admin privileges
        if (-not (Test-IsAdmin)) {
            throw "This function requires administrator privileges. Please run PowerShell as Administrator."
        }

        Write-Verbose "Starting disk optimization (Priority: $Priority, AnalyzeOnly: $AnalyzeOnly)"
        $results = [System.Collections.ArrayList]::new()

        # Set ConfirmPreference based on Force parameter
        if ($Force) {
            $ConfirmPreference = 'None'
        }
    }

    process {
        try {
            # Get all fixed drives
            $drives = Get-CimInstance -ClassName Win32_LogicalDisk -Filter "DriveType = 3" -ErrorAction Stop

            # Filter by specified drive letters if provided
            if ($DriveLetter) {
                $driveLettersUpper = $DriveLetter | ForEach-Object { $_.ToUpper() }
                $drives = $drives | Where-Object {
                    ($_.DeviceID -replace ':', '') -in $driveLettersUpper
                }
            }

            foreach ($drive in $drives) {
                $letter = $drive.DeviceID -replace ':', ''
                $volumePath = "$letter`:"

                # Determine drive type
                $mediaType = Get-DriveMediaType -DriveLetter $letter

                Write-Verbose "Processing drive $volumePath (Type: $mediaType)"

                # Determine operation based on drive type
                $operation = switch ($mediaType) {
                    'SSD' { 'Retrim' }
                    'HDD' { 'Defrag' }
                    default { 'Analyze' }
                }

                # Get current optimization status
                $optimizeStatus = $null
                try {
                    $volume = Get-Volume -DriveLetter $letter -ErrorAction SilentlyContinue
                    if ($volume) {
                        $optimizeStatus = Get-CimInstance -Namespace 'Root\Microsoft\Windows\Defrag' `
                            -ClassName 'MSFT_Volume' `
                            -Filter "DriveLetter = '$letter'" -ErrorAction SilentlyContinue
                    }
                }
                catch {
                    Write-Verbose "Could not get optimization status for drive $letter"
                }

                # Analyze-only mode
                if ($AnalyzeOnly) {
                    $analysisResult = [PSCustomObject]@{
                        DriveLetter        = $letter
                        MediaType          = $mediaType
                        RecommendedAction  = $operation
                        LastOptimized      = if ($optimizeStatus) { $optimizeStatus.LastAnalysisTime } else { 'Unknown' }
                        FragmentationLevel = 'N/A'
                        Status             = 'Analyzed'
                        Message            = "Drive type: $mediaType. Recommended: $operation"
                    }

                    # Try to get fragmentation info for HDDs
                    if ($mediaType -eq 'HDD') {
                        try {
                            $defragAnalysis = Optimize-Volume -DriveLetter $letter -Analyze -ErrorAction SilentlyContinue
                            if ($defragAnalysis) {
                                $analysisResult.FragmentationLevel = "$($defragAnalysis.PercentFileFragmentation)%"
                            }
                        }
                        catch {
                            Write-Verbose "Could not analyze fragmentation for $letter"
                        }
                    }

                    $analysisResult.PSObject.TypeNames.Insert(0, 'PC-AI.Performance.DiskOptimization')
                    [void]$results.Add($analysisResult)
                    continue
                }

                # Build description for ShouldProcess
                $actionDescription = switch ($operation) {
                    'Retrim' { "Run TRIM optimization on SSD drive $volumePath" }
                    'Defrag' { "Defragment HDD drive $volumePath" }
                    default { "Analyze drive $volumePath" }
                }

                # Execute optimization
                if ($PSCmdlet.ShouldProcess($volumePath, $actionDescription)) {
                    $startTime = Get-Date
                    $optimizeResult = [PSCustomObject]@{
                        DriveLetter = $letter
                        MediaType   = $mediaType
                        Operation   = $operation
                        Status      = 'Running'
                        Duration    = $null
                        Message     = ''
                        StartTime   = $startTime
                        EndTime     = $null
                    }

                    try {
                        Write-Verbose "Starting $operation on drive $volumePath"

                        # Build Optimize-Volume parameters
                        $optimizeParams = @{
                            DriveLetter = $letter
                            ErrorAction = 'Stop'
                            Verbose     = $false
                        }

                        switch ($operation) {
                            'Retrim' {
                                $optimizeParams['ReTrim'] = $true
                            }
                            'Defrag' {
                                $optimizeParams['Defrag'] = $true
                            }
                            default {
                                $optimizeParams['Analyze'] = $true
                            }
                        }

                        # Execute optimization
                        Optimize-Volume @optimizeParams

                        $endTime = Get-Date
                        $duration = $endTime - $startTime

                        $optimizeResult.Status = 'Success'
                        $optimizeResult.Duration = $duration
                        $optimizeResult.EndTime = $endTime
                        $optimizeResult.Message = "$operation completed successfully in $($duration.TotalSeconds.ToString('F2')) seconds"

                        Write-Verbose "Completed $operation on $volumePath ($($duration.TotalSeconds) seconds)"
                    }
                    catch {
                        $endTime = Get-Date
                        $duration = $endTime - $startTime

                        $optimizeResult.Status = 'Failed'
                        $optimizeResult.Duration = $duration
                        $optimizeResult.EndTime = $endTime
                        $optimizeResult.Message = "Error: $($_.Exception.Message)"

                        Write-Warning "Failed to optimize drive $volumePath`: $_"
                    }

                    $optimizeResult.PSObject.TypeNames.Insert(0, 'PC-AI.Performance.DiskOptimization')
                    [void]$results.Add($optimizeResult)
                }
                else {
                    # Operation was skipped (WhatIf or user declined)
                    $skippedResult = [PSCustomObject]@{
                        DriveLetter = $letter
                        MediaType   = $mediaType
                        Operation   = $operation
                        Status      = 'Skipped'
                        Duration    = $null
                        Message     = 'Operation skipped by user'
                        StartTime   = $null
                        EndTime     = $null
                    }

                    $skippedResult.PSObject.TypeNames.Insert(0, 'PC-AI.Performance.DiskOptimization')
                    [void]$results.Add($skippedResult)
                }
            }
        }
        catch {
            Write-Error "Failed to process drives: $_"
        }
    }

    end {
        # Summary
        $successCount = ($results | Where-Object { $_.Status -eq 'Success' }).Count
        $failedCount = ($results | Where-Object { $_.Status -eq 'Failed' }).Count
        $skippedCount = ($results | Where-Object { $_.Status -eq 'Skipped' }).Count

        Write-Verbose "Disk optimization complete: $successCount succeeded, $failedCount failed, $skippedCount skipped"

        if ($failedCount -gt 0) {
            Write-Warning "$failedCount drive(s) failed optimization. Check the results for details."
        }

        return $results
    }
}
