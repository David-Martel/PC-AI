function Measure-MemoryUsage {
    <#
    .SYNOPSIS
        Measures memory usage during inference
    #>
    [CmdletBinding()]
    param(
        [string]$Prompt = "Test memory usage with a moderate length prompt for measurement purposes.",
        [int]$MaxTokens = 256
    )

    [System.GC]::Collect()
    [System.GC]::WaitForPendingFinalizers()

    $before = [System.GC]::GetTotalMemory($true)

    try {
        $null = Invoke-PcaiGenerate -Prompt $Prompt -MaxTokens $MaxTokens
    } catch {
        Write-Warning "Inference failed: $_"
    }

    $after = [System.GC]::GetTotalMemory($false)

    return @{
        BeforeMB = [math]::Round($before / 1MB, 2)
        AfterMB = [math]::Round($after / 1MB, 2)
        DeltaMB = [math]::Round(($after - $before) / 1MB, 2)
    }
}
