function Get-PcaiProcessCategories {
    <#
    .SYNOPSIS
        Classifies running processes into LLM-workload-relevant categories.
    .DESCRIPTION
        Groups processes into: LLM/AI agents, browsers, terminals, build tools,
        and system services. Returns per-category totals for RAM optimization analysis.
    .OUTPUTS
        PSCustomObject with category breakdown including process count, working set,
        private memory, and handle count per category.
    .EXAMPLE
        Get-PcaiProcessCategories
    .EXAMPLE
        Get-PcaiProcessCategories -AsJson
    #>
    [CmdletBinding()]
    param(
        [switch]$AsJson
    )

    Import-Module PC-AI.Common -ErrorAction SilentlyContinue
    $nativeAvailable = $false
    try { $nativeAvailable = Initialize-PcaiNative } catch {}

    if ($nativeAvailable) {
        $json = [PcaiNative.OptimizerModule]::GetProcessCategoriesJson()
        if ($json) {
            if ($AsJson) { return $json }
            return $json | ConvertFrom-Json
        }
    }

    # Fallback: PowerShell-based classification
    Write-Verbose 'Native DLL unavailable, using PowerShell fallback'

    $categoryPatterns = @{
        'LLM_Agents'      = @('claude', 'codex', 'ollama', 'copilot', 'pcai', 'llama', 'mistral')
        'Browsers'        = @('chrome', 'brave', 'msedge', 'firefox')
        'Terminals'       = @('conhost', 'cmd', 'wezterm', 'WindowsTerminal', 'wt')
        'Shells'          = @('powershell', 'pwsh')
        'Node_Electron'   = @('node', 'electron', 'bun', 'deno')
        'IDEs'            = @('Code', 'cursor', 'devenv')
        'Build_Tools'     = @('rust-analyzer', 'cargo', 'dotnet', 'msbuild', 'cl')
        'Python'          = @('python', 'python3')
        'WSL'             = @('vmmemWSL', 'vmmem', 'wsl', 'wslhost')
        'System_Services' = @('svchost', 'MsMpEng', 'lsass', 'csrss', 'dwm', 'explorer')
    }

    $allProcs = Get-Process
    $result = @{}
    $categorized = @{}

    foreach ($cat in $categoryPatterns.Keys) {
        $matched = @()
        foreach ($proc in $allProcs) {
            if ($categorized.ContainsKey($proc.Id)) { continue }
            foreach ($pattern in $categoryPatterns[$cat]) {
                if ($proc.ProcessName -like "*$pattern*") {
                    $matched += $proc
                    $categorized[$proc.Id] = $cat
                    break
                }
            }
        }

        $result[$cat] = [PSCustomObject]@{
            Category      = $cat
            ProcessCount  = $matched.Count
            WorkingSetMB  = [math]::Round(($matched | Measure-Object -Property WorkingSet64 -Sum).Sum / 1MB, 0)
            PrivateMB     = [math]::Round(($matched | Measure-Object -Property PrivateMemorySize64 -Sum).Sum / 1MB, 0)
            HandleCount   = ($matched | Measure-Object -Property HandleCount -Sum).Sum
            TopProcess    = if ($matched.Count -gt 0) {
                ($matched | Sort-Object PrivateMemorySize64 -Descending | Select-Object -First 1).ProcessName
            } else { '' }
        }
    }

    # Uncategorized
    $uncat = $allProcs | Where-Object { -not $categorized.ContainsKey($_.Id) }
    $result['Other'] = [PSCustomObject]@{
        Category      = 'Other'
        ProcessCount  = $uncat.Count
        WorkingSetMB  = [math]::Round(($uncat | Measure-Object -Property WorkingSet64 -Sum).Sum / 1MB, 0)
        PrivateMB     = [math]::Round(($uncat | Measure-Object -Property PrivateMemorySize64 -Sum).Sum / 1MB, 0)
        HandleCount   = ($uncat | Measure-Object -Property HandleCount -Sum).Sum
        TopProcess    = if ($uncat.Count -gt 0) {
            ($uncat | Sort-Object PrivateMemorySize64 -Descending | Select-Object -First 1).ProcessName
        } else { '' }
    }

    $output = $result.Values | Sort-Object PrivateMB -Descending

    if ($AsJson) {
        return $output | ConvertTo-Json -Depth 3
    }
    return $output
}
