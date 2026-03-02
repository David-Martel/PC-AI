#Requires -Version 5.1
<#
.SYNOPSIS
    Reports the status of available Rust acceleration tools

.DESCRIPTION
    Detects and reports which Rust CLI tools are available for
    performance acceleration. Shows version and path for each tool.

.PARAMETER Tool
    Specific tool to check. If not specified, checks all tools.

.EXAMPLE
    Get-RustToolStatus
    Lists all available Rust tools with versions

.EXAMPLE
    Get-RustToolStatus -Tool rg
    Checks only ripgrep availability

.OUTPUTS
    PSCustomObject[] with tool status information
#>
function Get-RustToolStatus {
    [CmdletBinding()]
    [OutputType([PSCustomObject[]])]
    param(
        [Parameter()]
        [ValidateSet('rg', 'fd', 'bat', 'procs', 'tokei', 'sd', 'eza', 'hyperfine', 'dust', 'btm', 'All')]
        [string]$Tool = 'All'
    )

    $toolInfo = @{
        rg        = @{ Name = 'ripgrep';    Use = 'Fast text search (grep replacement)';     VersionArg = '--version' }
        fd        = @{ Name = 'fd';         Use = 'Fast file finder (find replacement)';     VersionArg = '--version' }
        bat       = @{ Name = 'bat';        Use = 'Cat with syntax highlighting';            VersionArg = '--version' }
        procs     = @{ Name = 'procs';      Use = 'Modern process viewer (ps replacement)';  VersionArg = '--version' }
        tokei     = @{ Name = 'tokei';      Use = 'Code statistics';                         VersionArg = '--version' }
        sd        = @{ Name = 'sd';         Use = 'Fast find & replace (sed replacement)';   VersionArg = '--version' }
        eza       = @{ Name = 'eza';        Use = 'Modern ls replacement';                   VersionArg = '--version' }
        hyperfine = @{ Name = 'hyperfine';  Use = 'Command benchmarking';                    VersionArg = '--version' }
        dust      = @{ Name = 'dust';       Use = 'Disk usage analyzer (du replacement)';    VersionArg = '--version' }
        btm       = @{ Name = 'bottom';     Use = 'System monitor (htop replacement)';       VersionArg = '--version' }
    }

    $toolsToCheck = if ($Tool -eq 'All') { $toolInfo.Keys } else { @($Tool) }
    $results = @()

    foreach ($t in $toolsToCheck) {
        $info = $toolInfo[$t]
        $path = Get-RustToolPath -ToolName $t
        $available = $null -ne $path -and (Test-Path $path)
        $version = $null

        if ($available) {
            try {
                $versionOutput = & $path $info.VersionArg 2>&1 | Select-Object -First 1
                if ($versionOutput -match '[\d]+\.[\d]+\.[\d]+') {
                    $version = $Matches[0]
                }
                elseif ($versionOutput) {
                    $version = $versionOutput.ToString().Trim()
                }
            }
            catch {
                $version = 'Unknown'
            }
        }

        $results += [PSCustomObject]@{
            Tool        = $t
            Name        = $info.Name
            Available   = $available
            Version     = $version
            Path        = $path
            Use         = $info.Use
            Accelerates = Get-AcceleratedFunction -Tool $t
        }
    }

    return $results
}

function Get-AcceleratedFunction {
    [CmdletBinding()]
    param([string]$Tool)

    $mapping = @{
        rg        = @('Search-LogsFast', 'Search-ContentFast', 'Get-SystemEvents')
        fd        = @('Find-FilesFast', 'Find-DuplicatesFast')
        procs     = @('Get-ProcessesFast', 'Get-ProcessPerformance')
        dust      = @('Get-DiskUsageFast', 'Get-DiskSpace')
        tokei     = @('Get-CodeStatistics')
        hyperfine = @('Measure-CommandPerformance', 'Compare-ToolPerformance')
        bat       = @('Show-FileWithHighlighting')
        eza       = @('Get-DirectoryListingFast')
    }

    if ($mapping.ContainsKey($Tool)) {
        return $mapping[$Tool] -join ', '
    }
    return 'N/A'
}

function Test-RustToolAvailable {
    <#
    .SYNOPSIS
        Tests if a specific Rust tool is available

    .PARAMETER Tool
        The tool to check

    .EXAMPLE
        if (Test-RustToolAvailable -Tool rg) { ... }
    #>
    [CmdletBinding()]
    [OutputType([bool])]
    param(
        [Parameter(Mandatory)]
        [string]$Tool
    )

    return Test-RustToolInternal -ToolName $Tool
}
