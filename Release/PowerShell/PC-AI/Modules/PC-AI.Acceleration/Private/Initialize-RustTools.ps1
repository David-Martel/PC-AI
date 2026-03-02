#Requires -Version 5.1
<#
.SYNOPSIS
    Initializes and caches Rust tool paths
#>

function Initialize-RustTool {
    [CmdletBinding()]
    param()

    $tools = @('rg', 'fd', 'bat', 'procs', 'tokei', 'sd', 'eza', 'hyperfine', 'dust', 'btm')

    foreach ($tool in $tools) {
        $script:ToolPaths[$tool] = Find-RustTool -ToolName $tool
    }
}

function Find-RustTool {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$ToolName
    )

    # Check cache first
    if ($script:RustToolCache.ContainsKey($ToolName)) {
        return $script:RustToolCache[$ToolName]
    }

    # Check common locations
    $searchPaths = @(
        "$env:USERPROFILE\.cargo\bin"
        "$env:USERPROFILE\bin"
        "$env:USERPROFILE\.local\bin"
        "$env:LOCALAPPDATA\Microsoft\WinGet\Links"
        'C:\Program Files\ripgrep'
        'C:\Program Files\fd'
    )

    foreach ($searchPath in $searchPaths) {
        $exePath = Join-Path $searchPath "$ToolName.exe"
        if (Test-Path $exePath) {
            $script:RustToolCache[$ToolName] = $exePath
            return $exePath
        }
    }

    # Try where.exe as fallback
    try {
        $result = & where.exe $ToolName 2>$null | Select-Object -First 1
        if ($result -and (Test-Path $result)) {
            $script:RustToolCache[$ToolName] = $result
            return $result
        }
    } catch {
        Write-Verbose "where.exe failed to find $ToolName : $_"
    }

    return $null
}

function Get-RustToolPath {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$ToolName
    )

    if ($script:ToolPaths.ContainsKey($ToolName) -and $script:ToolPaths[$ToolName]) {
        return $script:ToolPaths[$ToolName]
    }

    return Find-RustTool -ToolName $ToolName
}

function Test-RustToolInternal {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$ToolName
    )

    $path = Get-RustToolPath -ToolName $ToolName
    return ($null -ne $path -and (Test-Path $path))
}
