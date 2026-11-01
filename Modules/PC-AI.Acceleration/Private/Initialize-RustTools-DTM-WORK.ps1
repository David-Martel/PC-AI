#Requires -Version 5.1
<#
.SYNOPSIS
    Initializes and caches Rust tool paths
#>

function Initialize-RustTool {
    [CmdletBinding()]
    param()

    $tools = @('rg', 'fd', 'bat', 'procs', 'pcai-perf', 'tokei', 'sd', 'eza', 'hyperfine', 'dust', 'btm')

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

    if ($script:RustToolCache.ContainsKey($ToolName)) {
        return $script:RustToolCache[$ToolName]
    }

    $dynamicSearchPaths = @(
        $(if ($env:PCAI_ROOT) { Join-Path $env:PCAI_ROOT 'bin' })
        'C:\codedev\PC_AI\bin'
    ) | Where-Object { $_ }

    # Split $env:PATH by ';' to get individual directory candidates.
    # Never join 'bin\' with the raw PATH string — that produces invalid compound paths.
    $pathDirs = ($env:PATH -split ';') | Where-Object { $_ -and (Test-Path -LiteralPath $_ -PathType Container) }

    $searchPaths = @($dynamicSearchPaths + $script:SearchPaths + $pathDirs + 'C:\Program Files\fd') | Select-Object -Unique

    foreach ($searchPath in $searchPaths) {
        $exePath = Join-Path $searchPath "$ToolName.exe"
        if (Test-Path -LiteralPath $exePath) {
            $script:RustToolCache[$ToolName] = $exePath
            if ($script:ToolPaths.ContainsKey($ToolName)) {
                $script:ToolPaths[$ToolName] = $exePath
            }
            return $exePath
        }
    }

    try {
        $result = & where.exe $ToolName 2>$null | Select-Object -First 1
        if ($result -and (Test-Path -LiteralPath $result)) {
            $script:RustToolCache[$ToolName] = $result
            if ($script:ToolPaths.ContainsKey($ToolName)) {
                $script:ToolPaths[$ToolName] = $result
            }
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
    return ($null -ne $path -and (Test-Path -LiteralPath $path))
}
