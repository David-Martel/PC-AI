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
        $(if (-not [string]::IsNullOrWhiteSpace($env:USERPROFILE)) { Join-Path $env:USERPROFILE 'PC_AI\bin' })
    ) | Where-Object { $_ }

    # Preserve host PATH semantics: colon on Unix, semicolon on Windows.
    $pathDirs = ($env:PATH -split [regex]::Escape([string][IO.Path]::PathSeparator)) | Where-Object { $_ -and (Test-Path -LiteralPath $_ -PathType Container) }

    $searchPaths = @($dynamicSearchPaths + $script:SearchPaths + $pathDirs + 'C:\Program Files\fd') | Select-Object -Unique

    foreach ($searchPath in $searchPaths) {
        if (-not (Test-Path -LiteralPath $searchPath -PathType Container -ErrorAction SilentlyContinue)) { continue }
        $fileName = if ([Environment]::OSVersion.Platform -eq [PlatformID]::Win32NT) { "$ToolName.exe" } else { $ToolName }
        $exePath = Join-Path $searchPath $fileName
        if (Test-Path -LiteralPath $exePath) {
            $script:RustToolCache[$ToolName] = $exePath
            if ($script:ToolPaths.ContainsKey($ToolName)) {
                $script:ToolPaths[$ToolName] = $exePath
            }
            return $exePath
        }
    }

    try {
        $application = Get-Command $ToolName -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
        $result = if ($application) { $application.Path } else { $null }
        if ($result -and (Test-Path -LiteralPath $result)) {
            $script:RustToolCache[$ToolName] = $result
            if ($script:ToolPaths.ContainsKey($ToolName)) {
                $script:ToolPaths[$ToolName] = $result
            }
            return $result
        }
    } catch {
        Write-Verbose "Application lookup failed to find $ToolName : $_"
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
