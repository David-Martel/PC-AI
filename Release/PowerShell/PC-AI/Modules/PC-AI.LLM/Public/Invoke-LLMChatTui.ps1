function Invoke-LLMChatTui {
    <#
    .SYNOPSIS
        Launches the PC_AI chat TUI for LLM interaction.
    #>
    [CmdletBinding()]
    param(
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Arguments,
        [switch]$BuildIfMissing,
        [string]$ProjectRoot
    )

    $repoRoot = if ($ProjectRoot) {
        $ProjectRoot
    } else {
        Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
    }

    $artifactsRoot = if ($env:PCAI_ARTIFACTS_ROOT) {
        $env:PCAI_ARTIFACTS_ROOT
    } else {
        Join-Path $repoRoot '.pcai'
    }

    $candidates = @()
    if ($env:PCAI_TUI_EXE) {
        $candidates += $env:PCAI_TUI_EXE
    }

    $candidates += @(
        (Join-Path $artifactsRoot 'build\artifacts\pcai-chattui\PcaiChatTui.exe'),
        (Join-Path $repoRoot 'Native\PcaiChatTui\bin\Release\net8.0\win-x64\publish\PcaiChatTui.exe'),
        (Join-Path $repoRoot 'Native\PcaiChatTui\bin\Release\net8.0\win-x64\PcaiChatTui.exe'),
        (Join-Path $repoRoot 'Native\PcaiChatTui\bin\Debug\net8.0\win-x64\publish\PcaiChatTui.exe'),
        (Join-Path $repoRoot 'Native\PcaiChatTui\bin\Debug\net8.0\win-x64\PcaiChatTui.exe'),
        (Join-Path $env:USERPROFILE 'bin\PcaiChatTui.exe')
    )

    $exe = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $exe -and $BuildIfMissing) {
        $buildScript = Join-Path $repoRoot 'Build.ps1'
        if (Test-Path $buildScript) {
            & $buildScript -Component tui -Configuration Release
            $exe = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
        }
    }

    if (-not $exe) {
        throw "PcaiChatTui.exe not found. Build it with `.\Build.ps1 -Component tui` or rerun this command with -BuildIfMissing."
    }

    & $exe @Arguments
}
