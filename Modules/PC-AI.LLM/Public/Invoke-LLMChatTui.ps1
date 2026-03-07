#Requires -PSEdition Core

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

    $artifactsRoot = Join-Path $repoRoot '.pcai'
    $userBinRoot = Join-Path ([Environment]::GetFolderPath('UserProfile')) 'bin'

    $candidates = @()
    $candidates += @(
        (Join-Path $artifactsRoot 'build\artifacts\pcai-chattui\PcaiChatTui.exe'),
        (Join-Path $repoRoot 'Native\PcaiChatTui\bin\Release\net8.0\win-x64\publish\PcaiChatTui.exe'),
        (Join-Path $repoRoot 'Native\PcaiChatTui\bin\Release\net8.0\win-x64\PcaiChatTui.exe'),
        (Join-Path $repoRoot 'Native\PcaiChatTui\bin\Debug\net8.0\win-x64\publish\PcaiChatTui.exe'),
        (Join-Path $repoRoot 'Native\PcaiChatTui\bin\Debug\net8.0\win-x64\PcaiChatTui.exe'),
        (Join-Path $userBinRoot 'PcaiChatTui.exe')
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
