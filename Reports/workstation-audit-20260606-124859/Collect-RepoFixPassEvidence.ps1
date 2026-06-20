param(
    [string]$OutputDirectory = (Join-Path $PSScriptRoot 'repo-fix-pass-20260606')
)

$ErrorActionPreference = 'Continue'
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null

function Save-Text {
    param(
        [Parameter(Mandatory)] [string] $Name,
        [Parameter(Mandatory)] [scriptblock] $ScriptBlock
    )

    $path = Join-Path $OutputDirectory $Name
    try {
        & $ScriptBlock *> $path
    } catch {
        $_ | Out-String | Set-Content -LiteralPath $path -Encoding UTF8
    }
}

Save-Text 'cargo-tools-build-environment.txt' {
    Import-Module CargoTools -Force
    Test-BuildEnvironment -Detailed
}

Save-Text 'cargo-tools-commands.txt' {
    Import-Module CargoTools -Force
    Get-Command -Module CargoTools | Sort-Object Name | Format-Table Name, CommandType, Source -AutoSize
    ''
    Get-Command Invoke-CargoWrapper -All | Format-List *
}

Save-Text 'cargo-tools-module-files.txt' {
    Get-ChildItem -LiteralPath "$HOME\Documents\PowerShell\Modules\CargoTools" -Recurse -File -ErrorAction SilentlyContinue |
        Select-Object FullName, Length, LastWriteTime |
        Format-Table -AutoSize
}

Save-Text 'winget-nvidia-upgrades.txt' {
    winget upgrade --name NVIDIA
}

Save-Text 'winget-lenovo-upgrades.txt' {
    winget upgrade --name Lenovo
}

Save-Text 'nvidia-app-processes.txt' {
    Get-Process -ErrorAction SilentlyContinue |
        Where-Object { $_.ProcessName -match 'NVIDIA|Nv|FrameView|Broadcast|Nsight' } |
        Select-Object ProcessName, Id, Path, StartTime |
        Format-Table -AutoSize
}

Save-Text 'elan-input-devices.txt' {
    Get-PnpDevice -ErrorAction SilentlyContinue |
        Where-Object {
            $_.FriendlyName -match 'ELAN|Synaptics|TouchPad|Touchpad|Keyboard|HID|I2C|Fingerprint|Windows Hello'
        } |
        Select-Object Status, Class, FriendlyName, InstanceId, Problem, ConfigManagerErrorCode |
        Format-Table -AutoSize
}

Save-Text 'recent-input-events.txt' {
    $start = (Get-Date).AddHours(-24)
    Get-WinEvent -FilterHashtable @{ LogName = 'System'; StartTime = $start } -ErrorAction SilentlyContinue |
        Where-Object {
            $_.Message -match 'ELAN|Synaptics|HID|I2C|Keyboard|WUDFRd|WindowsHello|Fingerprint|Dynamic Tuning'
        } |
        Select-Object TimeCreated, ProviderName, Id, LevelDisplayName, Message |
        Format-List
}

Save-Text 'functiongemma-files.txt' {
    Get-ChildItem -LiteralPath (Join-Path (Resolve-Path "$PSScriptRoot\..\..").Path 'Deploy') -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -match 'functiongemma|vllm|router|tool' } |
        Select-Object FullName, Length, LastWriteTime |
        Format-Table -AutoSize
}

[pscustomobject]@{
    Machine = $env:COMPUTERNAME
    Timestamp = Get-Date
    OutputDirectory = (Resolve-Path -LiteralPath $OutputDirectory).Path
} | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $OutputDirectory 'manifest.json') -Encoding UTF8
