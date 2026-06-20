param(
    [string]$OutputDirectory = (Join-Path $PSScriptRoot 'direct-pass')
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

function Save-Json {
    param(
        [Parameter(Mandatory)] [string] $Name,
        [Parameter(Mandatory)] [scriptblock] $ScriptBlock,
        [int] $Depth = 6
    )

    $path = Join-Path $OutputDirectory $Name
    try {
        $data = & $ScriptBlock
        $data | ConvertTo-Json -Depth $Depth | Set-Content -LiteralPath $path -Encoding UTF8
    } catch {
        [pscustomobject]@{ Error = $_.Exception.Message } |
            ConvertTo-Json -Depth 4 |
            Set-Content -LiteralPath $path -Encoding UTF8
    }
}

$since = (Get-Date).AddHours(-2)

Save-Json 'events-critical-error-warning-2h.json' {
    Get-WinEvent -FilterHashtable @{ LogName = @('System', 'Application'); Level = 1, 2, 3; StartTime = $since } -ErrorAction SilentlyContinue |
        Select-Object TimeCreated, ProviderName, Id, LevelDisplayName, Message
} 8

Save-Json 'events-kernel-driver-storage-24h.json' {
    $start = (Get-Date).AddHours(-24)
    Get-WinEvent -FilterHashtable @{ LogName = 'System'; StartTime = $start } -ErrorAction SilentlyContinue |
        Where-Object {
            $_.ProviderName -in @(
                'Microsoft-Windows-Kernel-PnP',
                'Microsoft-Windows-DriverFrameworks-UserMode',
                'Microsoft-Windows-Kernel-Power',
                'EventLog',
                'disk',
                'Microsoft-Windows-FilterManager',
                'Microsoft-Windows-WHEA-Logger'
            )
        } |
        Select-Object TimeCreated, ProviderName, Id, LevelDisplayName, Message
} 8

Save-Text 'pnputil-problems.txt' { pnputil.exe /enum-devices /problem }
Save-Text 'pnputil-display-devices.txt' { pnputil.exe /enum-devices /class Display }
Save-Text 'pnputil-display-drivers.txt' { pnputil.exe /enum-drivers /class Display }
Save-Json 'pnp-display-devices.json' {
    Get-PnpDevice -Class Display -ErrorAction SilentlyContinue |
        Select-Object Status, Class, FriendlyName, InstanceId, Problem, ConfigManagerErrorCode
} 6

Save-Json 'onedrive-scheduled-tasks.json' {
    Get-ScheduledTask -ErrorAction SilentlyContinue |
        Where-Object { $_.TaskName -like '*OneDrive*' -or $_.TaskPath -like '*OneDrive*' } |
        ForEach-Object {
            $info = $_ | Get-ScheduledTaskInfo -ErrorAction SilentlyContinue
            [pscustomobject]@{
                TaskPath = $_.TaskPath
                TaskName = $_.TaskName
                State = $_.State
                Author = $_.Author
                UserId = $_.Principal.UserId
                RunLevel = $_.Principal.RunLevel
                LastRunTime = $info.LastRunTime
                LastTaskResult = $info.LastTaskResult
                NextRunTime = $info.NextRunTime
                Actions = ($_.Actions | ForEach-Object {
                    [pscustomobject]@{
                        Execute = $_.Execute
                        Arguments = $_.Arguments
                        WorkingDirectory = $_.WorkingDirectory
                    }
                })
                Triggers = ($_.Triggers | ForEach-Object { $_.ToString() })
            }
        }
} 10

Save-Text 'services-pcai-vtss-hvsock.txt' {
    foreach ($name in 'PC_AI-HVSockProxy', 'vtss', 'PC_AI-VLLM', 'PC_AI-ToolRouter') {
        "--- $name ---"
        sc.exe qc $name
        sc.exe queryex $name
    }
}

Save-Json 'powershell-profile-files.json' {
    $profileRoots = @(
        "$HOME\Documents\PowerShell",
        "$HOME\Documents\WindowsPowerShell",
        "$HOME\.machine",
        "$HOME\.local",
        "$HOME\bin",
        "$HOME\.codex",
        "$HOME\.agents"
    )
    foreach ($root in $profileRoots) {
        if (Test-Path -LiteralPath $root) {
            Get-ChildItem -LiteralPath $root -Force -Recurse -File -ErrorAction SilentlyContinue |
                Where-Object {
                    $_.FullName -match '\\(profile|profiles|startup|scripts|tools|bin|Machine|SystemScripts)\\' -or
                    $_.Name -match '(profile|startup|login|init|hvsock|vtss|onedrive|docker|vllm|toolrouter|rag|redis|dns|machine)'
                } |
                Select-Object FullName, Length, LastWriteTime
        }
    }
} 6

Save-Text 'docker-system-df.txt' { docker system df }
Save-Text 'docker-system-df-verbose.txt' { docker system df -v }
Save-Text 'docker-ps-a.txt' { docker ps -a --no-trunc }
Save-Json 'docker-images.json' {
    docker image ls --digests --no-trunc --format '{{json .}}' |
        ForEach-Object { $_ | ConvertFrom-Json }
} 8
Save-Json 'docker-containers.json' {
    docker ps -a --no-trunc --format '{{json .}}' |
        ForEach-Object { $_ | ConvertFrom-Json }
} 8
Save-Text 'docker-builder-du.txt' { docker builder du }
Save-Text 'docker-buildx-du.txt' { docker buildx du }

Save-Json 'startup-relevant-tasks.json' {
    Get-ScheduledTask -ErrorAction SilentlyContinue |
        Where-Object {
            $_.TaskName -match 'PC_AI|VLLM|HVSock|vtss|RAG|Redis|DNS|Docker|OneDrive|iCloud|NVIDIA|Razer|Logi|Bonjour' -or
            $_.TaskPath -match 'PC_AI|VLLM|HVSock|vtss|RAG|Redis|DNS|Docker|OneDrive|iCloud|NVIDIA|Razer|Logi|Bonjour'
        } |
        ForEach-Object {
            $info = $_ | Get-ScheduledTaskInfo -ErrorAction SilentlyContinue
            [pscustomobject]@{
                TaskPath = $_.TaskPath
                TaskName = $_.TaskName
                State = $_.State
                UserId = $_.Principal.UserId
                LastRunTime = $info.LastRunTime
                LastTaskResult = $info.LastTaskResult
                NextRunTime = $info.NextRunTime
                Actions = $_.Actions
            }
        }
} 10

[pscustomobject]@{
    Machine = $env:COMPUTERNAME
    Timestamp = Get-Date
    OutputDirectory = (Resolve-Path -LiteralPath $OutputDirectory).Path
} | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $OutputDirectory 'manifest.json') -Encoding UTF8
