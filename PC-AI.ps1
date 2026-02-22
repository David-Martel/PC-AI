#Requires -Version 5.1

<#
.SYNOPSIS
    PC-AI - Local LLM-Powered PC Diagnostics Framework

.DESCRIPTION
    Unified CLI for PC diagnostics, optimization, USB management, and LLM-powered analysis.
    Provides a comprehensive interface to all PC-AI modules including:
    - Hardware diagnostics (devices, disks, USB, network adapters)
    - Virtualization management (WSL2, Hyper-V, Docker)
    - USB/WSL passthrough management
    - Network diagnostics and VSock optimization
    - Performance monitoring and optimization
    - System cleanup (PATH, temp files, duplicates)
    - LLM-powered analysis via pcai-inference

.PARAMETER Command
    Main command: diagnose, optimize, usb, analyze, chat, llm, cleanup, perf, doctor, status, version, help

.PARAMETER Arguments
    Additional arguments for the command

.EXAMPLE
    .\PC-AI.ps1 diagnose all
    Run full system diagnostics

.EXAMPLE
    .\PC-AI.ps1 analyze --report "report.txt"
    Analyze diagnostic report with LLM

.EXAMPLE
    .\PC-AI.ps1 usb list
    List all USB devices

.EXAMPLE
    .\PC-AI.ps1 optimize wsl --dry-run
    Preview WSL optimization changes

.NOTES
    Author: PC_AI Framework
    Version: 1.0.0
    Requires: Windows 10/11 with PowerShell 5.1+
#>

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('diagnose', 'optimize', 'usb', 'analyze', 'chat', 'llm', 'cleanup', 'perf', 'doctor', 'status', 'version', 'help')]
    [string]$Command,

    [Parameter(Position = 1, ValueFromRemainingArguments)]
    [string[]]$Arguments,

    # Inference backend selection
    [Parameter()]
    [ValidateSet('auto', 'llamacpp', 'mistralrs', 'http')]
    [string]$InferenceBackend = 'auto',

    # Model path for native inference
    [Parameter()]
    [string]$ModelPath,

    # GPU layers for native inference (-1 = all, 0 = CPU only)
    [Parameter()]
    [int]$GpuLayers = -1,

    # Use native inference via FFI instead of HTTP
    [Parameter()]
    [switch]$UseNativeInference
)

#region Script Configuration
$script:Version = '1.0.0'
$script:ModulesPath = Join-Path $PSScriptRoot 'Modules'
$script:ConfigPath = Join-Path $PSScriptRoot 'Config'
$script:ReportsPath = Join-Path $PSScriptRoot 'Reports'
$script:LoadedModules = @{}
$script:Settings = $null
$script:LLMConfig = $null
$script:InferenceMode = 'http'  # 'http' or 'native'
$script:NativeInferenceReady = $false
#endregion

#region Core Module Loading
# Load common module for shared UI and utilities
$script:CommonPath = Join-Path $script:ModulesPath 'PC-AI.Common\PC-AI.Common.psm1'
if (Test-Path $script:CommonPath) {
    Import-Module $script:CommonPath -Force
}
#endregion

#region Module Loading Functions
function Ensure-Module {
    <#
    .SYNOPSIS
        Lazy-loads a PC-AI module only when needed
    #>
    param(
        [Parameter(Mandatory)]
        [string]$ModuleName
    )

    if ($script:LoadedModules[$ModuleName]) {
        return $true
    }

    $modulePath = Join-Path $script:ModulesPath "$ModuleName\$ModuleName.psd1"

    if (-not (Test-Path $modulePath)) {
        Write-Error "Module not found: $ModuleName"
        Write-Info "Expected path: $modulePath"
        return $false
    }

    try {
        Import-Module $modulePath -Force -ErrorAction Stop
        $script:LoadedModules[$ModuleName] = $true
        return $true
    } catch {
        Write-Error "Failed to load module $ModuleName`: $_"
        return $false
    }
}

function Get-LoadedModules {
    return $script:LoadedModules.Keys | Sort-Object
}
#endregion

#region Configuration Functions
function Load-Settings {
    if ($null -ne $script:Settings) {
        return $script:Settings
    }

    $settingsPath = Join-Path $script:ConfigPath 'settings.json'

    if (Test-Path $settingsPath) {
        try {
            $script:Settings = Get-Content $settingsPath -Raw | ConvertFrom-Json
            return $script:Settings
        } catch {
            Write-Warning "Could not load settings: $_"
            return $null
        }
    }

    return $null
}

function Load-LLMConfig {
    if ($null -ne $script:LLMConfig) {
        return $script:LLMConfig
    }

    $llmConfigPath = Join-Path $script:ConfigPath 'llm-config.json'

    if (Test-Path $llmConfigPath) {
        try {
            $script:LLMConfig = Get-Content $llmConfigPath -Raw | ConvertFrom-Json
            return $script:LLMConfig
        } catch {
            Write-Warning "Could not load LLM config: $_"
            return $null
        }
    }

    return $null
}

function Initialize-InferenceBackend {
    <#
    .SYNOPSIS
        Initialize the inference backend based on parameters
    #>
    param(
        [string]$Backend,
        [string]$ModelPath,
        [int]$GpuLayers
    )

    # Skip if HTTP mode
    if ($Backend -eq 'http') {
        Write-Verbose 'Using HTTP inference backend'
        $script:InferenceMode = 'http'
        return $true
    }

    # Try native inference
    Write-Verbose 'Attempting to initialize native inference backend...'

    try {
        # Load PcaiInference module
        $modulePath = Join-Path $script:ModulesPath 'PcaiInference.psm1'

        if (-not (Test-Path $modulePath)) {
            Write-Warning 'PcaiInference module not found. Falling back to HTTP.'
            $script:InferenceMode = 'http'
            return $false
        }

        Import-Module $modulePath -Force -ErrorAction Stop

        # Check DLL availability
        $status = Get-PcaiInferenceStatus
        if (-not $status.DllExists) {
            Write-Warning 'pcai_inference.dll not found. Build instructions:'
            Write-Warning '  .\Build.ps1 -Component inference'
            Write-Warning '  .\Build.ps1 -Component mistralrs   # backend-specific'
            Write-Warning 'Falling back to HTTP inference.'
            $script:InferenceMode = 'http'
            return $false
        }

        # Initialize backend
        $backendName = if ($Backend -eq 'auto') { 'mistralrs' } else { $Backend }
        $initResult = Initialize-PcaiInference -Backend $backendName -Verbose:$VerbosePreference

        if (-not $initResult.Success) {
            Write-Warning 'Failed to initialize native backend. Falling back to HTTP.'
            $script:InferenceMode = 'http'
            return $false
        }

        # Load model if path provided
        if ($ModelPath) {
            Write-Verbose "Loading model: $ModelPath"
            $loadResult = Import-PcaiModel -ModelPath $ModelPath -GpuLayers $GpuLayers -Verbose:$VerbosePreference

            if (-not $loadResult.Success) {
                Write-Warning 'Failed to load model. Falling back to HTTP.'
                Close-PcaiInference
                $script:InferenceMode = 'http'
                return $false
            }

            Write-Info "Native inference ready (backend: $backendName, model: $ModelPath)"
            $script:InferenceMode = 'native'
            $script:NativeInferenceReady = $true
            return $true
        } else {
            Write-Info "Native backend initialized (backend: $backendName). Model not loaded yet."
            Write-Info 'Use Import-PcaiModel to load a model, or inference will fall back to HTTP.'
            $script:InferenceMode = 'http'  # Fall back until model is loaded
            return $false
        }
    } catch {
        Write-Warning "Error initializing native inference: $_"
        Write-Warning 'Falling back to HTTP inference.'
        $script:InferenceMode = 'http'
        return $false
    }
}
#endregion

#region Admin Detection Functions
function Test-Administrator {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Require-Administrator {
    param([string]$Operation = 'this operation')

    if (-not (Test-Administrator)) {
        Write-Error "Administrator privileges required for $Operation"
        Write-Info 'Please run PowerShell as Administrator and try again.'
        return $false
    }
    return $true
}

function Warn-NonAdministrator {
    param([string]$Operation = 'this operation')

    if (-not (Test-Administrator)) {
        Write-Warning "Some features of $Operation may require Administrator privileges"
        Write-Warning 'Consider running as Administrator for full functionality'
    }
}
#endregion

#region Argument Parsing Functions
function Get-ParsedArguments {
    param(
        [string[]]$InputArgs,
        [hashtable]$Defaults = @{}
    )

    if (-not (Ensure-Module 'PC-AI.CLI')) {
        Write-Error 'CLI module unavailable; cannot parse arguments.'
        $fallback = @{
            SubCommand = $null
            Flags      = @{}
            Values     = @{}
            Positional = @()
        }
        foreach ($key in $Defaults.Keys) {
            $fallback.Values[$key] = $Defaults[$key]
        }
        return $fallback
    }

    return Resolve-PCArguments -InputArgs $InputArgs -Defaults $Defaults
}
#endregion

#region Help System
function Show-MainHelp {
    Write-Header "PC-AI v$script:Version - Local LLM-Powered PC Diagnostics"

    Write-Host 'USAGE:' -ForegroundColor Yellow
    Write-Host '    .\PC-AI.ps1 <command> [subcommand] [options]'
    Write-Host ''

    Write-Host 'COMMANDS:' -ForegroundColor Yellow
    $commandSummaries = @()
    if (Ensure-Module 'PC-AI.CLI') {
        $commandSummaries = Get-PCCommandSummary -ProjectRoot $PSScriptRoot
    }
    if ($commandSummaries -and $commandSummaries.Count -gt 0) {
        if (-not ($commandSummaries.Command -contains 'doctor')) {
            $commandSummaries += [PSCustomObject]@{
                Command     = 'doctor'
                Description = 'Run a one-command health check for common runtime failures.'
            }
        }
        foreach ($summary in $commandSummaries) {
            if ($summary.Description) {
                Write-Host "    $($summary.Command) - $($summary.Description)" -ForegroundColor White
            } else {
                Write-Host "    $($summary.Command)" -ForegroundColor White
            }
        }
    } else {
        $commands = @('diagnose', 'optimize', 'usb', 'analyze', 'chat', 'llm', 'cleanup', 'perf', 'status', 'doctor', 'version', 'help')
        foreach ($cmd in $commands) {
            Write-Host "    $cmd" -ForegroundColor White
        }
    }
    Write-Host ''

    Write-Host 'EXAMPLES:' -ForegroundColor Yellow
    $examples = @()
    if (Ensure-Module 'PC-AI.CLI') {
        $examples = Get-PCModuleHelpIndex -ProjectRoot $PSScriptRoot |
            Where-Object { $_.Examples -and $_.Examples.Count -gt 0 } |
            Select-Object -First 6
    }
    if ($examples -and $examples.Count -gt 0) {
        foreach ($entry in $examples) {
            foreach ($example in ($entry.Examples | Select-Object -First 1)) {
                $formatted = $example -replace "(`r`n|`n)", "`n    "
                Write-Host "    $formatted"
            }
        }
    } else {
        Write-Host '    .\PC-AI.ps1 diagnose all'
        Write-Host '    .\PC-AI.ps1 optimize wsl --dry-run'
        Write-Host '    .\PC-AI.ps1 usb list'
        Write-Host '    .\PC-AI.ps1 analyze'
        Write-Host '    .\PC-AI.ps1 help diagnose'
    }
    Write-Host ''

    Write-Host 'Module help is generated dynamically from module implementations.' -ForegroundColor DarkGray
    Write-Host "Run '.\\PC-AI.ps1 help <command>' to see module function help." -ForegroundColor DarkGray
}

function Show-ModuleHelp {
    param(
        [Parameter(Mandatory)]
        [string]$CommandName
    )

    if (-not (Ensure-Module 'PC-AI.CLI')) {
        Write-Warning 'Help module unavailable. Showing basic help.'
        Show-MainHelp
        return
    }

    $modules = Get-PCCommandModules -CommandName $CommandName -ProjectRoot $PSScriptRoot
    if (-not $modules -or $modules.Count -eq 0) {
        Show-MainHelp
        return
    }

    $helpEntries = Get-PCModuleHelpIndex -Modules $modules -ProjectRoot $PSScriptRoot
    Write-Header "$CommandName - Module Help"
    Write-Host 'USAGE:' -ForegroundColor Yellow
    Write-Host "    .\\PC-AI.ps1 $CommandName <subcommand> [options]"
    Write-Host ''

    foreach ($group in ($helpEntries | Group-Object Module)) {
        Write-SubHeader "$($group.Name)"
        foreach ($entry in $group.Group | Sort-Object Name) {
            if ($entry.Synopsis) {
                Write-Bullet "$($entry.Name) - $($entry.Synopsis)"
            } else {
                Write-Bullet "$($entry.Name)"
            }
        }
        Write-Host ''
    }
}

function Show-HelpEntry {
    param(
        [Parameter(Mandatory)]
        [object]$Entry
    )

    Write-Header "$($Entry.Name) - Help"
    if ($Entry.Synopsis) {
        Write-SubHeader 'Synopsis'
        Write-Host $Entry.Synopsis
        Write-Host ''
    }
    if ($Entry.Parameters -and $Entry.Parameters.Count -gt 0) {
        Write-SubHeader 'Parameters'
        if ($Entry.ParameterHelp -and $Entry.ParameterHelp.Count -gt 0) {
            foreach ($paramName in $Entry.Parameters) {
                $paramDesc = $Entry.ParameterHelp[$paramName]
                if ($paramDesc) {
                    Write-Host ('  {0} - {1}' -f $paramName, $paramDesc)
                } else {
                    Write-Host ('  {0}' -f $paramName)
                }
            }
        } else {
            Write-Host ($Entry.Parameters -join ', ')
        }
        Write-Host ''
    }
    if ($Entry.Description) {
        Write-SubHeader 'Description'
        Write-Host $Entry.Description
        Write-Host ''
    }
    if ($Entry.Examples -and $Entry.Examples.Count -gt 0) {
        Write-SubHeader 'Examples'
        foreach ($example in $Entry.Examples) {
            $formatted = $example -replace "(`r`n|`n)", "`n  "
            Write-Host "  $formatted"
            Write-Host ''
        }
    }
    if ($Entry.SourcePath) {
        Write-SubHeader 'Source'
        Write-Host $Entry.SourcePath -ForegroundColor DarkGray
    }
}

function Show-Help {
    param([string]$Topic)

    $knownCommands = @()
    if (Ensure-Module 'PC-AI.CLI') {
        $knownCommands = Get-PCCommandList -ProjectRoot $PSScriptRoot
    }
    if (-not $knownCommands -or $knownCommands.Count -eq 0) {
        $knownCommands = @('diagnose', 'optimize', 'usb', 'analyze', 'chat', 'llm', 'cleanup', 'perf', 'doctor', 'status', 'version', 'help')
    }
    if (-not $Topic) {
        Show-MainHelp
        return
    }

    if ($knownCommands -contains $Topic) {
        Show-ModuleHelp -CommandName $Topic
        return
    }

    if (Ensure-Module 'PC-AI.CLI') {
        $entry = Get-PCModuleHelpEntry -Name $Topic -ProjectRoot $PSScriptRoot | Select-Object -First 1
        if ($entry) {
            Show-HelpEntry -Entry $entry
            return
        }
    }

    Show-MainHelp
}
#endregion

#region Command Implementations

#region Diagnose Commands
function Invoke-DiagnoseCommand {
    param([string[]]$CmdArgs)

    $parsed = Get-ParsedArguments -InputArgs $CmdArgs -Defaults @{
        output = $null
        format = 'txt'
        days   = 3
    }

    $subCommand = $parsed.SubCommand

    switch ($subCommand) {
        'hardware' {
            Warn-NonAdministrator 'hardware diagnostics'

            if (-not (Ensure-Module 'PC-AI.Hardware')) { return }

            Write-Header 'Hardware Diagnostics'

            Write-SubHeader 'Device Manager Errors'
            $deviceErrors = Get-DeviceErrors
            if ($deviceErrors) {
                $deviceErrors | ForEach-Object {
                    Write-Bullet "$($_.Name) - Error Code: $($_.ConfigManagerErrorCode)" -Color Red
                }
            } else {
                Write-Success 'No device errors found'
            }

            Write-SubHeader 'Disk Health (SMART)'
            $diskHealth = Get-DiskHealth
            $diskHealth | ForEach-Object {
                $color = if ($_.Status -eq 'OK') { 'Green' } else { 'Red' }
                Write-Bullet "$($_.Model) - Status: $($_.Status)" -Color $color
            }

            Write-SubHeader 'USB Device Status'
            $usbStatus = Get-UsbStatus
            Write-Bullet "USB Controllers: $($usbStatus.Controllers)"
            Write-Bullet "Connected Devices: $($usbStatus.Devices)"

            Write-SubHeader 'Network Adapters'
            $adapters = Get-NetworkAdapters
            $adapters | Where-Object { $_.PhysicalAdapter } | ForEach-Object {
                $status = if ($_.NetEnabled) { 'Connected' } else { 'Disconnected' }
                $color = if ($_.NetEnabled) { 'Green' } else { 'Yellow' }
                Write-Bullet "$($_.Name) - $status" -Color $color
            }
        }

        'wsl' {
            if (-not (Ensure-Module 'PC-AI.Virtualization')) { return }

            Write-Header 'WSL2 Diagnostics'
            $wslStatus = Get-WSLStatus

            Write-SubHeader 'WSL Status'
            Write-Bullet "Version: $($wslStatus.Version)"
            Write-Bullet "Default Distribution: $($wslStatus.DefaultDistribution)"

            if ($wslStatus.Distributions) {
                Write-SubHeader 'Distributions'
                $wslStatus.Distributions | ForEach-Object {
                    $color = if ($_.State -eq 'Running') { 'Green' } else { 'Gray' }
                    Write-Bullet "$($_.Name) (WSL$($_.Version)) - $($_.State)" -Color $color
                }
            }

            if ($wslStatus.NetworkInfo) {
                Write-SubHeader 'Network Configuration'
                Write-Bullet "IP Address: $($wslStatus.NetworkInfo.IPAddress)"
                Write-Bullet "Gateway: $($wslStatus.NetworkInfo.Gateway)"
            }
        }

        'network' {
            if (-not (Ensure-Module 'PC-AI.Network')) { return }

            Write-Header 'Network Diagnostics'
            $netDiag = Get-NetworkDiagnostics

            Write-SubHeader 'Network Adapters'
            $netDiag.Adapters | ForEach-Object {
                $color = if ($_.Status -eq 'Up') { 'Green' } else { 'Yellow' }
                Write-Bullet "$($_.Name) - $($_.Status)" -Color $color
            }

            Write-SubHeader 'Connectivity Tests'
            $netDiag.Connectivity | ForEach-Object {
                $color = if ($_.Success) { 'Green' } else { 'Red' }
                Write-Bullet "$($_.Target): $(if ($_.Success) { 'OK' } else { 'Failed' })" -Color $color
            }
        }

        'hyperv' {
            if (-not (Require-Administrator 'Hyper-V diagnostics')) { return }
            if (-not (Ensure-Module 'PC-AI.Virtualization')) { return }

            Write-Header 'Hyper-V Diagnostics'
            $hypervStatus = Get-HyperVStatus

            Write-SubHeader 'Hyper-V Status'
            $color = if ($hypervStatus.Enabled) { 'Green' } else { 'Red' }
            Write-Bullet "Hyper-V Enabled: $($hypervStatus.Enabled)" -Color $color

            if ($hypervStatus.VMs) {
                Write-SubHeader 'Virtual Machines'
                $hypervStatus.VMs | ForEach-Object {
                    Write-Bullet "$($_.Name) - $($_.State)"
                }
            }
        }

        'docker' {
            if (-not (Ensure-Module 'PC-AI.Virtualization')) { return }

            Write-Header 'Docker Diagnostics'
            $dockerStatus = Get-DockerStatus

            Write-SubHeader 'Docker Desktop Status'
            $color = if ($dockerStatus.Running) { 'Green' } else { 'Red' }
            Write-Bullet "Running: $($dockerStatus.Running)" -Color $color

            if ($dockerStatus.Version) {
                Write-Bullet "Version: $($dockerStatus.Version)"
            }

            if ($dockerStatus.Containers) {
                Write-SubHeader 'Containers'
                Write-Bullet "Running: $($dockerStatus.Containers.Running)"
                Write-Bullet "Stopped: $($dockerStatus.Containers.Stopped)"
            }
        }

        'events' {
            Warn-NonAdministrator 'event log analysis'
            if (-not (Ensure-Module 'PC-AI.Hardware')) { return }

            $days = [int]$parsed.Values['days']
            Write-Header "System Events (Last $days Days)"

            $events = Get-SystemEvents -Days $days

            if ($events.Critical) {
                Write-SubHeader 'Critical Events'
                $events.Critical | Select-Object -First 10 | ForEach-Object {
                    Write-Bullet "$($_.TimeCreated): $($_.Message)" -Color Red
                }
            }

            if ($events.Error) {
                Write-SubHeader 'Error Events'
                $events.Error | Select-Object -First 10 | ForEach-Object {
                    Write-Bullet "$($_.TimeCreated): $($_.Message)" -Color Yellow
                }
            }

            if (-not $events.Critical -and -not $events.Error) {
                Write-Success 'No critical or error events found'
            }
        }

        'all' {
            Warn-NonAdministrator 'full system diagnostics'
            if (-not (Ensure-Module 'PC-AI.Hardware')) { return }

            Write-Header 'Full System Diagnostics'

            $outputPath = $parsed.Values['output']
            if (-not $outputPath) {
                $timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
                $outputPath = Join-Path $script:ReportsPath "Diagnostics-$timestamp.txt"
            }

            # Ensure Reports directory exists
            $reportsDir = Split-Path $outputPath -Parent
            if (-not (Test-Path $reportsDir)) {
                New-Item -Path $reportsDir -ItemType Directory -Force | Out-Null
            }

            Write-Info 'Generating comprehensive diagnostic report...'
            Write-Info "Output: $outputPath"

            try {
                $report = New-DiagnosticReport -OutputPath $outputPath
                Write-Success 'Diagnostic report generated successfully'
                Write-Info "Report saved to: $outputPath"

                # Show summary
                if ($report.Summary) {
                    Write-SubHeader 'Summary'
                    Write-Bullet "Device Errors: $($report.Summary.DeviceErrors)"
                    Write-Bullet "Disk Issues: $($report.Summary.DiskIssues)"
                    Write-Bullet "Network Issues: $($report.Summary.NetworkIssues)"
                }
            } catch {
                Write-Error "Failed to generate report: $_"
            }
        }

        default {
            if ($subCommand) {
                Write-Error "Unknown diagnose subcommand: $subCommand"
            }
            Show-ModuleHelp -CommandName 'diagnose'
        }
    }
}
#endregion

#region Optimize Commands
function Invoke-OptimizeCommand {
    param([string[]]$CmdArgs)

    $parsed = Get-ParsedArguments -InputArgs $CmdArgs -Defaults @{
        profile = 'default'
        backup  = 'true'
    }

    $subCommand = $parsed.SubCommand
    $dryRun = $parsed.Flags['dry-run'] -or $parsed.Flags['n']
    $force = $parsed.Flags['force'] -or $parsed.Flags['f']

    switch ($subCommand) {
        'wsl' {
            if (-not (Require-Administrator 'WSL optimization')) { return }
            if (-not (Ensure-Module 'PC-AI.Virtualization')) { return }

            Write-Header 'WSL2 Optimization'

            if ($dryRun) {
                Write-Info '[DRY RUN] Showing proposed changes...'
            }

            try {
                $result = Optimize-WSLConfig -DryRun:$dryRun -Force:$force

                if ($result.Changes) {
                    Write-SubHeader 'Proposed/Applied Changes'
                    $result.Changes | ForEach-Object {
                        Write-Bullet $_
                    }
                }

                if (-not $dryRun -and $result.Success) {
                    Write-Success 'WSL optimization completed successfully'
                    if ($result.RestartRequired) {
                        Write-Warning 'WSL restart required. Run: wsl --shutdown'
                    }
                }
            } catch {
                Write-Error "WSL optimization failed: $_"
            }
        }

        'disk' {
            if (-not (Require-Administrator 'disk optimization')) { return }
            if (-not (Ensure-Module 'PC-AI.Performance')) { return }

            Write-Header 'Disk Optimization'

            if ($dryRun) {
                Write-Info '[DRY RUN] Showing disk optimization plan...'
            }

            try {
                $result = Optimize-Disks -DryRun:$dryRun -Force:$force

                if ($result.Disks) {
                    Write-SubHeader 'Optimization Results'
                    $result.Disks | ForEach-Object {
                        $action = if ($_.IsSSD) { 'TRIM' } else { 'Defrag' }
                        Write-Bullet "$($_.DriveLetter): $action - $($_.Status)"
                    }
                }
            } catch {
                Write-Error "Disk optimization failed: $_"
            }
        }

        'vsock' {
            if (-not (Require-Administrator 'VSock optimization')) { return }
            if (-not (Ensure-Module 'PC-AI.Network')) { return }

            Write-Header 'VSock Optimization'

            $profile = $parsed.Values['profile']

            if ($dryRun) {
                Write-Info "[DRY RUN] Profile: $profile"
            }

            try {
                $result = Optimize-VSock -Profile $profile -DryRun:$dryRun

                if ($result.Settings) {
                    Write-SubHeader 'VSock Settings'
                    $result.Settings | ForEach-Object {
                        Write-Bullet "$($_.Name): $($_.Value)"
                    }
                }

                if ($result.Success -and -not $dryRun) {
                    Write-Success 'VSock optimization completed'
                }
            } catch {
                Write-Error "VSock optimization failed: $_"
            }
        }

        'defender' {
            if (-not (Require-Administrator 'Defender exclusions')) { return }
            if (-not (Ensure-Module 'PC-AI.Virtualization')) { return }

            Write-Header 'Windows Defender Exclusions'

            if ($dryRun) {
                Write-Info '[DRY RUN] Showing proposed exclusions...'
            }

            try {
                $result = Set-WSLDefenderExclusions -DryRun:$dryRun

                if ($result.Exclusions) {
                    Write-SubHeader 'Exclusions'
                    $result.Exclusions | ForEach-Object {
                        Write-Bullet "$($_.Type): $($_.Path)"
                    }
                }

                if ($result.Success -and -not $dryRun) {
                    Write-Success 'Defender exclusions configured'
                }
            } catch {
                Write-Error "Failed to set exclusions: $_"
            }
        }

        'network' {
            if (-not (Require-Administrator 'network repair')) { return }
            if (-not (Ensure-Module 'PC-AI.Virtualization')) { return }

            Write-Header 'WSL Network Repair'

            if ($dryRun) {
                Write-Info '[DRY RUN] Showing repair actions...'
            }

            try {
                $result = Repair-WSLNetworking -DryRun:$dryRun

                if ($result.Actions) {
                    Write-SubHeader 'Repair Actions'
                    $result.Actions | ForEach-Object {
                        Write-Bullet $_
                    }
                }

                if ($result.Success -and -not $dryRun) {
                    Write-Success 'Network repair completed'
                }
            } catch {
                Write-Error "Network repair failed: $_"
            }
        }

        default {
            if ($subCommand) {
                Write-Error "Unknown optimize subcommand: $subCommand"
            }
            Show-ModuleHelp -CommandName 'optimize'
        }
    }
}
#endregion

#region USB Commands
function Invoke-UsbCommand {
    param([string[]]$CmdArgs)

    $parsed = Get-ParsedArguments -InputArgs $CmdArgs -Defaults @{
        distribution = $null
        busid        = $null
    }

    $subCommand = $parsed.SubCommand
    $unbind = $parsed.Flags['unbind']

    if (-not (Ensure-Module 'PC-AI.USB')) { return }

    switch ($subCommand) {
        'list' {
            Write-Header 'USB Devices'

            try {
                $devices = Get-UsbDeviceList

                if ($devices) {
                    $devices | ForEach-Object {
                        $state = if ($_.Attached) { '[Attached to WSL]' } else { '' }
                        $color = if ($_.Attached) { 'Green' } else { 'White' }
                        Write-Host "  $($_.BusId)  " -NoNewline -ForegroundColor Cyan
                        Write-Host "$($_.Description) $state" -ForegroundColor $color
                    }
                } else {
                    Write-Info 'No USB devices found'
                }
            } catch {
                Write-Error "Failed to list USB devices: $_"
            }
        }

        'attach' {
            if (-not (Require-Administrator 'USB attach')) { return }

            $busid = $parsed.Values['busid']
            if (-not $busid -and $parsed.Positional) {
                $busid = $parsed.Positional[0]
            }

            if (-not $busid) {
                Write-Error 'Bus ID required. Use: PC-AI usb attach --busid <id>'
                Write-Info "Run 'PC-AI usb list' to see available devices"
                return
            }

            $distribution = $parsed.Values['distribution']

            Write-Header 'Attaching USB Device'
            Write-Info "Bus ID: $busid"
            if ($distribution) {
                Write-Info "Distribution: $distribution"
            }

            try {
                $result = Mount-UsbToWSL -BusId $busid -Distribution $distribution

                if ($result.Success) {
                    Write-Success 'Device attached successfully'
                } else {
                    Write-Error "Failed to attach device: $($result.Error)"
                }
            } catch {
                Write-Error "Failed to attach USB device: $_"
            }
        }

        'detach' {
            if (-not (Require-Administrator 'USB detach')) { return }

            $busid = $parsed.Values['busid']
            if (-not $busid -and $parsed.Positional) {
                $busid = $parsed.Positional[0]
            }

            if (-not $busid) {
                Write-Error 'Bus ID required. Use: PC-AI usb detach --busid <id>'
                return
            }

            Write-Header 'Detaching USB Device'
            Write-Info "Bus ID: $busid"

            try {
                $result = Dismount-UsbFromWSL -BusId $busid -Unbind:$unbind

                if ($result.Success) {
                    Write-Success 'Device detached successfully'
                    if ($unbind) {
                        Write-Info 'Device unbound from usbipd'
                    }
                } else {
                    Write-Error "Failed to detach device: $($result.Error)"
                }
            } catch {
                Write-Error "Failed to detach USB device: $_"
            }
        }

        'status' {
            Write-Header 'USB/WSL Passthrough Status'

            try {
                $status = Get-UsbWSLStatus

                Write-SubHeader 'usbipd Status'
                $color = if ($status.UsbIpdInstalled) { 'Green' } else { 'Red' }
                Write-Bullet "usbipd-win Installed: $($status.UsbIpdInstalled)" -Color $color

                if ($status.AttachedDevices) {
                    Write-SubHeader 'Attached to WSL'
                    $status.AttachedDevices | ForEach-Object {
                        Write-Bullet "$($_.BusId): $($_.Description)" -Color Green
                    }
                }

                if ($status.BoundDevices) {
                    Write-SubHeader 'Bound (Ready to Attach)'
                    $status.BoundDevices | ForEach-Object {
                        Write-Bullet "$($_.BusId): $($_.Description)" -Color Yellow
                    }
                }
            } catch {
                Write-Error "Failed to get USB status: $_"
            }
        }

        'bind' {
            if (-not (Require-Administrator 'USB bind')) { return }

            $busid = $parsed.Values['busid']
            if (-not $busid -and $parsed.Positional) {
                $busid = $parsed.Positional[0]
            }

            if (-not $busid) {
                Write-Error 'Bus ID required. Use: PC-AI usb bind --busid <id>'
                return
            }

            Write-Header 'Binding USB Device'
            Write-Info "Bus ID: $busid"

            try {
                $result = Invoke-UsbBind -BusId $busid

                if ($result.Success) {
                    Write-Success 'Device bound successfully'
                    Write-Info 'Device is now ready for WSL attachment'
                } else {
                    Write-Error "Failed to bind device: $($result.Error)"
                }
            } catch {
                Write-Error "Failed to bind USB device: $_"
            }
        }

        default {
            if ($subCommand) {
                Write-Error "Unknown usb subcommand: $subCommand"
            }
            Show-ModuleHelp -CommandName 'usb'
        }
    }
}
#endregion

#region Analyze Commands
function Invoke-AnalyzeCommand {
    param([string[]]$CmdArgs)

    $parsed = Get-ParsedArguments -InputArgs $CmdArgs -Defaults @{
        report      = $null
        model       = $null
        temperature = '0.3'
        output      = $null
    }

    if (-not (Ensure-Module 'PC-AI.LLM')) { return }

    Write-Header 'LLM Diagnostic Analysis'

    # Check LLM availability first
    $llmStatus = Get-LLMStatus
    if (-not $llmStatus.Available) {
        Write-Error 'No LLM provider available'
        Write-Info 'Ensure pcai-inference is running'
        return
    }

    $reportPath = $parsed.Values['report']

    # Find latest report if not specified
    if (-not $reportPath) {
        $latestReport = Get-ChildItem -Path $script:ReportsPath -Filter '*.txt' -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 1

        if ($latestReport) {
            $reportPath = $latestReport.FullName
            Write-Info "Using latest report: $reportPath"
        } else {
            Write-Warning "No diagnostic reports found. Run 'PC-AI diagnose all' first."

            # Offer to run diagnostics
            $response = Read-Host 'Run diagnostics now? (y/n)'
            if ($response -eq 'y') {
                Invoke-DiagnoseCommand @('all')
                $latestReport = Get-ChildItem -Path $script:ReportsPath -Filter '*.txt' |
                    Sort-Object LastWriteTime -Descending |
                    Select-Object -First 1
                if ($latestReport) {
                    $reportPath = $latestReport.FullName
                }
            } else {
                return
            }
        }
    }

    if (-not (Test-Path $reportPath)) {
        Write-Error "Report not found: $reportPath"
        return
    }

    $model = $parsed.Values['model']
    if (-not $model) {
        $llmConfig = Load-LLMConfig
        $model = $llmConfig.providers.'pcai-inference'.defaultModel
    }

    Write-Info "Model: $model"
    Write-Info 'Analyzing report...'

    try {
        $analysisParams = @{
            ReportPath = $reportPath
        }

        if ($model) {
            $analysisParams['Model'] = $model
        }

        if ($parsed.Values['temperature']) {
            $analysisParams['Temperature'] = [double]$parsed.Values['temperature']
        }

        $result = Invoke-PCDiagnosis @analysisParams

        Write-Header 'Analysis Results'

        if ($result.Summary) {
            Write-SubHeader 'Summary'
            Write-Host $result.Summary
        }

        if ($result.Issues) {
            Write-SubHeader 'Identified Issues'
            $result.Issues | ForEach-Object {
                $color = switch ($_.Priority) {
                    'Critical' { 'Red' }
                    'High' { 'Yellow' }
                    'Medium' { 'Cyan' }
                    default { 'White' }
                }
                Write-Bullet "[$($_.Priority)] $($_.Description)" -Color $color
            }
        }

        if ($result.Recommendations) {
            Write-SubHeader 'Recommendations'
            $i = 1
            $result.Recommendations | ForEach-Object {
                Write-Host "  $i. $_" -ForegroundColor Green
                $i++
            }
        }

        # Save analysis if output specified
        $outputPath = $parsed.Values['output']
        if ($outputPath) {
            $result | ConvertTo-Json -Depth 10 | Set-Content $outputPath
            Write-Info "Analysis saved to: $outputPath"
        }
    } catch {
        Write-Error "Analysis failed: $_"
    }
}
#endregion

#region Chat Commands
function Invoke-ChatCommand {
    param([string[]]$CmdArgs)

    $parsed = Get-ParsedArguments -InputArgs $CmdArgs -Defaults @{
        model  = $null
        system = $null
    }

    if (-not (Ensure-Module 'PC-AI.LLM')) { return }

    # Check LLM availability
    $llmStatus = Get-LLMStatus
    if (-not $llmStatus.Available) {
        Write-Error 'No LLM provider available'
        Write-Info 'Ensure pcai-inference is running'
        return
    }

    $model = $parsed.Values['model']
    if (-not $model) {
        $llmConfig = Load-LLMConfig
        $model = $llmConfig.providers.'pcai-inference'.defaultModel
    }

    $includeContext = $parsed.Flags['context']

    Write-Header 'PC-AI Chat'
    Write-Info "Model: $model"
    Write-Info "Type '/help' for commands, '/quit' to exit"
    Write-Host ''

    # Start chat session
    try {
        $chatParams = @{
            Model       = $model
            Interactive = $true
        }

        if ($includeContext) {
            $chatParams['IncludeDiagnosticContext'] = $true
        }

        if ($parsed.Values['system']) {
            $chatParams['SystemPromptPath'] = $parsed.Values['system']
        }

        Invoke-LLMChat @chatParams
    } catch {
        Write-Error "Chat session failed: $_"
    }
}
#endregion

#region LLM Commands
function Invoke-LLMCommand {
    param([string[]]$CmdArgs)

    $parsed = Get-ParsedArguments -InputArgs $CmdArgs -Defaults @{
        provider = 'pcai-inference'
        model    = $null
    }

    $subCommand = $parsed.SubCommand
    $values = $parsed.Values
    $flags = $parsed.Flags

    function Get-ArgValue {
        param([string]$Name, $Default = $null)
        if ($values.ContainsKey($Name)) { return $values[$Name] }
        return $Default
    }

    function Get-ArgFlag {
        param([string]$Name)
        return ($flags.ContainsKey($Name) -and $flags[$Name])
    }

    if (-not (Ensure-Module 'PC-AI.LLM')) { return }

    switch ($subCommand) {
        'status' {
            Write-Header 'LLM Provider Status'

            try {
                $status = Get-LLMStatus

                Write-SubHeader 'pcai-inference'
                $color = if ($status.PcaiInference.ApiConnected) { 'Green' } else { 'Red' }
                Write-Bullet "Available: $($status.PcaiInference.ApiConnected)" -Color $color
                if ($status.PcaiInference.ApiConnected) {
                    Write-Bullet "URL: $($status.PcaiInference.ApiUrl)"
                    $modelNames = $status.PcaiInference.Models | ForEach-Object { $_.Name } | Where-Object { $_ }
                    if ($modelNames) {
                        Write-Bullet "Models: $($modelNames -join ', ')"
                    }
                }

                if ($status.Router) {
                    Write-SubHeader 'FunctionGemma Router'
                    $color = if ($status.Router.ApiConnected) { 'Green' } else { 'Red' }
                    Write-Bullet "Available: $($status.Router.ApiConnected)" -Color $color
                    if ($status.Router.ApiConnected) {
                        Write-Bullet "URL: $($status.Router.ApiUrl)"
                        Write-Bullet "Model: $($status.Router.Model)"
                    }
                }
            } catch {
                Write-Error "Failed to get LLM status: $_"
            }
        }

        'models' {
            Write-Header 'Available LLM Models'

            $provider = $parsed.Values['provider']

            try {
                $status = Get-LLMStatus

                if ($provider -in @('pcai-inference', 'ollama') -and $status.PcaiInference.ApiConnected) {
                    Write-SubHeader 'pcai-inference Models'
                    $status.PcaiInference.Models | ForEach-Object {
                        $color = if ($_.Name -eq $status.PcaiInference.DefaultModel) { 'Green' } else { 'White' }
                        Write-Bullet "$($_.Name)" -Color $color
                    }
                }
            } catch {
                Write-Error "Failed to list models: $_"
            }
        }

        'config' {
            Write-Header 'LLM Configuration'

            $llmConfig = Load-LLMConfig

            if ($parsed.Values['model']) {
                # Set model
                $model = $parsed.Values['model']
                try {
                    Set-LLMConfig -DefaultModel $model
                    Write-Success "Default model set to: $model"
                } catch {
                    Write-Error "Failed to set model: $_"
                }
            } else {
                # Show config
                Write-SubHeader 'Current Configuration'
                Write-Bullet "Default Model: $($llmConfig.providers.'pcai-inference'.defaultModel)"
                Write-Bullet "Timeout: $($llmConfig.providers.'pcai-inference'.timeout)ms"
                Write-Bullet "Max Context: $($llmConfig.contextManagement.maxContextTokens) tokens"
            }
        }

        'test' {
            Write-Header 'LLM Connectivity Test'

            try {
                $testResult = Send-OllamaRequest -Prompt "Respond with 'OK' only." -Stream $false

                if ($testResult) {
                    Write-Success 'LLM connectivity test passed'
                    Write-Info "Response: $testResult"
                }
            } catch {
                Write-Error "LLM connectivity test failed: $_"
            }
        }

        'tui' {
            Write-Header 'LLM Chat TUI'

            $rawTuiArgs = @($CmdArgs | Select-Object -Skip 1)
            $buildIfMissing = $false
            $tuiArgs = @()
            foreach ($arg in $rawTuiArgs) {
                if ($arg -in @('--build-if-missing', '--build')) {
                    $buildIfMissing = $true
                    continue
                }
                $tuiArgs += $arg
            }

            try {
                Invoke-LLMChatTui -Arguments $tuiArgs -BuildIfMissing:$buildIfMissing
            } catch {
                Write-Error "Failed to launch TUI: $_"
            }
        }

        'router-prepare' {
            Write-Header 'FunctionGemma Router Dataset'

            $args = @{}
            $args.ToolsPath = Get-ArgValue 'tools'
            $args.DiagnosePrompt = Get-ArgValue 'diagnose-prompt'
            $args.ChatPrompt = Get-ArgValue 'chat-prompt'
            $args.ScenariosPath = Get-ArgValue 'scenarios'
            $args.Output = Get-ArgValue 'output'
            $args.TestVectors = Get-ArgValue 'test-vectors'

            $maxCases = Get-ArgValue 'max-cases'
            if ($null -ne $maxCases) { $args.MaxCases = [int]$maxCases }

            if (Get-ArgFlag 'no-tool-coverage') { $args.NoToolCoverage = $true }
            if (Get-ArgFlag 'stream') { $args.Stream = $true }
            if (Get-ArgFlag 'native') { $args.UseNative = $true }
            if (Get-ArgFlag 'native-only') { $args.NativeOnly = $true }
            if (Get-ArgFlag 'use-lld') { $args.UseLld = $true }
            if (Get-ArgFlag 'llm-debug') { $args.LlmDebug = $true }

            Invoke-FunctionGemmaDataset @args
        }

        'router-cache' {
            Write-Header 'FunctionGemma Token Cache'

            $args = @{}
            $args.Input = Get-ArgValue 'input'
            $args.TokenizerPath = Get-ArgValue 'tokenizer'
            $args.OutputDir = Get-ArgValue 'output-dir'
            if (Get-ArgFlag 'use-lld') { $args.UseLld = $true }
            if (Get-ArgFlag 'llm-debug') { $args.LlmDebug = $true }

            Invoke-FunctionGemmaTokenCache @args
        }

        'router-train' {
            Write-Header 'FunctionGemma Training'

            $args = @{}
            $args.ModelPath = Get-ArgValue 'model-path'
            $args.TrainData = Get-ArgValue 'train-data'
            $args.EvalData = Get-ArgValue 'eval-data'
            $args.EvalSplit = Get-ArgValue 'eval-split'
            $args.TokenCache = Get-ArgValue 'token-cache'
            $args.Output = Get-ArgValue 'output'
            $args.ConfigPath = Get-ArgValue 'config'

            $epochs = Get-ArgValue 'epochs'
            if ($null -ne $epochs) { $args.Epochs = [int]$epochs }
            $lr = Get-ArgValue 'lr'
            if ($null -ne $lr) { $args.Lr = [double]$lr }
            $loraR = Get-ArgValue 'lora-r'
            if ($null -ne $loraR) { $args.LoraR = [int]$loraR }
            $loraAlpha = Get-ArgValue 'lora-alpha'
            if ($null -ne $loraAlpha) { $args.LoraAlpha = [double]$loraAlpha }
            $loraDropout = Get-ArgValue 'lora-dropout'
            if ($null -ne $loraDropout) { $args.LoraDropout = [double]$loraDropout }
            $batchSize = Get-ArgValue 'batch-size'
            if ($null -ne $batchSize) { $args.BatchSize = [int]$batchSize }
            $gradAccum = Get-ArgValue 'grad-accum'
            if ($null -ne $gradAccum) { $args.GradAccum = [int]$gradAccum }
            $maxSeqLen = Get-ArgValue 'max-seq-len'
            if ($null -ne $maxSeqLen) { $args.MaxSeqLen = [int]$maxSeqLen }
            $eosTokenId = Get-ArgValue 'eos-token-id'
            if ($null -ne $eosTokenId) { $args.EosTokenId = [int]$eosTokenId }
            $warmupSteps = Get-ArgValue 'warmup-steps'
            if ($null -ne $warmupSteps) { $args.WarmupSteps = [int]$warmupSteps }
            $schedulerType = Get-ArgValue 'scheduler-type'
            if ($null -ne $schedulerType) { $args.SchedulerType = $schedulerType }
            $earlyStoppingPatience = Get-ArgValue 'early-stopping-patience'
            if ($null -ne $earlyStoppingPatience) { $args.EarlyStoppingPatience = [int]$earlyStoppingPatience }
            $earlyStoppingMinDelta = Get-ArgValue 'early-stopping-min-delta'
            if ($null -ne $earlyStoppingMinDelta) { $args.EarlyStoppingMinDelta = [double]$earlyStoppingMinDelta }
            $evalMaxBatches = Get-ArgValue 'eval-max-batches'
            if ($null -ne $evalMaxBatches) { $args.EvalMaxBatches = [int]$evalMaxBatches }

            if (Get-ArgFlag 'pack-sequences') { $args.PackSequences = $true }
            if (Get-ArgFlag 'disable-lora') { $args.DisableLora = $true }
            if (Get-ArgFlag 'use-4bit') { $args.Use4Bit = $true }
            if (Get-ArgFlag 'use-lld') { $args.UseLld = $true }
            if (Get-ArgFlag 'llm-debug') { $args.LlmDebug = $true }

            Invoke-FunctionGemmaTrain @args
        }

        'router-eval' {
            Write-Header 'FunctionGemma Evaluation'

            $args = @{}
            $args.ModelPath = Get-ArgValue 'model-path'
            $args.TestData = Get-ArgValue 'test-data'
            $args.Adapters = Get-ArgValue 'adapters'
            $args.Output = Get-ArgValue 'output'
            $args.ConfigPath = Get-ArgValue 'config'

            $maxNewTokens = Get-ArgValue 'max-new-tokens'
            if ($null -ne $maxNewTokens) { $args.MaxNewTokens = [int]$maxNewTokens }
            $loraR = Get-ArgValue 'lora-r'
            if ($null -ne $loraR) { $args.LoraR = [int]$loraR }
            $maxSamples = Get-ArgValue 'max-samples'
            if ($null -ne $maxSamples) { $args.MaxSamples = [int]$maxSamples }

            if (Get-ArgFlag 'quiet') { $args.Quiet = $true }
            if (Get-ArgFlag 'verbose') { $args.VerboseOutput = $true }
            if (Get-ArgFlag 'fast-eval') { $args.FastEval = $true }
            if (Get-ArgFlag 'no-schema-validate') { $args.NoSchemaValidate = $true }
            if (Get-ArgFlag 'use-lld') { $args.UseLld = $true }
            if (Get-ArgFlag 'llm-debug') { $args.LlmDebug = $true }

            Invoke-FunctionGemmaEval @args
        }

        'router-tests' {
            Write-Header 'FunctionGemma Test Suite'

            $args = @{}
            if (Get-ArgFlag 'use-lld') { $args.UseLld = $true }
            if (Get-ArgFlag 'llm-debug') { $args.LlmDebug = $true }

            Invoke-FunctionGemmaTests @args
        }

        default {
            if ($subCommand) {
                Write-Error "Unknown llm subcommand: $subCommand"
            }
            Show-ModuleHelp -CommandName 'llm'
        }
    }
}
#endregion

#region Cleanup Commands
function Invoke-CleanupCommand {
    param([string[]]$CmdArgs)

    $parsed = Get-ParsedArguments -InputArgs $CmdArgs -Defaults @{}

    $subCommand = $parsed.SubCommand
    $dryRun = $parsed.Flags['dry-run'] -or $parsed.Flags['n']
    $force = $parsed.Flags['force'] -or $parsed.Flags['f']
    $recursive = $parsed.Flags['recursive'] -or $parsed.Flags['r']

    if (-not (Ensure-Module 'PC-AI.Cleanup')) { return }

    switch ($subCommand) {
        'path' {
            Write-Header 'PATH Environment Cleanup'

            if ($dryRun) {
                Write-Info '[DRY RUN] Analyzing PATH...'
            }

            try {
                $analysis = Get-PathDuplicates

                if ($analysis.Duplicates) {
                    Write-SubHeader 'Duplicate Entries'
                    $analysis.Duplicates | ForEach-Object {
                        Write-Bullet $_ -Color Yellow
                    }
                }

                if ($analysis.NonExistent) {
                    Write-SubHeader 'Non-Existent Paths'
                    $analysis.NonExistent | ForEach-Object {
                        Write-Bullet $_ -Color Red
                    }
                }

                if (-not $analysis.Duplicates -and -not $analysis.NonExistent) {
                    Write-Success 'PATH is clean - no issues found'
                    return
                }

                if (-not $dryRun) {
                    if (-not $force) {
                        $response = Read-Host 'Clean up PATH? (y/n)'
                        if ($response -ne 'y') {
                            Write-Info 'Cleanup cancelled'
                            return
                        }
                    }

                    if (-not (Require-Administrator 'PATH cleanup')) { return }

                    $result = Repair-MachinePath -CreateBackup

                    if ($result.Success) {
                        Write-Success 'PATH cleaned successfully'
                        Write-Info "Entries removed: $($result.EntriesRemoved)"
                        if ($result.BackupPath) {
                            Write-Info "Backup saved to: $($result.BackupPath)"
                        }
                    }
                }
            } catch {
                Write-Error "PATH cleanup failed: $_"
            }
        }

        'temp' {
            Write-Header 'Temporary Files Cleanup'

            if ($dryRun) {
                Write-Info '[DRY RUN] Analyzing temp files...'
            }

            try {
                $result = Clear-TempFiles -DryRun:$dryRun

                Write-SubHeader 'Cleanup Summary'
                Write-Bullet "Files found: $($result.FilesFound)"
                Write-Bullet "Space to recover: $([math]::Round($result.SpaceBytes / 1MB, 2)) MB"

                if (-not $dryRun) {
                    Write-Bullet "Files deleted: $($result.FilesDeleted)"
                    Write-Bullet "Space recovered: $([math]::Round($result.SpaceRecovered / 1MB, 2)) MB"

                    if ($result.Errors) {
                        Write-SubHeader 'Errors (files in use)'
                        $result.Errors | Select-Object -First 5 | ForEach-Object {
                            Write-Bullet $_ -Color Yellow
                        }
                    }
                }
            } catch {
                Write-Error "Temp cleanup failed: $_"
            }
        }

        'duplicates' {
            $searchPath = if ($parsed.Positional) { $parsed.Positional[0] } else { $null }

            if (-not $searchPath) {
                Write-Error 'Path required. Use: PC-AI cleanup duplicates <path>'
                return
            }

            if (-not (Test-Path $searchPath)) {
                Write-Error "Path not found: $searchPath"
                return
            }

            Write-Header 'Duplicate File Detection'
            Write-Info "Searching: $searchPath"
            Write-Info "Recursive: $recursive"

            try {
                $result = Find-DuplicateFiles -Path $searchPath -Recursive:$recursive

                if ($result.DuplicateSets) {
                    Write-SubHeader "Duplicate Sets Found: $($result.DuplicateSets.Count)"

                    $result.DuplicateSets | Select-Object -First 10 | ForEach-Object {
                        Write-Host ''
                        Write-Host "  Hash: $($_.Hash.Substring(0, 16))..." -ForegroundColor DarkGray
                        Write-Host "  Size: $([math]::Round($_.Size / 1KB, 2)) KB" -ForegroundColor DarkGray
                        $_.Files | ForEach-Object {
                            Write-Bullet $_
                        }
                    }

                    Write-Host ''
                    Write-Bullet "Total duplicate sets: $($result.DuplicateSets.Count)" -Color Yellow
                    Write-Bullet "Potential space savings: $([math]::Round($result.WastedSpace / 1MB, 2)) MB" -Color Yellow
                } else {
                    Write-Success 'No duplicate files found'
                }
            } catch {
                Write-Error "Duplicate detection failed: $_"
            }
        }

        default {
            if ($subCommand) {
                Write-Error "Unknown cleanup subcommand: $subCommand"
            }
            Show-ModuleHelp -CommandName 'cleanup'
        }
    }
}
#endregion

#region Performance Commands
function Invoke-PerfCommand {
    param([string[]]$CmdArgs)

    $parsed = Get-ParsedArguments -InputArgs $CmdArgs -Defaults @{
        top      = 10
        duration = 30
        interval = 1000
        sort     = 'cpu'
    }

    $subCommand = $parsed.SubCommand

    if (-not (Ensure-Module 'PC-AI.Performance')) { return }

    switch ($subCommand) {
        'disk' {
            Write-Header 'Disk Space Analysis'

            try {
                $diskSpace = Get-DiskSpace

                $diskSpace | ForEach-Object {
                    $usedPercent = [math]::Round($_.UsedPercent, 1)
                    $freeGB = [math]::Round($_.FreeGB, 2)
                    $totalGB = [math]::Round($_.TotalGB, 2)

                    $color = if ($usedPercent -gt 90) { 'Red' }
                    elseif ($usedPercent -gt 75) { 'Yellow' }
                    else { 'Green' }

                    Write-Host ''
                    Write-Host "  Drive $($_.DriveLetter):" -ForegroundColor Cyan
                    Write-Host '    Used: ' -NoNewline
                    Write-Host "$usedPercent%" -ForegroundColor $color -NoNewline
                    Write-Host " ($($totalGB - $freeGB) GB / $totalGB GB)"
                    Write-Host "    Free: $freeGB GB"
                    Write-Host "    Type: $($_.DriveType)"

                    # Visual bar
                    $barLength = 30
                    $filledLength = [math]::Round($usedPercent / 100 * $barLength)
                    $bar = '=' * $filledLength + '-' * ($barLength - $filledLength)
                    Write-Host "    [$bar]" -ForegroundColor $color
                }
            } catch {
                Write-Error "Disk analysis failed: $_"
            }
        }

        'process' {
            Write-Header 'Top Processes by Resource Usage'

            $topN = [int]$parsed.Values['top']
            $sortBy = $parsed.Values['sort']

            try {
                $processes = Get-ProcessPerformance -Top $topN -SortBy $sortBy

                Write-SubHeader "Top $topN by $($sortBy.ToUpper())"

                $format = '{0,-6} {1,-30} {2,10} {3,12}'
                Write-Host ($format -f 'PID', 'Name', 'CPU %', 'Memory MB') -ForegroundColor DarkGray

                $processes | ForEach-Object {
                    $cpuColor = if ($_.CPUPercent -gt 50) { 'Red' }
                    elseif ($_.CPUPercent -gt 25) { 'Yellow' }
                    else { 'White' }

                    $memMB = [math]::Round($_.MemoryMB, 1)
                    $name = if ($_.Name.Length -gt 28) { $_.Name.Substring(0, 28) + '..' } else { $_.Name }

                    Write-Host ($format -f $_.PID, $name, "$($_.CPUPercent)%", "$memMB MB") -ForegroundColor $cpuColor
                }
            } catch {
                Write-Error "Process analysis failed: $_"
            }
        }

        'watch' {
            Write-Header 'Real-Time System Monitor'

            $duration = [int]$parsed.Values['duration']
            $interval = [int]$parsed.Values['interval']

            Write-Info "Duration: $duration seconds"
            Write-Info "Interval: $interval ms"
            Write-Info 'Press Ctrl+C to stop early'
            Write-Host ''

            try {
                Watch-SystemResources -Duration $duration -IntervalMs $interval
            } catch {
                Write-Error "Monitoring failed: $_"
            }
        }

        'vsock' {
            if (-not (Ensure-Module 'PC-AI.Network')) { return }

            Write-Header 'VSock Performance Monitor'

            try {
                Watch-VSockPerformance
            } catch {
                Write-Error "VSock monitoring failed: $_"
            }
        }

        default {
            if ($subCommand) {
                Write-Error "Unknown perf subcommand: $subCommand"
            }
            Show-ModuleHelp -CommandName 'perf'
        }
    }
}
#endregion

#region Doctor Command
function Invoke-DoctorCommand {
    param([string[]]$CmdArgs)

    $parsed = Get-ParsedArguments -InputArgs $CmdArgs -Defaults @{
        json   = $false
        legacy = $false
    }

    $emitJson = $false
    if ($parsed.Flags.ContainsKey('json') -or ($parsed.Values['json'] -eq $true) -or ($parsed.Values['json'] -eq 'true')) {
        $emitJson = $true
    }

    $checkLegacy = $false
    if ($parsed.Flags.ContainsKey('legacy') -or ($parsed.Values['legacy'] -eq $true) -or ($parsed.Values['legacy'] -eq 'true')) {
        $checkLegacy = $true
    }

    if (-not (Ensure-Module 'PC-AI.Virtualization')) { return }

    $llmConfig = Load-LLMConfig
    $pcaiUrl = $null
    $functionGemmaUrl = $null

    if ($llmConfig -and $llmConfig.providers -and $llmConfig.providers.'pcai-inference' -and $llmConfig.providers.'pcai-inference'.baseUrl) {
        $pcaiUrl = $llmConfig.providers.'pcai-inference'.baseUrl
    }
    if ($llmConfig -and $llmConfig.providers -and $llmConfig.providers.functiongemma -and $llmConfig.providers.functiongemma.baseUrl) {
        $functionGemmaUrl = $llmConfig.providers.functiongemma.baseUrl
    }

    $nativeDllPaths = @()
    if ($llmConfig -and $llmConfig.nativeInference -and $llmConfig.nativeInference.dllSearchPaths) {
        foreach ($path in $llmConfig.nativeInference.dllSearchPaths) {
            if ([string]::IsNullOrWhiteSpace($path)) { continue }
            if ([System.IO.Path]::IsPathRooted($path)) {
                $nativeDllPaths += $path
            } else {
                $nativeDllPaths += (Join-Path $PSScriptRoot $path)
            }
        }
    }

    $doctorParams = @{
        PcaiInferenceUrl = $pcaiUrl
        FunctionGemmaUrl = $functionGemmaUrl
    }
    if ($nativeDllPaths.Count -gt 0) {
        $doctorParams.NativeDllSearchPaths = $nativeDllPaths
    }
    if ($checkLegacy) {
        $doctorParams.CheckLegacyProviders = $true
    }

    try {
        $report = Invoke-PcaiDoctor @doctorParams
    } catch {
        Write-Error "Doctor checks failed: $_"
        return
    }

    if ($emitJson) {
        $report | ConvertTo-Json -Depth 6
        return
    }

    $health = $report.Health

    Write-Header 'PC-AI Doctor'
    Write-SubHeader 'Summary'

    $overallColor = if ($health.OverallStatus -eq 'Healthy') { 'Green' } elseif ($health.OverallStatus -eq 'Degraded') { 'Yellow' } else { 'Yellow' }
    Write-Bullet "Overall: $($health.OverallStatus)" -Color $overallColor

    $pcaiColor = if ($health.PcaiInference.Status -eq 'OK') { 'Green' } elseif ($health.PcaiInference.Status -eq 'NotRunning') { 'Red' } else { 'Yellow' }
    Write-Bullet "pcai-inference: $($health.PcaiInference.Status)" -Color $pcaiColor

    $routerColor = if ($health.FunctionGemma.Status -eq 'OK') { 'Green' } elseif ($health.FunctionGemma.Status -eq 'NotRunning') { 'Red' } else { 'Yellow' }
    Write-Bullet "FunctionGemma: $($health.FunctionGemma.Status)" -Color $routerColor

    $nativeColor = if ($health.NativeFFI.DllExists) { 'Green' } else { 'Yellow' }
    Write-Bullet "Native DLL: $($health.NativeFFI.Status)" -Color $nativeColor

    $gpuColor = if ($health.Gpu.Status -eq 'OK') { 'Green' } elseif ($health.Gpu.Status -eq 'NotFound') { 'Yellow' } else { 'Yellow' }
    Write-Bullet "GPU: $($health.Gpu.Status)" -Color $gpuColor

    Write-SubHeader 'pcai-inference'
    Write-Bullet "URL: $pcaiUrl"
    $respondColor = if ($health.PcaiInference.Responding) { 'Green' } else { 'Red' }
    Write-Bullet "Responding: $($health.PcaiInference.Responding)" -Color $respondColor
    if ($health.PcaiInference.Backend) {
        Write-Bullet "Backend: $($health.PcaiInference.Backend)"
    }
    $modelLoadedColor = if ($health.PcaiInference.ModelLoaded) { 'Green' } else { 'Yellow' }
    Write-Bullet "Model Loaded: $($health.PcaiInference.ModelLoaded)" -Color $modelLoadedColor

    Write-SubHeader 'FunctionGemma Router'
    Write-Bullet "URL: $functionGemmaUrl"
    $fgRespondColor = if ($health.FunctionGemma.Responding) { 'Green' } else { 'Red' }
    Write-Bullet "Responding: $($health.FunctionGemma.Responding)" -Color $fgRespondColor

    Write-SubHeader 'Native Inference DLL'
    Write-Bullet "Available: $($health.NativeFFI.DllExists)" -Color $nativeColor
    if ($health.NativeFFI.Path) {
        Write-Bullet "Path: $($health.NativeFFI.Path)"
    }

    Write-SubHeader 'GPU'
    if ($health.Gpu.Devices -and $health.Gpu.Devices.Count -gt 0) {
        foreach ($gpu in $health.Gpu.Devices) {
            $driver = if ($gpu.DriverVersion) { " (Driver $($gpu.DriverVersion))" } else { '' }
            Write-Bullet "$($gpu.Name)$driver"
        }
    } else {
        Write-Bullet 'No GPU devices detected'
    }

    $nvidiaSmiColor = if ($health.Gpu.NvidiaSmi) { 'Green' } else { 'Yellow' }
    Write-Bullet "NVIDIA SMI: $($health.Gpu.NvidiaSmi)" -Color $nvidiaSmiColor

    if ($health.Docker.Status -ne 'Unknown') {
        $dockerColor = if ($health.Docker.Status -eq 'OK') { 'Green' } elseif ($health.Docker.Status -in @('NotRunning', 'NotInstalled')) { 'Yellow' } else { 'Yellow' }
        Write-Bullet "Docker: $($health.Docker.Status)" -Color $dockerColor
        if ($health.Docker.Running) {
            $runtimeColor = if ($health.Gpu.NvidiaRuntime) { 'Green' } else { 'Yellow' }
            Write-Bullet "Docker NVIDIA Runtime: $($health.Gpu.NvidiaRuntime)" -Color $runtimeColor
        }
    }

    Write-SubHeader 'Environment'
    if ($health.WSL.Status -ne 'Unknown') {
        $wslColor = if ($health.WSL.Status -eq 'OK') { 'Green' } elseif ($health.WSL.Status -eq 'Stopped') { 'Yellow' } else { 'Red' }
        Write-Bullet "WSL: $($health.WSL.Status)" -Color $wslColor
    }

    if ($report.Recommendations -and $report.Recommendations.Count -gt 0) {
        Write-SubHeader 'Suggested Fixes'
        foreach ($rec in $report.Recommendations) {
            Write-Bullet $rec -Color Yellow
        }
    } else {
        Write-Success 'No issues detected.'
    }
}
#endregion

#region Status Command
function Invoke-StatusCommand {
    Write-Header 'PC-AI System Status'

    # Framework info
    Write-SubHeader 'Framework'
    Write-Bullet "Version: $script:Version"
    Write-Bullet "Modules Path: $script:ModulesPath"
    Write-Bullet "Config Path: $script:ConfigPath"

    # Admin status
    $isAdmin = Test-Administrator
    $adminColor = if ($isAdmin) { 'Green' } else { 'Yellow' }
    Write-Bullet "Administrator: $isAdmin" -Color $adminColor

    # Module status
    Write-SubHeader 'Modules'
    $modules = @(
        'PC-AI.Hardware',
        'PC-AI.Virtualization',
        'PC-AI.USB',
        'PC-AI.Network',
        'PC-AI.Performance',
        'PC-AI.Cleanup',
        'PC-AI.LLM',
        'PC-AI.CLI'
    )

    foreach ($moduleName in $modules) {
        $modulePath = Join-Path $script:ModulesPath "$moduleName\$moduleName.psd1"
        $exists = Test-Path $modulePath
        $color = if ($exists) { 'Green' } else { 'Red' }
        $status = if ($exists) { 'Available' } else { 'Missing' }
        Write-Bullet "$moduleName`: $status" -Color $color
    }

    # Inference backend status
    Write-SubHeader 'Inference Backend'
    Write-Bullet "Mode: $script:InferenceMode"

    if ($script:InferenceMode -eq 'native') {
        try {
            $status = Get-PcaiInferenceStatus
            Write-Bullet "Native Backend: $($status.CurrentBackend)" -Color Green
            Write-Bullet "Model Loaded: $($status.ModelLoaded)" -Color $(if ($status.ModelLoaded) { 'Green' } else { 'Yellow' })
        } catch {
            Write-Bullet 'Native Backend: Error' -Color Red
        }
    }

    # LLM status (quick check for HTTP mode)
    Write-SubHeader 'LLM Provider (HTTP)'
    try {
        $pcaiUrl = $null
        $cfgPath = Join-Path $script:ConfigPath 'llm-config.json'
        if (Test-Path $cfgPath) {
            $cfg = Get-Content $cfgPath -Raw | ConvertFrom-Json
            $pcaiUrl = $cfg.providers.'pcai-inference'.baseUrl
        }
        if (-not $pcaiUrl) { throw 'No BaseUrl' }
        $null = Invoke-RestMethod -Uri "$pcaiUrl/v1/models" -Method Get -TimeoutSec 2 -ErrorAction SilentlyContinue
        Write-Bullet 'pcai-inference: Running' -Color Green
    } catch {
        Write-Bullet 'pcai-inference: Not Running' -Color Yellow
    }

    # Quick system info
    Write-SubHeader 'System'
    $os = Get-CimInstance Win32_OperatingSystem
    $cs = Get-CimInstance Win32_ComputerSystem
    Write-Bullet "OS: $($os.Caption)"
    Write-Bullet "Memory: $([math]::Round($cs.TotalPhysicalMemory / 1GB, 1)) GB"
    Write-Bullet "Free Memory: $([math]::Round($os.FreePhysicalMemory / 1MB, 0)) MB"
}
#endregion


#region Version Command
function Invoke-VersionCommand {
    Write-Host ''
    Write-Host 'PC-AI Framework' -ForegroundColor Cyan
    Write-Host "Version: $script:Version" -ForegroundColor White
    Write-Host ''
    Write-Host 'Local LLM-Powered PC Diagnostics and Optimization'
    Write-Host 'Copyright (c) 2025 PC_AI Project'
    Write-Host ''

    # Module versions
    Write-Host 'Module Versions:' -ForegroundColor DarkGray
    $modules = Get-ChildItem -Path $script:ModulesPath -Directory -ErrorAction SilentlyContinue

    foreach ($module in $modules) {
        $manifest = Join-Path $module.FullName "$($module.Name).psd1"
        if (Test-Path $manifest) {
            try {
                $data = Import-PowerShellDataFile $manifest
                Write-Host "  $($module.Name): $($data.ModuleVersion)" -ForegroundColor DarkGray
            } catch {
                Write-Host "  $($module.Name): Unknown" -ForegroundColor DarkGray
            }
        }
    }
}
#endregion

#endregion

#region Main Entry Point
function Main {
    # Load settings
    $null = Load-Settings

    # Initialize inference backend if requested
    if ($UseNativeInference -or ($InferenceBackend -ne 'auto' -and $InferenceBackend -ne 'http')) {
        Write-Verbose "Inference backend: $InferenceBackend"
        if ($ModelPath) {
            Write-Verbose "Model path: $ModelPath"
        }

        $null = Initialize-InferenceBackend -Backend $InferenceBackend -ModelPath $ModelPath -GpuLayers $GpuLayers
    }

    # Handle empty command
    if (-not $Command) {
        Show-MainHelp
        return
    }

    # Route to command handler
    try {
        switch ($Command) {
            'diagnose' { Invoke-DiagnoseCommand -CmdArgs $Arguments }
            'optimize' { Invoke-OptimizeCommand -CmdArgs $Arguments }
            'usb' { Invoke-UsbCommand -CmdArgs $Arguments }
            'analyze' { Invoke-AnalyzeCommand -CmdArgs $Arguments }
            'chat' { Invoke-ChatCommand -CmdArgs $Arguments }
            'llm' { Invoke-LLMCommand -CmdArgs $Arguments }
            'cleanup' { Invoke-CleanupCommand -CmdArgs $Arguments }
            'perf' { Invoke-PerfCommand -CmdArgs $Arguments }
            'doctor' { Invoke-DoctorCommand -CmdArgs $Arguments }
            'status' { Invoke-StatusCommand }
            'version' { Invoke-VersionCommand }
            'help' {
                $topic = if ($Arguments) { $Arguments[0] } else { $null }
                Show-Help -Topic $topic
            }
            default {
                Write-Error "Unknown command: $Command"
                Show-MainHelp
            }
        }
    } finally {
        # Clean up native inference if initialized
        if ($script:NativeInferenceReady) {
            Write-Verbose 'Cleaning up native inference backend...'
            Close-PcaiInference
        }
    }
}

# Run main function
Main
#endregion
