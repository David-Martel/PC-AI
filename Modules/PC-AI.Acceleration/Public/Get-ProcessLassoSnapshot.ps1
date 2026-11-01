#Requires -Version 7.0
function Get-ProcessLassoSnapshot {
    [CmdletBinding()]
    [OutputType([PSCustomObject])]
    param(
        [Parameter()]
        [string]$ConfigPath = 'C:\ProgramData\ProcessLasso\config\prolasso.ini',

        [Parameter()]
        [string]$LogPath = 'C:\ProgramData\ProcessLasso\logs\processlasso.log',

        [Parameter()]
        [int]$LookbackMinutes = 60
    )

    function Get-ProcessLassoText {
        param([string]$Path)

        try {
            return Get-Content -LiteralPath $Path -Raw -Encoding Unicode -ErrorAction Stop
        } catch {
            return Get-Content -LiteralPath $Path -Raw -ErrorAction Stop
        }
    }

    function Get-CommaList {
        param([AllowNull()][string]$Value)

        if ([string]::IsNullOrWhiteSpace($Value)) {
            return @()
        }

        return @($Value -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ })
    }

    function Get-IniValue {
        param(
            [hashtable]$Sections,
            [string]$Section,
            [string]$Key
        )

        if ($Sections.ContainsKey($Section) -and $Sections[$Section].ContainsKey($Key)) {
            return $Sections[$Section][$Key]
        }

        return $null
    }

    $json = $null
    if (Get-Command Invoke-PcaiNativeProcessLassoSnapshot -ErrorAction SilentlyContinue) {
        try {
            $json = Invoke-PcaiNativeProcessLassoSnapshot -ConfigPath $ConfigPath -LogPath $LogPath -LookbackMinutes $LookbackMinutes
        } catch {}
    }

    if (-not $json) {
        $sections = @{}
        $currentSection = 'global'
        $sections[$currentSection] = @{}

        if (Test-Path -LiteralPath $ConfigPath) {
            $text = Get-ProcessLassoText -Path $ConfigPath
            foreach ($rawLine in ($text -split "`r?`n")) {
                $line = $rawLine.Trim()
                if (-not $line -or $line.StartsWith(';') -or $line.StartsWith('#')) {
                    continue
                }

                if ($line.StartsWith('[') -and $line.EndsWith(']')) {
                    $currentSection = $line.Trim('[', ']')
                    if (-not $sections.ContainsKey($currentSection)) {
                        $sections[$currentSection] = @{}
                    }
                    continue
                }

                $parts = $line -split '=', 2
                if ($parts.Count -eq 2) {
                    $sections[$currentSection][$parts[0].Trim()] = $parts[1].Trim()
                }
            }
        }

        $logSummary = [ordered]@{
            lookback_minutes       = $LookbackMinutes
            total_events           = 0
            efficiency_mode_events = 0
            cpu_set_events         = 0
            smart_trim_events      = 0
            power_profile_events   = 0
            actions                = @{}
            processes              = @{}
        }

        $cutoff = (Get-Date).AddMinutes(-1 * $LookbackMinutes)
        if (Test-Path -LiteralPath $LogPath) {
            foreach ($line in Get-Content -LiteralPath $LogPath -ErrorAction SilentlyContinue) {
                if ([string]::IsNullOrWhiteSpace($line)) {
                    continue
                }

                $trimmed = $line.Trim()
                if ($trimmed.StartsWith('"') -and $trimmed.EndsWith('"')) {
                    $trimmed = $trimmed.Substring(1, $trimmed.Length - 2)
                }

                $fields = $trimmed -split '","'
                if ($fields.Count -lt 9) {
                    continue
                }

                $timestamp = $null
                try {
                    $timestamp = [datetime]::ParseExact($fields[1], 'yyyy-MM-dd HH:mm:ss', [System.Globalization.CultureInfo]::InvariantCulture)
                } catch {
                    continue
                }

                if ($timestamp -lt $cutoff) {
                    continue
                }

                $processName = $fields[5].ToLowerInvariant()
                $action = $fields[7]
                $logSummary.total_events++

                if (-not $logSummary.actions.Contains($action)) {
                    $logSummary.actions[$action] = 0
                }
                $logSummary.actions[$action]++

                if (-not $logSummary.processes.Contains($processName)) {
                    $logSummary.processes[$processName] = 0
                }
                $logSummary.processes[$processName]++

                $lowerAction = $action.ToLowerInvariant()
                if ($lowerAction.Contains('efficiency mode')) { $logSummary.efficiency_mode_events++ }
                if ($lowerAction.Contains('cpu set')) { $logSummary.cpu_set_events++ }
                if ($lowerAction.Contains('smarttrim')) { $logSummary.smart_trim_events++ }
                if ($lowerAction.Contains('power profile')) { $logSummary.power_profile_events++ }
            }
        }

        $defaultPriorities = @{}
        $priorityParts = Get-CommaList (Get-IniValue -Sections $sections -Section 'ProcessDefaults' -Key 'DefaultPriorities')
        for ($i = 0; $i -lt $priorityParts.Count - 1; $i += 2) {
            $defaultPriorities[$priorityParts[$i]] = $priorityParts[$i + 1]
        }

        $efficiencyOff = @()
        $efficiencyParts = Get-CommaList (Get-IniValue -Sections $sections -Section 'ProcessAllowances' -Key 'EfficiencyMode')
        for ($i = 0; $i -lt $efficiencyParts.Count; $i += 2) {
            $efficiencyOff += $efficiencyParts[$i]
        }

        $json = [ordered]@{
            generated_at = (Get-Date).ToString('o')
            config_path  = $ConfigPath
            log_path     = $LogPath
            sections     = $sections
            summary      = [ordered]@{
                start_with_power_plan = Get-IniValue -Sections $sections -Section 'PowerManagement' -Key 'StartWithPowerPlan'
                gaming_mode_enabled   = Get-IniValue -Sections $sections -Section 'GamingMode' -Key 'GamingModeEnabled'
                target_power_plan     = Get-IniValue -Sections $sections -Section 'GamingMode' -Key 'TargetPowerPlan'
                ooc_exclusions        = @(Get-CommaList (Get-IniValue -Sections $sections -Section 'OutOfControlProcessRestraint' -Key 'OocExclusions'))
                smart_trim_exclusions = @(Get-CommaList (Get-IniValue -Sections $sections -Section 'MemoryManagement' -Key 'SmartTrimExclusions'))
                efficiency_mode_off   = @($efficiencyOff)
                default_priorities    = $defaultPriorities
                log_efficiency_mode   = Get-IniValue -Sections $sections -Section 'Logging' -Key 'LogEfficiencyMode'
                log_cpu_sets          = Get-IniValue -Sections $sections -Section 'Logging' -Key 'LogCPUSets'
            }
            log_summary  = $logSummary
        } | ConvertTo-Json -Depth 8
    }

    return $json | ConvertFrom-Json -Depth 8
}
