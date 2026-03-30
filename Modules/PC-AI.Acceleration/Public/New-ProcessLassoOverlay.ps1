#Requires -Version 7.0
function New-ProcessLassoOverlay {
    [CmdletBinding()]
    [OutputType([string])]
    param(
        [Parameter()]
        [string]$ProfilePath = (Join-Path ((Resolve-Path (Join-Path $PSScriptRoot '..\..\..\..\..\..')).Path) 'Config\process-lasso.ai-dev-workstation.json'),

        [Parameter()]
        [string]$OutputPath,

        [Parameter()]
        [switch]$PassThru
    )

    $profile = Get-Content -LiteralPath $ProfilePath -Raw | ConvertFrom-Json -Depth 10
    $classes = $profile.processClasses

    $oocExclusions = [System.Collections.Generic.List[string]]::new()
    $efficiencyMode = [System.Collections.Generic.List[string]]::new()
    $defaultPriorities = [System.Collections.Generic.List[string]]::new()

    foreach ($property in $classes.PSObject.Properties) {
        $class = $property.Value
        $matches = @($class.match)
        $intent = $class.intent

        if ($intent.excludeFromProBalance) {
            foreach ($match in $matches) {
                if ($match -and -not $oocExclusions.Contains($match)) {
                    $oocExclusions.Add($match)
                }
            }
        }

        if ($intent.disableEfficiencyMode) {
            foreach ($match in $matches) {
                if ($match) {
                    $efficiencyMode.Add($match)
                    $efficiencyMode.Add('0')
                }
            }
        }

        if ($intent.defaultPriority) {
            foreach ($match in $matches) {
                if ($match) {
                    $defaultPriorities.Add($match)
                    $defaultPriorities.Add([string]$intent.defaultPriority)
                }
            }
        }
    }

    $lines = @(
        '[OutOfControlProcessRestraint]'
        ('OocExclusions=' + ($oocExclusions -join ','))
        ''
        '[ProcessAllowances]'
        ('EfficiencyMode=' + ($efficiencyMode -join ','))
        ''
        '[ProcessDefaults]'
        ('DefaultPriorities=' + ($defaultPriorities -join ','))
        ''
        '[PowerManagement]'
        'StartWithPowerPlan=Balanced'
    )

    $overlay = ($lines -join [Environment]::NewLine)

    if ($OutputPath) {
        $parent = Split-Path -Path $OutputPath -Parent
        if ($parent -and -not (Test-Path -LiteralPath $parent)) {
            $null = New-Item -ItemType Directory -Path $parent -Force
        }
        Set-Content -LiteralPath $OutputPath -Value $overlay -Encoding UTF8
    }

    if ($PassThru -or -not $OutputPath) {
        return $overlay
    }
}
