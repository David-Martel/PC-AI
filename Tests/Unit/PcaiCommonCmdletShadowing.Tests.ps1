#Requires -Version 7.0
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }

# Why this file exists: PC-AI.Common used to define and export functions named
# Write-Warning and Write-Error that only called Write-Host. Command resolution
# prefers a function over a cmdlet, so every session or module scope that
# imported PC-AI.Common lost the error and warning streams. -ErrorAction Stop,
# $ErrorActionPreference = 'Stop' and try/catch silently stopped working for
# every caller (for example Get-DiskSpace). The module must never again publish
# a command that shadows a built-in cmdlet.
#
# Each check runs in a fresh `pwsh -NoProfile` child process so the result does
# not depend on what earlier test files imported into this Pester session
# (about a dozen test files import PC-AI.Common with -Force).

BeforeAll {
    $script:ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    $script:ModulesRoot = Join-Path $script:ProjectRoot 'Modules'
    $script:CommonModulePath = Join-Path $script:ModulesRoot 'PC-AI.Common\PC-AI.Common.psm1'
    $script:PwshPath = (Get-Process -Id $PID).Path

    # Runs a script block's text in a clean child pwsh and returns the parsed
    # JSON object it writes as its last stdout line. A child that fails to
    # produce JSON fails the test with the child's output, never a silent pass.
    function Invoke-CleanChild {
        param(
            [Parameter(Mandatory)][string]$ScriptText,
            [hashtable]$Replacements = @{}
        )
        $text = $ScriptText
        foreach ($key in $Replacements.Keys) {
            $text = $text.Replace("@@$key@@", [string]$Replacements[$key])
        }
        $scriptPath = Join-Path $TestDrive ("child-{0}.ps1" -f ([guid]::NewGuid().ToString('N')))
        [System.IO.File]::WriteAllText($scriptPath, $text, [System.Text.UTF8Encoding]::new($false))
        $output = & $script:PwshPath -NoProfile -NonInteractive -File $scriptPath 2>&1 | ForEach-Object { [string]$_ }
        $jsonLine = @($output | Where-Object { $_ -like '{*}' }) | Select-Object -Last 1
        if (-not $jsonLine) {
            throw "Child produced no JSON result. Output:`n$($output -join "`n")"
        }
        $jsonLine | ConvertFrom-Json
    }

    $script:ResolutionScript = @'
Import-Module -Name '@@COMMON@@' -Force -ErrorAction Stop
$module = Get-Module -Name 'PC-AI.Common'
$shadowFunctions = @(Get-Command -Name 'Write-Error', 'Write-Warning' -All | Where-Object { $_.CommandType -eq 'Function' })
[pscustomobject]@{
    ModulePath      = $module.Path
    WriteError      = (Get-Command -Name 'Write-Error').CommandType.ToString()
    WriteWarning    = (Get-Command -Name 'Write-Warning').CommandType.ToString()
    WriteInfo       = (Get-Command -Name 'Write-Info').CommandType.ToString()
    ShadowFunctions = @($shadowFunctions | ForEach-Object { $_.Name })
    Exported        = @($module.ExportedFunctions.Keys)
} | ConvertTo-Json -Compress
'@

    # Mirrors the production pattern in PC-AI.Performance: a function imports
    # PC-AI.Common by NAME (so PSModulePath must resolve it) and then reports a
    # failure with the built-in Write-Error.
    $script:InFunctionScript = @'
$env:PSModulePath = '@@MODULES@@' + [System.IO.Path]::PathSeparator + $env:PSModulePath

# Every probe runs INSIDE the function that imported the module: a module
# imported in a function is visible only in that function's scope, which is
# exactly where the production callers use Write-Error and Write-Warning.
function Invoke-ConsumerFunction {
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$Probe)
    Import-Module PC-AI.Common -ErrorAction SilentlyContinue
    switch ($Probe) {
        'ModulePath' {
            $common = Get-Module -Name 'PC-AI.Common'
            if ($common) { return $common.Path } else { return 'NOT-LOADED' }
        }
        'ErrorActionParameter' {
            Write-Error 'consumer failure' -ErrorAction Stop
            return 'REACHED-AFTER-WRITE-ERROR'
        }
        'ErrorActionPreference' {
            $ErrorActionPreference = 'Stop'
            Write-Error 'consumer failure'
            return 'REACHED-AFTER-WRITE-ERROR'
        }
        'ErrorStream' {
            $records = @(Write-Error 'stream probe' -ErrorAction Continue 2>&1)
            if ($records.Count -gt 0) { return $records[0].GetType().Name } else { return 'none' }
        }
        'WarningStream' {
            $records = @(Write-Warning 'stream probe' 3>&1)
            if ($records.Count -gt 0) { return $records[0].GetType().Name } else { return 'none' }
        }
        'WarningPreference' {
            $WarningPreference = 'Stop'
            Write-Warning 'preference probe'
            return 'REACHED-AFTER-WRITE-WARNING'
        }
    }
}

$result = [ordered]@{}
foreach ($probe in 'ModulePath', 'ErrorActionParameter', 'ErrorActionPreference', 'ErrorStream', 'WarningStream', 'WarningPreference') {
    # 6>$null hides the shadow's Write-Host text so only the return value is captured.
    $result[$probe] = try { [string](@(Invoke-ConsumerFunction -Probe $probe 6>$null)[-1]) } catch { 'THREW' }
}
[pscustomobject]$result | ConvertTo-Json -Compress
'@
}

Describe 'PC-AI.Common built-in cmdlet resolution' -Tag 'Unit', 'Portable' {
    Context 'after Import-Module PC-AI.Common in a clean session' {
        BeforeAll {
            $script:Resolution = Invoke-CleanChild -ScriptText $script:ResolutionScript -Replacements @{
                COMMON = $script:CommonModulePath
            }
        }

        It 'imported the PC-AI.Common from this checkout, not an installed copy' {
            # Protects: the other assertions exercise the code under test.
            # Detects: a path or PSModulePath mixup that would let a stale installed copy pass.
            (Resolve-Path -LiteralPath $script:Resolution.ModulePath).Path |
                Should -Be (Resolve-Path -LiteralPath $script:CommonModulePath).Path
        }

        It 'resolves Write-Error to the built-in cmdlet' {
            # Protects: -ErrorAction Stop, $ErrorActionPreference and try/catch keep working for every importer.
            # Detects: a function named Write-Error defined or exported by the module.
            $script:Resolution.WriteError | Should -Be 'Cmdlet'
        }

        It 'resolves Write-Warning to the built-in cmdlet' {
            # Protects: -WarningAction, $WarningPreference and the warning stream keep working for every importer.
            # Detects: a function named Write-Warning defined or exported by the module.
            $script:Resolution.WriteWarning | Should -Be 'Cmdlet'
        }

        It 'exposes no function that shares a name with Write-Error or Write-Warning' {
            # Protects: no scope sees a shadow even where the cmdlet would still win by precedence rules.
            # Detects: the shadow reappearing under -All enumeration or in ExportedFunctions.
            $script:Resolution.ShadowFunctions | Should -BeNullOrEmpty
            $script:Resolution.Exported | Should -Not -Contain 'Write-Error'
            $script:Resolution.Exported | Should -Not -Contain 'Write-Warning'
        }

        It 'still exports the console helpers callers use for informational output' {
            # Protects: positive control. Removing the shadows must not remove the real helpers.
            # Detects: an over-broad edit that dropped Write-Info or its siblings.
            $script:Resolution.WriteInfo | Should -Be 'Function'
            foreach ($name in 'Write-Success', 'Write-Info', 'Write-Header', 'Write-SubHeader', 'Write-Bullet') {
                $script:Resolution.Exported | Should -Contain $name
            }
        }
    }

    Context 'in a function that imports PC-AI.Common by name (the PC-AI.Performance pattern)' {
        BeforeAll {
            $script:Consumer = Invoke-CleanChild -ScriptText $script:InFunctionScript -Replacements @{
                MODULES = $script:ModulesRoot
            }
        }

        It 'imported PC-AI.Common from this checkout (the check is not vacuous)' {
            # Protects: a by-name import that silently resolves nothing would make the throw assertions pass for the wrong reason.
            # Detects: PSModulePath not reaching this checkout, or an installed copy being used instead.
            $script:Consumer.ModulePath | Should -Not -Be 'NOT-LOADED'
            (Resolve-Path -LiteralPath $script:Consumer.ModulePath).Path |
                Should -Be (Resolve-Path -LiteralPath $script:CommonModulePath).Path
        }

        It 'lets Write-Error -ErrorAction Stop throw' {
            # Protects: callers such as Get-DiskSpace can signal failure to try/catch.
            # Detects: the shadow swallowing the error and the function continuing.
            $script:Consumer.ErrorActionParameter | Should -Be 'THREW'
        }

        It 'lets $ErrorActionPreference = Stop turn Write-Error into a terminating error' {
            # Protects: scripts that set the preference once and rely on it.
            # Detects: the shadow ignoring the preference and the function reaching its next statement.
            $script:Consumer.ErrorActionPreference | Should -Be 'THREW'
        }

        It 'writes Write-Error to the error stream and Write-Warning to the warning stream' {
            # Protects: 2>&1 and 3>&1 capture, which the test suites and CI log scrapers use.
            # Detects: a Write-Host shim, which only writes to the information stream.
            $script:Consumer.ErrorStream | Should -Be 'ErrorRecord'
            $script:Consumer.WarningStream | Should -Be 'WarningRecord'
        }

        It 'honors $WarningPreference = Stop for Write-Warning' {
            # Protects: callers that escalate warnings to failures.
            # Detects: a shim that ignores the preference.
            $script:Consumer.WarningPreference | Should -Be 'THREW'
        }
    }
}
