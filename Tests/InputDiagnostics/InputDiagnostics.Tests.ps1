#Requires -Version 7.0
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }
<#
.SYNOPSIS
    Pester v5 structural tests for the Tools\InputDiagnostics toolkit.

.DESCRIPTION
    Validates that each InputDiagnostics script:
      - Parses without PowerShell syntax errors
      - Exposes comment-based help (.SYNOPSIS)
      - Read-only scripts contain no state-mutating cmdlets (AST check)
      - Mutating scripts declare [CmdletBinding(SupportsShouldProcess)] and -Revert

    All checks are static (ParseFile / Get-Content / Get-Help).
    No script is dot-sourced or invoked.  No -Apply or -Revert paths are exercised.

.NOTES
    Author: input-stack investigation (Claude Code) - 2026-05-30
    Run  : Invoke-Pester -Path .\InputDiagnostics.Tests.ps1
#>

# ---------------------------------------------------------------------------
# BeforeDiscovery: build the script-name arrays at discovery time so
# -ForEach can reference them.  Path resolution happens in BeforeAll (run
# phase) because $PSScriptRoot is not reliable at discovery in all hosts.
# ---------------------------------------------------------------------------
BeforeDiscovery {
    # All six scripts the toolkit specifies.  The 6th (Repair-Workstation...) may
    # not yet exist; per-It guards skip gracefully when absent.
    $script:AllScriptNames = @(
        'Invoke-InputStackDiagnostics.ps1'
        'Repair-InputStackQuickWins.ps1'
        'Start-LoadCapture.ps1'
        'Reset-AccessibilityKeysLive.ps1'
        'Repair-WorkstationInputReliability.ps1'
        'Optimize-StartupLoad.ps1'
    )

    # Subset: scripts that write system state and therefore must declare
    # SupportsShouldProcess and expose a -Revert parameter.
    # Reset-AccessibilityKeysLive and Start-LoadCapture are NOT in this list;
    # they use plain [CmdletBinding()] with read/live-query paths, not -Revert.
    $script:MutatingScriptNames = @(
        'Repair-InputStackQuickWins.ps1'
        'Repair-WorkstationInputReliability.ps1'
        'Optimize-StartupLoad.ps1'
    )
}

BeforeAll {
    # Resolve the Tools\InputDiagnostics directory relative to this test file.
    # This test lives at:  Tests\InputDiagnostics\InputDiagnostics.Tests.ps1
    # The tools live at:   Tools\InputDiagnostics\
    # => two levels up from $PSScriptRoot, then into Tools\InputDiagnostics.
    $script:ToolsDir = Join-Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) `
                                 'Tools\InputDiagnostics'

    # Helper: parse a script file using the PS parser (read-only, no execution).
    # Returns [pscustomobject]@{ Ast; Tokens; Errors }.
    function script:Invoke-ParseScript {
        param([Parameter(Mandatory)] [string]$Path)
        $tokens = $null
        $errors = $null
        $ast = [System.Management.Automation.Language.Parser]::ParseFile(
            $Path, [ref]$tokens, [ref]$errors)
        [pscustomobject]@{
            Ast    = $ast
            Tokens = $tokens
            Errors = $errors
        }
    }

    # Helper: collect all command names from an AST (case-insensitive, unique).
    function script:Get-AstCommandNames {
        param([Parameter(Mandatory)] $Ast)
        $Ast.FindAll({
            param($node)
            $node -is [System.Management.Automation.Language.CommandAst]
        }, $true) |
        ForEach-Object { $_.GetCommandName() } |
        Where-Object   { $_ -and $_ -ne '' } |
        Sort-Object -Unique
    }
}

# ===========================================================================
# Describe 1: Syntax — every present script must parse cleanly
# ===========================================================================
Describe 'InputDiagnostics scripts — syntax' -Tag 'Unit', 'InputDiagnostics', 'Portable' {

    Context 'each script parses without errors' -ForEach (
        $script:AllScriptNames | ForEach-Object { @{ ScriptName = $_ } }
    ) {
        It '<ScriptName> parses with zero syntax errors' {
            $fullPath = Join-Path $script:ToolsDir $ScriptName
            if (-not (Test-Path -LiteralPath $fullPath)) {
                Set-ItResult -Skipped -Because "$ScriptName not yet present at $fullPath"
                return
            }
            $parsed = Invoke-ParseScript -Path $fullPath
            $parsed.Errors.Count | Should -Be 0 -Because (
                "Parser reported errors: " + ($parsed.Errors -join '; ')
            )
        }
    }
}

# ===========================================================================
# Describe 2: Comment-based help — .SYNOPSIS required in every present script
# ===========================================================================
Describe 'InputDiagnostics scripts — comment-based help' -Tag 'Unit', 'InputDiagnostics', 'Portable' {

    Context 'each script exposes .SYNOPSIS via Get-Help' -ForEach (
        $script:AllScriptNames | ForEach-Object { @{ ScriptName = $_ } }
    ) {
        It '<ScriptName> has a non-empty .SYNOPSIS' {
            $fullPath = Join-Path $script:ToolsDir $ScriptName
            if (-not (Test-Path -LiteralPath $fullPath)) {
                Set-ItResult -Skipped -Because "$ScriptName not yet present at $fullPath"
                return
            }
            # Get-Help on a script path parses comment-based help without
            # executing the script body.
            $help = Get-Help $fullPath -ErrorAction SilentlyContinue
            $help | Should -Not -BeNullOrEmpty   -Because 'Get-Help returned nothing'
            $help.Synopsis | Should -Not -BeNullOrEmpty `
                -Because '.SYNOPSIS must be present in comment-based help'
        }
    }

    Context 'each script contains the .SYNOPSIS token in source' -ForEach (
        $script:AllScriptNames | ForEach-Object { @{ ScriptName = $_ } }
    ) {
        It '<ScriptName> source contains .SYNOPSIS' {
            $fullPath = Join-Path $script:ToolsDir $ScriptName
            if (-not (Test-Path -LiteralPath $fullPath)) {
                Set-ItResult -Skipped -Because "$ScriptName not yet present at $fullPath"
                return
            }
            $raw = Get-Content -LiteralPath $fullPath -Raw
            $raw | Should -Match '\.SYNOPSIS' -Because '.SYNOPSIS token required in source'
        }
    }
}

# ===========================================================================
# Describe 3: Read-only constraint — Invoke-InputStackDiagnostics must not
# contain Set-ItemProperty or New-ItemProperty anywhere in its AST.
# (Other diagnostic-only scripts are a separate concern; scope is what the
#  task specifies to keep the assertion precise and maintainable.)
# ===========================================================================
Describe 'InputDiagnostics scripts — read-only enforcement' -Tag 'Unit', 'InputDiagnostics', 'Portable' {

    It 'Invoke-InputStackDiagnostics.ps1 contains no Set-ItemProperty or New-ItemProperty calls' {
        $fullPath = Join-Path $script:ToolsDir 'Invoke-InputStackDiagnostics.ps1'
        if (-not (Test-Path -LiteralPath $fullPath)) {
            Set-ItResult -Skipped -Because 'Invoke-InputStackDiagnostics.ps1 not found'
            return
        }
        $parsed   = Invoke-ParseScript -Path $fullPath
        $commands = Get-AstCommandNames -Ast $parsed.Ast

        # Narrow check: these two cmdlets are the primary registry-mutation verbs.
        # Set-Content (used for report output), New-Item (creates log dir) are
        # intentionally allowed — they operate on files, not system state.
        $commands | Should -Not -Contain 'Set-ItemProperty' `
            -Because 'Invoke-InputStackDiagnostics is strictly read-only'
        $commands | Should -Not -Contain 'New-ItemProperty' `
            -Because 'Invoke-InputStackDiagnostics is strictly read-only'
    }

    It 'Reset-AccessibilityKeysLive.ps1 -DiagnoseOnly path contains no Set-ItemProperty or New-ItemProperty' {
        $fullPath = Join-Path $script:ToolsDir 'Reset-AccessibilityKeysLive.ps1'
        if (-not (Test-Path -LiteralPath $fullPath)) {
            Set-ItResult -Skipped -Because 'Reset-AccessibilityKeysLive.ps1 not found'
            return
        }
        $parsed   = Invoke-ParseScript -Path $fullPath
        $commands = Get-AstCommandNames -Ast $parsed.Ast

        # This script uses SystemParametersInfo (P/Invoke) for live state changes,
        # not registry cmdlets.  Both verbs should be absent in the entire file.
        $commands | Should -Not -Contain 'Set-ItemProperty' `
            -Because 'Reset-AccessibilityKeysLive uses P/Invoke, not registry cmdlets'
        $commands | Should -Not -Contain 'New-ItemProperty' `
            -Because 'Reset-AccessibilityKeysLive uses P/Invoke, not registry cmdlets'
    }
}

# ===========================================================================
# Describe 4: Mutating scripts — must declare SupportsShouldProcess and -Revert
# ===========================================================================
Describe 'InputDiagnostics scripts — mutating-script contract' -Tag 'Unit', 'InputDiagnostics', 'Portable' {

    Context 'each mutating script declares SupportsShouldProcess' -ForEach (
        $script:MutatingScriptNames | ForEach-Object { @{ ScriptName = $_ } }
    ) {
        It '<ScriptName> declares [CmdletBinding(SupportsShouldProcess)]' {
            $fullPath = Join-Path $script:ToolsDir $ScriptName
            if (-not (Test-Path -LiteralPath $fullPath)) {
                Set-ItResult -Skipped -Because "$ScriptName not yet present at $fullPath"
                return
            }
            $raw = Get-Content -LiteralPath $fullPath -Raw
            # Accept both SupportsShouldProcess and SupportsShouldProcess = $true
            $raw | Should -Match 'SupportsShouldProcess' `
                -Because 'mutating scripts must declare SupportsShouldProcess for -WhatIf/-Confirm support'
        }
    }

    Context 'each mutating script exposes a -Revert parameter' -ForEach (
        $script:MutatingScriptNames | ForEach-Object { @{ ScriptName = $_ } }
    ) {
        It '<ScriptName> exposes a [switch]$Revert or -Revert parameter' {
            $fullPath = Join-Path $script:ToolsDir $ScriptName
            if (-not (Test-Path -LiteralPath $fullPath)) {
                Set-ItResult -Skipped -Because "$ScriptName not yet present at $fullPath"
                return
            }
            # AST check: look for a ParameterAst named 'Revert'
            $parsed = Invoke-ParseScript -Path $fullPath
            $revertParam = $parsed.Ast.FindAll({
                param($node)
                $node -is [System.Management.Automation.Language.ParameterAst] -and
                $node.Name.VariablePath.UserPath -eq 'Revert'
            }, $true)
            @($revertParam).Count | Should -BeGreaterThan 0 `
                -Because 'mutating scripts must expose a -Revert path to undo their changes'
        }
    }

    Context 'each mutating script backs up state before applying changes' -ForEach (
        $script:MutatingScriptNames | ForEach-Object { @{ ScriptName = $_ } }
    ) {
        It '<ScriptName> references a backup dir or backup file in its source' {
            $fullPath = Join-Path $script:ToolsDir $ScriptName
            if (-not (Test-Path -LiteralPath $fullPath)) {
                Set-ItResult -Skipped -Because "$ScriptName not yet present at $fullPath"
                return
            }
            $raw = Get-Content -LiteralPath $fullPath -Raw
            # The script must mention a BackupDir/BackupFile/backups pattern —
            # evidence that it persists a pre-change snapshot.
            $raw | Should -Match 'Backup(Dir|File)|backups' `
                -Because 'mutating scripts must write a backup before changing system state'
        }
    }
}

# ===========================================================================
# Describe 5: Safety — no script self-executes when parsed / dot-sourced
#             (structural: all scripts use param() blocks, not bare code at
#              scope level that would fire on parse/import)
# ===========================================================================
Describe 'InputDiagnostics scripts — safe-to-parse' -Tag 'Unit', 'InputDiagnostics', 'Portable' {

    Context 'each script is structured as a param-block script (not a bare runbook)' -ForEach (
        $script:AllScriptNames | ForEach-Object { @{ ScriptName = $_ } }
    ) {
        It '<ScriptName> declares a top-level param() block' {
            $fullPath = Join-Path $script:ToolsDir $ScriptName
            if (-not (Test-Path -LiteralPath $fullPath)) {
                Set-ItResult -Skipped -Because "$ScriptName not yet present at $fullPath"
                return
            }
            $parsed = Invoke-ParseScript -Path $fullPath
            # The script's EndBlock (main body) should have a ParamBlock at the
            # script level, confirming it is parameter-driven.
            $scriptParamBlock = $parsed.Ast.ParamBlock
            $scriptParamBlock | Should -Not -BeNullOrEmpty `
                -Because 'scripts must declare a top-level param() block (prevents accidental execution on dot-source/parse)'
        }
    }

    Context 'each script has the #Requires -Version 7.0 directive' -ForEach (
        $script:AllScriptNames | ForEach-Object { @{ ScriptName = $_ } }
    ) {
        It '<ScriptName> requires PowerShell 7.0' {
            $fullPath = Join-Path $script:ToolsDir $ScriptName
            if (-not (Test-Path -LiteralPath $fullPath)) {
                Set-ItResult -Skipped -Because "$ScriptName not yet present at $fullPath"
                return
            }
            $raw = Get-Content -LiteralPath $fullPath -Raw -ErrorAction SilentlyContinue
            $raw | Should -Match '#Requires\s+-Version\s+7' `
                -Because 'all InputDiagnostics scripts target PowerShell 7+'
        }
    }
}
