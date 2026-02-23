#Requires -Version 5.1
#Requires -Modules Pester

. (Join-Path $PSScriptRoot '..\Helpers\Resolve-TestRepoRoot.ps1')

Describe "PC-AI SOLID Modularity (Phase 7)" {
    BeforeAll {
        $script:ModulesRoot = Join-Path (Resolve-TestRepoRoot -StartPath $PSScriptRoot) "Modules"
        $script:AccelerationPath = Join-Path $script:ModulesRoot "PC-AI.Acceleration\PC-AI.Acceleration.psd1"

        # Identify all PC-AI modules to test (excluding non-module directories like 'Archive')
        $script:ModuleDirs = Get-ChildItem -Path $script:ModulesRoot -Directory |
                            Where-Object { $_.Name -match "^PC-AI\." } |
                            Select-Object -ExpandProperty Name
    }

    Context "Module Isolation (with Acceleration for Native)" {
        # Using Pester 5 ForEach pattern
        foreach ($ModuleName in $script:ModuleDirs) {
            It "Module [$ModuleName] should import correctly" {
                $ModulePath = Join-Path $script:ModulesRoot "$ModuleName\$ModuleName.psd1"
                if (-not (Test-Path $ModulePath)) {
                    $ModulePath = Join-Path $script:ModulesRoot "$ModuleName\$ModuleName.psm1"
                }

                # Test environment isolation: Remove module if already loaded
                if (Get-Module $ModuleName) {
                    Remove-Module $ModuleName -ErrorAction SilentlyContinue
                }

                # Acceleration is a mandatory prerequisite for native features in other modules
                if ($ModuleName -ne "PC-AI.Acceleration") {
                    Import-Module $script:AccelerationPath -Force -ErrorAction Stop
                }

                # Special Case: PC-AI.LLM depends on PC-AI.Virtualization for some classes/tools
                if ($ModuleName -eq "PC-AI.LLM") {
                    $VirtPath = Join-Path $script:ModulesRoot "PC-AI.Virtualization\PC-AI.Virtualization.psd1"
                    Import-Module $VirtPath -Force -ErrorAction Stop
                }

                Import-Module $ModulePath -Force -ErrorAction Stop
                Get-Module $ModuleName | Should -Not -BeNull
            }
        }
    }
}
