#Requires -Version 7.0
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }

param([string]$CacheSourcePath)

# Windows owns the native capture contract. Portable capture has a separate contract.
Describe 'Process capture refuses invalid work before starting a child' -Tag 'Unit', 'Cache', 'ProcessCapture' -Skip:(-not $IsWindows) {
    BeforeAll {
        $script:CacheSource = if ($CacheSourcePath) { $CacheSourcePath } else {
            Join-Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) 'Modules/PC-AI.Common/Public/Get-PcaiSharedCache.ps1'
        }
        $script:OriginalCacheGlobals = @{}
        foreach ($variableName in @('PcaiSharedCache', 'PcaiSharedCacheProviderState')) {
            $originalVariable = Get-Variable -Name $variableName -Scope Global -ErrorAction SilentlyContinue
            $script:OriginalCacheGlobals[$variableName] = @{
                Present = $null -ne $originalVariable
                Value = if ($originalVariable) { ,$originalVariable.Value } else { $null }
            }
        }
        . $script:CacheSource

        # Mock only the initializer boundary; the complete public wrapper stays real.
        function Initialize-PcaiProcessCaptureRunner {
            throw 'The owned initializer boundary must be mocked before calling capture.'
        }
    }

    BeforeEach {
        $script:OriginalCaptureType = 'Pcai.Common.ProcessCaptureV1' -as [type]
        Mock Initialize-PcaiProcessCaptureRunner {
            throw [InvalidOperationException]::new('owned-prelaunch-initializer-refusal')
        }
    }

    AfterEach {
        # Preserve an ambient loaded type; these tests neither load nor replace it.
        [object]::ReferenceEquals($script:OriginalCaptureType, ('Pcai.Common.ProcessCaptureV1' -as [type])) | Should -BeTrue
    }

    AfterAll {
        foreach ($entry in $script:OriginalCacheGlobals.GetEnumerator()) {
            if ($entry.Value.Present) {
                Set-Variable -Name $entry.Key -Scope Global -Value $entry.Value.Value
            } else {
                Remove-Variable -Name $entry.Key -Scope Global -ErrorAction SilentlyContinue
            }
        }
    }

    # Protects: zero deadlines cannot consume a process slot or initialize native code.
    # Detects: accepted zero budgets or a false timeout/exit-success report before launch.
    # Needs: PowerShell 7 on Windows; an owned missing path, with no executable or backend.
    # Breadcrumb: Modules/PC-AI.Common/Public/Get-PcaiSharedCache.ps1::Invoke-PcaiProcessCapture
    It 'rejects a zero deadline without initializing the runner' {
        $missingPath = Join-Path $TestDrive 'zero-deadline-missing.exe'
        $result = Invoke-PcaiProcessCapture -FilePath $missingPath -TimeoutMs 0
        $result.Success | Should -BeFalse
        $result.ExitCode | Should -Be -3
        $result.TimedOut | Should -BeFalse
        $result.StdOut | Should -BeExactly ''
        $result.StdErr | Should -BeExactly 'TimeoutMs must be positive.'
        $result.Closure | Should -BeExactly 'NOT_STARTED'
        Should -Invoke Initialize-PcaiProcessCaptureRunner -Times 0 -Exactly
    }

    # Protects: negative deadlines are refused before native setup rather than treated as unlimited.
    # Detects: negative budgets reaching initialization or reporting an actual child timeout.
    # Needs: PowerShell 7 on Windows; an owned missing path, with no executable or backend.
    # Breadcrumb: Modules/PC-AI.Common/Public/Get-PcaiSharedCache.ps1::Invoke-PcaiProcessCapture
    It 'rejects a negative deadline without initializing the runner' {
        $missingPath = Join-Path $TestDrive 'negative-deadline-missing.exe'
        $result = Invoke-PcaiProcessCapture -FilePath $missingPath -TimeoutMs -1
        $result.Success | Should -BeFalse
        $result.ExitCode | Should -Be -3
        $result.TimedOut | Should -BeFalse
        $result.StdOut | Should -BeExactly ''
        $result.StdErr | Should -BeExactly 'TimeoutMs must be positive.'
        $result.Closure | Should -BeExactly 'NOT_STARTED'
        Should -Invoke Initialize-PcaiProcessCaptureRunner -Times 0 -Exactly
    }

    # Protects: an unavailable configured executable gives an honest missing-file result.
    # Detects: native initialization for a missing path or confusing absence with a timeout.
    # Needs: PowerShell 7 on Windows; a genuinely absent file beneath the owned TestDrive.
    # Breadcrumb: Modules/PC-AI.Common/Public/Get-PcaiSharedCache.ps1::Invoke-PcaiProcessCapture
    It 'reports a missing owned executable without initializing the runner' {
        $missingPath = Join-Path $TestDrive 'absent-capture.exe'
        Test-Path -LiteralPath $missingPath | Should -BeFalse
        $result = Invoke-PcaiProcessCapture -FilePath $missingPath -TimeoutMs 1200
        $result.Success | Should -BeFalse
        $result.ExitCode | Should -Be -1
        $result.TimedOut | Should -BeFalse
        $result.StdOut | Should -BeExactly ''
        $result.StdErr | Should -BeExactly "File not found: $missingPath"
        $result.Closure | Should -BeExactly 'NOT_STARTED'
        Should -Invoke Initialize-PcaiProcessCaptureRunner -Times 0 -Exactly
    }

    # Protects: initialization failures remain nonthrowing, diagnostic and explicitly unstarted.
    # Detects: lost initializer exceptions, fabricated child results or success after refusal.
    # Needs: PowerShell 7 on Windows; a real inert file and a throwing initializer boundary.
    # Breadcrumb: Modules/PC-AI.Common/Public/Get-PcaiSharedCache.ps1::Invoke-PcaiProcessCapture
    It 'retains an initializer refusal without starting a child or inventing cleanup failures' {
        $markerPath = Join-Path $TestDrive 'inert-capture-marker.txt'
        [IO.File]::WriteAllText($markerPath, 'owned inert marker', [Text.UTF8Encoding]::new($false))
        $results = @(Invoke-PcaiProcessCapture -FilePath $markerPath -TimeoutMs 1200)
        $results.Count | Should -Be 1
        $result = $results[0]
        $result.Success | Should -BeFalse
        $result.ExitCode | Should -Be -3
        $result.TimedOut | Should -BeFalse
        $result.StdOut | Should -BeExactly ''
        $result.Closure | Should -BeExactly 'NOT_STARTED'
        $result.PrimaryFailure | Should -Match 'System\.InvalidOperationException: owned-prelaunch-initializer-refusal'
        $result.StdErr | Should -BeExactly $result.PrimaryFailure
        @($result.SecondaryFailures).Count | Should -Be 0
        Should -Invoke Initialize-PcaiProcessCaptureRunner -Times 1 -Exactly
    }
}
