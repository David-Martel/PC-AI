# Tests/Integration/FFI.Preflight.Tests.ps1

Describe 'Test-PcaiGpuReadiness' {
    BeforeAll {
        Import-Module "$PSScriptRoot/../../Modules/PC-AI.Gpu" -Force -ErrorAction Stop

        # Detect whether a working preflight backend (FFI or CLI) is available.
        # When neither is present, GPU-specific tests are skipped rather than
        # failing -- the function contract still holds (Source='none', Verdict='fail').
        $script:preflightResult = Test-PcaiGpuReadiness
        $script:backendAvailable = $script:preflightResult.Source -in @('ffi', 'cli')
    }

    Context 'Module availability' {
        It 'Should be exported from PC-AI.Gpu module' {
            Get-Command -Name Test-PcaiGpuReadiness -Module PC-AI.Gpu | Should -Not -BeNullOrEmpty
        }
    }

    Context 'Inventory mode (no model)' {
        It 'Returns a result with Verdict property' {
            $result = Test-PcaiGpuReadiness
            $result.verdict | Should -BeIn @('go', 'warn', 'fail')
        }

        It 'Returns a result with Reason property' {
            $result = Test-PcaiGpuReadiness
            $result.reason | Should -Not -BeNullOrEmpty
        }

        It 'Returns Gpus array' -Skip:(-not $script:backendAvailable) {
            $result = Test-PcaiGpuReadiness
            $result.gpus | Should -Not -BeNullOrEmpty -Because 'at least one GPU should be present on this workstation'
        }

        It 'Returns Source property (ffi or cli)' -Skip:(-not $script:backendAvailable) {
            $result = Test-PcaiGpuReadiness
            $result.Source | Should -BeIn @('ffi', 'cli')
        }

        It 'Returns Source property (ffi, cli, or none)' {
            $result = Test-PcaiGpuReadiness
            $result.Source | Should -BeIn @('ffi', 'cli', 'none')
        }
    }

    Context 'Required MB mode' {
        It 'Returns Go for trivial requirement (1 MB)' -Skip:(-not $script:backendAvailable) {
            $result = Test-PcaiGpuReadiness -RequiredMB 1
            $result.verdict | Should -Be 'go'
        }

        It 'Returns Fail for impossible requirement (999999 MB)' {
            $result = Test-PcaiGpuReadiness -RequiredMB 999999
            $result.verdict | Should -Be 'fail'
        }
    }

    Context 'JSON output mode' {
        It 'Returns valid JSON with -AsJson' {
            $json = Test-PcaiGpuReadiness -AsJson
            { $json | ConvertFrom-Json } | Should -Not -Throw
        }

        It 'JSON contains verdict field' {
            $json = Test-PcaiGpuReadiness -AsJson
            $parsed = $json | ConvertFrom-Json
            $parsed.verdict | Should -BeIn @('go', 'warn', 'fail')
        }
    }

    Context 'GPU process audit' {
        It 'GPU snapshots include process list' -Skip:(-not $script:backendAvailable) {
            $result = Test-PcaiGpuReadiness
            foreach ($gpu in $result.gpus) {
                $gpu.PSObject.Properties.Name | Should -Contain 'processes'
            }
        }
    }

    Context 'No-backend fallback' {
        It 'Returns Source=none when no backend is available' -Skip:$script:backendAvailable {
            $result = Test-PcaiGpuReadiness
            $result.Source | Should -Be 'none'
            $result.Verdict | Should -Be 'fail'
            $result.Reason | Should -BeLike '*not found*'
        }
    }
}
