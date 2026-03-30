# Tests/Integration/FFI.Preflight.Tests.ps1

Describe 'Test-PcaiGpuReadiness' {
    BeforeAll {
        Import-Module "$PSScriptRoot/../../Modules/PC-AI.Gpu" -Force -ErrorAction Stop
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

        It 'Returns Source property (ffi, cli, or none)' {
            $result = Test-PcaiGpuReadiness
            $result.Source | Should -BeIn @('ffi', 'cli', 'none')
        }

        It 'Returns Gpus array when backend available' {
            $result = Test-PcaiGpuReadiness
            if ($result.Source -in @('ffi', 'cli')) {
                $result.gpus | Should -Not -BeNullOrEmpty -Because 'at least one GPU should be present on this workstation'
            } else {
                Set-ItResult -Skipped -Because 'no preflight backend available'
            }
        }
    }

    Context 'Required MB mode' {
        It 'Returns Go for trivial requirement (1 MB)' {
            $result = Test-PcaiGpuReadiness -RequiredMB 1
            if ($result.Source -in @('ffi', 'cli')) {
                $result.verdict | Should -Be 'go'
            } else {
                Set-ItResult -Skipped -Because 'no preflight backend available'
            }
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
        It 'GPU snapshots include process list' {
            $result = Test-PcaiGpuReadiness
            if ($result.Source -in @('ffi', 'cli')) {
                foreach ($gpu in $result.gpus) {
                    $gpu.PSObject.Properties.Name | Should -Contain 'processes'
                }
            } else {
                Set-ItResult -Skipped -Because 'no preflight backend available'
            }
        }
    }
}
