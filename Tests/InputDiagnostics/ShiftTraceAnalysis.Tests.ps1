#Requires -Version 7.0
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }

BeforeAll {
    $tools = Join-Path $PSScriptRoot '../../Tools/InputDiagnostics'
    $script:Analyzer = Join-Path $tools 'Analyze-ShiftTrace.ps1'
    $script:Ordering = Join-Path $tools 'Measure-ShiftOrdering.ps1'
    function New-Event {
        param([string]$Device = 'private-device-a', [string]$Side = 'LSHIFT', [string]$Direction = 'DOWN', [string]$Class = 'INTERNAL')
        [pscustomobject]@{ dev = $Device; name = $Side; dir = $Direction; cls = $Class; t = '12:00:00.000'; vk = 160 }
    }
    function Write-Trace {
        param([object[]]$Events = @())
        $path = Join-Path $TestDrive ('trace-' + [guid]::NewGuid().ToString('N') + '.jsonl')
        $lines = @($Events | ForEach-Object { $_ | ConvertTo-Json -Compress })
        [IO.File]::WriteAllText($path, ($lines -join [Environment]::NewLine))
        $path
    }
}

Describe 'Conservative offline Shift analysis' {
    It 'counts repeated downs as one observed press and one release' {
        $path = Write-Trace @((New-Event), (New-Event), (New-Event), (New-Event -Direction UP))
        $result = & $script:Analyzer -Path $path -PassThru
        $left = $result.Devices[0].ShiftSides[0]
        $left.DownEvents | Should -Be 3
        $left.ObservedPresses | Should -Be 1
        $left.RepeatedDownEvents | Should -Be 2
        $left.ObservedReleases | Should -Be 1
        $left.HeldAtCaptureEnd | Should -BeFalse
        $result.ActualFailureRate | Should -BeNullOrEmpty
        $result.Diagnosis | Should -Be 'undetermined'
    }

    It 'tracks each side and each device independently even within one class' {
        $path = Write-Trace @(
            (New-Event -Device a), (New-Event -Device a -Side RSHIFT),
            (New-Event -Device b), (New-Event -Device a -Direction UP),
            (New-Event -Device b), (New-Event -Device b -Direction UP)
        )
        $result = & $script:Analyzer -Path $path -PassThru
        $result.DeviceCount | Should -Be 2
        $result.Devices[0].ShiftSides[0].HeldAtCaptureEnd | Should -BeFalse
        $result.Devices[0].ShiftSides[1].HeldAtCaptureEnd | Should -BeTrue
        $result.Devices[1].ShiftSides[0].ObservedPresses | Should -Be 1
        $result.Devices[1].ShiftSides[0].RepeatedDownEvents | Should -Be 1
        $result.Devices[1].ShiftSides[0].HeldAtCaptureEnd | Should -BeFalse
    }

    It 'leaves leading releases and trailing holds ambiguous' {
        $path = Write-Trace @((New-Event -Direction UP), (New-Event), (New-Event))
        $result = & $script:Analyzer -Path $path -PassThru
        $left = $result.Devices[0].ShiftSides[0]
        $left.UnmatchedUpEvents | Should -Be 1
        $left.HeldAtCaptureEnd | Should -BeTrue
        $left.RepeatedDownEvents | Should -Be 1
        $result.Diagnosis | Should -Be 'undetermined'
        ($result.Limitations -join ' ') | Should -Match 'capture boundaries'
    }

    It 'handles an empty capture without claiming healthy input' {
        $path = Write-Trace
        $result = & $script:Analyzer -Path $path -PassThru
        $result.EventCount | Should -Be 0
        $result.DeviceCount | Should -Be 0
        @($result.Devices).Count | Should -Be 0
        $result.Diagnosis | Should -Be 'undetermined'
    }

    It 'handles modifier-only and non-Shift captures without inferring failure rate' {
        $path = Write-Trace @((New-Event -Side LCTRL), (New-Event -Side LCTRL -Direction UP))
        $result = & $script:Analyzer -Path $path -PassThru
        $result.Devices[0].OtherKeyDownEvents | Should -Be 1
        $result.Devices[0].OtherKeyUpEvents | Should -Be 1
        $result.Devices[0].ShiftSides[0].ObservedPresses | Should -Be 0
        $result.ActualFailureRate | Should -BeNullOrEmpty
    }

    It 'emits no private device path, key name, timestamp or arbitrary class text' {
        $path = Write-Trace @((New-Event -Side 'PRIVATE-TYPED-CONTENT' -Class 'PRIVATE-CLASS'))
        $output = & $script:Analyzer -Path $path
        $output | Should -Not -Match 'private-device|PRIVATE-TYPED|PRIVATE-CLASS|12:00:00'
        $result = $output | ConvertFrom-Json
        $result.Devices[0].DeviceLabel | Should -Be 'Device1'
        $result.Devices[0].ReportedClass | Should -Be 'UNKNOWN'
    }

    It 'rejects malformed JSON without repeating its contents' {
        $path = Join-Path $TestDrive 'malformed.jsonl'
        [IO.File]::WriteAllText($path, '{PRIVATE-INVALID-CONTENT')
        $caught = $null
        try { & $script:Analyzer -Path $path -PassThru } catch { $caught = $_ }
        $caught | Should -Not -BeNullOrEmpty
        $caught.Exception.Message | Should -Match 'line 1'
        $caught.Exception.Message | Should -Not -Match 'PRIVATE-INVALID'
    }

    It 'rejects missing identity, invalid direction and non-object records' {
        foreach ($raw in @('{"name":"LSHIFT","dir":"DOWN"}', '{"dev":"a","name":"LSHIFT","dir":"BAD"}', '[{"dev":"a","name":"LSHIFT","dir":"DOWN"}]', '[]', 'null', '42')) {
            $path = Join-Path $TestDrive 'invalid-fields.jsonl'
            [IO.File]::WriteAllText($path, $raw)
            { & $script:Analyzer -Path $path -PassThru } | Should -Throw '*Invalid trace record*'
        }
    }

    It 'keeps ordering compatibility output identical to conservative aggregates' {
        $path = Write-Trace @((New-Event), (New-Event -Direction UP))
        $expected = & $script:Analyzer -Path $path
        $actual = & $script:Ordering -Path $path
        $actual | Should -Be $expected
        (& $script:Ordering -Path $path -PassThru).Analysis | Should -Be 'observed_shift_state'
    }
}
