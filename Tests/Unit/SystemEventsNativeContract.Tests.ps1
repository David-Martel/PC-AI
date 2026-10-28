BeforeAll {
    . (Join-Path $PSScriptRoot '../../Modules/PC-AI.Hardware/Public/Get-SystemEvents.ps1')
    # Synthetic data only; no real event-log read or native library is permitted.
    [xml]$script:fixtureXml = Get-Content -LiteralPath (Join-Path $PSScriptRoot '../Fixtures/HardwareEvent.Inert.xml') -Raw
    $script:fixture = [ordered]@{
        time_created = [string]$script:fixtureXml.Event.System.TimeCreated.SystemTime
        provider_name = [string]$script:fixtureXml.Event.System.Provider.Name
        event_id = [int]$script:fixtureXml.Event.System.EventID
        id = [int]$script:fixtureXml.Event.System.EventID
        level = [int]$script:fixtureXml.Event.System.Level
        level_display = [string]$script:fixtureXml.Event.RenderingInfo.Level
        severity = 'Warning'
        message = ([string]$script:fixtureXml.Event.RenderingInfo.Message -split "`n")[0]
        full_message = [string]$script:fixtureXml.Event.RenderingInfo.Message
    }
    $script:fallback = [PSCustomObject]@{
        TimeCreated = [DateTime]'2000-01-01T00:00:00Z'
        ProviderName = 'disk'
        Id = 7
        Level = 2
        LevelDisplayName = 'Error'
        Message = "Inert fallback`nsecond line"
    }
}

Describe 'Private native event public contract (inert)' {
    BeforeEach {
        Mock Get-HardwareSystemEventsNative { ConvertTo-Json -InputObject @($script:fixture) -Depth 5 }
        Mock Get-WinEvent { @($script:fallback) }
    }

    # Protects: All seven public event fields and precise time.
    # Detects: Missing level/full text or lost timestamp ticks.
    # Needs: Inert native/Get-WinEvent mocks and repository fixtures; no real query or DLL.
    # Breadcrumb: Get-SystemEvents; SystemEventsNativeContract.Tests.ps1 / maps every public field from real-shape synthetic XML values.
    It 'maps every public field from real-shape synthetic XML values' {
        $row = @(Get-SystemEvents)
        $row.Count | Should -Be 1
        $row[0].ProviderName | Should -Be 'Microsoft-Windows-Ntfs'
        $row[0].Id | Should -Be 55
        $row[0].Level | Should -Be 'Warning'
        $row[0].Severity | Should -Be 'Warning'
        $row[0].Message | Should -Be 'Inert fixture first line'
        $row[0].FullMessage | Should -Be "Inert fixture first line`nInert fixture second line"
        $row[0].TimeCreated.ToUniversalTime().Ticks | Should -Be ([DateTimeOffset]::Parse($script:fixture.time_created).UtcDateTime.Ticks)
        Should -Invoke Get-WinEvent -Times 0 -Exactly
    }

    # Protects: Days/MaxEvents native argument transport.
    # Detects: Ignored caller window or caller-side 100-event cap.
    # Needs: Inert native/Get-WinEvent mocks and repository fixtures; no real query or DLL.
    # Breadcrumb: Get-SystemEvents; SystemEventsNativeContract.Tests.ps1 / passes the requested window and count above the former hardcap.
    It 'passes the requested window and count above the former hardcap' {
        Get-SystemEvents -Days 30 -MaxEvents 150 | Out-Null
        Should -Invoke Get-HardwareSystemEventsNative -Times 1 -Exactly -ParameterFilter { $Days -eq 30 -and $MaxEvents -eq 150 }
    }

    # Protects: Successful empty diagnostics.
    # Detects: Unnecessary fallback or empty success treated as failure.
    # Needs: Inert native/Get-WinEvent mocks and repository fixtures; no real query or DLL.
    # Breadcrumb: Get-SystemEvents; SystemEventsNativeContract.Tests.ps1 / accepts a successful empty native array without fallback.
    It 'accepts a successful empty native array without fallback' {
        Mock Get-HardwareSystemEventsNative { '[]' }
        @(Get-SystemEvents).Count | Should -Be 0
        Should -Invoke Get-WinEvent -Times 0 -Exactly
    }

    # Protects: Native failure fallback and full public row.
    # Detects: NULL mistaken for a successful quiet log.
    # Needs: Inert native/Get-WinEvent mocks and repository fixtures; no real query or DLL.
    # Breadcrumb: Get-SystemEvents; SystemEventsNativeContract.Tests.ps1 / falls back when native fails with NULL.
    It 'falls back when native fails with NULL' {
        Mock Get-HardwareSystemEventsNative { $null }
        $row = @(Get-SystemEvents)
        $row[0].Id | Should -Be 7
        $row[0].Level | Should -Be 'Error'
        $row[0].Message | Should -Be 'Inert fallback'
        $row[0].FullMessage | Should -Be "Inert fallback`nsecond line"
        Should -Invoke Get-WinEvent -Times 1 -Exactly
    }

    # Protects: Native exception recovery.
    # Detects: DLL failure swallowed as empty success.
    # Needs: Inert native/Get-WinEvent mocks and repository fixtures; no real query or DLL.
    # Breadcrumb: Get-SystemEvents; SystemEventsNativeContract.Tests.ps1 / falls back on native exceptions without declaring a quiet machine.
    It 'falls back on native exceptions without declaring a quiet machine' {
        Mock Get-HardwareSystemEventsNative { throw 'inert DLL failure' }
        @(Get-SystemEvents)[0].Id | Should -Be 7
        Should -Invoke Get-WinEvent -Times 1 -Exactly
    }

    # Protects: Malformed native-response rejection.
    # Detects: Parser failure preventing the established fallback.
    # Needs: Inert native/Get-WinEvent mocks and repository fixtures; no real query or DLL.
    # Breadcrumb: Get-SystemEvents; SystemEventsNativeContract.Tests.ps1 / falls back on malformed native JSON.
    It 'falls back on malformed native JSON' {
        Mock Get-HardwareSystemEventsNative { '[inert-invalid-json]' }
        @(Get-SystemEvents)[0].Id | Should -Be 7
    }

    # Protects: Current complete-field admission.
    # Detects: Old fabricated/partial native records reaching consumers.
    # Needs: Inert native/Get-WinEvent mocks and repository fixtures; no real query or DLL.
    # Breadcrumb: Get-SystemEvents; SystemEventsNativeContract.Tests.ps1 / rejects the old fabricated native shape.
    It 'rejects the old fabricated native shape' {
        Mock Get-HardwareSystemEventsNative {
            '[{"time_created":"2026-01-31T00:00:00Z","provider_name":"HardwareSource","event_id":0,"level":2,"severity":"Error","message":"pending"}]'
        }
        @(Get-SystemEvents)[0].Id | Should -Be 7
        Should -Invoke Get-WinEvent -Times 1 -Exactly
    }

    # Protects: JSON array contract.
    # Detects: Single object admitted as an event array.
    # Needs: Inert native/Get-WinEvent mocks and repository fixtures; no real query or DLL.
    # Breadcrumb: Get-SystemEvents; SystemEventsNativeContract.Tests.ps1 / rejects a top-level object instead of the array contract.
    It 'rejects a top-level object instead of the array contract' {
        Mock Get-HardwareSystemEventsNative { ConvertTo-Json -InputObject $script:fixture }
        @(Get-SystemEvents)[0].Id | Should -Be 7
    }

    # Protects: Whole native-array validation.
    # Detects: Earlier rows escaping a later malformed entry.
    # Needs: Inert native/Get-WinEvent mocks and repository fixtures; no real query or DLL.
    # Breadcrumb: Get-SystemEvents; SystemEventsNativeContract.Tests.ps1 / rejects bad fields without publishing an earlier valid partial row.
    It 'rejects bad fields without publishing an earlier valid partial row' {
        Mock Get-HardwareSystemEventsNative { ConvertTo-Json -InputObject @($script:fixture, @{ id = 9 }) -Depth 5 }
        $rows = @(Get-SystemEvents)
        $rows.Count | Should -Be 1
        $rows[0].Id | Should -Be 7
    }

    # Protects: Severity/time validity.
    # Detects: Inconsistent level semantics or invalid time admitted.
    # Needs: Inert native/Get-WinEvent mocks and repository fixtures; no real query or DLL.
    # Breadcrumb: Get-SystemEvents; SystemEventsNativeContract.Tests.ps1 / rejects inconsistent severity and invalid timestamp fields.
    It 'rejects inconsistent severity and invalid timestamp fields' {
        Mock Get-HardwareSystemEventsNative {
            $bad = @{} + $script:fixture
            $bad.severity = 'Error'
            $bad.time_created = 'invalid-time'
            ConvertTo-Json -InputObject @($bad)
        }
        @(Get-SystemEvents)[0].Id | Should -Be 7
    }

    # Protects: Informational-level public semantics.
    # Detects: Native levels1-3 path silently omitting requested level4.
    # Needs: Inert native/Get-WinEvent mocks and repository fixtures; no real query or DLL.
    # Breadcrumb: Get-SystemEvents; SystemEventsNativeContract.Tests.ps1 / bypasses the two-argument native ABI for IncludeInfo.
    It 'bypasses the two-argument native ABI for IncludeInfo' {
        Get-SystemEvents -IncludeInfo | Out-Null
        Should -Invoke Get-HardwareSystemEventsNative -Times 0 -Exactly
        Should -Invoke Get-WinEvent -Times 1 -Exactly -ParameterFilter {
            $FilterHashtable.LogName -eq 'System' -and $FilterHashtable.Level.Count -eq 4 -and 4 -in $FilterHashtable.Level
        }
    }

    # Protects: Fallback access-error propagation.
    # Detects: Inaccessible log mislabeled as successful emptiness.
    # Needs: Inert native/Get-WinEvent mocks and repository fixtures; no real query or DLL.
    # Breadcrumb: Get-SystemEvents; SystemEventsNativeContract.Tests.ps1 / holds real fallback failures instead of returning empty success.
    It 'holds real fallback failures instead of returning empty success' {
        Mock Get-HardwareSystemEventsNative { $null }
        Mock Get-WinEvent { throw 'inert Access denied' }
        { Get-SystemEvents -ErrorAction Stop } | Should -Throw '*Access denied*'
    }

    # Protects: Benign no-events handling.
    # Detects: Normal empty selection treated as a hard failure.
    # Needs: Inert native/Get-WinEvent mocks and repository fixtures; no real query or DLL.
    # Breadcrumb: Get-SystemEvents; SystemEventsNativeContract.Tests.ps1 / allows only the benign no-events fallback to return empty.
    It 'allows only the benign no-events fallback to return empty' {
        Mock Get-HardwareSystemEventsNative { $null }
        Mock Get-WinEvent { throw 'No events were found that match the specified selection criteria.' }
        @(Get-SystemEvents).Count | Should -Be 0
    }

    # Protects: Native/fallback CRLF parity across all seven fields.
    # Detects: Trailing CR removed from first line or complete CRLF text changed.
    # Needs: Inert native/Get-WinEvent mocks and repository fixtures; no real query or DLL.
    # Breadcrumb: Get-SystemEvents; SystemEventsNativeContract.Tests.ps1 / preserves CRLF first-line and full-message equality across native and fallback.
    It 'preserves CRLF first-line and full-message equality across native and fallback' {
        $script:crlfJson = Get-Content -LiteralPath (Join-Path $PSScriptRoot '../Fixtures/HardwareEvent.InertCrlf.json') -Raw
        $script:crlfFixture = @($script:crlfJson | ConvertFrom-Json)[0]
        Mock Get-HardwareSystemEventsNative { $script:crlfJson }
        $native = @(Get-SystemEvents)[0]
        $native.Message | Should -Be "Inert CRLF first line`r"
        $native.FullMessage | Should -Be "Inert CRLF first line`r`nInert CRLF second line"
        Should -Invoke Get-WinEvent -Times 0 -Exactly

        Mock Get-HardwareSystemEventsNative { $null }
        Mock Get-WinEvent {
            [PSCustomObject]@{
                TimeCreated = ([DateTimeOffset]$script:crlfFixture.time_created).LocalDateTime
                ProviderName = $script:crlfFixture.provider_name
                Id = [int]$script:crlfFixture.id
                Level = [int]$script:crlfFixture.level
                LevelDisplayName = $script:crlfFixture.level_display
                Message = $script:crlfFixture.full_message
            }
        }
        $fallback = @(Get-SystemEvents)[0]
        foreach ($property in @('TimeCreated', 'ProviderName', 'Id', 'Level', 'Severity', 'Message', 'FullMessage')) {
            $fallback.$property | Should -Be $native.$property
        }
        Should -Invoke Get-WinEvent -Times 1 -Exactly
    }
}
