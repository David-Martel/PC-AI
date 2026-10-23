BeforeAll {
    $script:Root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
    . (Join-Path $script:Root 'Modules/PC-AI.Acceleration/Private/Invoke-PcaiPerfWorker.ps1')
    . (Join-Path $script:Root 'Modules/PC-AI.Acceleration/Private/Get-PcaiPerfToolPath.ps1')
    . (Join-Path $script:Root 'Modules/PC-AI.Acceleration/Private/Initialize-RustTools.ps1')
    . (Join-Path $script:Root 'Modules/PC-AI.Acceleration/Public/Get-DiskUsageFast.ps1')
    . (Join-Path $script:Root 'Modules/PC-AI.Acceleration/Public/Get-ProcessesFast.ps1')
    . (Join-Path $script:Root 'Modules/PC-AI.Acceleration/Public/Get-FileHashParallel.ps1')
}

Describe 'Process and hash production transport fallback boundaries' {
    BeforeEach {
        $script:SavedEnvironment = @{}
        foreach ($name in @('PCAI_PREFER_RUST_CLI', 'PCAI_DISABLE_PERF_WORKER', 'PCAI_PREFER_RUST_HASHER')) {
            $script:SavedEnvironment[$name] = [Environment]::GetEnvironmentVariable($name)
        }
        $env:PCAI_PREFER_RUST_CLI = '1'
        $env:PCAI_DISABLE_PERF_WORKER = $null
        $env:PCAI_PREFER_RUST_HASHER = '1'
        $script:Failure = $null
        $script:Observed = $null
        $script:CustodyChecks = 0
        Mock Assert-PcaiPerfNoPendingCustody { $script:CustodyChecks++ }
        Mock Invoke-PcaiPerfWorkerRequest { throw $script:Failure }
        Mock Invoke-PcaiPerfCliCommand { throw $script:Failure }
        Mock Get-PcaiPerfToolPath { 'inert-never-launched-tool.exe' }
        Mock Get-RustToolPath { $null }
        Mock Get-ProcessesWithNative { [pscustomobject]@{ Backend = 'unexpected-native' } }
        Mock Get-ProcessesParallel { [pscustomobject]@{ Backend = 'managed-fallback' } }
    }
    AfterEach {
        foreach ($entry in $script:SavedEnvironment.GetEnumerator()) {
            [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value)
        }
    }

    It 'preserves <Kind> through <Wrapper> in the real <Consumer> helper' -ForEach @(
        foreach ($consumer in @('process', 'hash')) {
            foreach ($kind in @('timeout', 'cancellation', 'invalid-frame', 'EOF', 'custody')) {
                foreach ($wrapper in @('direct', 'operation-data')) {
                    @{ Consumer = $consumer; Kind = $kind; Wrapper = $wrapper }
                }
            }
        }
    ) {
        param($Consumer, $Kind, $Wrapper)
        $cause = switch ($Kind) {
            'timeout' { [TimeoutException]::new('Owned transport deadline') }
            'cancellation' { [OperationCanceledException]::new('Owned call cancelled') }
            'invalid-frame' { [IO.InvalidDataException]::new('Invalid correlated frame') }
            'EOF' { [IO.EndOfStreamException]::new('Missing correlated response') }
            'custody' {
                $error = [InvalidOperationException]::new('Unresolved exact owned process')
                $error.Data['PcaiProcessCustody'] = [object]::new()
                $error
            }
        }
        $script:Failure = $cause
        if ($Wrapper -eq 'operation-data') {
            $script:Failure = [IO.FileNotFoundException]::new('Availability wrapper retains original operation')
            $script:Failure.Data['OperationException'] = $cause
        }
        $result = $null
        try {
            $result = if ($Consumer -eq 'process') {
                @(Get-ProcessesWithPcaiPerf -Top 1 -SortBy cpu -ToolPath 'inert-never-launched-tool.exe')
            } else {
                @(Get-FileHashWithPcaiPerf -FilePaths @('owned-input') -Algorithm SHA256 -ToolPath 'inert-never-launched-tool.exe')
            }
        } catch { $script:Observed = $_.Exception }
        [object]::ReferenceEquals($script:Observed, $script:Failure) | Should -BeTrue
        $result | Should -BeNullOrEmpty
        Should -Invoke Invoke-PcaiPerfWorkerRequest -Times 1 -Exactly
        Should -Invoke Invoke-PcaiPerfCliCommand -Times 0 -Exactly
    }

    It 'does not replace a failed public process request with another scan' {
        $script:Failure = [InvalidOperationException]::new('Retained owned resources')
        $script:Failure.Data['PcaiProcessCustody'] = [object]::new()
        try { Get-ProcessesFast -Top 1 -SortBy cpu | Out-Null } catch { $script:Observed = $_.Exception }
        [object]::ReferenceEquals($script:Observed, $script:Failure) | Should -BeTrue
        Should -Invoke Get-ProcessesWithNative -Times 0 -Exactly
        Should -Invoke Get-ProcessesParallel -Times 0 -Exactly
        Should -Invoke Invoke-PcaiPerfCliCommand -Times 0 -Exactly
    }

    It 'stops a public hash batch before any replacement hashing after retained custody' {
        $inputPath = Join-Path $TestDrive 'one-byte.dat'
        [IO.File]::WriteAllBytes($inputPath, [byte[]]@(91))
        $script:Failure = [InvalidOperationException]::new('Retained owned hash resources')
        $script:Failure.Data['PcaiProcessCustody'] = [object]::new()
        Mock Get-FileHash { throw 'Unexpected replacement hash' }
        try { Get-FileHashParallel -Path ([string[]]@($inputPath) * 4096) | Out-Null } catch { $script:Observed = $_.Exception }
        [object]::ReferenceEquals($script:Observed, $script:Failure) | Should -BeTrue
        Should -Invoke Get-FileHash -Times 0 -Exactly
        Should -Invoke Invoke-PcaiPerfWorkerRequest -Times 1 -Exactly
        Should -Invoke Invoke-PcaiPerfCliCommand -Times 0 -Exactly
        [IO.File]::ReadAllBytes($inputPath)[0] | Should -Be 91
    }

    It 'checks preexisting custody before launching the <Consumer> transport' -ForEach @(
        @{ Consumer = 'process' }; @{ Consumer = 'hash' }
    ) {
        param($Consumer)
        $script:Failure = [InvalidOperationException]::new('Existing registry refuses a new transport')
        Mock Assert-PcaiPerfNoPendingCustody { throw $script:Failure }
        try {
            if ($Consumer -eq 'process') {
                Get-ProcessesWithPcaiPerf -Top 1 -SortBy cpu -ToolPath 'inert-never-launched-tool.exe' | Out-Null
            } else {
                Get-FileHashWithPcaiPerf -FilePaths @('owned-input') -Algorithm SHA256 -ToolPath 'inert-never-launched-tool.exe' | Out-Null
            }
        } catch { $script:Observed = $_.Exception }
        [object]::ReferenceEquals($script:Observed, $script:Failure) | Should -BeTrue
        Should -Invoke Invoke-PcaiPerfWorkerRequest -Times 0 -Exactly
        Should -Invoke Invoke-PcaiPerfCliCommand -Times 0 -Exactly
    }

    It 'uses the bounded CLI after genuine legacy negotiation for <Consumer>' -ForEach @(
        @{ Consumer = 'process' }; @{ Consumer = 'hash' }
    ) {
        param($Consumer)
        $script:Failure = [NotSupportedException]::new('Legacy protocol only')
        Mock Invoke-PcaiPerfCliCommand { [pscustomobject]@{ Path = 'owned-input'; Hash = 'fixture-digest'; CPU = 12; Marker = 'direct' } }
        $rows = if ($Consumer -eq 'process') {
            @(Get-ProcessesWithPcaiPerf -Top 1 -SortBy cpu -ToolPath 'inert-never-launched-tool.exe')
        } else {
            @(Get-FileHashWithPcaiPerf -FilePaths @('owned-input') -Algorithm SHA256 -ToolPath 'inert-never-launched-tool.exe')
        }
        $rows.Count | Should -Be 1
        $rows[0].Marker | Should -Be 'direct'
        Should -Invoke Invoke-PcaiPerfCliCommand -Times 1 -Exactly
        $script:CustodyChecks | Should -BeGreaterOrEqual 2
    }

    It 'keeps genuine tool unavailability eligible for a public managed process fallback' {
        $script:Failure = [IO.FileNotFoundException]::new('Selected tool disappeared before launch')
        $rows = @(Get-ProcessesFast -Top 1 -SortBy cpu)
        $rows[0].Backend | Should -Be 'managed-fallback'
        Should -Invoke Get-ProcessesParallel -Times 1 -Exactly
        Should -Invoke Invoke-PcaiPerfCliCommand -Times 0 -Exactly
    }

    It 'does not rescan after a successful zero-row public process response' {
        Mock Invoke-PcaiPerfWorkerRequest { @() }
        $rows = @(Get-ProcessesFast -Top 1 -SortBy cpu)
        $rows.Count | Should -Be 0
        Should -Invoke Get-ProcessesParallel -Times 0 -Exactly
        Should -Invoke Get-ProcessesWithNative -Times 0 -Exactly
        Should -Invoke Invoke-PcaiPerfCliCommand -Times 0 -Exactly
    }

    It 'does not launch a direct hash replacement after an empty worker response' {
        Mock Invoke-PcaiPerfWorkerRequest { @() }
        $rows = @(Get-FileHashWithPcaiPerf -FilePaths @('owned-input') -Algorithm SHA256 -ToolPath 'inert-never-launched-tool.exe')
        $rows.Count | Should -Be 0
        Should -Invoke Invoke-PcaiPerfCliCommand -Times 0 -Exactly
    }

    It 'preserves a direct CLI deadline for <Consumer> with the worker disabled' -ForEach @(
        @{ Consumer = 'process' }; @{ Consumer = 'hash' }
    ) {
        param($Consumer)
        $env:PCAI_DISABLE_PERF_WORKER = '1'
        $script:Failure = [TimeoutException]::new('Existing bounded CLI deadline')
        try {
            if ($Consumer -eq 'process') {
                Get-ProcessesWithPcaiPerf -Top 1 -SortBy cpu -ToolPath 'inert-never-launched-tool.exe' | Out-Null
            } else {
                Get-FileHashWithPcaiPerf -FilePaths @('owned-input') -Algorithm SHA256 -ToolPath 'inert-never-launched-tool.exe' | Out-Null
            }
        } catch { $script:Observed = $_.Exception }
        [object]::ReferenceEquals($script:Observed, $script:Failure) | Should -BeTrue
        Should -Invoke Invoke-PcaiPerfWorkerRequest -Times 0 -Exactly
        Should -Invoke Invoke-PcaiPerfCliCommand -Times 1 -Exactly
    }

    It 'does not treat failed cleanup as legacy negotiation for <Consumer>' -ForEach @(
        @{ Consumer = 'process' }; @{ Consumer = 'hash' }
    ) {
        param($Consumer)
        $script:Failure = [NotSupportedException]::new('Legacy negotiation with retained resources')
        $script:Failure.Data['PcaiProcessCustody'] = [object]::new()
        try {
            if ($Consumer -eq 'process') {
                Get-ProcessesWithPcaiPerf -Top 1 -SortBy cpu -ToolPath 'inert-never-launched-tool.exe' | Out-Null
            } else {
                Get-FileHashWithPcaiPerf -FilePaths @('owned-input') -Algorithm SHA256 -ToolPath 'inert-never-launched-tool.exe' | Out-Null
            }
        } catch { $script:Observed = $_.Exception }
        [object]::ReferenceEquals($script:Observed, $script:Failure) | Should -BeTrue
        Should -Invoke Invoke-PcaiPerfCliCommand -Times 0 -Exactly
    }

    It 'hashes a small real file under strict mode without a native transport' {
        $inputPath = Join-Path $TestDrive 'known abc.dat'
        [IO.File]::WriteAllBytes($inputPath, [byte[]]@(97, 98, 99))
        $rows = @(& {
            Set-StrictMode -Version Latest
            Get-FileHashParallel -Path $inputPath -Algorithm SHA256
        })
        $rows.Count | Should -Be 1
        $rows[0].Success | Should -BeTrue
        $rows[0].Hash | Should -Be 'BA7816BF8F01CFEA414140DE5DAE2223B00361A396177A9CB410FF61F20015AD'
        $rows[0].SizeBytes | Should -Be 3
        Should -Invoke Invoke-PcaiPerfWorkerRequest -Times 0 -Exactly
        Should -Invoke Invoke-PcaiPerfCliCommand -Times 0 -Exactly
        [Convert]::ToHexString([IO.File]::ReadAllBytes($inputPath)) | Should -Be '616263'
    }
}
