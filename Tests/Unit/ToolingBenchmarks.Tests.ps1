#Requires -Version 7.0
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }

BeforeDiscovery {
    # The NativeDll Describe below says it "require[s] a built DLL to run" and is
    # tagged 'Windows' rather than 'Portable' -- but a tag is only a label, and
    # nothing in CI filters on it, so the block ran anyway and asserted three
    # backends where a runner without the DLL can only produce 'powershell'.
    # Compute real availability here so -Skip: can act on it at discovery time.
    $script:NativeCoreDll = Join-Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) 'bin\pcai_core_lib.dll'
    $script:HasNativeCore = Test-Path $script:NativeCoreDll
}

BeforeAll {
    $script:ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    $script:BenchmarkScript = Join-Path $script:ProjectRoot 'Tests\Benchmarks\Invoke-PcaiToolingBenchmarks.ps1'
}

Describe "Invoke-PcaiToolingBenchmarks" -Tag 'Unit', 'Benchmarks', 'Acceleration', 'Portable' {
    It 'keeps real legacy matrix timings separate from unqualified native claims' {
        $matrix=Join-Path $script:ProjectRoot 'Modules/PC-AI.Acceleration/Tests/Benchmarks/Measure-PcaiCommandMatrix.ps1'
        $result=& $matrix -Iterations 2 -Warmup 1 -SkipProcesses -SkipDisk -SkipSearch
        $row=$result.FindFiles
        $row.Candidate.Mean|Should -BeGreaterThan 0
        $row.Fallback.Mean|Should -BeGreaterThan 0
        $row.Native|Should -BeNullOrEmpty
        $row.Speedup|Should -BeNullOrEmpty
        $row.ActualBackend|Should -Be @('unreported')
        $row.Qualification.NativeTimingQualified|Should -BeFalse
        $row.Qualification.UsefulOutputParity|Should -Be 'not-established'
        $row.Qualification.SourceSha256|Should -Be (Get-FileHash -LiteralPath $matrix).Hash
        @($row.Observations).Count|Should -Be 3
        foreach($observation in $row.Observations){$observation.RowCount|Should -Be 150}
        $row.Candidate.Name|Should -Not -Match 'Native'
    }
    It 'executes both startup cases through actual isolated child processes and hash-bound receipts' {
        $fixture=Join-Path $TestDrive 'benchmark-startup.ps1'
        "[Console]::WriteLine('matched benchmark startup')"|Set-Content -LiteralPath $fixture -Encoding utf8NoBOM
        $config=Get-Content (Join-Path $script:ProjectRoot 'Config/pcai-tooling-benchmarks.json') -Raw|ConvertFrom-Json
        $pwshPath=(Get-Command pwsh -CommandType Application|Select-Object -First 1).Source
        foreach($case in @($config.cases|Where-Object id -in 'chat-tui-startup','service-host-provider-show')){
            $case.iterations=1;$case.warmup=0
            $case|Add-Member -NotePropertyName commandPath -NotePropertyValue $pwshPath
            $case|Add-Member -NotePropertyName argumentList -NotePropertyValue @('-NoProfile','-File',$fixture)
        }
        $configPath=Join-Path $TestDrive 'benchmark-config.json'
        $config|ConvertTo-Json -Depth 8|Set-Content -LiteralPath $configPath -Encoding utf8NoBOM
        $result=& $script:BenchmarkScript -ConfigPath $configPath -CaseId 'chat-tui-startup','service-host-provider-show' -SkipCapabilities -OutputRoot (Join-Path $TestDrive 'startup') -PassThru
        $rows=@($result.Summary.Results)
        $rows.Count|Should -Be 2
        foreach($row in $rows){
            $row.Backend|Should -Be 'profiled-startup'
            $row.Tool|Should -Be 'ProcessMetrics'
            $row.MeanMs|Should -BeGreaterThan 0
            @($row.Evidence).Count|Should -Be 1
            $receipt=Get-Content @($row.Evidence)[0] -Raw|ConvertFrom-Json
            $receipt.Measurement.ExitCode|Should -Be 0
            $receipt.CommandSha256|Should -Be (Get-FileHash $pwshPath).Hash
            Get-Process -Id $receipt.Measurement.OwnedPid -ErrorAction SilentlyContinue|Should -BeNullOrEmpty
        }
    }
    It "uses the repo-imported acceleration module instead of a shadowed session function" {
        function global:Measure-CommandPerformance {
            param()

            [PSCustomObject]@{
                Name                   = 'shadowed'
                Command                = 'shadowed'
                Mean                   = 1
                StdDev                 = 0
                Min                    = 1
                Max                    = 1
                Median                 = 1
                Iterations             = 1
                Warmup                 = 0
                Unit                   = 'ms'
                Tool                   = 'shadowed'
                WorkingSetDeltaMeanBytes = $null
                WorkingSetDeltaMaxBytes  = $null
                PrivateMemoryDeltaMeanBytes = $null
                PrivateMemoryDeltaMaxBytes  = $null
                ManagedMemoryDeltaMeanBytes = $null
                ManagedMemoryDeltaMaxBytes  = $null
                ManagedAllocatedMeanBytes = $null
                ManagedAllocatedMaxBytes  = $null
            }
        }

        try {
            $result = & $script:BenchmarkScript -CaseId 'runtime-config' -SkipCapabilities -OutputRoot (Join-Path $TestDrive 'runtime') -PassThru
            $report = Get-Content -Path $result.JsonReportPath -Raw -Encoding UTF8 | ConvertFrom-Json
            $row = @($report.Results | Where-Object CaseId -eq 'runtime-config' | Select-Object -First 1)[0]

            $row | Should -Not -BeNullOrEmpty
            $row.Tool | Should -Be 'Measure-Command'
            $row.ManagedAllocatedMeanBytes | Should -Not -BeNullOrEmpty
            $row.ManagedMemoryDeltaMeanBytes | Should -Not -BeNullOrEmpty
        }
        finally {
            Remove-Item -Path Function:\global:Measure-CommandPerformance -ErrorAction SilentlyContinue
        }
    }
}

Describe 'Hash benchmark useful output admission' -Tag 'Unit', 'Benchmarks', 'Acceleration', 'Portable' {
    BeforeAll {
        # Load the real case/validation functions without executing the report runner.
        $tokens = $null; $parseErrors = $null
        $ast = [Management.Automation.Language.Parser]::ParseFile($script:BenchmarkScript, [ref]$tokens, [ref]$parseErrors)
        if ($parseErrors.Count) { throw 'Benchmark fixture source did not parse.' }
        foreach ($name in @('Get-ConfigValue', 'Assert-PcaiHashBenchmarkResults', 'Get-ToolingCaseBenchmarks',
                'Add-BenchmarkMeasurement', 'Invoke-AccelerationModuleCommand', 'Invoke-ImportedModuleCommand')) {
            $definition = $ast.Find({ param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq $name }, $false)
            if (-not $definition) { throw "Production benchmark function is absent: $name" }
            . ([scriptblock]::Create($definition.Extent.Text))
        }
        # Only the clock/transport boundaries are fixtures; real case construction,
        # dataset hashes, closures and output admission execute unchanged.
        function Invoke-BackendBenchmark {
            param($CaseId, $Backend, $Command, $Iterations, $Warmup)
            & $Command
            [pscustomobject]@{CaseId=$CaseId;Backend=$Backend;Iterations=$Iterations;Warmup=$Warmup}
        }
        $script:FixtureAcceleration = New-Module -ScriptBlock {
            $script:Route = ''; $script:Mode = ''; $script:Requests = [Collections.Generic.List[string]]::new()
            function Get-PcaiPerfToolPath { 'fixture-transport' }
            function Get-FixtureHashRows {
                param([string[]]$Paths, [string]$Route)
                $script:Requests.Add($Route)
                $rows = @(foreach ($path in $Paths) {
                    [pscustomobject]@{Path=$path;Hash=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash;Success=$true}
                })
                if ($Route -eq $script:Route) {
                    switch ($script:Mode) {
                        'duplicate' { $rows[1] = $rows[0] }
                        'missing' { $rows = @($rows | Select-Object -SkipLast 1) }
                        'unknown' { $rows[0].Path = Join-Path (Split-Path $rows[0].Path) 'unknown.txt' }
                        'null-hash' { $rows[0].Hash = $null }
                        'empty-hash' { $rows[0].Hash = '' }
                        'wrong-hash' { $rows[0].Hash = '0' * 64 }
                        'wrong-case' { $rows[0].Path = $rows[0].Path.ToUpperInvariant() }
                        'normalized' { $rows[0].Path = Join-Path (Split-Path $rows[0].Path) './sample-0.txt' }
                    }
                }
                [array]::Reverse($rows)
                $rows
            }
            function Invoke-PcaiPerfWorkerRequest {
                param($ToolPath, $Command, $Payload)
                Get-FixtureHashRows -Paths $Payload.paths -Route worker
            }
            function Invoke-PcaiPerfCliCommand {
                param($ToolPath, $Arguments)
                Get-FixtureHashRows -Paths $Arguments[3..($Arguments.Count-1)] -Route CLI
            }
            Export-ModuleMember -Function Get-PcaiPerfToolPath, Invoke-PcaiPerfWorkerRequest, Invoke-PcaiPerfCliCommand
        }
        $script:SavedAcceleration = $script:AccelerationModule
        $script:AccelerationModule = $script:FixtureAcceleration
        $script:CommonModule = $null; $script:CliModule = $null
        $script:HashCase = [pscustomobject]@{id='perf-worker-hash-list';iterations=1;warmup=0}
    }
    BeforeEach {
        $script:BenchmarkReportRoot = Join-Path $TestDrive ([guid]::NewGuid().ToString('N'))
        & $script:FixtureAcceleration { $script:Requests.Clear(); $script:Route=''; $script:Mode='' }
    }
    AfterAll {
        $script:AccelerationModule = $script:SavedAcceleration
        if ($script:FixtureAcceleration) { Remove-Module $script:FixtureAcceleration }
    }
    It 'rejects <Mode> results from the actual <Route> case pipeline' -TestCases @(
        foreach ($route in 'worker', 'CLI') {
            foreach ($mode in 'duplicate', 'missing', 'unknown', 'null-hash', 'empty-hash', 'wrong-hash', 'wrong-case') {
                @{Route=$route;Mode=$mode}
            }
        }
    ) {
        param($Route, $Mode)
        & $script:FixtureAcceleration { param($route,$mode) $script:Route=$route; $script:Mode=$mode } $Route $Mode
        { Get-ToolingCaseBenchmarks -Case $script:HashCase -Defaults ([pscustomobject]@{}) -RepoRoot $TestDrive -Capabilities ([pscustomobject]@{}) -TokenSample 'fixture' } |
            Should -Throw "Hash $Route*"
        $calls = @(& $script:FixtureAcceleration { $script:Requests.ToArray() })
        $calls | Should -Be $(if ($Route -eq 'worker') { @('worker') } else { @('worker','CLI') })
    }
    It 'admits all twelve real hashes in reordered and normalized <Route> results' -TestCases @(@{Route='worker'},@{Route='CLI'}) {
        param($Route)
        & $script:FixtureAcceleration { param($route) $script:Route=$route; $script:Mode='normalized' } $Route
        $result = Get-ToolingCaseBenchmarks -Case $script:HashCase -Defaults ([pscustomobject]@{}) -RepoRoot $TestDrive -Capabilities ([pscustomobject]@{}) -TokenSample 'fixture'
        @($result.Results.Backend) | Should -Be @('worker','direct-cli','powershell')
        @(& $script:FixtureAcceleration { $script:Requests.ToArray() }) | Should -Be @('worker','CLI')
    }
}

# These benchmark cases invoke the native pcai_core_lib.dll (direct-core-probe, content-search).
# Intentionally tagged Windows (not Portable) — require a built DLL to run.
Describe "Invoke-PcaiToolingBenchmarks - NativeDll" -Tag 'Unit', 'Benchmarks', 'Acceleration', 'Windows' -Skip:(-not $script:HasNativeCore) {
    It "records memory metrics for the direct Rust probe case" {
        $result = & $script:BenchmarkScript -CaseId 'direct-core-probe' -SkipCapabilities -OutputRoot (Join-Path $TestDrive 'core') -PassThru
        $report = Get-Content -Path $result.JsonReportPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $row = @($report.Results | Where-Object CaseId -eq 'direct-core-probe' | Select-Object -First 1)[0]

        $row | Should -Not -BeNullOrEmpty
        $row.MeanMs | Should -BeGreaterThan 0
        $row.WorkingSetDeltaMeanBytes | Should -Not -BeNullOrEmpty
        $row.PrivateMemoryDeltaMeanBytes | Should -Not -BeNullOrEmpty
        $row.ManagedAllocatedMeanBytes | Should -Not -BeNullOrEmpty
    }

    It "emits content-search rows for all expected backends with memory metrics" {
        $result = & $script:BenchmarkScript -CaseId 'content-search' -SkipCapabilities -OutputRoot (Join-Path $TestDrive 'content') -PassThru
        $report = Get-Content -Path $result.JsonReportPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $rows = @($report.Results | Where-Object CaseId -eq 'content-search')

        @($rows | Select-Object -ExpandProperty Backend | Sort-Object) | Should -Be @('accelerated', 'native', 'powershell')
        foreach ($row in $rows) {
            $row.WorkingSetDeltaMeanBytes | Should -Not -BeNullOrEmpty
            $row.PrivateMemoryDeltaMeanBytes | Should -Not -BeNullOrEmpty
            $row.ManagedAllocatedMeanBytes | Should -Not -BeNullOrEmpty
        }
    }

    It "emits a shared-cache benchmark row with memory metrics" {
        $result = & $script:BenchmarkScript -CaseId 'shared-cache-hit' -SkipCapabilities -OutputRoot (Join-Path $TestDrive 'cache') -PassThru
        $report = Get-Content -Path $result.JsonReportPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $row = @($report.Results | Where-Object CaseId -eq 'shared-cache-hit' | Select-Object -First 1)[0]

        $row | Should -Not -BeNullOrEmpty
        $row.Backend | Should -Be 'powershell'
        $row.MeanMs | Should -BeGreaterThan 0
        $row.ManagedAllocatedMeanBytes | Should -Not -BeNullOrEmpty
    }

    It "emits an external-cache-status benchmark row" {
        $result = & $script:BenchmarkScript -CaseId 'external-cache-status' -SkipCapabilities -OutputRoot (Join-Path $TestDrive 'external') -PassThru
        $report = Get-Content -Path $result.JsonReportPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $row = @($report.Results | Where-Object CaseId -eq 'external-cache-status' | Select-Object -First 1)[0]

        $row | Should -Not -BeNullOrEmpty
        $row.Backend | Should -Be 'powershell'
        $row.MeanMs | Should -BeGreaterThan 0
    }

    It "emits an acceleration-stack benchmark row" {
        $result = & $script:BenchmarkScript -CaseId 'acceleration-stack' -SkipCapabilities -OutputRoot (Join-Path $TestDrive 'stack') -PassThru
        $report = Get-Content -Path $result.JsonReportPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $row = @($report.Results | Where-Object CaseId -eq 'acceleration-stack' | Select-Object -First 1)[0]

        $row | Should -Not -BeNullOrEmpty
        $row.Backend | Should -Be 'powershell'
        $row.MeanMs | Should -BeGreaterThan 0
    }
}
