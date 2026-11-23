#Requires -Version 7.0
BeforeAll {
    $source = Join-Path $PSScriptRoot '../Hardware/Invoke-PcaiLlamaDeviceBenchmark.ps1'
    $runnerDir = Join-Path $TestDrive 'Tests/Hardware'
    $helperDir = Join-Path $TestDrive 'Tools/SystemScripts/Networking'
    $null = New-Item -ItemType Directory -Path $runnerDir, $helperDir -Force
    $script:Runner = Join-Path $runnerDir 'Invoke-PcaiLlamaDeviceBenchmark.ps1'
    $script:ReplyPath = Join-Path $helperDir 'replies.json'
    $script:CallPath = Join-Path $helperDir 'calls.jsonl'
    Copy-Item -LiteralPath $source -Destination $script:Runner
    $parseErrors = $null
    $ast = [Management.Automation.Language.Parser]::ParseFile($source, [ref]$null, [ref]$parseErrors)
    if ($parseErrors.Count) { throw ($parseErrors | Out-String) }
    $definition = $ast.Find({
        param($node)
        $node -is [Management.Automation.Language.FunctionDefinitionAst] -and
            $node.Name -eq 'Invoke-LlamaBenchmarkChild'
    }, $true)
    $script:LaunchDefinition = if ($definition) { [scriptblock]::Create($definition.Extent.Text) }
    @'
function Invoke-VigilBoundedProcess {
    param([string]$FilePath, [string]$WorkingDirectory, [string[]]$Arguments, [int]$TimeoutSeconds, [string]$InputText)
    $calls = Join-Path $PSScriptRoot 'calls.jsonl'
    $replies = ConvertFrom-Json -InputObject (Get-Content -LiteralPath (Join-Path $PSScriptRoot 'replies.json') -Raw) -NoEnumerate
    $index = @(Get-Content -LiteralPath $calls).Count
    $parentPrecision = [Environment]::GetEnvironmentVariable('GGML_CUDA_MMQ_PREC')
    $request = if ($InputText) { ConvertFrom-Json -InputObject $InputText -AsHashtable } else {
        @{ filePath = $FilePath; workingDirectory = $WorkingDirectory; arguments = $Arguments; precision = $parentPrecision }
    }
    $entry = @{
        Arguments = $request.arguments; FilePath = $request.filePath; WorkingDirectory = $request.workingDirectory
        Precision = $request.precision; ParentPrecision = $parentPrecision
        LauncherPath = $FilePath; LauncherArguments = $Arguments
    }
    Add-Content -LiteralPath $calls -Value (ConvertTo-Json -InputObject $entry -Compress)
    if ($index -ge $replies.Count) { throw 'Unexpected synthetic native call.' }
    $replies[$index]
}
'@ | Set-Content -LiteralPath (Join-Path $helperDir 'Invoke-VigilBoundedProcess.ps1') -Encoding utf8

    function Set-LlamaFixtureReplies {
        param([object[]]$Replies)
        [IO.File]::WriteAllText($script:ReplyPath, (ConvertTo-Json -InputObject $Replies -Depth 6))
        [IO.File]::WriteAllText($script:CallPath, '')
    }

    function New-LlamaFixtureRows {
        param([string]$Device)
        $layers = if ($Device -eq 'none') { 0 } else { 999 }
        $rows = foreach ($tokens in @(@(256, 0), @(0, 64))) {
            @{
                build_commit = '123456789a'; build_number = 12635; model_filename = $script:Inputs.ModelPath
                devices = $Device; n_gpu_layers = $layers; n_threads = 4
                n_prompt = $tokens[0]; n_gen = $tokens[1]; avg_ts = 200.25
                samples_ts = @(200.0, 200.25, 200.5); samples_ns = @(1000000, 1000001, 1000002)
            }
        }
        ConvertTo-Json -InputObject @($rows) -Depth 4
    }

    $exe = Join-Path $TestDrive 'staged benchmark.exe'
    $gguf = Join-Path $TestDrive 'staged model [A] & é.gguf'
    [IO.File]::WriteAllText($exe, 'inert executable')
    [IO.File]::WriteAllText($gguf, 'inert model')
    $script:Inputs = @{
        BenchPath = $exe; BenchSha256 = (Get-FileHash -LiteralPath $exe).Hash
        ModelPath = $gguf; ModelSha256 = (Get-FileHash -LiteralPath $gguf).Hash
    }
    foreach ($name in 'ggml-vulkan.dll', 'ggml-cuda.dll') {
        [IO.File]::WriteAllText((Join-Path $TestDrive $name), "inert $name sidecar")
    }
    $script:Listing = @(
        'Available devices:'
        '  Vulkan0: NVIDIA RTX Blackwell (8151 MiB, 7043 MiB free)'
        '  Vulkan7: Intel(R) Graphics (65536 MiB, 37035 MiB free)'
        '  CUDA3: NVIDIA RTX Blackwell (8151 MiB, 7043 MiB free)'
    ) -join [char]10
}

Describe 'llama-bench benchmark admission' -Tag 'Unit', 'Portable' {
    BeforeEach {
        $script:SavedPrecision = [Environment]::GetEnvironmentVariable('GGML_CUDA_MMQ_PREC')
        [Environment]::SetEnvironmentVariable('GGML_CUDA_MMQ_PREC', 'ambient-fixture')
        Set-LlamaFixtureReplies -Replies @()
    }

    AfterEach {
        [Environment]::SetEnvironmentVariable('GGML_CUDA_MMQ_PREC', $script:SavedPrecision)
    }

    # Protects: the CPU route requests no GPU and admits both measured slices.
    # Detects: ambient CUDA precision, unnecessary discovery, or wrong arguments.
    # Needs: inert hashed files and a process stub returning benchmark-shaped JSON.
    # Breadcrumb: Tests/Hardware/Invoke-PcaiLlamaDeviceBenchmark.ps1.
    It 'reports both CPU slices with explicit no-device selection' {
        Set-LlamaFixtureReplies -Replies @(@{ ExitCode = 0; Stdout = (New-LlamaFixtureRows 'none'); Stderr = 'offloaded 0/17 layers to GPU' })
        $params = $script:Inputs
        $report = & $script:Runner @params -Route CPU
        $report.PSTypeNames[0] | Should -BeExactly 'Pcai.LlamaDeviceBenchmark'
        $report.Device | Should -BeExactly 'none'
        $report.Prefill.TokensPerSecond | Should -Be 200.25
        $report.Decode.SamplesNanoseconds.Count | Should -Be 3
        $report.OffloadedLayers | Should -Be 0
        [Environment]::GetEnvironmentVariable('GGML_CUDA_MMQ_PREC') | Should -BeExactly 'ambient-fixture'
        $calls = @(Get-Content -LiteralPath $script:CallPath | ConvertFrom-Json)
        $calls.Count | Should -Be 1
        $calls[0].ParentPrecision | Should -BeExactly 'ambient-fixture'
        $calls[0].FilePath | Should -BeExactly $params.BenchPath
        $calls[0].WorkingDirectory | Should -BeExactly (Split-Path -Parent $params.BenchPath)
        $calls[0].Precision | Should -BeNullOrEmpty
        $expected = @('--offline', '-m', $params.ModelPath, '-p', '256', '-n', '64', '-r', '3',
            '-t', '4', '-dev', 'none', '-ngl', '0', '-o', 'json', '-v')
        ($calls[0].Arguments -join [char]0) | Should -BeExactly ($expected -join [char]0)
        ($calls[0].LauncherArguments[0..3] -join ' ') |
            Should -BeExactly '-NoLogo -NoProfile -NonInteractive -EncodedCommand'
    }

    # Protects: GPU qualification uses both enumerated device and observed placement.
    # Detects: hard-coded indices, partial offload acceptance, or a leaked MMQ override.
    # Needs: synthetic multi-vendor device list, runtime logs and measurement JSON.
    # Breadcrumb: Tests/Hardware/Invoke-PcaiLlamaDeviceBenchmark.ps1.
    It 'qualifies <Route> only when reported offload is complete' -ForEach @(
        @{ Route = 'IntelVulkan'; Device = 'Vulkan7'; Description = 'Intel(R) Graphics'; Precision = $null; Sidecar = 'ggml-vulkan.dll' },
        @{ Route = 'NvidiaCuda'; Device = 'CUDA3'; Description = 'NVIDIA RTX Blackwell'; Precision = 'q4'; Sidecar = 'ggml-cuda.dll' }
    ) {
        $probe = @{ ExitCode = 0; Stdout = $script:Listing; Stderr = '  Device 3: NVIDIA RTX Blackwell, compute capability 12.0, VMM: yes' }
        $log = @(
            "using device $Device ($Description) (unknown id) - 7000 MiB free"
            'offloaded 17/17 layers to GPU'
            "$Device model buffer size = 694.76 MiB"
        ) -join [char]10
        $run = @{ ExitCode = 0; Stdout = (New-LlamaFixtureRows $Device); Stderr = $log }
        Set-LlamaFixtureReplies -Replies @($probe, $run)

        $params = $script:Inputs.Clone()
        if ($Precision) { $params.MmqPrecision = $Precision }
        $report = & $script:Runner @params -Route $Route
        $report.Device | Should -BeExactly $Device
        $report.OffloadedLayers | Should -Be 17
        $report.DeviceModelBufferMiB | Should -Be 694.76
        $report.BackendSidecar.FileName | Should -BeExactly $Sidecar
        $report.BackendSidecar.Sha256 | Should -Be (Get-FileHash -LiteralPath (Join-Path $TestDrive $Sidecar)).Hash
        $calls = @(Get-Content -LiteralPath $script:CallPath | ConvertFrom-Json)
        $calls.Count | Should -Be 2
        $calls[0].Arguments | Should -Contain '--list-devices'
        $calls[1].Arguments | Should -Contain $Device
        foreach ($call in $calls) {
            $call.ParentPrecision | Should -BeExactly 'ambient-fixture'
            if ($Precision) { $call.Precision | Should -BeExactly $Precision }
            else { $call.Precision | Should -BeNullOrEmpty }
        }
        if ($Precision) {
            $calls[1].Precision | Should -BeExactly $Precision
            $report.RequestedMmqPrecision | Should -BeExactly $Precision
            $report.CudaComputeCapability | Should -BeExactly '12.0'
        } else {
            $calls[1].Precision | Should -BeNullOrEmpty
            $report.RequestedMmqPrecision | Should -BeNullOrEmpty
        }

        $run.Stderr = $log.Replace('17/17', '1/17')
        Set-LlamaFixtureReplies -Replies @($probe, $run)
        { & $script:Runner @params -Route $Route } | Should -Throw '*complete device and model offload evidence*'
        [Environment]::GetEnvironmentVariable('GGML_CUDA_MMQ_PREC') | Should -BeExactly 'ambient-fixture'
    }

    # Protects: the real Windows launcher scopes MMQ precision to the native workload.
    # Detects: rewriting parent state, mangling argv, recoding diagnostics, or losing the child status.
    # Needs: Windows, pwsh and an inert child in TestDrive. No staged executable or GPU is used.
    # Breadcrumb: Invoke-LlamaBenchmarkChild and Invoke-VigilBoundedProcess.
    It 'preserves literal argv, raw streams and exit code with child-only precision' -Skip:(-not $IsWindows) {
        $script:LaunchDefinition | Should -Not -BeNullOrEmpty
        . $script:LaunchDefinition
        . (Join-Path $PSScriptRoot '../../Tools/SystemScripts/Networking/Invoke-VigilBoundedProcess.ps1')
        $pwshPath = Join-Path $PSHOME 'pwsh.exe'
        $child = Join-Path $TestDrive 'capture args & settings.ps1'
        @'
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
$state = @{
    arguments = @($args)
    precision = [Environment]::GetEnvironmentVariable('GGML_CUDA_MMQ_PREC')
    input = [Console]::In.ReadToEnd()
}
[Console]::Write((ConvertTo-Json -InputObject $state -Compress))
[Console]::Error.Write('inert diagnostics éΩ')
exit 19
'@ | Set-Content -LiteralPath $child -Encoding utf8
        $expected = @('space value', 'a"b', '', 'C:\trailing\', 'éΩ')
        $arguments = @('-NoLogo', '-NoProfile', '-NonInteractive', '-File', $child) + $expected

        foreach ($precision in @('q4', 'q8', $null)) {
            $result = Invoke-LlamaBenchmarkChild -FilePath $pwshPath -WorkingDirectory $TestDrive `
                -Arguments $arguments -Precision $precision -TimeoutSeconds 10
            $result.ExitCode | Should -Be 19
            $result.Stderr | Should -BeExactly 'inert diagnostics éΩ'
            $actual = ConvertFrom-Json -InputObject $result.Stdout -AsHashtable
            ($actual.arguments -join [char]0) | Should -BeExactly ($expected -join [char]0)
            $actual.input | Should -BeExactly ''
            if ($precision) { $actual.precision | Should -BeExactly $precision }
            else { $actual.precision | Should -BeNullOrEmpty }
            [Environment]::GetEnvironmentVariable('GGML_CUDA_MMQ_PREC') | Should -BeExactly 'ambient-fixture'
        }
    }

    # Protects: the bounded launcher still owns the workload after adding a child shell.
    # Detects: a timed-out workload surviving after the helper closes its job.
    # Needs: Windows, pwsh and a sleeping inert child in TestDrive.
    # Breadcrumb: Invoke-LlamaBenchmarkChild and the bounded helper's job cleanup.
    It 'terminates the native workload at the launcher deadline' -Skip:(-not $IsWindows) {
        $script:LaunchDefinition | Should -Not -BeNullOrEmpty
        . $script:LaunchDefinition
        . (Join-Path $PSScriptRoot '../../Tools/SystemScripts/Networking/Invoke-VigilBoundedProcess.ps1')
        $pwshPath = Join-Path $PSHOME 'pwsh.exe'
        $child = Join-Path $TestDrive 'owned delay.ps1'
        $pidPath = Join-Path $TestDrive 'owned delay.pid'
        @'
[IO.File]::WriteAllText($args[0], [string]$PID)
Start-Sleep -Seconds 40
'@ | Set-Content -LiteralPath $child -Encoding utf8
        $arguments = @('-NoLogo', '-NoProfile', '-NonInteractive', '-File', $child, $pidPath)

        {
            Invoke-LlamaBenchmarkChild -FilePath $pwshPath -WorkingDirectory $TestDrive `
                -Arguments $arguments -Precision q4 -TimeoutSeconds 8
        } | Should -Throw '*timed out*process tree terminated*'
        $childId = [int](Get-Content -LiteralPath $pidPath -Raw)
        Get-Process -Id $childId -ErrorAction SilentlyContinue | Should -BeNullOrEmpty
        [Environment]::GetEnvironmentVariable('GGML_CUDA_MMQ_PREC') | Should -BeExactly 'ambient-fixture'
    }

    # Protects: q4 MMQ qualification requires confirmed native-capable CUDA hardware.
    # Detects: benchmarking a pre-Blackwell or unverified device under a q4 label.
    # Needs: inert assets and only a device-discovery reply. Breadcrumb: CUDA capability check.
    It 'rejects q4 before timing with a <Case> CUDA capability' -ForEach @(
        @{ Case = 'pre-Blackwell'; Detail = '  Device 3: NVIDIA RTX Blackwell, compute capability 8.9, VMM: yes' },
        @{ Case = 'missing'; Detail = '' },
        @{ Case = 'wrong-device'; Detail = '  Device 0: NVIDIA RTX Blackwell, compute capability 12.0, VMM: yes' }
    ) {
        Set-LlamaFixtureReplies -Replies @(@{ ExitCode = 0; Stdout = $script:Listing; Stderr = $Detail })
        $params = $script:Inputs
        { & $script:Runner @params -Route NvidiaCuda -MmqPrecision q4 } | Should -Throw '*confirmed CUDA compute capability*'
        @(Get-Content -LiteralPath $script:CallPath).Count | Should -Be 1
        [Environment]::GetEnvironmentVariable('GGML_CUDA_MMQ_PREC') | Should -BeExactly 'ambient-fixture'
    }

    # Protects: artifact and route checks apply before each benchmark or discovery.
    # Detects: checking a pin after launching or allowing CUDA precision on Intel.
    # Needs: two inert staged files and an empty list of native process replies.
    # Breadcrumb: Tests/Hardware/Invoke-PcaiLlamaDeviceBenchmark.ps1.
    It 'blocks either wrong digest and incompatible CUDA precision' {
        foreach ($field in 'BenchSha256', 'ModelSha256') {
            $params = $script:Inputs.Clone()
            $params[$field] = '0' * 64
            { & $script:Runner @params -Route IntelVulkan } | Should -Throw '*SHA256 mismatch*'
        }
        $params = $script:Inputs
        { & $script:Runner @params -Route IntelVulkan -MmqPrecision q4 } | Should -Throw '*only for NvidiaCuda*'
        @(Get-Content -LiteralPath $script:CallPath).Count | Should -Be 0
    }

    # Protects: an accelerator route requires exactly one eligible device.
    # Detects: selecting a wrong vendor or choosing the first of multiple matches.
    # Needs: synthetic device discovery output.
    # Breadcrumb: Tests/Hardware/Invoke-PcaiLlamaDeviceBenchmark.ps1.
    It 'rejects <Case> matching Intel Vulkan devices' -ForEach @(
        @{ Case = 'zero'; ExpectedCount = 0; Records = @('  Vulkan0: NVIDIA RTX (1000 MiB, 500 MiB free)') },
        @{ Case = 'two'; ExpectedCount = 2; Records = @('  Vulkan4: Intel(R) Graphics (1000 MiB, 500 MiB free)', '  Vulkan5: Intel Arc (1000 MiB, 500 MiB free)') }
    ) {
        Set-LlamaFixtureReplies -Replies @(@{ ExitCode = 0; Stdout = ($Records -join [char]10); Stderr = '' })
        $params = $script:Inputs
        { & $script:Runner @params -Route IntelVulkan } | Should -Throw "*found $ExpectedCount*"
        @(Get-Content -LiteralPath $script:CallPath).Count | Should -Be 1
    }

    # Protects: empty measurement arrays fail after a successful native exit.
    # Detects: admitting a child status without two expected measured slices.
    # Needs: an inert synthetic process reply.
    # Breadcrumb: Tests/Hardware/Invoke-PcaiLlamaDeviceBenchmark.ps1.
    It 'rejects a success status with no benchmark rows' {
        Set-LlamaFixtureReplies -Replies @(@{ ExitCode = 0; Stdout = '[]'; Stderr = '' })
        $params = $script:Inputs
        { & $script:Runner @params -Route CPU } | Should -Throw '*two benchmark rows*'
    }
}
