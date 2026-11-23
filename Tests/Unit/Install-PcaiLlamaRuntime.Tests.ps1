#Requires -Version 7.4

BeforeDiscovery {
    $supported = $IsWindows -and [Runtime.InteropServices.RuntimeInformation]::ProcessArchitecture -eq 'X64'
}
BeforeAll {
    $repoRoot = Join-Path $PSScriptRoot '../..'
    $sourceScript = Join-Path $repoRoot 'Tools/Install-PcaiLlamaRuntime.ps1'
    $sourceManifest = Join-Path $repoRoot 'Config/llama-cpp-runtime.json'

    function New-LlamaArchiveFixture {
        param([string]$Path, [System.Collections.IDictionary]$Entries)

        $zip = [IO.Compression.ZipFile]::Open($Path, [IO.Compression.ZipArchiveMode]::Create)
        try {
            foreach ($name in $Entries.Keys) {
                $writer = [IO.StreamWriter]::new($zip.CreateEntry($name).Open())
                try { $writer.Write($Entries[$name]) } finally { $writer.Dispose() }
            }
        } finally {
            $zip.Dispose()
        }
    }

    function Set-LlamaFixturePin {
        param([string]$Id, [string]$Path)
        $script:fixtureManifest.platforms['windows-x64'].assets[$Id].sha256 = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
        $script:fixtureManifest.platforms['windows-x64'].assets[$Id].sizeBytes = (Get-Item -LiteralPath $Path).Length
        $script:fixtureManifest | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $script:fixtureConfig
    }
}

Describe 'Pinned llama.cpp runtime staging' -Tag 'Unit', 'Portable', 'LlamaRuntime' {
    # Protects: the published Windows routes select the exact versioned upstream assets.
    # Detects: a missing CUDA runtime or drift from GitHub b12646 pin and release architecture.
    # Needs: shipped manifest only, on any PowerShell 7.4 host.
    # Breadcrumb: Config/llama-cpp-runtime.json; Tools/Install-PcaiLlamaRuntime.ps1.
    It 'pins all four published Windows x64 ZIP assets' {
        $manifest = Get-Content -LiteralPath $sourceManifest -Raw | ConvertFrom-Json -AsHashtable
        $manifest.version | Should -BeExactly 'b12646'
        $manifest.commit | Should -BeExactly 'f4f6f2e7db76e98c95cc9b7fc608ea669c806044'
        $profile = $manifest.platforms['windows-x64']
        $profile.routes.CPU -join ',' | Should -BeExactly 'cpu'
        $profile.routes.IntelVulkan -join ',' | Should -BeExactly 'vulkan'
        $profile.routes.NvidiaCuda -join ',' | Should -BeExactly 'cuda,cudart'
        $profile.assets.cpu.sha256 | Should -BeExactly 'a1ea7429343ff32fdd5533fe587b6ab88fb05839a281e285afbaf4d97b5d722f'
        $profile.assets.vulkan.sha256 | Should -BeExactly 'a410c2d5b1db2a6c709724da26b1a791ac5960d35baf7041e1ac8177ccd956f5'
        $profile.assets.cuda.sha256 | Should -BeExactly '9625603a500192585c3d272d74dcc3a0008c420c08bc044eb8d8d44eee1a1658'
        $profile.assets.cudart.sha256 | Should -BeExactly 'c2bf4cc202bfbaaa7a0b75f5082457c3811326947bb216059b72839d844e9cba'
    }

    # Protects: platforms lacking an admitted archive handler report Unsupported non-mutatingly.
    # Detects: accidentally treating an unsupported OS or process architecture as Windows x64.
    # Needs: a host outside the Windows x64 support set.
    # Breadcrumb: Tools/Install-PcaiLlamaRuntime.ps1.
    It 'returns structured Unsupported on an unavailable platform' -Skip:$supported {
        $destination = Join-Path $TestDrive 'unsupported-output'
        $result = & $sourceScript -Route IntelVulkan -OutputRoot $destination
        $result.Status | Should -BeExactly 'Unsupported'
        $result.Reason | Should -Not -BeNullOrEmpty
        Test-Path -LiteralPath $destination | Should -BeFalse
    }

    Context 'Windows x64 archive contracts' -Skip:(-not $supported) {
        BeforeEach {
            $fixtureRoot = Join-Path $TestDrive ([guid]::NewGuid().ToString('N'))
            foreach ($name in @('Tools', 'Config', 'cache')) {
                [void][IO.Directory]::CreateDirectory((Join-Path $fixtureRoot $name))
            }
            $installer = Join-Path $fixtureRoot 'Tools/Install-PcaiLlamaRuntime.ps1'
            $script:fixtureConfig = Join-Path $fixtureRoot 'Config/llama-cpp-runtime.json'
            Copy-Item -LiteralPath $sourceScript -Destination $installer
            $script:fixtureManifest = Get-Content -LiteralPath $sourceManifest -Raw | ConvertFrom-Json -AsHashtable
            $script:fixtureManifest | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $script:fixtureConfig
            $profile = $script:fixtureManifest.platforms['windows-x64']
            $cache = Join-Path $fixtureRoot 'cache'
            $output = Join-Path $fixtureRoot 'out'
            $final = Join-Path $output 'b12646/windows-x64/CPU'
            $cpuZip = Join-Path $cache $profile.assets.cpu.name
            Mock Invoke-WebRequest { throw 'Unexpected fixture download.' }
            Mock Start-Process { throw 'Unexpected process launch.' }
        }

        # Protects: DryRun and WhatIf produce identical valid routing without staging or network.
        # Detects: creating a destination or fetching any package during planning.
        # Needs: Windows x64 and a fixture installer/manifest.
        # Breadcrumb: Tools/Install-PcaiLlamaRuntime.ps1.
        It 'previews both modes without writing a destination' {
            $dry = & $installer -Route intelvulkan -OutputRoot $output -DryRun
            $what = & $installer -Route IntelVulkan -OutputRoot $output -WhatIf
            $dry.Status | Should -BeExactly 'DryRun'
            $what.Status | Should -BeExactly 'WhatIf'
            $dry.Destination | Should -BeExactly $what.Destination
            Test-Path -LiteralPath $output | Should -BeFalse
            Should -Invoke Invoke-WebRequest -Exactly -Times 0
        }

        # Protects: verified reusable cache, unique runner discovery, and no replace-on-collision.
        # Detects: launching files, publishing an incomplete runtime, or replacing an occupied route.
        # Needs: Windows x64 and private inert ZIP, with network and process calls mocked.
        # Breadcrumb: Tools/Install-PcaiLlamaRuntime.ps1.
        It 'stages verified cached runners and rejects an occupied destination' {
            New-LlamaArchiveFixture -Path $cpuZip -Entries @{ 'bin/llama-server.exe' = 'inert server'; 'bin/llama-bench.exe' = 'inert bench' }
            Set-LlamaFixturePin -Id cpu -Path $cpuZip

            $installed = & $installer -Route CPU -OutputRoot $output -CacheRoot $cache
            $installed.Status | Should -BeExactly 'Installed'
            $installed.Destination | Should -BeExactly $final
            Get-Content -LiteralPath $installed.ServerPath -Raw | Should -BeExactly 'inert server'
            Get-Content -LiteralPath $installed.BenchmarkPath -Raw | Should -BeExactly 'inert bench'
            $receipt = Get-Content -LiteralPath $installed.ReceiptPath -Raw | ConvertFrom-Json -AsHashtable
            $receipt.BenchmarkRelativePath | Should -BeExactly 'bin/llama-bench.exe'
            $receipt.Binaries.Count | Should -Be 2
            $receipt.Binaries['bin/llama-bench.exe'] | Should -BeExactly (Get-FileHash -LiteralPath $installed.BenchmarkPath).Hash.ToLowerInvariant()
            { & $installer -Route CPU -OutputRoot $output -CacheRoot $cache } | Should -Throw '*already occupied*'
            Get-Content -LiteralPath $installed.ServerPath -Raw | Should -BeExactly 'inert server'
            Should -Invoke Invoke-WebRequest -Exactly -Times 0
            Should -Invoke Start-Process -Exactly -Times 0
        }

        # Protects: exact SHA-256 is mandatory regardless of whether bytes arrive from cache or download.
        # Detects: publishing a different same-length ZIP or bypassing a bad cache with a refresh.
        # Needs: Windows x64 and a private ZIP with intentionally different pinned hash.
        # Breadcrumb: Tools/Install-PcaiLlamaRuntime.ps1.
        It 'rejects a same-length digest mismatch from <Origin> without publishing' -ForEach @(
            @{ Origin = 'cache' },
            @{ Origin = 'download' }
        ) {
            New-LlamaArchiveFixture -Path $cpuZip -Entries @{ 'llama-server.exe' = 'inert server'; 'llama-bench.exe' = 'inert bench' }
            Set-LlamaFixturePin -Id cpu -Path $cpuZip
            $expected = $script:fixtureManifest.platforms['windows-x64'].assets.cpu.sha256
            $different = if ($expected[-1] -ceq '0') { '1' } else { '0' }
            $script:fixtureManifest.platforms['windows-x64'].assets.cpu.sha256 = $expected.Substring(0, 63) + $different
            $script:fixtureManifest | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $script:fixtureConfig
            $params = @{ Route = 'CPU'; OutputRoot = $output }
            if ($Origin -eq 'cache') {
                $params.CacheRoot = $cache
            } else {
                Mock Invoke-WebRequest {
                    $stage = [IO.DirectoryInfo]::new([IO.Path]::GetDirectoryName($OutFile))
                    $fixture = $stage.Parent.Parent.Parent.Parent.FullName
                    $fixtureZip = Join-Path (Join-Path $fixture 'cache') ([IO.Path]::GetFileName($OutFile))
                    [IO.File]::Copy($fixtureZip, $OutFile)
                }
            }
            { & $installer @params } | Should -Throw '*SHA-256 mismatch*'
            Test-Path -LiteralPath $final | Should -BeFalse
            Test-Path -LiteralPath $cpuZip | Should -BeTrue
            $remainingStages = @(Get-ChildItem -LiteralPath (Split-Path $final) -Force -Filter '.stage-*')
            $remainingStages.Count | Should -Be 0
            if ($Origin -eq 'cache') { Should -Invoke Invoke-WebRequest -Exactly -Times 0 }
            else {
                $expectedUrl = "https://github.com/ggml-org/llama.cpp/releases/download/b12646/$($profile.assets.cpu.name)"
                Should -Invoke Invoke-WebRequest -Exactly -Times 1 -ParameterFilter { [string]$Uri -ceq $expectedUrl }
            }
        }

        # Protects: a pinned archive still has to stay inside its extraction root.
        # Detects: accepting a parent path and publishing a traversing ZIP.
        # Needs: an inert digest-pinned ZIP with a crafted entry name, no executable payload.
        # Breadcrumb: Install-PcaiLlamaRuntime.ps1 Expand-VerifiedLlamaZip.
        It 'rejects a pinned ZIP with a parent traversal entry' {
            New-LlamaArchiveFixture -Path $cpuZip -Entries @{
                'llama-server.exe' = 'inert server'; 'llama-bench.exe' = 'inert bench'; '../escaped.dll' = 'inert outside file'
            }
            Set-LlamaFixturePin -Id cpu -Path $cpuZip
            { & $installer -Route CPU -OutputRoot $output -CacheRoot $cache } | Should -Throw '*ZIP entry path*'
            Test-Path -LiteralPath $final | Should -BeFalse
            @(Get-ChildItem -LiteralPath $output -Filter 'escaped.dll' -File -Recurse -Force).Count | Should -Be 0
        }

        # Protects: a new release requires a single server and benchmark before atomic promotion.
        # Detects: ambiguous or incomplete runner discovery or leaked temporary staging.
        # Needs: Windows x64 and two validly pinned inert ZIP layouts.
        # Breadcrumb: Tools/Install-PcaiLlamaRuntime.ps1.
        It 'refuses <Issue> runner layouts after digest validation' -ForEach @(
            @{ Issue = 'missing'; Entries = @{ 'llama-server.exe' = 'inert server' } },
            @{ Issue = 'ambiguous'; Entries = @{ 'llama-server.exe' = 'inert server'; 'llama-bench.exe' = 'inert bench'; 'alt/llama-server.exe' = 'inert alternate' } }
        ) {
            New-LlamaArchiveFixture -Path $cpuZip -Entries $Entries
            Set-LlamaFixturePin -Id cpu -Path $cpuZip
            { & $installer -Route CPU -OutputRoot $output -CacheRoot $cache } | Should -Throw '*one nonempty server*'
            Test-Path -LiteralPath $final | Should -BeFalse
            @(Get-ChildItem -LiteralPath (Split-Path $final) -Force -Filter '.stage-*').Count | Should -Be 0
        }

        # Protects: CUDA packaging includes a separately verified and co-staged runtime asset.
        # Detects: omitting its dependency bundle or publishing after that asset fails admission.
        # Needs: Windows x64 with synthetic files standing in for CUDA payloads and dependencies.
        # Breadcrumb: Config/llama-cpp-runtime.json; Tools/Install-PcaiLlamaRuntime.ps1.
        It 'stages the matched CUDA runtime next to the unique server and benchmark' {
            $cudaZip = Join-Path $cache $profile.assets.cuda.name
            $cudartZip = Join-Path $cache $profile.assets.cudart.name
            New-LlamaArchiveFixture -Path $cudaZip -Entries @{ 'llama-server.exe' = 'inert server'; 'llama-bench.exe' = 'inert bench'; 'ggml-cuda.dll' = 'inert accelerator' }
            New-LlamaArchiveFixture -Path $cudartZip -Entries @{ 'cudart64_13.dll' = 'inert runtime' }
            Set-LlamaFixturePin -Id cuda -Path $cudaZip
            Set-LlamaFixturePin -Id cudart -Path $cudartZip
            $result = & $installer -Route NvidiaCuda -OutputRoot $output -CacheRoot $cache
            $result.Assets.Count | Should -Be 2
            Get-Content -LiteralPath (Join-Path $result.Destination 'cudart64_13.dll') -Raw | Should -BeExactly 'inert runtime'
            Should -Invoke Start-Process -Exactly -Times 0
        }
    }
}
