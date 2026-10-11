BeforeAll {
    $gpuRoot = Join-Path $PSScriptRoot '../../Modules/PC-AI.Gpu'
    $cudaSource = Join-Path $gpuRoot 'Private/Get-CudaVersionFromPath.ps1'
    $registrySource = Join-Path $gpuRoot 'Public/Get-NvidiaSoftwareRegistry.ps1'
    if ($env:PCAI_NVIDIA_METADATA_PREDECESSOR) {
        $cudaSource = Join-Path $env:PCAI_NVIDIA_METADATA_PREDECESSOR 'Get-CudaVersionFromPath.predecessor.ps1'
        $registrySource = Join-Path $env:PCAI_NVIDIA_METADATA_PREDECESSOR 'Get-NvidiaSoftwareRegistry.predecessor.ps1'
    }
    . $cudaSource
    . $registrySource
    $script:MetadataCaseNumber = 0
}

Describe 'CUDA optional JSON component/version contracts' -Tag 'Unit', 'Gpu', 'Portable' {
    BeforeEach {
        $script:MetadataCaseNumber++
        $fixtureBase = $TestDrive
        if ($env:PCAI_NVIDIA_METADATA_FIXTURE_ROOT) { $fixtureBase = $env:PCAI_NVIDIA_METADATA_FIXTURE_ROOT }
        $script:CudaFixture = Join-Path $fixtureBase "case-$script:MetadataCaseNumber/v12.8"
        [IO.Directory]::CreateDirectory($script:CudaFixture) | Out-Null
        [IO.File]::WriteAllText((Join-Path $script:CudaFixture 'version.txt'), 'CUDA Version 12.8.1')
    }

    # Protects: actual supported and arbitrary component JSON version beats stale text.
    # Detects: missing optional cuda/component.version throws under StrictMode into stale fallback.
    # Needs: real JSON/text bytes, the whole actual parser, no filesystem/parser mocks.
    # Breadcrumb: Get-CudaVersionFromPath JSON-first precedence and optional field guards.
    It 'reads a valid JSON version ahead of stale text: <Case>' -ForEach @(
        @{ Case='cuda'; Json='{"cuda":{"version":"13.1.2"}}' },
        @{ Case='cuda_cudart'; Json='{"cuda_cudart":{"version":"13.1.2"}}' },
        @{ Case='arbitrary after missing version'; Json='{"compiler":{"build":"public"},"library":{"version":"13.1.2"}}' },
        @{ Case='arbitrary after null version'; Json='{"compiler":{"version":null},"library":{"version":"13.1.2"}}' },
        @{ Case='null cuda before cudart'; Json='{"cuda":null,"cuda_cudart":{"version":"13.1.2"}}' }
    ) {
        [IO.File]::WriteAllText((Join-Path $script:CudaFixture 'version.json'), $Json)
        Get-CudaVersionFromPath -CudaPath $script:CudaFixture | Should -BeExactly '13.1.2'
        [IO.File]::ReadAllText((Join-Path $script:CudaFixture 'version.json')) | Should -BeExactly $Json
        [IO.File]::ReadAllText((Join-Path $script:CudaFixture 'version.txt')) | Should -BeExactly 'CUDA Version 12.8.1'
    }

    # Protects: known CUDA component priority and tolerant real text fallback remain unchanged.
    # Detects: optional-field repair changes precedence or rejects missing/null/malformed JSON.
    # Needs: actual whole parser on literal JSON files, no parser replacement.
    # Breadcrumb: documented JSON, text, directory fallback order is preserved.
    It 'prefers cuda over cudart and arbitrary components' {
        [IO.File]::WriteAllText((Join-Path $script:CudaFixture 'version.json'), '{"library":{"version":"15.0"},"cuda_cudart":{"version":"14.0"},"cuda":{"version":"13.1.2"}}')
        Get-CudaVersionFromPath -CudaPath $script:CudaFixture | Should -BeExactly '13.1.2'
    }
    It 'retains real text fallback for unavailable JSON version: <Case>' -ForEach @(
        @{ Case='missing version'; Json='{"cuda":{"build":"public"}}' },
        @{ Case='null version'; Json='{"cuda":{"version":null}}' },
        @{ Case='malformed'; Json='{bad json' }
    ) {
        [IO.File]::WriteAllText((Join-Path $script:CudaFixture 'version.json'), $Json)
        Get-CudaVersionFromPath -CudaPath $script:CudaFixture | Should -BeExactly '12.8.1'
    }
}

Describe 'NVIDIA registry optional metadata contracts' -Tag 'Unit', 'Gpu', 'Portable' {
    BeforeEach {
        $script:MetadataCaseNumber++
        $fixtureBase = $TestDrive
        if ($env:PCAI_NVIDIA_METADATA_FIXTURE_ROOT) { $fixtureBase = $env:PCAI_NVIDIA_METADATA_FIXTURE_ROOT }
        $caseRoot = Join-Path $fixtureBase "case-$script:MetadataCaseNumber"
        [IO.Directory]::CreateDirectory($caseRoot) | Out-Null
        $script:RegistryFixture = Join-Path $caseRoot 'registry.json'
    }

    # Protects: metadata omission has ordinary null semantics without losing actual component filtering.
    # Detects: StrictMode missing-property failure in the public registry return object.
    # Needs: actual deserialization/filtering from tiny real JSON, no registry/schema mocks.
    # Breadcrumb: optional return metadata does not authorize or relax a download URL trust check.
    It 'returns null for omitted or explicit-null optional metadata: <Case>' -ForEach @(
        @{ Case='omitted'; Metadata='' },
        @{ Case='null'; Metadata='"trustedSources":null,"categories":null,' }
    ) {
        $json = '{"version":"fixture","lastUpdated":"public",' + $Metadata + '"components":[{"id":"one","category":"runtime"},{"id":"two","category":"driver"}]}'
        [IO.File]::WriteAllText($script:RegistryFixture, $json)
        $result = Get-NvidiaSoftwareRegistry -RegistryPath $script:RegistryFixture -ComponentId one
        $result.TrustedSources | Should -BeNullOrEmpty
        $result.Categories | Should -BeNullOrEmpty
        $result.Components | Should -HaveCount 1
        $result.Components[0].id | Should -BeExactly 'one'
        $result.Version | Should -BeExactly 'fixture'
        $result.LastUpdated | Should -BeExactly 'public'
        [IO.File]::ReadAllText($script:RegistryFixture) | Should -BeExactly $json
    }

    # Protects: supplied metadata and empty filters retain their actual values and cardinality.
    # Detects: repair replaces caller metadata with defaults or mutates original JSON.
    # Needs: real metadata arrays/objects and public filtering on actual bytes.
    # Breadcrumb: returned structure is preserved; trust policy remains a separate consumer boundary.
    It 'preserves present metadata and category filtering' {
        $json = '{"version":"fixture","lastUpdated":"public","trustedSources":["public.nvidia.com","other.public.invalid"],"categories":{"runtime":"Runtime"},"components":[{"id":"one","category":"runtime"},{"id":"two","category":"driver"}]}'
        [IO.File]::WriteAllText($script:RegistryFixture, $json)
        $result = Get-NvidiaSoftwareRegistry -RegistryPath $script:RegistryFixture -Category runtime
        $result.TrustedSources | Should -HaveCount 2
        $result.TrustedSources[0] | Should -BeExactly 'public.nvidia.com'
        $result.TrustedSources[1] | Should -BeExactly 'other.public.invalid'
        $result.Categories.runtime | Should -BeExactly 'Runtime'
        $result.Components | Should -HaveCount 1
        $result.Components[0].id | Should -BeExactly 'one'
        [IO.File]::ReadAllText($script:RegistryFixture) | Should -BeExactly $json
        (Get-NvidiaSoftwareRegistry -RegistryPath $script:RegistryFixture -ComponentId absent).Components | Should -HaveCount 0
    }

    # Protects: supplied metadata arrays retain zero/one/many collection shape.
    # Detects: a conditional scriptblock enumerates or drops the optional metadata value.
    # Needs: actual JSON arrays and public result types, no metadata mocks.
    # Breadcrumb: presence-aware return expressions must preserve array cardinality and type.
    It 'retains the actual metadata array for <Count> sources' -ForEach @(
        @{ Count=0; Sources='[]' },
        @{ Count=1; Sources='["public.nvidia.com"]' },
        @{ Count=2; Sources='["public.nvidia.com","other.public.invalid"]' }
    ) {
        $json = '{"version":"fixture","lastUpdated":"public","trustedSources":' + $Sources + ',"categories":{},"components":[{"id":"one","category":"runtime"}]}'
        [IO.File]::WriteAllText($script:RegistryFixture, $json)
        $result = Get-NvidiaSoftwareRegistry -RegistryPath $script:RegistryFixture
        ($result.TrustedSources -is [array]) | Should -BeTrue
        $result.TrustedSources.Count | Should -Be $Count
        if ($Count -gt 0) { $result.TrustedSources[0] | Should -BeExactly 'public.nvidia.com' }
        $result.Categories.GetType().FullName | Should -BeExactly 'System.Management.Automation.PSCustomObject'
    }

    # Protects: required components and parse/read failures still refuse actual invalid inputs.
    # Detects: optional metadata handling swallows malformed/missing required registry errors.
    # Needs: actual invalid JSON and absent actual file, no error masking mocks.
    # Breadcrumb: only trustedSources/categories omission becomes presence-aware.
    It 'retains required components and malformed JSON failures: <Case>' -ForEach @(
        @{ Case='missing components'; Json='{"version":"fixture","lastUpdated":"public"}' },
        @{ Case='malformed'; Json='{bad json' }
    ) {
        [IO.File]::WriteAllText($script:RegistryFixture, $Json)
        { Get-NvidiaSoftwareRegistry -RegistryPath $script:RegistryFixture } | Should -Throw
    }
    It 'retains missing file failure' {
        { Get-NvidiaSoftwareRegistry -RegistryPath (Join-Path $TestDrive 'absent.json') } | Should -Throw '*not found*'
    }
}
