#Requires -Version 7.0

BeforeAll {
    $script:WorkflowText = Get-Content -LiteralPath (Join-Path $PSScriptRoot '../../.github/workflows/release-cuda.yml') -Raw
    $script:SourceCommit = '0123456789abcdef0123456789abcdef01234567'

    function Invoke-ReleaseSourceCheck {
        param(
            [string]$Event,
            [string]$Ref,
            [string]$RefType,
            [string]$RefName,
            [string]$CheckoutRef = $Ref,
            [string]$CheckoutCommit = $script:SourceCommit
        )

        $pattern = '(?ms)^      - name: Verify Source Revision\r?\n.*?^        run: \|\r?\n(?<body>.*?)(?=^      - |\z)'
        $body = [regex]::Match($script:WorkflowText, $pattern).Groups['body'].Value
        if (-not $body) { throw 'Missing native release source verification step.' }
        $check = [scriptblock]::Create(($body -replace '(?m)^          ', ''))
        $outputPath = Join-Path $TestDrive 'github-output'
        Remove-Item -LiteralPath $outputPath -ErrorAction SilentlyContinue
        $values = @{
            GITHUB_EVENT_NAME = $Event
            GITHUB_REF = $Ref
            GITHUB_REF_TYPE = $RefType
            GITHUB_REF_NAME = $RefName
            GITHUB_SHA = $script:SourceCommit
            GITHUB_OUTPUT = $outputPath
            PCAI_CHECKOUT_REF = $CheckoutRef
            PCAI_CHECKOUT_COMMIT = $CheckoutCommit
        }
        $prior = @{}
        try {
            foreach ($name in $values.Keys) {
                $prior[$name] = [Environment]::GetEnvironmentVariable($name)
                [Environment]::SetEnvironmentVariable($name, $values[$name])
            }
            & $check
            Get-Content -LiteralPath $outputPath -ErrorAction Stop
        }
        finally {
            foreach ($name in $prior.Keys) {
                [Environment]::SetEnvironmentVariable($name, $prior[$name])
            }
        }
    }
}

Describe 'Native release workflow source and permissions' -Tag 'Unit', 'Portable' {
    # Protects: Both stable and prerelease artifacts identify the exact pushed tag and checked-out commit.
    # Detects: Source admission drops the ref, treats the tag name as a different version, or loses the exact SHA.
    # Needs: PowerShell 7, Pester 5, workflow text, and TestDrive. No GitHub or network access.
    # Breadcrumb: .github/workflows/release-cuda.yml Verify Source Revision.
    It 'accepts a checked-out tag and source commit for <Tag>' -ForEach @(
        @{ Tag = 'v1.2.3' },
        @{ Tag = 'v1.2.3-rc.2' }
    ) {
        $ref = "refs/tags/$Tag"
        $output = Invoke-ReleaseSourceCheck -Event push -Ref $ref -RefType tag -RefName $Tag
        $output | Should -Contain "source_ref=$ref"
        $output | Should -Contain "source_commit=$script:SourceCommit"
        $output | Should -Contain "release_tag=$Tag"
        $output | Should -Contain "release_version=$($Tag.Substring(1))"
    }

    # Protects: Only an admitted release tag whose ref and checkout commit match can declare publication inputs.
    # Detects: Commit mismatches, ref mismatches, branch pushes and malformed stable or prerelease versions.
    # Needs: PowerShell 7, Pester 5, workflow text, and TestDrive. No GitHub or network access.
    # Breadcrumb: .github/workflows/release-cuda.yml Verify Source Revision.
    It 'refuses publication inputs when <Case>' -ForEach @(
        @{ Case = 'the checkout commit differs'; Override = @{ CheckoutCommit = '1111111111111111111111111111111111111111' }; ErrorText = '*event source commit*' },
        @{ Case = 'the checkout tag differs'; Override = @{ CheckoutRef = 'refs/tags/v1.2.4' }; ErrorText = '*release tag and checkout*' },
        @{ Case = 'the push is to a branch'; Override = @{ Ref = 'refs/heads/v1.2.3'; RefType = 'branch' }; ErrorText = '*release tag and checkout*' },
        @{ Case = 'the tag contains a leading zero'; Override = @{ Ref = 'refs/tags/v01.2.3'; RefName = 'v01.2.3' }; ErrorText = '*release tag and checkout*' },
        @{ Case = 'a numeric prerelease identifier contains a leading zero'; Override = @{ Ref = 'refs/tags/v1.2.3-rc.01'; RefName = 'v1.2.3-rc.01' }; ErrorText = '*release tag and checkout*' }
    ) {
        $params = @{ Event = 'push'; Ref = 'refs/tags/v1.2.3'; RefType = 'tag'; RefName = 'v1.2.3' }
        foreach ($name in $Override.Keys) { $params[$name] = $Override[$name] }
        { Invoke-ReleaseSourceCheck @params } | Should -Throw -ExpectedMessage $ErrorText
        Test-Path -LiteralPath (Join-Path $TestDrive 'github-output') | Should -BeFalse
    }

    # Protects: Selecting an existing tag during a manual dispatch yields a build-only source SHA.
    # Detects: A manual run supplies any tag or version output to the publication job.
    # Needs: PowerShell 7, Pester 5, workflow text, and TestDrive. No GitHub or network access.
    # Breadcrumb: .github/workflows/release-cuda.yml Verify Source Revision and release job condition.
    It 'admits a manual build without emitting a publication tag' {
        $output = Invoke-ReleaseSourceCheck -Event workflow_dispatch -Ref 'refs/tags/v1.2.3' -RefType tag -RefName 'v1.2.3'
        $output | Should -Contain "source_ref=$script:SourceCommit"
        $output | Should -Contain "source_commit=$script:SourceCommit"
        @($output -match '^release_').Count | Should -Be 0
    }

    # Protects: Only a successful tag build can write a release, with matrix builds using the admitted ref.
    # Detects: Workflow-wide write permissions, a release job callable by dispatch, omitted checkout pinning or missing prerequisites.
    # Needs: PowerShell 7, Pester 5, and workflow text. No GitHub or network access.
    # Breadcrumb: .github/workflows/release-cuda.yml test, build and release jobs.
    It 'limits write access and publication to the checked tag after test and build' {
        $script:WorkflowText | Should -Match '(?m)^permissions: \{\}\s*$'
        $writes = [regex]::Matches($script:WorkflowText, '(?m)^\s*contents: write\s*$')
        $writes.Count | Should -Be 1
        $forbiddenInput = [regex]::Matches($script:WorkflowText, 'github\.event\.inputs\.tag')
        $forbiddenInput.Count | Should -Be 0
        $build = [regex]::Match($script:WorkflowText, '(?ms)^  build:\r?\n(?<job>.*?)(?=^  release:)').Groups['job'].Value
        $release = [regex]::Match($script:WorkflowText, '(?ms)^  release:\r?\n(?<job>.*)$').Groups['job'].Value
        $build | Should -Match '(?m)^      contents: read\s*$'
        $build | Should -Match '(?m)^          ref: \$\{\{ needs\.test\.outputs\.source_ref \}\}\s*$'
        $build | Should -Match '(?m)^        if: steps\.source_checkout\.outputs\.commit != needs\.test\.outputs\.source_commit\s*$'
        $release | Should -Match "(?m)^    if: github\.event_name == 'push' && github\.ref_type == 'tag' && needs\.test\.outputs\.release_tag != ''\s*$"
        $release | Should -Match '(?m)^    needs: \[test, build\]\s*$'
        $release | Should -Match '(?m)^      contents: write\s*$'
        $release | Should -Match '(?m)^          ref: \$\{\{ needs\.test\.outputs\.source_ref \}\}\s*$'
        $release | Should -Match '(?m)^        if: steps\.source_checkout\.outputs\.commit != needs\.test\.outputs\.source_commit\s*$'
        $release | Should -Match '(?m)^          tag_name: \$\{\{ needs\.test\.outputs\.release_tag \}\}\s*$'
    }
}

Describe 'Native release workflow artifact staging' -Tag 'Unit', 'Portable' {
    BeforeAll {
        function Invoke-PrepareNativeArtifact {
            param(
                [string]$Root,
                [string]$Backend,
                [bool]$Cuda,
                [string]$MetadataJson,
                [int]$MetadataExit = 0
            )

            if (-not $PSBoundParameters.ContainsKey('MetadataJson')) {
                $features = @($Backend, 'ffi', 'server')
                if ($Cuda) { $features += "cuda-$Backend" }
                $MetadataJson = @{
                    git_hash = $script:SourceCommit
                    features = $features -join ','
                    target = 'x86_64-pc-windows-msvc'
                    profile = 'release'
                } |
                    ConvertTo-Json -Compress
            }

            function Invoke-ArtifactMetadataFixture {
                param([string]$ProbePath)
                $expectedPath = Join-Path $Root ".pcai/build/artifacts/pcai-$Backend/pcai-$Backend.exe"
                if ($args.Count -ne 1 -or $args[0] -ne '--version-json' -or
                    (Resolve-Path -LiteralPath $ProbePath).Path -ne (Resolve-Path -LiteralPath $expectedPath).Path) {
                    throw 'Version query did not use the canonical staged executable.'
                }
                $global:LASTEXITCODE = $MetadataExit
                $MetadataJson
            }

            $pattern = '(?ms)^      - name: Prepare Artifact\r?\n.*?^        run: \|\r?\n(?<body>.*?)(?=^      [#-]|\z)'
            $body = [regex]::Match($script:WorkflowText, $pattern).Groups['body'].Value
            if (-not $body) { throw 'Missing native release artifact preparation step.' }
            $body = $body.Replace('${{ matrix.backend }}', $Backend)
            $body = $body.Replace('${{ matrix.cuda }}', $Cuda.ToString().ToLowerInvariant())
            $body = $body.Replace('${{ steps.features.outputs.BIN_NAME }}', "pcai-$Backend")
            # Stub only the owned binary invocation. Run the YAML's other packaging code unchanged.
            $body = $body.Replace('& $nativeExecutable --version-json', 'Invoke-ArtifactMetadataFixture $nativeExecutable --version-json')
            $check = [scriptblock]::Create(($body -replace '(?m)^          ', ''))

            $priorOutput = $env:GITHUB_OUTPUT
            $priorCommit = $env:PCAI_EXPECTED_SOURCE_COMMIT
            $priorExit = $global:LASTEXITCODE
            Push-Location -LiteralPath $Root
            try {
                $env:GITHUB_OUTPUT = Join-Path $Root 'github-output'
                $env:PCAI_EXPECTED_SOURCE_COMMIT = $script:SourceCommit
                & $check
            }
            finally {
                $env:GITHUB_OUTPUT = $priorOutput
                $env:PCAI_EXPECTED_SOURCE_COMMIT = $priorCommit
                $global:LASTEXITCODE = $priorExit
                Pop-Location
            }
        }

        function Read-NativeArchive {
            param([string]$Path)

            $entries = @{}
            $digests = @{}
            $archive = [IO.Compression.ZipFile]::OpenRead($Path)
            try {
                foreach ($entry in $archive.Entries) {
                    $reader = [IO.StreamReader]::new($entry.Open())
                    try { $entries.Add($entry.FullName, $reader.ReadToEnd()) }
                    finally { $reader.Dispose() }

                    $stream = $entry.Open()
                    try {
                        $digest = (Get-FileHash -InputStream $stream -Algorithm SHA256).Hash
                        $digests.Add($entry.FullName, $digest.ToLowerInvariant())
                    }
                    finally { $stream.Dispose() }
                }
            }
            finally { $archive.Dispose() }
            return @{ Files = $entries; Digests = $digests }
        }
    }

    # Protects: packaged native variants include the staged executable and P/Invoke alias.
    # Detects: stale DLLs or staging residue, mismatched manifest files or incorrect SHA-256 checksums.
    # Needs: PowerShell 7, Pester 5, workflow text, and synthetic files in TestDrive.
    # Breadcrumb: .github/workflows/release-cuda.yml Prepare Artifact, Build.ps1 Invoke-InferenceBuild.
    It 'packs the canonical <Backend> <Variant> staged files' -ForEach @(
        @{ Backend = 'llamacpp'; Variant = 'cpu'; Cuda = $false },
        @{ Backend = 'llamacpp'; Variant = 'cuda'; Cuda = $true },
        @{ Backend = 'mistralrs'; Variant = 'cpu'; Cuda = $false },
        @{ Backend = 'mistralrs'; Variant = 'cuda'; Cuda = $true }
    ) {
        $name = "pcai-inference-$Backend-$Variant-win64"
        $root = Join-Path $TestDrive $name
        $staged = Join-Path $root ".pcai/build/artifacts/pcai-$Backend"
        $legacy = Join-Path $root 'Native/pcai_core/pcai_inference/target/release'
        $zipScratch = Join-Path $root "staging/$name"
        $null = New-Item -ItemType Directory -Path $staged, $legacy, $zipScratch -Force

        $exe = "pcai-$Backend.exe"
        Set-Content -LiteralPath (Join-Path $staged $exe) -Value 'canonical executable' -NoNewline
        Set-Content -LiteralPath (Join-Path $staged 'pcai_inference.dll') -Value 'canonical library' -NoNewline
        Set-Content -LiteralPath (Join-Path $staged 'pcai_inference_lib.dll') -Value 'canonical library' -NoNewline
        Set-Content -LiteralPath (Join-Path $legacy 'pcai_inference.dll') -Value 'stale alias' -NoNewline
        Set-Content -LiteralPath (Join-Path $legacy 'pcai_inference_lib.dll') -Value 'stale library' -NoNewline
        Set-Content -LiteralPath (Join-Path $zipScratch 'old-artifact.dll') -Value 'stale stage' -NoNewline

        Invoke-PrepareNativeArtifact -Root $root -Backend $Backend -Cuda $Cuda | Out-Null
        $archive = Read-NativeArchive -Path (Join-Path $root "$name.zip")
        $files = $archive.Files
        $files[$exe] | Should -BeExactly 'canonical executable'
        $files['pcai_inference.dll'] | Should -BeExactly 'canonical library'
        $files['pcai_inference_lib.dll'] | Should -BeExactly $files['pcai_inference.dll']
        $files['README.txt'] | Should -Match '--config \.\\pcai-inference\.json'
        $files['README.txt'] | Should -Not -Match '--(?:model|port)'
        $files.ContainsKey('old-artifact.dll') | Should -BeFalse
        $manifest = $files['manifest.json'] | ConvertFrom-Json
        $manifest.schemaVersion | Should -Be 1
        $manifest.commit | Should -BeExactly $script:SourceCommit
        $manifest.variant | Should -BeExactly $name
        [string[]]$payload = @($exe, 'pcai_inference.dll', 'pcai_inference_lib.dll', 'README.txt')
        [Array]::Sort($payload, [StringComparer]::Ordinal)
        ($manifest.files.path -join ',') | Should -BeExactly ($payload -join ',')
        foreach ($entry in $manifest.files) {
            $entry.sha256 | Should -BeExactly $archive.Digests[$entry.path]
        }

        [string[]]$coveredNames = $payload + 'manifest.json'
        [Array]::Sort($coveredNames, [StringComparer]::Ordinal)
        $expectedLines = foreach ($file in $coveredNames) {
            '{0}  {1}' -f $archive.Digests[$file], $file
        }
        $lineFeed = [string][char]10
        $files['SHA256SUMS'] | Should -BeExactly (($expectedLines -join $lineFeed) + $lineFeed)
        @($files.Keys).Count | Should -Be ($payload.Count + 2)
        Get-Content -LiteralPath (Join-Path $root 'github-output') | Should -Contain "ARTIFACT_PATH=$name.zip"
    }

    # Protects: incomplete canonical builds never emit a release ZIP.
    # Detects: stale Cargo output substituted for a missing executable or alias, or an empty alias accepted.
    # Needs: PowerShell 7, Pester 5, workflow text, and synthetic files in TestDrive.
    # Breadcrumb: .github/workflows/release-cuda.yml Prepare Artifact.
    It 'rejects an incomplete canonical artifact when <Case>' -ForEach @(
        @{ Case = 'the executable is absent'; Missing = 'pcai-llamacpp.exe'; Empty = $false },
        @{ Case = 'the alias is absent'; Missing = 'pcai_inference.dll'; Empty = $false },
        @{ Case = 'the alias is empty'; Missing = 'pcai_inference.dll'; Empty = $true },
        @{ Case = 'the Cargo library is absent'; Missing = 'pcai_inference_lib.dll'; Empty = $false }
    ) {
        $root = Join-Path $TestDrive $Case
        $staged = Join-Path $root '.pcai/build/artifacts/pcai-llamacpp'
        $legacy = Join-Path $root 'Native/pcai_core/pcai_inference/target/release'
        $null = New-Item -ItemType Directory -Path $staged, $legacy -Force
        foreach ($file in @('pcai-llamacpp.exe', 'pcai_inference.dll', 'pcai_inference_lib.dll')) {
            Set-Content -LiteralPath (Join-Path $legacy $file) -Value 'retired output' -NoNewline
            if ($file -ne $Missing) {
                Set-Content -LiteralPath (Join-Path $staged $file) -Value 'canonical output' -NoNewline
            }
            elseif ($Empty) {
                [IO.File]::WriteAllBytes((Join-Path $staged $file), [byte[]]::new(0))
            }
        }

        { Invoke-PrepareNativeArtifact -Root $root -Backend llamacpp -Cuda $false } |
            Should -Throw -ExpectedMessage "*required staged native release file*$Missing*"
        Test-Path -LiteralPath (Join-Path $root 'pcai-inference-llamacpp-cpu-win64.zip') | Should -BeFalse
        Test-Path -LiteralPath (Join-Path $root 'github-output') | Should -BeFalse
    }

    # Protects: the compatibility alias is the same native library as Cargo built.
    # Detects: a stale alias accepted and hashed as an independent binary.
    # Needs: PowerShell 7, Pester 5 and two different synthetic DLLs in TestDrive.
    # Breadcrumb: .github/workflows/release-cuda.yml Prepare Artifact digest comparison.
    It 'rejects a P/Invoke alias with different bytes from the Cargo library' {
        $root = Join-Path $TestDrive 'mismatched-alias'
        $staged = Join-Path $root '.pcai/build/artifacts/pcai-llamacpp'
        $null = New-Item -ItemType Directory -Path $staged -Force
        Set-Content -LiteralPath (Join-Path $staged 'pcai-llamacpp.exe') -Value 'canonical executable' -NoNewline
        Set-Content -LiteralPath (Join-Path $staged 'pcai_inference.dll') -Value 'stale alias' -NoNewline
        Set-Content -LiteralPath (Join-Path $staged 'pcai_inference_lib.dll') -Value 'canonical library' -NoNewline

        { Invoke-PrepareNativeArtifact -Root $root -Backend llamacpp -Cuda $false } |
            Should -Throw -ExpectedMessage '*P/Invoke alias does not match the Cargo library*'
        Test-Path -LiteralPath (Join-Path $root 'pcai-inference-llamacpp-cpu-win64.zip') | Should -BeFalse
        Test-Path -LiteralPath (Join-Path $root 'github-output') | Should -BeFalse
    }

    # Protects: source revision and CPU/CUDA variants have matching runtime build identity.
    # Detects: wrong backend, CPU masquerading as CUDA, stale commit, incompatible target or profile, or a failed query.
    # Needs: PowerShell 7, Pester 5, synthetic files and a stubbed child version response.
    # Breadcrumb: .github/workflows/release-cuda.yml Prepare Artifact and pcai_inference/build.rs.
    It 'rejects the staged executable when <Case>' -ForEach @(
        @{ Case = 'the feature set is CPU-only'; Sha = 'valid'; Features = 'llamacpp,ffi,server'; MetadataExit = 0; Malformed = $false; ErrorText = '*feature set does not match*' },
        @{ Case = 'another backend was built'; Sha = 'valid'; Features = 'mistralrs,cuda-mistralrs,ffi,server'; MetadataExit = 0; Malformed = $false; ErrorText = '*feature set does not match*' },
        @{ Case = 'the source commit is stale'; Sha = '1111111111111111111111111111111111111111'; Features = 'llamacpp,cuda-llamacpp,ffi,server'; MetadataExit = 0; Malformed = $false; ErrorText = '*source commit does not match*' },
        @{ Case = 'the child exits nonzero'; Sha = 'valid'; Features = 'llamacpp,cuda-llamacpp,ffi,server'; MetadataExit = 7; Malformed = $false; ErrorText = '*query failed with exit code 7*' },
        @{ Case = 'the child output is invalid JSON'; Sha = 'valid'; Features = 'llamacpp,cuda-llamacpp,ffi,server'; MetadataExit = 0; Malformed = $true; ErrorText = '*did not return JSON*' },
        @{ Case = 'a debug binary was staged'; Sha = 'valid'; MetadataProfile = 'debug'; MetadataExit = 0; Malformed = $false; ErrorText = '*target or profile is incompatible*' },
        @{ Case = 'an ARM64 binary was staged'; Sha = 'valid'; MetadataTarget = 'aarch64-pc-windows-msvc'; MetadataExit = 0; Malformed = $false; ErrorText = '*target or profile is incompatible*' },
        @{ Case = 'the target is an array'; Sha = 'valid'; MetadataTarget = @('x86_64-pc-windows-msvc'); MetadataExit = 0; Malformed = $false; ErrorText = '*target or profile is incompatible*' },
        @{ Case = 'the profile is an array'; Sha = 'valid'; MetadataProfile = @('release'); MetadataExit = 0; Malformed = $false; ErrorText = '*target or profile is incompatible*' }
    ) {
        $root = Join-Path $TestDrive $Case
        $staged = Join-Path $root '.pcai/build/artifacts/pcai-llamacpp'
        $null = New-Item -ItemType Directory -Path $staged -Force
        Set-Content -LiteralPath (Join-Path $staged 'pcai-llamacpp.exe') -Value 'canonical executable' -NoNewline
        Set-Content -LiteralPath (Join-Path $staged 'pcai_inference.dll') -Value 'canonical library' -NoNewline
        Set-Content -LiteralPath (Join-Path $staged 'pcai_inference_lib.dll') -Value 'canonical library' -NoNewline
        $commit = if ($Sha -eq 'valid') { $script:SourceCommit } else { $Sha }
        $identity = if ($Malformed) {
            '{broken'
        } else {
            @{
                git_hash = $commit
                features = $(if ($Features) { $Features } else { 'llamacpp,cuda-llamacpp,ffi,server' })
                target = $(if ($null -ne $MetadataTarget) { ,$MetadataTarget } else { 'x86_64-pc-windows-msvc' })
                profile = $(if ($null -ne $MetadataProfile) { ,$MetadataProfile } else { 'release' })
            } | ConvertTo-Json -Compress
        }

        $query = @{ Root = $root; Backend = 'llamacpp'; Cuda = $true; MetadataJson = $identity; MetadataExit = $MetadataExit }
        { Invoke-PrepareNativeArtifact @query } | Should -Throw -ExpectedMessage $ErrorText
        Test-Path -LiteralPath (Join-Path $root 'pcai-inference-llamacpp-cuda-win64.zip') | Should -BeFalse
        Test-Path -LiteralPath (Join-Path $root 'github-output') | Should -BeFalse
    }
}
