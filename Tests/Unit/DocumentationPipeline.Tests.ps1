#Requires -Version 7.0

BeforeAll {
    $script:repository = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
    $script:generator = Join-Path $script:repository 'Tools/generate-auto-docs.ps1'
    $script:pipeline = Join-Path $script:repository 'Tools/Invoke-DocPipeline.ps1'
    . $script:generator -LibraryOnly
    $script:fixtureRoot = Join-Path $TestDrive 'documentation fixtures'
    $null = New-Item -ItemType Directory $script:fixtureRoot
    $script:savedTarget = $env:CARGO_TARGET_DIR
    $script:savedWrapper = $env:RUSTC_WRAPPER
    $script:savedRustDocFlags = $env:RUSTDOCFLAGS
    $script:savedBuildTarget = $env:CARGO_BUILD_TARGET
    $env:CARGO_TARGET_DIR = Join-Path $script:fixtureRoot 'shared target'
    $env:RUSTC_WRAPPER = ''
    $env:RUSTDOCFLAGS = '-D warnings'
    $env:CARGO_BUILD_TARGET = $null
    $script:rustGood = Join-Path $script:fixtureRoot 'good rust'
    $script:rustBad = Join-Path $script:fixtureRoot 'bad rust'
    foreach ($path in @($script:rustGood, $script:rustBad)) {
        $null = New-Item -ItemType Directory (Join-Path $path 'src') -Force
        [IO.File]::WriteAllText((Join-Path $path 'Cargo.toml'), "[package]`nname = 'documentation_fixture'`nversion = '0.1.0'`nedition = '2021'`n[workspace]`n")
    }
    [IO.File]::WriteAllText((Join-Path $script:rustGood 'src/lib.rs'), "/// Returns the fixture sentinel.`npub fn sentinel() -> u32 { private_sentinel() }`nfn private_sentinel() -> u32 { 731 }`n")
    [IO.File]::WriteAllText((Join-Path $script:rustBad 'src/lib.rs'), 'compile_error!("intentional documentation child failure");')
    $script:rustFeature = Join-Path $script:fixtureRoot 'feature gated rust'
    $script:rustTriple = Join-Path $script:fixtureRoot 'configured target rust'
    foreach ($path in @($script:rustFeature, $script:rustTriple)) {
        $null = New-Item -ItemType Directory (Join-Path $path 'src') -Force
        [IO.File]::WriteAllText((Join-Path $path 'Cargo.toml'), [IO.File]::ReadAllText((Join-Path $script:rustGood 'Cargo.toml')))
        [IO.File]::WriteAllText((Join-Path $path 'src/lib.rs'), [IO.File]::ReadAllText((Join-Path $script:rustGood 'src/lib.rs')))
    }
    [IO.File]::AppendAllText((Join-Path $script:rustFeature 'Cargo.toml'), "[features]`naccelerated = []`n[[bin]]`nname = 'conditional_viewer'`npath = 'src/viewer.rs'`nrequired-features = ['accelerated']`n")
    [IO.File]::WriteAllText((Join-Path $script:rustFeature 'src/viewer.rs'), 'fn main() {}')
    $compilerVersion = Invoke-DocumentationCommand -Tool rustc.exe -ArgumentList @('-vV')
    if ($compilerVersion -notmatch '(?m)^host:\s*(.+)$') { throw 'Rust compiler did not report its host target' }
    $script:hostTriple = $Matches[1].Trim()
    $script:csRoot = Join-Path $script:fixtureRoot 'csharp'
    $null = New-Item -ItemType Directory $script:csRoot
    $sdkVersion = Invoke-DocumentationCommand -Tool dotnet.exe -ArgumentList @('--version')
    $framework = 'net' + ($sdkVersion -split '\.')[0] + '.0'
    $script:csProject = Join-Path $script:csRoot 'DocumentationFixture.csproj'
    [IO.File]::WriteAllText($script:csProject, "<Project Sdk=`"Microsoft.NET.Sdk`"><PropertyGroup><TargetFramework>$framework</TargetFramework><AssemblyName>ExactFixtureAssembly</AssemblyName><TreatWarningsAsErrors>true</TreatWarningsAsErrors></PropertyGroup></Project>")
    [IO.File]::WriteAllText((Join-Path $script:csRoot 'NuGet.Config'), '<configuration><packageSources><clear /></packageSources></configuration>')
    [IO.File]::WriteAllText((Join-Path $script:csRoot 'Fixture.cs'), "/// <summary>Fixture sentinel.</summary>`npublic class Fixture { /// <summary>Returns 731.</summary>`npublic static int Sentinel() => 731; }")
}

AfterAll {
    $env:CARGO_TARGET_DIR = $script:savedTarget
    $env:RUSTC_WRAPPER = $script:savedWrapper
    $env:RUSTDOCFLAGS = $script:savedRustDocFlags
    $env:CARGO_BUILD_TARGET = $script:savedBuildTarget
}

Describe 'Documentation workspace boundaries' {
    It 'selects only existing in-repository defaults' {
        $local = Join-Path $script:fixtureRoot 'repo/Native/pcai_core'
        $null = New-Item -ItemType Directory $local -Force
        [IO.File]::WriteAllText((Join-Path $local 'Cargo.toml'), '[workspace]')
        @(Get-DocumentationWorkspace -Repository (Join-Path $script:fixtureRoot 'repo')) | Should -Be @($local)
    }
    It 'rejects external and sibling-prefix roots unless explicitly authorized' {
        { Get-DocumentationWorkspace -Repository $script:rustGood -Roots $script:rustBad } | Should -Throw '*requires -AllowExternalRustRoots*'
        { Get-DocumentationWorkspace -Repository $script:rustGood -Roots ($script:rustGood + '-other') } | Should -Throw '*requires -AllowExternalRustRoots*'
        @(Get-DocumentationWorkspace -Repository $script:rustGood -Roots $script:rustBad -AllowExternal) | Should -Be @($script:rustBad)
    }
    It 'fails on a missing explicit workspace but allows absent optional defaults' {
        { Get-DocumentationWorkspace -Repository $script:fixtureRoot -Roots 'missing' } | Should -Throw '*Cargo.toml not found*'
        @(Get-DocumentationWorkspace -Repository $script:csRoot).Count | Should -Be 0
    }
    It 'rejects a requested missing native tool' {
        { Invoke-DocumentationCommand -Tool pcai-documentation-tool-does-not-exist.exe -ArgumentList @('--version') } | Should -Throw
    }
    It 'does not write reports when importing the production library' {
        $empty = Join-Path $script:fixtureRoot 'library only'
        $null = New-Item -ItemType Directory $empty
        . $script:generator -RepoRoot $empty -LibraryOnly
        Test-Path (Join-Path $empty 'Reports') | Should -BeFalse
    }
}

Describe 'Real native documentation generation' {
    It 'uses Cargo selection when a binary required feature is disabled' {
        $result = Invoke-RustDocumentation -Workspace $script:rustFeature -Build
        @($result.DocIndexes).Count | Should -Be 1
        $result.DocumentedTargets.Target | Should -Be 'documentation_fixture'
        $result.DocumentedTargets[0].Features.Count | Should -Be 0
        $result.DocIndexes[0] | Should -Not -BeLike '*conditional_viewer*'
    }
    It 'includes the feature-gated binary when Cargo enables its default feature' {
        $manifest = Join-Path $script:rustFeature 'Cargo.toml'
        $original = [IO.File]::ReadAllText($manifest)
        [IO.File]::WriteAllText($manifest, $original.Replace('[features]', "[features]`ndefault = ['accelerated']"))
        try {
            $result = Invoke-RustDocumentation -Workspace $script:rustFeature -Build
            @($result.DocIndexes).Count | Should -Be 2
            $binary = @($result.DocumentedTargets | Where-Object Target -EQ conditional_viewer)
            $binary.Count | Should -Be 1
            $binary[0].Features | Should -Contain 'accelerated'
            Test-Path -LiteralPath $binary[0].Index -PathType Leaf | Should -BeTrue
        }
        finally { [IO.File]::WriteAllText($manifest, $original) }
    }
    It 'uses actual Cargo artifact paths for a configured build target' {
        $configuration = Join-Path $script:rustTriple '.cargo'
        $null = New-Item -ItemType Directory $configuration -Force
        [IO.File]::WriteAllText((Join-Path $configuration 'config.toml'), "[build]`ntarget = '$script:hostTriple'`n")
        $result = Invoke-RustDocumentation -Workspace $script:rustTriple -Build
        @($result.DocIndexes).Count | Should -Be 1
        $result.DocIndexes[0] | Should -BeLike "$($result.TargetDirectory)*$script:hostTriple*doc*documentation_fixture*index.html"
        Test-Path -LiteralPath $result.DocIndexes[0] -PathType Leaf | Should -BeTrue
    }
    It 'uses actual Cargo artifact paths for CARGO_BUILD_TARGET' {
        $env:CARGO_BUILD_TARGET = $script:hostTriple
        try {
            $result = Invoke-RustDocumentation -Workspace $script:rustGood -Build
            @($result.DocIndexes).Count | Should -Be 1
            $result.DocIndexes[0] | Should -BeLike "$($result.TargetDirectory)*$script:hostTriple*doc*documentation_fixture*index.html"
            Test-Path -LiteralPath $result.DocIndexes[0] -PathType Leaf | Should -BeTrue
        }
        finally { $env:CARGO_BUILD_TARGET = $null }
    }
    It 'generates the exact Rust crate index from Cargo target metadata' {
        $result = Invoke-RustDocumentation -Workspace $script:rustGood -Build
        $result.Provenance | Should -Be 'Generated'
        $result.TargetDirectory | Should -BeLike "$env:CARGO_TARGET_DIR*pcai-docs*"
        @($result.DocIndexes).Count | Should -Be 1
        $result.DocIndexes[0] | Should -BeLike '*doc*documentation_fixture*index.html'
        [IO.File]::ReadAllText($result.DocIndexes[0]) | Should -Match 'documentation_fixture'
    }
    It 'honors workspace Cargo configuration and restores the caller location' {
        $configuration = Join-Path $script:rustGood '.cargo'
        $null = New-Item -ItemType Directory $configuration -Force
        [IO.File]::WriteAllText((Join-Path $configuration 'config.toml'), "[build]`ntarget-dir = 'configured documentation target'`n")
        $previousLocation = (Get-Location).Path
        $env:CARGO_TARGET_DIR = $null
        try {
            $result = Invoke-RustDocumentation -Workspace $script:rustGood -Build
            $result.TargetDirectory | Should -BeLike "$script:rustGood*configured documentation target*pcai-docs*"
            (Get-Location).Path | Should -Be $previousLocation
        }
        finally { $env:CARGO_TARGET_DIR = Join-Path $script:fixtureRoot 'shared target' }
    }
    It 'preserves the pipeline option to document private Rust items' {
        $result = Invoke-RustDocumentation -Workspace $script:rustGood -Build -DocumentPrivateItems
        $privatePage = Join-Path (Split-Path $result.DocIndexes[0]) 'fn.private_sentinel.html'
        Test-Path -LiteralPath $privatePage -PathType Leaf | Should -BeTrue
        [IO.File]::ReadAllText($privatePage) | Should -Match 'private_sentinel'
    }
    It 'rejects a real failed cargo child despite stale expected documentation' {
        $stale = Join-Path $env:CARGO_TARGET_DIR 'doc/documentation_fixture/index.html'
        $null = New-Item -ItemType Directory (Split-Path $stale) -Force
        [IO.File]::WriteAllText($stale, 'STALE-731')
        { Invoke-RustDocumentation -Workspace $script:rustBad -Build } | Should -Throw '*failed (exit*intentional documentation child failure*'
        [IO.File]::ReadAllText($stale) | Should -Be 'STALE-731'
    }
    It 'does not attribute an unrelated Rust index during inspection' {
        $env:CARGO_TARGET_DIR = Join-Path $script:fixtureRoot 'wrong target'
        try {
            $unrelated = Join-Path $env:CARGO_TARGET_DIR 'doc/unrelated/index.html'
            $null = New-Item -ItemType Directory (Split-Path $unrelated) -Force
            [IO.File]::WriteAllText($unrelated, 'WRONG-731')
            $result = Invoke-RustDocumentation -Workspace $script:rustGood
            @($result.DocIndexes).Count | Should -Be 0
            $result.Provenance | Should -Be 'ExistingUnverified'
        }
        finally { $env:CARGO_TARGET_DIR = Join-Path $script:fixtureRoot 'shared target' }
    }
    It 'labels an exact preexisting Rust index as unverified without rebuilding' {
        $result = Invoke-RustDocumentation -Workspace $script:rustGood
        @($result.DocIndexes).Count | Should -Be 1
        $result.Provenance | Should -Be 'ExistingUnverified'
        [IO.File]::ReadAllText($result.DocIndexes[0]) | Should -Be 'STALE-731'
    }
    It 'rejects empty Rust artifacts even during inspection' {
        $index = Join-Path $env:CARGO_TARGET_DIR 'doc/documentation_fixture/index.html'
        [IO.File]::WriteAllText($index, '')
        try { { Invoke-RustDocumentation -Workspace $script:rustGood } | Should -Throw '*Empty Rust documentation artifact*' }
        finally { [IO.File]::WriteAllText($index, 'STALE-731') }
    }
    It 'generates C# XML at a fresh explicit path with the actual assembly name' {
        $result = Invoke-CSharpDocumentation -Project $script:csProject -ReportDirectory $script:fixtureRoot -Build
        $result.Provenance | Should -Be 'Generated'
        [xml]$document = [IO.File]::ReadAllText($result.DocXml)
        $document.doc.assembly.name | Should -Be 'ExactFixtureAssembly'
        $document.doc.members.member.name | Should -Contain 'M:Fixture.Sentinel'
    }
    It 'rejects a real failed dotnet child despite stale project-name XML' {
        $stale = Join-Path $script:csRoot 'wrong/DocumentationFixture.xml'
        $null = New-Item -ItemType Directory (Split-Path $stale) -Force
        [IO.File]::WriteAllText($stale, 'STALE-731')
        [IO.File]::WriteAllText((Join-Path $script:csRoot 'Broken.cs'), 'this is not valid C#;')
        try {
            { Invoke-CSharpDocumentation -Project $script:csProject -ReportDirectory $script:fixtureRoot -Build } | Should -Throw '*dotnet.exe failed (exit*'
            [IO.File]::ReadAllText($stale) | Should -Be 'STALE-731'
        }
        finally { Remove-Item -LiteralPath (Join-Path $script:csRoot 'Broken.cs') }
    }
    It 'rejects wrong-target C# XML during inspection' {
        $result = Invoke-CSharpDocumentation -Project $script:csProject -ReportDirectory $script:fixtureRoot
        $result.DocXml | Should -BeNullOrEmpty
        $result.Provenance | Should -Be 'ExistingUnverified'
    }
    It 'rejects success without the required C# XML artifact' {
        $project = Join-Path $script:csRoot 'MissingArtifact.csproj'
        [IO.File]::WriteAllText($project, [IO.File]::ReadAllText($script:csProject).Replace('</Project>', '<Target Name="DeleteDocumentation" AfterTargets="Build"><Delete Files="$(DocumentationFile)" /></Target></Project>'))
        { Invoke-CSharpDocumentation -Project $project -ReportDirectory $script:fixtureRoot -Build } | Should -Throw '*Required C# documentation artifact missing*'
    }
}

Describe 'Production script failure propagation' {
    It 'fails a requested C# build when no projects are present' {
        $repo = Join-Path $script:fixtureRoot 'missing csharp'
        $output = & pwsh.exe -NoLogo -NoProfile -File $script:generator -RepoRoot $repo -IncludeCSharp -BuildDocs 2>&1
        $LASTEXITCODE | Should -Be 1
        ($output | Out-String) | Should -Match 'No C# projects found for requested documentation build'
    }
    It 'returns zero and writes attributable reports from the actual auto-docs script' {
        $repo = Join-Path $script:fixtureRoot 'successful script'
        $output = & pwsh.exe -NoLogo -NoProfile -File $script:generator -RepoRoot $repo -IncludeRust -BuildDocs -RustWorkspaceRoots $script:rustGood -AllowExternalRustRoots 2>&1
        $LASTEXITCODE | Should -Be 0
        ($output | Out-String) | Should -Match 'Wrote:'
        $report = [IO.File]::ReadAllText((Join-Path $repo 'Reports/RUST_DOCS.json')) | ConvertFrom-Json
        $report.Provenance | Should -Be 'Generated'
        $report.Workspace | Should -Be $script:rustGood
        Test-Path (Join-Path $repo 'Reports/AUTO_DOCS_SUMMARY.md') | Should -BeTrue
    }
    It 'returns nonzero from the actual auto-docs script on failed Rust generation' {
        $output = & pwsh.exe -NoLogo -NoProfile -File $script:generator -RepoRoot $script:fixtureRoot -IncludeRust -BuildDocs -RustWorkspaceRoots $script:rustBad 2>&1
        $code = $LASTEXITCODE
        $code | Should -Be 1
        ($output | Out-String) | Should -Match 'intentional documentation child failure'
        Test-Path (Join-Path $script:fixtureRoot 'Reports/AUTO_DOCS_SUMMARY.md') | Should -BeFalse
    }
    It 'records requested Rust failure as Error and exits the actual pipeline nonzero' {
        $fixtureTools = Join-Path $script:fixtureRoot 'pipeline/Tools'
        $null = New-Item -ItemType Directory $fixtureTools -Force
        Copy-Item -LiteralPath $script:pipeline, $script:generator -Destination $fixtureTools
        $output = & pwsh.exe -NoLogo -NoProfile -File (Join-Path $fixtureTools 'Invoke-DocPipeline.ps1') -Mode DocsOnly -RustWorkspaceRoots $script:rustBad -AllowExternalRustRoots 2>&1
        $code = $LASTEXITCODE
        $code | Should -Be 1
        $report = [IO.File]::ReadAllText((Join-Path $script:fixtureRoot 'pipeline/Reports/DOC_PIPELINE_REPORT.json')) | ConvertFrom-Json
        $report.success | Should -BeFalse
        @($report.steps | Where-Object Name -EQ RustDocs).Status | Should -Be 'Error'
        ($report.errors -join '') | Should -Match 'intentional documentation child failure'
        ($output | Out-String) | Should -Match 'FAIL Pipeline completed'
    }
    It 'keeps absent optional tooling from failing a valid validation-only run' {
        $fixture = Join-Path $script:fixtureRoot 'validation only'
        $tools = Join-Path $fixture 'Tools'
        $data = Join-Path $fixture 'Deploy/rust-functiongemma-train/data'
        $null = New-Item -ItemType Directory $tools, $data -Force
        Copy-Item -LiteralPath $script:pipeline -Destination $tools
        [IO.File]::WriteAllText((Join-Path $data 'rust_router_train.jsonl'), '{"messages":[],"tools":[]}')
        [IO.File]::WriteAllText((Join-Path $data 'test_vectors.json'), '[{"tool":"fixture","arguments":{}}]')
        $output = & pwsh.exe -NoLogo -NoProfile -File (Join-Path $tools 'Invoke-DocPipeline.ps1') -Mode Validate 2>&1
        $LASTEXITCODE | Should -Be 0
        $report = [IO.File]::ReadAllText((Join-Path $fixture 'Reports/DOC_PIPELINE_REPORT.json')) | ConvertFrom-Json
        $report.success | Should -BeTrue
        @($report.errors).Count | Should -Be 0
        @($report.steps | Where-Object Status -EQ Warning).Count | Should -Be 2
        ($output | Out-String) | Should -Match 'OK Pipeline completed successfully'
    }
}
