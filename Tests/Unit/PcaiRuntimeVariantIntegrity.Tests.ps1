param(
    [string]$RuntimeSourcePath,
    [string]$FixtureRoot
)

BeforeAll {
    if (-not $RuntimeSourcePath) { $RuntimeSourcePath = Join-Path $PSScriptRoot '../../Modules/PcaiInference.psm1' }
    $tokens = $null
    $parseErrors = $null
    $sourceAst = [Management.Automation.Language.Parser]::ParseFile($RuntimeSourcePath, [ref]$tokens, [ref]$parseErrors)
    if ($parseErrors.Count) { throw 'Runtime resolver source contains parse errors.' }
    foreach ($name in @('Resolve-PcaiRuntimeVariantDll', 'Resolve-PcaiInferenceDll')) {
        $definitions = @($sourceAst.FindAll({
            param($node)
            $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -ceq $name
        }, $true))
        if ($definitions.Count -ne 1) { throw "Exactly one complete source function required: $name" }
        Set-Item -LiteralPath "function:$name" -Value ($definitions[0].Body.GetScriptBlock())
    }

    # Explicit collaborators only: no module bootstrap, FFI, GPU command or HTTP.
    function Get-PcaiCudaCapability { throw 'Unmocked GPU boundary.' }
    function Get-PcaiConfig { throw 'Unmocked configuration boundary.' }
    function Get-PcaiProjectRoot { throw 'Unmocked project-root boundary.' }
    $script:RuntimeFixtureNumber = 0
    $script:OriginalBundle = [Environment]::GetEnvironmentVariable('PCAI_NATIVE_BUNDLE_ROOT', 'Process')

    function New-RuntimeFixtureMapping([string]$MappingKind, [Collections.IDictionary]$Pairs) {
        $mapping = if ($MappingKind -ceq 'Generic') { [Collections.Generic.Dictionary[string,object]]::new([StringComparer]::Ordinal) }
        elseif ($MappingKind -ceq 'Ordered') { [Collections.Specialized.OrderedDictionary]::new([StringComparer]::Ordinal) }
        else { throw 'Explicit fixture mapping kind required.' }
        foreach ($entry in $Pairs.GetEnumerator()) { $mapping.Add($entry.Key, $entry.Value) }
        return ,$mapping
    }
    function New-RuntimeFixtureMappingConfig([string]$MappingKind) {
        $variant = New-RuntimeFixtureMapping $MappingKind @{ DLLPATH=$script:Variant; KIND='cpu'; MINCOMPUTE=0; MAXCOMPUTE=0; SHA256=$script:ExpectedHash; Keys='literal variant metadata'; Count=99 }
        $selector = New-RuntimeFixtureMapping $MappingKind @{ ENABLED=$true; AUTODOWNLOAD=$false; VARIANTS=@($variant); Keys='literal selector metadata'; Count=98 }
        $native = New-RuntimeFixtureMapping $MappingKind @{ RUNTIMEBINARYSELECTION=$selector; DLLSEARCHPATHS=@(); Keys='literal native metadata'; Count=97 }
        return New-RuntimeFixtureMapping $MappingKind @{ NATIVEINFERENCE=$native; Keys='literal root metadata'; Count=96 }
    }
    function Read-RuntimeFixtureText([string]$Path) { [IO.File]::ReadAllText($Path) }
    function New-RuntimeFixtureConfig {
        param([string]$Path, [AllowNull()]$Hash, [bool]$Download = $false)
        @{
            nativeInference = @{
                dllSearchPaths = @()
                runtimeBinarySelection = @{
                    enabled = $true; autoDownload = $Download
                    variants = @(@{
                        name = 'public-fixture'; kind = 'cpu'; minCompute = -1; maxCompute = 9999
                        dllPath = $Path; url = 'https://runtime-fixture.invalid/public.dll'; sha256 = $Hash
                    })
                }
            }
        } | ConvertTo-Json -Depth 6 | ConvertFrom-Json
    }
}

Describe 'Runtime DLL variant integrity with real public file bytes' -Tag 'Unit', 'Inference', 'Portable' {
    BeforeEach {
        $script:RuntimeFixtureNumber++
        $base = if ($FixtureRoot) { $FixtureRoot } else { $TestDrive }
        $script:RuntimeFixture = Join-Path $base "case-$script:RuntimeFixtureNumber"
        [void][IO.Directory]::CreateDirectory((Join-Path $script:RuntimeFixture 'variants'))
        [void][IO.Directory]::CreateDirectory((Join-Path $script:RuntimeFixture 'bin'))
        $script:Variant = Join-Path $script:RuntimeFixture 'variants/runtime.dll'
        $script:Canonical = Join-Path $script:RuntimeFixture 'bin/pcai_inference.dll'
        $script:Alias = Join-Path $script:RuntimeFixture 'bin/pcai_inference_lib.dll'
        $script:ValidBytes = 'verified public runtime fixture'
        $script:OldBytes = 'preserved previous runtime fixture'
        $reference = Join-Path $script:RuntimeFixture 'expected-public-bytes'
        [IO.File]::WriteAllText($reference, $script:ValidBytes)
        $script:ExpectedHash = (Get-FileHash -LiteralPath $reference -Algorithm SHA256).Hash
        [IO.File]::WriteAllText($script:Canonical, $script:OldBytes)
        [IO.File]::WriteAllText($script:Alias, $script:OldBytes)
        $script:RuntimeConfig = New-RuntimeFixtureConfig -Path $script:Variant -Hash $script:ExpectedHash
        [Environment]::SetEnvironmentVariable('PCAI_NATIVE_BUNDLE_ROOT', $null, 'Process')
        Mock Get-PcaiCudaCapability { $null }
        Mock Get-PcaiConfig { $script:RuntimeConfig }
        Mock Get-PcaiProjectRoot { $script:RuntimeFixture }
        Mock Invoke-WebRequest { throw 'Unexpected fixture transport request.' }
    }

    AfterEach {
        try {
            # Test-only isolation for the prefix+throw transport detector: prove that
            # retained unqualified bytes block another activation before any reset.
            # Real production state has no automatic reset or path cleanup here.
            $retained = Get-Variable -Name PcaiRuntimeVariantUnresolved -Scope Script -ErrorAction SilentlyContinue
            if ($null -ne $retained -and $retained.Value.Count -gt 0) {
                { Resolve-PcaiRuntimeVariantDll -Config $script:RuntimeConfig -ProjectRoot $script:RuntimeFixture } | Should -Throw '*original-object recovery*'
                $recovery = [Collections.Generic.List[object]]::new()
                foreach ($entry in $retained.Value) {
                    # Other kinds can contain unresolved originals: leave those intact
                    # and fail teardown rather than pretending exceptional cleanup.
                    $entry.Kind | Should -BeExactly 'UnqualifiedDownload'
                    $entry.ReservationHandle.IsClosed | Should -BeTrue
                    $fullPath = [IO.Path]::GetFullPath($entry.Path)
                    $ownedParent = [IO.Path]::GetFullPath((Join-Path $script:RuntimeFixture 'variants'))
                    [IO.Path]::GetDirectoryName($fullPath).Equals($ownedParent, [StringComparison]::OrdinalIgnoreCase) | Should -BeTrue
                    [IO.Path]::GetFileName($fullPath) | Should -Match '^\.pcai-runtime-download-[0-9a-f]{32}\.tmp$'
                    $file = Get-Item -LiteralPath $fullPath -Force
                    ($file -is [IO.FileInfo]) | Should -BeTrue
                    ($file.Attributes -band [IO.FileAttributes]::ReparsePoint) | Should -Be 0
                    $file.Length | Should -BeLessOrEqual 1024
                    $probe = [IO.File]::Open($fullPath, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::None)
                    $probeHandle = $probe.SafeFileHandle
                    $memory = [IO.MemoryStream]::new()
                    try { $probe.CopyTo($memory); $bytes = $memory.ToArray() }
                    finally { $memory.Dispose(); $probe.Dispose() }
                    $probeHandle.IsClosed | Should -BeTrue
                    $preserved = Join-Path $script:RuntimeFixture ('retained-public-prefix-' + $recovery.Count + '.bin')
                    Test-Path -LiteralPath $preserved | Should -BeFalse
                    [IO.File]::WriteAllBytes($preserved, $bytes)
                    $pin = Get-FileHash -LiteralPath $preserved -Algorithm SHA256
                    $pin.Hash | Should -BeExactly (Get-FileHash -LiteralPath $fullPath -Algorithm SHA256).Hash
                    $recovery.Add([pscustomobject]@{Kind=$entry.Kind;OriginalPath=$fullPath;PreservedPath=$preserved;Bytes=$bytes.Length;SHA256=$pin.Hash;OriginalProbeHandleClosed=$probeHandle.IsClosed;BlockedActivationAsserted=$true})
                }
                $receiptPath = Join-Path $script:RuntimeFixture 'test-owned-recovery.json'
                Test-Path -LiteralPath $receiptPath | Should -BeFalse
                [IO.File]::WriteAllText($receiptPath, ($recovery | ConvertTo-Json -Depth 4))
                # Reset only this test-script state after its real public-byte evidence
                # was preserved. The original partial file is deliberately retained.
                Remove-Variable -Name PcaiRuntimeVariantUnresolved -Scope Script -ErrorAction Stop
            }
        } finally {
            [Environment]::SetEnvironmentVariable('PCAI_NATIVE_BUNDLE_ROOT', $script:OriginalBundle, 'Process')
        }
    }

    # Protects: optional/disabled variant configuration remains a non-mutating no-op.
    # Detects: absent optional properties throw under StrictMode or cause discovery/download.
    # Needs: complete actual resolver, null/empty/disabled public config and inert collaborators.
    # Breadcrumb: Modules/PcaiInference.psm1 Resolve-PcaiRuntimeVariantDll optional selector guards.
    It 'does not discover or mutate for <Case> optional selection' -ForEach @(
        @{ Case = 'null'; Value = $null }
        @{ Case = 'omitted'; Value = [pscustomobject]@{} }
        @{ Case = 'disabled'; Value = [pscustomobject]@{ nativeInference = [pscustomobject]@{ runtimeBinarySelection = [pscustomobject]@{ enabled = $false } } } }
    ) {
        Resolve-PcaiRuntimeVariantDll -Config $Value -ProjectRoot $script:RuntimeFixture | Should -BeNullOrEmpty
        Read-RuntimeFixtureText $script:Canonical | Should -BeExactly $script:OldBytes
        Read-RuntimeFixtureText $script:Alias | Should -BeExactly $script:OldBytes
        Should -Invoke Get-PcaiCudaCapability -Times 0 -Exactly
        Should -Invoke Invoke-WebRequest -Times 0 -Exactly
    }

    # Protects: enabled selection with an empty variant list remains a no-op.
    # Detects: integrity repair discovers GPU or changes files when no variant exists.
    # Needs: actual complete resolver, empty JSON variant array and inert collaborator counters.
    # Breadcrumb: Modules/PcaiInference.psm1 Resolve-PcaiRuntimeVariantDll empty selection guard.
    It 'does not discover or mutate for an enabled empty variant array' {
        $script:RuntimeConfig.nativeInference.runtimeBinarySelection.variants = @()
        Resolve-PcaiRuntimeVariantDll -Config $script:RuntimeConfig -ProjectRoot $script:RuntimeFixture | Should -BeNullOrEmpty
        Should -Invoke Get-PcaiCudaCapability -Times 0 -Exactly
        Should -Invoke Invoke-WebRequest -Times 0 -Exactly
        Read-RuntimeFixtureText $script:Canonical | Should -BeExactly $script:OldBytes
        Read-RuntimeFixtureText $script:Alias | Should -BeExactly $script:OldBytes
    }

    # Protects: verified cached bytes select both canonical filenames without network.
    # Detects: legitimate digest/selection compatibility regresses while tightening integrity.
    # Needs: actual tiny cache and two preexisting destination files, whole source function.
    # Breadcrumb: Modules/PcaiInference.psm1 Resolve-PcaiRuntimeVariantDll cache and staging.
    It 'stages a matching cached digest into both canonical files without HTTP' {
        [IO.File]::WriteAllText($script:Variant, $script:ValidBytes)
        $resolved = Resolve-PcaiRuntimeVariantDll -Config $script:RuntimeConfig -ProjectRoot $script:RuntimeFixture
        $resolved | Should -BeExactly (Resolve-Path -LiteralPath $script:Canonical).Path
        Read-RuntimeFixtureText $script:Canonical | Should -BeExactly $script:ValidBytes
        Read-RuntimeFixtureText $script:Alias | Should -BeExactly $script:ValidBytes
        Should -Invoke Invoke-WebRequest -Times 0 -Exactly
    }

    # Protects: a configured digest binds cached bytes before canonical activation.
    # Detects: existing corrupted DLL bypasses sha256 and overwrites the working pair.
    # Needs: actual corrupted cache/expected public digest, no hash/filesystem mocks.
    # Breadcrumb: Modules/PcaiInference.psm1 Resolve-PcaiRuntimeVariantDll existing-file branch.
    It 'refuses corrupted cached bytes and preserves both previous canonical files' {
        [IO.File]::WriteAllText($script:Variant, 'corrupted public runtime fixture')
        $resolved = $null
        try { $resolved = Resolve-PcaiRuntimeVariantDll -Config $script:RuntimeConfig -ProjectRoot $script:RuntimeFixture } catch { }
        Read-RuntimeFixtureText $script:Canonical | Should -BeExactly $script:OldBytes
        Read-RuntimeFixtureText $script:Alias | Should -BeExactly $script:OldBytes
        $resolved | Should -BeNullOrEmpty
        Should -Invoke Invoke-WebRequest -Times 0 -Exactly
    }

    # Protects: a successful verified download remains supported.
    # Detects: repair rejects valid public bytes or leaves canonical aliases inconsistent.
    # Needs: mocked transport writes exact public bytes to its supplied actual OutFile.
    # Breadcrumb: Modules/PcaiInference.psm1 Resolve-PcaiRuntimeVariantDll download/hash success.
    It 'activates a complete matching tiny download' {
        $script:RuntimeConfig = New-RuntimeFixtureConfig -Path $script:Variant -Hash $script:ExpectedHash -Download $true
        Mock Invoke-WebRequest { [IO.File]::WriteAllText($OutFile, $script:ValidBytes) }
        $resolved = Resolve-PcaiRuntimeVariantDll -Config $script:RuntimeConfig -ProjectRoot $script:RuntimeFixture
        $resolved | Should -BeExactly (Resolve-Path -LiteralPath $script:Canonical).Path
        Read-RuntimeFixtureText $script:Canonical | Should -BeExactly $script:ValidBytes
        Read-RuntimeFixtureText $script:Alias | Should -BeExactly $script:ValidBytes
        Should -Invoke Invoke-WebRequest -Times 1 -Exactly
    }

    # Protects: transport failure cannot activate partially written runtime bytes.
    # Detects: warning catch falls through to existence-only staging after prefix+throw.
    # Needs: real prefix file from mocked HTTP boundary, exact preexisting destination bytes.
    # Breadcrumb: Modules/PcaiInference.psm1 Resolve-PcaiRuntimeVariantDll download catch.
    It 'never stages a real partial file left by a throwing transport' {
        $script:RuntimeConfig = New-RuntimeFixtureConfig -Path $script:Variant -Hash $script:ExpectedHash -Download $true
        Mock Invoke-WebRequest { [IO.File]::WriteAllText($OutFile, 'partial'); throw 'public fixture transport failed after prefix' }
        $resolved = $null
        try { $resolved = Resolve-PcaiRuntimeVariantDll -Config $script:RuntimeConfig -ProjectRoot $script:RuntimeFixture } catch { }
        Read-RuntimeFixtureText $script:Canonical | Should -BeExactly $script:OldBytes
        Read-RuntimeFixtureText $script:Alias | Should -BeExactly $script:OldBytes
        $resolved | Should -BeNullOrEmpty
        Should -Invoke Invoke-WebRequest -Times 1 -Exactly
    }

    # Protects: canonical source selection still reconciles its compatibility alias.
    # Detects: source==canonical shortcut silently leaves missing or stale secondary DLL.
    # Needs: real canonical source and missing/stale alias parameter cases.
    # Breadcrumb: Modules/PcaiInference.psm1 Resolve-PcaiRuntimeVariantDll staging shortcut.
    It 'repairs a <Case> alias when the selected source is already canonical' -ForEach @(
        @{ Case = 'missing'; Missing = $true }
        @{ Case = 'stale'; Missing = $false }
    ) {
        [IO.File]::WriteAllText($script:Canonical, $script:ValidBytes)
        if ($Missing) { [IO.File]::Delete($script:Alias) }
        $config = New-RuntimeFixtureConfig -Path $script:Canonical -Hash $script:ExpectedHash
        $resolved = Resolve-PcaiRuntimeVariantDll -Config $config -ProjectRoot $script:RuntimeFixture
        $resolved | Should -BeExactly (Resolve-Path -LiteralPath $script:Canonical).Path
        Test-Path -LiteralPath $script:Alias -PathType Leaf | Should -BeTrue
        Read-RuntimeFixtureText $script:Alias | Should -BeExactly $script:ValidBytes
    }

    # Protects: a failed secondary replacement cannot damage the previous working pair.
    # Detects: first Copy-Item succeeds, second real locked-file copy fails, source is returned.
    # Needs: Windows file sharing, original test-owned read handle retained through assertions.
    # Breadcrumb: Modules/PcaiInference.psm1 Resolve-PcaiRuntimeVariantDll two-copy catch.
    It 'preserves the pair and refuses activation when the secondary destination is locked' -Skip:([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT) {
        [IO.File]::WriteAllText($script:Variant, $script:ValidBytes)
        $lock = [IO.File]::Open($script:Alias, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::Read)
        $originalHandle = $lock.SafeFileHandle
        try {
            $resolved = $null
            try { $resolved = Resolve-PcaiRuntimeVariantDll -Config $script:RuntimeConfig -ProjectRoot $script:RuntimeFixture } catch { }
            Read-RuntimeFixtureText $script:Canonical | Should -BeExactly $script:OldBytes
            Read-RuntimeFixtureText $script:Alias | Should -BeExactly $script:OldBytes
            $resolved | Should -BeNullOrEmpty
        } finally { $lock.Dispose() }
        $originalHandle.IsClosed | Should -BeTrue
    }

    # Protects: digest refusal propagates through the actual caller's fallback search.
    # Detects: resolver rejection/null is followed by acceptance of the same corrupt canonical.
    # Needs: complete caller+variant functions, actual corrupt canonical, inert config/root mocks.
    # Breadcrumb: Modules/PcaiInference.psm1 Resolve-PcaiInferenceDll runtime then canonical fallback.
    It 'does not reaccept a rejected canonical digest through ordinary caller fallback' {
        [IO.File]::WriteAllText($script:Canonical, 'corrupted public runtime fixture')
        $script:RuntimeConfig = New-RuntimeFixtureConfig -Path $script:Canonical -Hash $script:ExpectedHash
        $resolved = $null
        try { $resolved = Resolve-PcaiInferenceDll } catch { }
        $resolved | Should -BeNullOrEmpty
        Read-RuntimeFixtureText $script:Alias | Should -BeExactly $script:OldBytes
    }

    # Protects: malformed supplied SHA cannot downgrade an enabled selection to unchecked.
    # Detects: non-string hash ToLowerInvariant failure is caught after download then activated.
    # Needs: numeric configured hash, actual tiny transport bytes, no hash/parser mock.
    # Breadcrumb: Modules/PcaiInference.psm1 Resolve-PcaiRuntimeVariantDll supplied sha256 handling.
    It 'refuses a malformed nonempty digest before transport or canonical writes' {
        $script:RuntimeConfig = New-RuntimeFixtureConfig -Path $script:Variant -Hash 17 -Download $true
        Mock Invoke-WebRequest { [IO.File]::WriteAllText($OutFile, $script:ValidBytes) }
        $resolved = $null
        try { $resolved = Resolve-PcaiRuntimeVariantDll -Config $script:RuntimeConfig -ProjectRoot $script:RuntimeFixture } catch { }
        Should -Invoke Invoke-WebRequest -Times 0 -Exactly
        Read-RuntimeFixtureText $script:Canonical | Should -BeExactly $script:OldBytes
        Read-RuntimeFixtureText $script:Alias | Should -BeExactly $script:OldBytes
        $resolved | Should -BeNullOrEmpty
    }

    # Protects: DLL selection returns a literal file rather than a directory.
    # Detects: Test-Path existence admits a directory and staging catch returns it as a DLL.
    # Needs: actual directory at the configured variant path; no Copy-Item/filesystem mock.
    # Breadcrumb: Modules/PcaiInference.psm1 Resolve-PcaiRuntimeVariantDll existence-only guards.
    It 'refuses a directory masquerading as the configured DLL' {
        [void][IO.Directory]::CreateDirectory($script:Variant)
        $resolved = $null
        try { $resolved = Resolve-PcaiRuntimeVariantDll -Config $script:RuntimeConfig -ProjectRoot $script:RuntimeFixture } catch { }
        $resolved | Should -BeNullOrEmpty
        Read-RuntimeFixtureText $script:Canonical | Should -BeExactly $script:OldBytes
        Read-RuntimeFixtureText $script:Alias | Should -BeExactly $script:OldBytes
    }

    # Protects: documented no-SHA download compatibility remains explicit and unverified.
    # Detects: integrity repair silently makes optional SHA mandatory or suppresses its warning.
    # Needs: complete valid public download with null SHA; warning stream captured, no real HTTP.
    # Breadcrumb: Modules/PcaiInference.psm1 Resolve-PcaiRuntimeVariantDll missing-hash warning.
    It 'retains an explicitly unverified missing-SHA download with its warning' {
        $script:RuntimeConfig = New-RuntimeFixtureConfig -Path $script:Variant -Hash $null -Download $true
        Mock Invoke-WebRequest { [IO.File]::WriteAllText($OutFile, $script:ValidBytes) }
        $records = @(Resolve-PcaiRuntimeVariantDll -Config $script:RuntimeConfig -ProjectRoot $script:RuntimeFixture 3>&1)
        @($records | Where-Object { $_ -is [Management.Automation.WarningRecord] -and $_.Message -like '*without verification*' }).Count | Should -Be 1
        @($records | Where-Object { $_ -is [string] }) | Should -Contain (Resolve-Path -LiteralPath $script:Canonical).Path
        Read-RuntimeFixtureText $script:Canonical | Should -BeExactly $script:ValidBytes
        Read-RuntimeFixtureText $script:Alias | Should -BeExactly $script:ValidBytes
    }

    # Protects: supplied explicit override intentionally bypasses configuration/download selection.
    # Detects: repair consults latent auto-download or GPU discovery for an explicit file.
    # Needs: complete caller and real override marker file; no bridge/native engine load.
    # Breadcrumb: Modules/PcaiInference.psm1 Resolve-PcaiInferenceDll OverridePath branch.
    It 'preserves explicit override selection without configuration or GPU lookup' {
        [IO.File]::WriteAllText($script:Variant, $script:ValidBytes)
        Resolve-PcaiInferenceDll -OverridePath $script:Variant | Should -BeExactly (Resolve-Path -LiteralPath $script:Variant).Path
        Should -Invoke Get-PcaiConfig -Times 0 -Exactly
        Should -Invoke Get-PcaiCudaCapability -Times 0 -Exactly
        Should -Invoke Invoke-WebRequest -Times 0 -Exactly
    }

    # Protects: explicit paired bundle remains authoritative before legacy selection.
    # Detects: repair falls back to config/download instead of returning the selected paired root.
    # Needs: actual inert PcaiNative.dll/pcai_inference.dll files, no Assembly or NativeLibrary calls.
    # Breadcrumb: Modules/PcaiInference.psm1 Resolve-PcaiInferenceDll explicit bundle branch.
    It 'preserves the complete explicit bundle without loading either DLL' {
        [IO.File]::WriteAllText((Join-Path $script:RuntimeFixture 'bin/PcaiNative.dll'), 'managed selection marker')
        [Environment]::SetEnvironmentVariable('PCAI_NATIVE_BUNDLE_ROOT', (Split-Path -Parent $script:Canonical), 'Process')
        Resolve-PcaiInferenceDll | Should -BeExactly (Resolve-Path -LiteralPath $script:Canonical).ProviderPath
        Should -Invoke Get-PcaiConfig -Times 0 -Exactly
        Should -Invoke Get-PcaiCudaCapability -Times 0 -Exactly
        Should -Invoke Invoke-WebRequest -Times 0 -Exactly
    }

    # Protects: generic and ordered optional maps keep absent selectors a non-mutating no-op.
    # Detects: generic Contains binding and literal Keys/Count adapter collisions.
    # Needs: real declared IDictionary objects, complete resolver and inert collaborator counters.
    # Breadcrumb: Modules/PcaiInference.psm1 Resolve-PcaiRuntimeVariantDll Get-RuntimeOptionalValue.
    It 'preserves the pair and skips discovery for missing <MappingKind> dictionary configuration' -ForEach @(
        @{ MappingKind = 'Generic' }
        @{ MappingKind = 'Ordered' }
    ) {
        $configuration = New-RuntimeFixtureMapping $MappingKind @{ Keys='literal metadata'; Count=7 }
        Resolve-PcaiRuntimeVariantDll -Config $configuration -ProjectRoot $script:RuntimeFixture | Should -BeNullOrEmpty
        Read-RuntimeFixtureText $script:Canonical | Should -BeExactly $script:OldBytes
        Read-RuntimeFixtureText $script:Alias | Should -BeExactly $script:OldBytes
        Should -Invoke Get-PcaiCudaCapability -Times 0 -Exactly
        Should -Invoke Invoke-WebRequest -Times 0 -Exactly
    }

    # Protects: mixed-case string keys select verified cached bytes in both supported map kinds.
    # Detects: querying a normalized name instead of the actual key, lost zero values or Keys metadata.
    # Needs: nested case-sensitive Generic/Ordered maps, actual tiny cached bytes and canonical pair.
    # Breadcrumb: Modules/PcaiInference.psm1 Resolve-PcaiRuntimeVariantDll actual matched-key access.
    It 'activates exact cached bytes using mixed-case nested <MappingKind> dictionary keys' -ForEach @(
        @{ MappingKind = 'Generic' }
        @{ MappingKind = 'Ordered' }
    ) {
        [IO.File]::WriteAllText($script:Variant, $script:ValidBytes)
        $configuration = New-RuntimeFixtureMappingConfig $MappingKind
        Resolve-PcaiRuntimeVariantDll -Config $configuration -ProjectRoot $script:RuntimeFixture | Should -BeExactly $script:Canonical
        Read-RuntimeFixtureText $script:Variant | Should -BeExactly $script:ValidBytes
        Read-RuntimeFixtureText $script:Canonical | Should -BeExactly $script:ValidBytes
        Read-RuntimeFixtureText $script:Alias | Should -BeExactly $script:ValidBytes
        Should -Invoke Get-PcaiCudaCapability -Times 1 -Exactly
        Should -Invoke Invoke-WebRequest -Times 0 -Exactly
    }

    # Protects: map compatibility cannot bypass digest refusal or damage the previous pair.
    # Detects: ignored supplied SHA, Contains binder failure instead of the actual integrity cause.
    # Needs: complete production resolver, real corrupt cache and two known previous files.
    # Breadcrumb: Modules/PcaiInference.psm1 Resolve-PcaiRuntimeVariantDll Install-RuntimePair digest gate.
    It 'refuses actual corrupt cached bytes and preserves the pair for <MappingKind> dictionary configuration' -ForEach @(
        @{ MappingKind = 'Generic' }
        @{ MappingKind = 'Ordered' }
    ) {
        [IO.File]::WriteAllText($script:Variant, 'actual corrupted public bytes')
        $configuration = New-RuntimeFixtureMappingConfig $MappingKind
        { Resolve-PcaiRuntimeVariantDll -Config $configuration -ProjectRoot $script:RuntimeFixture } | Should -Throw '*SHA256 hash mismatch*'
        Read-RuntimeFixtureText $script:Canonical | Should -BeExactly $script:OldBytes
        Read-RuntimeFixtureText $script:Alias | Should -BeExactly $script:OldBytes
        Read-RuntimeFixtureText $script:Variant | Should -BeExactly 'actual corrupted public bytes'
        Should -Invoke Invoke-WebRequest -Times 0 -Exactly
    }

    # Protects: case-insensitive optional lookup cannot silently choose between conflicting keys.
    # Detects: insertion-order-dependent acceptance of case-sensitive maps with duplicate folded names.
    # Needs: actual Generic/Ordered maps allowing both exact key spellings and real unchanged files.
    # Breadcrumb: Modules/PcaiInference.psm1 Get-RuntimeOptionalValue ambiguity refusal.
    It 'refuses ambiguous folded names before discovery or mutation for <MappingKind> dictionary configuration' -ForEach @(
        @{ MappingKind = 'Generic' }
        @{ MappingKind = 'Ordered' }
    ) {
        $configuration = New-RuntimeFixtureMapping $MappingKind @{}
        $configuration.Add('nativeInference', [pscustomobject]@{})
        $configuration.Add('NATIVEINFERENCE', [pscustomobject]@{})
        { Resolve-PcaiRuntimeVariantDll -Config $configuration -ProjectRoot $script:RuntimeFixture } | Should -Throw '*Ambiguous runtime configuration key*'
        Read-RuntimeFixtureText $script:Canonical | Should -BeExactly $script:OldBytes
        Read-RuntimeFixtureText $script:Alias | Should -BeExactly $script:OldBytes
        Should -Invoke Get-PcaiCudaCapability -Times 0 -Exactly
        Should -Invoke Invoke-WebRequest -Times 0 -Exactly
    }
}
