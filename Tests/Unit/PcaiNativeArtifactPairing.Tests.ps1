#Requires -Version 7.0
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }

# Contract for how Initialize-PcaiNative pairs PcaiNative.dll (the managed wrapper) with pcai_core_lib (the Rust core).
# The two ship as a matched ABI pair, so a wrapper from one directory loaded against a core from another can fail with
# "PCAI Core library loaded but not functional". See Modules/PC-AI.Acceleration/Private/Initialize-PcaiNative.ps1.
#
# These tests never load a real native library: Add-Type, the managed load boundary, is mocked, and the paths the
# producer publishes before that boundary are the observable behaviour. This file is separate from
# PcaiNativeInitializationContracts.Tests.ps1 because that fixture needs an isolated child process and a
# PCAI_NATIVE_BUNDLE_ROOT contract the producer does not implement.

# Evaluated at discovery so -Skip can use it: another file in the same session may have loaded the real wrapper.
$script:AssemblyAlreadyLoaded = [bool]([AppDomain]::CurrentDomain.GetAssemblies() | Where-Object { $_.GetName().Name -eq 'PcaiNative' })

BeforeAll {
    $script:ProducerPath = Join-Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) 'Modules/PC-AI.Acceleration/Private/Initialize-PcaiNative.ps1'
    . $script:ProducerPath
    # Resolve-PcaiRepoRoot lives in another module; the producer only calls it when it exists.
    function Resolve-PcaiRepoRoot { param($StartPath) $script:OwnedRepo }
    $script:CoreName = Get-PcaiNativeLibraryFileName -LibraryName 'pcai_core_lib'
    $script:Rid = Get-PcaiNativeRuntimeIdentifier
    $script:EnvironmentNames = @('PCAI_CORE_LIB_PATH', 'PCAI_TEST_CORE_DLL', 'PCAI_NATIVE_DIR', 'PCAI_ROOT', 'LOCALAPPDATA', 'PATH')

    function New-ArtifactDirectory {
        param([Parameter(Mandatory)][string]$Path, [switch]$Wrapper, [switch]$Core, [switch]$RuntimeCore)
        $null = [IO.Directory]::CreateDirectory($Path)
        if ($Wrapper) { [IO.File]::WriteAllText((Join-Path $Path 'PcaiNative.dll'), 'inert wrapper marker') }
        if ($Core) { [IO.File]::WriteAllText((Join-Path $Path $script:CoreName), 'inert core marker') }
        if ($RuntimeCore) {
            $native = Join-Path $Path "runtimes/$($script:Rid)/native"
            $null = [IO.Directory]::CreateDirectory($native)
            [IO.File]::WriteAllText((Join-Path $native $script:CoreName), 'inert runtime core marker')
        }
        $Path
    }
}

Describe 'Select-PcaiNativeArtifactPair' -Tag 'Unit', 'Acceleration', 'Portable' {
    BeforeEach {
        $script:Root = Join-Path $TestDrive ([guid]::NewGuid().ToString('N'))
    }

    # Why: the reported bug was a wrapper paired with a core left over from a different, earlier directory.
    It 'returns the first directory holding both artifacts even when an earlier directory has only a core' {
        $stale = New-ArtifactDirectory -Path (Join-Path $script:Root 'stale') -Core
        $pair = New-ArtifactDirectory -Path (Join-Path $script:Root 'pair') -Wrapper -Core
        $selection = Select-PcaiNativeArtifactPair -SearchPaths @($stale, $pair) -CoreFileName $script:CoreName -RuntimeIdentifier $script:Rid
        $selection.WrapperPath | Should -BeExactly (Join-Path $pair 'PcaiNative.dll')
        $selection.CoreLibPath | Should -BeExactly (Join-Path $pair $script:CoreName)
        $selection.CoLocated | Should -BeTrue
    }

    # Why: a published runtimes/<rid>/native layout is a legitimate core location and must count as co-located.
    It 'treats a core under runtimes/<rid>/native as co-located with its wrapper' {
        $stale = New-ArtifactDirectory -Path (Join-Path $script:Root 'stale') -Core
        $pair = New-ArtifactDirectory -Path (Join-Path $script:Root 'runtime-pair') -Wrapper -RuntimeCore
        $selection = Select-PcaiNativeArtifactPair -SearchPaths @($stale, $pair) -CoreFileName $script:CoreName -RuntimeIdentifier $script:Rid
        $selection.WrapperPath | Should -BeExactly (Join-Path $pair 'PcaiNative.dll')
        $selection.CoreLibPath | Should -BeExactly (Join-Path $pair "runtimes/$($script:Rid)/native/$($script:CoreName)")
        $selection.CoLocated | Should -BeTrue
    }

    # Why: when only mismatched halves exist the loader must still try them, as it did before, but say they are split.
    It 'falls back to the first wrapper and first core across directories and reports the split in the verbose stream' {
        $coreOnly = New-ArtifactDirectory -Path (Join-Path $script:Root 'core-only') -Core
        $wrapperOnly = New-ArtifactDirectory -Path (Join-Path $script:Root 'wrapper-only') -Wrapper
        $records = Select-PcaiNativeArtifactPair -SearchPaths @($coreOnly, $wrapperOnly) -CoreFileName $script:CoreName -RuntimeIdentifier $script:Rid -Verbose 4>&1
        $selection = $records | Where-Object { $_ -isnot [System.Management.Automation.VerboseRecord] }
        $verbose = @($records | Where-Object { $_ -is [System.Management.Automation.VerboseRecord] } | ForEach-Object { $_.Message })
        $selection.WrapperPath | Should -BeExactly (Join-Path $wrapperOnly 'PcaiNative.dll')
        $selection.CoreLibPath | Should -BeExactly (Join-Path $coreOnly $script:CoreName)
        $selection.CoLocated | Should -BeFalse
        ($verbose -join "`n") | Should -Match 'cross-directory'
    }

    # Why: an explicit core override is authoritative and must survive pairing, as it did before the two-pass change.
    It 'keeps an explicit core path while taking the wrapper from the first directory that holds a core and wrapper' {
        $explicit = New-ArtifactDirectory -Path (Join-Path $script:Root 'explicit') -Core
        $stale = New-ArtifactDirectory -Path (Join-Path $script:Root 'stale') -Core
        $pair = New-ArtifactDirectory -Path (Join-Path $script:Root 'pair') -Wrapper -Core
        $explicitCore = Join-Path $explicit $script:CoreName
        $selection = Select-PcaiNativeArtifactPair -SearchPaths @($stale, $pair) -CoreFileName $script:CoreName -RuntimeIdentifier $script:Rid -ExplicitCorePath $explicitCore
        $selection.WrapperPath | Should -BeExactly (Join-Path $pair 'PcaiNative.dll')
        $selection.CoreLibPath | Should -BeExactly $explicitCore
    }

    # Why: callers test for null to decide the artifacts are missing, so an empty search must not invent a path.
    It 'returns no paths when nothing is found' {
        $empty = New-ArtifactDirectory -Path (Join-Path $script:Root 'empty')
        $selection = Select-PcaiNativeArtifactPair -SearchPaths @($empty) -CoreFileName $script:CoreName -RuntimeIdentifier $script:Rid
        $selection.WrapperPath | Should -BeNullOrEmpty
        $selection.CoreLibPath | Should -BeNullOrEmpty
        $selection.CoLocated | Should -BeFalse
    }
}

Describe 'Initialize-PcaiNative artifact resolution' -Tag 'Unit', 'Acceleration', 'Portable' {
    BeforeEach {
        $script:SavedEnvironment = @{}
        foreach ($name in $script:EnvironmentNames) {
            $script:SavedEnvironment[$name] = [Environment]::GetEnvironmentVariable($name)
            if ($name -ne 'PATH') { [Environment]::SetEnvironmentVariable($name, $null) }
        }
        $script:OwnedRepo = Join-Path $TestDrive ('repo-' + [guid]::NewGuid().ToString('N'))
        $null = [IO.Directory]::CreateDirectory($script:OwnedRepo)
        foreach ($name in @('PcaiNativeVersion', 'PcaiNativeDllPath', 'PcaiNativeCoreLibraryPath', 'PcaiNativeWrapperPath', 'PcaiNativeArtifactHash', 'PcaiNativeRid')) {
            Set-Variable -Name $name -Value $null -Scope Script
        }
        $script:PcaiNativeLoaded = $false
        Mock Get-Module { $null }
        # The managed load boundary: refusing here keeps every native type out of the process while the
        # producer has already published the paths it selected.
        Mock Add-Type { throw 'inert load boundary' }
    }
    AfterEach {
        foreach ($name in $script:EnvironmentNames) { [Environment]::SetEnvironmentVariable($name, $script:SavedEnvironment[$name]) }
    }

    # Why: this is the dtm-p1gen9 failure end to end; the Release wrapper was paired with a stale core from an earlier path.
    It 'publishes a wrapper and core from the same directory when an earlier search path holds only a core' -Skip:$script:AssemblyAlreadyLoaded {
        $null = New-ArtifactDirectory -Path (Join-Path $script:OwnedRepo 'bin') -Core
        $pair = New-ArtifactDirectory -Path (Join-Path $script:OwnedRepo 'Native/PcaiNative/bin/Release/net8.0') -Wrapper -Core
        Initialize-PcaiNative -WarningAction SilentlyContinue | Should -BeFalse
        $script:PcaiNativeWrapperPath | Should -BeExactly (Join-Path $pair 'PcaiNative.dll')
        $script:PcaiNativeCoreLibraryPath | Should -BeExactly (Join-Path $pair $script:CoreName)
        $script:PcaiNativeDllPath | Should -BeExactly $pair
        Should -Invoke Add-Type -Times 1 -Exactly -ParameterFilter { $Path -eq (Join-Path $pair 'PcaiNative.dll') }
    }

    # Why: the installed bridge is only visible to Get-Module when its parent is on PSModulePath, which is often not the case.
    It 'finds the installed bridge bin under LOCALAPPDATA when Get-Module cannot see the module' -Skip:$script:AssemblyAlreadyLoaded {
        $localAppData = Join-Path $TestDrive ('localappdata-' + [guid]::NewGuid().ToString('N'))
        $bridge = New-ArtifactDirectory -Path (Join-Path $localAppData 'PowerShell/Modules/PcaiNativeBridge/bin') -Wrapper -Core
        $env:LOCALAPPDATA = $localAppData
        Initialize-PcaiNative -WarningAction SilentlyContinue | Should -BeFalse
        $script:PcaiNativeWrapperPath | Should -BeExactly (Join-Path $bridge 'PcaiNative.dll')
        $script:PcaiNativeCoreLibraryPath | Should -BeExactly (Join-Path $bridge $script:CoreName)
    }

    # Why: LOCALAPPDATA is unset on Linux and macOS, and Join-Path would throw on a null base.
    It 'initializes without error when LOCALAPPDATA is not set' -Skip:$script:AssemblyAlreadyLoaded {
        $pair = New-ArtifactDirectory -Path (Join-Path $script:OwnedRepo 'bin') -Wrapper -Core
        $env:LOCALAPPDATA | Should -BeNullOrEmpty
        { Initialize-PcaiNative -WarningAction SilentlyContinue } | Should -Not -Throw
        $script:PcaiNativeWrapperPath | Should -BeExactly (Join-Path $pair 'PcaiNative.dll')
    }
}
