#Requires -Version 5.1
param([string]$ValidationSourcePath=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../Modules/PC-AI.Evaluation/ValidateDependencies.ps1')))
# The complete actual validation script runs against tiny owned files. DLL/EXE
# contents are availability markers only: nothing is imported, loaded or launched.
BeforeAll {
    if($PSVersionTable.PSVersion.Major -lt 7){return}
    $script:dependencySourceBytes=[IO.File]::ReadAllBytes($ValidationSourcePath)
    $script:dependencySourceSHA256=(Get-FileHash -LiteralPath $ValidationSourcePath).Hash
    $script:ambientDependencyRoot=[IO.Path]::GetFullPath((Join-Path ([Environment]::GetFolderPath('UserProfile')) '.local/bin'))
    function New-OwnedDependencyFixture([string]$Name,[AllowNull()][object]$Json){
        $fixtureBase=Join-Path $TestDrive $Name
        if(Test-Path -LiteralPath $fixtureBase){throw 'Fresh dependency fixture collision.'}
        $moduleDirectory=Join-Path $fixtureBase 'Modules/PC-AI.Evaluation'
        [void][IO.Directory]::CreateDirectory($moduleDirectory)
        [void][IO.Directory]::CreateDirectory((Join-Path $fixtureBase 'Config'))
        $validator=Join-Path $moduleDirectory 'ValidateDependencies.ps1'
        [IO.File]::WriteAllBytes($validator,$script:dependencySourceBytes)
        if((Get-FileHash -LiteralPath $validator).Hash-cne$script:dependencySourceSHA256){throw 'Actual full validation script copy differs.'}
        [IO.File]::WriteAllText((Join-Path $fixtureBase 'Modules/PcaiInference.psm1'),'# inert metadata marker',[Text.UTF8Encoding]::new($false))
        $fixtureConfigPath=Join-Path $fixtureBase 'Config/llm-config.json'
        if($null-ne$Json){[IO.File]::WriteAllText($fixtureConfigPath,$Json,[Text.UTF8Encoding]::new($false))}
        [pscustomobject]@{Root=$fixtureBase;Validator=$validator;ConfigPath=$fixtureConfigPath}
    }
    function Write-OwnedDependencyFile([string]$Path){
        [void][IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($Path))
        [IO.File]::WriteAllBytes($Path,[byte[]]@(1,2,3))
    }
    function Invoke-OwnedDependencyValidation($Fixture,[switch]$SeedStaleDllPath){
        $configBefore=if(Test-Path -LiteralPath $Fixture.ConfigPath){(Get-FileHash -LiteralPath $Fixture.ConfigPath).Hash}else{$null}
        $records=@(& {
            param([string]$ActualValidator,[bool]$Seed)
            Set-StrictMode -Version Latest
            if($Seed){$script:PcaiDllPath='stale-predecessor.dll'}
            . $ActualValidator
            [pscustomobject]$script:DependencyStatus
        } $Fixture.Validator $SeedStaleDllPath.IsPresent 3>&1)
        $statuses=@($records|Where-Object{$_ -isnot [Management.Automation.WarningRecord]})
        if($statuses.Count-ne1){throw 'Validation did not emit one actual status record.'}
        $configAfter=if(Test-Path -LiteralPath $Fixture.ConfigPath){(Get-FileHash -LiteralPath $Fixture.ConfigPath).Hash}else{$null}
        if($configBefore-cne$configAfter-or(Get-FileHash -LiteralPath $Fixture.Validator).Hash-cne$script:dependencySourceSHA256){throw 'Actual validation changed its input bytes.'}
        [pscustomobject]@{Status=$statuses[0];Warnings=@($records|Where-Object{$_ -is [Management.Automation.WarningRecord]}|ForEach-Object{$_.Message});InputsUnchanged=$true}
    }
}
Describe 'Actual Evaluation optional dependency configuration under StrictMode' -Tag 'Unit','Evaluation','Portable' -Skip:($PSVersionTable.PSVersion.Major -lt 7) {
    BeforeEach {
        # Only installed user dependency discovery is suppressed. Owned config,
        # validation script and native-path marker probes use the real filesystem.
        Mock Test-Path {$false} -ParameterFilter {
            $queryPath=if($LiteralPath){[string](@($LiteralPath)[0])}elseif($Path){[string](@($Path)[0])}else{''}
            $queryPath -and ([IO.Path]::GetFullPath($queryPath).Equals($script:ambientDependencyRoot,[StringComparison]::OrdinalIgnoreCase)-or[IO.Path]::GetFullPath($queryPath).StartsWith($script:ambientDependencyRoot+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase))
        }
    }
    # Protects: provider-only config can import the offline Evaluation surface.
    # Detects: nativeInference/evaluation missing-property errors under StrictMode.
    # Needs: actual full script, owned provider JSON and no ambient dependency files.
    # Breadcrumb: Modules/PC-AI.Evaluation/ValidateDependencies.ps1 optional sections.
    It 'accepts provider-only configuration and reports absent native dependencies' {
        $fixture=New-OwnedDependencyFixture 'providers' '{"providers":{"example":{"baseUrl":"http://localhost:1"}}}'
        $result=Invoke-OwnedDependencyValidation $fixture
        $result.Status.DllAvailable|Should -BeFalse
        $result.Status.DllPath|Should -BeNullOrEmpty
        $result.Status.ModuleAvailable|Should -BeTrue
        $result.Status.CompiledBackends.LlamaCppExe|Should -BeNullOrEmpty
        $result.Status.CompiledBackends.MistralRsExe|Should -BeNullOrEmpty
        $result.InputsUnchanged|Should -BeTrue
    }
    # Protects: missing, empty, partial and null optional settings retain defaults.
    # Detects: StrictMode dereferencing absent search-path properties or null sections.
    # Needs: group fixtures missing-file, empty-object, partial and explicit-null JSON.
    # Breadcrumb: Modules/PC-AI.Evaluation/ValidateDependencies.ps1 config presence checks.
    It 'accepts optional configuration shape <Case>' -ForEach @(
        @{Case='missing-file';Json=$null},
        @{Case='empty-object';Json='{}'},
        @{Case='partial';Json='{"nativeInference":{},"evaluation":{}}'},
        @{Case='null-sections';Json='{"nativeInference":null,"evaluation":null}'},
        @{Case='null-paths';Json='{"nativeInference":{"dllSearchPaths":null},"evaluation":{"binSearchPaths":null}}'}
    ) {
        $result=Invoke-OwnedDependencyValidation (New-OwnedDependencyFixture $Case $Json)
        $result.Status.DllAvailable|Should -BeFalse
        $result.Status.DllPath|Should -BeNullOrEmpty
        $result.Status.ModuleAvailable|Should -BeTrue
    }
    # Protects: metadata validation recognizes explicit relative DLL and backend files.
    # Detects: losing configured paths while repairing optional-section access.
    # Needs: three nonexecuted tiny files inside the owned fixture repository.
    # Breadcrumb: Modules/PC-AI.Evaluation/ValidateDependencies.ps1 configured searches.
    It 'discovers relative configured DLL and both backend files without loading them' {
        $fixture=New-OwnedDependencyFixture 'relative' '{"nativeInference":{"dllSearchPaths":["native/pcai_inference.dll"]},"evaluation":{"binSearchPaths":["native"]}}'
        $dll=Join-Path $fixture.Root 'native/pcai_inference.dll'
        $llama=Join-Path $fixture.Root 'native/pcai-llamacpp.exe'
        $mistral=Join-Path $fixture.Root 'native/pcai-mistralrs.exe'
        foreach($marker in @($dll,$llama,$mistral)){Write-OwnedDependencyFile $marker}
        $result=Invoke-OwnedDependencyValidation $fixture
        $result.Status.DllAvailable|Should -BeTrue
        $result.Status.DllPath|Should -BeExactly $dll
        $result.Status.CompiledBackends.LlamaCppExe|Should -BeExactly $llama
        $result.Status.CompiledBackends.MistralRsExe|Should -BeExactly $mistral
    }
    # Protects: absolute and scalar configured paths keep their existing contract.
    # Detects: path coercion or enumeration that breaks explicit machine configuration.
    # Needs: JSON serialized from real owned absolute paths and one marker DLL.
    # Breadcrumb: Modules/PC-AI.Evaluation/ValidateDependencies.ps1 rooted path selection.
    It 'discovers a scalar absolute configured DLL and directory containing brackets' {
        $fixture=New-OwnedDependencyFixture 'absolute' '{}'
        $dll=Join-Path $fixture.Root 'native[owned]/pcai_inference.dll'
        Write-OwnedDependencyFile $dll
        [IO.File]::WriteAllText($fixture.ConfigPath,(@{nativeInference=@{dllSearchPaths=$dll};evaluation=@{binSearchPaths=@()}}|ConvertTo-Json -Depth 4),[Text.UTF8Encoding]::new($false))
        $result=Invoke-OwnedDependencyValidation $fixture
        $result.Status.DllAvailable|Should -BeTrue
        $result.Status.DllPath|Should -BeExactly $dll
    }
    # Protects: optional native dependencies remain absent when only directories exist.
    # Detects: Test-Path existence falsely accepting DLL, EXE or module directories.
    # Needs: owned directories with dependency-like names, no binary execution.
    # Breadcrumb: Modules/PC-AI.Evaluation/ValidateDependencies.ps1 leaf-only discovery.
    It 'rejects dependency-named directories as DLL backend and module files' {
        $fixture=New-OwnedDependencyFixture 'directories' '{"nativeInference":{"dllSearchPaths":["native/pcai_inference.dll"]},"evaluation":{"binSearchPaths":["native"]}}'
        foreach($relative in @('native/pcai_inference.dll','native/pcai-llamacpp.exe','native/pcai-mistralrs.exe')){[void][IO.Directory]::CreateDirectory((Join-Path $fixture.Root $relative))}
        [IO.File]::Delete((Join-Path $fixture.Root 'Modules/PcaiInference.psm1'))
        [void][IO.Directory]::CreateDirectory((Join-Path $fixture.Root 'Modules/PcaiInference.psm1'))
        $result=Invoke-OwnedDependencyValidation $fixture
        $result.Status.DllAvailable|Should -BeFalse
        $result.Status.ModuleAvailable|Should -BeFalse
        $result.Status.CompiledBackends.LlamaCppExe|Should -BeNullOrEmpty
        $result.Status.CompiledBackends.MistralRsExe|Should -BeNullOrEmpty
    }
    # Protects: successive validation cannot publish an unavailable stale DLL path.
    # Detects: undefined or retained script:PcaiDllPath when discovery finds no file.
    # Needs: seeded predecessor state plus owned config and absent native markers.
    # Breadcrumb: Modules/PC-AI.Evaluation/ValidateDependencies.ps1 result initialization.
    It 'clears stale DLL path before reporting unavailable dependencies' {
        $fixture=New-OwnedDependencyFixture 'stale' '{"nativeInference":{"dllSearchPaths":[]},"evaluation":{"binSearchPaths":[]}}'
        $result=Invoke-OwnedDependencyValidation $fixture -SeedStaleDllPath
        $result.Status.DllAvailable|Should -BeFalse
        $result.Status.DllPath|Should -BeNullOrEmpty
    }
    # Protects: documentation/offline imports survive malformed optional JSON honestly.
    # Detects: silent parse failure, false availability or a new fatal import contract.
    # Needs: malformed JSON and nonobject-root group fixtures, no live configuration.
    # Breadcrumb: Modules/PC-AI.Evaluation/ValidateDependencies.ps1 warning/default branch.
    It 'warns and uses defaults for malformed configuration <Case>' -ForEach @(
        @{Case='invalid-json';Json='{"nativeInference":'},
        @{Case='array-root';Json='[]'}
    ) {
        $result=Invoke-OwnedDependencyValidation (New-OwnedDependencyFixture $Case $Json)
        $result.Status.DllAvailable|Should -BeFalse
        @($result.Warnings|Where-Object{$_ -like 'Failed to parse dependency configuration*Using default search paths.'}).Count|Should -Be 1
    }
    # Protects: malformed optional sections cannot fabricate dependency paths.
    # Detects: coerced object/number paths or StrictMode shape errors on invalid JSON values.
    # Needs: wrong-section and invalid-path group fixtures with real absent marker files.
    # Breadcrumb: Modules/PC-AI.Evaluation/ValidateDependencies.ps1 configured path validation.
    It 'warns and refuses invalid configured paths <Case>' -ForEach @(
        @{Case='wrong-section';Json='{"nativeInference":false,"evaluation":17}';WarningPattern='Dependency configuration section*'},
        @{Case='wrong-paths';Json='{"nativeInference":{"dllSearchPaths":[17,null," ",{}]},"evaluation":{"binSearchPaths":[false]}}';WarningPattern='Dependency configuration*contains an invalid path*'}
    ) {
        $result=Invoke-OwnedDependencyValidation (New-OwnedDependencyFixture $Case $Json)
        $result.Status.DllAvailable|Should -BeFalse
        $result.Status.CompiledBackends.LlamaCppExe|Should -BeNullOrEmpty
        $result.Status.CompiledBackends.MistralRsExe|Should -BeNullOrEmpty
        @($result.Warnings|Where-Object{$_ -like $WarningPattern}).Count|Should -BeGreaterThan 0
    }
}
