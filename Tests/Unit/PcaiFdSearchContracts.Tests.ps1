#Requires -Version 5.1
BeforeAll {
    # NativeCommandExitException preference behavior requires PowerShell7.3+.
    # Legacy compatibility discovery reports these cases SKIP, never PASS.
    if ($PSVersionTable.PSVersion -lt [version]'7.3') { return }
    $script:SavedNativeSearchPreference = $env:PCAI_PREFER_NATIVE_SEARCH
    $script:SavedNativeErrorPreference = $PSNativeCommandUseErrorActionPreference
    $repo = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
    . (Join-Path $repo 'Modules/PC-AI.Acceleration/Public/Find-FilesFast.ps1')
    # Use maintained path helpers without importing the native bridge or profile.
    $tokens = $null
    $parseErrors = $null
    $moduleAst = [Management.Automation.Language.Parser]::ParseFile(
        (Join-Path $repo 'Modules/PC-AI.Acceleration/PC-AI.Acceleration.psm1'),
        [ref]$tokens, [ref]$parseErrors)
    if ($parseErrors.Count) { throw 'Acceleration module cannot be parsed.' }
    foreach ($name in @('Resolve-PcaiPath', 'New-PcaiPathItem')) {
        $definitions = @($moduleAst.FindAll({
            param($node)
            $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -ceq $name
        }, $true))
        if ($definitions.Count -ne 1) { throw "Expected one maintained $name helper." }
        . ([ScriptBlock]::Create($definitions[0].Extent.Text))
    }
    $script:QualifiedFd = @(Get-Command fd -CommandType Application -ErrorAction Stop)[0].Source
    if (-not [IO.File]::Exists($script:QualifiedFd)) { throw 'Native fd is required; CI installs the pinned release.' }
    # Cache spies and backend routing are deliberate collaborators. The actual
    # dispatcher, fd invocation and path mapping run on three tiny owned files.
    function Get-RustToolPath { param($ToolName) $script:SelectedFd }
    function Get-PcaiCacheKey { param($Category, $Parameters) 'fd-contract-fixture' }
    function Get-PcaiCachedValue { param($Key, $TtlSeconds) $null }
    function Set-PcaiCachedValue {
        param($Key, $Value)
        $script:FdCacheCalls++
        $script:FdCachedCount = @($Value).Count
        $Value
    }
    function Get-FdContractObservation {
        param([scriptblock]$Action)
        $rows = [Collections.Generic.List[object]]::new()
        $failure = $null
        try { & $Action | ForEach-Object { $rows.Add($_) } }
        catch { $failure = $_ }
        [pscustomobject]@{ Rows = @($rows); ErrorRecord = $failure }
    }
    function Get-FdContractExitCode {
        param($ErrorRecord)
        $exception = $ErrorRecord.Exception
        while ($exception) {
            if ($exception.Data.Contains('ExitCode')) { return [int]$exception.Data['ExitCode'] }
            if ($exception.PSObject.Properties['ExitCode']) { return [int]$exception.ExitCode }
            $exception = $exception.InnerException
        }
        return $null
    }
    function Get-FdContractLocationState {
        $location = Microsoft.PowerShell.Management\Get-Location -ErrorAction Stop
        $stack = Microsoft.PowerShell.Management\Get-Location -Stack -ErrorAction Stop
        $frames = @(foreach ($frame in $stack) {
            [ordered]@{ Path = $frame.Path; ProviderPath = $frame.ProviderPath; Drive = $frame.Drive.Name }
        })
        [pscustomobject]@{
            Path = $location.Path
            ProviderPath = $location.ProviderPath
            Drive = $location.Drive.Name
            StackName = $stack.Name
            StackFrames = ConvertTo-Json -InputObject $frames -Compress -Depth 4
        }
    }
    $env:PCAI_PREFER_NATIVE_SEARCH = '0'
}

AfterAll {
    if ($PSVersionTable.PSVersion -lt [version]'7.3') { return }
    $env:PCAI_PREFER_NATIVE_SEARCH = $script:SavedNativeSearchPreference
    $script:PSNativeCommandUseErrorActionPreference = $script:SavedNativeErrorPreference
}

Describe 'fd search failure and cache contracts' -Tag 'Unit', 'Acceleration' -Skip:($PSVersionTable.PSVersion -lt [version]'7.3') {
    BeforeEach {
        $script:FdFixture = Join-Path $TestDrive ([guid]::NewGuid().ToString('N'))
        $nested = Join-Path $script:FdFixture 'sub'
        [void][IO.Directory]::CreateDirectory($nested)
        $script:FdFirst = Join-Path $script:FdFixture 'match.txt'
        $script:FdNested = Join-Path $nested 'match-nested.txt'
        foreach ($path in @($script:FdFirst, $script:FdNested, (Join-Path $script:FdFixture 'other.log'))) {
            [IO.File]::WriteAllText($path, 'tiny owned fixture')
        }
        $script:SelectedFd = $script:QualifiedFd
        $script:FdCacheCalls = 0
        $script:FdCachedCount = $null
        $script:PSNativeCommandUseErrorActionPreference = $false
        $ErrorActionPreference = 'Stop'
    }

    # Protects: invalid regex is a failure. Detects: swallowed native exit.
    # Needs: physical fd and owned files. Breadcrumb: Find-WithFd exit admission.
    It 'reports an actual invalid fd regex with native exit metadata and target' {
        $observed = Get-FdContractObservation {
            Find-WithFd -Path $script:FdFixture -Pattern '[' -NoIgnore -FdPath $script:QualifiedFd
        }
        $observed.ErrorRecord | Should -Not -BeNullOrEmpty
        (Get-FdContractExitCode $observed.ErrorRecord) | Should -BeGreaterThan 0
        $observed.ErrorRecord.FullyQualifiedErrorId | Should -BeExactly 'PcaiFdSearchFailed,Find-WithFd'
        $observed.ErrorRecord.TargetObject | Should -BeExactly $script:QualifiedFd
        $observed.Rows | Should -HaveCount 0
    }
    # Protects: failure cannot enter cache. Detects: false successful empty result.
    # Needs: real fd and cache spy. Breadcrumb: Find-FilesFast cache boundary.
    It 'does not cache or emit paths when actual fd fails through the dispatcher' {
        $observed = Get-FdContractObservation { Find-FilesFast -Path $script:FdFixture -Pattern '[' -NoIgnore }
        $observed.ErrorRecord | Should -Not -BeNullOrEmpty
        (Get-FdContractExitCode $observed.ErrorRecord) | Should -BeGreaterThan 0
        $script:FdCacheCalls | Should -Be 0
        $observed.Rows | Should -HaveCount 0
    }
    # Protects: successful no-match semantics. Detects: treating exit0 as failure.
    # Needs: real fd and cache spy. Breadcrumb: unchanged dispatcher success path.
    It 'continues caching a legitimate successful empty search' {
        $observed = Get-FdContractObservation {
            Find-FilesFast -Path $script:FdFixture -Pattern '^no-such-owned-file$' -NoIgnore
        }
        $observed.ErrorRecord | Should -BeNullOrEmpty
        $observed.Rows | Should -HaveCount 0
        $script:FdCacheCalls | Should -Be 1
        $script:FdCachedCount | Should -Be 0
    }
    # Protects: filters and FileInfo results. Detects: regex/extension/exclude drift.
    # Needs: three tiny files. Breadcrumb: Find-WithFd arguments and path mapper.
    It 'preserves actual filtered file objects and successful cache behavior' {
        $observed = Get-FdContractObservation {
            Find-FilesFast -Path $script:FdFixture -Pattern '^match' -Extension txt -Type file -Exclude 'match-nested.txt' -NoIgnore
        }
        $observed.ErrorRecord | Should -BeNullOrEmpty
        $observed.Rows | Should -HaveCount 1
        $observed.Rows[0] | Should -BeOfType ([IO.FileInfo])
        $observed.Rows[0].FullName | Should -BeExactly $script:FdFirst
        $script:FdCacheCalls | Should -Be 1
        $script:FdCachedCount | Should -Be 1
    }
    # Protects: failed partial output stays private. Detects: parsing before exit admission.
    # Needs: explicit script invocation seam. Breadcrumb: captured ExitCode23/cache0.
    It 'rejects plausible output and preserves exit23 at a synthetic invocation boundary' {
        $script:SelectedFd = Join-Path $script:FdFixture 'partial-fd.ps1'
        [IO.File]::WriteAllText($script:SelectedFd,
            ('Write-Output ' + "'" + $script:FdFirst.Replace("'", "''") + "'" + ';$global:LASTEXITCODE=23'))
        $observed = Get-FdContractObservation { Find-FilesFast -Path $script:FdFixture -Pattern '^match' -NoIgnore }
        $observed.ErrorRecord | Should -Not -BeNullOrEmpty
        (Get-FdContractExitCode $observed.ErrorRecord) | Should -Be 23
        $script:FdCacheCalls | Should -Be 0
        $observed.Rows | Should -HaveCount 0
    }
    # Protects: original invocation cause. Detects: warning-only empty success/replacement.
    # Needs: inert throwing script. Breadcrumb: actual Find-WithFd bare rethrow.
    It 'preserves the original synthetic invocation exception without caching it' {
        $script:SelectedFd = Join-Path $script:FdFixture 'throwing-fd.ps1'
        [IO.File]::WriteAllText($script:SelectedFd, "throw [IO.InvalidDataException]::new('owned invocation failure sentinel')")
        $observed = Get-FdContractObservation { Find-FilesFast -Path $script:FdFixture -Pattern '^match' -NoIgnore }
        $observed.ErrorRecord.Exception | Should -BeOfType ([IO.InvalidDataException])
        $observed.ErrorRecord.Exception.Message | Should -BeExactly 'owned invocation failure sentinel'
        $script:FdCacheCalls | Should -Be 0
        $observed.Rows | Should -HaveCount 0
    }
    # Protects: direct successful output parity. Detects: path/type transformation drift.
    # Needs: physical fd and maintained mapper. Breadcrumb: unchanged results collection.
    It 'preserves the direct successful fd file object and path set' {
        $observed = Get-FdContractObservation {
            Find-WithFd -Path $script:FdFixture -Pattern '^match' -Type file -NoIgnore -FdPath $script:QualifiedFd
        }
        $observed.ErrorRecord | Should -BeNullOrEmpty
        $observed.Rows | Should -HaveCount 2
        foreach ($row in $observed.Rows) { $row | Should -BeOfType ([IO.FileInfo]) }
        @($observed.Rows.FullName | Sort-Object) | Should -Be @(@($script:FdFirst, $script:FdNested) | Sort-Object)
    }
    # Real fd must receive a physical root, including custom PowerShell drives.
    # Native enumeration is not mocked; expected identities come from owned files.
    It 'forwards a physical root to actual fd for <RootMode>' -ForEach @(
        @{RootMode='custom-drive'}, @{RootMode='custom-drive-cwd'},
        @{RootMode='ordinary-cwd'}, @{RootMode='literal-bracket'}
    ) {
        $root = $script:FdFixture
        $expected = @($script:FdFirst, $script:FdNested)
        if ($RootMode -eq 'literal-bracket') {
            $root = Join-Path $script:FdFixture 'inputs[ab]'
            [void][IO.Directory]::CreateDirectory($root)
            $expected = @(Join-Path $root 'match-real.txt')
            [IO.File]::WriteAllText($expected[0], 'real')
            $decoy = Join-Path $script:FdFixture 'inputsa'
            [void][IO.Directory]::CreateDirectory($decoy)
            [IO.File]::WriteAllText((Join-Path $decoy 'match-wrong.txt'), 'wrong')
        }
        $originalLocation = Microsoft.PowerShell.Management\Get-Location -ErrorAction Stop
        if ($originalLocation.Provider.Name -cne 'FileSystem' -or
            -not [IO.Directory]::Exists($originalLocation.ProviderPath)) {
            throw 'The fd fixture requires an existing physical filesystem location before drive creation.'
        }
        if ($RootMode -like 'custom-drive*' -and
            @(Microsoft.PowerShell.Management\Get-PSDrive -ErrorAction Stop | Where-Object Name -EQ 'PcaiFdReview').Count -ne 0) {
            throw 'A pre-existing PcaiFdReview drive must be preserved.'
        }
        $originalProviderPath = $originalLocation.ProviderPath
        $driveCreated = $false
        $primaryFailure = $null
        $cleanupFailures = [Collections.Generic.List[object]]::new()
        try {
            if ($RootMode -like 'custom-drive*') {
                New-PSDrive -Name PcaiFdReview -PSProvider FileSystem -Root $root -Scope Script -ErrorAction Stop | Out-Null
                $driveCreated = $true
                $root = 'PcaiFdReview:\'
            }
            if ($RootMode -like '*cwd') {
                Microsoft.PowerShell.Management\Set-Location -LiteralPath $root -ErrorAction Stop
            }
            $beforeFdLocation = Get-FdContractLocationState
            if ($RootMode -like '*cwd') {
                $observed = Get-FdContractObservation { Find-FilesFast -Pattern '^match' -Type file -NoIgnore }
            } else {
                $observed = Get-FdContractObservation { Find-FilesFast -Path $root -Pattern '^match' -Type file -NoIgnore }
            }
            $afterFdLocation = Get-FdContractLocationState
            # Observe the producer before restoration so cleanup cannot conceal a side effect.
            foreach ($field in @('Path', 'ProviderPath', 'Drive', 'StackName', 'StackFrames')) {
                $afterFdLocation.$field | Should -BeExactly $beforeFdLocation.$field
            }
            $observed.ErrorRecord | Should -BeNullOrEmpty
            @($observed.Rows.FullName | Sort-Object) | Should -Be @($expected | Sort-Object)
            $script:FdCacheCalls | Should -Be 1
        } catch {
            $primaryFailure = $_
        } finally {
            # Restore the captured physical location without reading or mutating a location stack.
            try { Microsoft.PowerShell.Management\Set-Location -LiteralPath $originalProviderPath -ErrorAction Stop }
            catch { $cleanupFailures.Add($_) }
            try {
                if ($driveCreated) { Microsoft.PowerShell.Management\Remove-PSDrive -Name PcaiFdReview -Scope Script -ErrorAction Stop }
            } catch { $cleanupFailures.Add($_) }
        }
        $cleanupDiagnostics = @($cleanupFailures | ForEach-Object {
            [pscustomobject]@{ Message = $_.Exception.Message; ErrorId = $_.FullyQualifiedErrorId; Stack = $_.ScriptStackTrace }
        })
        if ($null -ne $primaryFailure) {
            if ($cleanupFailures.Count -gt 0) {
                $primaryFailure.Exception.Data['PcaiFdCleanupFailures'] = $cleanupDiagnostics
            }
            throw $primaryFailure
        }
        if ($cleanupFailures.Count -gt 0) {
            $cleanupFailures[0].Exception.Data['PcaiFdCleanupFailures'] = $cleanupDiagnostics
            throw $cleanupFailures[0]
        }
        (Microsoft.PowerShell.Management\Get-Location -ErrorAction Stop).ProviderPath | Should -BeExactly $originalProviderPath
        if ($driveCreated) { @(Microsoft.PowerShell.Management\Get-PSDrive -ErrorAction Stop | Where-Object Name -EQ 'PcaiFdReview').Count | Should -Be 0 }
    }
    # Protects: caller native-error preference. Detects: manual cause substitution/cache.
    # Needs: physical fd and preference true. Breadcrumb: exact NativeCommandExitException.
    It 'preserves the actual native error preference cause and avoids caching it' {
        $script:PSNativeCommandUseErrorActionPreference = $true
        $observed = Get-FdContractObservation { Find-FilesFast -Path $script:FdFixture -Pattern '[' -NoIgnore }
        $nativeFailure = $null
        $inner = $observed.ErrorRecord.Exception
        while ($inner) {
            if ($inner.GetType().FullName -ceq 'System.Management.Automation.NativeCommandExitException') {
                $nativeFailure = $inner
                break
            }
            $inner = $inner.InnerException
        }
        $nativeFailure | Should -Not -BeNullOrEmpty
        $nativeFailure.ExitCode | Should -BeGreaterThan 0
        $observed.ErrorRecord.FullyQualifiedErrorId | Should -BeExactly 'ProgramExitedWithNonZeroCode'
        $script:FdCacheCalls | Should -Be 0
        $observed.Rows | Should -HaveCount 0
    }
}
