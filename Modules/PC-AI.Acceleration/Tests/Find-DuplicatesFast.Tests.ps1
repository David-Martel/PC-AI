#Requires -Modules Pester

BeforeAll {
    $modulePath = Split-Path -Parent $PSScriptRoot
    Import-Module (Join-Path $modulePath 'PC-AI.Acceleration.psd1') -Force

    function New-ExplicitDuplicateFixture {
        param([string]$Parent)
        $root = Join-Path $Parent 'explicit'
        New-Item -ItemType Directory -Path (Join-Path $root 'nested') -Force | Out-Null
        & git init --quiet $Parent
        if ($LASTEXITCODE -ne 0) { throw 'Cannot initialize the isolated TestDrive ignore fixture' }
        [IO.File]::WriteAllText((Join-Path $Parent '.gitignore'), "*.bin`n")
        [byte[]]$bytes = 0..127
        [IO.File]::WriteAllBytes((Join-Path $root 'a.bin'), $bytes)
        [IO.File]::WriteAllBytes((Join-Path $root 'nested/b.bin'), $bytes)
        [IO.File]::WriteAllBytes((Join-Path $root 'different.bin'), [byte[]](128..255))
        & git -C $Parent check-ignore --quiet explicit/a.bin
        if ($LASTEXITCODE -ne 0) { throw 'The explicit fixture root must be ignored by its parent' }
        return $root
    }
}

Describe 'Find-DuplicatesFast' {
    # Protects: native duplicate dispatch; Detects: the selected native result losing provider/membership.
    # Needs: scoped native-call mocks; Breadcrumb: ../Public/Find-DuplicatesFast.ps1.
    It 'Uses the native duplicate path when available for SHA256 requests' {
        Mock -CommandName Get-Command -ModuleName 'PC-AI.Acceleration' -MockWith {
            [pscustomobject]@{ Name = 'Invoke-PcaiNativeDuplicates' }
        } -ParameterFilter { $Name -eq 'Invoke-PcaiNativeDuplicates' }
        Mock -CommandName Test-PcaiNativeAvailable -ModuleName 'PC-AI.Acceleration' -MockWith { $true }
        Mock -CommandName Invoke-PcaiNativeDuplicates -ModuleName 'PC-AI.Acceleration' -MockWith {
            [pscustomobject]@{
                IsSuccess = $true
                FilesScanned = 3
                DuplicateFiles = 1
                WastedBytes = 128
                ElapsedMs = 5
                Groups = @(
                    [pscustomobject]@{
                        Hash = 'abc'
                        Size = 128
                        WastedBytes = 128
                        Paths = @('C:\temp\a.txt', 'C:\temp\b.txt')
                    }
                )
            }
        }

        $result = @(Find-DuplicatesFast -Path $env:TEMP -Recurse)
        $result.Count | Should -Be 1
        $result[0].Provider | Should -Be 'PcaiNative'
        $result[0].Duplicates.Count | Should -Be 1
        Should -Invoke Invoke-PcaiNativeDuplicates -ModuleName 'PC-AI.Acceleration' -Times 1
    }

    # Protects: explicit fallback selection; Detects: DisableNative failing to invoke the managed scanner.
    # Needs: scoped scanner dispatch mock; Breadcrumb: ../Public/Find-DuplicatesFast.ps1.
    It 'Falls back when the native path is disabled' {
        Mock -CommandName Invoke-PowerShellDuplicateScan -ModuleName 'PC-AI.Acceleration' -MockWith {
            @([pscustomobject]@{ Hash = 'fallback'; WastedBytes = 0 })
        }

        $result = @(Find-DuplicatesFast -Path $env:TEMP -DisableNative)
        $result.Count | Should -Be 1
        $result[0].Hash | Should -Be 'fallback'
        Should -Invoke Invoke-PowerShellDuplicateScan -ModuleName 'PC-AI.Acceleration' -Times 1
    }
}

Describe 'Explicit-root duplicate scanning through real fallback implementations' {
    BeforeAll {
        $script:duplicateFd = & (Get-Module PC-AI.Acceleration) { Get-RustToolPath -ToolName 'fd' }
    }

    # Protects: explicit-root visibility and content identity; Detects: ancestor ignores hiding a selected root or equal sizes becoming false duplicates.
    # Needs: actual fd/Git, isolated TestDrive files and real .NET hashing; Breadcrumb: ../Public/Find-DuplicatesFast.ps1.
    It 'finds the exact duplicate pair with real fd when its explicit root is ignored by a parent' {
        if (-not $script:duplicateFd) { Set-ItResult -Skipped -Because 'fd is not installed'; return }
        $root = New-ExplicitDuplicateFixture -Parent (Join-Path $TestDrive 'fd-identity-parent')
        @(& $script:duplicateFd . -t f -a $root).Count | Should -Be 0

        $groups = @(Find-DuplicatesFast -Path $root -Recurse -MinimumSize 1 -ThrottleLimit 2 -DisableNative)

        $groups.Count | Should -Be 1
        $groups[0].Count | Should -Be 2
        $groups[0].FileSize | Should -Be 128
        $groups[0].WastedBytes | Should -Be 128
        $groups[0].Hash | Should -Be (Get-FileHash (Join-Path $root 'a.bin') -Algorithm SHA256).Hash
        $members = @($groups[0].Files | ForEach-Object { [IO.Path]::GetRelativePath($root, $_).Replace('\', '/') } | Sort-Object)
        ($members -join '|') | Should -BeExactly 'a.bin|nested/b.bin'
    }

    # Protects: selected-root ignore/hidden boundaries; Detects: a broad no-ignore/unrestricted workaround scanning excluded descendants.
    # Needs: actual fd/Git and synthetic ignored/hidden subtrees; Breadcrumb: ../Public/Find-DuplicatesFast.ps1.
    It 'keeps ignore files inside the selected root and hidden-file filtering effective' {
        if (-not $script:duplicateFd) { Set-ItResult -Skipped -Because 'fd is not installed'; return }
        $root = New-ExplicitDuplicateFixture -Parent (Join-Path $TestDrive 'fd-local-policy-parent')
        $excluded = New-Item -ItemType Directory -Path (Join-Path $root 'local-excluded')
        $hidden = New-Item -ItemType Directory -Path (Join-Path $root '.hidden')
        if ($IsWindows) { $hidden.Attributes = $hidden.Attributes -bor [IO.FileAttributes]::Hidden }
        [IO.File]::WriteAllText((Join-Path $root '.fdignore'), "local-excluded/`n")
        foreach ($directory in @($excluded, $hidden)) {
            [IO.File]::WriteAllText((Join-Path $directory.FullName 'one.bin'), 'excluded synthetic pair')
            [IO.File]::WriteAllText((Join-Path $directory.FullName 'two.bin'), 'excluded synthetic pair')
        }

        $groups = @(Find-DuplicatesFast -Path $root -Recurse -MinimumSize 1 -ThrottleLimit 2 -DisableNative)

        $groups.Count | Should -Be 1
        $members = @($groups[0].Files | ForEach-Object { [IO.Path]::GetRelativePath($root, $_).Replace('\', '/') } | Sort-Object)
        ($members -join '|') | Should -BeExactly 'a.bin|nested/b.bin'
    }

    # Protects: public recursion/pattern/size filters across the fd visibility fix; Detects: an excluded pair surviving any named filter.
    # Needs: real fd/Git and five explicit negative parameter cases; Breadcrumb: ../Public/Find-DuplicatesFast.ps1.
    It 'retains the <label> exclusion with real fd' -TestCases @(
        @{ label = 'nonrecursive'; selection = @{ Recurse = $false } }
        @{ label = 'minimum-size'; selection = @{ MinimumSize = 129 } }
        @{ label = 'maximum-size'; selection = @{ MaximumSize = 127 } }
        @{ label = 'include-pattern'; selection = @{ Include = '*.absent' } }
        @{ label = 'exclude-pattern'; selection = @{ Exclude = 'nested' } }
    ) {
        param($label, $selection)
        if (-not $script:duplicateFd) { Set-ItResult -Skipped -Because 'fd is not installed'; return }
        $root = New-ExplicitDuplicateFixture -Parent (Join-Path $TestDrive "fd-filter-$label-parent")
        $parameters = @{ Path = $root; Recurse = $true; MinimumSize = 1; ThrottleLimit = 2; DisableNative = $true }
        foreach ($key in $selection.Keys) { $parameters[$key] = $selection[$key] }

        @(Find-DuplicatesFast @parameters).Count | Should -Be 0
    }

    # Protects: genuine managed fallback without fd; Detects: a missing tool changing exact hash membership or permitting equal-size false matches.
    # Needs: Git fixture and simulated fd lookup absence; real public scanner/Get-ChildItem/.NET helper; Breadcrumb: ../Public/Find-DuplicatesFast.ps1.
    It 'returns the same exact pair through real managed enumeration and hashing when fd is unavailable' {
        $root = New-ExplicitDuplicateFixture -Parent (Join-Path $TestDrive 'managed-parent')
        Mock Get-RustToolPath -ModuleName PC-AI.Acceleration { $null }

        $groups = @(Find-DuplicatesFast -Path $root -Recurse -MinimumSize 1 -ThrottleLimit 2 -DisableNative)

        $groups.Count | Should -Be 1
        $groups[0].Count | Should -Be 2
        $groups[0].FileSize | Should -Be 128
        $groups[0].Hash | Should -Be (Get-FileHash (Join-Path $root 'a.bin') -Algorithm SHA256).Hash
        $members = @($groups[0].Files | ForEach-Object { [IO.Path]::GetRelativePath($root, $_).Replace('\', '/') } | Sort-Object)
        ($members -join '|') | Should -BeExactly 'a.bin|nested/b.bin'
        [bool]('PcAi.Acceleration.ParallelHashHelper' -as [type]) | Should -BeTrue
    }
}
