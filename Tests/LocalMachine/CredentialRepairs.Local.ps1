BeforeDiscovery {
    Import-Module (Join-Path $env:USERPROFILE '.machine\SecretsTier.psm1') -Force -DisableNameChecking
}

BeforeAll {
    $machineRoot = Join-Path $env:USERPROFILE '.machine'
    Import-Module (Join-Path $machineRoot 'SecretsTier.psm1') -Force -DisableNameChecking
    . (Join-Path $machineRoot 'SecretBackendUtilities.ps1')
    . (Join-Path $PSScriptRoot '..\..\Tools\SystemScripts\Machine\Update-BwArchive.ps1')
}

Describe 'Installed Bitwarden command selection' {
    It 'prefers a native executable over stale CMD wrappers and Node installations' {
        Mock Get-Command { [pscustomobject]@{ Source = 'C:\fixture\bw.exe' } } -ParameterFilter { $Name -eq 'bw.exe' }
        Mock Get-Command { throw 'Native selection must not inspect Node' } -ParameterFilter { $Name -eq 'node' }
        Mock Test-Path { $true }
        $spec = Get-BitwardenCliProcessSpec
        $spec.Kind | Should -Be 'native-executable'
        $spec.FilePath | Should -Be 'C:\fixture\bw.exe'
        @($spec.ArgumentPrefix).Count | Should -Be 0
        Should -Invoke Get-Command -Times 1 -Exactly -ParameterFilter { $Name -eq 'bw.exe' -and $CommandType -eq 'Application' }
        Should -Invoke Get-Command -Times 0 -Exactly -ParameterFilter { $Name -eq 'node' }
    }
    It 'retains direct Node execution when the native CLI is absent' {
        Mock Get-Command { $null } -ParameterFilter { $Name -eq 'bw.exe' }
        Mock Get-Command { [pscustomobject]@{ Source = 'C:\fixture\node.exe' } } -ParameterFilter { $Name -eq 'node' }
        Mock Test-Path { $true }
        $spec = Get-BitwardenCliProcessSpec
        $spec.Kind | Should -Be 'node-script'
        $spec.FilePath | Should -Be 'C:\fixture\node.exe'
        @($spec.ArgumentPrefix) | Should -Be @((Join-Path $env:APPDATA 'npm\node_modules\@bitwarden\cli\build\bw.js'))
    }
}

Describe 'Archive ACL identity normalization' {
    It 'deduplicates SYSTEM before ACL construction and verification' {
        $system = [Security.Principal.SecurityIdentifier]::new('S-1-5-18')
        $actual = @(Get-BwArchiveAllowedSid -CurrentUser $system)
        $actual.Count | Should -Be 2
        $actual.Value | Should -Be @('S-1-5-18', 'S-1-5-32-544')
    }
    It 'preserves a distinct user alongside SYSTEM and Administrators' {
        $userSid = 'S-1-5-21-1-2-3-1001'
        $actual = @(Get-BwArchiveAllowedSid -CurrentUser ([Security.Principal.SecurityIdentifier]::new($userSid)))
        $actual.Count | Should -Be 3
        $actual.Value | Should -Be @($userSid, 'S-1-5-18', 'S-1-5-32-544')
    }
}

Describe 'Actual SecretsTier initializer' {
    InModuleScope SecretsTier {
        BeforeEach {
            $script:oldToken = $env:BWS_ACCESS_TOKEN
            $env:BWS_ACCESS_TOKEN = $null
            $script:testKey = 'CODEX_CREDENTIAL_TEST_' + [guid]::NewGuid().ToString('N')
            $script:testSecrets = @{ $script:testKey = 'nonsecret-fixture' }
            Mock Resolve-BwsTokenFromBackends { $env:BWS_ACCESS_TOKEN = 'nonsecret-token-fixture' } # pragma: allowlist secret (nonsecret fixture)
            Mock Test-InternetConnection { $true }
            Mock Get-BwsSecretsList { $script:testSecrets }
            Mock Save-SecretsToCache {}
            Mock Get-CacheStatus { @{ IsValid = $true; Exists = $true; IsExpired = $false; LastUpdated = [DateTime]::UtcNow } }
            Mock Get-SecretsFromCache { $script:testSecrets }
        }
        AfterEach {
            $env:BWS_ACCESS_TOKEN = $script:oldToken
            [Environment]::SetEnvironmentVariable($script:testKey, [NullString]::Value, 'Process')
            [Environment]::SetEnvironmentVariable($script:testKey, [NullString]::Value, 'User')
        }
        It 'resolves an absent token before cloud fetch and persists selected fixture values' {
            $r = Initialize-MachineSecretsTiered -PersistentKeys @($script:testKey)
            $r.Source | Should -Be 'BWS Cloud'
            $r.SecretCount | Should -Be 1
            [Environment]::GetEnvironmentVariable($script:testKey, 'Process') | Should -Be 'nonsecret-fixture'
            [Environment]::GetEnvironmentVariable($script:testKey, 'User') | Should -Be 'nonsecret-fixture'
            Should -Invoke Resolve-BwsTokenFromBackends -Times 1 -Exactly
            Should -Invoke Save-SecretsToCache -Times 1 -Exactly -ParameterFilter { $Secrets[$script:testKey] -eq 'nonsecret-fixture' }
        }
        It 'preserves an existing process token without resolving another backend' {
            $env:BWS_ACCESS_TOKEN = 'explicit-nonsecret-fixture' # pragma: allowlist secret (nonsecret fixture)
            (Initialize-MachineSecretsTiered).Source | Should -Be 'BWS Cloud'
            $env:BWS_ACCESS_TOKEN | Should -Be 'explicit-nonsecret-fixture'
            Should -Invoke Resolve-BwsTokenFromBackends -Times 0 -Exactly
        }
        It 'uses cache without attempting cloud access when resolution returns no token' {
            Mock Resolve-BwsTokenFromBackends {}
            (Initialize-MachineSecretsTiered -PersistentKeys @($script:testKey)).Source | Should -Be 'Local Cache'
            [Environment]::GetEnvironmentVariable($script:testKey, 'User') | Should -Be 'nonsecret-fixture'
            Should -Invoke Get-BwsSecretsList -Times 0 -Exactly
            Should -Invoke Test-InternetConnection -Times 0 -Exactly
        }
        It 'reports resolver failure and persists cache values' {
            Mock Resolve-BwsTokenFromBackends { throw 'fixture resolver failure' }
            $r = Initialize-MachineSecretsTiered -PersistentKeys @($script:testKey)
            $r.Source | Should -Be 'Local Cache'
            $r.Warnings | Should -Contain 'BWS token resolution failed; trying the local cache.'
            [Environment]::GetEnvironmentVariable($script:testKey, 'User') | Should -Be 'nonsecret-fixture'
            Should -Invoke Get-BwsSecretsList -Times 0 -Exactly
        }
        It 'uses cache when the optional token resolver is unavailable' {
            Mock Get-Command { $null } -ParameterFilter { $Name -eq 'Resolve-BwsTokenFromBackends' }
            (Initialize-MachineSecretsTiered).Source | Should -Be 'Local Cache'
            Should -Invoke Resolve-BwsTokenFromBackends -Times 0 -Exactly
            Should -Invoke Get-BwsSecretsList -Times 0 -Exactly
        }
        It 'persists cache values offline and skips cloud fetch' {
            Mock Test-InternetConnection { $false }
            (Initialize-MachineSecretsTiered -PersistentKeys @($script:testKey)).Source | Should -Be 'Local Cache'
            [Environment]::GetEnvironmentVariable($script:testKey, 'User') | Should -Be 'nonsecret-fixture'
            Should -Invoke Get-BwsSecretsList -Times 0 -Exactly
        }
        It 'falls back from empty cloud response and honors default process-only scope' {
            Mock Get-BwsSecretsList { $null }
            (Initialize-MachineSecretsTiered).Source | Should -Be 'Local Cache'
            [Environment]::GetEnvironmentVariable($script:testKey, 'Process') | Should -Be 'nonsecret-fixture'
            [Environment]::GetEnvironmentVariable($script:testKey, 'User') | Should -BeNullOrEmpty
        }
        It 'persists expired cache while reporting its age' {
            Mock Test-InternetConnection { $false }
            Mock Get-CacheStatus { @{ IsValid = $false; Exists = $true; IsExpired = $true; Age = [TimeSpan]::FromDays(9) } }
            $r = Initialize-MachineSecretsTiered -PersistentKeys @($script:testKey, 'missing-fixture-key')
            $r.Source | Should -Be 'Local Cache (EXPIRED)'
            $r.Warnings | Should -Contain 'WARNING: Cache is 9 days old!'
            [Environment]::GetEnvironmentVariable($script:testKey, 'User') | Should -Be 'nonsecret-fixture'
            Should -Invoke Get-SecretsFromCache -Times 1 -Exactly -ParameterFilter { $IgnoreAge }
        }
        It 'fails explicitly when neither cloud nor cache supplies secrets' {
            Mock Resolve-BwsTokenFromBackends {}
            Mock Get-CacheStatus { @{ IsValid = $false; Exists = $false; IsExpired = $false } }
            $r = Initialize-MachineSecretsTiered
            $r.Success | Should -BeFalse
            $r.Source | Should -Be 'NONE'
            $r.SecretCount | Should -Be 0
            [Environment]::GetEnvironmentVariable($script:testKey, 'Process') | Should -BeNullOrEmpty
        }
    }
}

Describe 'Actual protected archive flow using nonsecret native I/O fixtures' {
    BeforeEach {
        $script:archiveRoot = Join-Path $TestDrive ([guid]::NewGuid().ToString('N'))
        Mock Initialize-BitwardenSessionFromBackends { throw 'Unexpected session bootstrap in fixture' }
        Mock Invoke-BitwardenCli {
            param($Arguments, $TimeoutSeconds)
            $TimeoutSeconds | Should -Be 120
            $stdout = ''
            if ($Arguments[0] -eq 'status') { $stdout = '{"status":"unlocked"}' }
            if ($Arguments[0] -eq 'export') {
                $output = $Arguments[4]
                $acl = Get-Acl -LiteralPath (Split-Path $output)
                if (-not $acl.AreAccessRulesProtected) { throw 'Export attempted before staging ACL protection.' }
                (Get-Acl -LiteralPath $script:archiveRoot).AreAccessRulesProtected | Should -BeTrue
                [IO.File]::WriteAllText($output, '{"fixture":"nonsecret"}')
            }
            [pscustomobject]@{ Success = $true; ExitCode = 0; TimedOut = $false; StdOut = $stdout }
        }
    }
    It 'protects data before export and preserves ACLs when publishing both archives' {
        Invoke-BwArchiveRefresh -Dir $script:archiveRoot -Quiet | Should -Be 0
        $files = @(Get-ChildItem -LiteralPath $script:archiveRoot -File)
        $files.Count | Should -Be 2
        foreach ($file in $files) {
            [IO.File]::ReadAllText($file.FullName) | Should -Be '{"fixture":"nonsecret"}'
            $acl = Get-Acl -LiteralPath $file.FullName
            $acl.AreAccessRulesProtected | Should -BeTrue
            @($acl.Access).Count | Should -Be 3
        }
        @(Get-ChildItem -LiteralPath $script:archiveRoot -Directory -Force).Count | Should -Be 0
        Should -Invoke Invoke-BitwardenCli -Times 1 -Exactly -ParameterFilter { $Arguments[0] -eq 'sync' -and $TimeoutSeconds -eq 120 }
        Should -Invoke Invoke-BitwardenCli -Times 1 -Exactly -ParameterFilter { $Arguments[0] -eq 'export' -and $Arguments[2] -eq 'json' -and $TimeoutSeconds -eq 120 }
    }
    It 'does no authentication or filesystem mutation in DryRun and WhatIf' {
        Invoke-BwArchiveRefresh -Dir $script:archiveRoot -DryRun | Should -Be 0
        Invoke-BwArchiveRefresh -Dir $script:archiveRoot -WhatIf | Should -Be 0
        Test-Path -LiteralPath $script:archiveRoot | Should -BeFalse
        Should -Invoke Invoke-BitwardenCli -Times 0 -Exactly
        Should -Invoke Initialize-BitwardenSessionFromBackends -Times 0 -Exactly
    }
    It 'aborts before sync and export when Set-Acl fails' {
        Mock Set-Acl { throw 'fixture ACL denied' }
        { Invoke-BwArchiveRefresh -Dir $script:archiveRoot -Quiet } | Should -Throw '*fixture ACL denied*'
        Should -Invoke Invoke-BitwardenCli -Times 0 -Exactly -ParameterFilter { $Arguments[0] -in @('sync', 'export') }
        @(Get-ChildItem -LiteralPath $script:archiveRoot -Force).Count | Should -Be 0
    }
    It 'detects an ACL operation that returns without applying protection' {
        Mock Set-Acl {}
        { Invoke-BwArchiveRefresh -Dir $script:archiveRoot -Quiet } | Should -Throw '*ACL verification failed*'
        Should -Invoke Invoke-BitwardenCli -Times 0 -Exactly -ParameterFilter { $Arguments[0] -eq 'export' }
    }
    It 'does not export stale data or replace latest after sync fails' {
        $null = New-Item -ItemType Directory -Path $script:archiveRoot
        $latest = Join-Path $script:archiveRoot 'bitwarden_archive_latest.json'
        [IO.File]::WriteAllText($latest, 'previous-fixture')
        Mock Invoke-BitwardenCli { [pscustomobject]@{ Success = $false; ExitCode = 7; TimedOut = $false; StdOut = 'untrusted-output' } } -ParameterFilter { $Arguments[0] -eq 'sync' }
        { Invoke-BwArchiveRefresh -Dir $script:archiveRoot -Quiet } | Should -Throw '*sync failed (exit=7, timeout=False)*'
        [IO.File]::ReadAllText($latest) | Should -Be 'previous-fixture'
        Should -Invoke Invoke-BitwardenCli -Times 0 -Exactly -ParameterFilter { $Arguments[0] -eq 'export' }
    }
    It 'surfaces native timeouts without exposing native output' {
        Mock Invoke-BitwardenCli { [pscustomobject]@{ Success = $false; ExitCode = $null; TimedOut = $true; StdOut = 'untrusted-output' } } -ParameterFilter { $Arguments[0] -eq 'export' }
        { Invoke-BwArchiveRefresh -Dir $script:archiveRoot -Quiet } | Should -Throw '*export failed (exit=, timeout=True)*'
        @(Get-ChildItem -LiteralPath $script:archiveRoot -Force).Count | Should -Be 0
    }
    It 'rejects empty exports without publishing a latest archive' {
        Mock Invoke-BitwardenCli { [pscustomobject]@{ Success = $true; ExitCode = 0; TimedOut = $false; StdOut = '' } } -ParameterFilter { $Arguments[0] -eq 'export' }
        { Invoke-BwArchiveRefresh -Dir $script:archiveRoot -Quiet } | Should -Throw '*nonempty archive*'
        @(Get-ChildItem -LiteralPath $script:archiveRoot -Force).Count | Should -Be 0
    }
    It 'rejects malformed status without including native data in the error' {
        Mock Invoke-BitwardenCli { [pscustomobject]@{ Success = $true; ExitCode = 0; TimedOut = $false; StdOut = 'untrusted status fixture' } } -ParameterFilter { $Arguments[0] -eq 'status' }
        { Invoke-BwArchiveRefresh -Dir $script:archiveRoot -Quiet } | Should -Throw -ExpectedMessage 'Bitwarden returned an invalid status response.'
        @(Get-ChildItem -LiteralPath $script:archiveRoot -Force).Count | Should -Be 0
        Should -Invoke Initialize-BitwardenSessionFromBackends -Times 0 -Exactly
    }
    It 'stops on failed locked-session bootstrap without syncing or exporting' {
        Mock Invoke-BitwardenCli { [pscustomobject]@{ Success = $true; ExitCode = 0; TimedOut = $false; StdOut = '{"status":"locked"}' } } -ParameterFilter { $Arguments[0] -eq 'status' }
        Mock Initialize-BitwardenSessionFromBackends { [pscustomobject]@{ Success = $false } }
        { Invoke-BwArchiveRefresh -Dir $script:archiveRoot -Quiet } | Should -Throw '*could not establish an unlocked session*'
        Should -Invoke Initialize-BitwardenSessionFromBackends -Times 1 -Exactly -ParameterFilter { $Quiet }
        Should -Invoke Invoke-BitwardenCli -Times 0 -Exactly -ParameterFilter { $Arguments[0] -in @('sync', 'export') }
    }
    It 'only prunes expired timestamped archives and keeps manual exports and latest' {
        $null = New-Item -ItemType Directory -Path $script:archiveRoot
        foreach ($name in @('bitwarden_archive_20200101T000000Z.json', 'bitwarden_export_20200101.json', 'bitwarden_archive_latest.json', 'bitwarden_archive_other.json')) {
            $p = Join-Path $script:archiveRoot $name
            [IO.File]::WriteAllText($p, 'old-fixture')
            (Get-Item -LiteralPath $p).LastWriteTime = (Get-Date).AddDays(-40)
        }
        Invoke-BwArchiveRefresh -Dir $script:archiveRoot -Quiet | Should -Be 0
        Test-Path (Join-Path $script:archiveRoot 'bitwarden_archive_20200101T000000Z.json') | Should -BeFalse
        [IO.File]::ReadAllText((Join-Path $script:archiveRoot 'bitwarden_export_20200101.json')) | Should -Be 'old-fixture'
        [IO.File]::ReadAllText((Join-Path $script:archiveRoot 'bitwarden_archive_other.json')) | Should -Be 'old-fixture'
        [IO.File]::ReadAllText((Join-Path $script:archiveRoot 'bitwarden_archive_latest.json')) | Should -Be '{"fixture":"nonsecret"}'
        $latestAcl = Get-Acl -LiteralPath (Join-Path $script:archiveRoot 'bitwarden_archive_latest.json')
        $latestAcl.AreAccessRulesProtected | Should -BeTrue
        @($latestAcl.Access).Count | Should -Be 3
    }
    It 'rejects invalid retention and timeout bounds before native calls' {
        { Invoke-BwArchiveRefresh -Dir $script:archiveRoot -KeepDays 0 } | Should -Throw
        { Invoke-BwArchiveRefresh -Dir $script:archiveRoot -TimeoutSeconds 0 } | Should -Throw
        Should -Invoke Invoke-BitwardenCli -Times 0 -Exactly
    }
}

Describe 'Actual Bitwarden bootstrap with nonsecret password/session fixtures' {
    BeforeEach {
        $script:savedBwSession = $env:BW_SESSION
        $env:BW_SESSION = 'previous-fixture-session-123456789'
        $script:sessionFile = Join-Path $TestDrive ([guid]::NewGuid().ToString('N') + '.session')
        [IO.File]::WriteAllText($script:sessionFile, $env:BW_SESSION)
        $script:capturedPasswordFile = $null
        Mock Get-BitwardenCliProcessSpec { @{ FilePath = 'fixture' } }
        Mock Resolve-BwBootstrapState {
            @{ Password = 'fixture-password'; PasswordSource = 'fixture'; SessionFile = $script:sessionFile; ClientId = $null; ClientSecret = $null } # pragma: allowlist secret (nonsecret fixture)
        }
        Mock Get-BwStatusSafe {
            @{ status = if ($env:BW_SESSION -eq 'candidate-fixture-session-123456789') { 'unlocked' } else { 'locked' } }
        }
        Mock Invoke-BitwardenCli {
            param($Arguments, $TimeoutSeconds)
            $Arguments[0] | Should -Be 'unlock'
            $Arguments[1] | Should -Be '--passwordfile'
            $Arguments[3] | Should -Be '--raw'
            $TimeoutSeconds | Should -Be 30
            $script:capturedPasswordFile = $Arguments[2]
            $acl = Get-Acl -LiteralPath $script:capturedPasswordFile
            $acl.AreAccessRulesProtected | Should -BeTrue
            @($acl.Access).Count | Should -Be 3
            [IO.File]::ReadAllText($script:capturedPasswordFile) | Should -Be 'fixture-password'
            [pscustomobject]@{ Success = $true; ExitCode = 0; TimedOut = $false; StdOut = 'candidate-fixture-session-123456789' }
        }
    }
    AfterEach { $env:BW_SESSION = $script:savedBwSession }
    It 'creates protected password data, validates unlock and protects the persistent session' {
        $r = Initialize-BitwardenSessionFromBackends -Refresh -Quiet
        $r.Success | Should -BeTrue
        $env:BW_SESSION | Should -Be 'candidate-fixture-session-123456789'
        [IO.File]::ReadAllText($script:sessionFile) | Should -Be 'candidate-fixture-session-123456789'
        (Get-Acl -LiteralPath $script:sessionFile).AreAccessRulesProtected | Should -BeTrue
        @((Get-Acl -LiteralPath $script:sessionFile).Access).Count | Should -Be 3
        Test-Path -LiteralPath $script:capturedPasswordFile | Should -BeFalse
    }
    It 'preserves previous session data on native failure with misleading long stdout' {
        Mock Invoke-BitwardenCli {
            param($Arguments)
            $script:capturedPasswordFile = $Arguments[2]
            [pscustomobject]@{ Success = $false; ExitCode = 9; TimedOut = $false; StdOut = 'untrusted-error-text-longer-than-twenty-characters' }
        }
        $r = Initialize-BitwardenSessionFromBackends -Refresh -Quiet
        $r.Success | Should -BeFalse
        $r.Message | Should -Match 'exit=9'
        $r.Message | Should -Not -Match 'untrusted-error-text'
        $env:BW_SESSION | Should -Be 'previous-fixture-session-123456789'
        [IO.File]::ReadAllText($script:sessionFile) | Should -Be 'previous-fixture-session-123456789'
        Test-Path -LiteralPath $script:capturedPasswordFile | Should -BeFalse
    }
    It 'rejects a candidate session that Bitwarden does not accept' {
        Mock Get-BwStatusSafe { @{ status = 'locked' } }
        $r = Initialize-BitwardenSessionFromBackends -Refresh -Quiet
        $r.Success | Should -BeFalse
        $env:BW_SESSION | Should -Be 'previous-fixture-session-123456789'
        [IO.File]::ReadAllText($script:sessionFile) | Should -Be 'previous-fixture-session-123456789'
        Test-Path -LiteralPath $script:capturedPasswordFile | Should -BeFalse
    }
    It 'does not call unlock if private password storage cannot be created' {
        Mock Write-BitwardenPrivateFile { throw 'fixture private file denied' }
        $r = Initialize-BitwardenSessionFromBackends -Refresh -Quiet
        $r.Success | Should -BeFalse
        $env:BW_SESSION | Should -Be 'previous-fixture-session-123456789'
        [IO.File]::ReadAllText($script:sessionFile) | Should -Be 'previous-fixture-session-123456789'
        Should -Invoke Invoke-BitwardenCli -Times 0 -Exactly
    }
}
