BeforeAll {
    $script:GpuRoot = Join-Path $PSScriptRoot '../../Modules/PC-AI.Gpu'
    foreach ($relative in @('Public/Get-NvidiaSoftwareRegistry.ps1', 'Public/Get-NvidiaSoftwareStatus.ps1',
            'Private/Test-NvidiaDownloadUrl.ps1', 'Private/Backup-NvidiaEnvironment.ps1',
            'Private/Invoke-NvidiaSilentInstall.ps1')) {
        . (Join-Path $script:GpuRoot $relative)
    }
    $script:CoordinatorSource = Join-Path $script:GpuRoot 'Public/Install-NvidiaSoftware.ps1'
    if ($env:PCAI_NVIDIA_COORDINATOR_SOURCE) { $script:CoordinatorSource = $env:PCAI_NVIDIA_COORDINATOR_SOURCE }
    . $script:CoordinatorSource
    $script:OriginalTemp = $env:TEMP
    $script:CaseNumber = 0
    $script:Payload = [Text.Encoding]::UTF8.GetBytes('public inert installer bytes')
    $script:PayloadHash = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($script:Payload))

    function Write-CoordinatorRegistry {
        param([string]$Url = 'https://developer.download.nvidia.com/fixture.exe', [string]$Hash = $script:PayloadHash)
        $registry = [ordered]@{
            version = 'fixture'; lastUpdated = 'fixture'; trustedSources = @('nvidia.com'); categories = @{}
            components = @([ordered]@{ id = 'fixture'; name = 'Public fixture'; category = 'fixture'
                latestVersion = '2.0'; downloadUrl = $Url; sha256 = $Hash })
        }
        [IO.File]::WriteAllText($script:RegistryPath, ($registry | ConvertTo-Json -Depth 5))
    }
}

Describe 'NVIDIA installer coordinator decisions and file contracts' -Tag 'Unit', 'Gpu', 'Portable' {
    BeforeEach {
        $script:CaseNumber++
        $fixtureBase = $TestDrive
        if ($env:PCAI_NVIDIA_FIXTURE_ROOT) { $fixtureBase = $env:PCAI_NVIDIA_FIXTURE_ROOT }
        $script:CaseRoot = Join-Path $fixtureBase "case-$script:CaseNumber"
        [IO.Directory]::CreateDirectory($script:CaseRoot) | Out-Null
        $script:CaseTemp = Join-Path $script:CaseRoot 'temp'
        [IO.Directory]::CreateDirectory($script:CaseTemp) | Out-Null
        $env:TEMP = $script:CaseTemp
        $script:RegistryPath = Join-Path $script:CaseRoot 'registry.json'
        $script:StagingDir = Join-Path $script:CaseTemp 'nvidia-installers'
        $script:StagedFile = Join-Path $script:StagingDir 'fixture.exe'
        Write-CoordinatorRegistry
        Mock Get-NvidiaSoftwareStatus { [pscustomobject]@{ InstalledVersion = '1.0'; Status = 'Outdated' } }
        Mock Test-NvidiaDownloadUrl { [pscustomobject]@{ IsTrusted = $true; IsValid = $true; StatusCode = 200 } }
        Mock Invoke-WebRequest { [IO.File]::WriteAllBytes($OutFile, $script:Payload) }
        Mock Backup-NvidiaEnvironment { 'public-backup-boundary' }
        Mock Invoke-NvidiaSilentInstall {
            [pscustomobject]@{ Success = $true; ExitCode = 0; RebootRequired = $false; LogPath = 'public-log-boundary' }
        }
    }
    AfterEach { $env:TEMP = $script:OriginalTemp }

    # Protects: declined download/install decisions cannot create staging or call vendor/network boundaries.
    # Detects: the real predecessor advertises a URL as a staged download and reports false success.
    # Needs: real registry, real filesystem and the public coordinator; only external boundaries are mocked.
    # Breadcrumb: Install-NvidiaSoftware ShouldProcess must precede URL reachability and all side effects.
    It 'declines explicit WhatIf without staging or external work: <Download>' -ForEach @(
        @{ Download = $true }, @{ Download = $false }
    ) {
        $result = Install-NvidiaSoftware -ComponentId fixture -RegistryPath $script:RegistryPath -DownloadOnly:$Download -WhatIf
        $result.Success | Should -BeFalse
        $result.InstallerPath | Should -BeNullOrEmpty
        $result.LogPath | Should -BeNullOrEmpty
        $result.VersionAfter | Should -BeNullOrEmpty
        $result.Message | Should -Match 'declined|not performed'
        [IO.Directory]::Exists($script:StagingDir) | Should -BeFalse
        @(Get-ChildItem -LiteralPath $script:CaseTemp -Recurse -Force).Count | Should -Be 0
        Should -Invoke Test-NvidiaDownloadUrl -Exactly -Times 0
        Should -Invoke Invoke-WebRequest -Exactly -Times 0
        Should -Invoke Backup-NvidiaEnvironment -Exactly -Times 0
        Should -Invoke Invoke-NvidiaSilentInstall -Exactly -Times 0
    }

    # Protects: ambient declines have the same effect as explicit WhatIf.
    # Detects: a flag-only check that misses the actual ShouldProcess decision.
    # Needs: normal public invocation with inherited preference, not a mocked PSCmdlet.
    # Breadcrumb: inherited WhatIfPreference drives the genuine cmdlet decision.
    It 'honors inherited declined decision without side effects: <Download>' -ForEach @(
        @{ Download = $true }, @{ Download = $false }
    ) {
        $WhatIfPreference = $true
        $result = Install-NvidiaSoftware -ComponentId fixture -RegistryPath $script:RegistryPath -DownloadOnly:$Download
        $result.Success | Should -BeFalse
        $result.InstallerPath | Should -BeNullOrEmpty
        [IO.Directory]::Exists($script:StagingDir) | Should -BeFalse
        Should -Invoke Test-NvidiaDownloadUrl -Exactly -Times 0
        Should -Invoke Invoke-WebRequest -Exactly -Times 0
        Should -Invoke Backup-NvidiaEnvironment -Exactly -Times 0
        Should -Invoke Invoke-NvidiaSilentInstall -Exactly -Times 0
    }

    # Protects: an approved download produces a verified local artifact without installation.
    # Detects: refusal incorrectly treated as download success, or successful bytes reported without a file.
    # Needs: tiny actual file writes and actual SHA256; only the HTTP transport is mocked.
    # Breadcrumb: DownloadOnly return follows completed local staging.
    It 'downloads exact tiny bytes with digest and no installer or backup' {
        $result = Install-NvidiaSoftware -ComponentId fixture -RegistryPath $script:RegistryPath -DownloadOnly -Confirm:$false
        $result.Success | Should -BeTrue
        $result.InstallerPath | Should -BeExactly $script:StagedFile
        (Get-FileHash -LiteralPath $result.InstallerPath).Hash | Should -BeExactly $script:PayloadHash
        [Convert]::ToBase64String([IO.File]::ReadAllBytes($result.InstallerPath)) | Should -BeExactly ([Convert]::ToBase64String($script:Payload))
        Should -Invoke Invoke-WebRequest -Exactly -Times 1
        Should -Invoke Backup-NvidiaEnvironment -Exactly -Times 0
        Should -Invoke Invoke-NvidiaSilentInstall -Exactly -Times 0
    }

    # Protects: cached installers still have their actual digest checked.
    # Detects: cache presence bypasses hash verification or unnecessarily downloads valid content.
    # Needs: genuine valid/invalid small cached files, unchanged coordinator digest code.
    # Breadcrumb: cached bytes are not accepted solely by existence.
    It 'accepts a digest-matched cached file without network download' {
        [IO.Directory]::CreateDirectory($script:StagingDir) | Out-Null
        [IO.File]::WriteAllBytes($script:StagedFile, $script:Payload)
        $result = Install-NvidiaSoftware -ComponentId fixture -RegistryPath $script:RegistryPath -DownloadOnly -Confirm:$false
        $result.Success | Should -BeTrue
        (Get-FileHash -LiteralPath $result.InstallerPath).Hash | Should -BeExactly $script:PayloadHash
        Should -Invoke Invoke-WebRequest -Exactly -Times 0
    }
    It 'refuses corrupted cached bytes without installing them' {
        [IO.Directory]::CreateDirectory($script:StagingDir) | Out-Null
        [IO.File]::WriteAllText($script:StagedFile, 'corrupt public bytes')
        { Install-NvidiaSoftware -ComponentId fixture -RegistryPath $script:RegistryPath -DownloadOnly -Confirm:$false } | Should -Throw '*SHA-256*'
        Should -Invoke Invoke-WebRequest -Exactly -Times 0
        Should -Invoke Invoke-NvidiaSilentInstall -Exactly -Times 0
    }

    # Protects: invalid origin, reachability and content retain failures before installation.
    # Detects: mocked validation trust overriding the independent host allowlist, or error-to-success conversion.
    # Needs: real URL parsing/hash and public function, mocked network validation/transport.
    # Breadcrumb: neither a vendor boundary nor backup is permitted after download failure.
    It 'rejects an untrusted host even if URL boundary incorrectly claims trust' {
        Write-CoordinatorRegistry -Url 'https://nvidia.com.attacker.invalid/fixture.exe'
        { Install-NvidiaSoftware -ComponentId fixture -RegistryPath $script:RegistryPath -DownloadOnly -Confirm:$false } | Should -Throw '*not from a trusted NVIDIA host*'
        Should -Invoke Invoke-WebRequest -Exactly -Times 0
        Should -Invoke Invoke-NvidiaSilentInstall -Exactly -Times 0
    }
    It 'preserves an unreachable URL refusal before staging' {
        Mock Test-NvidiaDownloadUrl { [pscustomobject]@{ IsTrusted = $true; IsValid = $false; StatusCode = 503 } }
        { Install-NvidiaSoftware -ComponentId fixture -RegistryPath $script:RegistryPath -DownloadOnly -Confirm:$false } | Should -Throw '*not reachable*503*'
        [IO.Directory]::Exists($script:StagingDir) | Should -BeFalse
        Should -Invoke Invoke-WebRequest -Exactly -Times 0
    }
    It 'retains a real downloaded hash mismatch and never installs' {
        Write-CoordinatorRegistry -Hash ('0' * 64)
        { Install-NvidiaSoftware -ComponentId fixture -RegistryPath $script:RegistryPath -Confirm:$false } | Should -Throw '*SHA-256*'
        Should -Invoke Invoke-WebRequest -Exactly -Times 1
        Should -Invoke Backup-NvidiaEnvironment -Exactly -Times 0
        Should -Invoke Invoke-NvidiaSilentInstall -Exactly -Times 0
    }
    It 'retains the HTTP boundary failure cause without installation' {
        Mock Invoke-WebRequest { throw 'public transport failure' }
        { Install-NvidiaSoftware -ComponentId fixture -RegistryPath $script:RegistryPath -Confirm:$false } | Should -Throw '*public transport failure*'
        Should -Invoke Backup-NvidiaEnvironment -Exactly -Times 0
        Should -Invoke Invoke-NvidiaSilentInstall -Exactly -Times 0
    }

    # Protects: local paths bypass download while real installer/post-status semantics remain intact.
    # Detects: local content accidentally treated as URL, lost 3010 status or failed installer promoted.
    # Needs: actual tiny local file and explicit mocks only at vendor, backup and machine-status boundaries.
    # Breadcrumb: post-status metadata is separate from successful vendor exit semantics.
    It 'uses the supplied actual file and propagates reboot and post-status' {
        $localFile = Join-Path $script:CaseRoot 'supplied.exe'
        [IO.File]::WriteAllBytes($localFile, $script:Payload)
        Mock Invoke-NvidiaSilentInstall { [pscustomobject]@{ Success = $true; ExitCode = 3010; RebootRequired = $true; LogPath = 'public-log-boundary' } }
        Mock Get-NvidiaSoftwareStatus { [pscustomobject]@{ InstalledVersion = '2.0'; Status = 'Current' } }
        $result = Install-NvidiaSoftware -ComponentId fixture -RegistryPath $script:RegistryPath -InstallerPath $localFile -Force -TimeoutSeconds 90 -Confirm:$false
        $result.Success | Should -BeTrue
        $result.RebootRequired | Should -BeTrue
        $result.InstallerPath | Should -BeExactly $localFile
        $result.VersionAfter | Should -BeExactly '2.0'
        Should -Invoke Invoke-NvidiaSilentInstall -Exactly -Times 1 -ParameterFilter { $InstallerPath -ceq $localFile -and $TimeoutSeconds -eq 90 -and $ComponentId -ceq 'fixture' }
        Should -Invoke Get-NvidiaSoftwareStatus -Exactly -Times 2
        Should -Invoke Test-NvidiaDownloadUrl -Exactly -Times 0
        [IO.Directory]::Exists($script:StagingDir) | Should -BeFalse
    }
    It 'does not call backup or installer for a declined local-file action' {
        $localFile = Join-Path $script:CaseRoot 'supplied.exe'
        [IO.File]::WriteAllBytes($localFile, $script:Payload)
        $before = (Get-FileHash -LiteralPath $localFile).Hash
        $result = Install-NvidiaSoftware -ComponentId fixture -RegistryPath $script:RegistryPath -InstallerPath $localFile -WhatIf
        $result.Success | Should -BeFalse
        (Get-FileHash -LiteralPath $localFile).Hash | Should -BeExactly $before
        Should -Invoke Backup-NvidiaEnvironment -Exactly -Times 0
        Should -Invoke Invoke-NvidiaSilentInstall -Exactly -Times 0
    }
    It 'refuses a missing supplied file before any backup or installer' {
        { Install-NvidiaSoftware -ComponentId fixture -RegistryPath $script:RegistryPath -InstallerPath (Join-Path $script:CaseRoot 'absent.exe') -Confirm:$false } | Should -Throw '*does not exist*'
        Should -Invoke Test-NvidiaDownloadUrl -Exactly -Times 0
        Should -Invoke Backup-NvidiaEnvironment -Exactly -Times 0
        Should -Invoke Invoke-NvidiaSilentInstall -Exactly -Times 0
    }
    It 'preserves a failed vendor result and does not query post-status' {
        $localFile = Join-Path $script:CaseRoot 'supplied.exe'
        [IO.File]::WriteAllBytes($localFile, $script:Payload)
        Mock Invoke-NvidiaSilentInstall { [pscustomobject]@{ Success = $false; ExitCode = 23; RebootRequired = $false; LogPath = 'public-log-boundary' } }
        { Install-NvidiaSoftware -ComponentId fixture -RegistryPath $script:RegistryPath -InstallerPath $localFile -Confirm:$false } | Should -Throw '*exit code 23*'
        Should -Invoke Get-NvidiaSoftwareStatus -Exactly -Times 1
        Should -Invoke Backup-NvidiaEnvironment -Exactly -Times 1
    }

    # Protects: a declined download performs no URL reachability request.
    # Detects: the predecessor probes the URL before its genuine download decision.
    # Needs: actual WhatIf decision and a counted explicit HTTP-validation boundary.
    # Breadcrumb: this assertion executes independently of the result.Success detector.
    It 'does not probe URL reachability after a declined download' {
        Install-NvidiaSoftware -ComponentId fixture -RegistryPath $script:RegistryPath -DownloadOnly -WhatIf | Out-Null
        Should -Invoke Test-NvidiaDownloadUrl -Exactly -Times 0
        Should -Invoke Invoke-WebRequest -Exactly -Times 0
    }

    # Protects: a declined local backup does not delegate a vendor installation.
    # Detects: the predecessor calls the installer boundary despite the backup refusal.
    # Needs: an actual tiny local file and actual WhatIf, with no real vendor process.
    # Breadcrumb: this boundary detector is separate from false-success accounting.
    It 'does not delegate an installer after a declined local backup' {
        $localFile = Join-Path $script:CaseRoot 'supplied.exe'
        [IO.File]::WriteAllBytes($localFile, $script:Payload)
        Install-NvidiaSoftware -ComponentId fixture -RegistryPath $script:RegistryPath -InstallerPath $localFile -WhatIf | Out-Null
        Should -Invoke Invoke-NvidiaSilentInstall -Exactly -Times 0
    }

    # Protects: already-current no-op success remains accurate without external work.
    # Detects: a blanket WhatIf-failure shortcut incorrectly rejects a genuine no-op.
    # Needs: actual coordinator with machine status mocked as already current.
    # Breadcrumb: the early current-status return does not claim an installation.
    It 'keeps already-current no-op success without staging or vendor work' {
        Mock Get-NvidiaSoftwareStatus { [pscustomobject]@{ InstalledVersion = '2.0'; Status = 'Current' } }
        $result = Install-NvidiaSoftware -ComponentId fixture -RegistryPath $script:RegistryPath -WhatIf
        $result.Success | Should -BeTrue
        $result.Message | Should -Match 'already Current'
        $result.InstallerPath | Should -BeNullOrEmpty
        $result.VersionAfter | Should -BeNullOrEmpty
        [IO.Directory]::Exists($script:StagingDir) | Should -BeFalse
        Should -Invoke Test-NvidiaDownloadUrl -Exactly -Times 0
        Should -Invoke Backup-NvidiaEnvironment -Exactly -Times 0
        Should -Invoke Invoke-NvidiaSilentInstall -Exactly -Times 0
    }

    # Protects: validating an existing local download-only artifact is nonmutating success.
    # Detects: refusal logic incorrectly treats a real supplied file as hypothetical staging.
    # Needs: actual file identity and digest, with WhatIf and DownloadOnly together.
    # Breadcrumb: this mode validates availability; it does not claim installed software.
    It 'keeps supplied-file DownloadOnly availability success under WhatIf' {
        $localFile = Join-Path $script:CaseRoot 'supplied.exe'
        [IO.File]::WriteAllBytes($localFile, $script:Payload)
        $result = Install-NvidiaSoftware -ComponentId fixture -RegistryPath $script:RegistryPath -InstallerPath $localFile -DownloadOnly -WhatIf
        $result.Success | Should -BeTrue
        $result.InstallerPath | Should -BeExactly $localFile
        $result.VersionAfter | Should -BeNullOrEmpty
        (Get-FileHash -LiteralPath $localFile).Hash | Should -BeExactly $script:PayloadHash
        Should -Invoke Test-NvidiaDownloadUrl -Exactly -Times 0
        Should -Invoke Backup-NvidiaEnvironment -Exactly -Times 0
        Should -Invoke Invoke-NvidiaSilentInstall -Exactly -Times 0
    }
}
