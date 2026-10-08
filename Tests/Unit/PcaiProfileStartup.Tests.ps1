#Requires -Version 7.0
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }

[Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSAvoidGlobalVars', '', Justification = 'Dedicated Pester race fixture crosses the real external-script mock boundary and is removed after every test.')]
param()

BeforeAll {
    $script:RepairTool = Join-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) 'Tools/Repair-PcaiProfileStartup.ps1'
    $script:OriginalFixture = @'
$script:ProfileLevel = 'minimal'
function script:Get-PreferredModulesRoot {
    return 'C:\fixture-local-modules'
}
function script:Set-ProfileEnvFromUser {
    param($Name)
    return $false
}
function script:Optimize-PSModulePath {
    param($AllowNetworkPaths)
}
$script:AllowNetworkModulePath = $false
Optimize-PSModulePath -AllowNetworkPaths:$script:AllowNetworkModulePath
$script:UserPowerShellDir = Split-Path -Parent $PROFILE.CurrentUserCurrentHost
$script:UserPSModulesRoot = Join-Path $script:UserPowerShellDir 'Modules'
function script:Initialize-AgentBusAuthToken {
    if (Set-ProfileEnvFromUser -Name 'AGENT_BUS_AUTH_TOKEN') {
        return $true
    }

    $secretValue = Get-ProfileSecretFromSecretManagement -Name @(
        'AGENT_BUS_AUTH_TOKEN',
        'agent-bus/auth-token',
        'agent_bus/auth_token'
    )
    if (-not [string]::IsNullOrWhiteSpace($secretValue)) {
        $env:AGENT_BUS_AUTH_TOKEN = $secretValue
        return $true
    }
    return $false
}
# Keep unrelated profile text exactly as written: λ and 日本語.
'@
    function Get-TestPatchedFunction {
        param([string]$Text, [string]$Name)
        $tokens = $null; $errors = $null
        $ast = [Management.Automation.Language.Parser]::ParseInput($Text, [ref]$tokens, [ref]$errors)
        $function = $ast.Find({ param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq $Name }, $true)
        return [scriptblock]::Create($function.Extent.Text)
    }
}

Describe 'Guarded private profile startup repair' -Skip:(-not $IsWindows) {
    BeforeEach {
        $script:FixtureRoot = Join-Path $TestDrive ([guid]::NewGuid().ToString('N'))
        [void](New-Item -ItemType Directory -Path $script:FixtureRoot)
        $script:FixtureProfile = Join-Path $script:FixtureRoot 'canonical-profile.ps1'
        $script:FixtureBackup = Join-Path $script:FixtureRoot 'private-backup'
        [IO.File]::WriteAllText($script:FixtureProfile, $script:OriginalFixture, [Text.UTF8Encoding]::new($false))
        $script:ExpectedHash = (Get-FileHash -LiteralPath $script:FixtureProfile).Hash
        $script:SavedModulePath = $env:PSModulePath
        $script:SavedBusToken = $env:AGENT_BUS_AUTH_TOKEN
        $script:SavedLocalAppData = $env:LOCALAPPDATA
        $script:SavedProgramData = $env:ProgramData
        $script:Parameters = @{ ProfilePath = $script:FixtureProfile; BackupRoot = $script:FixtureBackup }
        # Pester 6 rejects calls outside a filtered mock. Keep real hashing for
        # originals/displaced custody; only the specific race paths are injected.
        Mock Get-FileHash {
            $hashCommand = Get-Command Microsoft.PowerShell.Utility\Get-FileHash -CommandType Cmdlet
            & $hashCommand -LiteralPath $LiteralPath -Algorithm SHA256
        }
    }
    AfterEach {
        $env:PSModulePath = $script:SavedModulePath
        $env:AGENT_BUS_AUTH_TOKEN = $script:SavedBusToken
        $env:LOCALAPPDATA = $script:SavedLocalAppData
        $env:ProgramData = $script:SavedProgramData
        Remove-Variable -Name PcaiProfileRaceFixture -Scope Global -ErrorAction SilentlyContinue
    }

    It 'plans without file, registry or environment writes using <Mode>' -ForEach @(
        @{ Mode = 'Default' }, @{ Mode = 'DryRun' }, @{ Mode = 'WhatIf' }, @{ Mode = 'Help' }, @{ Mode = 'h' }
    ) {
        $userPath = [Environment]::GetEnvironmentVariable('PSModulePath', 'User')
        $arguments = $script:Parameters.Clone()
        if ($Mode -eq 'WhatIf') { $arguments.Apply = $true; $arguments.WhatIf = $true }
        elseif ($Mode -ne 'Default') { $arguments[$Mode] = $true }
        & $script:RepairTool @arguments
        (Get-FileHash -LiteralPath $script:FixtureProfile).Hash | Should -BeExactly $script:ExpectedHash
        Test-Path -LiteralPath $script:FixtureBackup | Should -BeFalse
        $env:PSModulePath | Should -BeExactly $script:SavedModulePath
        [Environment]::GetEnvironmentVariable('PSModulePath', 'User') | Should -BeExactly $userPath
        @(Get-ChildItem -LiteralPath $script:FixtureRoot -File).Count | Should -Be 1
    }

    It 'accepts --help without reading a missing profile' {
        & $script:RepairTool -ProfilePath (Join-Path $TestDrive 'absent') --help | Should -Match '^Usage:'
    }

    It 'requires expected hash for apply and rejects stale hash without writes' {
        { & $script:RepairTool @script:Parameters -Apply } | Should -Throw '*requires ExpectedSha256*'
        { & $script:RepairTool @script:Parameters -Apply -ExpectedSha256 ('0' * 64) } | Should -Throw '*does not match*'
        Test-Path -LiteralPath $script:FixtureBackup | Should -BeFalse
    }

    It 'preserves exact original bytes with private ACL/hash receipt and is idempotent' {
        $bytes = [IO.File]::ReadAllBytes($script:FixtureProfile)
        $result = & $script:RepairTool @script:Parameters -Apply -ExpectedSha256 $script:ExpectedHash -Confirm:$false
        $result.State | Should -Be Applied
        $result.Changes.Count | Should -Be 3
        $receipt = Get-Content -LiteralPath $result.Receipt -Raw | ConvertFrom-Json
        [Convert]::ToHexString([IO.File]::ReadAllBytes($receipt.BackupPath)) | Should -BeExactly ([Convert]::ToHexString($bytes))
        $receipt.BeforeSha256 | Should -BeExactly $script:ExpectedHash
        $receipt.AfterSha256 | Should -BeExactly (Get-FileHash -LiteralPath $script:FixtureProfile).Hash
        (Get-Acl -LiteralPath (Split-Path $receipt.BackupPath -Parent)).AreAccessRulesProtected | Should -BeTrue
        $patched = [IO.File]::ReadAllText($script:FixtureProfile)
        $patched | Should -Match 'PS_SKIP_PSMODULEPATH_OPTIMIZE'
        $patched | Should -Match '\$script:UserPSModulesRoot = Get-PreferredModulesRoot'
        $patched | Should -Match '# Keep unrelated profile text exactly as written: λ and 日本語\.'
        $again = & $script:RepairTool @script:Parameters -Apply -ExpectedSha256 $result.AfterSha256 -Confirm:$false
        $again.State | Should -Be AlreadyApplied
        @(Get-ChildItem -LiteralPath (Split-Path (Split-Path $result.Receipt -Parent) -Parent) -Directory).Count | Should -Be 1
    }

    It 'preserves UTF16 BOM and CRLF bytes surrounding the three repairs' {
        $encoding = [Text.UnicodeEncoding]::new($false, $true)
        $body = $script:OriginalFixture.Replace("`r`n", "`n").Replace("`n", "`r`n")
        [IO.File]::WriteAllText($script:FixtureProfile, $body, $encoding)
        $original = [IO.File]::ReadAllBytes($script:FixtureProfile)
        $hash = (Get-FileHash -LiteralPath $script:FixtureProfile).Hash
        $result = & $script:RepairTool @script:Parameters -Apply -ExpectedSha256 $hash -Confirm:$false
        $bytes = [IO.File]::ReadAllBytes($script:FixtureProfile)
        $bytes[0] | Should -Be 255
        $bytes[1] | Should -Be 254
        [IO.File]::ReadAllText($script:FixtureProfile).Replace("`r`n", '').Contains("`n") | Should -BeFalse
        $receipt = Get-Content -LiteralPath $result.Receipt -Raw | ConvertFrom-Json
        [Convert]::ToHexString([IO.File]::ReadAllBytes($receipt.BackupPath)) | Should -BeExactly ([Convert]::ToHexString($original))
    }

    It 'fails closed on a missing or duplicate anchor: <Kind>' -ForEach @(
        @{ Kind = 'Missing' }, @{ Kind = 'Duplicate' }
    ) {
        $anchor = 'Optimize-PSModulePath -AllowNetworkPaths:$script:AllowNetworkModulePath'
        $body = if ($Kind -eq 'Missing') { $script:OriginalFixture.Replace($anchor, '# absent') } else { $script:OriginalFixture + "`n$anchor" }
        [IO.File]::WriteAllText($script:FixtureProfile, $body)
        $hash = (Get-FileHash -LiteralPath $script:FixtureProfile).Hash
        { & $script:RepairTool @script:Parameters -Apply -ExpectedSha256 $hash -Confirm:$false } | Should -Throw '*profile anchor*'
        Test-Path -LiteralPath $script:FixtureBackup | Should -BeFalse
        (Get-FileHash -LiteralPath $script:FixtureProfile).Hash | Should -BeExactly $hash
    }

    It 'preserves mixed existing line separators outside the precise repaired anchors' {
        $body = $script:OriginalFixture.Replace("`r`n", "`n").Replace("return 'C:\fixture-local-modules'`n", "return 'C:\fixture-local-modules'`r`n")
        $body = $body.Replace("function script:Initialize-AgentBusAuthToken {`n", "function script:Initialize-AgentBusAuthToken {`r`n")
        [IO.File]::WriteAllText($script:FixtureProfile, $body, [Text.UTF8Encoding]::new($false))
        $result = & $script:RepairTool @script:Parameters -Apply -ExpectedSha256 (Get-FileHash -LiteralPath $script:FixtureProfile).Hash -Confirm:$false
        $patched = [IO.File]::ReadAllText($script:FixtureProfile)
        $patched.Contains("return 'C:\fixture-local-modules'`r`n") | Should -BeTrue
        $patched.Contains("function script:Initialize-AgentBusAuthToken {`r`n") | Should -BeTrue
        $patched.Contains("    if (Set-ProfileEnvFromUser -Name 'AGENT_BUS_AUTH_TOKEN') {`n") | Should -BeTrue
        (& $script:RepairTool @script:Parameters -DryRun).State | Should -Be AlreadyApplied
        $result.State | Should -Be Applied
    }

    It 'rejects a broken source parse before creating backup or staging' {
        [IO.File]::WriteAllText($script:FixtureProfile, $script:OriginalFixture + "`nfunction Broken {")
        { & $script:RepairTool @script:Parameters -Apply -ExpectedSha256 (Get-FileHash -LiteralPath $script:FixtureProfile).Hash -Confirm:$false } | Should -Throw '*parse validation*'
        Test-Path -LiteralPath $script:FixtureBackup | Should -BeFalse
    }

    It 'refuses replacement if the reviewed profile hash changes before publication' {
        Mock Get-FileHash { [pscustomobject]@{ Hash = '0' * 64 } } -ParameterFilter { $LiteralPath -like '*canonical-profile.ps1' }
        { & $script:RepairTool @script:Parameters -Apply -ExpectedSha256 $script:ExpectedHash -Confirm:$false } | Should -Throw '*changed after review*'
        [IO.File]::ReadAllText($script:FixtureProfile) | Should -BeExactly $script:OriginalFixture
        $receipt = Get-ChildItem -LiteralPath $script:FixtureBackup -Filter receipt.json -Recurse -File | Get-Content -Raw | ConvertFrom-Json
        $receipt.State | Should -Be FailedBeforeReplace
    }

    It 'restores original bytes after a post-publication receipt failure' {
        Mock Set-Content {
            $text = $Value -join "`n"
            if ($text -match '"State": "Applied"') { throw 'Injected post-publication receipt failure' }
            [IO.File]::WriteAllText($LiteralPath, $text, [Text.UTF8Encoding]::new($false))
        }
        { & $script:RepairTool @script:Parameters -Apply -ExpectedSha256 $script:ExpectedHash -Confirm:$false } | Should -Throw '*Injected post-publication*'
        (Get-FileHash -LiteralPath $script:FixtureProfile).Hash | Should -BeExactly $script:ExpectedHash
        $receipt = Get-ChildItem -LiteralPath $script:FixtureBackup -Filter receipt.json -Recurse -File | Get-Content -Raw | ConvertFrom-Json
        $receipt.State | Should -Be RolledBack
        (Get-FileHash -LiteralPath $receipt.BackupPath).Hash | Should -BeExactly $script:ExpectedHash
        (Get-FileHash -LiteralPath $receipt.DisplacedPath).Hash | Should -BeExactly $script:ExpectedHash
        $receipt.RecoveryAttempts.Count | Should -Be 1
        (Get-FileHash -LiteralPath $receipt.RecoveryAttempts[0].DisplacedPath).Hash | Should -BeExactly $receipt.AfterSha256
    }

    It 'captures and restores writer bytes arriving between the last hash check and replacement' {
        $writer = [Text.Encoding]::UTF8.GetBytes($script:OriginalFixture + "`n# concurrent writer one")
        $writerHash = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($writer))
        $global:PcaiProfileRaceFixture = @{ Reads = 0; Bytes = @($writer); Profile = $script:FixtureProfile }
        Mock Get-FileHash {
            $observed = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData([IO.File]::ReadAllBytes($LiteralPath)))
            $global:PcaiProfileRaceFixture.Reads++
            if ($global:PcaiProfileRaceFixture.Reads -eq 1) { [IO.File]::WriteAllBytes($LiteralPath, [byte[]]$global:PcaiProfileRaceFixture.Bytes) }
            [pscustomobject]@{ Hash = $observed }
        } -ParameterFilter { $LiteralPath -like '*canonical-profile.ps1' }
        { & $script:RepairTool @script:Parameters -Apply -ExpectedSha256 $script:ExpectedHash -Confirm:$false } | Should -Throw '*changed at replacement boundary*'
        [Convert]::ToHexString([IO.File]::ReadAllBytes($script:FixtureProfile)) | Should -BeExactly ([Convert]::ToHexString($writer))
        $receipt = Get-ChildItem -LiteralPath $script:FixtureBackup -Filter receipt.json -Recurse -File | Get-Content -Raw | ConvertFrom-Json
        $receipt.State | Should -Be RolledBack
        $receipt.DisplacedSha256 | Should -BeExactly $writerHash
        (Get-FileHash -LiteralPath $receipt.DisplacedPath).Hash | Should -BeExactly $writerHash
        (Get-FileHash -LiteralPath $receipt.BackupPath).Hash | Should -BeExactly $script:ExpectedHash
        $receipt.RecoveryAttempts.Count | Should -Be 1
    }

    It 'captures a second writer racing rollback and restores its newer bytes' {
        $first = [Text.Encoding]::UTF8.GetBytes($script:OriginalFixture + "`n# concurrent writer one")
        $second = [Text.Encoding]::UTF8.GetBytes($script:OriginalFixture + "`n# concurrent writer two")
        $firstHash = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($first))
        $secondHash = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($second))
        $global:PcaiProfileRaceFixture = @{ Reads = 0; First = $first; Second = $second }
        Mock Get-FileHash {
            $observed = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData([IO.File]::ReadAllBytes($LiteralPath)))
            $global:PcaiProfileRaceFixture.Reads++
            if ($global:PcaiProfileRaceFixture.Reads -eq 1) { [IO.File]::WriteAllBytes($LiteralPath, $global:PcaiProfileRaceFixture.First) }
            if ($global:PcaiProfileRaceFixture.Reads -eq 2) { [IO.File]::WriteAllBytes($LiteralPath, $global:PcaiProfileRaceFixture.Second) }
            [pscustomobject]@{ Hash = $observed }
        } -ParameterFilter { $LiteralPath -like '*canonical-profile.ps1' }
        { & $script:RepairTool @script:Parameters -Apply -ExpectedSha256 $script:ExpectedHash -Confirm:$false } | Should -Throw '*changed at replacement boundary*'
        [Convert]::ToHexString([IO.File]::ReadAllBytes($script:FixtureProfile)) | Should -BeExactly ([Convert]::ToHexString($second))
        $receipt = Get-ChildItem -LiteralPath $script:FixtureBackup -Filter receipt.json -Recurse -File | Get-Content -Raw | ConvertFrom-Json
        $receipt.DisplacedSha256 | Should -BeExactly $firstHash
        $receipt.RecoveryAttempts.Count | Should -Be 2
        $receipt.RecoveryAttempts[0].DisplacedSha256 | Should -BeExactly $secondHash
        $receipt.RecoveryAttempts[1].DisplacedSha256 | Should -BeExactly $firstHash
        foreach ($attempt in $receipt.RecoveryAttempts) {
            (Get-FileHash -LiteralPath $attempt.DisplacedPath).Hash | Should -BeExactly $attempt.DisplacedSha256
        }
    }

    It 'stops persistent boundary races with explicit review state and custody of every displaced version' {
        $global:PcaiProfileRaceFixture = @{ Reads = 0; Original = $script:OriginalFixture; Writers = [Collections.Generic.List[byte[]]]::new() }
        Mock Get-FileHash {
            $observed = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData([IO.File]::ReadAllBytes($LiteralPath)))
            $global:PcaiProfileRaceFixture.Reads++
            $writer = [Text.Encoding]::UTF8.GetBytes($global:PcaiProfileRaceFixture.Original + "`n# writer $($global:PcaiProfileRaceFixture.Reads)")
            $global:PcaiProfileRaceFixture.Writers.Add($writer)
            [IO.File]::WriteAllBytes($LiteralPath, $writer)
            [pscustomobject]@{ Hash = $observed }
        } -ParameterFilter { $LiteralPath -like '*canonical-profile.ps1' }
        { & $script:RepairTool @script:Parameters -Apply -ExpectedSha256 $script:ExpectedHash -Confirm:$false } | Should -Throw '*recovery requires review*'
        $receipt = Get-ChildItem -LiteralPath $script:FixtureBackup -Filter receipt.json -Recurse -File | Get-Content -Raw | ConvertFrom-Json
        $receipt.State | Should -Be RecoveryRequiresReview
        $receipt.RecoveryAttempts.Count | Should -Be 3
        $paths = @($receipt.DisplacedPath) + @($receipt.RecoveryAttempts.DisplacedPath)
        $paths.Count | Should -Be $global:PcaiProfileRaceFixture.Writers.Count
        for ($index = 0; $index -lt $paths.Count; $index++) {
            [Convert]::ToHexString([IO.File]::ReadAllBytes($paths[$index])) | Should -BeExactly ([Convert]::ToHexString($global:PcaiProfileRaceFixture.Writers[$index]))
        }
    }

    It 'rejects backup roots inside Git before storing private original bytes' {
        [void](New-Item -ItemType Directory -Path (Join-Path $script:FixtureRoot '.git'))
        { & $script:RepairTool @script:Parameters -Apply -ExpectedSha256 $script:ExpectedHash -Confirm:$false } | Should -Throw '*outside Git*'
        Test-Path -LiteralPath $script:FixtureBackup | Should -BeFalse
    }

    It 'resolves whole-home Git default custody outside that checkout without read-only writes' {
        $gitHome = Join-Path $script:FixtureRoot 'home-checkout'
        $profileRoot = Join-Path $gitHome '.config/powershell'
        [void](New-Item -ItemType Directory -Path $profileRoot -Force)
        [void](New-Item -ItemType Directory -Path (Join-Path $gitHome '.git'))
        $profile = Join-Path $profileRoot 'canonical-profile.ps1'
        [IO.File]::Move($script:FixtureProfile, $profile)
        $env:LOCALAPPDATA = Join-Path $gitHome 'AppData/Local'
        $env:ProgramData = Join-Path $script:FixtureRoot 'shared-machine-data'
        $beforeFiles = @(Get-ChildItem -LiteralPath $script:FixtureRoot -File -Recurse | Select-Object -ExpandProperty FullName)
        $planned = & $script:RepairTool -ProfilePath $profile -DryRun
        $sid = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
        $planned.BackupRoot | Should -BeExactly (Join-Path $env:ProgramData "PC_AI/ProfileRepairs/$sid")
        $planned.BackupRootRequired | Should -BeFalse
        $planned.Changes.Count | Should -Be 3
        Test-Path -LiteralPath $env:ProgramData | Should -BeFalse
        Test-Path -LiteralPath $env:LOCALAPPDATA | Should -BeFalse
        @(Get-ChildItem -LiteralPath $script:FixtureRoot -File -Recurse | Select-Object -ExpandProperty FullName) | Should -Be $beforeFiles
        (Get-FileHash -LiteralPath $profile).Hash | Should -BeExactly $script:ExpectedHash
    }

    It 'applies whole-home Git profile repairs only with outside-Git backup and staging custody' {
        $gitHome = Join-Path $script:FixtureRoot 'home-checkout'
        [void](New-Item -ItemType Directory -Path $gitHome)
        [void](New-Item -ItemType Directory -Path (Join-Path $gitHome '.git'))
        $profile = Join-Path $gitHome 'canonical-profile.ps1'
        [IO.File]::Move($script:FixtureProfile, $profile)
        $env:LOCALAPPDATA = Join-Path $gitHome 'AppData/Local'
        $env:ProgramData = Join-Path $gitHome 'machine-data-also-in-git'
        $plan = & $script:RepairTool -ProfilePath $profile -DryRun
        $plan.BackupRootRequired | Should -BeTrue
        $plan.CustodyIssue | Should -Match 'Supply -BackupRoot'
        { & $script:RepairTool -ProfilePath $profile -Apply -ExpectedSha256 $script:ExpectedHash -Confirm:$false } | Should -Throw '*Supply -BackupRoot*'
        Test-Path -LiteralPath $env:LOCALAPPDATA | Should -BeFalse
        Test-Path -LiteralPath $env:ProgramData | Should -BeFalse
        $result = & $script:RepairTool -ProfilePath $profile -BackupRoot $script:FixtureBackup -Apply -ExpectedSha256 $script:ExpectedHash -Confirm:$false
        $receipt = Get-Content -LiteralPath $result.Receipt -Raw | ConvertFrom-Json
        $receipt.BackupPath.StartsWith($gitHome, [StringComparison]::OrdinalIgnoreCase) | Should -BeFalse
        $receipt.StagePath.StartsWith($gitHome, [StringComparison]::OrdinalIgnoreCase) | Should -BeFalse
        $receipt.DisplacedPath.StartsWith($gitHome, [StringComparison]::OrdinalIgnoreCase) | Should -BeFalse
        $result.State | Should -Be Applied
        (Get-FileHash -LiteralPath $receipt.BackupPath).Hash | Should -BeExactly $script:ExpectedHash
        @(Get-ChildItem -LiteralPath $gitHome -File -Recurse).Count | Should -Be 1
    }

    It 'keeps WhatIf readonly when every automatic custody candidate is inside Git' {
        [void](New-Item -ItemType Directory -Path (Join-Path $script:FixtureRoot '.git'))
        $env:LOCALAPPDATA = Join-Path $script:FixtureRoot 'local-app-data'
        $env:ProgramData = Join-Path $script:FixtureRoot 'machine-data'
        $result = & $script:RepairTool -ProfilePath $script:FixtureProfile -Apply -WhatIf
        $result.State | Should -Be Planned
        $result.BackupRootRequired | Should -BeTrue
        Test-Path -LiteralPath $env:LOCALAPPDATA | Should -BeFalse
        Test-Path -LiteralPath $env:ProgramData | Should -BeFalse
        (Get-FileHash -LiteralPath $script:FixtureProfile).Hash | Should -BeExactly $script:ExpectedHash
    }

    It 'rejects cross-volume explicit custody in read-only and Apply modes before writes' {
        $drive = if ([IO.Path]::GetPathRoot($script:FixtureProfile) -eq 'Z:\') { 'Y:\' } else { 'Z:\' }
        $otherVolume = $drive + 'pcai-private-fixture'
        { & $script:RepairTool -ProfilePath $script:FixtureProfile -BackupRoot $otherVolume -DryRun } | Should -Throw '*share the profile volume*'
        { & $script:RepairTool -ProfilePath $script:FixtureProfile -BackupRoot $otherVolume -Apply -ExpectedSha256 $script:ExpectedHash -Confirm:$false } | Should -Throw '*share the profile volume*'
        (Get-FileHash -LiteralPath $script:FixtureProfile).Hash | Should -BeExactly $script:ExpectedHash
        Test-Path -LiteralPath $script:FixtureBackup | Should -BeFalse
    }

    It 'keeps minimal token bootstrap cheap while retaining existing token and full fallback' {
        $result = & $script:RepairTool @script:Parameters -Apply -ExpectedSha256 $script:ExpectedHash -Confirm:$false
        . (Get-TestPatchedFunction -Text ([IO.File]::ReadAllText($script:FixtureProfile)) -Name 'script:Initialize-AgentBusAuthToken')
        function Set-ProfileEnvFromUser { param($Name) return $false }
        function Get-ProfileSecretFromSecretManagement { param($Name) throw 'Expensive fallback invoked' }
        $script:ProfileLevel = 'minimal'
        Initialize-AgentBusAuthToken | Should -BeFalse
        function Set-ProfileEnvFromUser { param($Name) return $true }
        Initialize-AgentBusAuthToken | Should -BeTrue
        function Set-ProfileEnvFromUser { param($Name) return $false }
        $script:ProfileLevel = 'full'
        { Initialize-AgentBusAuthToken } | Should -Throw '*Expensive fallback invoked*'
        $result.State | Should -Be Applied
    }
}
