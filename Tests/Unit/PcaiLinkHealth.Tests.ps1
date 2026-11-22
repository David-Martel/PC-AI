# Controls for the USB4 link-health watcher and its trigger installer
# (docs/usb4/link-health.md). They use in-memory adapters and plans, so they change no host state.
# The plan fixture mirrors the shape `pcai-tb route-plan` printed for the 2026-11-21 Gen7-NUC-Gen9
# chain, with documentation-range addresses and made-up MACs and GUIDs.

BeforeAll {
    $repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
    . (Join-Path $repoRoot 'Tools/SystemScripts/Networking/Invoke-PcaiLinkHealth.ps1')
    . (Join-Path $repoRoot 'Tools/SystemScripts/Networking/Install-PcaiLinkHealthTrigger.ps1')

    $guid = '11111111-2222-3333-4444-555555555555'
    $mac = '02-00-00-00-00-01'
    $journal = "Add-Content -LiteralPath (Join-Path `$d '0123456789abcdef.journal') -Value 'x'"
    $script:Plan = [pscustomobject]@{
        schema = 'pcai.usb4-route-plan/v1'
        expires_at_utc = '2026-11-21T11:51:17Z'
        edges = @([pscustomobject]@{
                subnet = '172.31.255.20/30'
                endpoints = @(
                    [pscustomobject]@{ host = 'hosta'; interface_index = 28; interface_guid = $guid; mac = $mac; ipv4_address = '172.31.255.21' },
                    [pscustomobject]@{ host = 'hostb'; interface_index = 19; interface_guid = '99999999-2222-3333-4444-555555555555'; mac = '02-00-00-00-00-02'; ipv4_address = '172.31.255.22' })
            })
        paths = @(
            [pscustomobject]@{ from = 'hosta'; to = 'hostb'; status = 'Direct'; destination_ipv4 = '172.31.255.22' },
            [pscustomobject]@{ from = 'hosta'; to = 'hostc'; status = 'Routed'; destination_ipv4 = '172.31.248.197' },
            [pscustomobject]@{ from = 'hostb'; to = 'hosta'; status = 'Direct'; destination_ipv4 = '172.31.255.21' })
        hosts = @([pscustomobject]@{
                host = 'hosta'; transit = $false
                guard = @("`$a = Get-NetAdapter -InterfaceIndex 28 -ErrorAction Stop; if (`$a.MacAddress -ne '$mac') { throw 'guard' }")
                preflight = @('Get-NetIPAddress -InterfaceIndex 28 -AddressFamily IPv4')
                actions = @(
                    [pscustomobject]@{ kind = 'EdgeAddress'; interface_index = 28; target = '172.31.255.21/30'; already_present = $false; apply = "`$d = 'x'; $journal" },
                    [pscustomobject]@{ kind = 'HostRoute'; interface_index = 28; target = '172.31.248.197/32 via 172.31.255.22'; already_present = $false; apply = "`$d = 'x'; $journal" })
            })
    }
    function New-TestAdapter {
        param([int]$Index = 28, [string]$Guid = $guid, [string]$Mac = $mac, [string]$Status = 'Up', [string[]]$Ipv4 = @('172.31.255.21/30'), [string]$Forwarding = 'Disabled')
        [pscustomobject]@{ InterfaceIndex = $Index; Name = 'Ethernet 18'; Description = 'USB4(TM) P2P Network Adapter'; Guid = $Guid; Mac = $Mac; Status = $Status; LinkSpeed = '20 Gbps'; Ipv4 = $Ipv4; Forwarding = $Forwarding }
    }
    $script:Route = @([pscustomobject]@{ InterfaceIndex = 28; DestinationPrefix = '172.31.248.197/32'; NextHop = '172.31.255.22' })
}

Describe 'Compare-PcaiLinkPlan classifies the live USB4 state against the plan' -Tag 'Unit', 'Portable' {
    It 'reports InSync when guards match and every planned address and route is present' {
        $r = Compare-PcaiLinkPlan -Plan $Plan -HostName 'hosta' -Adapter @(New-TestAdapter) -Route $Route
        $r.State | Should -Be 'InSync'
        $r.GuardsMatch | Should -BeTrue
    }
    It 'reports Drift, with the missing items, when guards match but the address is gone (reboot or replug on the same port)' {
        $r = Compare-PcaiLinkPlan -Plan $Plan -HostName 'hosta' -Adapter @(New-TestAdapter -Ipv4 @()) -Route @()
        $r.State | Should -Be 'Drift'
        @($r.Findings | Where-Object Status -eq 'Missing').Target | Should -Be @('172.31.255.21/30', '172.31.248.197/32 via 172.31.255.22')
    }
    It 'reports InstanceChanged when the planned ifIndex now names another adapter (new GUID and MAC after a hub-port change)' {
        $r = Compare-PcaiLinkPlan -Plan $Plan -HostName 'hosta' -Adapter @(New-TestAdapter -Guid 'aaaaaaaa-2222-3333-4444-555555555555' -Mac '02-00-00-00-00-09') -Route $Route
        $r.State | Should -Be 'InstanceChanged'
        ($r.Findings | Where-Object Kind -eq 'Guard').Status | Should -Be 'InstanceReplaced'
        ($r.Findings | Where-Object Kind -eq 'UnplannedAdapter') | Should -Not -BeNullOrEmpty
    }
    It 'reports InstanceChanged when the planned adapter moved to another ifIndex' {
        $r = Compare-PcaiLinkPlan -Plan $Plan -HostName 'hosta' -Adapter @(New-TestAdapter -Index 40) -Route @()
        $r.State | Should -Be 'InstanceChanged'
        ($r.Findings | Where-Object Kind -eq 'Guard').Status | Should -Be 'IndexChanged'
    }
    It 'reports InstanceChanged when the planned adapter is absent' {
        (Compare-PcaiLinkPlan -Plan $Plan -HostName 'hosta' -Adapter @() -Route @()).State | Should -Be 'InstanceChanged'
    }
    It 'reports LinkDown when the planned adapter is present but down' {
        (Compare-PcaiLinkPlan -Plan $Plan -HostName 'hosta' -Adapter @(New-TestAdapter -Status 'Disconnected') -Route $Route).State | Should -Be 'LinkDown'
    }
    It 'treats a host with no USB4 adapter and no plan as Absent, which is healthy' {
        (Compare-PcaiLinkPlan -Plan $null -HostName 'ms01' -Adapter @()).State | Should -Be 'Absent'
        Get-PcaiLinkHealthSeverity -State 'Absent' | Should -Be 'ok'
    }
    It 'reports LinkDown for a present but down adapter even without a plan (Linux discovery-only mode)' {
        (Compare-PcaiLinkPlan -Plan $null -HostName 'ms01' -Adapter @(New-TestAdapter -Status 'Down')).State | Should -Be 'LinkDown'
        Get-PcaiLinkHealthSeverity -State 'LinkDown' | Should -Be 'warn'
        (Compare-PcaiLinkPlan -Plan $Plan -HostName 'other' -Adapter @(New-TestAdapter -Status 'Down')).State | Should -Be 'LinkDown'
    }
    It 'reports NotInPlan for a host the plan does not mention' {
        (Compare-PcaiLinkPlan -Plan $Plan -HostName 'other' -Adapter @(New-TestAdapter)).State | Should -Be 'NotInPlan'
    }
    It 'checks forwarding for a transit host action' {
        $p = $Plan | ConvertTo-Json -Depth 8 | ConvertFrom-Json
        $p.hosts[0].actions = @([pscustomobject]@{ kind = 'Forwarding'; interface_index = 28; target = 'IPv4 forwarding on ifIndex 28'; apply = 'x' })
        (Compare-PcaiLinkPlan -Plan $p -HostName 'hosta' -Adapter @(New-TestAdapter -Forwarding 'Disabled')).State | Should -Be 'Drift'
        (Compare-PcaiLinkPlan -Plan $p -HostName 'hosta' -Adapter @(New-TestAdapter -Forwarding 'Enabled')).State | Should -Be 'InSync'
    }
}

# Protects: the watcher reads what the real planner writes. Detects: the hand-built $Plan above
# drifting from `pcai-tb route-plan` output (codex review of #344: "the hand-built test fixture
# masks the missing integration"). Needs: Tests/Fixtures/LinkHealth/usb4-route-plan-hub-chain.json,
# printed by pcai-tb built from main 9310df20 (#335) for the sanitized hub-chain observation in
# Native/pcai_core/pcai_tb_cli/tests/fixtures with transit dtm-super-nuc. Regenerate it with
# `pcai-tb route-plan` (docs/usb4/multi-hop-routing.md) when the plan schema changes.
Describe 'The watcher reads a plan printed by pcai-tb route-plan' -Tag 'Unit', 'Portable' {
    BeforeAll {
        $script:RealPlan = Read-PcaiRoutePlan -Path (Join-Path $repoRoot 'Tests/Fixtures/LinkHealth/usb4-route-plan-hub-chain.json')
        function New-PlannedAdapter {
            param($Plan, [string]$HostName, [switch]$NoAddress)
            foreach ($e in @($Plan.edges)) {
                foreach ($ep in @($e.endpoints | Where-Object host -eq $HostName)) {
                    $ipv4 = if ($NoAddress) { @() } else { @("$($ep.ipv4_address)/30") }
                    [pscustomobject]@{ InterfaceIndex = [int]$ep.interface_index; Name = $ep.interface_alias; Description = 'USB4(TM) P2P Network Adapter'
                        Guid = ([string]$ep.interface_guid).Trim('{}').ToLowerInvariant(); Mac = ([string]$ep.mac).ToUpperInvariant(); Status = 'Up'
                        LinkSpeed = '40 Gbps'; Ipv4 = $ipv4; Forwarding = 'Enabled' }
                }
            }
        }
        function Get-PlannedRoute {
            param($Plan, [string]$HostName)
            foreach ($a in @(($Plan.hosts | Where-Object host -eq $HostName).actions | Where-Object kind -eq 'HostRoute')) {
                if ([string]$a.target -match '^(?<dst>\S+)\s+via\s+(?<hop>\S+)$') {
                    [pscustomobject]@{ InterfaceIndex = [int]$a.interface_index; DestinationPrefix = $Matches.dst; NextHop = $Matches.hop }
                }
            }
        }
    }
    It 'accepts the schema and finds the plan ID in the journal paths' {
        $RealPlan.schema | Should -Be 'pcai.usb4-route-plan/v1'
        Get-PcaiRoutePlanId -Plan $RealPlan | Should -Match '^[0-9a-f]{16}$'
    }
    # Protects: direct and routed probes use the first physical edge in the plan.
    # Detects: destination-only probes and using a remote edge's address as the local source.
    # Needs: the real planner fixture above. Breadcrumb: Invoke-PcaiLinkHealth.ps1::Get-PcaiReachabilityTarget.
    It 'resolves the first-hop source and next hop for direct and routed targets' {
        $targets = Get-PcaiReachabilityTarget -Plan $RealPlan -HostName 'dtm-p1gen9'
        $routed = $targets | Where-Object Status -eq 'Routed'
        $direct = $targets | Where-Object Status -eq 'Direct'
        $routed.SourceAddress | Should -Be '172.31.255.21'
        $routed.InterfaceIndex | Should -Be 28
        $routed.NextHop | Should -Be '172.31.255.22'
        $direct.SourceAddress | Should -Be '172.31.255.21'
        $direct.NextHop | Should -Be '0.0.0.0'
    }
    # Protects: only traffic routed through the enrolled USB4 adapter counts as healthy.
    # Detects: a successful LAN fallback hiding a disconnected or misrouted USB4 peer.
    # Needs: Windows and mocked Windows route and probe calls. Breadcrumb: Invoke-PcaiLinkHealth.ps1::Test-PcaiPeerReachability.
    It 'suppresses probes when Windows selects a LAN interface and source' -Skip:(-not $IsWindows) {
        Mock Find-NetRoute {
            [pscustomobject]@{ InterfaceIndex = 42; IPAddress = '192.0.2.5' }
            [pscustomobject]@{ InterfaceIndex = 42; NextHop = '192.0.2.1' }
        }
        Mock Test-Connection { $true }
        Mock Test-PcaiTcpPort { $true }
        $adapters = @(New-PlannedAdapter -Plan $RealPlan -HostName 'dtm-p1gen9')
        $targets = Get-PcaiReachabilityTarget -Plan $RealPlan -HostName 'dtm-p1gen9'
        $reach = Test-PcaiPeerReachability -Target $targets -Adapter $adapters
        @($reach | Where-Object { $_.RouteStatus -eq 'RouteMismatch' }).Count | Should -Be 2
        Get-PcaiLinkHealthSeverity -State 'InSync' -Reachability $reach | Should -Be 'warn'
        Should -Invoke Test-Connection -Times 0 -Exactly
        Should -Invoke Test-PcaiTcpPort -Times 0 -Exactly
    }
    # Protects: a verified planned TCP probe uses the planned local IP.
    # Detects: a guard that blocks healthy traffic or silently drops the socket's source address.
    # Needs: Windows, the fixture adapter and mocked route/probe calls. Breadcrumb: Invoke-PcaiLinkHealth.ps1::Test-PcaiPeerReachability.
    It 'probes a matching direct route from the planned source' -Skip:(-not $IsWindows) {
        Mock Find-NetRoute {
            [pscustomobject]@{ InterfaceIndex = 28; IPAddress = '172.31.255.21' }
            [pscustomobject]@{ InterfaceIndex = 28; NextHop = '0.0.0.0' }
        }
        Mock Test-Connection { $true }
        Mock Test-PcaiTcpPort { $true }
        $adapters = @(New-PlannedAdapter -Plan $RealPlan -HostName 'dtm-p1gen9')
        $target = Get-PcaiReachabilityTarget -Plan $RealPlan -HostName 'dtm-p1gen9' | ForEach-Object { $_ } | Where-Object Status -eq 'Direct'
        $reach = Test-PcaiPeerReachability -Target $target -Adapter $adapters
        $reach[0].RouteStatus | Should -Be 'Match'
        $reach[0].Ping | Should -BeTrue
        $reach[0].Tcp | Should -BeTrue
        Should -Invoke Test-PcaiTcpPort -Times 1 -Exactly -ParameterFilter { $SourceAddress -eq '172.31.255.21' }
    }
    It 'reports InSync on <HostName> when every planned address and route is live' -ForEach @(
        @{ HostName = 'dtm-p1gen7' }, @{ HostName = 'dtm-super-nuc' }, @{ HostName = 'dtm-p1gen9' }
    ) {
        $r = Compare-PcaiLinkPlan -Plan $RealPlan -HostName $HostName -Adapter @(New-PlannedAdapter -Plan $RealPlan -HostName $HostName) -Route @(Get-PlannedRoute -Plan $RealPlan -HostName $HostName)
        $r.GuardsMatch | Should -BeTrue
        $r.State | Should -Be 'InSync'
    }
    It 'reports Drift when only state the plan owns is gone' {
        $r = Compare-PcaiLinkPlan -Plan $RealPlan -HostName 'dtm-p1gen9' -Adapter @(New-PlannedAdapter -Plan $RealPlan -HostName 'dtm-p1gen9' -NoAddress) -Route @()
        $r.State | Should -Be 'Drift'
    }
    # Protects: Drift means the generated script can repair it. Detects: Drift reported for an
    # address the plan relies on but does not own (apply null; the NUC's 172.31.255.22), which the
    # script cannot restore and whose prefix guard stops it (codex review of #344 at c463db6d).
    It 'reports NeedsRegeneration, and applies nothing, when a not-owned address is gone' {
        $r = Compare-PcaiLinkPlan -Plan $RealPlan -HostName 'dtm-super-nuc' -Adapter @(New-PlannedAdapter -Plan $RealPlan -HostName 'dtm-super-nuc' -NoAddress) -Route @()
        $r.State | Should -Be 'NeedsRegeneration'
        @($r.Findings | Where-Object Status -eq 'MissingNotOwned').Target | Should -Be @('172.31.255.22/30')
        $policy = [pscustomobject]@{ Enabled = $true; Reason = 'enabled'; PlanPath = 'x' }
        (Get-PcaiAutoApplyDecision -RunMode 'Apply' -Elevated $true -Policy $policy -Comparison $r).Apply | Should -BeFalse
        Get-PcaiLinkHealthSeverity -State $r.State | Should -Be 'warn'
    }
}

Describe 'Plan helpers' -Tag 'Unit', 'Portable' {
    It 'extracts the plan ID from the journal path embedded in the apply lines' {
        Get-PcaiRoutePlanId -Plan $Plan | Should -Be '0123456789abcdef'
    }
    It 'extracts the plan ID whatever the quoting of the journal path' {
        $p = [pscustomobject]@{ hosts = @([pscustomobject]@{ actions = @([pscustomobject]@{ apply = "`$j = (Join-Path `$env:ProgramData 'pcai-usb4\route-plan\24851368c5f4eaa3.journal'); x" }) }) }
        Get-PcaiRoutePlanId -Plan $p | Should -Be '24851368c5f4eaa3'
    }
    It 'lists only this host''s reachability targets' {
        @(Get-PcaiReachabilityTarget -Plan $Plan -HostName 'hosta' | ForEach-Object { $_ } | ForEach-Object Address) | Should -Be @('172.31.255.22', '172.31.248.197')
    }
    It 'renders the guard before any apply line, and the script parses' {
        $text = New-PcaiHostApplyScript -PlanHost $Plan.hosts[0] -PlanId '0123456789abcdef' -Source 'test'
        $text.IndexOf('Get-NetAdapter -InterfaceIndex 28') | Should -BeLessThan $text.IndexOf('# EdgeAddress')
        $text | Should -Match '#Requires -RunAsAdministrator'
        $tokens = $null; $errors = $null
        [void][Management.Automation.Language.Parser]::ParseInput($text, [ref]$tokens, [ref]$errors)
        @($errors).Count | Should -Be 0
    }
}

Describe 'Auto-apply stays off unless every condition holds' -Tag 'Unit', 'Portable' {
    BeforeAll {
        $enabled = [pscustomobject]@{ Enabled = $true; Reason = 'Enabled by admin-owned policy'; PlanPath = 'x' }
        $disabled = [pscustomobject]@{ Enabled = $false; Reason = 'No auto-apply policy (default off)'; PlanPath = $null }
        $drift = [pscustomobject]@{ State = 'Drift' }
    }
    It 'never applies in Observe mode' {
        (Get-PcaiAutoApplyDecision -RunMode 'Observe' -Elevated $true -Policy $enabled -Comparison $drift).Apply | Should -BeFalse
    }
    It 'never applies without the admin-owned policy (the default)' {
        (Get-PcaiAutoApplyDecision -RunMode 'Apply' -Elevated $true -Policy $disabled -Comparison $drift).Apply | Should -BeFalse
    }
    It 'never applies when the adapter instance changed' {
        (Get-PcaiAutoApplyDecision -RunMode 'Apply' -Elevated $true -Policy $enabled -Comparison ([pscustomobject]@{ State = 'InstanceChanged' })).Apply | Should -BeFalse
    }
    It 'never applies from a non-elevated process' -Skip:(-not $IsWindows) {
        (Get-PcaiAutoApplyDecision -RunMode 'Apply' -Elevated $false -Policy $enabled -Comparison $drift).Apply | Should -BeFalse
    }
    It 'applies only for Drift with guards matching, elevated, and enabled' -Skip:(-not $IsWindows) {
        (Get-PcaiAutoApplyDecision -RunMode 'Apply' -Elevated $true -Policy $enabled -Comparison $drift).Apply | Should -BeTrue
    }
    It 'ignores a policy file that a standard user owns' -Skip:(-not $IsWindows) {
        $dir = Join-Path $TestDrive 'policy'
        $null = New-Item -ItemType Directory -Path $dir
        $file = Join-Path $dir 'auto-apply.json'
        '{"schema":"pcai.link-health-auto-apply/v1","enabled":true,"planPath":"x"}' | Set-Content -LiteralPath $file
        $owner = (Get-Acl -LiteralPath $file).GetOwner([Security.Principal.SecurityIdentifier]).Value
        if ($owner -in @('S-1-5-32-544', 'S-1-5-18')) { Set-ItResult -Skipped -Because 'this elevated session creates admin-owned files' }
        Test-PcaiAdminOwnedPath -Path $file | Should -BeFalse
        (Get-PcaiAutoApplyPolicy -Path $file).Enabled | Should -BeFalse
    }
}

Describe 'Trust checks walk every ancestor' -Tag 'Unit' {
    It 'rejects a protected directory whose ancestor a standard user owns (user temp)' -Skip:(-not $IsWindows) {
        if (-not (Test-PcaiElevated)) { Set-ItResult -Skipped -Because 'creating an Administrators-owned directory needs elevation' }
        $dir = Join-Path $TestDrive 'protected'
        New-PcaiProtectedStatusRoot -Path $dir
        (Get-Acl -LiteralPath $dir).GetOwner([Security.Principal.SecurityIdentifier]).Value | Should -Be 'S-1-5-32-544'
        Test-PcaiAdminOwnedPath -Path $dir | Should -BeFalse
    }
    It 'returns false off Windows and for a missing path' {
        Test-PcaiAdminOwnedPath -Path (Join-Path $TestDrive 'missing.json') | Should -BeFalse
    }
    # Protects: the volume root is an ancestor like any other. Detects: the root skipped before its
    # ACL was read, so a custom volume whose root grants Users DeleteChild passed although they can
    # replace the whole protected tree (codex review of #344 at 45700340). Needs: Get-Acl mocked to
    # an admin-owned chain whose root alone varies. Positive control: C:\'s real shape (Users may
    # only add entries; TrustedInstaller owns it and holds full control) passes.
    It 'checks the volume root: <Name>' -Skip:(-not $IsWindows) -ForEach @(
        @{ Name = 'Users DeleteChild on the root is refused'; Rights = 'DeleteSubdirectoriesAndFiles'; Expected = $false }
        @{ Name = 'Users create-folder on the root (C:\ default) passes'; Rights = 'AppendData'; Expected = $true }
    ) {
        $file = Join-Path $TestDrive 'root-check.json'
        Set-Content -LiteralPath $file -Value '{}'
        $global:PcaiTestRootRights = $Rights
        try {
            Mock Get-Acl {
                $isRoot = $LiteralPath -match '^[A-Za-z]:\\?$'
                $rules = if ($isRoot) {
                    @([Security.AccessControl.FileSystemAccessRule]::new([Security.Principal.SecurityIdentifier]'S-1-5-11', $global:PcaiTestRootRights, 'None', 'None', 'Allow'),
                        [Security.AccessControl.FileSystemAccessRule]::new([Security.Principal.SecurityIdentifier]'S-1-5-80-956008885-3418522649-1831038044-1853292631-2271478464', 'FullControl', 'None', 'None', 'Allow'))
                } else { @() }
                $owner = if ($isRoot) { 'S-1-5-80-956008885-3418522649-1831038044-1853292631-2271478464' } else { 'S-1-5-32-544' }
                [pscustomobject]@{ Rules = $rules; Owner = $owner } |
                    Add-Member -MemberType ScriptMethod -Name GetOwner -Value { [pscustomobject]@{ Value = $this.Owner } } -PassThru |
                    Add-Member -MemberType ScriptMethod -Name GetAccessRules -Value { $this.Rules } -PassThru
            }
            Test-PcaiAdminOwnedPath -Path $file | Should -Be $Expected
            Test-PcaiSystemTrustedPath -Path $file | Should -Be $Expected
            Should -Invoke Get-Acl -ParameterFilter { $LiteralPath -match '^[A-Za-z]:\\?$' }
        } finally { Remove-Variable -Name PcaiTestRootRights -Scope Global -ErrorAction SilentlyContinue }
    }
}

Describe 'Invoke-PcaiLinkHealth side effects' -Tag 'Unit', 'Portable' {
    BeforeAll {
        $planFile = Join-Path $TestDrive 'plan.json'
        $Plan | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $planFile
    }
    BeforeEach {
        Mock Get-PcaiUsb4Adapter { , @(New-TestAdapter -Ipv4 @()) }
        Mock Get-PcaiIpv4HostRoute { , @() }
        Mock Test-PcaiPeerReachability { , @() }
        Mock Get-PcaiUsb4Service { , @([pscustomobject]@{ Name = '\PC-AI\Configure-ThunderboltSwitchRoutes'; Kind = 'ScheduledTask'; State = 'Ready'; LastResult = 1 }) }
        Mock Invoke-PcaiServiceRecovery { , @() }
        Mock Send-PcaiBusStatus { 'posted' }
        Mock Get-PcaiHostName { 'hosta' }
        Mock Invoke-PcaiBoundedCommand { [pscustomobject]@{ ExitCode = 0; Stdout = ''; Stderr = ''; TimedOut = $false } }
    }
    It 'writes nothing, restarts nothing and posts nothing with -IsDryRun' {
        $root = Join-Path $TestDrive 'dry'
        $status = @(Invoke-PcaiLinkHealth -RunMode Observe -RunTrigger Event -Root $root -Plan $planFile -IsDryRun) | Where-Object { $_ -isnot [string] }
        $status.state | Should -Be 'Drift'
        Test-Path -LiteralPath $root | Should -BeFalse
        Should -Invoke Invoke-PcaiServiceRecovery -Times 0 -Exactly
        Should -Invoke Send-PcaiBusStatus -Times 0 -Exactly
        Should -Invoke Invoke-PcaiBoundedCommand -Times 0 -Exactly
    }
    It 'writes status and a review script but runs no apply in Observe mode' {
        $root = Join-Path $TestDrive 'observe'
        $null = Invoke-PcaiLinkHealth -RunMode Observe -RunTrigger Manual -Root $root -Plan $planFile -SkipBus
        Test-Path -LiteralPath (Join-Path $root 'status.json') | Should -BeTrue
        Test-Path -LiteralPath (Join-Path $root 'pending-apply-0123456789abcdef.ps1') | Should -BeTrue
        Test-Path -LiteralPath (Join-Path $root 'apply') | Should -BeFalse
        Should -Invoke Invoke-PcaiBoundedCommand -Times 0 -Exactly
    }
    It 'waits out the minimum interval and still checks, so a quick reconnect is not dropped' {
        $root = Join-Path $TestDrive 'debounce'
        $null = New-Item -ItemType Directory -Path $root
        Set-Content -LiteralPath (Join-Path $root 'last-run.txt') -Value ([DateTime]::UtcNow.ToString('o')) -NoNewline
        Mock Start-Sleep { }
        $status = @(Invoke-PcaiLinkHealth -RunMode Observe -RunTrigger Event -Root $root -Plan $planFile -MinInterval 20 -SkipBus) | Where-Object { $_ -isnot [string] }
        Should -Invoke Start-Sleep -Times 1 -Exactly -ParameterFilter { $Milliseconds -gt 0 -and $Milliseconds -le 20000 }
        $status.state | Should -Be 'Drift'
        Test-Path -LiteralPath (Join-Path $root 'status.json') | Should -BeTrue
    }
    # Protects: an elevated run never writes into a status root another account can change.
    # Detects: run.lock, last-run.txt, status.json and the review script written through links a
    # user planted in an untrusted root (gemini review of #344 at 45700340 and 2b45b3db). Needs:
    # elevation and the trust check mocked; the root is a TestDrive directory. Breadcrumb: $rootTrusted.
    It 'writes nothing into an untrusted status root when elevated in Apply mode' -Skip:(-not $IsWindows) {
        $root = Join-Path $TestDrive 'untrusted-root'
        $null = New-Item -ItemType Directory -Path $root
        Mock Test-PcaiElevated { $true }
        Mock Test-PcaiAdminOwnedPath { $false }
        $out = @(Invoke-PcaiLinkHealth -RunMode Apply -RunTrigger Manual -Root $root -Plan $planFile -SkipBus)
        ($out | Where-Object { $_ -is [string] }) -join ' ' | Should -Match 'writes nothing there'
        @(Get-ChildItem -LiteralPath $root -Force).Count | Should -Be 0
        # Writing nothing must still surface: the run reports a warning, so the task exits 2, not 0.
        $status = @($out | Where-Object { $_ -isnot [string] }) | Select-Object -Last 1
        $status.severity | Should -Be 'warn'
        @($status.guidance) -join ' ' | Should -Match 'not admin-owned and protected'
        Should -Invoke Invoke-PcaiBoundedCommand -Times 0 -Exactly
    }
    # Protects: a hardened root's existing contents are checked too. Detects: SYSTEM reading
    # last-run.txt or appending to history.jsonl that a user planted before the root was protected
    # (codex review of #344 at c463db6d). Needs: the trust check mocked per path.
    It 'treats a trusted root holding a planted file as untrusted' -Skip:(-not $IsWindows) {
        $root = Join-Path $TestDrive 'planted-root'
        $null = New-Item -ItemType Directory -Path $root
        Set-Content -LiteralPath (Join-Path $root 'history.jsonl') -Value 'planted'
        Mock Test-PcaiElevated { $true }
        Mock Test-PcaiAdminOwnedPath { $Path -notlike '*history.jsonl' }
        $out = @(Invoke-PcaiLinkHealth -RunMode Apply -RunTrigger Manual -Root $root -Plan $planFile -SkipBus)
        ($out | Where-Object { $_ -is [string] }) -join ' ' | Should -Match 'holds entries that are not admin-owned and protected: history.jsonl'
        (Get-Content -LiteralPath (Join-Path $root 'history.jsonl')) | Should -Be 'planted'
        @(Get-ChildItem -LiteralPath $root -Force).Name | Should -Be @('history.jsonl')
        (@($out | Where-Object { $_ -isnot [string] }) | Select-Object -Last 1).severity | Should -Be 'warn'
    }
    It 'reads only the policy''s plan in Apply mode, ignoring a -PlanPath override' {
        Mock Get-PcaiAutoApplyPolicy { [pscustomobject]@{ Path = 'p'; Enabled = $true; PlanPath = 'Z:\admin-only\route-plan.json'; Reason = 'Enabled by admin-owned policy' } }
        Mock Read-PcaiRoutePlan { $null }
        $null = Invoke-PcaiLinkHealth -RunMode Apply -RunTrigger Manual -Root (Join-Path $TestDrive 'apply-mode') -Plan $planFile -SkipBus -IsDryRun
        Should -Invoke Read-PcaiRoutePlan -Times 1 -Exactly -ParameterFilter { $Path -eq 'Z:\admin-only\route-plan.json' }
        Should -Invoke Invoke-PcaiBoundedCommand -Times 0 -Exactly
    }
}

Describe 'Linux reconnect monitor' -Tag 'Unit', 'Portable' {
    It 'starts the link-health unit once after a thunderbolt burst, even while veth churn continues' -Skip:(-not (Get-Command bash -ErrorAction SilentlyContinue)) {
        $bin = Join-Path $TestDrive 'fakebin'
        $null = New-Item -ItemType Directory -Path $bin
        $log = Join-Path $TestDrive 'systemctl.log'
        # Fake `ip`: two events for a thunderbolt interface, then docker veth churn every 0.3 s until
        # the timeout kills it. Churn must neither start the unit nor postpone the start.
        [IO.File]::WriteAllText((Join-Path $bin 'ip'), "#!/usr/bin/env bash`nprintf '7: thunderbolt0: <BROADCAST> state UP\n7: thunderbolt0    inet 172.31.255.21/30\n'`nwhile true; do printf '9: veth12ab@if8: <UP>\n'; sleep 0.3; done`n")
        [IO.File]::WriteAllText((Join-Path $bin 'systemctl'), "#!/usr/bin/env bash`necho `"`$*`" >> '$($log -replace '\\', '/')'`n")
        $script = Join-Path $repoRoot 'Tools/SystemScripts/Networking/pcai-link-health-monitor.sh'
        # Git Bash needs /c/... in PATH; cygpath is absent on Linux, where the path is already POSIX.
        $b = $bin -replace '\\', '/'
        $cmd = "B=`$(cygpath -u '$b' 2>/dev/null || echo '$b'); chmod +x `"`$B`"/*; PATH=`"`$B:`$PATH`" PCAI_LINK_HEALTH_QUIET_SECONDS=1 timeout 12 bash '$($script -replace '\\','/')'; true"
        & bash -c $cmd
        $calls = @(Get-Content -LiteralPath $log -ErrorAction SilentlyContinue)
        $calls | Should -Be @('--user start --no-block pcai-link-health.service')
    }
}

Describe 'SYSTEM install never trusts a pre-existing version directory' -Tag 'Unit' {
    # Protects: a live payload stays intact when a conflicting directory exists.
    # Detects: rebuilding a version that may still be referenced by a SYSTEM task.
    # Needs: Windows and mocked Task Scheduler/ACL providers.
    # Breadcrumb: Install-PcaiLinkHealthTrigger.ps1::Install-PcaiHealthWindows.
    It 'refuses a planted file without changing the version or registering a task' -Skip:(-not $IsWindows) {
        Mock Get-ScheduledTask { }
        Mock Register-ScheduledTask { }
        Mock Start-ScheduledTask { }
        Mock Get-Acl { [pscustomobject]@{} | Add-Member -MemberType ScriptMethod -Name GetOwner -Value { [pscustomobject]@{ Value = 'S-1-5-32-544' } } -PassThru }
        Mock New-PcaiProtectedDirectory { $null = New-Item -ItemType Directory -Force -Path $Path }
        Mock Test-PcaiSystemTrustedPath { $true }
        Mock Protect-PcaiSystemFile { }
        $target = Join-Path $TestDrive 'pcai-usb4/link-health/bin'
        $preview = @(Install-PcaiHealthWindows -Root $repoRoot -Target $target -Components @('LinkHealth') -SystemTask -IsDryRun)
        $bin = ($preview | Where-Object { $_ -like 'would install *' }) -replace '^would install \d+ file\(s\) to ', ''
        $null = New-Item -ItemType Directory -Force -Path $bin
        Set-Content -LiteralPath (Join-Path $bin 'planted.ps1') -Value 'evil'
        { Install-PcaiHealthWindows -Root $repoRoot -Target $target -Components @('LinkHealth') -SystemTask | Out-Null } |
            Should -Throw '*payload*'
        Get-Content -LiteralPath (Join-Path $bin 'planted.ps1') | Should -Be 'evil'
        Should -Invoke Register-ScheduledTask -Times 0 -Exactly
    }

    # Protects: SYSTEM reuse requires trusted custody of the version directory and each published file.
    # Detects: accepting hash-matching content after a directory or script loses trusted ownership or ACLs.
    # Needs: Windows, a real temporary payload, and mocked task and ACL providers.
    # Breadcrumb: Install-PcaiLinkHealthTrigger.ps1::Assert-PcaiHealthPayload.
    It 'refuses reuse when the matching SYSTEM <TargetKind> is untrusted' -Skip:(-not $IsWindows) -ForEach @(
        @{ TargetKind = 'version directory'; Leaf = '' }
        @{ TargetKind = 'payload entry'; Leaf = 'Invoke-PcaiLinkHealth.ps1' }
    ) {
        Mock Get-ScheduledTask {}
        Mock Register-ScheduledTask {}
        Mock Get-Acl { [pscustomobject]@{} | Add-Member -MemberType ScriptMethod -Name GetOwner -Value { [pscustomobject]@{ Value = 'S-1-5-32-544' } } -PassThru }
        Mock New-PcaiProtectedDirectory { $null = New-Item -ItemType Directory -Force -Path $Path }
        Mock Test-PcaiSystemTrustedPath { $true }
        Mock Protect-PcaiSystemFile {}
        $target = Join-Path $TestDrive 'reuse-custody/link-health/bin'
        $null = New-Item -ItemType Directory -Force -Path $target
        $files = @(Get-PcaiHealthFile -Root $repoRoot -Components @('LinkHealth') | Where-Object { Test-Path -LiteralPath $_.Source })
        $payload = Get-PcaiHealthPayload -File $files
        $bin = Join-Path $target $payload.Version
        Publish-PcaiHealthPayload -Root $repoRoot -Directory $bin -Payload $payload
        $paths = @(Get-ChildItem -LiteralPath $bin -File -Force | Sort-Object Name | ForEach-Object FullName)
        $before = @(Get-FileHash -LiteralPath $paths).Hash
        $denied = if ($Leaf) { Join-Path $bin $Leaf } else { $bin }
        Mock Test-PcaiSystemTrustedPath { $false } -ParameterFilter { $Path -eq $denied }
        { Install-PcaiHealthWindows -Root $repoRoot -Target $target -Components @('LinkHealth') -SystemTask | Out-Null } |
            Should -Throw '*Cannot use health payload*'
        @(Get-FileHash -LiteralPath $paths).Hash | Should -Be $before
        Should -Invoke Register-ScheduledTask -Times 0 -Exactly
    }

    # Protects: a write-restricted parent holds the SYSTEM payload while it is staged.
    # Detects: treating permission to create children as safe for a staging parent.
    # Needs: Windows and mocked task and ACL providers.
    # Breadcrumb: Install-PcaiLinkHealthTrigger.ps1::Install-PcaiHealthWindows.
    It 'refuses a SYSTEM staging parent that allows other accounts to create children' -Skip:(-not $IsWindows) {
        Mock Get-ScheduledTask {}
        Mock Register-ScheduledTask {}
        Mock Get-Acl { [pscustomobject]@{} | Add-Member -MemberType ScriptMethod -Name GetOwner -Value { [pscustomobject]@{ Value = 'S-1-5-32-544' } } -PassThru }
        Mock New-PcaiProtectedDirectory { $null = New-Item -ItemType Directory -Force -Path $Path }
        Mock Test-PcaiSystemTrustedPath { $true }
        Mock Test-PcaiSystemTrustedPath { $false } -ParameterFilter { $Path -eq $usb4Root -and -not $AncestorOnly }
        Mock Protect-PcaiSystemFile {}
        $usb4Root = Join-Path $TestDrive 'writable-usb4'
        $null = New-Item -ItemType Directory -Path $usb4Root
        $target = Join-Path $usb4Root 'link-health/bin'
        { Install-PcaiHealthWindows -Root $repoRoot -Target $target -Components @('LinkHealth') -SystemTask } |
            Should -Throw '*staging parent*'
        @(Get-ChildItem -LiteralPath $usb4Root -Force).Count | Should -Be 0
        Should -Invoke Register-ScheduledTask -Times 0 -Exactly
    }

    # Protects: a running SYSTEM observer never sees the partial payload or a premature task update.
    # Detects: staging under link-health, publishing before ACL protection, or registering before publication.
    # Needs: Windows, temporary storage and mocked task and ACL providers.
    # Breadcrumb: Install-PcaiLinkHealthTrigger.ps1::Publish-PcaiHealthPayload.
    It 'protects the complete SYSTEM stage outside link-health before registering its task' -Skip:(-not $IsWindows) {
        Mock Get-ScheduledTask {}
        Mock Register-ScheduledTask {
            @(Get-ChildItem -LiteralPath $target -Directory | Where-Object Name -ne 'backup').Count | Should -Be 1
            [IO.File]::Exists((Join-Path $target "$expectedVersion/manifest.json")) | Should -BeTrue
            Should -Invoke Protect-PcaiSystemFile -Times 1 -Exactly -ParameterFilter { $Directory -like "$usb4Root*stage-*" }
        }
        Mock Get-Acl { [pscustomobject]@{} | Add-Member -MemberType ScriptMethod -Name GetOwner -Value { [pscustomobject]@{ Value = 'S-1-5-32-544' } } -PassThru }
        Mock New-PcaiProtectedDirectory { $null = New-Item -ItemType Directory -Force -Path $Path }
        Mock Test-PcaiSystemTrustedPath { $true }
        Mock Protect-PcaiSystemFile {
            if ((Split-Path -Leaf $Directory) -like '.pcai-health-stage-*') {
                (Split-Path -Parent $Directory) | Should -BeExactly $usb4Root
                @(Get-ChildItem -LiteralPath $Directory -File).Name | Should -Contain 'manifest.json'
                Test-Path -LiteralPath (Join-Path $target $expectedVersion) | Should -BeFalse
            }
        }
        $usb4Root = Join-Path $TestDrive 'protected-usb4'
        $target = Join-Path $usb4Root 'link-health/bin'
        $files = @(Get-PcaiHealthFile -Root $repoRoot -Components @('LinkHealth') | Where-Object { Test-Path -LiteralPath $_.Source })
        $expectedVersion = (Get-PcaiHealthPayload -File $files).Version
        Install-PcaiHealthWindows -Root $repoRoot -Target $target -Components @('LinkHealth') -SystemTask | Out-Null
        Should -Invoke Register-ScheduledTask -Times 1 -Exactly
    }
}

Describe 'Health payload publication' -Tag 'Unit', 'Portable' {
    # Protects: reinstall keeps live bytes and their original provenance.
    # Detects: writing an existing content version while a task can be running.
    # Needs: Windows and mocked scheduled task operations.
    # Breadcrumb: Install-PcaiLinkHealthTrigger.ps1::Install-PcaiHealthWindows.
    It 'reuses a matching installed payload without opening its files for writing' -Skip:(-not $IsWindows) {
        Mock Get-ScheduledTask {}
        Mock Register-ScheduledTask {}
        $target = Join-Path $TestDrive 'repeat-install'
        Install-PcaiHealthWindows -Root $repoRoot -Target $target -Components @('LinkHealth') | Out-Null
        $bin = Get-ChildItem -LiteralPath $target -Directory | Where-Object Name -ne 'backup'
        $payload = Join-Path $bin.FullName 'Invoke-PcaiLinkHealth.ps1'
        $manifest = Join-Path $bin.FullName 'manifest.json'
        $before = Get-Content -LiteralPath $manifest -Raw
        $held = [IO.File]::Open($payload, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::Read)
        try {
            { Install-PcaiHealthWindows -Root $repoRoot -Target $target -Components @('LinkHealth') | Out-Null } |
                Should -Not -Throw
        } finally { $held.Dispose() }
        Get-Content -LiteralPath $manifest -Raw | Should -BeExactly $before
    }

    # Protects: the Linux directory identity describes the bytes systemd will execute.
    # Detects: separate versions for CRLF and LF copies of identical Linux payloads.
    # Needs: PowerShell, owned temporary files and the write-free Linux installer plan.
    # Breadcrumb: Install-PcaiLinkHealthTrigger.ps1::Install-PcaiHealthLinux.
    It 'plans the same Linux version for LF and CRLF source bytes' {
        $root = Join-Path $TestDrive 'newline-source'
        $source = Join-Path $root 'Tools/SystemScripts/Machine/Invoke-PcaiToolingHealth.ps1'
        $null = New-Item -ItemType Directory -Force -Path (Split-Path -Parent $source)
        $target = Join-Path $TestDrive 'newline-target'
        $lf = [string][char]10
        $crlf = [string][char]13 + $lf
        [IO.File]::WriteAllText($source, ('one' + $lf + 'two' + $lf))
        $first = @(Install-PcaiHealthLinux -Root $root -Target $target -Components @('ToolingHealth') -IsDryRun)
        [IO.File]::WriteAllText($source, ('one' + $crlf + 'two' + $crlf))
        $second = @(Install-PcaiHealthLinux -Root $root -Target $target -Components @('ToolingHealth') -IsDryRun)
        $second[0] | Should -BeExactly $first[0]
        Test-Path -LiteralPath $target | Should -BeFalse
    }

    # Protects: executable bytes, manifest integrity and preservation of conflicting payloads.
    # Detects: stale source rereads, lost execute mode, or writing over a damaged version.
    # Needs: Linux PowerShell and a user-owned temporary filesystem with Unix file modes.
    # Breadcrumb: Install-PcaiLinkHealthTrigger.ps1::Publish-PcaiHealthPayload.
    It 'publishes the captured Linux bytes once and preserves a damaged version' -Skip:(-not $IsLinux) {
        $target = Join-Path $TestDrive 'linux-publish'
        $null = New-Item -ItemType Directory -Path $target
        $source = Join-Path $TestDrive 'source.sh'
        [IO.File]::WriteAllText($source, "#!/bin/sh`r`nexit 0`r`n")
        $payload = Get-PcaiHealthPayload -File @([pscustomobject]@{ Name = 'monitor.sh'; Source = $source }) -NormalizeText
        $bin = Join-Path $target $payload.Version
        [IO.File]::WriteAllText($source, "#!/bin/sh`nexit 1`n")
        Publish-PcaiHealthPayload -Root $repoRoot -Directory $bin -Payload $payload
        $installed = Join-Path $bin 'monitor.sh'
        $manifest = Join-Path $bin 'manifest.json'
        [IO.File]::ReadAllText($installed) | Should -BeExactly "#!/bin/sh`nexit 0`n"
        & $installed
        $LASTEXITCODE | Should -Be 0
        $before = Get-Content -LiteralPath $manifest -Raw
        Publish-PcaiHealthPayload -Root $repoRoot -Directory $bin -Payload $payload
        Get-Content -LiteralPath $manifest -Raw | Should -BeExactly $before
        [IO.File]::WriteAllText($installed, 'damaged')
        { Publish-PcaiHealthPayload -Root $repoRoot -Directory $bin -Payload $payload } | Should -Throw '*payload*'
        [IO.File]::ReadAllText($installed) | Should -BeExactly 'damaged'
        @(Get-ChildItem -LiteralPath $target -Force).Name | Should -Be $payload.Version
    }
}

Describe 'SYSTEM install and restore trust the whole ancestor chain' -Tag 'Unit' {
    It 'refuses a SYSTEM install root under a directory the user owns' -Skip:(-not $IsWindows) {
        Mock Get-ScheduledTask { }
        Mock Register-ScheduledTask { }
        Mock New-PcaiProtectedDirectory { $null = New-Item -ItemType Directory -Force -Path $Path }
        # Only the last three levels look admin-owned; TestDrive (a user profile path) does not.
        Mock Get-Acl {
            [pscustomobject]@{} | Add-Member -MemberType ScriptMethod -Name GetOwner -Value { [pscustomobject]@{ Value = 'S-1-5-32-544' } } -PassThru |
                Add-Member -MemberType ScriptMethod -Name GetAccessRules -Value { @() } -PassThru
        } -ParameterFilter { $LiteralPath -like '*pcai-usb4*' }
        Mock Resolve-PcaiSystemPwsh { Join-Path $env:ProgramFiles 'PowerShell/7/pwsh.exe' }
        $target = Join-Path $TestDrive 'staging/pcai-usb4/link-health/bin'
        { Install-PcaiHealthWindows -Root $repoRoot -Target $target -Components @('LinkHealth') -SystemTask } | Should -Throw '*not trusted for a SYSTEM task*'
        Should -Invoke Register-ScheduledTask -Times 0 -Exactly
        # The ancestor check runs before any directory is created or re-permissioned (codex review of #344 at 77bcc55c).
        Should -Invoke New-PcaiProtectedDirectory -Times 0 -Exactly
    }
    # Protects: the SYSTEM task never runs an interpreter a standard user can replace. Detects: the
    # task registered with $PSHOME\pwsh.exe from a per-user or portable install (codex review of
    # #344 at 45700340; dtm-p1gen9's $PSHOME grants Authenticated Users Modify). Needs: a pwsh.exe
    # under TestDrive, which the test user owns. Breadcrumb: Resolve-PcaiSystemPwsh.
    It 'refuses a SYSTEM install when no candidate interpreter is admin-owned and protected' -Skip:(-not $IsWindows) {
        $userPwsh = Join-Path $TestDrive 'portable/pwsh.exe'
        $null = New-Item -ItemType File -Force -Path $userPwsh
        { Resolve-PcaiSystemPwsh -Candidate @($userPwsh, (Join-Path $TestDrive 'missing/pwsh.exe')) } |
            Should -Throw '*No PowerShell 7 is trusted to run as SYSTEM*portable*absent*'
        Mock Get-ScheduledTask { }
        Mock Register-ScheduledTask { }
        Mock Resolve-PcaiSystemPwsh { throw 'No PowerShell 7 is trusted to run as SYSTEM: test' }
        { Install-PcaiHealthWindows -Root $repoRoot -Target (Join-Path $TestDrive 'dry/bin') -Components @('LinkHealth') -SystemTask -IsDryRun } |
            Should -Throw '*No PowerShell 7 is trusted*'
        Should -Invoke Register-ScheduledTask -Times 0 -Exactly
        # A real install refuses before it creates, protects or rebuilds any directory, so an
        # existing install (and a task pointing into it) is left as it was.
        Mock New-PcaiProtectedDirectory { }
        $target = Join-Path $TestDrive 'real/pcai-usb4/link-health/bin'
        { Install-PcaiHealthWindows -Root $repoRoot -Target $target -Components @('LinkHealth') -SystemTask } |
            Should -Throw '*No PowerShell 7 is trusted*'
        Should -Invoke New-PcaiProtectedDirectory -Times 0 -Exactly
        Test-Path -LiteralPath (Join-Path $TestDrive 'real') | Should -BeFalse
    }
    It 'uses the first candidate whose executable and directory are trusted' -Skip:(-not $IsWindows) {
        $first = Join-Path $TestDrive 'a/pwsh.exe'
        $second = Join-Path $TestDrive 'b/pwsh.exe'
        $null = New-Item -ItemType File -Force -Path $first, $second
        Mock Test-PcaiSystemTrustedPath { $Path -like '*\b*' }
        Resolve-PcaiSystemPwsh -Candidate @($first, $second) | Should -Be $second
        Should -Invoke Test-PcaiSystemTrustedPath -ParameterFilter { $Path -eq (Join-Path $TestDrive 'b') }
    }
    It 'refuses a SYSTEM install whose link-health root holds a planted state file' -Skip:(-not $IsWindows) {
        Mock Get-ScheduledTask { }
        Mock Register-ScheduledTask { }
        Mock Resolve-PcaiSystemPwsh { Join-Path $env:ProgramFiles 'PowerShell/7/pwsh.exe' }
        Mock New-PcaiProtectedDirectory { $null = New-Item -ItemType Directory -Force -Path $Path }
        Mock Get-Acl { [pscustomobject]@{} | Add-Member -MemberType ScriptMethod -Name GetOwner -Value { [pscustomobject]@{ Value = 'S-1-5-32-544' } } -PassThru }
        Mock Test-PcaiSystemTrustedPath { $Path -notlike '*last-run.txt' }
        $linkRoot = Join-Path $TestDrive 'planted/pcai-usb4/link-health'
        $null = New-Item -ItemType Directory -Force -Path $linkRoot
        Set-Content -LiteralPath (Join-Path $linkRoot 'last-run.txt') -Value 'planted'
        { Install-PcaiHealthWindows -Root $repoRoot -Target (Join-Path $linkRoot 'bin') -Components @('LinkHealth') -SystemTask } |
            Should -Throw '*holds entries that are not admin-owned and protected*last-run.txt*'
        Should -Invoke Register-ScheduledTask -Times 0 -Exactly
    }
    # Protects: -System never changes ACLs on or deletes through a planted backup junction. Detects:
    # bin\backup excluded from the planted-entry scan, then protected and cleaned (codex review of #344
    # at 4ec06bb1). Needs: a real junction under TestDrive pointing at an outside directory.
    It 'refuses a SYSTEM install whose backup directory is a planted junction' -Skip:(-not $IsWindows) {
        Mock Get-ScheduledTask { }
        Mock Register-ScheduledTask { }
        Mock Resolve-PcaiSystemPwsh { Join-Path $env:ProgramFiles 'PowerShell/7/pwsh.exe' }
        Mock New-PcaiProtectedDirectory { $null = New-Item -ItemType Directory -Force -Path $Path }
        Mock Get-Acl { [pscustomobject]@{} | Add-Member -MemberType ScriptMethod -Name GetOwner -Value { [pscustomobject]@{ Value = 'S-1-5-32-544' } } -PassThru }
        # Everything is trusted except a reparse point, as the real check decides.
        Mock Test-PcaiSystemTrustedPath { -not ((Get-Item -LiteralPath $Path -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) }
        $outside = Join-Path $TestDrive 'outside'
        $null = New-Item -ItemType Directory -Force -Path $outside
        Set-Content -LiteralPath (Join-Path $outside 'victim.txt') -Value 'keep'
        $target = Join-Path $TestDrive 'junction/pcai-usb4/link-health/bin'
        $null = New-Item -ItemType Directory -Force -Path $target
        $null = New-Item -ItemType Junction -Path (Join-Path $target 'backup') -Target $outside
        { Install-PcaiHealthWindows -Root $repoRoot -Target $target -Components @('LinkHealth') -SystemTask } |
            Should -Throw '*holds entries that are not admin-owned and protected*backup*'
        Should -Invoke New-PcaiProtectedDirectory -Times 0 -Exactly -ParameterFilter { $Path -like '*backup' }
        Test-Path -LiteralPath (Join-Path $outside 'victim.txt') | Should -BeTrue
        Should -Invoke Register-ScheduledTask -Times 0 -Exactly
    }
    It 'does not restore a SYSTEM baseline that is not admin-owned and protected' -Skip:(-not $IsWindows) {
        Mock Get-ScheduledTask { [pscustomobject]@{ TaskName = 'LinkHealthSystem' } }
        Mock Unregister-ScheduledTask { }
        Mock Register-ScheduledTask { }
        $target = Join-Path $TestDrive 'restore'
        $null = New-Item -ItemType Directory -Force -Path (Join-Path $target 'backup')
        Set-Content -LiteralPath (Join-Path $target 'backup/LinkHealthSystem.baseline.xml') -Value '<Task>planted</Task>'
        $out = Install-PcaiHealthWindows -Root $repoRoot -Target $target -Components @('LinkHealth') -SystemTask -Remove -RestoreSaved
        ($out -join ' ') | Should -Match 'refusing to restore LinkHealthSystem'
        Should -Invoke Register-ScheduledTask -Times 0 -Exactly
    }
}

Describe 'Uninstall -Restore returns the pre-install definition' -Tag 'Unit' {
    It 'restores the baseline captured by the first install, not a later copy of the health task' -Skip:(-not $IsWindows) {
        # Mock bodies do not share this test's script scope, so the fake task store is global. Get-ScheduledTask
        # binds -TaskName as string[], hence the [string] casts.
        $global:PcaiTestTasks = @{ LinkHealth = '<Task>owner original</Task>' }
        try {
            Mock Get-ScheduledTask { if ($global:PcaiTestTasks.ContainsKey([string]$TaskName)) { [pscustomobject]@{ TaskName = [string]$TaskName } } }
            Mock Export-ScheduledTask { $global:PcaiTestTasks[[string]$TaskName] }
            Mock Register-ScheduledTask { $global:PcaiTestTasks[$TaskName] = $Xml }
            Mock Unregister-ScheduledTask { $global:PcaiTestTasks.Remove([string]$TaskName) }
            Mock Start-ScheduledTask { }
            $target = Join-Path $TestDrive 'health-bin'
            Install-PcaiHealthWindows -Root $repoRoot -Target $target -Components @('LinkHealth') | Out-Null
            Install-PcaiHealthWindows -Root $repoRoot -Target $target -Components @('LinkHealth') | Out-Null
            $global:PcaiTestTasks['LinkHealth'] | Should -Match 'PC-AI USB4 link health'
            Install-PcaiHealthWindows -Root $repoRoot -Target $target -Components @('LinkHealth') -Remove -RestoreSaved | Out-Null
            $global:PcaiTestTasks['LinkHealth'].Trim() | Should -Be '<Task>owner original</Task>'
        } finally { Remove-Variable -Name PcaiTestTasks -Scope Global -ErrorAction SilentlyContinue }
    }
}

Describe 'A health task registered before baselines existed is not a baseline' -Tag 'Unit' {
    It 'records .none when the existing task is this installer''s own' -Skip:(-not $IsWindows) {
        Mock Get-ScheduledTask { [pscustomobject]@{ TaskName = 'LinkHealth'; Description = 'PC-AI USB4 link health (read-only): logon' } }
        Mock Export-ScheduledTask { '<Task>ours</Task>' }
        Mock Register-ScheduledTask { }
        Mock Start-ScheduledTask { }
        $target = Join-Path $TestDrive 'own'
        Install-PcaiHealthWindows -Root $repoRoot -Target $target -Components @('LinkHealth') | Out-Null
        Test-Path -LiteralPath (Join-Path $target 'backup/LinkHealth.baseline.xml') | Should -BeFalse
        Test-Path -LiteralPath (Join-Path $target 'backup/LinkHealth.baseline.none') | Should -BeTrue
    }
}

Describe 'Help switches (AGENTS.md: -h and --help)' -Tag 'Unit', 'Portable' {
    It 'prints the comment-based help for <Flag> on <Script>' -ForEach @(
        @{ Script = 'Invoke-PcaiLinkHealth.ps1'; Flag = '-h' }, @{ Script = 'Invoke-PcaiLinkHealth.ps1'; Flag = '--help' },
        @{ Script = 'Install-PcaiLinkHealthTrigger.ps1'; Flag = '-h' }, @{ Script = 'Install-PcaiLinkHealthTrigger.ps1'; Flag = '--help' }
    ) {
        # The help block is only found when a blank line separates it from #Requires; without
        # it, Get-Help prints just the syntax line.
        $path = Join-Path $repoRoot "Tools/SystemScripts/Networking/$Script"
        $out = & (Join-Path $PSHOME 'pwsh') -NoProfile -NonInteractive -File $path $Flag 2>&1 | Out-String
        $LASTEXITCODE | Should -Be 0
        $out | Should -Match 'SYNOPSIS'
    }
    It 'rejects an unknown argument instead of ignoring it' {
        $path = Join-Path $repoRoot 'Tools/SystemScripts/Networking/Invoke-PcaiLinkHealth.ps1'
        $out = & (Join-Path $PSHOME 'pwsh') -NoProfile -NonInteractive -File $path --bogus 2>&1 | Out-String
        $out | Should -Match 'Unknown arguments: --bogus'
    }
}

Describe 'Service recovery leaves disabled and mutating tasks alone' -Tag 'Unit', 'Portable' {
    It 'restarts only an allow-listed, enabled observer that last failed' {
        $services = @(
            [pscustomobject]@{ Name = '\PC-AI\Configure-ThunderboltSwitchRoutes'; Kind = 'ScheduledTask'; State = 'Ready'; LastResult = 1 },
            [pscustomobject]@{ Name = '\PC-AI\Usb4PeerNetwork'; Kind = 'ScheduledTask'; State = 'Ready'; LastResult = 2 },
            [pscustomobject]@{ Name = '\PC-AI\Other'; Kind = 'ScheduledTask'; State = 'Disabled'; LastResult = 1 })
        $services[0] | Add-Member -NotePropertyName Action -NotePropertyValue 'pwsh.exe -NoProfile -File C:\ProgramData\pcai-usb4\observer\Get-Usb4Observation.ps1 -Observe'
        $d = Get-PcaiServiceRecoveryPlan -Service $services -AllowList @('\PC-AI\Configure-ThunderboltSwitchRoutes', '\PC-AI\Other')
        ($d | Where-Object Name -eq '\PC-AI\Configure-ThunderboltSwitchRoutes').Decision | Should -Be 'Restart'
        ($d | Where-Object Name -eq '\PC-AI\Usb4PeerNetwork').Decision | Should -Be 'ReportOnly'
        ($d | Where-Object Name -eq '\PC-AI\Other').Decision | Should -Be 'LeaveDisabled'
    }
    # Protects: recovery never starts a task that can change the network. Detects: restarting an
    # allow-listed task by name alone when its action runs -Apply or the startup coordinator, which
    # applies the enrolled controller (codex review of #344 at c463db6d). Needs: Windows and synthetic tasks.
    # Breadcrumb: Invoke-PcaiLinkHealth.ps1::Invoke-PcaiServiceRecovery.
    It 'leaves an allow-listed task alone when its action is <Name>' -Skip:(-not $IsWindows) -ForEach @(
        @{ Name = 'an -Apply run'; Action = 'pwsh.exe -File C:\x\Configure-ThunderboltSwitchRoutes.ps1 -Apply' }
        @{ Name = 'the startup coordinator'; Action = 'pwsh.exe -File C:\x\Invoke-ThunderboltStartup.ps1' }
        @{ Name = 'unknown'; Action = '' }
    ) {
        Mock Start-ScheduledTask { }
        $svc = [pscustomobject]@{ Name = '\PC-AI\Configure-ThunderboltSwitchRoutes'; Kind = 'ScheduledTask'; State = 'Ready'; LastResult = 1; Action = $Action }
        $d = Get-PcaiServiceRecoveryPlan -Service @($svc) -AllowList @('\PC-AI\Configure-ThunderboltSwitchRoutes')
        $d[0].Decision | Should -Be 'MutatingAction'
        (Invoke-PcaiServiceRecovery -Decision $d).Count | Should -Be 0
        Should -Invoke Start-ScheduledTask -Times 0 -Exactly
    }
    It 'no longer allow-lists Configure-ThunderboltSwitchRoutes by default' {
        (Read-PcaiLinkHealthConfig -Path (Join-Path $TestDrive 'absent.json') -Root $TestDrive).restartAllowList | Should -Not -Contain '\PC-AI\Configure-ThunderboltSwitchRoutes'
    }
}

Describe 'Notification fingerprint' -Tag 'Unit', 'Portable' {
    It 'ignores timestamps and changes with the state' {
        $base = [ordered]@{ state = 'InSync'; severity = 'ok'; adapters = @(New-TestAdapter); comparison = [pscustomobject]@{ Findings = @() }
            reachability = @(); services = @(); plan = [ordered]@{ planId = 'p' }; evaluatedAtUtc = '2026-11-21T00:00:00Z' }
        $a = Get-PcaiLinkHealthFingerprint -Status ([pscustomobject]$base)
        $base.evaluatedAtUtc = '2026-11-22T00:00:00Z'
        Get-PcaiLinkHealthFingerprint -Status ([pscustomobject]$base) | Should -Be $a
        $base.state = 'Drift'
        Get-PcaiLinkHealthFingerprint -Status ([pscustomobject]$base) | Should -Not -Be $a
    }
}

Describe 'Trigger definitions' -Tag 'Unit', 'Portable' {
    # Protects: event coverage and least privilege for the per-user observer.
    # Detects: missing or mutating triggers, or a privileged task principal.
    # Needs: PowerShell and synthetic task XML. Breadcrumb: Install-PcaiLinkHealthTrigger.ps1::Get-PcaiTaskSpec.
    It 'registers the user task least-privileged with logon, USB4, network, resume and periodic triggers' {
        $spec = Get-PcaiTaskSpec -BinDir $TestDrive -Pwsh 'C:\pwsh.exe' -UserId 'HOST\user' -Components @('LinkHealth')
        $xml = [xml]($spec[0].Xml -replace 'encoding="UTF-16"', '')
        $spec[0].Name | Should -Be 'LinkHealth'
        $xml.Task.Principals.Principal.RunLevel | Should -Be 'LeastPrivilege'
        $xml.Task.Principals.Principal.LogonType | Should -Be 'S4U'
        $xml.Task.Settings.MultipleInstancesPolicy | Should -Be 'IgnoreNew'
        $xml.Task.Triggers.LogonTrigger.UserId | Should -Be 'HOST\user'
        $sub = $xml.Task.Triggers.EventTrigger.Subscription
        foreach ($id in 10000, 10001, 400, 410, 1010) { $sub | Should -Match "EventID=$id\b" }
        $sub | Should -Match 'Power-Troubleshooter'
        $xml.Task.Triggers.TimeTrigger.Repetition.Interval | Should -Be 'PT30M'
        $xml.Task.Triggers.InnerXml | Should -Not -Match 'BootTrigger'
        $xml.Task.Actions.Exec.Arguments | Should -Not -Match '-Mode Apply'
    }
    # Protects: desktop fallback for hosts missing S4U rights.
    # Detects: losing the caller's logon type. Needs: PowerShell and synthetic task XML.
    # Breadcrumb: Install-PcaiLinkHealthTrigger.ps1::Get-PcaiTaskSpec.
    It 'can build the interactive-token fallback for hosts without the batch logon right' {
        $spec = Get-PcaiTaskSpec -BinDir $TestDrive -Pwsh 'C:\pwsh.exe' -UserId 'HOST\user' -Components @('ToolingHealth') -LogonType InteractiveToken
        ([xml]($spec[0].Xml -replace 'encoding="UTF-16"', '')).Task.Principals.Principal.LogonType | Should -Be 'InteractiveToken'
    }
    # Protects: boot-time recovery is scoped to the opt-in SYSTEM principal.
    # Detects: losing the SYSTEM principal, delay or Apply mode. Needs: PowerShell and synthetic task XML.
    # Breadcrumb: Install-PcaiLinkHealthTrigger.ps1::Get-PcaiTaskSpec.
    It 'gives only the owner''s SYSTEM task a boot trigger and Apply mode' {
        $spec = Get-PcaiTaskSpec -BinDir $TestDrive -Pwsh 'C:\pwsh.exe' -UserId 'HOST\user' -Components @('LinkHealth') -SystemTask
        $xml = [xml]($spec[0].Xml -replace 'encoding="UTF-16"', '')
        $spec[0].Name | Should -Be 'LinkHealthSystem'
        $xml.Task.Principals.Principal.UserId | Should -Be 'S-1-5-18'
        $xml.Task.Triggers.BootTrigger.Delay | Should -Be 'PT1M'
        $xml.Task.Actions.Exec.Arguments | Should -Match '-Mode Apply'
    }
    It 'writes systemd user units that treat exit 2 as a finding, not a failure' {
        $units = Get-PcaiSystemdUnit -BinDir '/opt/x' -Pwsh '/usr/bin/pwsh' -Components @('LinkHealth', 'ToolingHealth')
        $units.Keys | Should -Contain 'pcai-link-health-monitor.service'
        $units['pcai-link-health.service'] | Should -Match 'SuccessExitStatus=2'
        $units['pcai-tooling-health.service'] | Should -Match 'Environment=PATH=%h/.local/bin:%h/bin:'
        $units['pcai-link-health.timer'] | Should -Match 'OnUnitActiveSec=30min'
        $units['pcai-tooling-health.timer'] | Should -Match 'OnUnitActiveSec=4h'
    }
    # Protects: new source content selects a new version.
    # Detects: overwriting a version when only file bytes change.
    # Needs: PowerShell and an isolated temporary file.
    # Breadcrumb: Install-PcaiLinkHealthTrigger.ps1::Get-PcaiHealthPayload.
    It 'keys the install directory on file content' {
        $f = Join-Path $TestDrive 'a.ps1'
        'one' | Set-Content -LiteralPath $f
        $files = @([pscustomobject]@{ Name = 'a.ps1'; Source = $f })
        $v1 = (Get-PcaiHealthPayload -File $files).Version
        'two' | Set-Content -LiteralPath $f
        (Get-PcaiHealthPayload -File $files).Version | Should -Not -Be $v1
    }
    It 'keeps the Linux monitor script valid bash' -Skip:(-not (Get-Command bash -ErrorAction SilentlyContinue)) {
        $script = Join-Path $repoRoot 'Tools/SystemScripts/Networking/pcai-link-health-monitor.sh'
        & bash -n $script
        $LASTEXITCODE | Should -Be 0
    }
}

# Protects: a Linux install that could not activate its units fails. Detects: systemctl failures
# (no user manager, a unit that will not start) followed by "enabled" and a successful install
# (codex review of #344 at 77bcc55c). Needs: systemctl replaced by a function that sets the exit
# code, so it runs on any host and touches no real user manager.
Describe 'systemd activation checks every systemctl exit status' -Tag 'Unit', 'Portable' {
    BeforeAll {
        if (-not (Get-Command systemctl -CommandType Function -ErrorAction SilentlyContinue)) {
            function global:systemctl { $global:LASTEXITCODE = 0 }
        }
        $script:Units = @('pcai-link-health.service', 'pcai-link-health.timer', 'pcai-link-health-monitor.service')
    }
    AfterAll { Remove-Item -LiteralPath Function:\systemctl -ErrorAction SilentlyContinue }
    It 'enables the timer and the monitor when every call succeeds, passing every argument' {
        # Each call's full argument list is recorded, so a flag or unit name that failed to reach
        # systemctl fails here (codex review of #344 at 87305579 suspected exactly that).
        $global:PcaiTestCalls = [Collections.Generic.List[string]]::new()
        try {
            Mock systemctl { $global:PcaiTestCalls.Add($args -join ' '); $global:LASTEXITCODE = 0 }
            $out = @(Invoke-PcaiSystemdActivation -Unit $Units -StartNow)
            $out | Should -Be @('enabled pcai-link-health.timer', 'enabled pcai-link-health-monitor.service', 'started pcai-link-health.service')
            @($global:PcaiTestCalls) | Should -Be @(
                '--user daemon-reload'
                '--user enable --now pcai-link-health.timer'
                '--user enable --now pcai-link-health-monitor.service'
                '--user restart pcai-link-health-monitor.service'
                '--user start --no-block pcai-link-health.service')
        } finally { Remove-Variable -Name PcaiTestCalls -Scope Global -ErrorAction SilentlyContinue }
    }
    It 'throws, naming the failed call, when <Name>' -ForEach @(
        @{ Name = 'the user manager is unavailable'; Fail = 'daemon-reload'; Expect = '*daemon-reload exited 1*' }
        @{ Name = 'a unit does not start'; Fail = 'pcai-link-health.timer'; Expect = '*enable --now pcai-link-health.timer exited 1*' }
    ) {
        $global:PcaiTestFail = $Fail
        try {
            Mock systemctl { $global:LASTEXITCODE = if ($args -contains $global:PcaiTestFail) { 1 } else { 0 } }
            { Invoke-PcaiSystemdActivation -Unit $Units } | Should -Throw $Expect
        } finally { Remove-Variable -Name PcaiTestFail -Scope Global -ErrorAction SilentlyContinue }
    }
}

# Protects: files the SYSTEM task runs or restores are owned by Administrators. Detects: scripts,
# manifest and baselines left owned by the installing user, because client Windows makes an elevated
# administrator the owner of what it creates (codex review of #344 at 61707f9e). Needs: Get-Acl and
# Set-Acl mocked, since assigning Administrators as owner needs elevation.
Describe 'SYSTEM task files get Administrators as owner' -Tag 'Unit' {
    BeforeEach {
        $script:dir = Join-Path $TestDrive ([guid]::NewGuid().ToString('n'))
        $null = New-Item -ItemType Directory -Path $dir
        Set-Content -LiteralPath (Join-Path $dir 'Invoke-PcaiLinkHealth.ps1') -Value 'x'
        Set-Content -LiteralPath (Join-Path $dir 'manifest.json') -Value '{}'
        $global:PcaiTestOwners = [Collections.Generic.List[string]]::new()
        Mock Get-Acl { [pscustomobject]@{} | Add-Member -MemberType ScriptMethod -Name SetOwner -Value { param($sid) $global:PcaiTestOwners.Add($sid.Value) } -PassThru }
        Mock Set-Acl { }
    }
    AfterEach { Remove-Variable -Name PcaiTestOwners -Scope Global -ErrorAction SilentlyContinue }
    It 'sets the owner of every file and verifies it' -Skip:(-not $IsWindows) {
        Mock Test-PcaiSystemTrustedPath { $true }
        Protect-PcaiSystemFile -Directory $dir
        @($global:PcaiTestOwners) | Should -Be @('S-1-5-32-544', 'S-1-5-32-544')
        Should -Invoke Set-Acl -Times 2 -Exactly
        Should -Invoke Test-PcaiSystemTrustedPath -Times 2 -Exactly
    }
    It 'throws when a file still fails the trust check' -Skip:(-not $IsWindows) {
        Mock Test-PcaiSystemTrustedPath { $Path -notlike '*manifest.json' }
        { Protect-PcaiSystemFile -Directory $dir } | Should -Throw '*manifest.json is not admin-owned and protected*'
    }
}
