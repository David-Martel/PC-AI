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

Describe 'Plan helpers' -Tag 'Unit', 'Portable' {
    It 'extracts the plan ID from the journal path embedded in the apply lines' {
        Get-PcaiRoutePlanId -Plan $Plan | Should -Be '0123456789abcdef'
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
}

Describe 'Service recovery leaves disabled and mutating tasks alone' -Tag 'Unit', 'Portable' {
    It 'restarts only an allow-listed, enabled observer that last failed' {
        $services = @(
            [pscustomobject]@{ Name = '\PC-AI\Configure-ThunderboltSwitchRoutes'; Kind = 'ScheduledTask'; State = 'Ready'; LastResult = 1 },
            [pscustomobject]@{ Name = '\PC-AI\Usb4PeerNetwork'; Kind = 'ScheduledTask'; State = 'Ready'; LastResult = 2 },
            [pscustomobject]@{ Name = '\PC-AI\Other'; Kind = 'ScheduledTask'; State = 'Disabled'; LastResult = 1 })
        $d = Get-PcaiServiceRecoveryPlan -Service $services -AllowList @('\PC-AI\Configure-ThunderboltSwitchRoutes', '\PC-AI\Other')
        ($d | Where-Object Name -eq '\PC-AI\Configure-ThunderboltSwitchRoutes').Decision | Should -Be 'Restart'
        ($d | Where-Object Name -eq '\PC-AI\Usb4PeerNetwork').Decision | Should -Be 'ReportOnly'
        ($d | Where-Object Name -eq '\PC-AI\Other').Decision | Should -Be 'LeaveDisabled'
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
    It 'registers the user task least-privileged with logon, USB4, network, resume and periodic triggers' {
        $spec = Get-PcaiTaskSpec -BinDir 'C:\bin\v1' -Pwsh 'C:\pwsh.exe' -UserId 'HOST\user' -Components @('LinkHealth')
        $xml = [xml]($spec[0].Xml -replace 'encoding="UTF-16"', '')
        $spec[0].Name | Should -Be 'LinkHealth'
        $xml.Task.Principals.Principal.RunLevel | Should -Be 'LeastPrivilege'
        $xml.Task.Principals.Principal.LogonType | Should -Be 'InteractiveToken'
        $xml.Task.Settings.MultipleInstancesPolicy | Should -Be 'IgnoreNew'
        $xml.Task.Triggers.LogonTrigger.UserId | Should -Be 'HOST\user'
        $sub = $xml.Task.Triggers.EventTrigger.Subscription
        foreach ($id in 10000, 10001, 400, 410, 1010) { $sub | Should -Match "EventID=$id\b" }
        $sub | Should -Match 'Power-Troubleshooter'
        $xml.Task.Triggers.TimeTrigger.Repetition.Interval | Should -Be 'PT30M'
        $xml.Task.Triggers.InnerXml | Should -Not -Match 'BootTrigger'
        $xml.Task.Actions.Exec.Arguments | Should -Not -Match '-Mode Apply'
    }
    It 'gives only the owner''s SYSTEM task a boot trigger and Apply mode' {
        $spec = Get-PcaiTaskSpec -BinDir 'C:\ProgramData\x' -Pwsh 'C:\pwsh.exe' -UserId 'HOST\user' -Components @('LinkHealth') -SystemTask
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
        $units['pcai-link-health.timer'] | Should -Match 'OnUnitActiveSec=30min'
        $units['pcai-tooling-health.timer'] | Should -Match 'OnUnitActiveSec=4h'
    }
    It 'keys the install directory on file content' {
        $f = Join-Path $TestDrive 'a.ps1'
        'one' | Set-Content -LiteralPath $f
        $v1 = Get-PcaiContentVersion -File @([pscustomobject]@{ Name = 'a.ps1'; Source = $f })
        'two' | Set-Content -LiteralPath $f
        Get-PcaiContentVersion -File @([pscustomobject]@{ Name = 'a.ps1'; Source = $f }) | Should -Not -Be $v1
    }
    It 'keeps the Linux monitor script valid bash' -Skip:(-not (Get-Command bash -ErrorAction SilentlyContinue)) {
        $script = Join-Path $repoRoot 'Tools/SystemScripts/Networking/pcai-link-health-monitor.sh'
        & bash -n $script
        $LASTEXITCODE | Should -Be 0
    }
}
