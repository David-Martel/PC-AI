#Requires -Version 7.0
<#
.SYNOPSIS
Install the standalone USB4 peer-network controller without changing networking.
.DESCRIPTION
Plans by default. Apply stages verified immutable revisions, protects the runtime,
registers a bounded SYSTEM task, and retains task/configuration preimages. The task
applies the explicitly enrolled policy later. No machine registry is overwritten.
#>
[CmdletBinding()]
param(
    [string]$PolicyPath,
    [string]$ControllerPath = (Join-Path $PSScriptRoot 'Invoke-Usb4PeerNetwork.ps1'),
    [string]$InstallRoot = (Join-Path ([Environment]::GetFolderPath('CommonApplicationData')) 'pcai-usb4/peer-network'),
    [string]$PwshPath = (Join-Path $PSHOME 'pwsh.exe'),
    [string]$ExpectedControllerSha256,
    [switch]$Apply,
    [switch]$DryRun,
    [switch]$Verify,
    [switch]$ReplaceLegacyTasks,
    [Alias('h', 'help')][switch]$ShowHelp
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$script:Usb4InstallOwner = 'PC-AI standalone USB4 peer-network installer v1'

function Assert-Usb4InstallPath {
    <# .SYNOPSIS
    Reject roots, reserved leaves, control characters and reparse-point paths.
    #>
    param([Parameter(Mandatory)][string]$Path)
    $full = [IO.Path]::GetFullPath($Path)
    if ($full.TrimEnd('\', '/') -eq [IO.Path]::GetPathRoot($full).TrimEnd('\', '/')) { throw 'A filesystem root cannot be an installation directory.' }
    if ($full -match '[\x00-\x1f]') { throw 'Installation paths cannot contain control characters.' }
    foreach ($part in ($full -split '[\\/]')) {
        if ($part -match '^(?i:\$null|AUX|CON|NUL|PRN|COM[1-9]|LPT[1-9])(?:\..*)?[ .]*$') { throw 'Reserved installation path component.' }
    }
    $cursor = $full
    while ($cursor) {
        if (Test-Path -LiteralPath $cursor) {
            $item = Get-Item -LiteralPath $cursor -Force
            if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Installation path traverses a reparse point: $cursor" }
        }
        $parent = [IO.Path]::GetDirectoryName($cursor)
        if ($parent -eq $cursor) { break }
        $cursor = $parent
    }
    $full
}

function ConvertTo-Usb4TaskArgument {
    <# .SYNOPSIS
    Quote one literal Windows process argument without shell expansion.
    #>
    param([Parameter(Mandatory)][AllowEmptyString()][string]$Value)
    if ($Value -match '[\x00-\x1f]') { throw 'Task arguments cannot contain control characters.' }
    '"' + ([regex]::Replace([regex]::Replace($Value, '(\\*)"', '$1$1\"'), '(\\+)$', '$1$1')) + '"'
}

function New-Usb4PeerTaskXml {
    <# .SYNOPSIS
    Build the disabled SYSTEM task with boot, link, resume and timer triggers.
    #>
    param([Parameter(Mandatory)][string]$Executable, [Parameter(Mandatory)][string]$Controller,
        [Parameter(Mandatory)][string]$Policy, [Parameter(Mandatory)][string]$State,
        [Parameter(Mandatory)][string]$Report)
    $escape = { param($value) [Security.SecurityElement]::Escape([string]$value) }
    $arguments = '-NoLogo -NoProfile -NonInteractive -WindowStyle Hidden -File ' +
    (ConvertTo-Usb4TaskArgument $Controller) + ' -PolicyPath ' + (ConvertTo-Usb4TaskArgument $Policy) +
    ' -StatePath ' + (ConvertTo-Usb4TaskArgument $State) + ' -ReportPath ' + (ConvertTo-Usb4TaskArgument $Report) + ' -Apply'
    $subscription = @'
<QueryList><Query Id="0" Path="Microsoft-Windows-NetworkProfile/Operational"><Select Path="Microsoft-Windows-NetworkProfile/Operational">*[System[(EventID=10000 or EventID=10001)]]</Select></Query><Query Id="1" Path="System"><Select Path="System">*[System[Provider[@Name='Microsoft-Windows-Power-Troubleshooter'] and (EventID=1)]]</Select></Query></QueryList>
'@
    $start = [DateTime]::Today.ToString('yyyy-MM-ddTHH:mm:ss')
    @"
<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.4" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo><Description>$(& $escape $script:Usb4InstallOwner)</Description></RegistrationInfo>
  <Triggers>
    <BootTrigger><Enabled>true</Enabled><Delay>PT15S</Delay></BootTrigger>
    <EventTrigger><Enabled>true</Enabled><Subscription>$(& $escape $subscription)</Subscription><Delay>PT3S</Delay></EventTrigger>
    <CalendarTrigger><Repetition><Interval>PT1M</Interval><Duration>P1D</Duration><StopAtDurationEnd>false</StopAtDurationEnd></Repetition><StartBoundary>$start</StartBoundary><Enabled>true</Enabled><ScheduleByDay><DaysInterval>1</DaysInterval></ScheduleByDay></CalendarTrigger>
  </Triggers>
  <Principals><Principal id="System"><UserId>S-1-5-18</UserId><RunLevel>HighestAvailable</RunLevel></Principal></Principals>
  <Settings><MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy><DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries><StopIfGoingOnBatteries>false</StopIfGoingOnBatteries><AllowHardTerminate>true</AllowHardTerminate><StartWhenAvailable>true</StartWhenAvailable><RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable><Enabled>false</Enabled><Hidden>true</Hidden><WakeToRun>false</WakeToRun><ExecutionTimeLimit>PT2M</ExecutionTimeLimit></Settings>
  <Actions Context="System"><Exec><Command>$(& $escape $Executable)</Command><Arguments>$(& $escape $arguments)</Arguments></Exec></Actions>
</Task>
"@
}

function Get-Usb4PeerInstallPlan {
    <# .SYNOPSIS
    Read and validate sources without creating files, tasks or receipts.
    #>
    param([Parameter(Mandatory)][string]$PolicySource, [Parameter(Mandatory)][string]$ControllerSource,
        [Parameter(Mandatory)][string]$Destination, [Parameter(Mandatory)][string]$Executable,
        [string]$ExpectedHash, [string]$CurrentHost = [Environment]::MachineName)
    $root = Assert-Usb4InstallPath $Destination
    foreach ($source in @($PolicySource, $ControllerSource, $Executable)) {
        if (-not [IO.Path]::IsPathFullyQualified($source) -or -not (Test-Path -LiteralPath $source -PathType Leaf)) { throw "Required absolute source file is unavailable: $source" }
    }
    $hash = (Get-FileHash -LiteralPath $ControllerSource -Algorithm SHA256).Hash
    if ($ExpectedHash -and ($ExpectedHash -notmatch '^[A-Fa-f0-9]{64}$' -or $hash -ine $ExpectedHash)) { throw 'Controller SHA-256 does not match the expected reviewed source.' }
    $parseTokens = $null; $parseErrors = $null
    $null = [Management.Automation.Language.Parser]::ParseFile($ControllerSource, [ref]$parseTokens, [ref]$parseErrors)
    if ($parseErrors.Count) { throw 'Controller source does not parse.' }
    $policy = Get-Content -LiteralPath $PolicySource -Raw | ConvertFrom-Json -AsHashtable
    . $ControllerSource
    if (-not (Get-Command Assert-Usb4NetworkPolicy -CommandType Function -ErrorAction SilentlyContinue)) { throw 'Controller policy validator is unavailable.' }
    $null = Assert-Usb4NetworkPolicy -Policy $policy -CurrentHost $CurrentHost
    $revisions = @(if (Test-Path -LiteralPath $root) {
            Get-ChildItem -LiteralPath $root -Directory | Where-Object Name -Match '^r[1-9][0-9]*$' | ForEach-Object { [int]$_.Name.Substring(1) }
        })
    $number = if ($revisions.Count) { [int](($revisions | Measure-Object -Maximum).Maximum) + 1 } else { 1 }
    $revision = 'r' + $number
    $revisionPath = Join-Path $root $revision
    $controller = Join-Path $revisionPath 'Invoke-Usb4PeerNetwork.ps1'
    $policyTarget = Join-Path $revisionPath 'policy.json'
    $state = Join-Path $root 'owned-state.json'; $report = Join-Path $root 'current.json'
    [pscustomobject]@{
        schemaVersion = 1; owner = $script:Usb4InstallOwner; root = $root; revision = $revision
        revisionPath = $revisionPath; stagingPath = (Join-Path $root ('.staging-' + $revision))
        controllerSource = [IO.Path]::GetFullPath($ControllerSource); policySource = [IO.Path]::GetFullPath($PolicySource)
        controllerPath = $controller; policyPath = $policyTarget; statePath = $state; reportPath = $report
        controllerSha256 = $hash; policySha256 = (Get-FileHash -LiteralPath $PolicySource -Algorithm SHA256).Hash
        executable = [IO.Path]::GetFullPath($Executable); taskName = 'Usb4PeerNetwork'; taskPath = '\PC-AI\'
        taskXml = (New-Usb4PeerTaskXml -Executable $Executable -Controller $controller -Policy $policyTarget -State $state -Report $report)
        mode = $policy.mode; hostName = $policy.hostName
    }
}

function Get-Usb4ScheduledTaskSnapshot {
    <# .SYNOPSIS
    Read one scheduler task and preserve its complete XML and enabled state.
    #>
    param([string]$Name, [string]$Path)
    $tasks = @(Get-ScheduledTask -ErrorAction Stop | Where-Object { $_.TaskName -ceq $Name -and $_.TaskPath -ceq $Path })
    if (-not $tasks.Count) { return $null }
    if ($tasks.Count -ne 1) { throw 'Task lookup returned ambiguous results.' }
    $xml = Export-ScheduledTask -TaskName $Name -TaskPath $Path -ErrorAction Stop
    ConvertTo-Usb4ScheduledTaskSnapshot -Task $tasks[0] -Xml $xml
}

function ConvertTo-Usb4ScheduledTaskSnapshot {
    <# .SYNOPSIS
    Preserve the CIM enabled flag when exported XML omits its default value.
    #>
    param([Parameter(Mandatory)][object]$Task, [Parameter(Mandatory)][string]$Xml)
    $enabled = $Task.Settings.Enabled
    if ($enabled -isnot [bool]) { throw 'Scheduled task returned an invalid enabled flag.' }
    $state = [string]$Task.State
    if ($state -notin @('Unknown', 'Disabled', 'Queued', 'Ready', 'Running', '0', '1', '2', '3', '4')) { throw 'Scheduled task returned an invalid execution state.' }
    $principalUser = [string]$Task.Principal.UserId
    $principalLogon = [string]$Task.Principal.LogonType
    $principalLevel = [string]$Task.Principal.RunLevel
    [pscustomobject]@{ Name = [string]$Task.TaskName; Path = [string]$Task.TaskPath; Xml = $Xml; Enabled = $enabled; Running = ($state -notin @('Disabled', 'Ready', '1', '3'))
        SystemPrincipal  = ($principalUser -in @('S-1-5-18', 'SYSTEM', 'NT AUTHORITY\SYSTEM') -and $principalLogon -in @('ServiceAccount', '5'))
        HighestPrincipal = ($principalLevel -in @('Highest', 'HighestAvailable', '1'))
    }
}

function Set-Usb4RuntimeProtection {
    <# .SYNOPSIS
    Restrict the runtime directory to SYSTEM and Administrators only.
    #>
    param([string]$Path, [switch]$Recurse)
    $targets = @(Get-Item -LiteralPath $Path -Force)
    if ($Recurse) { $targets += @(Get-ChildItem -LiteralPath $Path -Recurse -Force) }
    if (@($targets | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint }).Count) { throw 'Refusing ACL changes across a reparse point.' }
    foreach ($target in $targets) {
        $security = if ($target.PSIsContainer) { [Security.AccessControl.DirectorySecurity]::new() } else { [Security.AccessControl.FileSecurity]::new() }
        $security.SetAccessRuleProtection($true, $false)
        $security.SetOwner([Security.Principal.SecurityIdentifier]::new('S-1-5-32-544'))
        $inheritance = if ($target.PSIsContainer) { 'ContainerInherit,ObjectInherit' } else { 'None' }
        foreach ($sidText in @('S-1-5-18', 'S-1-5-32-544')) {
            $sid = [Security.Principal.SecurityIdentifier]::new($sidText)
            $rule = [Security.AccessControl.FileSystemAccessRule]::new($sid, 'FullControl', $inheritance, 'None', 'Allow')
            $security.AddAccessRule($rule)
        }
        Set-Acl -LiteralPath $target.FullName -AclObject $security -ErrorAction Stop
        Assert-Usb4RuntimeProtection -Path $target.FullName
    }
}

function Assert-Usb4RuntimeProtection {
    <# .SYNOPSIS
    Require a protected administrator-owned runtime with no other access grants.
    #>
    param([string]$Path)
    $readback = Get-Acl -LiteralPath $Path -ErrorAction Stop
    if (-not $readback.AreAccessRulesProtected) { throw 'Runtime ACL inheritance remains enabled.' }
    $allowed = @('S-1-5-18', 'S-1-5-32-544')
    if ($readback.GetOwner([Security.Principal.SecurityIdentifier]).Value -notin $allowed) { throw 'Runtime owner is not SYSTEM or Administrators.' }
    $grants = @()
    foreach ($rule in $readback.Access) {
        $identity = $rule.IdentityReference.Translate([Security.Principal.SecurityIdentifier]).Value
        if ($rule.AccessControlType -ne 'Allow' -or $identity -notin $allowed -or
            ($rule.FileSystemRights -band [Security.AccessControl.FileSystemRights]::FullControl) -ne [Security.AccessControl.FileSystemRights]::FullControl) { throw 'Runtime ACL contains an unexpected principal or access grant.' }
        $grants += $identity
    }
    if (@($allowed | Where-Object { $_ -notin $grants }).Count) { throw 'Runtime is missing required SYSTEM/Administrators grants.' }
}

function New-Usb4InstallerOperations {
    <# .SYNOPSIS
    Supply the real operating-system boundaries for installation and rollback.
    #>
    @{
        GetTask         = { param($Name, $Path) Get-Usb4ScheduledTaskSnapshot -Name $Name -Path $Path }
        RegisterTask    = { param($Name, $Path, $Xml) Register-ScheduledTask -TaskName $Name -TaskPath $Path -Xml $Xml -Force -ErrorAction Stop | Out-Null }
        SetEnabled      = { param($Name, $Path, $Enabled)
            if ($Enabled) { Enable-ScheduledTask -TaskName $Name -TaskPath $Path -ErrorAction Stop | Out-Null }
            else { Disable-ScheduledTask -TaskName $Name -TaskPath $Path -ErrorAction Stop | Out-Null }
        }
        RemoveTask      = { param($Name, $Path) Unregister-ScheduledTask -TaskName $Name -TaskPath $Path -Confirm:$false -ErrorAction Stop }
        ProtectRoot     = { param($Path) Set-Usb4RuntimeProtection -Path $Path }
        ProtectRevision = { param($Path) Set-Usb4RuntimeProtection -Path $Path -Recurse }
        ProtectFile     = { param($Path) Set-Usb4RuntimeProtection -Path $Path }
        AssertRoot      = { param($Path) Assert-Usb4RuntimeProtection -Path $Path }
    }
}

function Test-Usb4OwnedTask {
    <# .SYNOPSIS
    Require the installer marker and an action within the owned runtime.
    #>
    param([object]$Snapshot, [string]$Root)
    if ($null -eq $Snapshot) { return $false }
    $document = [xml]$Snapshot.Xml
    $prefix = $Root.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
    $Snapshot.SystemPrincipal -and $Snapshot.HighestPrincipal -and
    ([string]$document.Task.RegistrationInfo.Description -ceq $script:Usb4InstallOwner) -and
    ([string]$document.Task.Actions.Exec.Arguments).Contains($prefix) -and
    ([string]$document.Task.Principals.Principal.UserId -eq 'S-1-5-18')
}

function Assert-Usb4InstalledTask {
    <# .SYNOPSIS
    Verify privilege, action, recovery triggers and bounded scheduler settings.
    #>
    param([object]$Snapshot, [string]$Executable, [string]$Controller, [string]$Policy,
        [string]$State, [string]$Report, [bool]$Enabled)
    if (-not $Snapshot) { throw 'Installed task is missing.' }
    $document = [xml]$Snapshot.Xml
    $expected = [xml](New-Usb4PeerTaskXml -Executable $Executable -Controller $Controller -Policy $Policy -State $State -Report $Report)
    if (-not $Snapshot.SystemPrincipal -or -not $Snapshot.HighestPrincipal -or @($document.Task.Actions.Exec).Count -ne 1 -or
        [string]$document.Task.Actions.Exec.Command -cne $Executable -or
        [string]$document.Task.Actions.Exec.Arguments -cne [string]$expected.Task.Actions.Exec.Arguments -or
        [string]$document.Task.RegistrationInfo.Description -cne $script:Usb4InstallOwner -or
        [string]$document.Task.Principals.Principal.UserId -cne 'S-1-5-18' -or
        [string]$document.Task.Principals.Principal.RunLevel -cne 'HighestAvailable' -or
        (Get-Usb4TaskXmlValue -Parent $document.Task.Settings -Name 'MultipleInstancesPolicy' -DefaultValue 'IgnoreNew') -cne 'IgnoreNew' -or
        [Xml.XmlConvert]::ToTimeSpan([string]$document.Task.Settings.ExecutionTimeLimit) -ne [TimeSpan]::FromMinutes(2) -or
        (Get-Usb4TaskXmlValue -Parent $document.Task.Settings -Name 'WakeToRun' -DefaultValue 'false') -cne 'false' -or
        (Get-Usb4TaskXmlValue -Parent $document.Task.Triggers.BootTrigger -Name 'Enabled' -DefaultValue 'true') -cne 'true' -or
        (Get-Usb4TaskXmlValue -Parent $document.Task.Triggers.EventTrigger -Name 'Enabled' -DefaultValue 'true') -cne 'true' -or
        [string]$document.Task.Triggers.EventTrigger.Subscription -cne [string]$expected.Task.Triggers.EventTrigger.Subscription -or
        (Get-Usb4TaskXmlValue -Parent $document.Task.Triggers.CalendarTrigger -Name 'Enabled' -DefaultValue 'true') -cne 'true' -or
        [Xml.XmlConvert]::ToTimeSpan([string]$document.Task.Triggers.CalendarTrigger.Repetition.Interval) -ne [TimeSpan]::FromMinutes(1) -or
        [Xml.XmlConvert]::ToTimeSpan([string]$document.Task.Triggers.CalendarTrigger.Repetition.Duration) -ne [TimeSpan]::FromDays(1) -or
        $Snapshot.Enabled -ne $Enabled) { throw 'Installed task action, privilege, enabled state or recovery settings differ.' }
}

function Get-Usb4TaskXmlValue {
    <# .SYNOPSIS
    Read task XML while honoring only explicitly supplied schema defaults.
    #>
    param([Parameter(Mandatory)][Xml.XmlNode]$Parent, [Parameter(Mandatory)][string]$Name, [string]$DefaultValue)
    $nodes = @($Parent.ChildNodes | Where-Object LocalName -CEQ $Name)
    if ($nodes.Count -gt 1) { throw "Task XML repeats the $Name setting." }
    if ($nodes.Count -eq 1) { return [string]$nodes[0].InnerText }
    $DefaultValue
}

function Invoke-Usb4PeerInstall {
    <# .SYNOPSIS
    Stage verified source and publish a task with explicit rollback preimages.
    #>
    param([Parameter(Mandatory)][object]$Plan, [hashtable]$Operations = (New-Usb4InstallerOperations),
        [switch]$Apply, [switch]$ReplaceLegacyTasks)
    $existing = & $Operations.GetTask $Plan.taskName $Plan.taskPath
    if ($existing -and -not (Test-Usb4OwnedTask $existing $Plan.root)) { throw 'Refusing to replace a foreign task with the same name.' }
    if ($existing -and $existing.Running) { throw 'The installed controller task is active; retry after its bounded run completes.' }
    $legacy = @()
    if ($ReplaceLegacyTasks) {
        foreach ($identity in @(@{ Name = 'PC-AI-ThunderboltPeerAgent'; Path = '\'; Pattern = 'Register-ThunderboltPeer' },
                @{ Name = 'Configure-ThunderboltSwitchRoutes'; Path = '\PC-AI\'; Pattern = 'Configure-ThunderboltSwitchRoutes|Invoke-ThunderboltStartup' })) {
            $snapshot = & $Operations.GetTask $identity.Name $identity.Path
            if ($snapshot) {
                if ($snapshot.Running) { throw "Legacy controller task is active; retry after its bounded run completes: $($identity.Name)" }
                $legacyDocument = [xml]$snapshot.Xml
                if (@($legacyDocument.Task.Actions.Exec).Count -ne 1 -or
                    [string]$legacyDocument.Task.Actions.Exec.Arguments -notmatch $identity.Pattern -or
                    [IO.Path]::GetFileName([string]$legacyDocument.Task.Actions.Exec.Command) -notmatch '^(?i:pwsh|powershell)(?:\.exe)?$') { throw "Legacy task action does not match the recognized network helper: $($identity.Name)" }
                $legacy += $snapshot
            }
        }
    }
    if (Test-Path -LiteralPath $Plan.root) {
        $children = @(Get-ChildItem -LiteralPath $Plan.root -Force)
        if ($children.Count) {
            $unexpected = @($children | Where-Object {
                    if ($_.PSIsContainer) { $_.Name -notmatch '^(?:r[1-9][0-9]*|\.staging-r[1-9][0-9]*)$' }
                    else { $_.Name -notmatch '^(?:installed\.json(?:\.pending)?|owned-state\.json(?:\.(?:pending|tmp))?|current\.json(?:\.(?:pending|tmp))?)$' }
                })
            if ($unexpected.Count) { throw 'Existing nonempty runtime contains unowned paths; refusing ACL or file changes.' }
            $markers = @($children | Where-Object PSIsContainer | ForEach-Object { Join-Path $_.FullName 'manifest.json' })
            $markers += (Join-Path $Plan.root 'installed.json')
            $owned = $false
            foreach ($marker in $markers) {
                if (Test-Path -LiteralPath $marker -PathType Leaf) {
                    $value = Get-Content -LiteralPath $marker -Raw | ConvertFrom-Json
                    if ($value.owner -ceq $script:Usb4InstallOwner) { $owned = $true }
                }
            }
            if (-not $owned) { throw 'Existing nonempty runtime has no owned installation manifest.' }
            & $Operations.AssertRoot $Plan.root | Out-Null
        }
    }
    if (-not $Apply) { return [pscustomobject]@{ State = 'Planned'; Applied = $false; RuntimeQualified = $false; Plan = $Plan; LegacyTasks = @($legacy | ForEach-Object Name) } }
    if ((Get-FileHash -LiteralPath $Plan.controllerSource -Algorithm SHA256).Hash -cne $Plan.controllerSha256 -or
        (Get-FileHash -LiteralPath $Plan.policySource -Algorithm SHA256).Hash -cne $Plan.policySha256) { throw 'Source changed after planning; no installation performed.' }
    if (Test-Path -LiteralPath $Plan.stagingPath) { throw 'A prior staging directory exists; preserve and reconcile it before retry.' }
    if (Test-Path -LiteralPath $Plan.revisionPath) { throw 'The planned revision already exists; replan without overwriting it.' }
    $pointerPath = Join-Path $Plan.root 'installed.json'
    $oldPointer = if (Test-Path -LiteralPath $pointerPath -PathType Leaf) { [IO.File]::ReadAllBytes($pointerPath) } else { $null }
    $taskAttempted = $false; $pointerWritten = $false; $legacyAttempted = @(); $revisionCreated = $false
    try {
        [IO.Directory]::CreateDirectory($Plan.root) | Out-Null
        & $Operations.ProtectRoot $Plan.root | Out-Null
        [IO.Directory]::CreateDirectory($Plan.stagingPath) | Out-Null
        Copy-Item -LiteralPath $Plan.controllerSource -Destination (Join-Path $Plan.stagingPath 'Invoke-Usb4PeerNetwork.ps1') -ErrorAction Stop
        Copy-Item -LiteralPath $Plan.policySource -Destination (Join-Path $Plan.stagingPath 'policy.json') -ErrorAction Stop
        foreach ($entry in @(@{ Name = 'Invoke-Usb4PeerNetwork.ps1'; Hash = $Plan.controllerSha256 }, @{ Name = 'policy.json'; Hash = $Plan.policySha256 })) {
            if ((Get-FileHash -LiteralPath (Join-Path $Plan.stagingPath $entry.Name) -Algorithm SHA256).Hash -cne $entry.Hash) { throw 'Staged file readback failed.' }
        }
        $preimages = Join-Path $Plan.stagingPath 'preimages'; [IO.Directory]::CreateDirectory($preimages) | Out-Null
        $preimageManifest = @()
        $allSnapshots = @($legacy); if ($existing) { $allSnapshots += $existing }
        foreach ($snapshot in $allSnapshots) {
            $leaf = $snapshot.Name + '.xml'
            [IO.File]::WriteAllText((Join-Path $preimages $leaf), $snapshot.Xml, [Text.UTF8Encoding]::new($false))
            $preimageManifest += @{ name = $snapshot.Name; path = $snapshot.Path; enabled = $snapshot.Enabled; file = $leaf }
        }
        if ($null -ne $oldPointer) { [IO.File]::WriteAllBytes((Join-Path $preimages 'installed.json'), $oldPointer) }
        $manifest = @{ schemaVersion = 1; owner = $script:Usb4InstallOwner; revision = $Plan.revision; hostName = $Plan.hostName
            controllerSha256 = $Plan.controllerSha256; policySha256 = $Plan.policySha256; controllerPath = $Plan.controllerPath; policyPath = $Plan.policyPath
            statePath = $Plan.statePath; reportPath = $Plan.reportPath
            taskName = $Plan.taskName; taskPath = $Plan.taskPath; executable = $Plan.executable; installedAtUtc = [DateTimeOffset]::UtcNow.ToString('o')
            preimages = $preimageManifest; runtimeQualified = $false
        }
        $manifestText = $manifest | ConvertTo-Json -Depth 10
        [IO.File]::WriteAllText((Join-Path $Plan.stagingPath 'manifest.json'), $manifestText, [Text.UTF8Encoding]::new($false))
        & $Operations.ProtectRevision $Plan.stagingPath | Out-Null
        [IO.Directory]::Move($Plan.stagingPath, $Plan.revisionPath); $revisionCreated = $true
        $freshTask = & $Operations.GetTask $Plan.taskName $Plan.taskPath
        if (($null -eq $existing) -ne ($null -eq $freshTask) -or
            ($existing -and ($freshTask.Xml -cne $existing.Xml -or $freshTask.Enabled -ne $existing.Enabled -or $freshTask.Running))) { throw 'Scheduled task changed after planning; refusing force replacement.' }
        $taskAttempted = $true
        & $Operations.RegisterTask $Plan.taskName $Plan.taskPath $Plan.taskXml | Out-Null
        $check = & $Operations.GetTask $Plan.taskName $Plan.taskPath
        Assert-Usb4InstalledTask -Snapshot $check -Executable $Plan.executable -Controller $Plan.controllerPath -Policy $Plan.policyPath -State $Plan.statePath -Report $Plan.reportPath -Enabled $false
        foreach ($snapshot in $legacy) {
            $freshLegacy = & $Operations.GetTask $snapshot.Name $snapshot.Path
            if (-not $freshLegacy -or $freshLegacy.Xml -cne $snapshot.Xml -or $freshLegacy.Enabled -ne $snapshot.Enabled -or $freshLegacy.Running) { throw "Legacy task changed or started after planning: $($snapshot.Name)" }
            $legacyAttempted += $snapshot
            & $Operations.SetEnabled $snapshot.Name $snapshot.Path $false | Out-Null
            $check = & $Operations.GetTask $snapshot.Name $snapshot.Path
            if ($null -eq $check -or $check.Enabled -or $check.Running) { throw "Legacy task was not disabled and quiescent: $($snapshot.Name)" }
        }
        [IO.File]::WriteAllText(($pointerPath + '.pending'), $manifestText, [Text.UTF8Encoding]::new($false))
        [IO.File]::Move(($pointerPath + '.pending'), $pointerPath, $true); $pointerWritten = $true
        & $Operations.ProtectFile $pointerPath | Out-Null
        & $Operations.SetEnabled $Plan.taskName $Plan.taskPath $true | Out-Null
        $check = & $Operations.GetTask $Plan.taskName $Plan.taskPath
        Assert-Usb4InstalledTask -Snapshot $check -Executable $Plan.executable -Controller $Plan.controllerPath -Policy $Plan.policyPath -State $Plan.statePath -Report $Plan.reportPath -Enabled $true
        [pscustomobject]@{ State = 'Installed'; Applied = $true; RuntimeQualified = $false; Revision = $Plan.revision
            ControllerPath = $Plan.controllerPath; PolicyPath = $Plan.policyPath; StatePath = $Plan.statePath; ReportPath = $Plan.reportPath
            ControllerSha256 = $Plan.controllerSha256; PolicySha256 = $Plan.policySha256; Preimages = (Join-Path $Plan.revisionPath 'preimages')
        }
    }
    catch {
        $failure = $_.Exception.Message; $rollbackErrors = @(); $taskRollbackComplete = $true
        if ($taskAttempted) {
            try {
                $current = & $Operations.GetTask $Plan.taskName $Plan.taskPath
                if ($current -and -not (Test-Usb4OwnedTask $current $Plan.root)) { throw 'Task ownership changed; refusing rollback overwrite.' }
                if ($current -and $current.Running) { throw 'Controller task is active; preserve its task and do not restart legacy reconcilers during rollback.' }
                if ($existing) {
                    & $Operations.RegisterTask $existing.Name $existing.Path $existing.Xml | Out-Null
                    & $Operations.SetEnabled $existing.Name $existing.Path $existing.Enabled | Out-Null
                    $restored = & $Operations.GetTask $existing.Name $existing.Path
                    if (-not $restored -or $restored.Enabled -ne $existing.Enabled -or
                        ([xml]$restored.Xml).Task.Actions.InnerXml -cne ([xml]$existing.Xml).Task.Actions.InnerXml) { throw 'Previous task rollback readback failed.' }
                }
                elseif ($current) {
                    & $Operations.RemoveTask $Plan.taskName $Plan.taskPath | Out-Null
                    if ($null -ne (& $Operations.GetTask $Plan.taskName $Plan.taskPath)) { throw 'New task rollback removal readback failed.' }
                }
            }
            catch { $rollbackErrors += $_.Exception.Message; $taskRollbackComplete = $false }
        }
        foreach ($snapshot in $legacyAttempted) {
            if (-not $taskRollbackComplete) { $rollbackErrors += 'Legacy restoration held until new task custody is reconciled.'; break }
            try {
                $current = & $Operations.GetTask $snapshot.Name $snapshot.Path
                if (-not $current -or ([xml]$current.Xml).Task.Actions.InnerXml -cne ([xml]$snapshot.Xml).Task.Actions.InnerXml) { throw "Legacy task changed; refusing rollback overwrite: $($snapshot.Name)" }
                & $Operations.SetEnabled $snapshot.Name $snapshot.Path $snapshot.Enabled | Out-Null
                $restored = & $Operations.GetTask $snapshot.Name $snapshot.Path
                if (-not $restored -or $restored.Enabled -ne $snapshot.Enabled) { throw "Legacy task rollback readback failed: $($snapshot.Name)" }
            }
            catch { $rollbackErrors += $_.Exception.Message }
        }
        if ($pointerWritten -and $taskRollbackComplete) {
            try {
                $currentPointer = Get-Content -LiteralPath $pointerPath -Raw | ConvertFrom-Json
                if ($currentPointer.revision -cne $Plan.revision -or $currentPointer.controllerSha256 -cne $Plan.controllerSha256) { throw 'Install pointer ownership changed; refusing rollback overwrite.' }
                if ($null -ne $oldPointer) { [IO.File]::WriteAllBytes($pointerPath, $oldPointer) }
                else { Remove-Item -LiteralPath $pointerPath -ErrorAction Stop }
            }
            catch { $rollbackErrors += $_.Exception.Message }
        }
        $retained = if ($revisionCreated) { $Plan.revisionPath } else { $Plan.stagingPath }
        throw "Installation failed: $failure. Preserved artifacts: $retained. Rollback errors: $($rollbackErrors -join '; ')"
    }
}

function Test-Usb4PeerInstallation {
    <# .SYNOPSIS
    Verify installed bytes and scheduler identity without changing the machine.
    #>
    param([string]$Root, [hashtable]$Operations = (New-Usb4InstallerOperations))
    $rootPath = Assert-Usb4InstallPath $Root
    & $Operations.AssertRoot $rootPath | Out-Null
    $manifest = Get-Content -LiteralPath (Join-Path $rootPath 'installed.json') -Raw | ConvertFrom-Json
    if ($manifest.owner -cne $script:Usb4InstallOwner -or $manifest.revision -notmatch '^r[1-9][0-9]*$') { throw 'Install manifest ownership or revision is invalid.' }
    foreach ($entry in @(@{ Path = $manifest.controllerPath; Hash = $manifest.controllerSha256 }, @{ Path = $manifest.policyPath; Hash = $manifest.policySha256 })) {
        if (-not ([IO.Path]::GetFullPath($entry.Path)).StartsWith((Join-Path $rootPath $manifest.revision) + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) { throw 'Install manifest path escapes its revision.' }
        if ((Get-FileHash -LiteralPath $entry.Path -Algorithm SHA256).Hash -cne $entry.Hash) { throw 'Installed file hash mismatch.' }
        & $Operations.AssertRoot $entry.Path | Out-Null
    }
    $task = & $Operations.GetTask $manifest.taskName $manifest.taskPath
    Assert-Usb4InstalledTask -Snapshot $task -Executable $manifest.executable -Controller $manifest.controllerPath -Policy $manifest.policyPath -State $manifest.statePath -Report $manifest.reportPath -Enabled $true
    [pscustomobject]@{ State = 'Verified'; Installed = $true; RuntimeQualified = $false; Revision = $manifest.revision }
}

if ($MyInvocation.InvocationName -eq '.') { return }
if ($ShowHelp) { Get-Help $PSCommandPath -Detailed; return }
if ($Apply -and $DryRun) { throw 'Apply and DryRun cannot be combined.' }
if (-not $IsWindows) { throw 'The installer runtime requires Windows; portable planning helpers can be dot-sourced.' }
if ($Verify) { Test-Usb4PeerInstallation -Root $InstallRoot | ConvertTo-Json -Depth 12; return }
if (-not $PolicyPath) { throw 'PolicyPath is required.' }
$plan = Get-Usb4PeerInstallPlan -PolicySource $PolicyPath -ControllerSource $ControllerPath -Destination $InstallRoot -Executable $PwshPath -ExpectedHash $ExpectedControllerSha256
if (-not $Apply) { Invoke-Usb4PeerInstall -Plan $plan -ReplaceLegacyTasks:$ReplaceLegacyTasks | ConvertTo-Json -Depth 15; return }
$principal = [Security.Principal.WindowsPrincipal]::new([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) { throw 'Installation requires an elevated administrator.' }
$mutex = [Threading.Mutex]::new($false, 'Global\PcaiUsb4PeerNetworkInstall')
$held = $false
try {
    $held = $mutex.WaitOne(0)
    if (-not $held) { throw 'Another USB4 installation is active.' }
    $plan = Get-Usb4PeerInstallPlan -PolicySource $PolicyPath -ControllerSource $ControllerPath -Destination $InstallRoot -Executable $PwshPath -ExpectedHash $ExpectedControllerSha256
    Invoke-Usb4PeerInstall -Plan $plan -Apply -ReplaceLegacyTasks:$ReplaceLegacyTasks | ConvertTo-Json -Depth 15
}
finally {
    if ($held) { $mutex.ReleaseMutex() }
    $mutex.Dispose()
}
