#Requires -Version 7.0
[CmdletBinding()]
param([string]$SourcePath = (Join-Path $PSScriptRoot '../../Tools/SystemScripts/Networking/Install-Usb4PeerNetwork.ps1'))
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. $SourcePath

function Assert-InstallCondition {
    param([bool]$Condition, [string]$Message)
    if (-not $Condition) { throw $Message }
}
function Assert-InstallThrows {
    param([scriptblock]$Operation, [string]$Pattern)
    $caught = $null
    try { & $Operation | Out-Null } catch { $caught = $_.Exception.Message }
    if (-not $caught -or $caught -notmatch $Pattern) { throw "Expected refusal /$Pattern/; observed '$caught'." }
}
function New-TaskFixture {
    $script:TaskStore = @{}
    $script:TaskCalls = [Collections.Generic.List[string]]::new()
    $script:FailNewEnable = $false
    $script:IgnoreRemoval = $false
    @{
        GetTask         = { param($Name, $Path)
            $key = $Path + $Name
            if ($script:TaskStore.ContainsKey($key)) {
                $item = $script:TaskStore[$key]
                [pscustomobject]@{ Name = $item.Name; Path = $item.Path; Xml = $item.Xml; Enabled = $item.Enabled; Running = $(if ($item.PSObject.Properties['Running']) { $item.Running } else { $false })
                    SystemPrincipal  = $(if ($item.PSObject.Properties['SystemPrincipal']) { $item.SystemPrincipal } else { $true })
                    HighestPrincipal = $(if ($item.PSObject.Properties['HighestPrincipal']) { $item.HighestPrincipal } else { $true })
                }
            }
            else { $null }
        }
        RegisterTask    = { param($Name, $Path, $Xml)
            $script:TaskCalls.Add('register:' + $Path + $Name)
            $document = [xml]$Xml
            $script:TaskStore[$Path + $Name] = [pscustomobject]@{ Name = $Name; Path = $Path; Xml = $Xml; Enabled = ([string]$document.Task.Settings.Enabled -ne 'false') }
        }
        SetEnabled      = { param($Name, $Path, $Enabled)
            $script:TaskCalls.Add('enabled:' + $Path + $Name + ':' + $Enabled)
            $item = $script:TaskStore[$Path + $Name]
            $document = [xml]$item.Xml
            $document.Task.Settings.Enabled = ([string][bool]$Enabled).ToLowerInvariant()
            $item.Xml = $document.OuterXml; $item.Enabled = [bool]$Enabled
            if ($script:FailNewEnable -and $Name -eq 'Usb4PeerNetwork' -and $Enabled) { throw 'injected enable failure after mutation' }
        }
        RemoveTask      = { param($Name, $Path)
            $script:TaskCalls.Add('remove:' + $Path + $Name)
            if (-not $script:IgnoreRemoval) { $script:TaskStore.Remove($Path + $Name) }
        }
        ProtectRoot     = { param($Path) $script:TaskCalls.Add('protect:' + $Path) }
        ProtectRevision = { param($Path) $script:TaskCalls.Add('protect-revision:' + $Path) }
        ProtectFile     = { param($Path) $script:TaskCalls.Add('protect-file:' + $Path) }
        AssertRoot      = { param($Path) $script:TaskCalls.Add('verify-acl:' + $Path) }
    }
}
function New-LegacyTaskFixture {
    param([string]$Name = 'PC-AI-ThunderboltPeerAgent', [string]$Path = '\', [string]$Action = 'Register-ThunderboltPeer -Apply')
    $arguments = [Security.SecurityElement]::Escape($Action)
    $xml = '<Task><RegistrationInfo><Description>legacy fixture</Description></RegistrationInfo><Settings><Enabled>true</Enabled></Settings><Actions><Exec><Command>pwsh.exe</Command><Arguments>' + $arguments + '</Arguments></Exec></Actions></Task>'
    $snapshot = [pscustomobject]@{ Name = $Name; Path = $Path; Xml = $xml; Enabled = $true; Running = $false }
    $script:TaskStore[$Path + $Name] = $snapshot
    $snapshot
}

$workspace = Join-Path ([IO.Path]::GetTempPath()) ('pcai-usb4-installer-' + [guid]::NewGuid().ToString('N'))
[IO.Directory]::CreateDirectory($workspace) | Out-Null
try {
    # This inert source is the installer input, not a replacement for installer
    # logic. Controller policy semantics have their own real controller controls.
    $controller = Join-Path $workspace 'reviewed-controller.ps1'
    $controllerText = @'
function Assert-Usb4NetworkPolicy {
    param([hashtable]$Policy, [string]$CurrentHost)
    if ($Policy.schemaVersion -ne 1 -or $Policy.hostName -cne $CurrentHost) { throw 'fixture policy host refused' }
    $Policy
}
'@
    [IO.File]::WriteAllText($controller, $controllerText)
    $policy = Join-Path $workspace 'approved-policy.json'
    [IO.File]::WriteAllText($policy, (@{ schemaVersion = 1; hostName = 'fixture-host'; mode = 'Peer' } | ConvertTo-Json))
    $executable = (Get-Command pwsh -ErrorAction Stop).Source
    $destination = Join-Path $workspace 'runtime & escaped'
    $plan = Get-Usb4PeerInstallPlan -PolicySource $policy -ControllerSource $controller -Destination $destination -Executable $executable -CurrentHost 'fixture-host'

    # Selector: SchedulerCimShapeRetainsDefaultEnabled. Real Task Scheduler
    # exposes Settings.Enabled and can omit Enabled entirely from exported XML.
    $cimTask = [pscustomobject]@{ TaskName = 'legacy'; TaskPath = '\'; State = 'Ready'; Settings = [pscustomobject]@{ Enabled = $true }
        Principal = [pscustomobject]@{ UserId = 'SYSTEM'; LogonType = 'ServiceAccount'; RunLevel = 'Highest' }
    }
    $defaultXml = '<Task><Settings><Hidden>true</Hidden></Settings></Task>'
    $snapshot = ConvertTo-Usb4ScheduledTaskSnapshot -Task $cimTask -Xml $defaultXml
    Assert-InstallCondition ($snapshot.Enabled -and $snapshot.Xml -ceq $defaultXml) 'Missing XML default changed the CIM enabled state.'
    Assert-InstallCondition ($snapshot.SystemPrincipal -and $snapshot.HighestPrincipal) 'Actual SYSTEM service-account principal was not retained.'
    $cimTask.Principal.LogonType = 'Interactive'
    $snapshot = ConvertTo-Usb4ScheduledTaskSnapshot -Task $cimTask -Xml $defaultXml
    Assert-InstallCondition (-not $snapshot.SystemPrincipal) 'Interactive principal was accepted as a SYSTEM service account.'
    $cimTask.Principal.LogonType = 'ServiceAccount'
    $cimTask.Settings.Enabled = $false
    $snapshot = ConvertTo-Usb4ScheduledTaskSnapshot -Task $cimTask -Xml $defaultXml
    Assert-InstallCondition (-not $snapshot.Enabled) 'Disabled task became enabled while exporting its preimage.'
    $cimTask.State = 'Running'
    $snapshot = ConvertTo-Usb4ScheduledTaskSnapshot -Task $cimTask -Xml $defaultXml
    Assert-InstallCondition $snapshot.Running 'Running task was treated as quiescent.'
    $cimTask.Settings.Enabled = $null
    Assert-InstallThrows { ConvertTo-Usb4ScheduledTaskSnapshot -Task $cimTask -Xml $defaultXml } 'invalid enabled flag'
    Write-Output 'PASS: SchedulerCimShapeRetainsDefaultEnabled'

    # Selector: PurePlanAndExplicitTaskContract. Detects unsafe startup identity,
    # quoting, missing hotplug/resume/fallback, wake behavior and planning writes.
    Assert-InstallCondition (-not (Test-Path -LiteralPath $destination)) 'Planning created a runtime directory.'
    $xml = [xml]$plan.taskXml
    Assert-InstallCondition ($xml.Task.Principals.Principal.UserId -eq 'S-1-5-18' -and $xml.Task.Principals.Principal.RunLevel -eq 'HighestAvailable') 'Task privilege contract is wrong.'
    Assert-InstallCondition ($null -eq $xml.Task.Principals.Principal.SelectSingleNode("*[local-name()='LogonType']")) 'The API ServiceAccount enum leaked into invalid task XML.'
    Assert-InstallCondition ($xml.Task.Settings.MultipleInstancesPolicy -eq 'IgnoreNew' -and $xml.Task.Settings.ExecutionTimeLimit -eq 'PT2M') 'Task runtime/overlap is unbounded.'
    Assert-InstallCondition ($xml.Task.Settings.WakeToRun -eq 'false' -and $xml.Task.Settings.Enabled -eq 'false') 'Task starts enabled or wakes the machine.'
    Assert-InstallCondition ($xml.Task.Triggers.CalendarTrigger.Repetition.Interval -eq 'PT1M' -and $null -ne $xml.Task.Triggers.BootTrigger) 'Boot or one-minute recovery trigger missing.'
    Assert-InstallCondition ($xml.Task.Triggers.EventTrigger.Subscription -match 'EventID=10000' -and $xml.Task.Triggers.EventTrigger.Subscription -match 'EventID=10001' -and $xml.Task.Triggers.EventTrigger.Subscription -match 'Power-Troubleshooter') 'Link/resume event recovery missing.'
    Assert-InstallCondition ($xml.Task.Actions.Exec.Command -ceq $executable -and $xml.Task.Actions.Exec.Arguments.Contains($plan.controllerPath)) 'Escaped absolute action did not roundtrip XML.'
    Assert-InstallCondition ($xml.Task.Actions.Exec.Arguments -match '-NoProfile -NonInteractive -WindowStyle Hidden' -and $xml.Task.Actions.Exec.Arguments.EndsWith(' -Apply')) 'Task action is profile-dependent or not applying policy.'
    Assert-InstallCondition ($plan.revision -ceq 'r1' -and $plan.taskXml -notmatch 'vigil|pcai-tb|agent-bus|Register-ThunderboltPeer') 'Standalone task depends on a lab/native discovery service.'
    $operations = New-TaskFixture
    $legacy = New-LegacyTaskFixture
    $result = Invoke-Usb4PeerInstall -Plan $plan -Operations $operations -ReplaceLegacyTasks
    Assert-InstallCondition ($result.State -eq 'Planned' -and -not $result.Applied -and $script:TaskCalls.Count -eq 0 -and -not (Test-Path -LiteralPath $destination)) 'DryRun changed tasks, ACLs or files.'
    Write-Output 'PASS: PurePlanAndExplicitTaskContract'

    # Selector: SourceAndOwnershipRefusalsPreserveOldTask. Detects changed bytes,
    # malformed source, hash mismatch and foreign-task overwrite before mutation.
    Assert-InstallThrows { Get-Usb4PeerInstallPlan -PolicySource $policy -ControllerSource $controller -Destination $destination -Executable $executable -CurrentHost 'fixture-host' -ExpectedHash ('0' * 64) } 'SHA-256'
    Assert-InstallThrows { Get-Usb4PeerInstallPlan -PolicySource $policy -ControllerSource $controller -Destination $destination -Executable $executable -CurrentHost 'other-host' } 'host refused'
    [IO.File]::AppendAllText($controller, "`n# source changed after plan")
    Assert-InstallThrows { Invoke-Usb4PeerInstall -Plan $plan -Operations $operations -Apply -ReplaceLegacyTasks } 'Source changed'
    Assert-InstallCondition ($script:TaskCalls.Count -eq 0 -and $script:TaskStore['\PC-AI-ThunderboltPeerAgent'].Enabled -and -not (Test-Path -LiteralPath $destination)) 'Source refusal changed legacy task or filesystem.'
    [IO.File]::WriteAllText($controller, $controllerText)
    $script:TaskStore['\PC-AI\Usb4PeerNetwork'] = [pscustomobject]@{ Name = 'Usb4PeerNetwork'; Path = '\PC-AI\'; Xml = '<Task><RegistrationInfo><Description>foreign</Description></RegistrationInfo><Settings><Enabled>true</Enabled></Settings><Actions><Exec><Arguments>foreign</Arguments></Exec></Actions><Principals><Principal><UserId>S-1-5-18</UserId></Principal></Principals></Task>'; Enabled = $true }
    Assert-InstallThrows { Invoke-Usb4PeerInstall -Plan $plan -Operations $operations -Apply } 'foreign task'
    Assert-InstallCondition ($script:TaskCalls.Count -eq 0) 'Foreign task refusal mutated scheduler state.'
    Write-Output 'PASS: SourceAndOwnershipRefusalsPreserveOldTask'

    # Selector: StagedBytesAndExactLegacyRetirement. Exercises actual copy/hash,
    # immutable revisions/preimages, task readback and no manifest clobbering.
    $operations = New-TaskFixture
    $legacy = New-LegacyTaskFixture
    $other = New-LegacyTaskFixture -Name 'UnrelatedNetworkTask' -Action 'unrelated'
    $legacyOriginalXml = $legacy.Xml
    $result = Invoke-Usb4PeerInstall -Plan $plan -Operations $operations -Apply -ReplaceLegacyTasks
    Assert-InstallCondition ($result.State -eq 'Installed' -and -not $result.RuntimeQualified) 'Installation falsely claims runtime qualification or did not finish.'
    Assert-InstallCondition ($script:TaskStore['\PC-AI\Usb4PeerNetwork'].Enabled -and -not $script:TaskStore['\PC-AI-ThunderboltPeerAgent'].Enabled -and $script:TaskStore['\UnrelatedNetworkTask'].Enabled) 'Installer disabled unrelated task or failed intended task transition.'
    Assert-InstallCondition ((Get-FileHash -LiteralPath $result.ControllerPath -Algorithm SHA256).Hash -ceq $plan.controllerSha256) 'Controller copy hash differs.'
    Assert-InstallCondition ((Get-FileHash -LiteralPath $result.PolicyPath -Algorithm SHA256).Hash -ceq $plan.policySha256) 'Policy copy hash differs.'
    Assert-InstallCondition ([IO.File]::ReadAllText((Join-Path $result.Preimages 'PC-AI-ThunderboltPeerAgent.xml')) -ceq $legacyOriginalXml) 'Legacy XML preimage missing or changed.'
    $verification = Test-Usb4PeerInstallation -Root $destination -Operations $operations
    Assert-InstallCondition ($verification.State -eq 'Verified' -and -not $verification.RuntimeQualified) 'Verification incorrectly claims runtime qualification.'
    $taskBeforeDrift = $script:TaskStore['\PC-AI\Usb4PeerNetwork'].Xml
    $script:TaskStore['\PC-AI\Usb4PeerNetwork'] | Add-Member -NotePropertyName SystemPrincipal -NotePropertyValue $false -Force
    Assert-InstallThrows { Test-Usb4PeerInstallation -Root $destination -Operations $operations } 'settings differ'
    $script:TaskStore['\PC-AI\Usb4PeerNetwork'].SystemPrincipal = $true
    $script:TaskStore['\PC-AI\Usb4PeerNetwork'] | Add-Member -NotePropertyName HighestPrincipal -NotePropertyValue $false -Force
    Assert-InstallThrows { Test-Usb4PeerInstallation -Root $destination -Operations $operations } 'settings differ'
    $script:TaskStore['\PC-AI\Usb4PeerNetwork'].HighestPrincipal = $true
    $defaultOmissions = [xml]$taskBeforeDrift
    foreach ($entry in @(@{ Parent = $defaultOmissions.Task.Settings; Name = 'MultipleInstancesPolicy' }, @{ Parent = $defaultOmissions.Task.Settings; Name = 'WakeToRun' },
            @{ Parent = $defaultOmissions.Task.Triggers.BootTrigger; Name = 'Enabled' }, @{ Parent = $defaultOmissions.Task.Triggers.EventTrigger; Name = 'Enabled' },
            @{ Parent = $defaultOmissions.Task.Triggers.CalendarTrigger; Name = 'Enabled' })) {
        $node = $entry.Parent.SelectSingleNode("*[local-name()='$($entry.Name)']")
        $entry.Parent.RemoveChild($node) | Out-Null
    }
    $script:TaskStore['\PC-AI\Usb4PeerNetwork'].Xml = $defaultOmissions.OuterXml
    $null = Test-Usb4PeerInstallation -Root $destination -Operations $operations
    $script:TaskStore['\PC-AI\Usb4PeerNetwork'].Xml = $taskBeforeDrift
    $taskDrift = [xml]$taskBeforeDrift
    $taskDrift.Task.Actions.Exec.Command = 'C:\foreign.exe'
    $script:TaskStore['\PC-AI\Usb4PeerNetwork'].Xml = $taskDrift.OuterXml
    Assert-InstallThrows { Test-Usb4PeerInstallation -Root $destination -Operations $operations } 'settings differ'
    $taskDrift = [xml]$taskBeforeDrift
    $taskDrift.Task.Actions.Exec.Arguments += ' -UnexpectedSwitch'
    $script:TaskStore['\PC-AI\Usb4PeerNetwork'].Xml = $taskDrift.OuterXml
    Assert-InstallThrows { Test-Usb4PeerInstallation -Root $destination -Operations $operations } 'settings differ'
    $taskDrift = [xml]$taskBeforeDrift
    $taskDrift.Task.Settings.MultipleInstancesPolicy = 'Parallel'
    $script:TaskStore['\PC-AI\Usb4PeerNetwork'].Xml = $taskDrift.OuterXml
    Assert-InstallThrows { Test-Usb4PeerInstallation -Root $destination -Operations $operations } 'settings differ'
    $script:TaskStore['\PC-AI\Usb4PeerNetwork'].Xml = $taskBeforeDrift
    [IO.File]::AppendAllText($result.ControllerPath, "`n# tamper")
    Assert-InstallThrows { Test-Usb4PeerInstallation -Root $destination -Operations $operations } 'hash mismatch'
    $plan2 = Get-Usb4PeerInstallPlan -PolicySource $policy -ControllerSource $controller -Destination $destination -Executable $executable -CurrentHost 'fixture-host'
    Assert-InstallCondition ($plan2.revision -ceq 'r2' -and $plan2.revisionPath -cne $plan.revisionPath) 'Revision naming overwrites a prior source.'
    Write-Output 'PASS: StagedBytesAndExactLegacyRetirement'

    # Selector: PartialInstallRollsBackAndReportsFailedRollback. A mutation that
    # succeeds and then throws must restore previous enabled states and pointer.
    $rollbackDestination = Join-Path $workspace 'rollback-runtime'
    $rollbackPlan = Get-Usb4PeerInstallPlan -PolicySource $policy -ControllerSource $controller -Destination $rollbackDestination -Executable $executable -CurrentHost 'fixture-host'
    $operations = New-TaskFixture
    $legacy = New-LegacyTaskFixture
    $script:FailNewEnable = $true
    Assert-InstallThrows { Invoke-Usb4PeerInstall -Plan $rollbackPlan -Operations $operations -Apply -ReplaceLegacyTasks } 'injected enable failure'
    Assert-InstallCondition ($script:TaskStore['\PC-AI-ThunderboltPeerAgent'].Enabled -and -not $script:TaskStore.ContainsKey('\PC-AI\Usb4PeerNetwork')) 'Partial failure left new task or failed to restore legacy enabled state.'
    Assert-InstallCondition (-not (Test-Path -LiteralPath (Join-Path $rollbackDestination 'installed.json')) -and (Test-Path -LiteralPath (Join-Path $rollbackPlan.revisionPath 'preimages/PC-AI-ThunderboltPeerAgent.xml'))) 'Pointer rollback failed or recovery preimages were discarded.'
    $rollbackPlan2 = Get-Usb4PeerInstallPlan -PolicySource $policy -ControllerSource $controller -Destination $rollbackDestination -Executable $executable -CurrentHost 'fixture-host'
    $script:IgnoreRemoval = $true
    Assert-InstallThrows { Invoke-Usb4PeerInstall -Plan $rollbackPlan2 -Operations $operations -Apply -ReplaceLegacyTasks } 'rollback removal readback failed'
    Write-Output 'PASS: PartialInstallRollsBackAndReportsFailedRollback'

    # Selector: DangerousInstallPathsAndLegacyMismatchRefused. Detects reserved
    # Windows leaves, filesystem-root targets and unrelated legacy-name actions.
    Assert-InstallThrows { Assert-Usb4InstallPath ([IO.Path]::GetPathRoot($workspace)) } 'filesystem root'
    Assert-InstallThrows { Assert-Usb4InstallPath (Join-Path $workspace 'NUL.txt') } 'Reserved'
    Assert-InstallThrows { Assert-Usb4InstallPath (Join-Path $workspace '$null') } 'Reserved'
    $operations = New-TaskFixture
    $null = New-LegacyTaskFixture -Action 'unrelated program'
    Assert-InstallThrows { Invoke-Usb4PeerInstall -Plan $plan -Operations $operations -ReplaceLegacyTasks } 'does not match'
    Assert-InstallCondition ($script:TaskCalls.Count -eq 0) 'Legacy mismatch changed host state.'
    $operations = New-TaskFixture
    $activeLegacy = New-LegacyTaskFixture
    $activeLegacy.Running = $true
    Assert-InstallThrows { Invoke-Usb4PeerInstall -Plan $plan -Operations $operations -Apply -ReplaceLegacyTasks } 'Legacy controller task is active'
    Assert-InstallCondition ($script:TaskCalls.Count -eq 0 -and $activeLegacy.Enabled) 'Active legacy controller was interrupted or overlapped.'
    $foreignRoot = Join-Path $workspace 'unowned-runtime'
    [IO.Directory]::CreateDirectory($foreignRoot) | Out-Null
    [IO.File]::WriteAllText((Join-Path $foreignRoot 'foreign.txt'), 'retain this work')
    $foreignPlan = Get-Usb4PeerInstallPlan -PolicySource $policy -ControllerSource $controller -Destination $foreignRoot -Executable $executable -CurrentHost 'fixture-host'
    $operations = New-TaskFixture
    Assert-InstallThrows { Invoke-Usb4PeerInstall -Plan $foreignPlan -Operations $operations -Apply } 'unowned paths'
    Assert-InstallCondition ($script:TaskCalls.Count -eq 0 -and [IO.File]::ReadAllText((Join-Path $foreignRoot 'foreign.txt')) -ceq 'retain this work') 'Unowned runtime was modified.'
    Write-Output 'PASS: DangerousInstallPathsAndLegacyMismatchRefused'

    # Selector: ConcurrentTaskWriterIsPreserved. Staging may complete, but a new
    # task owner appearing before publication must never be force-overwritten.
    $raceRoot = Join-Path $workspace 'race-runtime'
    $racePlan = Get-Usb4PeerInstallPlan -PolicySource $policy -ControllerSource $controller -Destination $raceRoot -Executable $executable -CurrentHost 'fixture-host'
    $operations = New-TaskFixture
    $script:RaceReads = 0
    $operations.GetTask = { param($Name, $Path)
        $script:RaceReads++
        if ($script:RaceReads -ge 2) {
            [pscustomobject]@{ Name = $Name; Path = $Path; Xml = '<Task><RegistrationInfo><Description>new foreign owner</Description></RegistrationInfo><Settings><Enabled>true</Enabled></Settings></Task>'; Enabled = $true; Running = $false }
        }
        else { $null }
    }
    Assert-InstallThrows { Invoke-Usb4PeerInstall -Plan $racePlan -Operations $operations -Apply } 'changed after planning'
    Assert-InstallCondition (@($script:TaskCalls | Where-Object { $_ -match '^(?:register|enabled|remove):' }).Count -eq 0) 'Concurrent task owner was mutated.'
    Write-Output 'PASS: ConcurrentTaskWriterIsPreserved'

    # Release manifests bind declared source provenance to the exact staged bytes.
    $releasePolicy = Join-Path $workspace 'release-policy.json'
    $releaseHost = [Environment]::MachineName
    [IO.File]::WriteAllText($releasePolicy, (@{ schemaVersion = 1; hostName = $releaseHost; mode = 'Peer' } | ConvertTo-Json))
    $releaseSource = Join-Path $workspace 'release.json'
    $policyRecord = @{ path = 'Config/usb4-fleet/fixture-policy.json'; sha256 = (Get-FileHash -LiteralPath $releasePolicy -Algorithm SHA256).Hash; size = (Get-Item -LiteralPath $releasePolicy).Length; gitBlob = ('a' * 40) }
    $release = @{ schemaVersion = 1; artifactType = 'pcai-usb4-fleet-release'; kind = 'deployable'; runtimeQualified = $false;
        source = @{ repository = 'https://github.com/David-Martel/PC-AI'; commit = ('b' * 40) };
        host   = @{ id = $releaseHost; platform = 'windows'; architecture = (Get-Usb4ReleaseArchitecture) };
        policy = $policyRecord;
        files  = @(@{ path = 'Tools/SystemScripts/Networking/Invoke-Usb4PeerNetwork.ps1'; sha256 = (Get-FileHash -LiteralPath $controller -Algorithm SHA256).Hash; size = (Get-Item -LiteralPath $controller).Length; gitBlob = ('c' * 40) },
            @{ path = 'Tools/SystemScripts/Networking/Install-Usb4PeerNetwork.ps1'; sha256 = (Get-FileHash -LiteralPath $SourcePath -Algorithm SHA256).Hash; size = (Get-Item -LiteralPath $SourcePath).Length; gitBlob = ('d' * 40) }, $policyRecord)
    }
    $releaseDestination = Join-Path $workspace 'release-runtime'
    foreach ($case in @('Malformed', 'ForeignHost', 'HostArray', 'SchemaString', 'SchemaBoolean', 'KindArray', 'WrongArchitecture', 'WrongPlatform', 'WrongKind', 'ControllerHash', 'InstallerHash', 'PolicyHash', 'PolicyRecord', 'DuplicatePayload', 'Commit', 'PayloadSize')) {
        $bad = $release | ConvertTo-Json -Depth 10 | ConvertFrom-Json -AsHashtable
        switch ($case) {
            'ForeignHost' { $bad.host.id = 'foreign-host' }
            'HostArray' { $bad.host.id = @($releaseHost) }
            'SchemaString' { $bad.schemaVersion = '1' }
            'SchemaBoolean' { $bad.schemaVersion = $true }
            'KindArray' { $bad.kind = @('deployable') }
            'WrongArchitecture' { $bad.host.architecture = $(if ((Get-Usb4ReleaseArchitecture) -eq 'x86') { 'x86_64' } else { 'x86' }) }
            'WrongPlatform' { $bad.host.platform = 'linux' }
            'WrongKind' { $bad.kind = 'source-only' }
            'ControllerHash' { $bad.files[0].sha256 = '0' * 64 }
            'InstallerHash' { $bad.files[1].sha256 = '0' * 64 }
            'PolicyHash' { $bad.files[2].sha256 = '0' * 64 }
            'PolicyRecord' { $bad.policy.sha256 = '0' * 64 }
            'DuplicatePayload' { $bad.files += $bad.files[0] }
            'Commit' { $bad.source.commit = 'main' }
            'PayloadSize' { $bad.files[0].size++ }
        }
        $text = $(if ($case -eq 'Malformed') { '{invalid' } else { $bad | ConvertTo-Json -Depth 10 })
        [IO.File]::WriteAllText($releaseSource, $text)
        $operations = New-TaskFixture
        Assert-InstallThrows { Get-Usb4PeerInstallPlan -PolicySource $releasePolicy -ControllerSource $controller -Destination $releaseDestination -Executable $executable -CurrentHost $releaseHost -ReleaseManifestSource $releaseSource } '.'
        Assert-InstallCondition ($script:TaskCalls.Count -eq 0 -and -not (Test-Path -LiteralPath $releaseDestination)) "$case release refusal touched task or runtime state."
    }
    Write-Output 'PASS: ReleaseManifestMalformedForeignOrMismatchedRefusedBeforeMutation'

    [IO.File]::WriteAllText($releaseSource, ($release | ConvertTo-Json -Depth 10))
    $releasePlan = Get-Usb4PeerInstallPlan -PolicySource $releasePolicy -ControllerSource $controller -Destination $releaseDestination -Executable $executable -CurrentHost $releaseHost -ReleaseManifestSource $releaseSource
    $changed = $release | ConvertTo-Json -Depth 10 | ConvertFrom-Json -AsHashtable
    $changed.source.commit = 'e' * 40
    [IO.File]::WriteAllText($releaseSource, ($changed | ConvertTo-Json -Depth 10))
    $operations = New-TaskFixture
    Assert-InstallThrows { Invoke-Usb4PeerInstall -Plan $releasePlan -Operations $operations -Apply } 'changed after planning'
    Assert-InstallCondition ($script:TaskCalls.Count -eq 0 -and -not (Test-Path -LiteralPath $releaseDestination)) 'Release race made task or runtime changes.'
    Write-Output 'PASS: ChangedReleaseManifestPreservesTasks'

    [IO.File]::WriteAllText($releaseSource, ($release | ConvertTo-Json -Depth 10))
    $releaseResult = Invoke-Usb4PeerInstall -Plan $releasePlan -Operations $operations -Apply
    $installedPath = Join-Path $releaseDestination 'installed.json'
    $installedBytes = [IO.File]::ReadAllBytes($installedPath)
    $installedRelease = Get-Content -LiteralPath $installedPath -Raw | ConvertFrom-Json -AsHashtable
    Assert-InstallCondition (-not $releaseResult.RuntimeQualified -and -not $installedRelease.runtimeQualified -and $installedRelease.sourceCommit -eq ('b' * 40) -and $installedRelease.sourceRepository -eq $release.source.repository) 'Install lost declared source provenance or claimed runtime qualification.'
    Assert-InstallCondition ($installedRelease.releaseManifestPath -eq $releasePlan.releaseManifestPath -and (Get-FileHash -LiteralPath $releasePlan.releaseManifestPath -Algorithm SHA256).Hash -eq $releasePlan.releaseManifestSha256) 'Immutable release manifest copy changed bytes or path.'
    $verification = Test-Usb4PeerInstallation -Root $releaseDestination -Operations $operations
    Assert-InstallCondition ($verification.Installed -and -not $verification.RuntimeQualified) 'Release-bound verification lost optional-less semantics.'
    $releaseBytes = [IO.File]::ReadAllBytes($releasePlan.releaseManifestPath)
    [IO.File]::AppendAllText($releasePlan.releaseManifestPath, "`n")
    Assert-InstallThrows { Test-Usb4PeerInstallation -Root $releaseDestination -Operations $operations } 'release manifest path or hash mismatch'
    [IO.File]::WriteAllBytes($releasePlan.releaseManifestPath, $releaseBytes)
    $raceOperations = $operations.Clone()
    $script:ReleaseReadbackRacePath = $releasePlan.releaseManifestPath
    $raceOperations.AssertRoot = { param($Path)
        if ($Path -ceq $script:ReleaseReadbackRacePath) { [IO.File]::AppendAllText($Path, "`n") }
    }
    Assert-InstallThrows { Test-Usb4PeerInstallation -Root $releaseDestination -Operations $raceOperations } 'source binding mismatch'
    [IO.File]::WriteAllBytes($releasePlan.releaseManifestPath, $releaseBytes)
    $changed = $release | ConvertTo-Json -Depth 10 | ConvertFrom-Json -AsHashtable
    $changed.files[0].sha256 = '0' * 64
    [IO.File]::WriteAllText($releasePlan.releaseManifestPath, ($changed | ConvertTo-Json -Depth 10))
    $installedRelease.releaseManifestSha256 = (Get-FileHash -LiteralPath $releasePlan.releaseManifestPath -Algorithm SHA256).Hash
    [IO.File]::WriteAllText($installedPath, ($installedRelease | ConvertTo-Json -Depth 10))
    Assert-InstallThrows { Test-Usb4PeerInstallation -Root $releaseDestination -Operations $operations } 'Release payload hash or size'
    [IO.File]::WriteAllBytes($releasePlan.releaseManifestPath, $releaseBytes); [IO.File]::WriteAllBytes($installedPath, $installedBytes)
    Write-Output 'PASS: ImmutableReleaseReadbackAndInstalledPayloadBinding'

    $releaseRollbackRoot = Join-Path $workspace 'release-rollback'
    $releaseRollbackPlan = Get-Usb4PeerInstallPlan -PolicySource $releasePolicy -ControllerSource $controller -Destination $releaseRollbackRoot -Executable $executable -CurrentHost $releaseHost -ReleaseManifestSource $releaseSource
    $operations = New-TaskFixture; $legacy = New-LegacyTaskFixture; $script:FailNewEnable = $true
    Assert-InstallThrows { Invoke-Usb4PeerInstall -Plan $releaseRollbackPlan -Operations $operations -Apply -ReplaceLegacyTasks } 'injected enable failure'
    Assert-InstallCondition ($script:TaskStore['\PC-AI-ThunderboltPeerAgent'].Enabled -and -not $script:TaskStore.ContainsKey('\PC-AI\Usb4PeerNetwork') -and -not (Test-Path -LiteralPath (Join-Path $releaseRollbackRoot 'installed.json'))) 'Release install failure did not restore original task/pointer custody.'
    Assert-InstallCondition ((Get-FileHash -LiteralPath $releaseRollbackPlan.releaseManifestPath -Algorithm SHA256).Hash -eq $releaseRollbackPlan.releaseManifestSha256 -and (Test-Path -LiteralPath (Join-Path $releaseRollbackPlan.revisionPath 'preimages/PC-AI-ThunderboltPeerAgent.xml'))) 'Failed release install discarded exact recovery evidence.'
    Write-Output 'PASS: FailedReleaseInstallationPreservesPreimagesAndRollsBack'

    if ($IsWindows) {
        # Selector: WindowsTaskSchemaAndServiceAccount. This invokes the actual
        # scheduler parser in memory, never registering or running a task.
        $scheduler = New-Object -ComObject Schedule.Service
        $scheduler.Connect()
        $definition = $scheduler.NewTask(0)
        $definition.XmlText = $plan.taskXml
        Assert-InstallCondition ($definition.Principal.UserId -in @('SYSTEM', 'S-1-5-18') -and [int]$definition.Principal.LogonType -eq 5 -and [int]$definition.Principal.RunLevel -eq 1) 'Windows parser did not retain SYSTEM/ServiceAccount/Highest.'
        Assert-InstallCondition (-not $definition.Settings.Enabled) 'Schema validation unexpectedly enabled the fixture task.'
        $invalidXml = $plan.taskXml.Replace('<UserId>S-1-5-18</UserId>', '<UserId>S-1-5-18</UserId><LogonType>ServiceAccount</LogonType>')
        Assert-InstallThrows { $definition.XmlText = $invalidXml } 'LogonType|incorrectly formatted|out of range|invalid'
        $systemTask = Get-ScheduledTask -ErrorAction Stop | Where-Object { $_.Principal.UserId -in @('SYSTEM', 'S-1-5-18', 'NT AUTHORITY\SYSTEM') -and [string]$_.Principal.RunLevel -in @('Highest', '1') } | Select-Object -First 1
        if (-not $systemTask) { throw 'No existing highest SYSTEM task is available for native CIM readback validation.' }
        $systemSnapshot = Get-Usb4ScheduledTaskSnapshot -Name $systemTask.TaskName -Path $systemTask.TaskPath
        Assert-InstallCondition ($systemSnapshot.SystemPrincipal -and $systemSnapshot.HighestPrincipal) 'Actual SYSTEM task CIM principal did not retain service-account/highest proof.'
        Write-Output 'PASS: WindowsTaskSchemaAndServiceAccount'
    }
    else { Write-Output 'NOT_RUN: WindowsTaskSchemaAndServiceAccount requires Windows Task Scheduler.' }

    if ($IsWindows -and [Security.Principal.WindowsPrincipal]::new([Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        # Selector: RealImmutableFileOwnership. Uses the actual Windows ACL
        # implementation; a protected parent alone leaves copied files user-owned.
        $privateFixture = Join-Path $workspace 'real-acl'
        [IO.Directory]::CreateDirectory($privateFixture) | Out-Null
        $privateScript = Join-Path $privateFixture 'controller.ps1'
        [IO.File]::WriteAllText($privateScript, '# inert ACL fixture')
        Set-Usb4RuntimeProtection -Path $privateFixture -Recurse
        Assert-Usb4RuntimeProtection -Path $privateFixture
        Assert-Usb4RuntimeProtection -Path $privateScript
        $owner = (Get-Acl -LiteralPath $privateScript).GetOwner([Security.Principal.SecurityIdentifier]).Value
        Assert-InstallCondition ($owner -eq 'S-1-5-32-544') 'Copied runtime file remains owned by an interactive user.'
        $liveTask = @(Get-ScheduledTask -ErrorAction Stop | Where-Object TaskName -CEQ 'PC-AI-ThunderboltPeerAgent')
        if ($liveTask.Count -eq 1) {
            $liveSnapshot = Get-Usb4ScheduledTaskSnapshot -Name $liveTask[0].TaskName -Path $liveTask[0].TaskPath
            Assert-InstallCondition ($liveSnapshot.Enabled -eq $liveTask[0].Settings.Enabled -and $liveSnapshot.Xml -match 'Register-ThunderboltPeer') 'Actual scheduler snapshot shape or action was lost.'
        }
        Write-Output 'PASS: RealImmutableFileOwnership'
    }
    else { Write-Output 'NOT_RUN: RealImmutableFileOwnership requires an elevated Windows token.' }
}
finally {
    $absolute = [IO.Path]::GetFullPath($workspace)
    $temporary = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
    if (-not $absolute.StartsWith($temporary, [StringComparison]::OrdinalIgnoreCase) -or [IO.Path]::GetFileName($absolute) -notmatch '^pcai-usb4-installer-[a-f0-9]{32}$') { throw 'Refusing cleanup outside the owned temporary fixture.' }
    if (Test-Path -LiteralPath $absolute) { Remove-Item -LiteralPath $absolute -Recurse -Force -ErrorAction Stop }
}
