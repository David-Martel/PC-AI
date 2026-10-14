#Requires -Modules Pester
BeforeAll {
    $script:Tool = Join-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) 'Tools/SystemScripts/Networking/Start-PcaiCloudMount.ps1'
    . $script:Tool
    . (Join-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) 'Tools/Repair-CloudCacheMount.ps1')
    $script:Sid = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
    $script:Binding = @{
        TaskName = 'Owned-Fixture'; Drive = 'U'; ProviderUnc = '\\dropbox-cloud\gmail-u'
        CacheDirectory = 'F:\RcloneCache\gmail-u'; ExpectedUserSid = $script:Sid
        ExpectedTaskXmlSha256 = 'A' * 64; Apply = $true; TimeoutSeconds = 5; PollMilliseconds = 10
    }
    $script:Healthy = [pscustomobject]@{
        State = 'Healthy'; Reason = ''; CacheGuid = 'c3e0daf0-3678-4842-8d9c-66aa3619b11a'
        BackingGuid = 'b9f25542-491b-4718-8556-4cbe67e599b8'; BackingFs = 'ReFS'
        CacheFree = 200GB; BackingFree = 300GB
        ActualBackingGuid = 'b9f25542-491b-4718-8556-4cbe67e599b8'; BackingFileId = 'A' * 48
    }
    $script:Absent = [pscustomobject]@{ NetResult = 2250; Unc = ''; DosResult = 0; DosError = 2; DriveType = 1 }
    $script:Mounted = [pscustomobject]@{ NetResult = 0; Unc = '\\dropbox-cloud\gmail-u'; DosResult = 1; DosError = 0; DriveType = 4 }
    # Real native compilation and current metadata-only mapping query. No mount/task mutation.
    $null = Get-PcaiCloudMountMapping U
    function Update-OwnedTaskFixtureXml {
        $execute = [Security.SecurityElement]::Escape($script:TaskFixture.Actions[0].Execute)
        $arguments = [Security.SecurityElement]::Escape($script:TaskFixture.Actions[0].Arguments)
        $script:Xml = '<Task xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task"><Principals><Principal id="Author"><UserId>' + $script:Sid + '</UserId><LogonType>InteractiveToken</LogonType></Principal></Principals><Settings><MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy></Settings><Actions Context="Author"><Exec><Command>' + $execute + '</Command><Arguments>' + $arguments + '</Arguments></Exec></Actions></Task>'
        $script:XmlHash = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData([Text.Encoding]::UTF8.GetBytes($script:Xml)))
    }
    function New-OwnedBackendFixture {
        param([string]$Mode, [string]$Root)
        $file = Join-Path $Root 'ordinary.vhdx'
        [IO.File]::WriteAllBytes($file, [byte[]]::new(32))
        $lease = Get-PcaiCloudMountBackingLease $file
        try { $actualGuid = $lease.VolumeGuid } finally { $lease.Dispose() }
        $marker = Join-Path $Root 'api-entered.txt'
        $childMarker = Join-Path $Root 'descendant.json'
        $arguments = 'mount remote: U: --read-only --cache-dir F:\RcloneCache\gmail-u --volname \\dropbox-cloud\gmail-u'
        $xml = '<Task xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task"><Principals><Principal id="Author"><UserId>' + $script:Sid + '</UserId><LogonType>InteractiveToken</LogonType></Principal></Principals><Settings><MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy></Settings><Actions Context="Author"><Exec><Command>C:\Owned\rclone.exe</Command><Arguments>' + $arguments + '</Arguments></Exec></Actions></Task>'
        $xml64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($xml))
        $api = @'
$script:Mode = '__MODE__'
$script:Marker = '__MARKER__'
$script:ChildMarker = '__CHILD__'
$script:File = '__FILE__'
$script:OwnerSid = '__SID__'
$script:BackingGuid = '__GUID__'
$script:Xml = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('__XML64__'))
$script:OwnedArguments = 'mount remote: U: --read-only --cache-dir F:\RcloneCache\gmail-u --volname \\dropbox-cloud\gmail-u'
$script:Started = [IO.File]::Exists($script:Marker)
function Get-PcaiCloudMountContext { [pscustomobject]@{ Sid = $script:OwnerSid; SessionId = [Diagnostics.Process]::GetCurrentProcess().SessionId; Elevated = $false } }
if ($script:Mode -eq 'Compile') {
    function Add-Type { [CmdletBinding()]param($TypeDefinition) [IO.File]::WriteAllText($script:Marker,'Compile'); Start-Sleep -Seconds 60 }
}
function Get-DiskImage {
    [CmdletBinding()]param($ImagePath)
    if ($script:Mode -eq 'Overflow') { [Console]::Out.Write([string]::new('x',1000000)); throw 'Deliberate fixture output overflow.' }
    if ($script:Mode -eq 'ErrorOverflow') { [Console]::Error.Write([string]::new('x',1000000)); throw 'Deliberate fixture error-stream overflow.' }
    if ($script:Mode -eq 'Noise') { [Console]::Out.Write('unexpected-output'); }
    if ($script:Mode -eq 'Pipe') {
        $info = [Diagnostics.ProcessStartInfo]::new((Join-Path $PSHOME 'pwsh.exe'))
        $info.UseShellExecute = $false; $info.CreateNoWindow = $true
        $info.ArgumentList.Add('-NoProfile'); $info.ArgumentList.Add('-Command'); $info.ArgumentList.Add('Start-Sleep -Seconds 30')
        $descendant = [Diagnostics.Process]::Start($info)
        [IO.File]::WriteAllText($script:ChildMarker,([pscustomobject]@{Pid=$descendant.Id;Creation=$descendant.StartTime.ToUniversalTime().ToString('o')}|ConvertTo-Json))
        [IO.File]::WriteAllText($script:Marker,'Pipe')
        throw 'Fixture backend exits while descendant retains inherited pipes.'
    }
    [pscustomobject]@{ ImagePath = $script:File; Attached = $true; Number = 4; DevicePath = '\\.\PhysicalDrive4' }
}
function Get-Disk { [CmdletBinding()]param($Number) [pscustomobject]@{ Number = 4 } }
function Get-Partition { [CmdletBinding()]param($DiskNumber) [pscustomobject]@{ DiskNumber = 4; DriveLetter = 'F' } }
function Get-Volume {
    [CmdletBinding()]param($DriveLetter,$Partition)
    $cache = [pscustomobject]@{ Path = '\\?\Volume{c3e0daf0-3678-4842-8d9c-66aa3619b11a}\'; DriveLetter = 'F'; FileSystem = 'NTFS'; FileSystemLabel = 'cloud-cache-disk'; SizeRemaining = 200GB }
    $backing = [pscustomobject]@{ Path = '\\?\Volume{' + $script:BackingGuid + '}\'; DriveLetter = 'T'; FileSystem = 'ReFS'; FileSystemLabel = 'dev-drive'; SizeRemaining = 300GB }
    if ($DriveLetter -eq 'T') { $backing } elseif ($Partition) { $cache } else { @($cache,$backing) }
}
function Export-ScheduledTask {
    [CmdletBinding()]param($TaskName)
    if ($script:Mode -eq 'Scheduler') { [IO.File]::WriteAllText($script:Marker,'Scheduler'); Start-Sleep -Seconds 60 }
    $script:Xml
}
function Get-ScheduledTask {
    [CmdletBinding()]param($TaskName)
    [pscustomobject]@{Principal=[pscustomobject]@{UserId=[Security.Principal.WindowsIdentity]::GetCurrent().Name;RunLevel='Limited';LogonType='Interactive'};
        Settings=[pscustomobject]@{Enabled=$true;MultipleInstances='IgnoreNew'};
        Actions=@([pscustomobject]@{Execute='C:\Owned\rclone.exe';Arguments=$script:OwnedArguments});State=if($script:Mode -eq 'Start' -or ($script:Mode -eq 'StartHappy' -and -not $script:Started)){'Ready'}else{'Running'}}
}
function Get-PcaiCloudMountMapping {
    param($Letter)
    if ($script:Mode -eq 'Start' -or ($script:Mode -eq 'StartHappy' -and -not $script:Started)) { [pscustomobject]@{ NetResult=2250;Unc='';DosResult=0;DosError=2;DriveType=1 } }
    else { [pscustomobject]@{ NetResult=0;Unc='\\dropbox-cloud\gmail-u';DosResult=1;DosError=0;DriveType=4 } }
}
function Get-CimInstance {
    [CmdletBinding()]param($ClassName,$Filter)
    if ($script:Mode -eq 'Wmi') { [IO.File]::WriteAllText($script:Marker,'Wmi'); Start-Sleep -Seconds 60 }
    [pscustomobject]@{ ProcessId=6000;CreationDate=[datetime]'2026-01-01T00:00:00Z';ExecutablePath='C:\Owned\rclone.exe';CommandLine='C:\Owned\rclone.exe '+$script:OwnedArguments }
}
function Invoke-CimMethod { [CmdletBinding()]param($InputObject,$MethodName) [pscustomobject]@{ReturnValue=0;Sid=$script:OwnerSid} }
function Start-ScheduledTask { [CmdletBinding()]param($TaskName) [IO.File]::WriteAllText($script:Marker,'Start'); if($script:Mode -eq 'StartHappy'){$script:Started=$true}else{Start-Sleep -Seconds 60} }
'@
        foreach ($pair in @(@('__MODE__',$Mode),@('__MARKER__',$marker),@('__CHILD__',$childMarker),@('__FILE__',$file),@('__SID__',$script:Sid),@('__GUID__',$actualGuid),@('__XML64__',$xml64))) { $api = $api.Replace($pair[0],$pair[1].Replace("'","''")) }
        $profile = Join-Path $Root 'external-api.ps1'
        [IO.File]::WriteAllText($profile,$api)
        [pscustomobject]@{ Profile = $profile; Marker = $marker; ChildMarker = $childMarker; Request = [pscustomobject]@{
            RequestId = [guid]::NewGuid().ToString('D'); Operation = if ($Mode -eq 'Start') { 'Start' } else { 'Snapshot' }
            EndTicks = [Environment]::TickCount64 + 8000; ExpectedSessionId = [Diagnostics.Process]::GetCurrentProcess().SessionId; ExpectedElevated = $false
            TaskName = 'Owned-Fixture'; Drive = 'U'; ProviderUnc = '\\dropbox-cloud\gmail-u'; CacheDirectory = 'F:\RcloneCache\gmail-u'
            ExpectedUserSid = $script:Sid; ExpectedTaskXmlSha256 = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData([Text.Encoding]::UTF8.GetBytes($xml)))
            CacheVolumeGuid = 'c3e0daf0-3678-4842-8d9c-66aa3619b11a'; BackingVolumeGuid = $actualGuid; VhdPath = $file
            MinimumCacheFreeBytes = 100GB; MinimumBackingFreeBytes = 140GB; ExpectedBackingFileId = ''
        } }
    }
}

Describe 'Cloud mount admission predicates' {
    It 'accepts physical disk zero instead of confusing it with absent identity' {
        Get-CacheImageDiskNumber ([pscustomobject]@{ Number = $null; DevicePath = '\\.\PhysicalDrive0' }) | Should -Be 0
    }
    It 'accepts the actual Limited null Number with canonical DevicePath' {
        Get-CacheImageDiskNumber ([pscustomobject]@{ Number = $null; DevicePath = '\\.\PhysicalDrive4' }) | Should -Be 4
    }
    It 'rejects conflicting image Number and DevicePath' {
        { Get-CacheImageDiskNumber ([pscustomobject]@{ Number = 5; DevicePath = '\\.\PhysicalDrive4' }) } | Should -Throw '*conflicts*'
    }
    It 'rejects malformed live disk paths: <Path>' -ForEach @(
        @{ Path = '\\.\PhysicalDrive04' }, @{ Path = '\\.\PhysicalDrive-1' },
        @{ Path = '\\.\PhysicalDrive2147483648' }, @{ Path = '\\foreign\PhysicalDrive4' },
        @{ Path = '\\.\PhysicalDrive4\extra' }) {
        { Get-CacheImageDiskNumber ([pscustomobject]@{ Number = $null; DevicePath = $Path }) } | Should -Throw
    }
    It 'admits the exact pair with sufficient reserve' {
        Test-PcaiCloudMountAdmission $script:Healthy ([guid]$script:Healthy.CacheGuid) ([guid]$script:Healthy.BackingGuid) 100GB 140GB | Should -BeTrue
    }
    It 'returns pending for unavailable or delayed Storage association: <State>' -ForEach @(@{ State = 'Pending' }, @{ State = 'Missing' }) {
        Test-PcaiCloudMountAdmission ([pscustomobject]@{ State = $State }) ([guid]::NewGuid()) ([guid]::NewGuid()) 100GB 140GB | Should -BeFalse
    }
    It 'rejects invalid association instead of retrying it' {
        { Test-PcaiCloudMountAdmission ([pscustomobject]@{ State = 'InvalidIdentity'; Reason = 'Wrong disk' }) ([guid]::NewGuid()) ([guid]::NewGuid()) 100GB 140GB } | Should -Throw '*identity refused*'
    }
    It 'rejects wrong cache GUID' {
        { Test-PcaiCloudMountAdmission $script:Healthy ([guid]::NewGuid()) ([guid]$script:Healthy.BackingGuid) 100GB 140GB } | Should -Throw '*identity differs*'
    }
    It 'rejects wrong backing GUID' {
        { Test-PcaiCloudMountAdmission $script:Healthy ([guid]$script:Healthy.CacheGuid) ([guid]::NewGuid()) 100GB 140GB } | Should -Throw '*identity differs*'
    }
    It 'rejects low cache reserve' {
        { Test-PcaiCloudMountAdmission $script:Healthy ([guid]$script:Healthy.CacheGuid) ([guid]$script:Healthy.BackingGuid) 201GB 140GB } | Should -Throw '*reserve*'
    }
    It 'rejects low backing reserve' {
        { Test-PcaiCloudMountAdmission $script:Healthy ([guid]$script:Healthy.CacheGuid) ([guid]$script:Healthy.BackingGuid) 100GB 301GB } | Should -Throw '*reserve*'
    }
    It 'rejects an actual backing file on another volume despite valid T label and reserve' {
        $copy = $script:Healthy.PSObject.Copy(); $copy.ActualBackingGuid = [guid]::NewGuid().ToString('D')
        { Test-PcaiCloudMountAdmission $copy ([guid]$copy.CacheGuid) ([guid]$copy.BackingGuid) 100GB 140GB } | Should -Throw '*resolved VHD backing volume differs*'
    }
    It 'retains exact native command line token boundaries' {
        $tokens = [Pcai.CloudMountNative]::Parse('"C:\Program Files\rclone.exe" mount remote: U: --cache-dir "F:\RcloneCache\space name"')
        $tokens.Count | Should -Be 6
        $tokens[0] | Should -Be 'C:\Program Files\rclone.exe'
        $tokens[5] | Should -Be 'F:\RcloneCache\space name'
    }
}

Describe 'Complete cloud mount gate orchestration' {
    BeforeEach {
        $script:PollCount = 0
        $script:StartCount = 0
        Mock Get-PcaiCloudMountContext { [pscustomobject]@{ Sid = $script:Sid; Elevated = $false; SessionId = 1 } }
        Mock Get-PcaiCloudMountStorage { $script:Healthy }
        Mock Get-PcaiCloudMountMapping { $script:Mounted }
        Mock Get-PcaiCloudMountTask { [pscustomobject]@{ Task = [pscustomobject]@{ State = 'Running' }; HashBefore = 'A' * 64; HashAfter = 'A' * 64 } }
        Mock Test-PcaiCloudMountOwner {
            $script:PcaiCloudOwnerWitness = [pscustomobject]@{ProcessId=6000;CreationUtc='2026-01-01T00:00:00Z';Sid=$script:Sid}
            $true
        }
        Mock Start-ScheduledTask { $script:StartCount++ }
        Mock Get-PcaiCloudMountBackingLease {
            $lease = [pscustomobject]@{ VolumeGuid = $script:Healthy.ActualBackingGuid; FileId = 'A' * 48; FinalPath = '\\?\Volume{b9f25542-491b-4718-8556-4cbe67e599b8}\vm\cloud-cache-disk.vhdx' }
            $lease | Add-Member -MemberType ScriptMethod -Name Dispose -Value { }
            $lease
        }
        Mock Get-DiskImage { [pscustomobject]@{ ImagePath = 'T:\vm\cloud-cache-disk.vhdx'; Attached = $true; Number = 4; DevicePath = '\\.\PhysicalDrive4' } }
        Mock Get-Disk { [pscustomobject]@{ Number = 4 } }
        $script:Partition = New-CimInstance -Namespace root/Microsoft/Windows/Storage -ClassName MSFT_Partition -ClientOnly -Property @{ DiskNumber = [uint32]4; PartitionNumber = [uint32]2; DriveLetter = [char]'F' }
        Mock Get-Partition { $script:Partition }
        Mock Get-Volume {
            $cache = [pscustomobject]@{ Path = '\\?\Volume{c3e0daf0-3678-4842-8d9c-66aa3619b11a}\'; DriveLetter = 'F'; FileSystem = 'NTFS'; FileSystemLabel = 'cloud-cache-disk'; SizeRemaining = $script:Healthy.CacheFree }
            $backing = [pscustomobject]@{ Path = '\\?\Volume{b9f25542-491b-4718-8556-4cbe67e599b8}\'; DriveLetter = 'T'; FileSystem = 'ReFS'; FileSystemLabel = 'dev-drive'; SizeRemaining = $script:Healthy.BackingFree }
            if ($DriveLetter -eq 'T') { $backing } elseif ($Partition) { $cache } else { @($cache, $backing) }
        }
        # Only the process transport is replaced. The entire backend, shared
        # image association and admission predicates execute against OS adapters.
        Mock Invoke-PcaiCloudMountWorker { Invoke-PcaiCloudMountBackend $Request }
    }
    It 'plans without any probes or mutation: <Mode>' -ForEach @(
        @{ Mode = 'Default' }, @{ Mode = 'DryRun' }, @{ Mode = 'WhatIf' }, @{ Mode = 'LongDryRun' }, @{ Mode = 'Help' }) {
        $options = $script:Binding.Clone()
        switch ($Mode) {
            Default { $options.Apply = $false }
            DryRun { $options.DryRun = $true }
            WhatIf { $options.WhatIf = $true }
            LongDryRun { $options.CliArgs = @('--DryRun') }
            Help { $options.Help = $true }
        }
        $result = Invoke-PcaiCloudMount @options
        if ($Mode -ne 'Help') { $result.State | Should -Be Plan }
        Should -Invoke Get-PcaiCloudMountContext -Times 0 -Exactly
        Should -Invoke Get-PcaiCloudMountStorage -Times 0 -Exactly
        Should -Invoke Get-PcaiCloudMountMapping -Times 0 -Exactly
        Should -Invoke Get-PcaiCloudMountTask -Times 0 -Exactly
        Should -Invoke Start-ScheduledTask -Times 0 -Exactly
        Should -Invoke Invoke-PcaiCloudMountWorker -Times 0 -Exactly
    }
    It 'qualifies an existing exact provider without starting another writer' {
        (Invoke-PcaiCloudMount @script:Binding).State | Should -Be AlreadyRunning
        Should -Invoke Start-ScheduledTask -Times 0 -Exactly
    }
    It 'ValidateOnly still qualifies a real admitted existing provider' {
        (Invoke-PcaiCloudMount @script:Binding -ValidateOnly).State | Should -Be AlreadyRunning
        Should -Invoke Start-ScheduledTask -Times 0 -Exactly
    }
    It 'ValidateOnly prohibits provider starts when the expected provider is absent' {
        Mock Get-PcaiCloudMountMapping { $script:Absent }
        Mock Get-PcaiCloudMountTask { [pscustomobject]@{ Task = [pscustomobject]@{ State = 'Ready' }; HashBefore = 'A' * 64; HashAfter = 'A' * 64 } }
        (Invoke-PcaiCloudMount @script:Binding -ValidateOnly).State | Should -Be Hold
        Should -Invoke Start-ScheduledTask -Times 0 -Exactly
    }
    It 'executes the actual diagnostic storage dispatch without scheduler or owner probes' {
        $request = [pscustomobject]@{ RequestId = [guid]::NewGuid().ToString('D'); Operation = 'Storage'; EndTicks = [Environment]::TickCount64 + 5000
            VhdPath = 'T:\vm\cloud-cache-disk.vhdx'; ExpectedUserSid = $script:Sid; ExpectedSessionId = 1; ExpectedElevated = $false }
        $result = Invoke-PcaiCloudMountBackend $request
        $result.State | Should -Be Observed
        $result.Snapshot.Storage.State | Should -Be Healthy
        $result.Snapshot.Storage.ActualBackingGuid | Should -Be $script:Healthy.ActualBackingGuid
        (Get-PcaiCloudMountStorage 'T:\vm\cloud-cache-disk.vhdx' 5000).State | Should -Be Healthy
        Should -Invoke Get-PcaiCloudMountTask -Times 0 -Exactly
        Should -Invoke Test-PcaiCloudMountOwner -Times 0 -Exactly
        Should -Invoke Start-ScheduledTask -Times 0 -Exactly
    }
    It 'refuses changed helper bytes before invoking any storage provider' {
        $original = $script:PcaiCloudMountSourceRoot
        $fixtureRoot = Join-Path $TestDrive 'helper-mismatch'; $tools = Join-Path $fixtureRoot 'Tools'
        New-Item -ItemType Directory -Path $tools | Out-Null
        [IO.File]::WriteAllText((Join-Path $tools 'Repair-CloudCacheMount.ps1'),'Owned changed helper bytes')
        try {
            $script:PcaiCloudMountSourceRoot = $fixtureRoot
            { Get-PcaiCloudMountStorageObservation 'T:\vm\cloud-cache-disk.vhdx' } | Should -Throw '*Reviewed helper bytes changed*'
            Should -Invoke Get-Volume -Times 0 -Exactly
        } finally { $script:PcaiCloudMountSourceRoot = $original }
    }
    It 'rejects another owner context' {
        Mock Get-PcaiCloudMountContext { [pscustomobject]@{ Sid = 'S-1-5-21-1-2-3-4'; Elevated = $false; SessionId = 1 } }
        { Invoke-PcaiCloudMount @script:Binding } | Should -Throw '*Limited interactive*'
        Should -Invoke Get-PcaiCloudMountStorage -Times 0 -Exactly
    }
    It 'rejects elevated context before probing or starting' {
        Mock Get-PcaiCloudMountContext { [pscustomobject]@{ Sid = $script:Sid; Elevated = $true; SessionId = 1 } }
        { Invoke-PcaiCloudMount @script:Binding } | Should -Throw '*Limited interactive*'
        Should -Invoke Start-ScheduledTask -Times 0 -Exactly
    }
    It 'preserves an occupied drive with foreign provider UNC' {
        Mock Get-PcaiCloudMountMapping { [pscustomobject]@{ NetResult = 0; Unc = '\\foreign\share'; DosResult = 1; DosError = 0; DriveType = 4 } }
        { Invoke-PcaiCloudMount @script:Binding } | Should -Throw '*collision*'
        Should -Invoke Start-ScheduledTask -Times 0 -Exactly
    }
    It 'preserves an apparently absent drive when DOS query is denied' {
        Mock Get-PcaiCloudMountMapping { [pscustomobject]@{ NetResult = 2250; Unc = ''; DosResult = 0; DosError = 5; DriveType = 1 } }
        { Invoke-PcaiCloudMount @script:Binding } | Should -Throw '*collision*'
    }
    It 'does not qualify a matching provider without live exact owner process' {
        Mock Test-PcaiCloudMountOwner { $false }
        { Invoke-PcaiCloudMount @script:Binding } | Should -Throw '*unique matching live owner*'
        Should -Invoke Start-ScheduledTask -Times 0 -Exactly
    }
    It 'waits for delayed Storage then qualifies the existing mount' {
        Mock Get-Volume {
            if ($DriveLetter -eq 'T') { [pscustomobject]@{ Path = '\\?\Volume{b9f25542-491b-4718-8556-4cbe67e599b8}\'; FileSystem = 'ReFS'; SizeRemaining = 300GB } }
            else { if (-not $Partition) { $script:PollCount++ }; if ($script:PollCount -ge 3) { [pscustomobject]@{ Path = '\\?\Volume{c3e0daf0-3678-4842-8d9c-66aa3619b11a}\'; DriveLetter = 'F'; FileSystem = 'NTFS'; FileSystemLabel = 'cloud-cache-disk'; SizeRemaining = 200GB } } }
        }
        (Invoke-PcaiCloudMount @script:Binding).State | Should -Be AlreadyRunning
        $script:PollCount | Should -Be 3
    }
    It 'holds within a deadline on permanently missing Storage' {
        Mock Get-Volume { if ($DriveLetter -eq 'T') { [pscustomobject]@{ Path = '\\?\Volume{b9f25542-491b-4718-8556-4cbe67e599b8}\'; FileSystem = 'ReFS'; SizeRemaining = 300GB } } }
        $options = $script:Binding.Clone(); $options.TimeoutSeconds = 1
        $timer = [Diagnostics.Stopwatch]::StartNew()
        (Invoke-PcaiCloudMount @options).State | Should -Be Hold
        $timer.Elapsed.TotalSeconds | Should -BeLessThan 3
        Should -Invoke Start-ScheduledTask -Times 0 -Exactly
    }
    It 'waits for a genuinely unattached image rather than denying a normal logon race' {
        Mock Get-DiskImage {
            $script:PollCount++
            [pscustomobject]@{ ImagePath = 'T:\vm\cloud-cache-disk.vhdx'; Attached = ($script:PollCount -gt 2); Number = 4; DevicePath = '\\.\PhysicalDrive4' }
        }
        (Invoke-PcaiCloudMount @script:Binding).State | Should -Be AlreadyRunning
        $script:PollCount | Should -BeGreaterThan 2
        Should -Invoke Start-ScheduledTask -Times 0 -Exactly
    }
    It 'recovers an unavailable image observation through the actual snapshot' {
        Mock Get-DiskImage {
            $script:PollCount++
            if ($script:PollCount -gt 1) { [pscustomobject]@{ ImagePath = 'T:\vm\cloud-cache-disk.vhdx'; Attached = $true; Number = 4; DevicePath = '\\.\PhysicalDrive4' } }
        }
        (Invoke-PcaiCloudMount @script:Binding).State | Should -Be AlreadyRunning
        Should -Invoke Start-ScheduledTask -Times 0 -Exactly
    }
    It 'recovers a transient image API error without treating it as healthy' {
        Mock Get-DiskImage {
            $script:PollCount++
            if ($script:PollCount -eq 1) { throw [UnauthorizedAccessException]::new('Actual external API fixture denied first query.') }
            [pscustomobject]@{ ImagePath = 'T:\vm\cloud-cache-disk.vhdx'; Attached = $true; Number = 4; DevicePath = '\\.\PhysicalDrive4' }
        }
        (Invoke-PcaiCloudMount @script:Binding).State | Should -Be AlreadyRunning
        $script:PollCount | Should -BeGreaterThan 1
        Should -Invoke Start-ScheduledTask -Times 0 -Exactly
    }
    It 'keeps missing backing metadata pending under the original deadline' {
        Mock Get-PcaiCloudMountBackingLease { throw [ComponentModel.Win32Exception]::new(2) }
        $options = $script:Binding.Clone(); $options.TimeoutSeconds = 1
        $result = Invoke-PcaiCloudMount @options
        $result.State | Should -Be Hold
        $result.StartRequested | Should -BeFalse
        Should -Invoke Start-ScheduledTask -Times 0 -Exactly
    }
    It 'does not hide definitive wrong backing GUID behind missing image observations' {
        Mock Get-PcaiCloudMountBackingLease {
            $lease = [pscustomobject]@{ VolumeGuid = [guid]::NewGuid().ToString('D'); FileId = 'A' * 48; FinalPath = 'fixture' }
            $lease | Add-Member -MemberType ScriptMethod -Name Dispose -Value { }
            $lease
        }
        { Invoke-PcaiCloudMount @script:Binding } | Should -Throw '*resolved VHD backing volume differs*'
        Should -Invoke Get-DiskImage -Times 0 -Exactly
        Should -Invoke Start-ScheduledTask -Times 0 -Exactly
    }
    It 'starts once, then qualifies the exact resulting provider' {
        Mock Get-PcaiCloudMountMapping { if ($script:StartCount) { $script:Mounted } else { $script:Absent } }
        Mock Get-PcaiCloudMountTask { [pscustomobject]@{ Task = [pscustomobject]@{ State = if ($script:StartCount) { 'Running' } else { 'Ready' } }; HashBefore = 'A' * 64; HashAfter = 'A' * 64 } }
        (Invoke-PcaiCloudMount @script:Binding).State | Should -Be Started
        Should -Invoke Start-ScheduledTask -Times 1 -Exactly -ParameterFilter { $TaskName -eq 'Owned-Fixture' }
    }
    It 'refuses a reserve drop on the final Storage recheck' {
        Mock Get-PcaiCloudMountMapping { $script:Absent }
        Mock Get-PcaiCloudMountTask { [pscustomobject]@{ Task = [pscustomobject]@{ State = 'Ready' }; HashBefore = 'A' * 64; HashAfter = 'A' * 64 } }
        Mock Get-Volume {
            if ($DriveLetter -eq 'T') { [pscustomobject]@{ Path = '\\?\Volume{b9f25542-491b-4718-8556-4cbe67e599b8}\'; FileSystem = 'ReFS'; SizeRemaining = 300GB } }
            else { if (-not $Partition) { $script:PollCount++ }; [pscustomobject]@{ Path = '\\?\Volume{c3e0daf0-3678-4842-8d9c-66aa3619b11a}\'; DriveLetter = 'F'; FileSystem = 'NTFS'; FileSystemLabel = 'cloud-cache-disk'; SizeRemaining = if ($script:PollCount -gt 1) { 99GB } else { 200GB } } }
        }
        { Invoke-PcaiCloudMount @script:Binding } | Should -Throw '*reserve*'
        Should -Invoke Start-ScheduledTask -Times 0 -Exactly
    }
    It 'refuses a drive collision appearing immediately before task start' {
        Mock Get-PcaiCloudMountMapping { $script:PollCount++; if ($script:PollCount -eq 1) { $script:Absent } else { [pscustomobject]@{ NetResult = 0; Unc = '\\foreign\share'; DosResult = 1; DosError = 0; DriveType = 4 } } }
        Mock Get-PcaiCloudMountTask { [pscustomobject]@{ Task = [pscustomobject]@{ State = 'Ready' }; HashBefore = 'A' * 64; HashAfter = 'A' * 64 } }
        { Invoke-PcaiCloudMount @script:Binding } | Should -Throw '*collision*'
        Should -Invoke Start-ScheduledTask -Times 0 -Exactly
    }
    It 'does not double launch a Running task whose provider is delayed' {
        Mock Get-PcaiCloudMountMapping { $script:Absent }
        $options = $script:Binding.Clone(); $options.TimeoutSeconds = 1
        (Invoke-PcaiCloudMount @options).State | Should -Be Hold
        Should -Invoke Start-ScheduledTask -Times 0 -Exactly
    }
    It 'never accepts a qualified response that arrives after the global deadline' {
        Mock Invoke-PcaiCloudMountWorker {
            Start-Sleep -Milliseconds 1100
            [pscustomobject]@{ State = 'Qualified'; StartOutcome = 'NotRequested'; Snapshot = $null }
        }
        $options = $script:Binding.Clone(); $options.TimeoutSeconds = 1
        (Invoke-PcaiCloudMount @options).State | Should -Be Hold
        Should -Invoke Start-ScheduledTask -Times 0 -Exactly
    }
    It 'preserves an unknown scheduler outcome without issuing another request' {
        Mock Get-PcaiCloudMountMapping { $script:Absent }
        Mock Get-PcaiCloudMountTask { [pscustomobject]@{ Task = [pscustomobject]@{ State = 'Ready' }; HashBefore = 'A' * 64; HashAfter = 'A' * 64 } }
        Mock Invoke-PcaiCloudMountWorker {
            if ($Request.Operation -eq 'Start') { [pscustomobject]@{ State = 'Expired'; StartOutcome = 'Unknown' } }
            else { Invoke-PcaiCloudMountBackend $Request }
        }
        $result = Invoke-PcaiCloudMount @script:Binding
        $result.State | Should -Be Hold
        $result.StartRequested | Should -Be Unknown
        Should -Invoke Invoke-PcaiCloudMountWorker -Times 1 -Exactly -ParameterFilter { $Request.Operation -eq 'Start' }
    }
    It 'refuses a concurrent process holding the same real session mutex' {
        $mutexName = "Local\PcaiCloudMount-$script:Sid-U"
        $code = '$m = [Threading.Mutex]::new($false, ''' + $mutexName + '''); $null = $m.WaitOne(); [Console]::WriteLine(''owned''); Start-Sleep -Seconds 5; $m.ReleaseMutex(); $m.Dispose()'
        $start = [Diagnostics.ProcessStartInfo]::new((Join-Path $PSHOME 'pwsh.exe'))
        $start.UseShellExecute = $false; $start.CreateNoWindow = $true; $start.RedirectStandardOutput = $true
        foreach ($arg in @('-NoProfile', '-EncodedCommand', [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($code)))) { $start.ArgumentList.Add($arg) }
        $child = [Diagnostics.Process]::new(); $child.StartInfo = $start
        try {
            $child.Start() | Should -BeTrue
            $ready = $child.StandardOutput.ReadLineAsync()
            $ready.Wait(3000) | Should -BeTrue
            $ready.Result | Should -Be owned
            (Invoke-PcaiCloudMount @script:Binding).State | Should -Be Busy
            Should -Invoke Get-PcaiCloudMountStorage -Times 0 -Exactly
            Should -Invoke Start-ScheduledTask -Times 0 -Exactly
        } finally {
            if (-not $child.HasExited) { $child.Kill($true); $child.WaitForExit() }
            $child.Dispose()
        }
    }
}

Describe 'Registered task pinning and live owner qualification' {
    BeforeEach {
        $script:TaskFixture = [pscustomobject]@{
            Principal = [pscustomobject]@{ RunLevel = 'Limited'; LogonType = 'Interactive'; UserId = [Security.Principal.WindowsIdentity]::GetCurrent().Name }
            Settings = [pscustomobject]@{ Enabled = $true; MultipleInstances = 'IgnoreNew' }
            Actions = @([pscustomobject]@{ Execute = 'C:\Owned\rclone.exe'; Arguments = 'mount remote: U: --read-only --cache-dir F:\RcloneCache\gmail-u --volname \\dropbox-cloud\gmail-u' })
            State = 'Running'
        }
        Update-OwnedTaskFixtureXml
        Mock Export-ScheduledTask { $script:Xml }
        Mock Get-ScheduledTask { $script:TaskFixture }
    }
    It 'pins exact task bytes, principal, cache and read-only flags' {
        $result = Get-PcaiCloudMountTask 'Owned-Fixture' $script:XmlHash $script:Sid U 'F:\RcloneCache\gmail-u' '\\dropbox-cloud\gmail-u'
        $result.Action.Execute | Should -Be 'C:\Owned\rclone.exe'
        $result.Tokens[3] | Should -Be 'U:'
    }
    It 'rejects changed serialized task bytes' {
        { Get-PcaiCloudMountTask 'Owned-Fixture' ('B' * 64) $script:Sid U 'F:\RcloneCache\gmail-u' '\\dropbox-cloud\gmail-u' } | Should -Throw '*XML changed*'
    }
    It 'rejects a live action differing from the hash-validated XML authority' {
        $script:TaskFixture.Actions[0].Arguments += ' --cache-dir C:\Unexpected'
        { Get-PcaiCloudMountTask 'Owned-Fixture' $script:XmlHash $script:Sid U 'F:\RcloneCache\gmail-u' '\\dropbox-cloud\gmail-u' } | Should -Throw '*semantics differ*'
    }
    It 'rejects a definition changed between export and live semantic snapshot' {
        $script:Exports = 0
        Mock Export-ScheduledTask { $script:Exports++; if ($script:Exports -eq 1) { $script:Xml } else { $script:Xml + '<!--changed-->' } }
        { Get-PcaiCloudMountTask 'Owned-Fixture' $script:XmlHash $script:Sid U 'F:\RcloneCache\gmail-u' '\\dropbox-cloud\gmail-u' } | Should -Throw '*changed during admission*'
    }
    It 'rejects a Highest provider task' {
        $script:TaskFixture.Principal.RunLevel = 'Highest'
        { Get-PcaiCloudMountTask 'Owned-Fixture' $script:XmlHash $script:Sid U 'F:\RcloneCache\gmail-u' '\\dropbox-cloud\gmail-u' } | Should -Throw '*principal*'
    }
    It 'rejects missing read-only flag' {
        $script:TaskFixture.Actions[0].Arguments = $script:TaskFixture.Actions[0].Arguments.Replace('--read-only', '')
        Update-OwnedTaskFixtureXml
        { Get-PcaiCloudMountTask 'Owned-Fixture' $script:XmlHash $script:Sid U 'F:\RcloneCache\gmail-u' '\\dropbox-cloud\gmail-u' } | Should -Throw '*read-only*'
    }
    It 'rejects a second read-only flag that overrides the first' {
        $script:TaskFixture.Actions[0].Arguments += ' --read-only=false'
        Update-OwnedTaskFixtureXml
        { Get-PcaiCloudMountTask 'Owned-Fixture' $script:XmlHash $script:Sid U 'F:\RcloneCache\gmail-u' '\\dropbox-cloud\gmail-u' } | Should -Throw '*read-only*'
    }
    It 'rejects a duplicate cache flag' {
        $script:TaskFixture.Actions[0].Arguments += ' --cache-dir C:\Wrong'
        Update-OwnedTaskFixtureXml
        { Get-PcaiCloudMountTask 'Owned-Fixture' $script:XmlHash $script:Sid U 'F:\RcloneCache\gmail-u' '\\dropbox-cloud\gmail-u' } | Should -Throw '*ambiguous*'
    }
    It 'rejects changed cache identity' {
        { Get-PcaiCloudMountTask 'Owned-Fixture' $script:XmlHash $script:Sid U 'F:\RcloneCache\foreign' '\\dropbox-cloud\gmail-u' } | Should -Throw '*binding differs*'
    }
    It 'rejects changed target drive' {
        { Get-PcaiCloudMountTask 'Owned-Fixture' $script:XmlHash $script:Sid V 'F:\RcloneCache\gmail-u' '\\dropbox-cloud\gmail-u' } | Should -Throw '*mount drive*'
    }
    It 'matches exact owner argv and rejects additional matching writers' {
        $binding = Get-PcaiCloudMountTask 'Owned-Fixture' $script:XmlHash $script:Sid U 'F:\RcloneCache\gmail-u' '\\dropbox-cloud\gmail-u'
        $script:ProcessFixture = New-CimInstance -ClassName Win32_Process -ClientOnly -Property @{
            ExecutablePath = 'C:\Owned\rclone.exe'; CommandLine = 'C:\Owned\rclone.exe ' + $binding.Action.Arguments
            ProcessId = [uint32]6000; CreationDate = [datetime]'2026-01-01T00:00:00Z'
        }
        Mock Get-CimInstance { $script:ProcessFixture }
        Mock Invoke-CimMethod { [pscustomobject]@{ ReturnValue = 0; Sid = $script:Sid } }
        Test-PcaiCloudMountOwner $binding $script:Sid | Should -BeTrue
        Mock Get-CimInstance { @($script:ProcessFixture, $script:ProcessFixture) }
        Test-PcaiCloudMountOwner $binding $script:Sid | Should -BeFalse
    }
    It 'rejects argv suffix differences despite matching executable and owner' {
        $binding = Get-PcaiCloudMountTask 'Owned-Fixture' $script:XmlHash $script:Sid U 'F:\RcloneCache\gmail-u' '\\dropbox-cloud\gmail-u'
        Mock Get-CimInstance { [pscustomobject]@{ ExecutablePath = 'C:\Owned\rclone.exe'; CommandLine = 'C:\Owned\rclone.exe ' + $script:TaskFixture.Actions[0].Arguments + ' --write-back-cache' } }
        Test-PcaiCloudMountOwner $binding $script:Sid | Should -BeFalse
    }
    It 'refuses a reused PID whose creation identity changes after owner lookup' {
        $binding = Get-PcaiCloudMountTask 'Owned-Fixture' $script:XmlHash $script:Sid U 'F:\RcloneCache\gmail-u' '\\dropbox-cloud\gmail-u'
        $script:OwnerQueries = 0
        Mock Get-CimInstance {
            $script:OwnerQueries++
            New-CimInstance -ClassName Win32_Process -ClientOnly -Property @{
                ProcessId = [uint32]6000; CreationDate = if ($script:OwnerQueries -eq 1) { [datetime]'2026-01-01T00:00:00Z' } else { [datetime]'2026-01-01T00:00:01Z' }
                ExecutablePath = 'C:\Owned\rclone.exe'; CommandLine = 'C:\Owned\rclone.exe ' + $script:TaskFixture.Actions[0].Arguments
            }
        }
        Mock Invoke-CimMethod { [pscustomobject]@{ ReturnValue = 0; Sid = $script:Sid } }
        Test-PcaiCloudMountOwner $binding $script:Sid | Should -BeFalse
        $script:PcaiCloudOwnerWitness | Should -BeNullOrEmpty
    }
}

Describe 'Native metadata backing proof' {
    It 'resolves an ordinary owned file and same-volume hardlink alias to the same GUID and full ID' {
        $file = Join-Path $TestDrive 'ordinary.bin'; $alias = Join-Path $TestDrive 'alias.bin'
        [IO.File]::WriteAllBytes($file,[byte[]]::new(32))
        New-Item -ItemType HardLink -Path $alias -Target $file | Out-Null
        $first = Get-PcaiCloudMountBackingLease $file
        $second = Get-PcaiCloudMountBackingLease $alias
        try {
            $first.VolumeGuid | Should -Be $second.VolumeGuid
            $first.FileId | Should -Be $second.FileId
            $first.FileId | Should -Match '^[A-F0-9]{48}$'
            $first.FinalPath | Should -Match '^\\\\\?\\Volume\{[a-f0-9-]{36}\}\\'
        } finally { $second.Dispose(); $first.Dispose() }
    }
    It 'rejects actual reparse ancestry before opening the backing leaf' {
        $real = Join-Path $TestDrive 'real'; $link = Join-Path $TestDrive 'junction'
        New-Item -ItemType Directory -Path $real | Out-Null
        [IO.File]::WriteAllBytes((Join-Path $real 'image.bin'),[byte[]]::new(32))
        New-Item -ItemType Junction -Path $link -Target $real | Out-Null
        { Get-PcaiCloudMountBackingLease (Join-Path $link 'image.bin') } | Should -Throw '*reparse point*'
    }
}

Describe 'Real child process deadline and stream bounds' {
    It 'validates a successful child snapshot with exact task and owner witness schema' {
        $root = Join-Path $TestDrive 'successful'; New-Item -ItemType Directory -Path $root | Out-Null
        $fixture = New-OwnedBackendFixture Success $root
        $result = Invoke-PcaiCloudMountWorker $fixture.Request -ApiFixturePath $fixture.Profile
        $result.State | Should -Be Qualified
        $result.Snapshot.OwnerWitness.ProcessId | Should -Be 6000
        $result.Snapshot.OwnerWitness.Sid | Should -Be $script:Sid
        $result.Snapshot.TaskHashBefore | Should -Be $fixture.Request.ExpectedTaskXmlSha256
        $result.Snapshot.TaskHashAfter | Should -Be $fixture.Request.ExpectedTaskXmlSha256
        $result.Snapshot.Storage.ActualBackingGuid | Should -Be $fixture.Request.BackingVolumeGuid
    }
    It 'acknowledges exactly one fake external scheduler request and qualifies its next snapshot' {
        $root = Join-Path $TestDrive 'start-happy'; New-Item -ItemType Directory -Path $root | Out-Null
        $fixture = New-OwnedBackendFixture StartHappy $root
        $fixture.Request.Operation = 'Start'
        $first = Invoke-PcaiCloudMountWorker $fixture.Request -ApiFixturePath $fixture.Profile
        $first.State | Should -Be Requested
        $first.StartRequested | Should -BeTrue
        $first.StartOutcome | Should -Be Acknowledged
        [IO.File]::ReadAllText($fixture.Marker) | Should -Be Start
        $fixture.Request.Operation = 'Snapshot'; $fixture.Request.RequestId = [guid]::NewGuid().ToString('D'); $fixture.Request.EndTicks = [Environment]::TickCount64 + 8000
        $second = Invoke-PcaiCloudMountWorker $fixture.Request -ApiFixturePath $fixture.Profile
        $second.State | Should -Be Qualified
        $second.StartRequested | Should -BeFalse
    }
    It 'bounds an actually blocked external <Mode> call under the same deadline' -ForEach @(
        @{Mode='Scheduler'},@{Mode='Wmi'},@{Mode='Compile'},@{Mode='Start'}) {
        $root = Join-Path $TestDrive $Mode; New-Item -ItemType Directory -Path $root | Out-Null
        $fixture = New-OwnedBackendFixture $Mode $root
        $timer = [Diagnostics.Stopwatch]::StartNew()
        $result = Invoke-PcaiCloudMountWorker $fixture.Request -ApiFixturePath $fixture.Profile
        $result.State | Should -Be Expired
        $timer.Elapsed.TotalSeconds | Should -BeLessThan 10
        Test-Path -LiteralPath $fixture.Marker | Should -BeTrue
        [IO.File]::ReadAllText($fixture.Marker) | Should -Be $Mode
        $result.StartOutcome | Should -Be $(if ($Mode -eq 'Start') { 'Unknown' } else { 'NotRequested' })
    }
    It 'enforces the streaming output cap before buffering a megabyte' {
        $root = Join-Path $TestDrive 'overflow'; New-Item -ItemType Directory -Path $root | Out-Null
        $fixture = New-OwnedBackendFixture Overflow $root
        { Invoke-PcaiCloudMountWorker $fixture.Request -ApiFixturePath $fixture.Profile } | Should -Throw '*stdout exceeds its output limit*'
    }
    It 'enforces the discarded error-stream cap while the process is still writing' {
        $root = Join-Path $TestDrive 'error-overflow'; New-Item -ItemType Directory -Path $root | Out-Null
        $fixture = New-OwnedBackendFixture ErrorOverflow $root
        { Invoke-PcaiCloudMountWorker $fixture.Request -ApiFixturePath $fixture.Profile } | Should -Throw '*stderr exceeds its output limit*'
    }
    It 'rejects non-JSON process output instead of accepting a hidden success response' {
        $root = Join-Path $TestDrive 'noise'; New-Item -ItemType Directory -Path $root | Out-Null
        $fixture = New-OwnedBackendFixture Noise $root
        { Invoke-PcaiCloudMountWorker $fixture.Request -ApiFixturePath $fixture.Profile } | Should -Throw '*JSON*'
    }
    It 'bounds draining inherited pipes after the backend process exits' {
        $root = Join-Path $TestDrive 'pipe'; New-Item -ItemType Directory -Path $root | Out-Null
        $fixture = New-OwnedBackendFixture Pipe $root
        try {
            $timer = [Diagnostics.Stopwatch]::StartNew()
            (Invoke-PcaiCloudMountWorker $fixture.Request -ApiFixturePath $fixture.Profile).State | Should -Be Expired
            $timer.Elapsed.TotalSeconds | Should -BeLessThan 10
            Test-Path -LiteralPath $fixture.ChildMarker | Should -BeTrue
        } finally {
            if (Test-Path -LiteralPath $fixture.ChildMarker) {
                $identity = [IO.File]::ReadAllText($fixture.ChildMarker) | ConvertFrom-Json
                $child = [Diagnostics.Process]::GetProcessById($identity.Pid)
                try {
                    $child.StartTime.ToUniversalTime().Ticks | Should -Be ([datetime]$identity.Creation).ToUniversalTime().Ticks
                    if (-not $child.HasExited) { $child.Kill($true); $child.WaitForExit(1000) | Should -BeTrue }
                } finally { $child.Dispose() }
            }
        }
    }
}
