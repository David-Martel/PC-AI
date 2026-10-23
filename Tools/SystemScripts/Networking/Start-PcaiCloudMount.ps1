#Requires -Version 7.3
<#
.SYNOPSIS
Admit an existing read-only cloud mount task after cache identity checks.
.DESCRIPTION
This launcher does not create, change, stop or unregister tasks. Pin the existing
task XML SHA256, user SID, provider UNC and cache directory during deployment.
R is deliberately excluded: it belongs to the dtm-work array. Each launch checks
the cache volume GUID, backing-volume GUID and actual VHD-to-partition association,
then rechecks admission immediately before starting the existing Limited task.
Storage probes run in bounded child processes. Access-denied or incomplete Storage
associations remain pending; there is no label-only or stale-receipt fallback.
Native mapping qualification does not establish cloud directory/read performance.
.PARAMETER Apply
Explicitly enable admission and task start. Default, DryRun and WhatIf return a
plan before querying Storage, tasks, mappings, compiling native code or writing.
.PARAMETER DryRun
Return a plan with zero IO and no task or process side effects.
.PARAMETER ValidateOnly
Perform actual runtime admission and qualification while prohibiting provider
task starts, even if the provider is unexpectedly absent.
.PARAMETER Help
Print usage. -h and --help are supported.
.EXAMPLE
./Start-PcaiCloudMount.ps1 -TaskName Rclone-Dropbox-Gmail-U -Drive U -DryRun
#>
[CmdletBinding(SupportsShouldProcess)]
param(
    [string]$TaskName,
    [ValidateSet('U', 'V')][string]$Drive = 'U',
    [string]$ProviderUnc,
    [string]$CacheDirectory,
    [string]$ExpectedUserSid,
    [string]$ExpectedTaskXmlSha256,
    [guid]$CacheVolumeGuid = 'c3e0daf0-3678-4842-8d9c-66aa3619b11a',
    [guid]$BackingVolumeGuid = 'b9f25542-491b-4718-8556-4cbe67e599b8',
    [string]$VhdPath = 'T:\vm\cloud-cache-disk.vhdx',
    [uint64]$MinimumCacheFreeBytes = 107374182400,
    [uint64]$MinimumBackingFreeBytes = 150323855360,
    [ValidateRange(1, 600)][int]$TimeoutSeconds = 180,
    [ValidateRange(10, 5000)][int]$PollMilliseconds = 1000,
    [switch]$Apply,
    [switch]$ValidateOnly,
    [switch]$DryRun,
    [Alias('h', '?')][switch]$Help,
    [Parameter(ValueFromRemainingArguments)][string[]]$CliArgs
)

$script:PcaiCloudMountSourceRoot = Split-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) -Parent
$script:PcaiCloudMountBackendSource = $PSCommandPath

function Get-PcaiCloudMountContext {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    [pscustomobject]@{
        Sid = $identity.User.Value
        Elevated = $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
        SessionId = [Diagnostics.Process]::GetCurrentProcess().SessionId
    }
}

function Get-PcaiCloudMountStorage {
    param([string]$ImagePath, [int]$BudgetMilliseconds)
    $endTicks = [Environment]::TickCount64 + $BudgetMilliseconds
    $context = Get-PcaiCloudMountContext
    $request = [pscustomobject]@{ RequestId = [guid]::NewGuid().ToString('D'); Operation = 'Storage'; EndTicks = $endTicks
        VhdPath = $ImagePath; ExpectedUserSid = $context.Sid; ExpectedSessionId = $context.SessionId; ExpectedElevated = $context.Elevated }
    $result = Invoke-PcaiCloudMountWorker $request
    if ($result.State -eq 'Observed') { return $result.Snapshot.Storage }
    return [pscustomobject]@{ State = 'Pending'; Reason = if ($result.State -eq 'Expired') { 'Storage probe deadline expired.' } else { 'Storage query failed; no identity fallback admitted.' } }
}
function Test-PcaiCloudMountAdmission {
    param($Storage, [guid]$CacheGuid, [guid]$BackingGuid, [uint64]$CacheReserve, [uint64]$BackingReserve)
    if ($Storage.State -eq 'InvalidIdentity') { throw "Cache identity refused: $($Storage.Reason)" }
    if ($Storage.State -ne 'Healthy') { return $false }
    if ([guid]$Storage.CacheGuid -ne $CacheGuid -or [guid]$Storage.BackingGuid -ne $BackingGuid -or
        $Storage.BackingFs -ne 'ReFS') { throw 'Permanent cache/backing identity differs.' }
    if (-not $Storage.PSObject.Properties['ActualBackingGuid'] -or [guid]$Storage.ActualBackingGuid -ne $BackingGuid) {
        throw 'Actual resolved VHD backing volume differs.'
    }
    if ([uint64]$Storage.CacheFree -lt $CacheReserve -or [uint64]$Storage.BackingFree -lt $BackingReserve) {
        throw 'Cache or backing reserve is below admission threshold.'
    }
    return $true
}

function Initialize-PcaiCloudMountNative {
    if (-not ('Pcai.CloudMountNative' -as [type])) {
        Add-Type -TypeDefinition @'
using System;
using System.Text;
using System.Runtime.InteropServices;
using System.IO;
using System.Collections.Generic;
using Microsoft.Win32.SafeHandles;
namespace Pcai {
public static class CloudMountNative {
 [DllImport("mpr.dll", CharSet=CharSet.Unicode)] public static extern int WNetGetConnection(string drive, StringBuilder name, ref int length);
 [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)] public static extern uint QueryDosDevice(string drive, StringBuilder name, int size);
 [DllImport("kernel32.dll", CharSet=CharSet.Unicode)] public static extern uint GetDriveType(string root);
 [DllImport("shell32.dll", CharSet=CharSet.Unicode, SetLastError=true)] static extern IntPtr CommandLineToArgvW(string command, out int count);
 [DllImport("kernel32.dll")] static extern IntPtr LocalFree(IntPtr memory);
 public static string[] Parse(string command) {
  int count; IntPtr memory=CommandLineToArgvW(command, out count);
  if(memory==IntPtr.Zero) throw new System.ComponentModel.Win32Exception();
  try { var result=new string[count]; for(int i=0;i<count;i++) result[i]=Marshal.PtrToStringUni(Marshal.ReadIntPtr(memory,i*IntPtr.Size)); return result; }
  finally { LocalFree(memory); }
 }
}
public sealed class CloudBackingLease : IDisposable {
 const uint ReadAttributes=0x80, SharedReadWrite=3, OpenExisting=3;
 const uint MetadataFlags=0x02300000, Reparse=0x400, DirectoryAttribute=0x10;
 [StructLayout(LayoutKind.Sequential)] struct TagInfo { public uint Attributes; public uint Tag; }
 [DllImport("kernel32.dll",CharSet=CharSet.Unicode,SetLastError=true)] static extern SafeFileHandle CreateFileW(string path,uint access,uint share,IntPtr security,uint creation,uint flags,IntPtr template);
 [DllImport("kernel32.dll",SetLastError=true)] static extern bool GetFileInformationByHandleEx(SafeFileHandle file,int kind,out TagInfo result,uint size);
 [DllImport("kernel32.dll",EntryPoint="GetFileInformationByHandleEx",SetLastError=true)] static extern bool GetId(SafeFileHandle file,int kind,IntPtr result,uint size);
 [DllImport("kernel32.dll",CharSet=CharSet.Unicode,SetLastError=true)] static extern uint GetFinalPathNameByHandleW(SafeFileHandle file,StringBuilder result,uint size,uint flags);
 readonly List<SafeFileHandle> handles=new List<SafeFileHandle>();
 public string VolumeGuid { get; private set; }
 public string FinalPath { get; private set; }
 public string FileId { get; private set; }
 public static CloudBackingLease Open(string path) {
  if(!Path.IsPathFullyQualified(path) || path.StartsWith(@"\\") || path.Length<4 || path[1]!=':') throw new ArgumentException("Backing path must be a fully qualified local file.");
  string full=Path.GetFullPath(path);
  if(!String.Equals(full,path,StringComparison.OrdinalIgnoreCase)) throw new ArgumentException("Backing path contains a noncanonical alias.");
  var lease=new CloudBackingLease();
  try {
   string root=Path.GetPathRoot(full); string relative=full.Substring(root.Length);
   string[] pieces=relative.Split('\\');
   if(pieces.Length==0 || Array.Exists(pieces,p=>p.Length==0)) throw new ArgumentException("Backing file components are incomplete.");
   lease.Add(root,true);
   string current=root;
   for(int i=0;i<pieces.Length;i++) {current=Path.Combine(current,pieces[i]);lease.Add(current,i<pieces.Length-1);}
   var file=lease.handles[lease.handles.Count-1];
   var name=new StringBuilder(32768);
   uint count=GetFinalPathNameByHandleW(file,name,(uint)name.Capacity,1);
   if(count==0) throw new System.ComponentModel.Win32Exception();
   if(count>=name.Capacity) throw new IOException("Backing final path exceeds the supported limit.");
   lease.FinalPath=name.ToString();
   const string prefix=@"\\?\Volume{";
   if(!lease.FinalPath.StartsWith(prefix,StringComparison.OrdinalIgnoreCase) || lease.FinalPath.Length<49 || lease.FinalPath[47]!='}' || lease.FinalPath[48]!='\\') throw new IOException("Backing final path has no volume GUID.");
   lease.VolumeGuid=Guid.Parse(lease.FinalPath.Substring(prefix.Length,36)).ToString("D");
   IntPtr id=Marshal.AllocHGlobal(24);
   try {
    if(!GetId(file,18,id,24)) throw new System.ComponentModel.Win32Exception();
    byte[] data=new byte[24];Marshal.Copy(id,data,0,data.Length);
    lease.FileId=BitConverter.ToString(data).Replace("-","");
   } finally {Marshal.FreeHGlobal(id);}
   return lease;
  } catch {lease.Dispose();throw;}
 }
 void Add(string path,bool directory) {
  var file=CreateFileW(path,ReadAttributes,SharedReadWrite,IntPtr.Zero,OpenExisting,MetadataFlags,IntPtr.Zero);
  if(file.IsInvalid) {int error=Marshal.GetLastWin32Error();file.Dispose();throw new System.ComponentModel.Win32Exception(error);}
  handles.Add(file);
  TagInfo info;
  if(!GetFileInformationByHandleEx(file,9,out info,8)) throw new System.ComponentModel.Win32Exception();
  if((info.Attributes&Reparse)!=0) throw new InvalidDataException("Backing path traverses a reparse point.");
  if(((info.Attributes&DirectoryAttribute)!=0)!=directory) throw new InvalidDataException("Backing path component type differs.");
 }
 public void Dispose() {for(int i=handles.Count-1;i>=0;i--) handles[i].Dispose();handles.Clear();}
}
}
'@ -ErrorAction Stop
    }
}

function Get-PcaiCloudMountMapping {
    param([string]$Letter)
    Initialize-PcaiCloudMountNative
    $unc = [Text.StringBuilder]::new(32768)
    $length = $unc.Capacity
    $net = [Pcai.CloudMountNative]::WNetGetConnection("${Letter}:", $unc, [ref]$length)
    $dos = [Text.StringBuilder]::new(32768)
    $count = [Pcai.CloudMountNative]::QueryDosDevice("${Letter}:", $dos, $dos.Capacity)
    $dosError = if ($count -eq 0) { [Runtime.InteropServices.Marshal]::GetLastWin32Error() } else { 0 }
    [pscustomobject]@{ Unc = $unc.ToString(); NetResult = $net; DosResult = $count; DosError = $dosError;
        DriveType = [Pcai.CloudMountNative]::GetDriveType("${Letter}:\") }
}

function New-PcaiCloudMountPendingError {
    param([string]$Message, [Exception]$Cause)
    $pendingError = [InvalidOperationException]::new($Message, $Cause)
    $pendingError.Data['PcaiPending'] = $true
    return $pendingError
}

function Get-PcaiCloudMountTask {
    param([string]$Name, [string]$Hash, [string]$Sid, [string]$Letter, [string]$Cache, [string]$Unc)
    try { $xml = Export-ScheduledTask -TaskName $Name -ErrorAction Stop }
    catch { throw (New-PcaiCloudMountPendingError 'Task definition query is pending.' $_.Exception) }
    $actualHash = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData([Text.Encoding]::UTF8.GetBytes($xml)))
    if ($actualHash -ne $Hash) { throw 'Registered task XML changed; preserved.' }
    $document = [xml]::new()
    $readerSettings = [Xml.XmlReaderSettings]::new()
    $readerSettings.DtdProcessing = [Xml.DtdProcessing]::Prohibit
    $readerSettings.XmlResolver = $null
    $readerSettings.MaxCharactersInDocument = 1048576
    $reader = [Xml.XmlReader]::Create([IO.StringReader]::new($xml), $readerSettings)
    try { $document.Load($reader) } finally { $reader.Dispose() }
    $namespaces = [Xml.XmlNamespaceManager]::new($document.NameTable)
    $namespaces.AddNamespace('t', 'http://schemas.microsoft.com/windows/2004/02/mit/task')
    $principalXml = @($document.SelectNodes('/t:Task/t:Principals/t:Principal', $namespaces))
    $actionsXml = @($document.SelectNodes('/t:Task/t:Actions/*', $namespaces))
    if ($principalXml.Count -ne 1 -or $actionsXml.Count -ne 1 -or $actionsXml[0].LocalName -ne 'Exec' -or
        $document.SelectSingleNode('/t:Task/t:Actions', $namespaces).GetAttribute('Context') -ne $principalXml[0].GetAttribute('id')) {
        throw 'Pinned task principal/action definition is ambiguous.'
    }
    $xmlUser = $principalXml[0].SelectSingleNode('t:UserId', $namespaces).InnerText
    $xmlLogon = $principalXml[0].SelectSingleNode('t:LogonType', $namespaces).InnerText
    $runLevelXml = $principalXml[0].SelectSingleNode('t:RunLevel', $namespaces)
    if ($xmlLogon -ne 'InteractiveToken' -or ($runLevelXml -and $runLevelXml.InnerText -ne 'LeastPrivilege')) { throw 'Pinned provider principal is not Limited Interactive.' }
    $xmlOwner = if ($xmlUser -match '^S-1-') { [Security.Principal.SecurityIdentifier]::new($xmlUser).Value } else {
        [Security.Principal.NTAccount]::new($xmlUser).Translate([Security.Principal.SecurityIdentifier]).Value
    }
    if ($xmlOwner -ne $Sid) { throw 'Pinned task belongs to another user.' }
    $action = [pscustomobject]@{
        Execute = $actionsXml[0].SelectSingleNode('t:Command', $namespaces).InnerText
        Arguments = $actionsXml[0].SelectSingleNode('t:Arguments', $namespaces).InnerText
    }
    $enabledXml = $document.SelectSingleNode('/t:Task/t:Settings/t:Enabled', $namespaces)
    $instancesXml = $document.SelectSingleNode('/t:Task/t:Settings/t:MultipleInstancesPolicy', $namespaces)
    if (($enabledXml -and -not [Xml.XmlConvert]::ToBoolean($enabledXml.InnerText)) -or
        ($instancesXml -and $instancesXml.InnerText -ne 'IgnoreNew')) { throw 'Pinned task settings are unsuitable.' }
    try { $task = Get-ScheduledTask -TaskName $Name -ErrorAction Stop }
    catch { throw (New-PcaiCloudMountPendingError 'Task state query is pending.' $_.Exception) }
    if (@($task.Actions).Count -ne 1 -or $task.Principal.RunLevel -ne 'Limited' -or
        $task.Principal.LogonType -ne 'Interactive' -or -not $task.Settings.Enabled -or
        $task.Settings.MultipleInstances -ne 'IgnoreNew') { throw 'Task principal, instance policy or action is unsuitable.' }
    $owner = [Security.Principal.NTAccount]::new($task.Principal.UserId).Translate([Security.Principal.SecurityIdentifier]).Value
    if ($owner -ne $Sid) { throw 'Task belongs to another user.' }
    if ($task.Actions[0].Execute -ine $action.Execute -or $task.Actions[0].Arguments -cne $action.Arguments) {
        throw 'Live task semantics differ from pinned XML; preserved.'
    }
    try { $afterXml = Export-ScheduledTask -TaskName $Name -ErrorAction Stop }
    catch { throw (New-PcaiCloudMountPendingError 'Task definition recheck is pending.' $_.Exception) }
    $afterHash = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData([Text.Encoding]::UTF8.GetBytes($afterXml)))
    if ($afterHash -ne $Hash) { throw 'Task XML changed during admission; preserved.' }
    $tokens = [Pcai.CloudMountNative]::Parse('rclone.exe ' + $action.Arguments)
    if ($tokens.Count -lt 4 -or $tokens[1] -ne 'mount' -or $tokens[3] -ne "${Letter}:") { throw 'Task is not the expected mount drive.' }
    if (@($tokens | Where-Object { $_ -match '^--read-only(?:=|$)' }).Count -ne 1 -or
        $tokens -notcontains '--read-only') { throw 'Task is not unambiguously read-only.' }
    foreach ($pair in @(@('--cache-dir', $Cache), @('--volname', $Unc))) {
        if (@($tokens | Where-Object { $_ -eq $pair[0] -or $_.StartsWith($pair[0] + '=') }).Count -ne 1) { throw 'Task cache/provider flag is ambiguous.' }
        $index = [Array]::IndexOf($tokens, $pair[0])
        if ($index -lt 0 -or $index + 1 -ge $tokens.Count -or $tokens[$index + 1] -ne $pair[1]) { throw 'Task cache/provider binding differs.' }
    }
    [pscustomobject]@{ Task = $task; Action = $action; Tokens = $tokens; HashBefore = $actualHash; HashAfter = $afterHash }
}

function Test-PcaiCloudMountOwner {
    param($TaskBinding, [string]$Sid)
    $script:PcaiCloudOwnerWitness = $null
    $matches = [Collections.Generic.List[object]]::new()
    try { $processes = @(Get-CimInstance Win32_Process -Filter "Name='rclone.exe'" -ErrorAction Stop) }
    catch { throw (New-PcaiCloudMountPendingError 'Owner/process inventory is pending.' $_.Exception) }
    foreach ($process in $processes) {
        if ([string]$process.ExecutablePath -ne [string]$TaskBinding.Action.Execute -or -not $process.CommandLine) { continue }
        $actual = [Pcai.CloudMountNative]::Parse($process.CommandLine)
        if ($actual.Count -ne $TaskBinding.Tokens.Count) { continue }
        $equal = $true
        for ($i = 1; $i -lt $actual.Count; $i++) { if ($actual[$i] -cne $TaskBinding.Tokens[$i]) { $equal = $false; break } }
        if (-not $equal) { continue }
        try { $owner = Invoke-CimMethod -InputObject $process -MethodName GetOwnerSid -ErrorAction Stop }
        catch { throw (New-PcaiCloudMountPendingError 'Process owner query is pending.' $_.Exception) }
        if ($owner.ReturnValue -eq 0 -and $owner.Sid -eq $Sid -and $process.ProcessId -gt 0 -and $process.CreationDate) {
            try { $fresh = @(Get-CimInstance Win32_Process -Filter "ProcessId=$($process.ProcessId)" -ErrorAction Stop) }
            catch { throw (New-PcaiCloudMountPendingError 'Process identity recheck is pending.' $_.Exception) }
            if ($fresh.Count -eq 1 -and $fresh[0].ProcessId -eq $process.ProcessId -and
                ([datetime]$fresh[0].CreationDate).ToUniversalTime().Ticks -eq ([datetime]$process.CreationDate).ToUniversalTime().Ticks -and
                $fresh[0].ExecutablePath -eq $process.ExecutablePath -and $fresh[0].CommandLine -ceq $process.CommandLine) { $matches.Add($fresh[0]) }
        }
    }
    $qualified = $matches.Count -eq 1 -and $TaskBinding.Task.State -eq 'Running'
    if ($qualified) {
        $script:PcaiCloudOwnerWitness = [pscustomobject]@{ ProcessId = $matches[0].ProcessId;
            CreationUtc = ([datetime]$matches[0].CreationDate).ToUniversalTime().ToString('o'); Sid = $Sid }
    }
    return $qualified
}

function Get-PcaiCloudMountBackingLease {
    param([string]$Path)
    Initialize-PcaiCloudMountNative
    return [Pcai.CloudBackingLease]::Open($Path)
}

function Get-PcaiCloudMountStorageObservation {
    param([string]$ImagePath)
    $helper = Join-Path $script:PcaiCloudMountSourceRoot 'Tools/Repair-CloudCacheMount.ps1'
    if ((Get-FileHash -LiteralPath $helper -Algorithm SHA256 -ErrorAction Stop).Hash -ne
        '1F7E3876EDED10CD6ED9B7F974F30D3AEC02391AF237524633A445D14A8825B6') { throw 'Reviewed helper bytes changed.' }
    . $helper
    $lease = Get-PcaiCloudMountBackingLease $ImagePath
    try {
        $state = Get-CacheVolumeState -Label cloud-cache-disk -ExpectedVhd $ImagePath -ExpectedDriveLetter F
        $backing = @(Get-Volume -DriveLetter T -ErrorAction Stop)
        if ($backing.Count -ne 1) { throw 'Backing volume is ambiguous.' }
        [pscustomobject]@{ State = $state.State; Reason = $state.Reason; CacheGuid = $state.Identity
            CacheFree = if ($state.Volume) { $state.Volume.SizeRemaining } else { 0 }
            BackingGuid = Get-CacheVolumeIdentity $backing[0]; BackingFs = $backing[0].FileSystem
            BackingFree = $backing[0].SizeRemaining; ActualBackingGuid = $lease.VolumeGuid
            BackingFileId = $lease.FileId; BackingFinalPath = $lease.FinalPath }
    } finally { $lease.Dispose() }
}

function Get-PcaiCloudMountSnapshot {
    param($Request)
    $helper = Join-Path $script:PcaiCloudMountSourceRoot 'Tools/Repair-CloudCacheMount.ps1'
    if ((Get-FileHash -LiteralPath $helper -Algorithm SHA256 -ErrorAction Stop).Hash -ne
        '1F7E3876EDED10CD6ED9B7F974F30D3AEC02391AF237524633A445D14A8825B6') { throw 'Reviewed helper bytes changed.' }
    . $helper
    Initialize-PcaiCloudMountNative
    $lease = Get-PcaiCloudMountBackingLease $Request.VhdPath
    try {
        if ([guid]$lease.VolumeGuid -ne [guid]$Request.BackingVolumeGuid) { throw 'Actual resolved VHD backing volume differs.' }
        try { $beforeImage = @(Get-DiskImage -ImagePath $Request.VhdPath -ErrorAction Stop) }
        catch { throw (New-PcaiCloudMountPendingError 'Backing image query is pending.' $_.Exception) }
        if ($beforeImage.Count -eq 0 -or ($beforeImage.Count -eq 1 -and -not $beforeImage[0].Attached)) { return [pscustomobject]@{ State = 'Pending'; Reason = 'Backing image is not yet attached.' } }
        if ($beforeImage.Count -ne 1 -or $beforeImage[0].ImagePath -ne $Request.VhdPath) { throw 'Backing image identity is ambiguous or mismatched.' }
        $beforeNumber = Get-CacheImageDiskNumber $beforeImage[0]
        if ($null -eq $beforeNumber) { return [pscustomobject]@{ State = 'Pending'; Reason = 'Image disk association is incomplete.' } }
        $state = Get-CacheVolumeState -Label cloud-cache-disk -ExpectedVhd $Request.VhdPath -ExpectedDriveLetter F
        try { $backing = @(Get-Volume -DriveLetter T -ErrorAction Stop) }
        catch { throw (New-PcaiCloudMountPendingError 'Backing volume query is pending.' $_.Exception) }
        if ($backing.Count -eq 0) { return [pscustomobject]@{ State = 'Pending'; Reason = 'Backing volume is unavailable.' } }
        if ($backing.Count -ne 1) { throw 'Backing volume is ambiguous.' }
        $storage = [pscustomobject]@{
            State = $state.State; Reason = $state.Reason; CacheGuid = $state.Identity
            CacheFree = if ($state.Volume) { $state.Volume.SizeRemaining } else { 0 }
            BackingGuid = Get-CacheVolumeIdentity $backing[0]; BackingFs = $backing[0].FileSystem
            BackingFree = $backing[0].SizeRemaining; ActualBackingGuid = $lease.VolumeGuid
            BackingFileId = $lease.FileId; BackingFinalPath = $lease.FinalPath
        }
        $admissionParameters = @{ Storage = $storage; CacheGuid = [guid]$Request.CacheVolumeGuid; BackingGuid = [guid]$Request.BackingVolumeGuid
            CacheReserve = [uint64]$Request.MinimumCacheFreeBytes; BackingReserve = [uint64]$Request.MinimumBackingFreeBytes }
        if (-not (Test-PcaiCloudMountAdmission @admissionParameters)) {
            return [pscustomobject]@{ State = 'Pending'; Storage = $storage }
        }
        $mapping = Get-PcaiCloudMountMapping $Request.Drive
        $binding = Get-PcaiCloudMountTask $Request.TaskName $Request.ExpectedTaskXmlSha256 $Request.ExpectedUserSid $Request.Drive $Request.CacheDirectory $Request.ProviderUnc
        try { $afterImage = @(Get-DiskImage -ImagePath $Request.VhdPath -ErrorAction Stop) }
        catch { throw (New-PcaiCloudMountPendingError 'Image association recheck is pending.' $_.Exception) }
        if ($afterImage.Count -eq 0 -or ($afterImage.Count -eq 1 -and -not $afterImage[0].Attached)) { return [pscustomobject]@{ State = 'Pending'; Reason = 'Image detached during admission.' } }
        if ($afterImage.Count -ne 1 -or $afterImage[0].ImagePath -ne $Request.VhdPath) { throw 'Image identity changed during admission.' }
        if ((Get-CacheImageDiskNumber $afterImage[0]) -ne $beforeNumber) { return [pscustomobject]@{ State = 'Pending'; Reason = 'Image disk association changed during admission.' } }
        $secondLease = Get-PcaiCloudMountBackingLease $Request.VhdPath
        try {
            if ($secondLease.VolumeGuid -ne $lease.VolumeGuid -or $secondLease.FileId -ne $lease.FileId) { throw 'Resolved backing file changed during admission.' }
        } finally { $secondLease.Dispose() }
        if ($mapping.NetResult -eq 0 -and $mapping.Unc -eq $Request.ProviderUnc -and $mapping.DriveType -eq 4) {
            if (-not (Test-PcaiCloudMountOwner $binding $Request.ExpectedUserSid)) { throw 'Provider has no unique matching live owner/task process; preserved.' }
            return [pscustomobject]@{ State = 'Qualified'; Storage = $storage; TaskState = [string]$binding.Task.State;
                TaskHashBefore = $binding.HashBefore; TaskHashAfter = $binding.HashAfter; Mapping = $mapping; OwnerQualified = $true;
                OwnerWitness = $script:PcaiCloudOwnerWitness }
        }
        if ($mapping.DosResult -ne 0 -or $mapping.DosError -ne 2 -or $mapping.NetResult -ne 2250) { throw 'Drive collision or inconclusive mapping query; preserved.' }
        [pscustomobject]@{ State = if ($binding.Task.State -eq 'Running') { 'Pending' } else { 'Ready' };
            Storage = $storage; TaskState = [string]$binding.Task.State; Mapping = $mapping;
            TaskHashBefore = $binding.HashBefore; TaskHashAfter = $binding.HashAfter; OwnerQualified = $false }
    } finally { $lease.Dispose() }
}

function Invoke-PcaiCloudMountBackend {
    param($Request)
    Set-StrictMode -Version Latest
    $ErrorActionPreference = 'Stop'
    $context = Get-PcaiCloudMountContext
    $result = [ordered]@{ SchemaVersion = 1; RequestId = $Request.RequestId; WorkerPid = $PID;
        WorkerCreationUtc = [Diagnostics.Process]::GetCurrentProcess().StartTime.ToUniversalTime().ToString('o');
        Sid = $context.Sid; SessionId = $context.SessionId; Elevated = $context.Elevated;
        EndTicks = [long]$Request.EndTicks; Operation = $Request.Operation; State = 'Pending';
        StartRequested = $false; StartOutcome = 'NotRequested'; Snapshot = $null; Reason = '' }
    try {
        if ($context.Sid -ne $Request.ExpectedUserSid -or $context.SessionId -ne $Request.ExpectedSessionId -or
            $context.Elevated -ne $Request.ExpectedElevated -or ($Request.Operation -ne 'Storage' -and $context.Elevated)) { throw 'Backend owner/session context differs.' }
        if ([Environment]::TickCount64 -ge $Request.EndTicks) { throw [TimeoutException]::new('Admission deadline expired before observation.') }
        if ($Request.Operation -eq 'Storage') {
            $result.Snapshot = [pscustomobject]@{ Storage = Get-PcaiCloudMountStorageObservation $Request.VhdPath }
            if ([Environment]::TickCount64 -ge $Request.EndTicks) { throw [TimeoutException]::new('Storage response arrived after deadline.') }
            $result.State = 'Observed'
            return [pscustomobject]$result
        }
        $result.Snapshot = Get-PcaiCloudMountSnapshot $Request
        if ([Environment]::TickCount64 -ge $Request.EndTicks) { throw [TimeoutException]::new('Admission deadline expired after observation.') }
        $result.State = $result.Snapshot.State
        if ($result.State -eq 'Pending' -and $result.Snapshot.PSObject.Properties['Reason']) { $result.Reason = $result.Snapshot.Reason }
        if ($Request.Operation -eq 'Start' -and $result.State -eq 'Ready') {
            # Recheck every identity immediately before the start request. The
            # scheduler has no atomic start-by-definition-hash API; task-definition
            # editing must remain under deployment custody during admission.
            $startLease = Get-PcaiCloudMountBackingLease $Request.VhdPath
            try {
                $result.Snapshot = Get-PcaiCloudMountSnapshot $Request
                if ($Request.ExpectedBackingFileId -and $result.Snapshot.Storage.BackingFileId -ne $Request.ExpectedBackingFileId) { throw 'Backing file changed between admission and launch.' }
                if ([Environment]::TickCount64 -ge $Request.EndTicks) { throw [TimeoutException]::new('Admission deadline expired before start.') }
                $result.State = $result.Snapshot.State
                if ($result.State -eq 'Ready') {
                    $result.StartRequested = $true
                    $result.StartOutcome = 'Unknown'
                    Start-ScheduledTask -TaskName $Request.TaskName -ErrorAction Stop
                    if ([Environment]::TickCount64 -ge $Request.EndTicks) { throw [TimeoutException]::new('Scheduler response arrived after admission deadline.') }
                    $result.StartOutcome = 'Acknowledged'
                    $result.State = 'Requested'
                }
            } finally { $startLease.Dispose() }
        }
    } catch {
        $result.Reason = $_.Exception.Message
        $cause = $_.Exception; $pending = $false
        while ($cause) {
            if ($cause.Data.Contains('PcaiPending') -or ($cause -is [ComponentModel.Win32Exception] -and $cause.NativeErrorCode -in @(2,3,5,32))) { $pending = $true }
            $cause = $cause.InnerException
        }
        $result.State = if ($result.StartRequested) { 'Unknown' } elseif ($_.Exception -is [TimeoutException]) { 'Expired' } elseif ($pending) { 'Pending' } else { 'Denied' }
    }
    [pscustomobject]$result
}

function Invoke-PcaiCloudMountWorker {
    param($Request, [string]$ApiFixturePath)
    # ApiFixturePath is an internal test adapter, absent from the public launcher.
    # It replaces external APIs in an isolated child, never the admission logic.
    $source = $script:PcaiCloudMountBackendSource
    $sourceHash = (Get-FileHash -LiteralPath $source -Algorithm SHA256 -ErrorAction Stop).Hash
    $sourceLiteral = $source.Replace("'", "''")
    $fixtureCode = if ($ApiFixturePath) { ". '" + $ApiFixturePath.Replace("'", "''") + "'" } else { '' }
    $json = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes(($Request | ConvertTo-Json -Compress -Depth 5)))
    $code = @"
`$ErrorActionPreference='Stop'
if((Get-FileHash -LiteralPath '$sourceLiteral' -Algorithm SHA256).Hash -ne '$sourceHash'){throw 'Backend source changed.'}
. '$sourceLiteral'
$fixtureCode
`$request=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('$json'))|ConvertFrom-Json
Invoke-PcaiCloudMountBackend `$request|ConvertTo-Json -Compress -Depth 8
"@
    $remaining = [int][Math]::Max(0, [Math]::Min([int]::MaxValue, $Request.EndTicks - [Environment]::TickCount64))
    if ($remaining -le 0) { return [pscustomobject]@{ State = 'Expired'; StartOutcome = 'NotRequested'; Reason = 'Deadline expired before worker start.' } }
    $start = [Diagnostics.ProcessStartInfo]::new((Join-Path $PSHOME 'pwsh.exe'))
    $start.UseShellExecute = $false; $start.CreateNoWindow = $true
    $start.RedirectStandardOutput = $true; $start.RedirectStandardError = $true
    foreach ($arg in @('-NoLogo', '-NoProfile', '-NonInteractive', '-EncodedCommand', [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($code)))) { $start.ArgumentList.Add($arg) }
    $child = [Diagnostics.Process]::new(); $child.StartInfo = $start
    try {
        if (-not $child.Start()) { throw 'Admission worker could not start.' }
        $outBuffer = [char[]]::new(4096); $errBuffer = [char[]]::new(4096)
        $output = [Text.StringBuilder]::new(4096); $errorCount = 0
        $outRead = $child.StandardOutput.ReadAsync($outBuffer, 0, $outBuffer.Length)
        $errRead = $child.StandardError.ReadAsync($errBuffer, 0, $errBuffer.Length)
        $outEnded = $false; $errEnded = $false; $complete = $false
        while ([Environment]::TickCount64 -lt $Request.EndTicks) {
            if (-not $outEnded -and $outRead.IsCompleted) {
                $count = $outRead.GetAwaiter().GetResult()
                if ($count -eq 0) { $outEnded = $true } else {
                    if ($output.Length + $count -gt 65536) { throw 'Admission worker stdout exceeds its output limit.' }
                    $null = $output.Append($outBuffer, 0, $count)
                    $outRead = $child.StandardOutput.ReadAsync($outBuffer, 0, $outBuffer.Length)
                }
            }
            if (-not $errEnded -and $errRead.IsCompleted) {
                $count = $errRead.GetAwaiter().GetResult()
                if ($count -eq 0) { $errEnded = $true } else {
                    $errorCount += $count
                    if ($errorCount -gt 8192) { throw 'Admission worker stderr exceeds its output limit.' }
                    $errRead = $child.StandardError.ReadAsync($errBuffer, 0, $errBuffer.Length)
                }
            }
            if ($child.HasExited -and $outEnded -and $errEnded) { $complete = $true; break }
            $remaining = [int][Math]::Max(0, $Request.EndTicks - [Environment]::TickCount64)
            if ($remaining -gt 0) { [Threading.Thread]::Sleep([Math]::Min(10, $remaining)) }
        }
        if (-not $complete) {
            if (-not $child.HasExited) {
                $child.Kill($true)
                if (-not $child.WaitForExit(1000)) { throw 'Owned worker cancellation could not be confirmed.' }
            }
            return [pscustomobject]@{ State = 'Expired'; StartOutcome = if ($Request.Operation -eq 'Start') { 'Unknown' } else { 'NotRequested' }; Reason = 'Admission worker deadline expired.' }
        }
        if ([Environment]::TickCount64 -ge $Request.EndTicks) { return [pscustomobject]@{ State = 'Expired'; StartOutcome = if ($Request.Operation -eq 'Start') { 'Unknown' } else { 'NotRequested' }; Reason = 'Worker completed after deadline.' } }
        if ($child.ExitCode -ne 0) { throw 'Admission worker failed before a qualified response.' }
        $raw = $output.ToString()
        $result = $raw | ConvertFrom-Json -ErrorAction Stop
        foreach ($field in @('SchemaVersion','RequestId','WorkerPid','WorkerCreationUtc','Sid','SessionId','Elevated','EndTicks','Operation','State','StartRequested','StartOutcome','Snapshot','Reason')) {
            if (-not $result.PSObject.Properties[$field]) { throw 'Admission worker response is incomplete.' }
        }
        if ($result.SchemaVersion -ne 1 -or $result.RequestId -ne $Request.RequestId -or $result.WorkerPid -ne $child.Id -or
            $result.Sid -ne $Request.ExpectedUserSid -or $result.SessionId -ne $Request.ExpectedSessionId -or $result.Elevated -ne $Request.ExpectedElevated -or
            $result.EndTicks -ne $Request.EndTicks -or $result.Operation -ne $Request.Operation -or
            $result.State -notin @('Qualified','Ready','Pending','Requested','Expired','Denied','Unknown','Observed') -or
            $result.Elevated -isnot [bool] -or $result.StartRequested -isnot [bool] -or
            $result.StartOutcome -notin @('NotRequested','Acknowledged','Unknown') -or
            ([datetime]$result.WorkerCreationUtc).ToUniversalTime().Ticks -ne $child.StartTime.ToUniversalTime().Ticks) { throw 'Admission worker response identity/schema differs.' }
        if ($result.State -in @('Qualified','Ready','Requested','Observed') -and -not $result.Snapshot) { throw 'Worker response has no admission snapshot.' }
        return $result
    } catch {
        if (-not $child.HasExited) { $child.Kill($true); if (-not $child.WaitForExit(1000)) { throw 'Owned worker cancellation could not be confirmed.' } }
        if ($Request.Operation -eq 'Start') { return [pscustomobject]@{ State = 'Unknown'; StartOutcome = 'Unknown'; Reason = 'Scheduler response failed qualification; request outcome is unknown.' } }
        throw
    } finally { $child.Dispose() }
}

function Invoke-PcaiCloudMount {
    [CmdletBinding(SupportsShouldProcess)]
    param([string]$TaskName, [ValidateSet('U', 'V')][string]$Drive = 'U', [string]$ProviderUnc,
        [string]$CacheDirectory, [string]$ExpectedUserSid, [string]$ExpectedTaskXmlSha256,
        [guid]$CacheVolumeGuid = 'c3e0daf0-3678-4842-8d9c-66aa3619b11a',
        [guid]$BackingVolumeGuid = 'b9f25542-491b-4718-8556-4cbe67e599b8',
        [string]$VhdPath = 'T:\vm\cloud-cache-disk.vhdx',
        [uint64]$MinimumCacheFreeBytes = 107374182400, [uint64]$MinimumBackingFreeBytes = 150323855360,
        [ValidateRange(1, 600)][int]$TimeoutSeconds = 180, [ValidateRange(10, 5000)][int]$PollMilliseconds = 1000,
        [switch]$Apply, [switch]$ValidateOnly, [switch]$DryRun, [Alias('h', '?')][switch]$Help,
        [Parameter(ValueFromRemainingArguments)][string[]]$CliArgs)
    Set-StrictMode -Version Latest
    $ErrorActionPreference = 'Stop'
    $CliArgs = @($CliArgs | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    if ($Help -or '--help' -in $CliArgs) { return 'Use -Apply with reviewed task XML SHA256, user SID, provider UNC and F cache path; -DryRun performs zero IO.' }
    if (@($CliArgs | Where-Object { $_ -notin @('--help', '--DryRun') }).Count) { throw 'Unknown arguments.' }
    if (-not $Apply -or $DryRun -or '--DryRun' -in $CliArgs -or $WhatIfPreference) {
        return [pscustomobject]@{ State = 'Plan'; Task = $TaskName; Drive = $Drive; SideEffects = $false }
    }
    if ($TaskName -notmatch '^[A-Za-z0-9][A-Za-z0-9_. -]{0,127}$' -or
        $ExpectedTaskXmlSha256 -notmatch '^[A-Fa-f0-9]{64}$' -or $ExpectedUserSid -notmatch '^S-1-5-21-(\d+-){3}\d+$' -or
        $ProviderUnc -notmatch '^\\\\[^\\]+\\[^\\]+$' -or $CacheDirectory -notmatch '^F:\\RcloneCache\\[A-Za-z0-9][A-Za-z0-9_-]*$' -or
        $VhdPath -ne 'T:\vm\cloud-cache-disk.vhdx' -or $CacheVolumeGuid -eq [guid]::Empty -or $BackingVolumeGuid -eq [guid]::Empty) {
        throw 'Deployment bindings are missing or malformed.'
    }
    $context = Get-PcaiCloudMountContext
    if ($context.Sid -ne $ExpectedUserSid -or $context.Elevated -or $context.SessionId -eq 0) { throw 'Run in the exact Limited interactive owner session.' }
    $beginTicks = [Environment]::TickCount64
    $endTicks = $beginTicks + [long]$TimeoutSeconds * 1000
    $mutex = [Threading.Mutex]::new($false, "Local\PcaiCloudMount-$ExpectedUserSid-$Drive")
    $owned = $false
    $started = $false
    $uncertain = $false
    $lastReason = ''
    $request = [pscustomobject]@{
        RequestId = ''; Operation = 'Snapshot'; EndTicks = $endTicks; ExpectedSessionId = $context.SessionId; ExpectedElevated = $false
        TaskName = $TaskName; Drive = $Drive; ProviderUnc = $ProviderUnc; CacheDirectory = $CacheDirectory
        ExpectedUserSid = $ExpectedUserSid; ExpectedTaskXmlSha256 = $ExpectedTaskXmlSha256
        CacheVolumeGuid = $CacheVolumeGuid.ToString('D'); BackingVolumeGuid = $BackingVolumeGuid.ToString('D')
        VhdPath = $VhdPath; MinimumCacheFreeBytes = $MinimumCacheFreeBytes; MinimumBackingFreeBytes = $MinimumBackingFreeBytes
        ExpectedBackingFileId = ''
    }
    try {
        try { $owned = $mutex.WaitOne(0) } catch [Threading.AbandonedMutexException] { $owned = $true }
        if (-not $owned) { return [pscustomobject]@{ State = 'Busy'; Drive = $Drive } }
        do {
            if ([Environment]::TickCount64 -ge $endTicks) { break }
            $request.Operation = 'Snapshot'; $request.RequestId = [guid]::NewGuid().ToString('D')
            $result = Invoke-PcaiCloudMountWorker ($request.PSObject.Copy())
            if ($result.PSObject.Properties['Reason']) { $lastReason = $result.Reason }
            if ($result.StartOutcome -eq 'Unknown') { $uncertain = $true; break }
            if ([Environment]::TickCount64 -ge $endTicks -or $result.State -eq 'Expired') { break }
            if ($result.State -eq 'Denied') { throw $result.Reason }
            if ($result.State -in @('Qualified', 'Ready')) {
                $snapshot = $result.Snapshot
                if (-not (Test-PcaiCloudMountAdmission $snapshot.Storage $CacheVolumeGuid $BackingVolumeGuid $MinimumCacheFreeBytes $MinimumBackingFreeBytes)) { throw 'Worker response has no healthy identity.' }
                if ($snapshot.TaskHashBefore -ne $ExpectedTaskXmlSha256 -or $snapshot.TaskHashAfter -ne $ExpectedTaskXmlSha256 -or
                    $snapshot.Storage.BackingFileId -notmatch '^[A-Fa-f0-9]{48}$') { throw 'Worker snapshot bindings are incomplete.' }
                if ($result.State -eq 'Qualified') {
                    if ($snapshot.TaskState -ne 'Running' -or $snapshot.OwnerQualified -ne $true -or
                        $snapshot.Mapping.NetResult -ne 0 -or $snapshot.Mapping.Unc -ne $ProviderUnc -or $snapshot.Mapping.DriveType -ne 4 -or
                        -not $snapshot.OwnerWitness -or $snapshot.OwnerWitness.ProcessId -le 0 -or
                        $snapshot.OwnerWitness.Sid -ne $ExpectedUserSid -or -not $snapshot.OwnerWitness.CreationUtc) { throw 'Worker provider qualification differs.' }
                    if ([Environment]::TickCount64 -ge $endTicks) { break }
                    return [pscustomobject]@{ State = if ($started) { 'Started' } else { 'AlreadyRunning' }; Drive = $Drive;
                        ElapsedSeconds = ([Environment]::TickCount64 - $beginTicks) / 1000.0; Storage = $snapshot.Storage; OwnerWitness = $snapshot.OwnerWitness }
                }
                if ($snapshot.Mapping.DosResult -ne 0 -or $snapshot.Mapping.DosError -ne 2 -or $snapshot.Mapping.NetResult -ne 2250) { throw 'Worker reported Ready with a drive collision.' }
                if ($ValidateOnly) { return [pscustomobject]@{ State = 'Hold'; Drive = $Drive; Reason = 'Provider absent; ValidateOnly prohibits task start.'; StartRequested = $false } }
                if (-not $started -and $PSCmdlet.ShouldProcess($TaskName, 'Start existing reviewed read-only Limited mount task')) {
                    if ([Environment]::TickCount64 -ge $endTicks) { break }
                    $request.ExpectedBackingFileId = $snapshot.Storage.BackingFileId
                    $request.Operation = 'Start'; $request.RequestId = [guid]::NewGuid().ToString('D')
                    $launch = Invoke-PcaiCloudMountWorker ($request.PSObject.Copy())
                    if ($launch.StartOutcome -eq 'Unknown') { $uncertain = $true; break }
                    if ($launch.StartOutcome -eq 'Acknowledged') { $started = $true }
                    if ([Environment]::TickCount64 -ge $endTicks -or $launch.State -eq 'Expired') { break }
                    if ($launch.State -eq 'Denied') { throw $launch.Reason }
                }
            }
            $remaining = [int][Math]::Max(0, $endTicks - [Environment]::TickCount64)
            if ($remaining -gt 0) { Start-Sleep -Milliseconds ([Math]::Min($PollMilliseconds, $remaining)) }
        } while ([Environment]::TickCount64 -lt $endTicks)
        [pscustomobject]@{ State = 'Hold'; Drive = $Drive; Reason = if ($uncertain) { 'Scheduler request outcome is unknown; provider preserved.' } else { 'Admission/provider deadline expired; provider preserved. Last observation: ' + $lastReason };
            StartRequested = if ($uncertain) { 'Unknown' } else { $started } }
    } finally { if ($owned) { $mutex.ReleaseMutex() }; $mutex.Dispose() }
}

if ($MyInvocation.InvocationName -ne '.') {
    $result = Invoke-PcaiCloudMount @PSBoundParameters
    $result
    if ($result -isnot [string] -and $result.State -in @('Hold', 'Busy')) { exit 1 }
}
