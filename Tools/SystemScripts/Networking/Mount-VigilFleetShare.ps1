#Requires -Version 7.0
<#
.SYNOPSIS
Inspect or explicitly mount the public ASUS fleet NFS share on a reachable LAN.
.DESCRIPTION
Defaults to read-only status. Mount and Unmount require -Apply and honor WhatIf.
DryRun and help do not invoke native programs or write files. Only the public
/srv/vigil-share export and the two approved Windows source addresses are used.
Existing foreign drive mappings are preserved. Native calls have a deadline.
.EXAMPLE
./Mount-VigilFleetShare.ps1 -Action Mount -DryRun
.EXAMPLE
./Mount-VigilFleetShare.ps1 -Action Mount -Apply
.EXAMPLE
./Mount-VigilFleetShare.ps1 -Action Unmount -Apply
#>
[CmdletBinding(SupportsShouldProcess, PositionalBinding = $false)]
param(
    [ValidateSet('Status', 'Mount', 'Unmount')][string]$Action = 'Status',
    [ValidatePattern('^[D-Zd-z]:?$')][string]$Drive = 'N:',
    [switch]$Apply,
    [switch]$DryRun,
    [ValidateRange(100, 3000)][int]$ProbeTimeoutMilliseconds = 750,
    [ValidateRange(1, 30)][int]$NativeTimeoutSeconds = 8,
    [switch]$ShowTaskTemplate,
    [Alias('h', 'help')][switch]$ShowHelp,
    [Parameter(ValueFromRemainingArguments)][string[]]$RemainingArguments = @()
)

Set-StrictMode -Version Latest

function Invoke-VigilNfsNative {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [string[]]$Arguments = @(),
        [ValidateRange(1, 30)][int]$TimeoutSeconds = 8
    )
    Initialize-VigilNfsProcessRunner
    [Vigil.NfsBoundedProcess]::Run($FilePath, $Arguments, $TimeoutSeconds)
}

function Initialize-VigilNfsProcessRunner {
    [CmdletBinding()]
    param()
    if ('Vigil.NfsBoundedProcess' -as [type]) { return }
    # Suspend before assigning a kill-on-close job, so descendants cannot escape.
    Add-Type -TypeDefinition @'
using System;
using System.ComponentModel;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Win32.SafeHandles;

namespace Vigil {
    public sealed class NfsProcessResult {
        public int ExitCode { get; set; }
        public string Stdout { get; set; }
        public string Stderr { get; set; }
    }
    public static class NfsBoundedProcess {
        [StructLayout(LayoutKind.Sequential)]
        struct SecurityAttributes { public int Size; public IntPtr Descriptor; public int Inherit; }
        [StructLayout(LayoutKind.Sequential)]
        struct StartupInfo {
            public int Size; public IntPtr Reserved, Desktop, Title;
            public int X, Y, XSize, YSize, XChars, YChars, Fill, Flags;
            public short ShowWindow, ReservedSize; public IntPtr ReservedBytes;
            public IntPtr Input, Output, Error;
        }
        [StructLayout(LayoutKind.Sequential)]
        struct ProcessInfo { public IntPtr Process, Thread; public int ProcessId, ThreadId; }
        [StructLayout(LayoutKind.Sequential)]
        struct BasicLimits {
            public long ProcessTime, JobTime; public uint Flags;
            public UIntPtr MinimumWorkingSet, MaximumWorkingSet;
            public uint ActiveProcessLimit; public UIntPtr Affinity;
            public uint Priority, SchedulingClass;
        }
        [StructLayout(LayoutKind.Sequential)]
        struct IoCounters { public ulong ReadOperations, WriteOperations, OtherOperations, ReadBytes, WriteBytes, OtherBytes; }
        [StructLayout(LayoutKind.Sequential)]
        struct ExtendedLimits {
            public BasicLimits Basic; public IoCounters Io;
            public UIntPtr ProcessMemory, JobMemory, PeakProcessMemory, PeakJobMemory;
        }
        [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)]
        static extern IntPtr CreateJobObject(IntPtr attributes, string name);
        [DllImport("kernel32.dll", SetLastError=true)]
        static extern bool SetInformationJobObject(IntPtr job, int infoClass, ref ExtendedLimits limits, int size);
        [DllImport("kernel32.dll", SetLastError=true)]
        static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);
        [DllImport("kernel32.dll", SetLastError=true)]
        static extern bool CreatePipe(out IntPtr read, out IntPtr write, ref SecurityAttributes attributes, int size);
        [DllImport("kernel32.dll", SetLastError=true)]
        static extern bool SetHandleInformation(IntPtr handle, uint mask, uint flags);
        [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)]
        static extern IntPtr CreateFile(string name, uint access, uint sharing, ref SecurityAttributes attributes, uint creation, uint flags, IntPtr template);
        [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)]
        static extern bool CreateProcess(string application, StringBuilder command, IntPtr processAttributes, IntPtr threadAttributes, bool inherit, uint flags, IntPtr environment, string directory, ref StartupInfo startup, out ProcessInfo process);
        [DllImport("kernel32.dll", SetLastError=true)]
        static extern uint ResumeThread(IntPtr thread);
        [DllImport("kernel32.dll", SetLastError=true)]
        static extern uint WaitForSingleObject(IntPtr handle, uint milliseconds);
        [DllImport("kernel32.dll", SetLastError=true)]
        static extern bool GetExitCodeProcess(IntPtr process, out uint code);
        [DllImport("kernel32.dll", SetLastError=true)]
        static extern bool TerminateProcess(IntPtr process, uint code);
        [DllImport("kernel32.dll", SetLastError=true)]
        static extern bool CloseHandle(IntPtr handle);

        static void Check(bool success, string operation) {
            if (!success) throw new Win32Exception(Marshal.GetLastWin32Error(), operation);
        }
        static void Close(ref IntPtr handle) {
            if (handle != IntPtr.Zero && handle != new IntPtr(-1)) {
                IntPtr value = handle; handle = IntPtr.Zero;
                Check(CloseHandle(value), "CloseHandle");
            }
        }
        static string Quote(string value) {
            var text = new StringBuilder("\""); int slashes = 0;
            foreach (char c in value) {
                if (c == '\\') { slashes++; continue; }
                if (c == '"') text.Append('\\', slashes * 2 + 1);
                else text.Append('\\', slashes);
                text.Append(c); slashes = 0;
            }
            text.Append('\\', slashes * 2); return text.Append('"').ToString();
        }
        static Task<string> ReadPipe(IntPtr handle) {
            var owned = new SafeFileHandle(handle, true);
            return Task.Run(() => {
                using (owned)
                using (var stream = new FileStream(owned, FileAccess.Read))
                using (var reader = new StreamReader(stream, Encoding.UTF8))
                    return reader.ReadToEnd();
            });
        }
        public static NfsProcessResult Run(string path, string[] arguments, int timeoutSeconds) {
            IntPtr job = IntPtr.Zero, outRead = IntPtr.Zero, outWrite = IntPtr.Zero;
            IntPtr errRead = IntPtr.Zero, errWrite = IntPtr.Zero, input = IntPtr.Zero;
            ProcessInfo process = new ProcessInfo();
            try {
                job = CreateJobObject(IntPtr.Zero, null);
                if (job == IntPtr.Zero) throw new Win32Exception(Marshal.GetLastWin32Error(), "CreateJobObject");
                var limits = new ExtendedLimits(); limits.Basic.Flags = 0x2000;
                Check(SetInformationJobObject(job, 9, ref limits, Marshal.SizeOf<ExtendedLimits>()), "SetInformationJobObject");
                var security = new SecurityAttributes { Size = Marshal.SizeOf<SecurityAttributes>(), Inherit = 1 };
                Check(CreatePipe(out outRead, out outWrite, ref security, 0), "CreatePipe stdout");
                Check(CreatePipe(out errRead, out errWrite, ref security, 0), "CreatePipe stderr");
                Check(SetHandleInformation(outRead, 1, 0), "Non-inherited stdout read handle");
                Check(SetHandleInformation(errRead, 1, 0), "Non-inherited stderr read handle");
                input = CreateFile("NUL", 0x80000000, 3, ref security, 3, 0, IntPtr.Zero);
                if (input == new IntPtr(-1)) throw new Win32Exception(Marshal.GetLastWin32Error(), "Open noninteractive input");
                var startup = new StartupInfo { Size = Marshal.SizeOf<StartupInfo>(), Flags = 0x101, Input = input, Output = outWrite, Error = errWrite };
                var command = new StringBuilder(Quote(path));
                foreach (string argument in arguments) command.Append(' ').Append(Quote(argument));
                Check(CreateProcess(path, command, IntPtr.Zero, IntPtr.Zero, true, 0x08000004, IntPtr.Zero, null, ref startup, out process), "CreateProcess suspended");
                Check(AssignProcessToJobObject(job, process.Process), "AssignProcessToJobObject");
                if (ResumeThread(process.Thread) == uint.MaxValue) throw new Win32Exception(Marshal.GetLastWin32Error(), "ResumeThread");
                Close(ref outWrite); Close(ref errWrite); Close(ref input);
                var output = ReadPipe(outRead); outRead = IntPtr.Zero;
                var error = ReadPipe(errRead); errRead = IntPtr.Zero;
                uint wait = WaitForSingleObject(process.Process, checked((uint)timeoutSeconds * 1000));
                // Job closure kills orphaned descendants even after parent exit.
                Close(ref job);
                if (wait == 0x102) {
                    if (WaitForSingleObject(process.Process, 1000) != 0) throw new TimeoutException("Native process cleanup did not finish.");
                    throw new TimeoutException(path + " timed out after " + timeoutSeconds + " seconds; process tree terminated.");
                }
                if (wait != 0) throw new Win32Exception(Marshal.GetLastWin32Error(), "WaitForSingleObject");
                uint code; Check(GetExitCodeProcess(process.Process, out code), "GetExitCodeProcess");
                if (!Task.WaitAll(new Task[] { output, error }, 1000)) throw new TimeoutException("Native output capture exceeded cleanup deadline.");
                return new NfsProcessResult { ExitCode = unchecked((int)code), Stdout = output.Result, Stderr = error.Result };
            }
            finally {
                try {
                    Close(ref job);
                    if (process.Process != IntPtr.Zero && WaitForSingleObject(process.Process, 0) == 0x102) {
                        Check(TerminateProcess(process.Process, 1), "Terminate suspended/unassigned process");
                        if (WaitForSingleObject(process.Process, 1000) != 0) throw new TimeoutException("Process remained alive during cleanup.");
                    }
                }
                finally {
                    Close(ref process.Thread); Close(ref process.Process);
                    Close(ref outRead); Close(ref outWrite); Close(ref errRead); Close(ref errWrite); Close(ref input);
                }
            }
        }
    }
}
'@
}

function Test-VigilNfsPort {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$Server,
        [Parameter(Mandatory)][string]$SourceAddress,
        [ValidateRange(100, 3000)][int]$TimeoutMilliseconds = 750
    )
    $client = [Net.Sockets.TcpClient]::new([Net.Sockets.AddressFamily]::InterNetwork)
    try {
        $client.Client.Bind([Net.IPEndPoint]::new([Net.IPAddress]::Parse($SourceAddress), 0))
        $connection = $client.ConnectAsync([Net.IPAddress]::Parse($Server), 2049)
        if (-not $connection.Wait($TimeoutMilliseconds)) { return $false }
        return $client.Connected
    }
    catch [Net.Sockets.SocketException] { return $false }
    catch [AggregateException] {
        if ($_.Exception.GetBaseException() -is [Net.Sockets.SocketException]) { return $false }
        throw
    }
    finally { $client.Dispose() }
}

function Select-VigilNfsEndpoint {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][AllowEmptyCollection()][object[]]$Addresses,
        [Parameter(Mandatory)][AllowEmptyCollection()][object[]]$Adapters,
        [ValidateRange(100, 3000)][int]$TimeoutMilliseconds = 750,
        [ValidateSet('192.168.50.2', '10.60.4.1')][string]$ServerHint
    )
    foreach ($pair in @(
            @{ Server = '192.168.50.2'; Source = '192.168.50.42'; Prefix = 24 },
            @{ Server = '10.60.4.1'; Source = '10.60.4.4'; Prefix = 29 }
        )) {
        if ($ServerHint -and $pair.Server -ne $ServerHint) { continue }
        foreach ($address in $Addresses) {
            if ($address.IPAddress -ne $pair.Source -or $address.PrefixLength -ne $pair.Prefix -or
                $address.AddressState -ne 'Preferred') { continue }
            $up = @($Adapters | Where-Object { $_.ifIndex -eq $address.InterfaceIndex -and $_.Status -eq 'Up' })
            if ($up.Count -ne 1) { continue }
            if (Test-VigilNfsPort -Server $pair.Server -SourceAddress $pair.Source -TimeoutMilliseconds $TimeoutMilliseconds) {
                return [pscustomobject]@{
                    Server         = $pair.Server
                    SourceAddress  = $pair.Source
                    InterfaceIndex = $address.InterfaceIndex
                    Remote         = "$($pair.Server):/srv/vigil-share"
                }
            }
        }
    }
    return $null
}

function ConvertFrom-VigilNfsMount {
    [CmdletBinding()]
    param([AllowEmptyString()][string]$Text)
    foreach ($line in $Text -split '\r?\n') {
        if ($line -match '^\s*(?<drive>[A-Za-z]:)\s+(?<remote>\\\\\S+|[0-9.]+:/\S+)') {
            [pscustomobject]@{ Drive = $Matches.drive.ToUpperInvariant(); Remote = $Matches.remote }
        }
    }
}

function Test-VigilOwnedNfsMount {
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$Remote)
    return $Remote -cin @(
        '192.168.50.2:/srv/vigil-share', '10.60.4.1:/srv/vigil-share',
        '\\192.168.50.2\srv\vigil-share', '\\10.60.4.1\srv\vigil-share'
    )
}

function Get-VigilDriveLetter {
    [CmdletBinding()]
    param()
    @([Environment]::GetLogicalDrives() | ForEach-Object { $_.Substring(0, 2).ToUpperInvariant() }) +
    @(Get-PSDrive -PSProvider FileSystem | Where-Object { $_.Name -match '^[A-Za-z]$' } |
            ForEach-Object { "$($_.Name.ToUpperInvariant()):" })
}

function Get-VigilNfsTaskTemplate {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][ValidateScript({ Test-Path -LiteralPath $_ -PathType Leaf })][string]$ScriptPath,
        [ValidatePattern('^S-1-\d+(?:-\d+)+$')][string]$UserId = ([Security.Principal.WindowsIdentity]::GetCurrent().User.Value),
        [string]$PowerShellPath = (Get-Command pwsh -CommandType Application -ErrorAction Stop | Select-Object -First 1).Source
    )
    $resolved = (Resolve-Path -LiteralPath $ScriptPath).Path
    if ($resolved.Contains('"')) { throw 'Task script path cannot contain a double quote.' }
    $escape = { param([string]$Value) [Security.SecurityElement]::Escape($Value) }
    $commandXml = & $escape $PowerShellPath
    $argumentsXml = & $escape "-NoProfile -NonInteractive -WindowStyle Hidden -File `"$resolved`" -Action Mount -Apply"
    $subscriptionXml = & $escape '<QueryList><Query Id="0" Path="Microsoft-Windows-NetworkProfile/Operational"><Select Path="Microsoft-Windows-NetworkProfile/Operational">*[System[(EventID=10000)]]</Select></Query></QueryList>'
    $xml = @"
<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.4" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo><Description>Mount public ASUS NFS share only when its approved LAN is reachable.</Description></RegistrationInfo>
  <Triggers>
    <LogonTrigger><Enabled>true</Enabled><UserId>$UserId</UserId><Delay>PT15S</Delay></LogonTrigger>
    <EventTrigger><Enabled>true</Enabled><Subscription>$subscriptionXml</Subscription><Delay>PT10S</Delay></EventTrigger>
  </Triggers>
  <Principals><Principal id="User"><UserId>$UserId</UserId><LogonType>InteractiveToken</LogonType><RunLevel>LeastPrivilege</RunLevel></Principal></Principals>
  <Settings><MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy><DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries><StopIfGoingOnBatteries>false</StopIfGoingOnBatteries><StartWhenAvailable>true</StartWhenAvailable><ExecutionTimeLimit>PT1M</ExecutionTimeLimit><Enabled>true</Enabled></Settings>
  <Actions Context="User"><Exec><Command>$commandXml</Command><Arguments>$argumentsXml</Arguments></Exec></Actions>
</Task>
"@
    [pscustomobject]@{ TaskName = 'VIGIL-Public-ASUS-NFS'; UserId = $UserId; Xml = $xml; Registered = $false }
}

function Invoke-VigilFleetShare {
    [CmdletBinding(SupportsShouldProcess)]
    param(
        [ValidateSet('Status', 'Mount', 'Unmount')][string]$Action = 'Status',
        [ValidatePattern('^[D-Zd-z]:?$')][string]$Drive = 'N:',
        [switch]$Apply,
        [switch]$DryRun,
        [ValidateRange(100, 3000)][int]$ProbeTimeoutMilliseconds = 750,
        [ValidateRange(1, 30)][int]$NativeTimeoutSeconds = 8
    )
    $driveLetter = $Drive.TrimEnd(':').ToUpperInvariant() + ':'
    $result = [ordered]@{ Action = $Action; Drive = $driveLetter; State = 'Plan'; Endpoint = $null; Native = $null }
    if ($DryRun) { return [pscustomobject]$result }
    if (-not $IsWindows) { throw 'This helper requires Windows Client for NFS.' }
    $mountPath = Join-Path $env:WINDIR 'System32/mount.exe'
    $unmountPath = Join-Path $env:WINDIR 'System32/umount.exe'
    if (-not (Test-Path -LiteralPath $mountPath -PathType Leaf)) { throw 'Windows Client for NFS mount.exe is unavailable.' }
    $inventory = Invoke-VigilNfsNative -FilePath $mountPath -TimeoutSeconds $NativeTimeoutSeconds
    if ($inventory.ExitCode -ne 0) { throw "NFS inventory failed: $($inventory.Stderr) $($inventory.Stdout)" }
    $mappings = @(ConvertFrom-VigilNfsMount -Text $inventory.Stdout | Where-Object { $_.Drive -eq $driveLetter })
    if ($mappings.Count -gt 1) { throw "Ambiguous NFS mappings for $driveLetter; preserved." }
    if ($mappings.Count -eq 1) {
        if (-not (Test-VigilOwnedNfsMount -Remote $mappings[0].Remote)) {
            $result.State = 'ForeignDrivePreserved'
            return [pscustomobject]$result
        }
        $result.Endpoint = $mappings[0].Remote
        $result.State = 'AlreadyMounted'
        if ($Action -ne 'Unmount') {
            $existingServer = if ($mappings[0].Remote.Contains('192.168.50.2')) { '192.168.50.2' } else { '10.60.4.1' }
            $reachable = Select-VigilNfsEndpoint -Addresses @(Get-NetIPAddress -AddressFamily IPv4) -Adapters @(Get-NetAdapter -IncludeHidden) -TimeoutMilliseconds $ProbeTimeoutMilliseconds -ServerHint $existingServer
            if ($null -eq $reachable) { $result.State = 'ExistingEndpointUnavailable' }
            return [pscustomobject]$result
        }
        if ($Apply -and $PSCmdlet.ShouldProcess($driveLetter, 'Unmount public ASUS NFS share')) {
            $current = Invoke-VigilNfsNative -FilePath $mountPath -TimeoutSeconds $NativeTimeoutSeconds
            if ($current.ExitCode -ne 0) { throw 'Cannot verify NFS custody before unmount; preserved.' }
            $currentMappings = @(ConvertFrom-VigilNfsMount -Text $current.Stdout | Where-Object { $_.Drive -eq $driveLetter })
            if ($currentMappings.Count -ne 1 -or $currentMappings[0].Remote -cne $mappings[0].Remote) {
                throw 'NFS drive identity changed before unmount; preserved.'
            }
            $native = Invoke-VigilNfsNative -FilePath $unmountPath -Arguments @($driveLetter) -TimeoutSeconds $NativeTimeoutSeconds
            if ($native.ExitCode -ne 0) { throw "NFS unmount failed (exit $($native.ExitCode)): $($native.Stderr) $($native.Stdout)" }
            $verified = Invoke-VigilNfsNative -FilePath $mountPath -TimeoutSeconds $NativeTimeoutSeconds
            if ($verified.ExitCode -ne 0) { throw 'NFS unmount returned success but verification inventory failed.' }
            if (@(ConvertFrom-VigilNfsMount -Text $verified.Stdout | Where-Object { $_.Drive -eq $driveLetter }).Count -gt 0) {
                throw 'NFS unmount returned success but a mapping is still observed on the drive.'
            }
            $result.Native = $native
            $result.State = 'Unmounted'
        }
        else { $result.State = 'WouldUnmount' }
        return [pscustomobject]$result
    }
    if ($driveLetter -in @(Get-VigilDriveLetter)) {
        $result.State = 'ForeignDrivePreserved'
        return [pscustomobject]$result
    }
    if ($Action -eq 'Unmount') {
        $result.State = 'AlreadyUnmounted'
        return [pscustomobject]$result
    }
    $endpoint = Select-VigilNfsEndpoint -Addresses @(Get-NetIPAddress -AddressFamily IPv4) -Adapters @(Get-NetAdapter -IncludeHidden) -TimeoutMilliseconds $ProbeTimeoutMilliseconds
    $result.Endpoint = $endpoint
    if ($null -eq $endpoint) {
        $result.State = 'LanUnavailable'
        return [pscustomobject]$result
    }
    $result.State = 'Available'
    if ($Action -eq 'Mount') {
        if ($Apply -and $PSCmdlet.ShouldProcess($driveLetter, "Mount $($endpoint.Remote)")) {
            # Recheck the drive immediately before changing state; never rebind it.
            if ($driveLetter -in @(Get-VigilDriveLetter)) { throw "$driveLetter became occupied; preserved." }
            $native = Invoke-VigilNfsNative -FilePath $mountPath -Arguments @(
                '-o', 'anon', 'mtype=soft', 'timeout=1', 'retry=1', $endpoint.Remote, $driveLetter
            ) -TimeoutSeconds $NativeTimeoutSeconds
            if ($native.ExitCode -ne 0) { throw "NFS mount failed (exit $($native.ExitCode)): $($native.Stderr) $($native.Stdout)" }
            $verified = Invoke-VigilNfsNative -FilePath $mountPath -TimeoutSeconds $NativeTimeoutSeconds
            if ($verified.ExitCode -ne 0) { throw 'NFS mount returned success but verification inventory failed.' }
            $match = @(ConvertFrom-VigilNfsMount -Text $verified.Stdout | Where-Object {
                    $_.Drive -eq $driveLetter -and (Test-VigilOwnedNfsMount -Remote $_.Remote)
                })
            if ($match.Count -ne 1) { throw 'NFS mount returned success but the expected mapping was not observed.' }
            $result.Native = $native
            $result.State = 'Mounted'
        }
        else { $result.State = 'WouldMount' }
    }
    [pscustomobject]$result
}

if ($MyInvocation.InvocationName -ne '.') {
    if ($ShowHelp -or '--help' -in $RemainingArguments) { Get-Help $PSCommandPath -Detailed; return }
    if ($RemainingArguments.Count -gt 0) { throw 'Unrecognized positional argument; use -h for help.' }
    if ($ShowTaskTemplate) { Get-VigilNfsTaskTemplate -ScriptPath $PSCommandPath; return }
    $forward = @{
        Action = $Action; Drive = $Drive; Apply = $Apply; DryRun = $DryRun
        ProbeTimeoutMilliseconds = $ProbeTimeoutMilliseconds; NativeTimeoutSeconds = $NativeTimeoutSeconds
    }
    if ($PSBoundParameters.ContainsKey('WhatIf')) { $forward.WhatIf = $PSBoundParameters.WhatIf }
    if ($PSBoundParameters.ContainsKey('Confirm')) { $forward.Confirm = $PSBoundParameters.Confirm }
    Invoke-VigilFleetShare @forward
}
