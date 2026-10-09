#Requires -Version 7.0
# Dot-source only: load lazily so planning/help creates no jobs or native processes.
function Initialize-VigilBoundedProcessRunner {
    [CmdletBinding()]
    param()
    if ('Vigil.BoundedProcess' -as [type]) {
        $contract = [Vigil.BoundedProcess].GetField('ContractVersion')
        if (-not $contract -or $contract.GetValue($null) -ne 2) { throw 'An older Vigil process runner is loaded. Use a fresh PowerShell process.' }
        return
    }
    Add-Type -TypeDefinition @'
using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Win32.SafeHandles;

namespace Vigil {
    public sealed class BoundedProcessResult {
        public int ExitCode { get; set; }
        public string Stdout { get; set; }
        public string Stderr { get; set; }
    }
    public static class BoundedProcess {
        public const int ContractVersion = 2;
        static readonly object CustodyGate = new object();
        static readonly Dictionary<Guid, SafeProcessHandle> PendingCleanup = new Dictionary<Guid, SafeProcessHandle>();
        static readonly Dictionary<Guid, List<SafeFileHandle>> PendingOtherHandles = new Dictionary<Guid, List<SafeFileHandle>>();
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
                Check(CloseHandle(handle), "CloseHandle");
                handle = IntPtr.Zero;
            }
        }
        static void TryClose(ref IntPtr handle, List<Exception> failures) {
            try { Close(ref handle); } catch (Exception failure) { failures.Add(failure); }
        }
        static void Retain(ref IntPtr handle, List<SafeFileHandle> retained) {
            if (handle != IntPtr.Zero && handle != new IntPtr(-1)) {
                retained.Add(new SafeFileHandle(handle, true));
                handle = IntPtr.Zero;
            }
        }
        static void CloseRetained(SafeHandle handle) {
            if (handle.IsClosed || handle.IsInvalid) return;
            Check(CloseHandle(handle.DangerousGetHandle()), "Close retained owned handle");
            handle.SetHandleAsInvalid();
            handle.Dispose();
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
        static void ConfirmExit(IntPtr handle) {
            uint state = WaitForSingleObject(handle, 0);
            if (state == 0) return;
            if (state != 0x102) throw new Win32Exception(Marshal.GetLastWin32Error(), "Observe owned process exit");
            bool requested = TerminateProcess(handle, 1);
            int terminationError = Marshal.GetLastWin32Error();
            // Kill-on-close may already be terminating this exact process.
            // TerminateProcess can fail during that race; confirmed exit is
            // authoritative, rather than the redundant termination request.
            if (WaitForSingleObject(handle, 1000) == 0) return;
            if (!requested) throw new Win32Exception(terminationError, "Owned process termination remains unconfirmed");
            throw new TimeoutException("Owned process remained alive during cleanup.");
        }
        public static Guid[] GetPendingCleanupIds() {
            lock (CustodyGate) {
                var ids = new Guid[PendingCleanup.Count];
                PendingCleanup.Keys.CopyTo(ids, 0);
                return ids;
            }
        }
        public static void RetryCleanup(Guid custodyId) {
            lock (CustodyGate) {
                SafeProcessHandle owned;
                if (!PendingCleanup.TryGetValue(custodyId, out owned)) throw new ArgumentException("Unknown owned process custody identity.");
                var failures = new List<Exception>();
                List<SafeFileHandle> otherHandles;
                if (PendingOtherHandles.TryGetValue(custodyId, out otherHandles)) {
                    foreach (SafeFileHandle handle in otherHandles) {
                        try { CloseRetained(handle); } catch (Exception failure) { failures.Add(failure); }
                    }
                }
                try {
                    if (!owned.IsInvalid && !owned.IsClosed) {
                        ConfirmExit(owned.DangerousGetHandle());
                        CloseRetained(owned);
                    }
                } catch (Exception failure) { failures.Add(failure); }
                if (failures.Count != 0) {
                    var failure = new AggregateException("Owned handle cleanup remains unconfirmed; custody retained.", failures);
                    failure.Data["VigilProcessCustodyId"] = custodyId;
                    throw failure;
                }
                PendingCleanup.Remove(custodyId);
                PendingOtherHandles.Remove(custodyId);
            }
        }
        public static BoundedProcessResult Run(string path, string[] arguments, int timeoutSeconds, string directory, string inputText) {
            lock (CustodyGate) {
                if (PendingCleanup.Count != 0) throw new InvalidOperationException("Unconfirmed process closure blocks another launch. Retry the retained custody identity first.");
            }
            var clock = Stopwatch.StartNew();
            int deadline = checked(timeoutSeconds * 1000);
            IntPtr job = IntPtr.Zero, outRead = IntPtr.Zero, outWrite = IntPtr.Zero;
            IntPtr errRead = IntPtr.Zero, errWrite = IntPtr.Zero, input = IntPtr.Zero, inputWrite = IntPtr.Zero;
            ProcessInfo process = new ProcessInfo();
            Exception operationFailure = null;
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
                if (inputText == null) {
                    input = CreateFile("NUL", 0x80000000, 3, ref security, 3, 0, IntPtr.Zero);
                    if (input == new IntPtr(-1)) throw new Win32Exception(Marshal.GetLastWin32Error(), "Open noninteractive input");
                } else {
                    Check(CreatePipe(out input, out inputWrite, ref security, 0), "CreatePipe stdin");
                    Check(SetHandleInformation(inputWrite, 1, 0), "Non-inherited stdin write handle");
                }
                var startup = new StartupInfo { Size = Marshal.SizeOf<StartupInfo>(), Flags = 0x101, Input = input, Output = outWrite, Error = errWrite };
                var command = new StringBuilder(Quote(path));
                foreach (string argument in arguments) command.Append(' ').Append(Quote(argument));
                Check(CreateProcess(path, command, IntPtr.Zero, IntPtr.Zero, true, 0x08000004, IntPtr.Zero, string.IsNullOrEmpty(directory) ? null : directory, ref startup, out process), "CreateProcess suspended");
                Check(AssignProcessToJobObject(job, process.Process), "AssignProcessToJobObject");
                if (ResumeThread(process.Thread) == uint.MaxValue) throw new Win32Exception(Marshal.GetLastWin32Error(), "ResumeThread");
                Close(ref outWrite); Close(ref errWrite); Close(ref input);
                var output = ReadPipe(outRead); outRead = IntPtr.Zero;
                var error = ReadPipe(errRead); errRead = IntPtr.Zero;
                Task inputTask = Task.CompletedTask;
                if (inputWrite != IntPtr.Zero) {
                    var ownedInput = new SafeFileHandle(inputWrite, true); inputWrite = IntPtr.Zero;
                    inputTask = Task.Run(() => {
                        using (ownedInput)
                        using (var stream = new FileStream(ownedInput, FileAccess.Write))
                        using (var writer = new StreamWriter(stream, new UTF8Encoding(false))) {
                            writer.Write(inputText); writer.Flush();
                        }
                    });
                }
                uint wait = WaitForSingleObject(process.Process, (uint)Math.Max(0, deadline - clock.ElapsedMilliseconds));
                // Job closure kills orphaned descendants even after parent exit.
                Close(ref job);
                if (wait == 0x102) {
                    if (WaitForSingleObject(process.Process, 1000) != 0) throw new TimeoutException("Native process cleanup did not finish.");
                    throw new TimeoutException(path + " timed out after " + timeoutSeconds + " seconds; process tree terminated.");
                }
                if (wait != 0) throw new Win32Exception(Marshal.GetLastWin32Error(), "WaitForSingleObject");
                uint code; Check(GetExitCodeProcess(process.Process, out code), "GetExitCodeProcess");
                if (!Task.WaitAll(new Task[] { output, error, inputTask }, (int)Math.Max(0, deadline - clock.ElapsedMilliseconds))) throw new TimeoutException(path + " timed out after " + timeoutSeconds + " seconds during input/output capture; process tree terminated.");
                return new BoundedProcessResult { ExitCode = unchecked((int)code), Stdout = output.Result, Stderr = error.Result };
            }
            catch (Exception failure) { operationFailure = failure; throw; }
            finally {
                var failures = new List<Exception>();
                TryClose(ref job, failures);
                bool exited = process.Process == IntPtr.Zero;
                try {
                    if (process.Process != IntPtr.Zero) ConfirmExit(process.Process);
                    exited = true;
                }
                catch (Exception failure) { failures.Add(failure); }
                TryClose(ref process.Thread, failures);
                if (exited) TryClose(ref process.Process, failures);
                TryClose(ref outRead, failures); TryClose(ref outWrite, failures);
                TryClose(ref errRead, failures); TryClose(ref errWrite, failures);
                TryClose(ref input, failures); TryClose(ref inputWrite, failures);
                if (failures.Count != 0) {
                    Guid custodyId = Guid.NewGuid();
                    var retained = new List<SafeFileHandle>();
                    Retain(ref job, retained); Retain(ref process.Thread, retained);
                    Retain(ref outRead, retained); Retain(ref outWrite, retained);
                    Retain(ref errRead, retained); Retain(ref errWrite, retained);
                    Retain(ref input, retained); Retain(ref inputWrite, retained);
                    lock (CustodyGate) {
                        PendingCleanup.Add(custodyId, new SafeProcessHandle(process.Process, true));
                        PendingOtherHandles.Add(custodyId, retained);
                    }
                    process.Process = IntPtr.Zero;
                    if (operationFailure != null) failures.Insert(0, operationFailure);
                    Exception surfaced = new AggregateException("Native operation or cleanup failed; exact owned handle custody retained.", failures);
                    surfaced.Data["VigilProcessCustodyId"] = custodyId;
                    surfaced.Data["OwnedProcessId"] = process.ProcessId;
                    throw surfaced;
                }
            }
        }
    }
}

'@
}

function Get-VigilPendingProcessCustody {
    <# .SYNOPSIS
    List retained identities whose owned process closure is unconfirmed.
    #>
    [CmdletBinding()]
    param()
    Initialize-VigilBoundedProcessRunner
    [Vigil.BoundedProcess]::GetPendingCleanupIds()
}

function Stop-VigilPendingProcessCustody {
    <# .SYNOPSIS
    Retry termination using a retained owned process handle.
    #>
    [CmdletBinding(SupportsShouldProcess)]
    param([Parameter(Mandatory)][guid]$CustodyId)
    if ($PSCmdlet.ShouldProcess($CustodyId.ToString(), 'Confirm owned process termination and release handle custody')) {
        Initialize-VigilBoundedProcessRunner
        [Vigil.BoundedProcess]::RetryCleanup($CustodyId)
    }
}

function Invoke-VigilBoundedProcess {
    <#
    .SYNOPSIS
    Run one Windows process tree with bounded UTF8 input and output capture.
    .DESCRIPTION
    Assigns a suspended parent to a private kill-on-close Windows job before it
    runs. Input, process exit and capture share one elapsed runtime deadline;
    timeout and failure cleanup wait at most two extra seconds for parent termination.
    Closing the job also terminates descendants after a normal parent exit.
    This API intentionally does not support detached background child processes.
    With no InputText, stdin is noninteractive EOF. Arguments use Windows command
    line escaping, and output is decoded as UTF8. Initialization compiles the
    helper once per PowerShell process before the native runtime deadline starts.
    Returns ExitCode, Stdout and Stderr; launch/capture/timeout errors throw.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [string[]]$Arguments = @(),
        [ValidateRange(1, 3600)][int]$TimeoutSeconds = 30,
        [string]$WorkingDirectory,
        [AllowNull()][string]$InputText
    )
    if (-not $IsWindows) { throw 'The owned job process runner requires Windows.' }
    Initialize-VigilBoundedProcessRunner
    # Resolve PATH applications before CreateProcess, retaining its launch error for absent files.
    $application = Get-Command -Name $FilePath -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    $resolvedPath = if ($application) { $application.Source } else { $FilePath }
    $directory = if ($WorkingDirectory) { [IO.Path]::GetFullPath($WorkingDirectory) } else { $null }
    $inputValue = if ($PSBoundParameters.ContainsKey('InputText')) { $InputText } else { $null }
    [Vigil.BoundedProcess]::Run($resolvedPath, $Arguments, $TimeoutSeconds, $directory, $inputValue)
}
