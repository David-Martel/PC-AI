#Requires -Version 7.0
# Dot-source only: load lazily so planning/help creates no jobs or native processes.
function Initialize-VigilBoundedProcessRunner {
    [CmdletBinding()]
    param()
    if ('Vigil.BoundedProcess' -as [type]) { return }
    Add-Type -TypeDefinition @'
using System;
using System.ComponentModel;
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
        public static BoundedProcessResult Run(string path, string[] arguments, int timeoutSeconds, string directory, string inputText) {
            var clock = Stopwatch.StartNew();
            int deadline = checked(timeoutSeconds * 1000);
            IntPtr job = IntPtr.Zero, outRead = IntPtr.Zero, outWrite = IntPtr.Zero;
            IntPtr errRead = IntPtr.Zero, errWrite = IntPtr.Zero, input = IntPtr.Zero, inputWrite = IntPtr.Zero;
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
                    Close(ref outRead); Close(ref outWrite); Close(ref errRead); Close(ref errWrite); Close(ref input); Close(ref inputWrite);
                }
            }
        }
    }
}
'@
}

function Invoke-VigilBoundedProcess {
    <#
    .SYNOPSIS
    Run one Windows process tree with bounded UTF8 input and output capture.
    .DESCRIPTION
    Assigns a suspended parent to a private kill-on-close Windows job before it
    runs. Input, process exit and capture share one elapsed runtime deadline;
    failure cleanup waits at most one extra second for parent termination.
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
