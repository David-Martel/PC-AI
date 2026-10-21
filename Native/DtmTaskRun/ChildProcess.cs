using System.Collections;
using System.ComponentModel;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;
using Microsoft.Win32.SafeHandles;

namespace DtmTaskRun;

/// <summary>Raised when the child was created but could not be put in the job, so it was never run.</summary>
internal sealed class ContainmentException : Exception
{
    /// <summary>Creates the exception.</summary>
    public ContainmentException(string message)
        : base(message)
    {
    }
}

/// <summary>
/// The child process, started with <c>CreateProcessW</c> so that two things hold that
/// <see cref="Process.Start(ProcessStartInfo)"/> cannot give:
/// <list type="number">
/// <item>The child's command line is the caller's text, byte for byte, as under <c>conhost --headless</c>
/// (see <see cref="RawCommandLine"/>).</item>
/// <item>The child is created suspended, put in the job, and only then resumed, so nothing it starts
/// can run outside the job. If the job cannot take it, the child is terminated before its first
/// instruction and <see cref="ContainmentException"/> is thrown.</item>
/// </list>
/// The three standard streams are pipes, and the window is suppressed with <c>CREATE_NO_WINDOW</c>.
/// </summary>
internal sealed class ChildProcess : IDisposable
{
    /// <summary>CREATE_SUSPENDED | CREATE_NO_WINDOW | CREATE_UNICODE_ENVIRONMENT.</summary>
    internal const uint CreationFlags = CreateSuspended | CreateNoWindow | CreateUnicodeEnvironment;

    private const uint CreateSuspended = 0x00000004;
    private const uint CreateNoWindow = 0x08000000;
    private const uint CreateUnicodeEnvironment = 0x00000400;
    private const uint StartfUseStdHandles = 0x00000100;
    private const uint HandleFlagInherit = 0x00000001;
    private const uint WaitObject0 = 0;
    private const uint Infinite = 0xFFFFFFFF;

    private IntPtr _process;
    private StreamWriter? _stdin;

    private ChildProcess(IntPtr process, int pid, StreamWriter stdin, StreamReader stdout, StreamReader stderr)
    {
        _process = process;
        Pid = pid;
        _stdin = stdin;
        StandardOutput = stdout;
        StandardError = stderr;
    }

    /// <summary>Process id.</summary>
    public int Pid { get; }

    /// <summary>The child's stdout (UTF-8).</summary>
    public StreamReader StandardOutput { get; }

    /// <summary>The child's stderr (UTF-8).</summary>
    public StreamReader StandardError { get; }

    /// <summary>
    /// Creates the child suspended, assigns it to <paramref name="job"/>, then resumes it.
    /// </summary>
    /// <param name="commandLine">Full child command line, program name first.</param>
    /// <param name="environment">Extra variables layered over the launcher's own environment.</param>
    /// <param name="job">The kill-on-close job; required, because a child that cannot be contained is not run.</param>
    /// <exception cref="Win32Exception">CreateProcessW failed (missing exe, bad image, access denied).</exception>
    /// <exception cref="ContainmentException">The job refused the child; it was terminated unrun.</exception>
    public static ChildProcess Start(string commandLine, IReadOnlyDictionary<string, string> environment, JobObject job)
    {
        var (inRead, inWrite) = CreatePipeFor(childReads: true);
        var (outRead, outWrite) = CreatePipeFor(childReads: false);
        var (errRead, errWrite) = CreatePipeFor(childReads: false);

        var si = new StartupInfo
        {
            Cb = Marshal.SizeOf<StartupInfo>(),
            Flags = StartfUseStdHandles,
            StdInput = inRead.DangerousGetHandle(),
            StdOutput = outWrite.DangerousGetHandle(),
            StdError = errWrite.DangerousGetHandle(),
        };

        // CreateProcessW may write to lpCommandLine, so it gets its own buffer.
        var cmd = Marshal.StringToHGlobalUni(commandLine);
        var env = Marshal.StringToHGlobalUni(BuildEnvironmentBlock(environment));
        ProcessInformation pi;
        try
        {
            if (!CreateProcessW(null, cmd, IntPtr.Zero, IntPtr.Zero, true, CreationFlags, env, null, ref si, out pi))
            {
                var error = Marshal.GetLastWin32Error();
                Dispose(inRead, inWrite, outRead, outWrite, errRead, errWrite);
                throw new Win32Exception(error);
            }
        }
        finally
        {
            Marshal.FreeHGlobal(cmd);
            Marshal.FreeHGlobal(env);
        }

        // The child holds its own copies now; the parent keeps only its ends.
        Dispose(inRead, outWrite, errWrite);

        if (!job.TryAssign(pi.Process))
        {
            var error = Marshal.GetLastWin32Error();
            _ = TerminateProcess(pi.Process, TaskRunner.InternalErrorExitCode);
            _ = CloseHandle(pi.Thread);
            _ = CloseHandle(pi.Process);
            Dispose(inWrite, outRead, errRead);
            throw new ContainmentException($"the job refused the child (Win32 error {error}); it was terminated before it ran");
        }

        if (ResumeThread(pi.Thread) == Infinite)
        {
            var error = Marshal.GetLastWin32Error();
            _ = TerminateProcess(pi.Process, TaskRunner.InternalErrorExitCode);
            _ = CloseHandle(pi.Thread);
            _ = CloseHandle(pi.Process);
            Dispose(inWrite, outRead, errRead);
            throw new Win32Exception(error);
        }

        _ = CloseHandle(pi.Thread);

        var utf8 = new UTF8Encoding(false);
        return new ChildProcess(
            pi.Process,
            pi.ProcessId,
            new StreamWriter(new FileStream(inWrite, FileAccess.Write, 1), utf8),
            new StreamReader(new FileStream(outRead, FileAccess.Read, 4096), utf8),
            new StreamReader(new FileStream(errRead, FileAccess.Read, 4096), utf8));
    }

    /// <summary>The child's exit code; valid after <see cref="WaitForExit(int)"/> returned true.</summary>
    public int ExitCode => GetExitCodeProcess(_process, out var code) ? unchecked((int)code) : TaskRunner.InternalErrorExitCode;

    /// <summary>Closes the child's stdin so a child that reads it sees end-of-file.</summary>
    public void CloseStandardInput()
    {
        _stdin?.Dispose();
        _stdin = null;
    }

    /// <summary>Waits up to <paramref name="milliseconds"/>; true when the child has exited.</summary>
    public bool WaitForExit(int milliseconds) => WaitForSingleObject(_process, unchecked((uint)milliseconds)) == WaitObject0;

    /// <summary>Waits until the child exits.</summary>
    public void WaitForExit() => _ = WaitForSingleObject(_process, Infinite);

    /// <summary>Best-effort kill of the child and its descendants (the job already covers them).</summary>
    public void TryKillTree()
    {
        try
        {
            using var p = Process.GetProcessById(Pid);
            p.Kill(entireProcessTree: true);
        }
        catch (Exception ex) when (ex is ArgumentException or InvalidOperationException or Win32Exception or NotSupportedException)
        {
            // Already gone.
        }
    }

    /// <inheritdoc />
    public void Dispose()
    {
        CloseStandardInput();
        StandardOutput.Dispose();
        StandardError.Dispose();
        if (_process != IntPtr.Zero)
        {
            _ = CloseHandle(_process);
            _process = IntPtr.Zero;
        }
    }

    /// <summary>
    /// The Unicode environment block: the launcher's variables plus <paramref name="extra"/>, sorted
    /// case-insensitively as CreateProcess expects, each <c>name=value</c> NUL-terminated, then a final NUL.
    /// </summary>
    internal static string BuildEnvironmentBlock(IReadOnlyDictionary<string, string> extra)
    {
        var vars = new SortedDictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        foreach (DictionaryEntry e in Environment.GetEnvironmentVariables())
        {
            if (e.Key is string k && k.Length > 0)
            {
                vars[k] = e.Value as string ?? string.Empty;
            }
        }

        foreach (var (k, v) in extra)
        {
            vars[k] = v;
        }

        var sb = new StringBuilder();
        foreach (var (k, v) in vars)
        {
            _ = sb.Append(k).Append('=').Append(v).Append('\0');
        }

        return sb.Append('\0').ToString();
    }

    private static (SafeFileHandle Read, SafeFileHandle Write) CreatePipeFor(bool childReads)
    {
        var sa = new SecurityAttributes { Length = Marshal.SizeOf<SecurityAttributes>(), InheritHandle = 1 };
        if (!CreatePipe(out var read, out var write, ref sa, 0))
        {
            throw new Win32Exception(Marshal.GetLastWin32Error());
        }

        var r = new SafeFileHandle(read, ownsHandle: true);
        var w = new SafeFileHandle(write, ownsHandle: true);

        // Only the child's end may be inherited; the parent's end must not leak into the child.
        var parentEnd = childReads ? w : r;
        if (!SetHandleInformation(parentEnd.DangerousGetHandle(), HandleFlagInherit, 0))
        {
            var error = Marshal.GetLastWin32Error();
            r.Dispose();
            w.Dispose();
            throw new Win32Exception(error);
        }

        return (r, w);
    }

    private static void Dispose(params SafeFileHandle[] handles)
    {
        foreach (var h in handles)
        {
            h.Dispose();
        }
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct SecurityAttributes
    {
        public int Length;
        public IntPtr SecurityDescriptor;
        public int InheritHandle;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct StartupInfo
    {
        public int Cb;
        public IntPtr Reserved;
        public IntPtr Desktop;
        public IntPtr Title;
        public uint X;
        public uint Y;
        public uint XSize;
        public uint YSize;
        public uint XCountChars;
        public uint YCountChars;
        public uint FillAttribute;
        public uint Flags;
        public ushort ShowWindow;
        public ushort Reserved2Size;
        public IntPtr Reserved2;
        public IntPtr StdInput;
        public IntPtr StdOutput;
        public IntPtr StdError;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct ProcessInformation
    {
        public IntPtr Process;
        public IntPtr Thread;
        public int ProcessId;
        public int ThreadId;
    }

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool CreateProcessW(
        string? applicationName,
        IntPtr commandLine,
        IntPtr processAttributes,
        IntPtr threadAttributes,
        [MarshalAs(UnmanagedType.Bool)] bool inheritHandles,
        uint creationFlags,
        IntPtr environment,
        string? currentDirectory,
        ref StartupInfo startupInfo,
        out ProcessInformation processInformation);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool CreatePipe(out IntPtr readPipe, out IntPtr writePipe, ref SecurityAttributes attributes, uint size);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool SetHandleInformation(IntPtr handle, uint mask, uint flags);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern uint ResumeThread(IntPtr thread);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool TerminateProcess(IntPtr process, int exitCode);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern uint WaitForSingleObject(IntPtr handle, uint milliseconds);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool GetExitCodeProcess(IntPtr process, out uint exitCode);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool CloseHandle(IntPtr handle);
}
