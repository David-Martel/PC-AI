using System.Diagnostics;
using System.Runtime.InteropServices;

namespace DtmTaskRun;

/// <summary>
/// A Windows Job Object that owns the child's process tree.
/// </summary>
/// <remarks>
/// While the launcher runs the job has <c>KILL_ON_JOB_CLOSE</c> set, so if Task Scheduler (or
/// anything else) kills the launcher, the whole tree dies with it instead of being orphaned and
/// then duplicated by the next trigger. When the child exits normally the flag is cleared before
/// the handle closes: tasks such as LspmuxServer deliberately start a detached daemon and exit,
/// and that daemon must survive exactly as it did under <c>conhost --headless</c>.
/// </remarks>
internal sealed class JobObject : IDisposable
{
    private const int JobObjectExtendedLimitInformation = 9;
    private const uint JobObjectLimitKillOnJobClose = 0x2000;

    private IntPtr _handle;

    private JobObject(IntPtr handle) => _handle = handle;

    /// <summary>Creates a kill-on-close job, or null when the OS refuses (the launcher then runs without one).</summary>
    public static JobObject? TryCreate()
    {
        var h = CreateJobObjectW(IntPtr.Zero, null);
        if (h == IntPtr.Zero)
        {
            return null;
        }

        var job = new JobObject(h);
        return job.SetKillOnClose(true) ? job : Discard(job);
    }

    /// <summary>Puts <paramref name="process"/> (and so its future children) in the job.</summary>
    public bool TryAssign(Process process)
    {
        try
        {
            return TryAssign(process.Handle);
        }
        catch (InvalidOperationException)
        {
            return false; // the child already exited
        }
    }

    /// <summary>
    /// Puts the process behind <paramref name="processHandle"/> in the job. <see cref="ChildProcess"/> calls
    /// this while the child is still suspended, so the child and everything it starts are in the job from
    /// its first instruction. On failure, <c>Marshal.GetLastWin32Error</c> holds the reason.
    /// </summary>
    public bool TryAssign(IntPtr processHandle) => _handle != IntPtr.Zero && AssignProcessToJobObject(_handle, processHandle);

    /// <summary>Kills every process in the job.</summary>
    public void Terminate(uint exitCode)
    {
        if (_handle != IntPtr.Zero)
        {
            _ = TerminateJobObject(_handle, exitCode);
        }
    }

    /// <summary>Lets surviving members outlive the launcher (normal-exit path).</summary>
    public void ReleaseSurvivors() => SetKillOnClose(false);

    /// <inheritdoc />
    public void Dispose()
    {
        if (_handle != IntPtr.Zero)
        {
            _ = CloseHandle(_handle);
            _handle = IntPtr.Zero;
        }
    }

    private static JobObject? Discard(JobObject job)
    {
        job.Dispose();
        return null;
    }

    private bool SetKillOnClose(bool on)
    {
        var info = new JobObjectExtendedLimitInfo();
        info.BasicLimitInformation.LimitFlags = on ? JobObjectLimitKillOnJobClose : 0;
        var size = Marshal.SizeOf<JobObjectExtendedLimitInfo>();
        var ptr = Marshal.AllocHGlobal(size);
        try
        {
            Marshal.StructureToPtr(info, ptr, false);
            return SetInformationJobObject(_handle, JobObjectExtendedLimitInformation, ptr, (uint)size);
        }
        finally
        {
            Marshal.FreeHGlobal(ptr);
        }
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct JobObjectBasicLimitInfo
    {
        public long PerProcessUserTimeLimit;
        public long PerJobUserTimeLimit;
        public uint LimitFlags;
        public UIntPtr MinimumWorkingSetSize;
        public UIntPtr MaximumWorkingSetSize;
        public uint ActiveProcessLimit;
        public UIntPtr Affinity;
        public uint PriorityClass;
        public uint SchedulingClass;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct IoCounters
    {
        public ulong ReadOperationCount;
        public ulong WriteOperationCount;
        public ulong OtherOperationCount;
        public ulong ReadTransferCount;
        public ulong WriteTransferCount;
        public ulong OtherTransferCount;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct JobObjectExtendedLimitInfo
    {
        public JobObjectBasicLimitInfo BasicLimitInformation;
        public IoCounters IoInfo;
        public UIntPtr ProcessMemoryLimit;
        public UIntPtr JobMemoryLimit;
        public UIntPtr PeakProcessMemoryUsed;
        public UIntPtr PeakJobMemoryUsed;
    }

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern IntPtr CreateJobObjectW(IntPtr attributes, string? name);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool SetInformationJobObject(IntPtr job, int infoClass, IntPtr info, uint size);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool TerminateJobObject(IntPtr job, uint exitCode);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool CloseHandle(IntPtr handle);
}
