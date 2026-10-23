// Private source proposal. Adapted from frozen UniFi OwnedNativeObserver E4F6B456.
// No compilation or execution is implied by this file.
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Win32.SafeHandles;

namespace PcaiCarbonQualificationR4 {
    public sealed class Observation {
        public string Reason, PrimaryFailure, CustodyId, StartedUtc, BirthUtc, Disposition;
        public string StdoutFault, StderrFault, StdoutFailureCode, StderrFailureCode;
        public int CleanupAttempts;
        public readonly List<string> SecondaryFailures = new List<string>();
        public int? OriginalPid, ExitCode;
        public long ElapsedMilliseconds, StdoutBytes, StderrBytes;
        public bool Started, ProcessHandleCaptured, JobAssigned, ParentExitConfirmed, JobEmptyConfirmed;
        public bool DrainsSettled, DrainsSucceeded, FilesFlushed, ResourcesDisposed, OriginalHandleDisposed;
        public bool ChildHoldRetained; // Sticky receipt marker; closure never promotes the held operation.
    }
    public sealed class Recovery {
        public string CustodyId, OriginalPrimaryFailure;
        public bool ClosureConfirmed;
        public long RecoveryWallMilliseconds;
        public Observation LatestObservation;
    }
    public sealed class StreamLimitException : IOException {
        public readonly string StreamName;
        public StreamLimitException(string name) : base("Owned "+name+" exceeded its finite byte limit.") { StreamName=name; }
    }
    public static class Supervisor {
        public const int ContractVersion = 4;
        public static string SourceBinding { get; private set; }
        public static void BindSource(string hash) {
            if (SourceBinding != null && SourceBinding != hash) throw new InvalidOperationException("Different supervisor source is loaded.");
            SourceBinding = hash;
        }
        [StructLayout(LayoutKind.Sequential)] struct BasicLimits {
            public long ProcessTime, JobTime; public uint Flags;
            public UIntPtr MinimumWorkingSet, MaximumWorkingSet;
            public uint ActiveProcessLimit; public UIntPtr Affinity; public uint Priority, SchedulingClass;
        }
        [StructLayout(LayoutKind.Sequential)] struct IoCounters { public ulong ReadOperations, WriteOperations, OtherOperations, ReadBytes, WriteBytes, OtherBytes; }
        [StructLayout(LayoutKind.Sequential)] struct ExtendedLimits { public BasicLimits Basic; public IoCounters Io; public UIntPtr ProcessMemory, JobMemory, PeakProcessMemory, PeakJobMemory; }
        [StructLayout(LayoutKind.Sequential)] struct Accounting { public long UserTime, KernelTime, PeriodUserTime, PeriodKernelTime; public uint PageFaults, TotalProcesses, ActiveProcesses, TerminatedProcesses; }
        [DllImport("kernel32.dll", SetLastError=true)] static extern IntPtr CreateJobObject(IntPtr security, string name);
        [DllImport("kernel32.dll", SetLastError=true)] static extern bool SetInformationJobObject(SafeFileHandle job, int kind, ref ExtendedLimits limits, int size);
        [DllImport("kernel32.dll", SetLastError=true)] static extern bool AssignProcessToJobObject(SafeFileHandle job, SafeProcessHandle process);
        [DllImport("kernel32.dll", SetLastError=true)] static extern bool TerminateJobObject(SafeFileHandle job, uint code);
        [DllImport("kernel32.dll", SetLastError=true)] static extern bool QueryInformationJobObject(SafeFileHandle job, int kind, out Accounting information, int size, IntPtr returned);
        [DllImport("kernel32.dll", SetLastError=true)] static extern uint WaitForSingleObject(SafeProcessHandle process, uint milliseconds);
        [DllImport("kernel32.dll", SetLastError=true)] static extern bool TerminateProcess(SafeProcessHandle process, uint code);
        sealed class Custody {
            public Process Process; public SafeProcessHandle ProcessHandle; public SafeFileHandle Job;
            public Observation Observation; public bool Started, Assigned, ChildHoldRetained;
            public FileStream Output, Error; public Task<long> OutputTask, ErrorTask;
            public Task InputTask, InputFlush, OutputFlush, ErrorFlush; public readonly CancellationTokenSource Cancellation = new CancellationTokenSource();
        }
        static readonly object Gate = new object();
        static readonly Dictionary<string, Custody> Retained = new Dictionary<string, Custody>();
        static bool Running;
        static void Check(bool result, string action) { if (!result) throw new Win32Exception(Marshal.GetLastWin32Error(), action); }
        static int Remaining(Stopwatch clock, long limit) { return (int)Math.Max(0, Math.Min(int.MaxValue, limit-clock.ElapsedMilliseconds)); }
        static bool Completed(Task task, Stopwatch clock, long limit) {
            if (task == null) return true;
            if (!task.IsCompleted) task.Wait(Remaining(clock, limit));
            return task.IsCompleted;
        }
        static bool Settled(Task task, Stopwatch clock, long limit) {
            try { return Completed(task,clock,limit); } catch(AggregateException) { return task.IsCompleted; }
        }
        static async Task<long> Drain(Stream source, FileStream destination, long maximum, string streamName, CancellationToken cancellation) {
            byte[] buffer = new byte[16384]; long total=0;
            for (;;) {
                int count=await source.ReadAsync(buffer,0,buffer.Length,cancellation).ConfigureAwait(false);
                if (count==0) return total;
                int writable=(int)Math.Min(count,maximum-total);
                if (writable>0) { await destination.WriteAsync(buffer,0,writable,cancellation).ConfigureAwait(false); total+=writable; }
                if (writable!=count) throw new StreamLimitException(streamName);
            }
        }
        static bool CaptureDrain(Task<long> task, string name, Observation observation) {
            if(task==null) return true;
            if(!task.IsCompleted) return false;
            try {
                long bytes=task.GetAwaiter().GetResult();
                if(name=="stdout") observation.StdoutBytes=bytes; else observation.StderrBytes=bytes;
                return true;
            } catch(Exception error) {
                string code=error is StreamLimitException ? "STREAM_BYTE_LIMIT" : "DRAIN_FAULT";
                if(name=="stdout") { observation.StdoutFault=error.ToString(); observation.StdoutFailureCode=code; }
                else { observation.StderrFault=error.ToString(); observation.StderrFailureCode=code; }
                observation.SecondaryFailures.Add(name+" drain: "+error);
                return false;
            }
        }
        static void CaptureInput(Task task, string name, Observation observation) {
            if(task==null || !task.IsCompleted) return;
            try { task.GetAwaiter().GetResult(); } catch(Exception error) { observation.SecondaryFailures.Add(name+": "+error); }
        }
        static void Cleanup(Custody custody, int limitMilliseconds) {
            Observation observation=custody.Observation;
            observation.CleanupAttempts++;
            var cleanup=Stopwatch.StartNew();
            try {
                if(!custody.ChildHoldRetained) {
                if(custody.Assigned && custody.Job!=null && !custody.Job.IsClosed) Check(TerminateJobObject(custody.Job,1),"Retire only exact retained private job");
                else if(custody.Started && !observation.ParentExitConfirmed) {
                    if(custody.ProcessHandle!=null && !custody.ProcessHandle.IsClosed) {
                        if(WaitForSingleObject(custody.ProcessHandle,0)==0x102) Check(TerminateProcess(custody.ProcessHandle,1),"Retire exact retained handshake parent");
                    } else if(custody.ProcessHandle==null && !custody.Process.HasExited) custody.Process.Kill();
                }
                }
            } catch(Exception error) { observation.SecondaryFailures.Add("Termination: "+error); }
            try {
                if(!observation.ParentExitConfirmed) observation.ParentExitConfirmed=!custody.Started ||
                    (custody.ProcessHandle!=null ? (!custody.ProcessHandle.IsClosed && WaitForSingleObject(custody.ProcessHandle,(uint)Remaining(cleanup,limitMilliseconds))==0) : custody.Process.WaitForExit(Remaining(cleanup,limitMilliseconds)));
                if(!observation.ParentExitConfirmed) throw new TimeoutException("Original owned parent closure unconfirmed.");
                if(custody.Started && observation.ExitCode==null) observation.ExitCode=custody.Process.ExitCode;
            } catch(Exception error) { observation.SecondaryFailures.Add("Parent closure: "+error); }
            try {
                if(!custody.Assigned) observation.JobEmptyConfirmed=true;
                while(!observation.JobEmptyConfirmed) {
                    if(custody.Job==null || custody.Job.IsClosed) throw new InvalidOperationException("Original job closed before empty proof; do not reconstruct identity.");
                    Accounting accounting;
                    Check(QueryInformationJobObject(custody.Job,1,out accounting,Marshal.SizeOf<Accounting>(),IntPtr.Zero),"Recheck exact retained job count");
                    if(accounting.ActiveProcesses==0) { observation.JobEmptyConfirmed=true; break; }
                    int remaining=Remaining(cleanup,limitMilliseconds);
                    if(remaining==0) throw new TimeoutException("Original private job descendants closure unconfirmed.");
                    Thread.Sleep(Math.Min(20,remaining));
                }
            } catch(Exception error) { observation.SecondaryFailures.Add("Job closure: "+error); }
            bool outputSettled=Settled(custody.OutputTask,cleanup,limitMilliseconds);
            bool errorSettled=Settled(custody.ErrorTask,cleanup,limitMilliseconds);
            bool inputSettled=Settled(custody.InputTask,cleanup,limitMilliseconds);
            bool inputFlushSettled=Settled(custody.InputFlush,cleanup,limitMilliseconds);
            observation.DrainsSettled=outputSettled && errorSettled && inputSettled && inputFlushSettled;
            // Deliberately independent calls: stdout failure cannot hide stderr failure.
            bool outputOk=CaptureDrain(custody.OutputTask,"stdout",observation);
            bool errorOk=CaptureDrain(custody.ErrorTask,"stderr",observation);
            CaptureInput(custody.InputTask,"stdin write",observation);
            CaptureInput(custody.InputFlush,"stdin flush",observation);
            observation.DrainsSucceeded=outputOk && errorOk;
            if(!observation.DrainsSettled) {
                observation.SecondaryFailures.Add("Owned stream tasks remain unresolved.");
                try { custody.Cancellation.Cancel(); } catch(Exception error) { observation.SecondaryFailures.Add("Cancel: "+error); }
            }
            if(observation.ParentExitConfirmed && observation.JobEmptyConfirmed && observation.DrainsSettled) {
                try {
                    if(!observation.FilesFlushed) {
                        if(custody.Output!=null && custody.OutputFlush==null) custody.OutputFlush=custody.Output.FlushAsync();
                        if(custody.Error!=null && custody.ErrorFlush==null) custody.ErrorFlush=custody.Error.FlushAsync();
                        bool outputFlushSettled=Settled(custody.OutputFlush,cleanup,limitMilliseconds);
                        bool errorFlushSettled=Settled(custody.ErrorFlush,cleanup,limitMilliseconds);
                        if(!outputFlushSettled || !errorFlushSettled) throw new TimeoutException("Original owned file flush unconfirmed.");
                        // Independently preserve both file-flush exceptions too.
                        bool outputFlushOk=true,errorFlushOk=true;
                        try { if(custody.OutputFlush!=null) custody.OutputFlush.GetAwaiter().GetResult(); } catch(Exception error) { outputFlushOk=false; observation.SecondaryFailures.Add("stdout flush: "+error); }
                        try { if(custody.ErrorFlush!=null) custody.ErrorFlush.GetAwaiter().GetResult(); } catch(Exception error) { errorFlushOk=false; observation.SecondaryFailures.Add("stderr flush: "+error); }
                        if(!outputFlushOk || !errorFlushOk) throw new IOException("Original file flush failed; custody retained.");
                        observation.FilesFlushed=true;
                    }
                    if(custody.Output!=null) custody.Output.Dispose();
                    if(custody.Error!=null) custody.Error.Dispose();
                    if(custody.Job!=null) custody.Job.Dispose();
                    if(custody.Process!=null) custody.Process.Dispose();
                    custody.Cancellation.Dispose(); observation.ResourcesDisposed=true;
                    observation.OriginalHandleDisposed=custody.ProcessHandle==null || custody.ProcessHandle.IsClosed;
                } catch(Exception error) { observation.SecondaryFailures.Add("Resource disposal: "+error); }
            }
            observation.Disposition=observation.ResourcesDisposed && observation.OriginalHandleDisposed && observation.ParentExitConfirmed && observation.JobEmptyConfirmed && observation.DrainsSettled ? "CLOSED" : "HOLD";
        }
        // Nonblocking observation of the retained original, including partial handle capture.
        // A query failure is not exit proof; callers retain the actual Process and all custody.
        static bool ObserveOriginalParentExit(Custody custody) {
            Observation observation=custody.Observation;
            if(!custody.Started) return true;
            if(!observation.ParentExitConfirmed) {
                bool exited;
                if(custody.ProcessHandle!=null) {
                    if(custody.ProcessHandle.IsClosed) throw new InvalidOperationException("Original child handle closed before exit proof; no identity reconstruction.");
                    uint state=WaitForSingleObject(custody.ProcessHandle,0);
                    if(state==0x102) return false;
                    if(state!=0) throw new Win32Exception(Marshal.GetLastWin32Error(),"Observe original managed child handle");
                    exited=true;
                } else {
                    if(custody.Process==null) throw new InvalidOperationException("Original started Process unavailable; no identity reconstruction.");
                    exited=custody.Process.HasExited;
                }
                if(!exited) return false;
                observation.ParentExitConfirmed=true;
            }
            if(observation.ExitCode==null) observation.ExitCode=custody.Process.ExitCode;
            return true;
        }
        // No termination, cancellation or disposal while a retained managed child is alive.
        // Recovery authority is these retained original handles, never PID metadata or a new handle.
        static bool ObserveChildHoldClosure(Custody custody) {
            Observation observation=custody.Observation;
            if(!custody.Started) throw new InvalidOperationException("Retained child HOLD requires the original started Process.");
            if(!ObserveOriginalParentExit(custody)) return false;
            // An unassigned readiness-gated child cannot launch work; never reconstruct a handle.
            if(!custody.Assigned) observation.JobEmptyConfirmed=true;
            if(!observation.JobEmptyConfirmed) {
                if(custody.Job==null || custody.Job.IsClosed) throw new InvalidOperationException("Original retained child job unavailable before empty proof.");
                Accounting accounting;
                Check(QueryInformationJobObject(custody.Job,1,out accounting,Marshal.SizeOf<Accounting>(),IntPtr.Zero),"Observe retained child HOLD original job count");
                if(accounting.ActiveProcesses!=0) return false;
                observation.JobEmptyConfirmed=true;
            }
            return true;
        }
        public static Recovery RetryOwnedCustody(string custodyId, int cleanupMilliseconds) {
            if(cleanupMilliseconds<1 || cleanupMilliseconds>15000) throw new ArgumentOutOfRangeException("Finite recovery limit required.");
            Custody custody;
            lock(Gate) {
                if(Running) throw new InvalidOperationException("Concurrent run/recovery refused.");
                if(!Retained.TryGetValue(custodyId,out custody)) throw new ArgumentException("Unknown exact-owned custody identity.");
                Running=true;
            }
            var clock=Stopwatch.StartNew();
            string originalPrimary=custody.Observation.PrimaryFailure;
            try {
                bool ready=true;
                if(custody.ChildHoldRetained) {
                    try { ready=ObserveChildHoldClosure(custody); }
                    catch(Exception error) { ready=false; custody.Observation.SecondaryFailures.Add("Retained child HOLD observation: "+error); }
                }
                if(ready) Cleanup(custody,cleanupMilliseconds);
                else custody.Observation.Disposition="HOLD";
                bool closed=custody.Observation.Disposition=="CLOSED";
                if(closed) { lock(Gate) Retained.Remove(custodyId); }
                return new Recovery { CustodyId=custodyId,OriginalPrimaryFailure=originalPrimary,ClosureConfirmed=closed,RecoveryWallMilliseconds=clock.ElapsedMilliseconds,LatestObservation=custody.Observation };
            } finally { lock(Gate) Running=false; }
        }
        public static string[] PendingCustody() { lock(Gate) { var keys=new string[Retained.Count]; Retained.Keys.CopyTo(keys,0); return keys; } }
        public static Observation Run(string application, string[] arguments, string directory, string outputPath, string errorPath, int timeoutSeconds, long maximumBytes) {
            return Run(application,arguments,directory,outputPath,errorPath,timeoutSeconds,maximumBytes,null);
        }
        public static Observation Run(string application, string[] arguments, string directory, string outputPath, string errorPath, int timeoutSeconds, long maximumBytes, string ownedHoldSignalPath) {
            if (timeoutSeconds<1 || timeoutSeconds>900 || maximumBytes<1 || maximumBytes>16777216) throw new ArgumentOutOfRangeException("Finite supervisor limits required.");
            lock(Gate) { if(Running || Retained.Count!=0) throw new InvalidOperationException("Active or unconfirmed exact-owned custody refuses launch."); Running=true; }
            var observation=new Observation { StartedUtc=DateTime.UtcNow.ToString("o") };
            var custody=new Custody { Observation=observation };
            var clock=Stopwatch.StartNew(); long deadline=timeoutSeconds*1000L;
            try {
                custody.Output=new FileStream(outputPath,FileMode.CreateNew,FileAccess.Write,FileShare.Read,16384,FileOptions.Asynchronous);
                custody.Error=new FileStream(errorPath,FileMode.CreateNew,FileAccess.Write,FileShare.Read,16384,FileOptions.Asynchronous);
                IntPtr raw=CreateJobObject(IntPtr.Zero,null);
                if(raw==IntPtr.Zero) throw new Win32Exception(Marshal.GetLastWin32Error(),"Create private outer job");
                custody.Job=new SafeFileHandle(raw,true);
                var limits=new ExtendedLimits(); limits.Basic.Flags=0x2000; // KILL_ON_JOB_CLOSE; no breakaway flags.
                Check(SetInformationJobObject(custody.Job,9,ref limits,Marshal.SizeOf<ExtendedLimits>()),"Set private job kill-on-close");
                if(Remaining(clock,deadline)==0) throw new TimeoutException("Deadline before owned launch.");
                var info=new ProcessStartInfo(application) { UseShellExecute=false,CreateNoWindow=true,RedirectStandardInput=true,RedirectStandardOutput=true,RedirectStandardError=true,WorkingDirectory=directory };
                // Child-only PATH makes unchanged maintained Get-Command pwsh
                // resolve the admitted physical interpreter. No parent/global edit.
                info.Environment["PATH"]=Path.GetDirectoryName(application)+Path.PathSeparator+info.Environment["PATH"];
                foreach(string argument in arguments) info.ArgumentList.Add(argument);
                custody.Process=new Process { StartInfo=info };
                if(!custody.Process.Start()) throw new InvalidOperationException("Owned handshake stage did not start.");
                custody.Started=true; observation.Started=true; custody.ProcessHandle=custody.Process.SafeHandle; observation.ProcessHandleCaptured=true;
                observation.OriginalPid=custody.Process.Id; observation.BirthUtc=custody.Process.StartTime.ToUniversalTime().ToString("o");
                custody.OutputTask=Drain(custody.Process.StandardOutput.BaseStream,custody.Output,maximumBytes,"stdout",custody.Cancellation.Token);
                custody.ErrorTask=Drain(custody.Process.StandardError.BaseStream,custody.Error,maximumBytes,"stderr",custody.Cancellation.Token);
                // Frozen stage may spawn nothing until this original SafeHandle is job-assigned and released.
                Check(AssignProcessToJobObject(custody.Job,custody.ProcessHandle),"Assign exact original stage handle");
                custody.Assigned=true; observation.JobAssigned=true;
                if(Remaining(clock,deadline)==0) throw new TimeoutException("Deadline during owned launch/assignment.");
                custody.InputTask=custody.Process.StandardInput.WriteLineAsync("PCAI_CARBON_HOOK_READY");
                if(!Completed(custody.InputTask,clock,deadline)) throw new TimeoutException("Owned readiness write exceeded aggregate deadline.");
                custody.InputTask.GetAwaiter().GetResult();
                custody.InputFlush=custody.Process.StandardInput.FlushAsync();
                if(!Completed(custody.InputFlush,clock,deadline)) throw new TimeoutException("Owned readiness flush exceeded aggregate deadline.");
                custody.InputFlush.GetAwaiter().GetResult(); custody.Process.StandardInput.Close();
                for(;;) {
                    if(custody.OutputTask.IsFaulted || custody.ErrorTask.IsFaulted) throw new IOException("Owned output drainage failed.");
                    if(custody.OutputTask.IsCanceled || custody.ErrorTask.IsCanceled) throw new IOException("Owned output drainage cancelled.");
                    int remaining=Remaining(clock,deadline);
                    if(remaining==0) throw new TimeoutException("Owned aggregate deadline exceeded.");
                    // Only a fresh exact child-host private marker supplied by its retained owner parent.
                    if(ownedHoldSignalPath!=null && File.Exists(ownedHoldSignalPath)) { custody.ChildHoldRetained=true; observation.ChildHoldRetained=true; observation.Reason="CHILD_HOLD"; break; }
                    uint state=WaitForSingleObject(custody.ProcessHandle,0);
                    if(state==0) { observation.Reason="EXIT"; break; }
                    if(state!=0x102) throw new Win32Exception(Marshal.GetLastWin32Error(),"Observe original owned SafeHandle");
                    Thread.Sleep(Math.Min(50,remaining));
                }
            } catch(Exception error) { observation.Reason="SUPERVISOR_ERROR"; observation.PrimaryFailure=error.ToString(); }
            finally {
                try {
                    // A nonnull marker path declares managed-resource handoff. Marker IO is
                    // optional early notification, never a prerequisite for preserving a live child.
                    if(ownedHoldSignalPath!=null && custody.Started && !custody.ChildHoldRetained) {
                        bool exited=false;
                        try { exited=ObserveOriginalParentExit(custody); }
                        catch(Exception error) { observation.SecondaryFailures.Add("Managed child exit observation: "+error); }
                        if(!exited) { custody.ChildHoldRetained=true; observation.ChildHoldRetained=true; }
                    }
                    if(custody.ChildHoldRetained) observation.Disposition="HOLD";
                    else Cleanup(custody,15000);
                }
                catch(Exception error) { observation.Disposition="HOLD"; observation.SecondaryFailures.Add("Unexpected cleanup: "+error); }
                finally {
                    try {
                        if(observation.Disposition!="CLOSED") { observation.CustodyId=Guid.NewGuid().ToString("N"); lock(Gate) Retained.Add(observation.CustodyId,custody); }
                    } finally {
                        observation.ElapsedMilliseconds=clock.ElapsedMilliseconds;
                        lock(Gate) Running=false;
                    }
                }
            }
            if(observation.SecondaryFailures.Count!=0 && observation.PrimaryFailure==null) observation.Reason="CLEANUP_ERROR";
            return observation;
        }
    }
}
