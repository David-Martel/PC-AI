// Private candidate; not compiled or executed. Reviewed MCP R3 ownership lineage
// is preserved under references. This capture API does not call that relay.
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Win32.SafeHandles;

namespace Pcai.Common {
    public sealed class CaptureCleanupReceipt {
        public readonly string Closure, PrimaryFailure, SourceBinding, CustodyId;
        public readonly uint? OriginalPid;
        public readonly int? OriginalExitCode;
        public readonly bool TimedOut, ParentExited, JobEmpty, ReadersSettled, HandlesClosed;
        public readonly long CleanupElapsedMilliseconds, OperationElapsedMilliseconds, RepeatedFailures;
        public readonly int CleanupBudgetMilliseconds;
        public readonly ReadOnlyCollection<string> SecondaryFailures;
        internal CaptureCleanupReceipt(CaptureResult value, long cleanupElapsed, string binding) {
            Closure = value.Closure; PrimaryFailure = value.PrimaryFailure; SourceBinding = binding; CustodyId = value.CustodyId; OriginalPid = value.OriginalPid; TimedOut = value.TimedOut;
            OriginalExitCode = value.OriginalExitCode;
            ParentExited = value.ParentExited; JobEmpty = value.JobEmpty; ReadersSettled = value.ReadersSettled; HandlesClosed = value.HandlesClosed;
            CleanupElapsedMilliseconds = cleanupElapsed; OperationElapsedMilliseconds = value.ElapsedMilliseconds;
            RepeatedFailures = value.RepeatedFailures; CleanupBudgetMilliseconds = ProcessCaptureV1.CleanupMilliseconds;
            SecondaryFailures = new List<string>(value.SecondaryFailures).AsReadOnly();
        }
    }
    public sealed class CaptureResult {
        public bool Success, TimedOut;
        public int ExitCode;
        public uint? OriginalPid;
        public int? OriginalExitCode;
        public string StdOut = "", StdErr = "", Closure = "NOT_STARTED", CustodyId;
        public string PrimaryFailure;
        public string[] SecondaryFailures = new string[0];
        public long RepeatedFailures;
        public long StdoutBytes, StderrBytes, ElapsedMilliseconds;
        public bool AssignedBeforeResume, ParentExited, JobEmpty, ReadersSettled, HandlesClosed;
        public bool CleanupContinuationUsed;
        public CaptureCleanupReceipt FirstCleanup;
    }
    public sealed class CaptureOperation {
        internal CancellationTokenSource Cancellation;
        internal volatile bool Closed;
        internal Thread Controller;
        volatile CaptureCleanupReceipt firstCleanup;
        public CaptureCleanupReceipt FirstCleanup { get { return firstCleanup; } internal set { firstCleanup = value; } }
        public Task<CaptureResult> Completion { get; internal set; }
        public void Cancel() {
            if (Closed || (Completion != null && Completion.IsCompleted)) return;
            if (Cancellation != null) {
                try { Cancellation.Cancel(); }
                catch (ObjectDisposedException) { if (!Closed && (Completion == null || !Completion.IsCompleted)) throw; }
            }
        }
    }
    public static class ProcessCaptureV1 {
        public const int ContractVersion = 1;
        public const int StdoutLimit = 8 * 1024 * 1024, StderrLimit = 1024 * 1024, CleanupMilliseconds = 1000;
        public static string SourceBinding { get; private set; }
        static readonly object Gate = new object();
        static readonly Dictionary<string, Custody> Pending = new Dictionary<string, Custody>();
        static bool Running;
        static Custody Active;
        sealed class OperationDeadlineException : TimeoutException { internal OperationDeadlineException() : base("Capture operation deadline exceeded.") { } }
        public static void BindSource(string hash) {
            lock (Gate) {
                if (SourceBinding != null && !String.Equals(SourceBinding, hash, StringComparison.Ordinal)) throw new InvalidOperationException("Different capture source is already loaded; use a fresh process.");
                if (hash == null || hash.Length != 64) throw new ArgumentException("Exact source hash is required.");
                SourceBinding = hash;
            }
        }
        sealed class Custody {
            internal SafeFileHandle Job, Thread, Input, OutRead, OutWrite, ErrRead, ErrWrite;
            internal SafeProcessHandle Process;
            internal Reader Output, Error;
            internal bool Started, Assigned, Resumed, ParentExited, JobEmpty, TimedOut;
            internal volatile bool Closed;
            internal int? ExitCode;
            internal uint? OriginalPid;
            internal readonly Stopwatch Clock = Stopwatch.StartNew();
            internal readonly object FailureGate = new object();
            internal readonly object CleanupGate = new object();
            internal Exception Primary;
            internal readonly List<Exception> Secondary = new List<Exception>();
            internal long RepeatedFailures;
            internal CaptureOperation Operation;
            internal string Id;
            internal bool ContinuationRequired;
            internal void Fail(Exception failure) {
                lock (FailureGate) {
                    if (Primary == null) { Primary = failure; TimedOut = failure is OperationDeadlineException; return; }
                    if (SameFailure(Primary, failure)) { if (RepeatedFailures < Int64.MaxValue) RepeatedFailures++; return; }
                    foreach (Exception prior in Secondary) if (SameFailure(prior, failure)) { if (RepeatedFailures < Int64.MaxValue) RepeatedFailures++; return; }
                    Secondary.Add(failure);
                }
            }
            static bool SameFailure(Exception a, Exception b) { return Object.ReferenceEquals(a, b) || (a.GetType() == b.GetType() && a.HResult == b.HResult && a.Message == b.Message); }
            internal bool Failed { get { lock (FailureGate) return Primary != null; } }
        }
        sealed class Buffer {
            readonly int limit;
            readonly List<byte[]> chunks = new List<byte[]>();
            internal int Length { get; private set; }
            internal Buffer(int maximum) { limit = maximum; }
            internal void Append(byte[] bytes, int count) {
                if (count < 0 || count > bytes.Length || count > limit - Length) throw new IOException("Owned output exceeded its finite byte limit.");
                byte[] owned = new byte[count]; System.Buffer.BlockCopy(bytes, 0, owned, 0, count); chunks.Add(owned); Length = checked(Length + count);
            }
            internal string Decode() {
                // No flattening allocation; StreamReader preserves predecessor BOM detection.
                using (var stream = new ChunkStream(chunks))
                using (var reader = new StreamReader(stream, Encoding.UTF8, true, 8192)) return reader.ReadToEnd().Trim();
            }
        }
        sealed class ChunkStream : Stream {
            readonly List<byte[]> chunks; int index, offset;
            internal ChunkStream(List<byte[]> value) { chunks = value; }
            public override bool CanRead { get { return true; } }
            public override bool CanSeek { get { return false; } }
            public override bool CanWrite { get { return false; } }
            public override long Length { get { throw new NotSupportedException(); } }
            public override long Position { get { throw new NotSupportedException(); } set { throw new NotSupportedException(); } }
            public override int Read(byte[] bytes, int start, int maximum) {
                int copied = 0;
                while (copied < maximum && index < chunks.Count) {
                    byte[] chunk = chunks[index]; int count = Math.Min(maximum - copied, chunk.Length - offset);
                    System.Buffer.BlockCopy(chunk, offset, bytes, start + copied, count); copied += count; offset += count;
                    if (offset == chunk.Length) { index++; offset = 0; }
                }
                return copied;
            }
            public override void Flush() { }
            public override long Seek(long offsetValue, SeekOrigin origin) { throw new NotSupportedException(); }
            public override void SetLength(long value) { throw new NotSupportedException(); }
            public override void Write(byte[] bytes, int start, int count) { throw new NotSupportedException(); }
        }
        sealed class Reader {
            readonly Custody custody; readonly SafeFileHandle pipe; readonly string name; readonly int deadline;
            readonly Thread worker; readonly ManualResetEventSlim ready = new ManualResetEventSlim(false);
            SafeFileHandle thread; int cancelled; bool readyClosed;
            internal readonly Buffer Bytes;
            internal volatile bool Started;
            internal bool Eof;
            internal bool Complete { get { return !Started || worker.Join(0); } }
            internal bool Ready { get { return ready.IsSet; } }
            internal bool HandlesClosed { get { SafeFileHandle original = Volatile.Read(ref thread); return readyClosed && (original == null || original.IsClosed || original.IsInvalid); } }
            internal Reader(Custody value, SafeFileHandle handle, string stream, int limit, int timeout) {
                custody = value; pipe = handle; name = stream; deadline = timeout; Bytes = new Buffer(limit);
                worker = new Thread(Read) { IsBackground = false, Name = "PC-AI capture " + stream };
            }
            internal void Start() { worker.Start(); Started = true; }
            void Read() {
                try {
                    Admit(custody, deadline);
                    Volatile.Write(ref thread, Native.Handle(Native.OpenThread(1, false, Native.GetCurrentThreadId()), "Retain original capture reader thread")); ready.Set();
                    byte[] scratch = new byte[8192];
                    while (Volatile.Read(ref cancelled) == 0) {
                        uint count;
                        Admit(custody, deadline);
                        if (!Native.ReadFile(pipe, scratch, (uint)scratch.Length, out count, IntPtr.Zero)) {
                            int error = Marshal.GetLastWin32Error();
                            if (error == 109) { Eof = true; return; }
                            if (error == 995 && Volatile.Read(ref cancelled) != 0) return;
                            throw new Win32Exception(error, "Read original " + name + " pipe");
                        }
                        if (count == 0) continue;
                        try { Bytes.Append(scratch, checked((int)count)); }
                        catch (IOException failure) { throw new IOException(name + " capture exceeded its finite byte limit.", failure); }
                    }
                } catch (Exception failure) { custody.Fail(failure); }
                finally { ready.Set(); }
            }
            internal void Cancel() {
                Volatile.Write(ref cancelled, 1);
                SafeFileHandle original = Volatile.Read(ref thread);
                if (!Complete && original != null && !Native.CancelSynchronousIo(original)) {
                    int error = Marshal.GetLastWin32Error();
                    if (error != 1168) custody.Fail(new Win32Exception(error, "Cancel original " + name + " reader"));
                    // NOT_FOUND is the inter-call race, not proof of settlement.
                }
            }
            internal void Close() {
                if (!Complete) throw new InvalidOperationException("Original reader remains live.");
                SafeFileHandle original = Volatile.Read(ref thread); if (original != null) Native.Close(original);
                if (!readyClosed) { ready.Dispose(); readyClosed = true; }
            }
        }
        static void Admit(Custody custody, int milliseconds) {
            if (custody.Operation.Cancellation.IsCancellationRequested) throw new OperationCanceledException("Capture was cancelled by its original caller.");
            if (custody.Clock.ElapsedMilliseconds >= milliseconds) throw new OperationDeadlineException();
            if (custody.Failed) throw new IOException("An original capture reader failed.");
        }
        static void Close(Custody custody, SafeHandle handle) { if (handle != null) { try { Native.Close(handle); } catch (Exception failure) { custody.Fail(failure); } } }
        static void Observe(Custody custody) {
            if (custody.Process == null) { custody.ParentExited = !custody.Started; return; }
            if (custody.Process.IsClosed) { if (!custody.ParentExited) throw new InvalidOperationException("Original process closed before exit proof."); return; }
            uint state = Native.WaitForSingleObject(custody.Process, 0);
            if (state == 0) {
                custody.ParentExited = true;
                if (!custody.ExitCode.HasValue) { uint code; Native.Check(Native.GetExitCodeProcess(custody.Process, out code), "Read original exit code"); custody.ExitCode = unchecked((int)code); }
            } else if (state != 258) throw new Win32Exception(Marshal.GetLastWin32Error(), "Observe original process handle");
        }
        static void ObserveJob(Custody custody) {
            if (custody.Job == null) { custody.JobEmpty = !custody.Assigned; return; }
            if (custody.Job.IsClosed) { if (!custody.JobEmpty) throw new InvalidOperationException("Original job closed before empty proof."); return; }
            Native.Accounting value; uint returned;
            Native.Check(Native.QueryInformationJobObject(custody.Job, 1, out value, (uint)Marshal.SizeOf<Native.Accounting>(), out returned), "Query original job membership");
            if (returned != Marshal.SizeOf<Native.Accounting>()) throw new InvalidOperationException("Unexpected original job accounting size.");
            custody.JobEmpty = value.ActiveProcesses == 0;
        }
        static void Start(Custody custody, string path, string[] arguments, int timeout) {
            IntPtr attributes = IntPtr.Zero, handleList = IntPtr.Zero; bool initialized = false;
            try {
                Admit(custody, timeout); custody.Job = Native.Handle(Native.CreateJobObjectW(IntPtr.Zero, null), "Create private capture job");
                var limits = new Native.ExtendedLimits(); limits.Basic.Flags = 0x2000;
                Admit(custody, timeout); Native.Check(Native.SetInformationJobObject(custody.Job, 9, ref limits, (uint)Marshal.SizeOf<Native.ExtendedLimits>()), "Set original job kill-on-close");
                Native.Pipe(custody, timeout, out custody.OutRead, out custody.OutWrite);
                Native.Pipe(custody, timeout, out custody.ErrRead, out custody.ErrWrite);
                var security = new Native.SecurityAttributes { Size = Marshal.SizeOf<Native.SecurityAttributes>(), Inherit = 1 };
                Admit(custody, timeout); custody.Input = Native.Handle(Native.CreateFileW("NUL", 0x80000000, 3, ref security, 3, 0, IntPtr.Zero), "Open EOF capture stdin");
                UIntPtr bytes = UIntPtr.Zero;
                Admit(custody, timeout); bool sized = Native.InitializeProcThreadAttributeList(IntPtr.Zero, 1, 0, ref bytes);
                if (sized || Marshal.GetLastWin32Error() != 122 || bytes == UIntPtr.Zero) throw new InvalidOperationException("Unexpected inheritance-list sizing result.");
                attributes = Marshal.AllocHGlobal(checked((int)bytes.ToUInt64()));
                Admit(custody, timeout); Native.Check(Native.InitializeProcThreadAttributeList(attributes, 1, 0, ref bytes), "Initialize capture inheritance allowlist"); initialized = true;
                handleList = Marshal.AllocHGlobal(IntPtr.Size * 3);
                Marshal.WriteIntPtr(handleList, 0, custody.Input.DangerousGetHandle()); Marshal.WriteIntPtr(handleList, IntPtr.Size, custody.OutWrite.DangerousGetHandle()); Marshal.WriteIntPtr(handleList, IntPtr.Size * 2, custody.ErrWrite.DangerousGetHandle());
                Admit(custody, timeout); Native.Check(Native.UpdateProcThreadAttribute(attributes, 0, new UIntPtr(0x20002), handleList, new UIntPtr((uint)(IntPtr.Size * 3)), IntPtr.Zero, IntPtr.Zero), "Bind only capture standard handles");
                var startup = new Native.StartupInfoEx { Startup = new Native.StartupInfo { Size = Marshal.SizeOf<Native.StartupInfoEx>(), Flags = 0x100, Input = custody.Input.DangerousGetHandle(), Output = custody.OutWrite.DangerousGetHandle(), Error = custody.ErrWrite.DangerousGetHandle() }, Attributes = attributes };
                var command = new StringBuilder(Native.Quote(path)); foreach (string argument in arguments) command.Append(' ').Append(Native.Quote(argument));
                Native.ProcessInfo original;
                // Null env/cwd preserve exact inherited environment and native cwd.
                Admit(custody, timeout); Native.Check(Native.CreateProcessW(path, command, IntPtr.Zero, IntPtr.Zero, true, 0x08080404, IntPtr.Zero, null, ref startup, out original), "Create original capture process suspended");
                custody.Started = true; custody.Process = new SafeProcessHandle(original.Process, true); custody.Thread = Native.Handle(original.Thread, "Retain original primary thread"); custody.OriginalPid = original.ProcessId;
                Admit(custody, timeout); Native.Check(Native.AssignProcessToJobObject(custody.Job, custody.Process), "Assign original suspended process to private job"); custody.Assigned = true;
                custody.Output = new Reader(custody, custody.OutRead, "stdout", StdoutLimit, timeout); custody.Error = new Reader(custody, custody.ErrRead, "stderr", StderrLimit, timeout);
                custody.Output.Start(); custody.Error.Start();
                while (!custody.Output.Ready || !custody.Error.Ready) { Admit(custody, timeout); Thread.Sleep(1); }
                Admit(custody, timeout); if (Native.ResumeThread(custody.Thread) == UInt32.MaxValue) throw new Win32Exception(Marshal.GetLastWin32Error(), "Resume original assigned capture process"); custody.Resumed = true;
                Close(custody, custody.Thread); Close(custody, custody.Input); Close(custody, custody.OutWrite); Close(custody, custody.ErrWrite);
            } finally {
                if (initialized) Native.DeleteProcThreadAttributeList(attributes);
                if (attributes != IntPtr.Zero) Marshal.FreeHGlobal(attributes);
                if (handleList != IntPtr.Zero) Marshal.FreeHGlobal(handleList);
            }
        }
        static bool ReadersSettled(Custody custody) { return (custody.Output == null || custody.Output.Complete) && (custody.Error == null || custody.Error.Complete); }
        static void Cleanup(Custody custody, int milliseconds) {
            var clock = Stopwatch.StartNew();
            try { Observe(custody); } catch (Exception failure) { custody.Fail(failure); }
            try { ObserveJob(custody); } catch (Exception failure) { custody.Fail(failure); }
            try {
                if (custody.Assigned && custody.Job != null && !custody.Job.IsClosed && !custody.JobEmpty) Native.Check(Native.TerminateJobObject(custody.Job, 1), "Terminate only original private capture job");
                else if (custody.Started && !custody.Assigned && !custody.ParentExited) Native.Check(Native.TerminateProcess(custody.Process, 1), "Terminate only original unassigned suspended process");
            } catch (Exception failure) { custody.Fail(failure); }
            // Child-side duplicates are never used by readers.
            Close(custody, custody.Input); Close(custody, custody.OutWrite); Close(custody, custody.ErrWrite);
            do {
                try { Observe(custody); } catch (Exception failure) { custody.Fail(failure); }
                try { ObserveJob(custody); } catch (Exception failure) { custody.Fail(failure); }
                if (custody.Failed) { if (custody.Output != null) custody.Output.Cancel(); if (custody.Error != null) custody.Error.Cancel(); }
                if (custody.ParentExited && custody.JobEmpty && ReadersSettled(custody)) break;
                if (clock.ElapsedMilliseconds >= milliseconds) break;
                Thread.Sleep((int)Math.Max(0, Math.Min(5, milliseconds - clock.ElapsedMilliseconds)));
            } while (true);
            if (!custody.ParentExited || !custody.JobEmpty || !ReadersSettled(custody)) { custody.Fail(new TimeoutException("Original parent/job/readers closure remains unconfirmed; custody retained.")); return; }
            if (custody.Output != null) { try { custody.Output.Close(); } catch (Exception failure) { custody.Fail(failure); } }
            if (custody.Error != null) { try { custody.Error.Close(); } catch (Exception failure) { custody.Fail(failure); } }
            Close(custody, custody.OutRead); Close(custody, custody.ErrRead); Close(custody, custody.Thread); Close(custody, custody.Process); Close(custody, custody.Job);
            bool readerHandlesClosed = (custody.Output == null || custody.Output.HandlesClosed) && (custody.Error == null || custody.Error.HandlesClosed);
            custody.Closed = readerHandlesClosed && Closed(custody.Input) && Closed(custody.OutWrite) && Closed(custody.ErrWrite) && Closed(custody.OutRead) && Closed(custody.ErrRead) && Closed(custody.Thread) && Closed(custody.Process) && Closed(custody.Job);
            if (custody.Closed) { custody.Operation.Closed = true; custody.Operation.Cancellation.Dispose(); }
        }
        static bool Closed(SafeHandle handle) { return handle == null || handle.IsClosed || handle.IsInvalid; }
        static CaptureResult Result(Custody custody, bool includeText = true) {
            var result = new CaptureResult { TimedOut = custody.TimedOut, ExitCode = custody.TimedOut ? -2 : -3, Closure = custody.Closed ? "CLOSED" : "HOLD", CustodyId = custody.Id, ElapsedMilliseconds = custody.Clock.ElapsedMilliseconds, AssignedBeforeResume = custody.Assigned && custody.Resumed, ParentExited = custody.ParentExited, JobEmpty = custody.JobEmpty, ReadersSettled = ReadersSettled(custody), HandlesClosed = custody.Closed };
            result.StdoutBytes = custody.Output == null ? 0 : custody.Output.Bytes.Length; result.StderrBytes = custody.Error == null ? 0 : custody.Error.Bytes.Length;
            result.OriginalPid = custody.OriginalPid;
            result.OriginalExitCode = custody.ExitCode;
            bool completeCapture = includeText && custody.Closed && !custody.Failed && custody.ExitCode.HasValue && custody.Output != null && custody.Output.Eof && custody.Error != null && custody.Error.Eof;
            if (completeCapture) {
                try { result.StdOut = custody.Output.Bytes.Decode(); result.StdErr = custody.Error.Bytes.Decode(); result.ExitCode = custody.ExitCode.Value; result.Success = result.ExitCode == 0; }
                catch (Exception failure) { custody.Fail(failure); result.StdOut = ""; result.StdErr = ""; result.ExitCode = -3; }
            }
            lock (custody.FailureGate) {
                result.PrimaryFailure = custody.Primary == null ? null : custody.Primary.ToString();
                result.SecondaryFailures = custody.Secondary.ConvertAll(error => error.ToString()).ToArray();
                result.RepeatedFailures = custody.RepeatedFailures;
            }
            result.FirstCleanup = custody.Operation.FirstCleanup;
            result.CleanupContinuationUsed = custody.ContinuationRequired;
            if (result.PrimaryFailure != null) result.StdErr = result.PrimaryFailure + (result.SecondaryFailures.Length == 0 ? "" : Environment.NewLine + String.Join(Environment.NewLine, result.SecondaryFailures));
            return result;
        }
        static void CleanupOwned(Custody custody) {
            lock (custody.CleanupGate) {
                try { Cleanup(custody, CleanupMilliseconds); }
                catch (Exception failure) { custody.Fail(failure); }
            }
        }
        static void PreserveFirstCleanup(Custody custody, long elapsed) {
            if (custody.Operation.FirstCleanup != null) return;
            // A synchronous final close/dispose can return after the grace has
            // expired. Eventual CLOSED does not erase that first-attempt failure.
            if (elapsed >= CleanupMilliseconds) custody.Fail(new TimeoutException("First original cleanup exceeded its " + CleanupMilliseconds + "ms grace (observed " + elapsed + "ms); original failure retained."));
            custody.Operation.FirstCleanup = new CaptureCleanupReceipt(Result(custody, false), elapsed, SourceBinding);
        }
        static void FirstCleanupOwned(Custody custody) {
            lock (custody.CleanupGate) {
                var cleanupClock = Stopwatch.StartNew();
                try { Cleanup(custody, CleanupMilliseconds); }
                catch (Exception failure) { custody.Fail(failure); }
                // Includes all original reader/handle closes and cancellation
                // source disposal. The clock is cooperative, not an OS kill timer.
                long elapsed = cleanupClock.ElapsedMilliseconds;
                if (!custody.Closed) {
                    lock (Gate) { if (custody.Id == null) custody.Id = Guid.NewGuid().ToString("N"); }
                }
                // No manual retry can acquire CleanupGate or observe Pending
                // until this immutable first-attempt snapshot has been frozen.
                PreserveFirstCleanup(custody, elapsed);
                if (!custody.Closed) PublishHold(custody);
            }
        }
        static void PublishHold(Custody custody) {
            lock (Gate) {
                if (custody.Id == null) custody.Id = Guid.NewGuid().ToString("N");
                if (!Pending.ContainsKey(custody.Id)) Pending.Add(custody.Id, custody);
            }
        }
        static void ContinueOriginalClosure(Custody custody) {
            if (custody.Closed) return;
            custody.ContinuationRequired = true;
            PublishHold(custody);
            // Exceptional HOLD is explicit owner retention, not a finite success.
            // FirstCleanup is never rewritten and the original Primary stays a failure.
            while (!custody.Closed) {
                Thread.Sleep(1000);
                CleanupOwned(custody);
            }
        }
        static CaptureResult Run(Custody custody, string path, string[] arguments, int timeout) {
            try {
                Start(custody, path, arguments, timeout);
                while (true) {
                    Admit(custody, timeout); Observe(custody); ObserveJob(custody);
                    if (custody.ParentExited && !custody.JobEmpty) { Admit(custody, timeout); Native.Check(Native.TerminateJobObject(custody.Job, 1), "Retire only original detached capture descendants"); }
                    if (custody.ParentExited && custody.JobEmpty && ReadersSettled(custody)) break;
                    Thread.Sleep((int)Math.Min(5, Math.Max(1, timeout - custody.Clock.ElapsedMilliseconds)));
                }
            } catch (Exception failure) { custody.Fail(failure); }
            finally {
                FirstCleanupOwned(custody);
                ContinueOriginalClosure(custody);
                lock (Gate) { if (custody.Id != null) Pending.Remove(custody.Id); Active = null; Running = false; }
            }
            CaptureResult completed = Result(custody);
            // Include final observation/cleanup and bounded decode in admission.
            // This observes late completion; it cannot preempt synchronous OS calls.
            if (!custody.Failed && custody.Clock.ElapsedMilliseconds >= timeout) {
                custody.Fail(new OperationDeadlineException());
                return Result(custody, false);
            }
            return completed;
        }
        public static CaptureOperation Begin(string path, string[] arguments, int timeoutMilliseconds, CancellationToken cancellation) {
            if (Environment.OSVersion.Platform != PlatformID.Win32NT) throw new PlatformNotSupportedException("External cache capture requires the Windows owned-job backend; local cache remains available.");
            if (timeoutMilliseconds <= 0) throw new ArgumentOutOfRangeException("timeoutMilliseconds", "A positive bounded deadline is required.");
            if (!Path.IsPathFullyQualified(path) || !File.Exists(path)) throw new ArgumentException("Capture executable must be an existing absolute file.");
            lock (Gate) {
                if (SourceBinding == null) throw new InvalidOperationException("Capture source has not been bound.");
                if (Running || Pending.Count != 0) throw new InvalidOperationException("Active or unconfirmed original capture custody refuses another launch.");
                Running = true;
            }
            var custody = new Custody();
            try {
                var operation = new CaptureOperation { Cancellation = CancellationTokenSource.CreateLinkedTokenSource(cancellation) }; custody.Operation = operation;
                string[] exactArguments = arguments == null ? new string[0] : (string[])arguments.Clone();
                lock (Gate) Active = custody;
                var completion = new TaskCompletionSource<CaptureResult>(TaskCreationOptions.RunContinuationsAsynchronously);
                operation.Completion = completion.Task;
                operation.Controller = new Thread(() => {
                    try { completion.TrySetResult(Run(custody, path, exactArguments, timeoutMilliseconds)); }
                    catch (Exception failure) {
                        // Ordinary unexpected bookkeeping failure cannot abandon a live owner.
                        custody.Fail(failure);
                        FirstCleanupOwned(custody);
                        ContinueOriginalClosure(custody);
                        lock (Gate) { if (custody.Id != null) Pending.Remove(custody.Id); Active = null; Running = false; }
                        completion.TrySetResult(Result(custody));
                    }
                }) { IsBackground = false, Name = "PC-AI original capture controller" };
                operation.Controller.Start();
                return operation;
            } catch { if (custody.Operation != null) custody.Operation.Cancellation.Dispose(); lock (Gate) { Active = null; Running = false; } throw; }
        }
        public static CaptureResult ActiveCustody() { lock (Gate) { if (Active == null) return null; var result = Result(Active, false); result.Closure = Active.Operation.FirstCleanup != null && Active.Operation.FirstCleanup.Closure == "HOLD" ? "OWNED_HOLD_CONTINUATION" : "RUNNING"; return result; } }
        public static CaptureResult[] PendingCustody() { lock (Gate) { var rows = new List<CaptureResult>(); foreach (Custody custody in Pending.Values) rows.Add(Result(custody, false)); return rows.ToArray(); } }
        public static CaptureResult RetryOwnedCustody(string id) {
            Custody custody;
            lock (Gate) { if (!Pending.TryGetValue(id, out custody)) throw new ArgumentException("Unknown original custody identity."); }
            if (!Monitor.TryEnter(custody.CleanupGate)) throw new InvalidOperationException("Original cleanup is currently in progress; custody remains retained.");
            try {
                try { Cleanup(custody, CleanupMilliseconds); } catch (Exception failure) { custody.Fail(failure); }
                // Controller alone releases global admission after it observes closure.
                return Result(custody, false);
            } finally { Monitor.Exit(custody.CleanupGate); }
        }

        static class Native {
            [StructLayout(LayoutKind.Sequential)] internal struct SecurityAttributes { public int Size; public IntPtr Descriptor; public int Inherit; }
            [StructLayout(LayoutKind.Sequential)] internal struct StartupInfo { public int Size; public IntPtr Reserved, Desktop, Title; public int X, Y, XSize, YSize, XChars, YChars, Fill, Flags; public short ShowWindow, ReservedSize; public IntPtr ReservedBytes, Input, Output, Error; }
            [StructLayout(LayoutKind.Sequential)] internal struct StartupInfoEx { public StartupInfo Startup; public IntPtr Attributes; }
            [StructLayout(LayoutKind.Sequential)] internal struct ProcessInfo { public IntPtr Process, Thread; public uint ProcessId, ThreadId; }
            [StructLayout(LayoutKind.Sequential)] internal struct BasicLimits { public long ProcessTime, JobTime; public uint Flags; public UIntPtr MinimumWorkingSet, MaximumWorkingSet; public uint ActiveProcessLimit; public UIntPtr Affinity; public uint Priority, SchedulingClass; }
            [StructLayout(LayoutKind.Sequential)] internal struct IoCounters { public ulong ReadOperations, WriteOperations, OtherOperations, ReadBytes, WriteBytes, OtherBytes; }
            [StructLayout(LayoutKind.Sequential)] internal struct ExtendedLimits { public BasicLimits Basic; public IoCounters Io; public UIntPtr ProcessMemory, JobMemory, PeakProcessMemory, PeakJobMemory; }
            [StructLayout(LayoutKind.Sequential)] internal struct Accounting { public long UserTime, KernelTime, PeriodUserTime, PeriodKernelTime; public uint PageFaults, TotalProcesses, ActiveProcesses, TerminatedProcesses; }
            [DllImport("kernel32.dll", SetLastError=true, CharSet=CharSet.Unicode)] internal static extern IntPtr CreateJobObjectW(IntPtr attributes, string name);
            [DllImport("kernel32.dll", SetLastError=true)] internal static extern bool SetInformationJobObject(SafeFileHandle job, int kind, ref ExtendedLimits limits, uint size);
            [DllImport("kernel32.dll", SetLastError=true)] internal static extern bool QueryInformationJobObject(SafeFileHandle job, int kind, out Accounting value, uint size, out uint returned);
            [DllImport("kernel32.dll", SetLastError=true)] internal static extern bool AssignProcessToJobObject(SafeFileHandle job, SafeProcessHandle process);
            [DllImport("kernel32.dll", SetLastError=true)] internal static extern bool TerminateJobObject(SafeFileHandle job, uint code);
            [DllImport("kernel32.dll", SetLastError=true)] internal static extern bool CreatePipe(out IntPtr read, out IntPtr write, ref SecurityAttributes attributes, uint size);
            [DllImport("kernel32.dll", SetLastError=true)] internal static extern bool SetHandleInformation(SafeFileHandle handle, uint mask, uint flags);
            [DllImport("kernel32.dll", SetLastError=true, CharSet=CharSet.Unicode)] internal static extern IntPtr CreateFileW(string name, uint access, uint share, ref SecurityAttributes attributes, uint creation, uint flags, IntPtr template);
            [DllImport("kernel32.dll", SetLastError=true, CharSet=CharSet.Unicode)] internal static extern bool CreateProcessW(string application, StringBuilder command, IntPtr processAttributes, IntPtr threadAttributes, bool inherit, uint flags, IntPtr environment, string directory, ref StartupInfoEx startup, out ProcessInfo process);
            [DllImport("kernel32.dll", SetLastError=true)] internal static extern uint ResumeThread(SafeFileHandle thread);
            [DllImport("kernel32.dll", SetLastError=true)] internal static extern uint WaitForSingleObject(SafeProcessHandle process, uint milliseconds);
            [DllImport("kernel32.dll", SetLastError=true)] internal static extern bool GetExitCodeProcess(SafeProcessHandle process, out uint code);
            [DllImport("kernel32.dll", SetLastError=true)] internal static extern bool TerminateProcess(SafeProcessHandle process, uint code);
            [DllImport("kernel32.dll", SetLastError=true)] internal static extern bool InitializeProcThreadAttributeList(IntPtr list, int count, uint flags, ref UIntPtr size);
            [DllImport("kernel32.dll", SetLastError=true)] internal static extern bool UpdateProcThreadAttribute(IntPtr list, uint flags, UIntPtr attribute, IntPtr value, UIntPtr size, IntPtr previous, IntPtr returned);
            [DllImport("kernel32.dll")] internal static extern void DeleteProcThreadAttributeList(IntPtr list);
            [DllImport("kernel32.dll")] internal static extern uint GetCurrentThreadId();
            [DllImport("kernel32.dll", SetLastError=true)] internal static extern IntPtr OpenThread(uint access, bool inherit, uint id);
            [DllImport("kernel32.dll", SetLastError=true)] internal static extern bool CancelSynchronousIo(SafeFileHandle thread);
            [DllImport("kernel32.dll", SetLastError=true)] internal static extern bool ReadFile(SafeFileHandle file, [Out] byte[] bytes, uint count, out uint read, IntPtr overlapped);
            [DllImport("kernel32.dll", SetLastError=true)] internal static extern bool CloseHandle(IntPtr handle);
            internal static void Check(bool success, string operation) { if (!success) throw new Win32Exception(Marshal.GetLastWin32Error(), operation); }
            internal static SafeFileHandle Handle(IntPtr value, string operation) { if (value == IntPtr.Zero || value == new IntPtr(-1)) throw new Win32Exception(Marshal.GetLastWin32Error(), operation); return new SafeFileHandle(value, true); }
            internal static void Close(SafeHandle handle) { if (handle.IsClosed || handle.IsInvalid) return; Check(CloseHandle(handle.DangerousGetHandle()), "Close original owned handle"); handle.SetHandleAsInvalid(); handle.Dispose(); }
            internal static void Pipe(Custody custody, int timeout, out SafeFileHandle read, out SafeFileHandle write) {
                read = null; write = null; IntPtr r, w; var security = new SecurityAttributes { Size = Marshal.SizeOf<SecurityAttributes>(), Inherit = 1 };
                Admit(custody, timeout); Check(CreatePipe(out r, out w, ref security, 0), "Create capture pipe"); read = Handle(r, "Retain capture read handle"); write = Handle(w, "Retain capture write handle");
                Admit(custody, timeout); Check(SetHandleInformation(read, 1, 0), "Exclude original read handle from inheritance");
            }
            internal static string Quote(string value) {
                if (value == null || value.IndexOf('\0') >= 0) throw new ArgumentException("Invalid NUL/null child argument.");
                var text = new StringBuilder("\""); int slashes = 0;
                foreach (char c in value) { if (c == '\\') { slashes++; continue; } text.Append('\\', c == '"' ? slashes * 2 + 1 : slashes); text.Append(c); slashes = 0; }
                return text.Append('\\', slashes * 2).Append('"').ToString();
            }
        }
    }
}
