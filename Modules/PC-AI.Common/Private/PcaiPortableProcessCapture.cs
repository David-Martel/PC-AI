using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace Pcai.Common {
    public sealed class PortableCaptureResult {
        public bool Success, TimedOut, Cancelled, ParentExited, ReadersSettled, HandlesClosed;
        public int ExitCode;
        public int? OriginalPid, OriginalExitCode;
        public string StdOut = "", StdErr = "", Closure = "NOT_STARTED", CustodyId, PrimaryFailure;
        public string Descendants = "UNVERIFIED";
        public long StdoutBytes, StderrBytes, ElapsedMilliseconds;
    }

    public sealed class PortableCaptureOperation {
        internal CancellationTokenSource Stop;
        internal volatile bool Closed;
        internal volatile bool CancellationRequested;
        public string CustodyId { get; internal set; }
        public Task<PortableCaptureResult> Completion { get; internal set; }
        public void Cancel() {
            if (Closed) return;
            CancellationRequested = true;
            try { Stop.Cancel(); } catch (ObjectDisposedException) { if (!Closed) throw; }
        }
    }

    // Root-process and redirected-stream custody only. Kill(true) is best effort;
    // neither normal EOF nor disposal proves escaped/reparented descendant closure.
    public static class PortableProcessCaptureV1 {
        public const int ContractVersion = 1;
        // Match the separately maintained Windows capture's finite output budgets.
        public const int StdoutLimit = 8 * 1024 * 1024, StderrLimit = 1024 * 1024;
        public const int CleanupMilliseconds = 1000, MaximumOutstanding = 4;
        static readonly object Gate = new object();
        static readonly Dictionary<string, Owner> Outstanding = new Dictionary<string, Owner>();
        public static string SourceBinding { get; private set; }

        public static void BindSource(string hash) {
            lock (Gate) {
                if (hash == null || hash.Length != 64) throw new ArgumentException("Exact source hash required.");
                if (SourceBinding != null && !String.Equals(SourceBinding, hash, StringComparison.Ordinal))
                    throw new InvalidOperationException("Different portable capture source is loaded; use a fresh process.");
                SourceBinding = hash;
            }
        }

        sealed class Output {
            readonly MemoryStream bytes = new MemoryStream();
            readonly int limit;
            internal bool Eof;
            internal Output(int maximum) { limit = maximum; }
            internal void Append(byte[] buffer, int count) {
                lock (bytes) {
                    if (count > limit - bytes.Length) throw new IOException("Capture exceeded its finite output byte limit.");
                    bytes.Write(buffer, 0, count);
                }
            }
            internal long Length { get { lock (bytes) return bytes.Length; } }
            internal string Text() {
                lock (bytes) {
                    using (var copy = new MemoryStream(bytes.ToArray(), false))
                    using (var reader = new StreamReader(copy, Encoding.UTF8, true)) return reader.ReadToEnd().Trim();
                }
            }
        }

        sealed class Owner {
            internal readonly object FailureGate = new object();
            internal readonly Stopwatch Clock = Stopwatch.StartNew();
            internal readonly Output Out = new Output(StdoutLimit), Err = new Output(StderrLimit);
            internal readonly TaskCompletionSource<bool> ExitSignal = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
            internal readonly TaskCompletionSource<bool> FailureSignal = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
            internal Process Process;
            internal PortableCaptureOperation Operation;
            internal CancellationToken Caller;
            internal Task OutTask = Task.CompletedTask, ErrTask = Task.CompletedTask, ExitTask = Task.CompletedTask, CleanupTask;
            internal bool Started, ParentExited, HandlesClosed, TimedOut, Cancelled;
            internal int? Pid, ExitCode;
            internal string Failure;
            internal void Fail(string value) {
                lock (FailureGate) { if (Failure == null) Failure = value; }
                FailureSignal.TrySetResult(true);
            }
        }

        public static PortableCaptureOperation Begin(string file, string[] arguments, int timeoutMs, CancellationToken cancellation) {
            if (timeoutMs <= 0) throw new ArgumentOutOfRangeException(nameof(timeoutMs));
            var operation = new PortableCaptureOperation {
                CustodyId = Guid.NewGuid().ToString("N"),
                Stop = CancellationTokenSource.CreateLinkedTokenSource(cancellation)
            };
            var owner = new Owner { Operation = operation, Caller = cancellation };
            lock (Gate) {
                // Includes HOLDs: raw retained buffers are capped at 36 MiB.
                // Decoded result strings and snapshot copies are separately bounded.
                if (Outstanding.Count >= MaximumOutstanding) {
                    operation.Stop.Dispose(); operation.Closed = true;
                    operation.Completion = Task.FromResult(new PortableCaptureResult { ExitCode = -3, StdErr = "Portable capture custody capacity is occupied.", PrimaryFailure = "ADMISSION" });
                    return operation;
                }
                Outstanding.Add(operation.CustodyId, owner);
            }
            operation.Completion = Task.Run(() => Run(owner, file, arguments ?? new string[0], timeoutMs));
            return operation;
        }

        static async Task Pump(Owner owner, Stream stream, Output output) {
            byte[] scratch = new byte[8192];
            try {
                while (true) {
                    int count = await stream.ReadAsync(scratch, 0, scratch.Length, owner.Operation.Stop.Token).ConfigureAwait(false);
                    if (count == 0) { output.Eof = true; return; }
                    output.Append(scratch, count);
                }
            } catch (OperationCanceledException) {
                // The controller retains the original timeout/cancellation/failure.
            } catch (Exception failure) {
                owner.Fail(failure.GetType().Name + ": " + failure.Message);
            }
        }

        static async Task ObserveExit(Owner owner) {
            await owner.ExitSignal.Task.ConfigureAwait(false);
            try { owner.ExitCode = owner.Process.ExitCode; owner.ParentExited = true; }
            catch (Exception failure) { owner.Fail("Observe original process: " + failure.Message); }
        }

        static async Task<PortableCaptureResult> Run(Owner owner, string file, string[] arguments, int timeoutMs) {
            try {
                owner.Operation.Stop.Token.ThrowIfCancellationRequested();
                var start = new ProcessStartInfo { FileName = file, UseShellExecute = false, RedirectStandardOutput = true, RedirectStandardError = true, CreateNoWindow = true };
                foreach (string argument in arguments) start.ArgumentList.Add(argument ?? "");
                owner.Process = new Process { StartInfo = start, EnableRaisingEvents = true };
                owner.Process.Exited += (sender, args) => owner.ExitSignal.TrySetResult(true);
                long remaining = timeoutMs - owner.Clock.ElapsedMilliseconds;
                if (remaining <= 0) { owner.TimedOut = true; throw new TimeoutException("Capture operation deadline exceeded before launch."); }
                owner.Operation.Stop.CancelAfter((int)remaining);
                if (!owner.Process.Start()) throw new IOException("Child process did not start.");
                owner.Started = true; owner.Pid = owner.Process.Id;
                owner.OutTask = Pump(owner, owner.Process.StandardOutput.BaseStream, owner.Out);
                owner.ErrTask = Pump(owner, owner.Process.StandardError.BaseStream, owner.Err);
                owner.ExitTask = ObserveExit(owner);
                if (owner.Process.HasExited) owner.ExitSignal.TrySetResult(true);
                Task complete = Task.WhenAll(owner.OutTask, owner.ErrTask, owner.ExitTask);
                Task stopped = Task.Delay(Timeout.Infinite, owner.Operation.Stop.Token);
                await Task.WhenAny(complete, stopped, owner.FailureSignal.Task).ConfigureAwait(false);
                if (owner.Failure == null && (!complete.IsCompleted || owner.Operation.Stop.IsCancellationRequested)) {
                    owner.Cancelled = owner.Caller.IsCancellationRequested || owner.Operation.CancellationRequested;
                    owner.TimedOut = !owner.Cancelled;
                    owner.Fail(owner.Cancelled ? "Capture cancelled by its original caller." : "Capture operation deadline exceeded.");
                }
            } catch (OperationCanceledException) {
                owner.Cancelled = true; owner.Fail("Capture cancelled by its original caller.");
            } catch (Exception failure) {
                owner.Fail(failure.GetType().Name + ": " + failure.Message);
            }
            if (owner.Operation.Stop.IsCancellationRequested && owner.Failure == null) {
                owner.Cancelled = owner.Caller.IsCancellationRequested || owner.Operation.CancellationRequested;
                owner.TimedOut = !owner.Cancelled;
                owner.Fail(owner.Cancelled ? "Capture cancelled by its original caller." : "Capture operation deadline exceeded.");
            }
            // Cancellation and stream close may require OS work. The cleanup task
            // retains every original object if that work exceeds this separate budget.
            owner.CleanupTask = Cleanup(owner);
            await Task.WhenAny(owner.CleanupTask, Task.Delay(CleanupMilliseconds)).ConfigureAwait(false);
            return Snapshot(owner);
        }

        static async Task Cleanup(Owner owner) {
            try {
                owner.Operation.Stop.Cancel();
                Task kill = Task.CompletedTask, close = Task.CompletedTask;
                if (owner.Started && !owner.ParentExited) {
                    kill = Task.Run(() => {
                        try { if (!owner.Process.HasExited) owner.Process.Kill(true); }
                        catch (Exception failure) { owner.Fail("Kill original process tree: " + failure.Message); }
                    });
                }
                if (owner.Started && (!owner.OutTask.IsCompleted || !owner.ErrTask.IsCompleted)) {
                    close = Task.Run(() => {
                        try { owner.Process.StandardOutput.BaseStream.Dispose(); }
                        catch (Exception failure) { owner.Fail("Close original stdout: " + failure.Message); }
                        try { owner.Process.StandardError.BaseStream.Dispose(); }
                        catch (Exception failure) { owner.Fail("Close original stderr: " + failure.Message); }
                    });
                }
                await Task.WhenAll(kill, close, owner.OutTask, owner.ErrTask, owner.ExitTask).ConfigureAwait(false);
                if (owner.Started && !owner.ParentExited) return;
                if (owner.Process != null) {
                    // Disposal also stays in retained task custody if an OS disposal stalls.
                    await Task.Run(() => owner.Process.Dispose()).ConfigureAwait(false);
                }
                owner.HandlesClosed = true;
                owner.Operation.Closed = true;
                owner.Operation.Stop.Dispose();
                lock (Gate) Outstanding.Remove(owner.Operation.CustodyId);
            } catch (Exception failure) { owner.Fail("Original cleanup remains held: " + failure.Message); }
        }

        static PortableCaptureResult Snapshot(Owner owner) {
            bool readers = owner.OutTask.IsCompleted && owner.ErrTask.IsCompleted;
            bool closed = owner.HandlesClosed;
            string stderr = owner.Err.Text();
            if (owner.Failure != null) stderr = stderr.Length == 0 ? owner.Failure : stderr + Environment.NewLine + owner.Failure;
            return new PortableCaptureResult {
                Success = owner.Started && owner.Failure == null && owner.ExitCode == 0 && owner.Out.Eof && owner.Err.Eof && closed,
                ExitCode = owner.TimedOut ? -2 : owner.Cancelled ? -4 : owner.Failure != null ? -3 : owner.ExitCode ?? -3,
                TimedOut = owner.TimedOut, Cancelled = owner.Cancelled,
                StdOut = owner.Out.Text(), StdErr = stderr,
                Closure = closed ? (owner.Started ? "ROOT_AND_PIPES_CLOSED" : "NOT_STARTED") : "HOLD",
                CustodyId = owner.Operation.CustodyId, PrimaryFailure = owner.Failure,
                ParentExited = owner.ParentExited, ReadersSettled = readers, HandlesClosed = closed,
                OriginalPid = owner.Pid, OriginalExitCode = owner.ExitCode,
                StdoutBytes = owner.Out.Length, StderrBytes = owner.Err.Length, ElapsedMilliseconds = owner.Clock.ElapsedMilliseconds
            };
        }

        public static PortableCaptureResult[] PendingCustody() {
            Owner[] owners;
            lock (Gate) { owners = new Owner[Outstanding.Count]; Outstanding.Values.CopyTo(owners, 0); }
            var result = new List<PortableCaptureResult>();
            foreach (Owner owner in owners) if (owner.CleanupTask != null) result.Add(Snapshot(owner));
            return result.ToArray();
        }
    }
}
