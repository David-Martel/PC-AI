using System.ComponentModel;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using Microsoft.Win32.SafeHandles;
using System.Runtime.CompilerServices;

[assembly: InternalsVisibleTo("PcaiServiceHost.Tests")]

namespace PcaiServiceHost;

// Internal failure boundaries permit inert-process qualification without exposing
// process/file overrides through CLI options or mutating installed services.
internal sealed class HvsockOperations
{
    public Func<ProcessStartInfo, Process?> Launch = Process.Start;
    public Func<int, Process> Find = Process.GetProcessById;
    public Func<Process, IntPtr> Handle = p => p.Handle;
    public Func<Process, string> Executable = p => p.MainModule?.FileName ?? throw new IOException("Process image unavailable.");
    public Func<Process, long> CreationTicks = p => p.StartTime.ToUniversalTime().Ticks;
    public Action<Process> Terminate = p => p.Kill();
    public Func<Process, int, bool> Wait = (p, milliseconds) => p.WaitForExit(milliseconds);
    public Action? BeforePublication;
    public Action? AfterDisplacement;
    public CancellationToken Cancellation;
}

internal sealed class HvsockStateSnapshot : IDisposable
{
    public required FileStream Stream { get; init; }
    public required string Identity { get; init; }
    public required byte[] Bytes { get; init; }
    public required string Hash { get; init; }
    public void Dispose() => Stream.Dispose();
}

internal static class HvsockCustody
{
    internal const int DefaultWaitMilliseconds = 5000;
    private static readonly JsonSerializerOptions Json = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        PropertyNameCaseInsensitive = true,
        WriteIndented = true
    };

    internal static string StatePath(string path)
    {
        if (!OperatingSystem.IsWindows()) throw new PlatformNotSupportedException("HVSOCK custody requires Windows file/process identity.");
        if (string.IsNullOrWhiteSpace(path)) throw new ArgumentException("State path is required.");
        foreach (var component in path.Replace('/', '\\').Split('\\', StringSplitOptions.RemoveEmptyEntries))
        {
            if (!Regex.IsMatch(component, @"^[A-Za-z]:$", RegexOptions.CultureInvariant) && component.IndexOfAny(Path.GetInvalidFileNameChars()) >= 0)
                throw new IOException("Invalid or alternate-stream state path refused.");
            var name = component.TrimEnd(' ', '.');
            if (Regex.IsMatch(name, @"^(\$null|CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])($|\.)", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant))
                throw new IOException("Reserved state path refused.");
        }
        var full = Path.GetFullPath(path);
        for (var current = full; current != null; current = Path.GetDirectoryName(current))
        {
            if ((File.Exists(current) || Directory.Exists(current)) && (File.GetAttributes(current) & FileAttributes.ReparsePoint) != 0)
                throw new IOException("Linked state paths are unsupported.");
        }
        return full;
    }

    internal static HvsockStateSnapshot Snapshot(string path)
    {
        path = StatePath(path);
        var handle = Native.CreateFileW(path, 0x80000000, 7, IntPtr.Zero, 3, 0x00200000, IntPtr.Zero);
        if (handle.IsInvalid) { handle.Dispose(); throw new Win32Exception(Marshal.GetLastWin32Error()); }
        FileStream? stream = null;
        try
        {
            if (!Native.GetFileInformationByHandle(handle, out var basic)) throw new Win32Exception(Marshal.GetLastWin32Error());
            if (basic.Links != 1 || (basic.Attributes & ((uint)FileAttributes.ReparsePoint | (uint)FileAttributes.Directory)) != 0)
                throw new IOException("Linked or non-ordinary state object refused.");
            if (!Native.GetFileInformationByHandleEx(handle, 18, out var id, 24)) throw new Win32Exception(Marshal.GetLastWin32Error());
            stream = new FileStream(handle, FileAccess.Read);
            using var buffer = new MemoryStream();
            stream.CopyTo(buffer);
            var bytes = buffer.ToArray();
            return new HvsockStateSnapshot { Stream = stream, Identity = $"{id.Volume:X16}:{id.Id:N}", Bytes = bytes, Hash = Convert.ToHexString(SHA256.HashData(bytes)) };
        }
        catch { if (stream != null) stream.Dispose(); else handle.Dispose(); throw; }
    }

    private static bool Same(string path, HvsockStateSnapshot expected)
    {
        if (!File.Exists(path)) return false;
        using var actual = Snapshot(path);
        return actual.Identity == expected.Identity && actual.Hash == expected.Hash;
    }

    internal static string? Publish(string path, HvsockStateSnapshot? expected, List<HvsockProxyStateEntry> entries, bool retire, HvsockOperations? operations = null)
    {
        var ops = operations ?? new HvsockOperations();
        path = StatePath(path);
        if (expected == null ? File.Exists(path) || Directory.Exists(path) : !Same(path, expected))
            throw new IOException("State changed concurrently or already exists.");
        ops.Cancellation.ThrowIfCancellationRequested();
        var directory = Path.GetDirectoryName(path) ?? throw new IOException("State parent unavailable.");
        Directory.CreateDirectory(directory);
        StatePath(path);
        var stage = path + ".pcai-stage-" + Guid.NewGuid().ToString("N");
        if (!retire)
        {
            var bytes = JsonSerializer.SerializeToUtf8Bytes(entries, Json);
            using var writer = new FileStream(stage, FileMode.CreateNew, FileAccess.Write, FileShare.None);
            writer.Write(bytes); writer.Flush(true);
        }
        using var staged = retire ? null : Snapshot(stage);
        string? previous = null;
        try
        {
            ops.BeforePublication?.Invoke();
            ops.Cancellation.ThrowIfCancellationRequested();
            if (expected == null ? File.Exists(path) || Directory.Exists(path) : !Same(path, expected))
                throw new IOException("State changed concurrently or already exists.");
            if (staged != null && !Same(stage, staged)) throw new IOException("Staged state changed concurrently; publication refused.");
            if (expected != null)
            {
                previous = path + ".pcai-previous-" + Guid.NewGuid().ToString("N");
                File.Move(path, previous, false);
                ops.AfterDisplacement?.Invoke();
                if (!Same(previous, expected)) throw new IOException("Displaced state changed concurrently; custody retained.");
            }
            ops.Cancellation.ThrowIfCancellationRequested();
            if (!retire)
            {
                File.Move(stage, path, false);
                if (staged == null || !Same(path, staged)) throw new IOException("Published state identity changed; custody requires review.");
            }
            else if (File.Exists(path) || Directory.Exists(path)) throw new IOException("Later writer state retained; retirement refused.");
            return previous;
        }
        catch (Exception error)
        {
            error.Data["ProxyStateStagePath"] = stage;
            if (previous != null)
            {
                error.Data["ProxyPreviousStatePath"] = previous;
                // Retain the actual displaced object. A recovery copy can only
                // fill an absent public name; it never overwrites a later writer.
                if (!File.Exists(path) && !Directory.Exists(path) && File.Exists(previous))
                {
                    var recovery = path + ".pcai-restoration-" + Guid.NewGuid().ToString("N");
                    try { File.Copy(previous, recovery, false); File.Move(recovery, path, false); }
                    catch (Exception restoration) { error.Data["ProxyStateRestorationError"] = restoration; }
                }
            }
            throw;
        }
    }

    internal static void NewState(string path, List<HvsockProxyStateEntry> entries) => Publish(path, null, entries, false);

    private static List<HvsockProxyStateEntry> Read(HvsockStateSnapshot snapshot)
    {
        var entries = JsonSerializer.Deserialize<List<HvsockProxyStateEntry>>(Encoding.UTF8.GetString(snapshot.Bytes).TrimStart('\uFEFF'), Json) ?? throw new IOException("State is null or malformed.");
        if (entries.Any(entry => entry is null)) throw new IOException("State contains a null custody entry.");
        return entries;
    }

    private static string CanonicalExecutable(string path)
    {
        using var stream = File.OpenRead(Path.GetFullPath(path));
        var length = Native.GetFinalPathNameByHandleW(stream.SafeFileHandle, null, 0, 0);
        if (length == 0 || length >= 32768) throw new Win32Exception(Marshal.GetLastWin32Error());
        var buffer = new StringBuilder(checked((int)length + 1));
        var written = Native.GetFinalPathNameByHandleW(stream.SafeFileHandle, buffer, (uint)buffer.Capacity, 0);
        if (written == 0 || written >= buffer.Capacity) throw new Win32Exception(Marshal.GetLastWin32Error());
        return buffer.ToString();
    }

    private static Process Owned(HvsockProxyStateEntry entry, HvsockOperations ops)
    {
        if (entry.Pid <= 0 || entry.MetadataIncomplete == true || entry.ProcessStartTimeUtcTicks is not > 0 || string.IsNullOrWhiteSpace(entry.ExecutablePath))
            throw new IOException("Unverified legacy or incomplete process custody.");
        var p = ops.Find(entry.Pid);
        try
        {
            var handle = ops.Handle(p);
            if (handle == IntPtr.Zero || handle == new IntPtr(-1)) throw new IOException("Process handle unavailable.");
            var image = CanonicalExecutable(ops.Executable(p));
            var expected = CanonicalExecutable(entry.ExecutablePath);
            var ticks = ops.CreationTicks(p);
            if (p.HasExited || !string.Equals(image, expected, StringComparison.OrdinalIgnoreCase) || ticks != entry.ProcessStartTimeUtcTicks)
                throw new IOException("Process custody identity differs.");
            return p;
        }
        catch { p.Dispose(); throw; }
    }

    internal static int Status(string statePath)
    {
        statePath = StatePath(statePath);
        if (!File.Exists(statePath)) { Console.WriteLine("No HVSOCK custody state."); return 1; }
        using var snapshot = Snapshot(statePath);
        var entries = Read(snapshot);
        var running = 0;
        foreach (var entry in entries)
        {
            try
            {
                using var process = Owned(entry, new HvsockOperations());
                running++;
                Console.WriteLine($"{entry.Name}: RUNNING pid={entry.Pid} target={entry.TcpTarget}");
            }
            catch (Exception error) { Console.WriteLine($"{entry.Name}: UNVERIFIED pid={entry.Pid}: {error.Message}"); }
        }
        Console.WriteLine($"Active verified: {running}/{entries.Count}");
        return entries.Count > 0 && running == entries.Count ? 0 : 1;
    }

    internal static int Stop(string statePath, int waitMilliseconds = DefaultWaitMilliseconds, HvsockOperations? operations = null)
    {
        if (waitMilliseconds < 0) throw new ArgumentOutOfRangeException(nameof(waitMilliseconds));
        var ops = operations ?? new HvsockOperations();
        statePath = StatePath(statePath);
        if (!File.Exists(statePath)) { Console.WriteLine("No state file found."); return 0; }
        using var snapshot = Snapshot(statePath);
        var entries = Read(snapshot);
        var remaining = new List<HvsockProxyStateEntry>();
        var stopped = 0;
        foreach (var entry in entries)
        {
            try
            {
                ops.Cancellation.ThrowIfCancellationRequested();
                using var p = Owned(entry, ops);
                ops.Terminate(p);
                if (!ops.Wait(p, waitMilliseconds) || !p.HasExited) throw new TimeoutException("Proxy exit remains unresolved.");
                stopped++;
            }
            catch (Exception error) { remaining.Add(entry); Console.Error.WriteLine($"Unresolved proxy {entry.Name}: {error.Message}"); }
        }
        // With no confirmed closure, retain exact original bytes and identity.
        if (stopped > 0)
        {
            var previous = Publish(statePath, snapshot, remaining, remaining.Count == 0, ops);
            Console.WriteLine($"Previous state custody: {previous}");
        }
        Console.WriteLine($"Stopped {stopped}; unresolved {remaining.Count}.");
        return remaining.Count == 0 ? 0 : 1;
    }

    private sealed class Launched
    {
        public required Process Process;
        public required HvsockProxyStateEntry Entry;
        public bool VerifiedHandle;
        public bool Retain;
    }

    internal static int Start(string statePath, string executable, List<HvsockProxyEntry> entries, HvsockOperations? operations = null)
    {
        var ops = operations ?? new HvsockOperations();
        statePath = StatePath(statePath);
        var directory = Path.GetDirectoryName(statePath) ?? throw new IOException("State parent unavailable.");
        if (File.Exists(statePath) || Directory.Exists(statePath)) throw new IOException("Existing proxy custody blocks launch; reconcile it explicitly.");
        if (Directory.Exists(directory) && Directory.EnumerateFiles(directory, Path.GetFileName(statePath) + ".pcai-recovery-*.json").Any())
            throw new IOException("Pending proxy recovery blocks launch.");
        if (entries.Count == 0 || entries.Any(e => string.IsNullOrWhiteSpace(e.Name) || string.IsNullOrWhiteSpace(e.ServiceId) || string.IsNullOrWhiteSpace(e.TcpHost) || string.IsNullOrWhiteSpace(e.TcpPort)))
            throw new IOException("Complete proxy configuration is required.");
        var image = CanonicalExecutable(executable);
        var launched = new List<Launched>();
        try
        {
            foreach (var entry in entries)
            {
                ops.Cancellation.ThrowIfCancellationRequested();
                var info = new ProcessStartInfo { FileName = executable, UseShellExecute = false, CreateNoWindow = true };
                info.ArgumentList.Add($"HVSock-LISTEN:{entry.ServiceId}");
                info.ArgumentList.Add($"TCP:{entry.TcpHost}:{entry.TcpPort}");
                var process = ops.Launch(info) ?? throw new IOException("Proxy launch did not return process custody.");
                var record = new Launched { Process = process, Entry = new HvsockProxyStateEntry { Name = entry.Name, ServiceId = entry.ServiceId, TcpTarget = $"{entry.TcpHost}:{entry.TcpPort}", ExecutablePath = executable, Started = DateTime.UtcNow.ToString("o"), MetadataIncomplete = true } };
                launched.Add(record); // Retain before every fallible getter.
                record.Entry.Pid = process.Id;
                var handle = ops.Handle(process);
                if (handle == IntPtr.Zero || handle == new IntPtr(-1)) throw new IOException("New proxy handle unavailable.");
                record.VerifiedHandle = true;
                record.Entry.ProcessStartTimeUtcTicks = ops.CreationTicks(process);
                if (process.HasExited || !string.Equals(CanonicalExecutable(ops.Executable(process)), image, StringComparison.OrdinalIgnoreCase))
                    throw new IOException("Launched proxy identity differs.");
                record.Entry.MetadataIncomplete = false;
            }
            Publish(statePath, null, launched.Select(r => r.Entry).ToList(), false, ops);
            Console.WriteLine($"Started {launched.Count} HVSOCK proxies.");
            return 0;
        }
        catch (Exception original)
        {
            var pending = new List<HvsockProxyStateEntry>();
            var retained = new List<Process>();
            foreach (var record in launched)
            {
                try
                {
                    if (!record.VerifiedHandle) throw new IOException("Unverified new-child handle; automatic termination refused.");
                    ops.Terminate(record.Process);
                    if (!ops.Wait(record.Process, DefaultWaitMilliseconds) || !record.Process.HasExited)
                        throw new TimeoutException("New-child exit remains unresolved.");
                }
                catch
                {
                    record.Retain = true;
                    pending.Add(record.Entry); retained.Add(record.Process);
                }
            }
            original.Data["ProxyOriginalOperationException"] = original;
            original.Data["ProxyRecoveryEntries"] = pending;
            original.Data["ProxyRecoveryProcessReferences"] = retained;
            if (pending.Count > 0)
            {
                var recovery = statePath + ".pcai-recovery-" + Guid.NewGuid().ToString("N") + ".json";
                original.Data["ProxyRecoveryStatePath"] = recovery;
                try
                {
                    Publish(recovery, null, pending, false);
                    Console.Error.WriteLine($"Unresolved proxy recovery custody: {recovery}. No automatic recovery is authorized.");
                }
                catch (Exception persistence)
                {
                    original.Data["ProxyRecoveryPersistenceException"] = persistence;
                    Console.Error.WriteLine($"Recovery marker persistence failed; candidate path: {recovery}; unresolved PIDs: {string.Join(",", pending.Select(e => e.Pid))}. Retained process references remain attached to the original exception.");
                }
            }
            throw;
        }
        finally { foreach (var record in launched) if (!record.Retain) record.Process.Dispose(); }
    }

    private static class Native
    {
        [StructLayout(LayoutKind.Sequential)] internal struct FileId { public ulong Volume; public Guid Id; }
        [StructLayout(LayoutKind.Sequential)] internal struct BasicFileInfo
        {
            public uint Attributes, CreationLow, CreationHigh, AccessLow, AccessHigh, WriteLow, WriteHigh, Volume, SizeHigh, SizeLow, Links, IndexHigh, IndexLow;
        }
        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        internal static extern SafeFileHandle CreateFileW(string name, uint access, uint share, IntPtr security, uint disposition, uint flags, IntPtr template);
        [DllImport("kernel32.dll", SetLastError = true)] [return: MarshalAs(UnmanagedType.Bool)]
        internal static extern bool GetFileInformationByHandle(SafeFileHandle handle, out BasicFileInfo information);
        [DllImport("kernel32.dll", SetLastError = true)] [return: MarshalAs(UnmanagedType.Bool)]
        internal static extern bool GetFileInformationByHandleEx(SafeFileHandle handle, int kind, out FileId information, uint size);
        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        internal static extern uint GetFinalPathNameByHandleW(SafeFileHandle handle, StringBuilder? path, uint length, uint flags);
    }
}
