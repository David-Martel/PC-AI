using System.ComponentModel;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Security.AccessControl;
using System.Security.Principal;
using System.Runtime.Versioning;
using PcaiServiceHost;

namespace PcaiServiceHost.Tests;

public sealed class WindowsFactAttribute : FactAttribute
{
    public WindowsFactAttribute() { if (!OperatingSystem.IsWindows()) Skip = "Windows process/file identity is required; unsupported platform is not qualified."; }
}
public sealed class WindowsTheoryAttribute : TheoryAttribute
{
    public WindowsTheoryAttribute() { if (!OperatingSystem.IsWindows()) Skip = "Windows process/file identity is required; unsupported platform is not qualified."; }
}

public sealed class HvsockCustodyTests : IDisposable
{
    private readonly string root = Path.Combine(Path.GetTempPath(), "pcai-servicehost-custody", Guid.NewGuid().ToString("N"));
    private readonly List<Process> children = new();
    private string State => Path.Combine(root, "state", "literal[private].json");
    private static readonly JsonSerializerOptions Json = new() { PropertyNameCaseInsensitive = true };

    private Process Child()
    {
        Directory.CreateDirectory(root);
        var image = Path.Combine(root, "winsocat-owned-" + Guid.NewGuid().ToString("N") + ".exe");
        File.Copy(Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.System), "cmd.exe"), image);
        var info = new ProcessStartInfo(image) { UseShellExecute = false, CreateNoWindow = true, RedirectStandardInput = true, RedirectStandardOutput = true, RedirectStandardError = true };
        info.ArgumentList.Add("/c"); info.ArgumentList.Add("pause");
        var child = Process.Start(info) ?? throw new IOException("Inert child could not start.");
        children.Add(child);
        _ = child.Handle;
        Thread.Sleep(150);
        Assert.False(child.HasExited);
        return child;
    }

    private static HvsockProxyStateEntry Entry(Process child) => new()
    {
        Name = "synthetic", ServiceId = "synthetic-not-registered", TcpTarget = "unused:0", Pid = child.Id,
        ExecutablePath = child.MainModule!.FileName, ProcessStartTimeUtcTicks = child.StartTime.ToUniversalTime().Ticks, MetadataIncomplete = false
    };
    private static List<HvsockProxyEntry> Config(int count = 1) => Enumerable.Range(0, count).Select(i => new HvsockProxyEntry { Name = "synthetic-" + i, ServiceId = "synthetic-not-registered", TcpHost = "unused", TcpPort = "0" }).ToList();
    private void Write(params HvsockProxyStateEntry[] entries)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(State)!);
        File.WriteAllText(State, JsonSerializer.Serialize(entries));
    }
    private static HvsockOperations Launch(Process child) => new() { Launch = _ => Process.GetProcessById(child.Id) };

    [WindowsFact]
    public void LegacyPidDoesNotAuthorizeStatusOrTermination()
    {
        var child = Child(); Write(new HvsockProxyStateEntry { Name = "legacy", Pid = child.Id });
        var before = File.ReadAllBytes(State);
        Assert.Equal(1, HvsockCustody.Status(State)); Assert.Equal(1, HvsockCustody.Stop(State));
        Assert.False(child.HasExited); Assert.Equal(before, File.ReadAllBytes(State));
    }

    [WindowsTheory]
    [InlineData("Executable")]
    [InlineData("Ticks")]
    [InlineData("Incomplete")]
    public void ForeignOrIncompleteIdentityCannotStopSameNameChild(string mismatch)
    {
        var child = Child(); var entry = Entry(child);
        if (mismatch == "Executable") entry.ExecutablePath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.System), "notepad.exe");
        if (mismatch == "Ticks") entry.ProcessStartTimeUtcTicks--;
        if (mismatch == "Incomplete") entry.MetadataIncomplete = true;
        Write(entry); var before = File.ReadAllBytes(State);
        Assert.Equal(1, HvsockCustody.Status(State)); Assert.Equal(1, HvsockCustody.Stop(State));
        Assert.False(child.HasExited); Assert.Equal(before, File.ReadAllBytes(State));
    }

    [WindowsFact]
    public void NullLedgerEntryRejectsWholeLedgerBeforeProcessEffects()
    {
        var child = Child();
        Directory.CreateDirectory(Path.GetDirectoryName(State)!);
        File.WriteAllText(State, JsonSerializer.Serialize(new HvsockProxyStateEntry?[] { Entry(child), null }));
        using var before = HvsockCustody.Snapshot(State);
        Assert.Equal("State contains a null custody entry.", Assert.Throws<IOException>(() => HvsockCustody.Status(State)).Message);
        Assert.Equal("State contains a null custody entry.", Assert.Throws<IOException>(() => HvsockCustody.Stop(State)).Message);
        Assert.False(child.HasExited);
        using var after = HvsockCustody.Snapshot(State);
        Assert.Equal(before.Identity, after.Identity); Assert.Equal(before.Bytes, after.Bytes);
    }

    [WindowsFact]
    public void VerifiedClosureRetiresStateIntoOriginalObjectCustody()
    {
        var child = Child(); Write(Entry(child));
        using var before = HvsockCustody.Snapshot(State);
        Assert.Equal(0, HvsockCustody.Status(State)); Assert.Equal(0, HvsockCustody.Stop(State));
        Assert.True(child.HasExited); Assert.False(File.Exists(State));
        var previous = Assert.Single(Directory.GetFiles(Path.GetDirectoryName(State)!, "*.pcai-previous-*"));
        using var saved = HvsockCustody.Snapshot(previous);
        Assert.Equal(before.Identity, saved.Identity); Assert.Equal(before.Bytes, saved.Bytes);
    }

    [WindowsFact]
    public void RefusedTerminationPreservesExactOriginalObjectAndBytes()
    {
        var child = Child(); Write(Entry(child)); using var before = HvsockCustody.Snapshot(State);
        var calls = 0;
        var ops = new HvsockOperations { Terminate = p => { calls++; Assert.Equal(child.Id, p.Id); throw new IOException("Synthetic refusal."); } };
        Assert.Equal(1, HvsockCustody.Stop(State, operations: ops));
        Assert.Equal(1, calls); Assert.False(child.HasExited);
        using var after = HvsockCustody.Snapshot(State); Assert.Equal(before.Identity, after.Identity); Assert.Equal(before.Bytes, after.Bytes);
    }

    [WindowsFact]
    public void ActualZeroWaitTimeoutCannotBeCountedAsConfirmedClosure()
    {
        var child = Child(); Write(Entry(child)); var before = File.ReadAllBytes(State);
        var waits = 0;
        var ops = new HvsockOperations { Terminate = _ => { }, Wait = (p, ms) => { waits++; Assert.Equal(0, ms); return p.WaitForExit(ms); } };
        Assert.Equal(1, HvsockCustody.Stop(State, 0, ops)); Assert.Equal(1, waits);
        Assert.False(child.HasExited); Assert.Equal(before, File.ReadAllBytes(State));
    }

    [WindowsFact]
    public void LaterChangedWriterSurvivesConfirmedStopPublicationFailure()
    {
        var child = Child(); Write(Entry(child)); var foreign = "[{\"Name\":\"later-writer\",\"Pid\":0}]";
        var ops = new HvsockOperations { BeforePublication = () => File.WriteAllText(State, foreign) };
        Assert.Throws<IOException>(() => HvsockCustody.Stop(State, operations: ops));
        Assert.True(child.HasExited); Assert.Equal(foreign, File.ReadAllText(State));
    }

    [WindowsFact]
    public void SameBytesDifferentFileObjectCannotBeRetired()
    {
        var child = Child(); Write(Entry(child)); using var before = HvsockCustody.Snapshot(State);
        var ops = new HvsockOperations { BeforePublication = () => { File.Move(State, State + ".original"); File.WriteAllBytes(State, before.Bytes); } };
        Assert.Throws<IOException>(() => HvsockCustody.Stop(State, operations: ops));
        using var after = HvsockCustody.Snapshot(State); Assert.NotEqual(before.Identity, after.Identity); Assert.Equal(before.Bytes, after.Bytes);
        Assert.True(child.HasExited);
    }

    [WindowsFact]
    public void SameBytesLaterWriterAfterDisplacementIsNeverOverwritten()
    {
        var child = Child(); Write(Entry(child)); using var before = HvsockCustody.Snapshot(State);
        var ops = new HvsockOperations { AfterDisplacement = () => File.WriteAllBytes(State, before.Bytes) };
        // Retire has no new bytes, but must still refuse a new public name.
        Assert.Throws<IOException>(() => HvsockCustody.Publish(State, before, new(), true, ops));
        using var after = HvsockCustody.Snapshot(State); Assert.NotEqual(before.Identity, after.Identity); Assert.Equal(before.Bytes, after.Bytes);
        var previous = Assert.Single(Directory.GetFiles(Path.GetDirectoryName(State)!, "*.pcai-previous-*"));
        using var saved = HvsockCustody.Snapshot(previous); Assert.Equal(before.Identity, saved.Identity);
    }

    [WindowsFact]
    public void DisplacementFailureRestoresAbsentNameAndKeepsActualOriginalCustody()
    {
        var child = Child(); Write(Entry(child)); using var before = HvsockCustody.Snapshot(State);
        var original = new IOException("Synthetic after-displacement failure.");
        var ops = new HvsockOperations { AfterDisplacement = () => throw original };
        Assert.Same(original, Assert.Throws<IOException>(() => HvsockCustody.Publish(State, before, new(), false, ops)));
        Assert.Equal(before.Bytes, File.ReadAllBytes(State));
        using var saved = HvsockCustody.Snapshot((string)original.Data["ProxyPreviousStatePath"]!);
        Assert.Equal(before.Identity, saved.Identity); Assert.Equal(before.Bytes, saved.Bytes);
    }

    [WindowsTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void ForeignStageBytesOrSameBytesNewObjectCannotBePublished(bool sameBytes)
    {
        Directory.CreateDirectory(root);
        var ops = new HvsockOperations { BeforePublication = () =>
        {
            var stage = Assert.Single(Directory.GetFiles(Path.GetDirectoryName(State)!, "*.pcai-stage-*"));
            var bytes = File.ReadAllBytes(stage);
            if (sameBytes) { File.Move(stage, stage + ".original"); File.WriteAllBytes(stage, bytes); }
            else File.WriteAllText(stage, "[{\"Name\":\"foreign-stage\",\"Pid\":0}]");
        } };
        Assert.Throws<IOException>(() => HvsockCustody.Publish(State, null, new(), false, ops));
        Assert.False(File.Exists(State));
    }

    [WindowsFact]
    public void PartialStopPreservesUnknownFieldsOnUnresolvedEntries()
    {
        var first = Child(); var second = Child(); var retained = Entry(second);
        retained.AdditionalFields = new() { ["PowerShellExtension"] = JsonDocument.Parse("{\"supported\":true}").RootElement.Clone() };
        Write(Entry(first), retained);
        var ops = new HvsockOperations { Terminate = p => { if (p.Id == second.Id) throw new IOException("Retained."); p.Kill(); } };
        Assert.Equal(1, HvsockCustody.Stop(State, operations: ops)); Assert.True(first.HasExited); Assert.False(second.HasExited);
        var actual = Assert.Single(JsonSerializer.Deserialize<List<HvsockProxyStateEntry>>(File.ReadAllText(State), Json)!);
        Assert.Equal(retained.ExecutablePath, actual.ExecutablePath); Assert.Equal(retained.ProcessStartTimeUtcTicks, actual.ProcessStartTimeUtcTicks);
        Assert.True(actual.AdditionalFields!["PowerShellExtension"].GetProperty("supported").GetBoolean());
    }

    [WindowsTheory]
    [InlineData("[]")]
    [InlineData("null")]
    [InlineData("malformed")]
    public void ExistingStateAlwaysBlocksLaunchWithoutChangingOriginal(string bytes)
    {
        var child = Child(); Directory.CreateDirectory(Path.GetDirectoryName(State)!); File.WriteAllText(State, bytes);
        var calls = 0; var ops = new HvsockOperations { Launch = _ => { calls++; return child; } };
        Assert.Throws<IOException>(() => HvsockCustody.Start(State, child.MainModule!.FileName, Config(), ops));
        Assert.Equal(0, calls); Assert.Equal(bytes, File.ReadAllText(State)); Assert.False(child.HasExited);
    }

    [WindowsFact]
    public void EmptyStateIsNotHealthy()
    {
        Write(); Assert.Equal(1, HvsockCustody.Status(State));
    }

    [WindowsFact]
    public void PendingRecoveryBlocksNewLaunch()
    {
        var child = Child(); Directory.CreateDirectory(Path.GetDirectoryName(State)!);
        File.WriteAllText(State + ".pcai-recovery-owned.json", "[]"); var calls = 0;
        var ops = new HvsockOperations { Launch = _ => { calls++; return child; } };
        Assert.Throws<IOException>(() => HvsockCustody.Start(State, child.MainModule!.FileName, Config(), ops)); Assert.Equal(0, calls);
    }

    [WindowsFact]
    public void SuccessfulInertLaunchPublishesStrongSharedState()
    {
        var child = Child(); Assert.Equal(0, HvsockCustody.Start(State, child.MainModule!.FileName, Config(), Launch(child)));
        var entry = Assert.Single(JsonSerializer.Deserialize<List<HvsockProxyStateEntry>>(File.ReadAllText(State), Json)!);
        Assert.Equal(child.Id, entry.Pid); Assert.Equal(child.StartTime.ToUniversalTime().Ticks, entry.ProcessStartTimeUtcTicks); Assert.False(entry.MetadataIncomplete);
        Assert.Equal(0, HvsockCustody.Status(State)); Assert.Equal(0, HvsockCustody.Stop(State)); Assert.True(child.HasExited);
    }

    [WindowsTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void GetterOrUnverifiedHandleFailureRetainsExactReferencesAndOriginalError(bool invalidHandle)
    {
        var child = Child(); var original = new IOException("Synthetic identity getter failure."); var getter = 0; var kills = 0;
        var ops = Launch(child);
        ops.Launch = _ => child;
        ops.CreationTicks = _ => { getter++; throw original; };
        ops.Terminate = _ => { kills++; throw new IOException("Synthetic cleanup refusal."); };
        if (invalidHandle) ops.Handle = _ => IntPtr.Zero;
        var error = Assert.Throws<IOException>(() => HvsockCustody.Start(State, child.MainModule!.FileName, Config(), ops));
        if (!invalidHandle) Assert.Same(original, error);
        Assert.Same(error, error.Data["ProxyOriginalOperationException"]);
        Assert.Same(child, Assert.Single((List<Process>)error.Data["ProxyRecoveryProcessReferences"]!));
        Assert.Equal(invalidHandle ? 0 : 1, getter); Assert.Equal(invalidHandle ? 0 : 1, kills); Assert.False(child.HasExited);
        var recovery = (string)error.Data["ProxyRecoveryStatePath"]!;
        var entry = Assert.Single(JsonSerializer.Deserialize<List<HvsockProxyStateEntry>>(File.ReadAllText(recovery), Json)!); Assert.True(entry.MetadataIncomplete);
        Assert.Equal(child.Id, entry.Pid);
        Assert.Equal(1, HvsockCustody.Status(recovery)); Assert.Equal(1, HvsockCustody.Stop(recovery)); Assert.False(child.HasExited);
        Assert.Throws<IOException>(() => HvsockCustody.Start(State, child.MainModule!.FileName, Config(), Launch(child)));
    }

    [WindowsFact]
    public void SecondLaunchFailureConfirmsExactFirstChildCleanup()
    {
        var child = Child(); var original = new IOException("Synthetic second launch failure."); var launches = 0;
        var ops = new HvsockOperations { Launch = _ => ++launches == 1 ? Process.GetProcessById(child.Id) : throw original };
        Assert.Same(original, Assert.Throws<IOException>(() => HvsockCustody.Start(State, child.MainModule!.FileName, Config(2), ops)));
        Assert.True(child.HasExited); Assert.Empty((List<Process>)original.Data["ProxyRecoveryProcessReferences"]!); Assert.False(File.Exists(State));
    }

    [WindowsFact]
    public void PublicationFailureKeepsUnresolvedExactChildAndDurableRecovery()
    {
        var child = Child(); var original = new IOException("Synthetic publication failure."); var ops = Launch(child);
        ops.Launch = _ => child;
        ops.BeforePublication = () => throw original; ops.Terminate = _ => throw new IOException("Synthetic refusal.");
        Assert.Same(original, Assert.Throws<IOException>(() => HvsockCustody.Start(State, child.MainModule!.FileName, Config(), ops)));
        Assert.Same(child, Assert.Single((List<Process>)original.Data["ProxyRecoveryProcessReferences"]!)); Assert.False(child.HasExited);
        Assert.True(File.Exists((string)original.Data["ProxyRecoveryStatePath"]!)); Assert.False(File.Exists(State));
    }

    [WindowsFact]
    [SupportedOSPlatform("windows")]
    public void RecoveryPersistenceFailureCannotReplaceOriginalErrorOrLoseChild()
    {
        var child = Child(); var original = new IOException("Synthetic publication failure."); var ops = Launch(child);
        ops.Launch = _ => child;
        Directory.CreateDirectory(Path.GetDirectoryName(State)!);
        var parent = new DirectoryInfo(Path.GetDirectoryName(State)!);
        var saved = FileSystemAclExtensions.GetAccessControl(parent);
        var denied = new DirectorySecurity(); denied.SetSecurityDescriptorBinaryForm(saved.GetSecurityDescriptorBinaryForm());
        denied.AddAccessRule(new FileSystemAccessRule(WindowsIdentity.GetCurrent().User!, FileSystemRights.CreateFiles, InheritanceFlags.None, PropagationFlags.None, AccessControlType.Deny));
        ops.BeforePublication = () => { FileSystemAclExtensions.SetAccessControl(parent, denied); throw original; };
        ops.Terminate = _ => throw new IOException("Synthetic refusal.");
        try
        {
            Assert.Same(original, Assert.Throws<IOException>(() => HvsockCustody.Start(State, child.MainModule!.FileName, Config(), ops)));
            Assert.Same(child, Assert.Single((List<Process>)original.Data["ProxyRecoveryProcessReferences"]!));
            Assert.IsType<UnauthorizedAccessException>(original.Data["ProxyRecoveryPersistenceException"]); Assert.False(child.HasExited);
            Assert.False(File.Exists((string)original.Data["ProxyRecoveryStatePath"]!));
        }
        finally { FileSystemAclExtensions.SetAccessControl(parent, saved); }
    }

    [WindowsFact]
    public void CancellationAfterLaunchConfirmsCleanupBeforePublishing()
    {
        var child = Child(); using var cancellation = new CancellationTokenSource(); var ops = Launch(child);
        ops.Cancellation = cancellation.Token; ops.BeforePublication = cancellation.Cancel;
        Assert.Throws<OperationCanceledException>(() => HvsockCustody.Start(State, child.MainModule!.FileName, Config(), ops));
        Assert.True(child.HasExited); Assert.False(File.Exists(State));
    }

    [WindowsFact]
    public void PowerShellUtf8BomStateRetainsStrongIdentity()
    {
        var child = Child(); Directory.CreateDirectory(Path.GetDirectoryName(State)!);
        File.WriteAllText(State, JsonSerializer.Serialize(new[] { Entry(child) }), new System.Text.UTF8Encoding(true));
        Assert.Equal(0, HvsockCustody.Status(State)); Assert.Equal(0, HvsockCustody.Stop(State)); Assert.True(child.HasExited);
    }

    [WindowsTheory]
    [InlineData("NUL.json")]
    [InlineData("$null")]
    [InlineData("COM1. ")]
    [InlineData("state:stream")]
    public void ReservedOrAlternateStreamWritesFailBeforeFileEffects(string leaf)
    {
        Directory.CreateDirectory(root); Assert.Throws<IOException>(() => HvsockCustody.NewState(Path.Combine(root, leaf), new())); Assert.Empty(Directory.GetFiles(root));
    }

    [WindowsFact]
    public void HardLinkedStateFailsBeforeTermination()
    {
        var child = Child(); Write(Entry(child)); var link = State + ".alias";
        if (!CreateHardLinkW(link, State, IntPtr.Zero)) throw new Win32Exception(Marshal.GetLastWin32Error());
        var calls = 0; var ops = new HvsockOperations { Terminate = _ => calls++ };
        Assert.Throws<IOException>(() => HvsockCustody.Stop(State, operations: ops)); Assert.Equal(0, calls); Assert.False(child.HasExited);
    }

    [WindowsFact]
    public void JunctionStateParentFailsBeforeFileEffects()
    {
        Directory.CreateDirectory(root); var target = Path.Combine(root, "target"); Directory.CreateDirectory(target); var alias = Path.Combine(root, "alias");
        var info = new ProcessStartInfo(Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.System), "cmd.exe")) { UseShellExecute = false, CreateNoWindow = true, RedirectStandardOutput = true, RedirectStandardError = true };
        foreach (var a in new[] { "/c", "mklink", "/J", alias, target }) info.ArgumentList.Add(a);
        using var process = Process.Start(info)!; children.Add(process);
        Assert.True(process.WaitForExit(10000)); Assert.Equal(0, process.ExitCode);
        Assert.Throws<IOException>(() => HvsockCustody.NewState(Path.Combine(alias, "state.json"), new())); Assert.Empty(Directory.GetFiles(target));
        children.Remove(process);
    }

    public void Dispose()
    {
        foreach (var process in children)
        {
            try { if (!process.HasExited) { process.Kill(); Assert.True(process.WaitForExit(5000), "Exact owned fixture child must close."); } }
            finally { process.Dispose(); }
        }
        // Keep private source-bound state/custody witnesses; no broad deletion.
    }

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool CreateHardLinkW(string name, string existing, IntPtr security);
}
