#Requires -Version 7.0
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }

BeforeAll {
    $script:CollectorPath = Join-Path $PSScriptRoot '../../Tools/InputDiagnostics/Trace-ShiftKeySource.ps1'
    $source = Get-Content -LiteralPath $script:CollectorPath -Raw
    $code = [regex]::Match($source, "(?s)\`$cs = @'\r?\n(.*?)\r?\n'@").Groups[1].Value
    Add-Type -AssemblyName System.Windows.Forms
    Add-Type -TypeDefinition $code -ReferencedAssemblies System.Windows.Forms, System.Windows.Forms.Primitives, System.Drawing, System.Collections, System.Runtime.InteropServices
    Add-Type -TypeDefinition @'
using System;
using System.Collections.Generic;
public static class RawKbFixture {
    public class Event {
        public string Time = "12:00:00.000", Dev = "fixture", Class = "INTERNAL", Name = "RSHIFT", Dir = "UP";
        public int Make = 54, VKey = 161; public bool E0; public long MonotonicTicks = 1; public uint ForegroundPid = 2;
    }
    public static List<Event> Events = new List<Event>();
    public static int StopCalls, PumpCalls;
    public static long TimestampFrequency = 1000;
    public static object Health = new object();
    public static bool Start(bool all, bool navigation) { Events.Clear(); StopCalls = 0; PumpCalls = 0; return true; }
    public static void Pump() { if (++PumpCalls == 2) Events.Add(new Event()); }
    public static void Stop() { StopCalls++; }
}
namespace KbMonFixture {
    public static class Hook {
        public delegate IntPtr HookProc(int code, IntPtr w, IntPtr l);
        public static int UnhookCalls;
        public static IntPtr SetWindowsHookExW(int id, HookProc proc, IntPtr module, uint thread) { return (IntPtr)123; }
        public static bool UnhookWindowsHookEx(IntPtr handle) { UnhookCalls++; return true; }
        public static IntPtr GetModuleHandleW(string name) { return IntPtr.Zero; }
        public static IntPtr CallNextHookEx(IntPtr handle, int code, IntPtr w, IntPtr l) { return IntPtr.Zero; }
    }
}
'@
    # Exercise the real PowerShell orchestration with only its native boundary replaced.
    $normalized = $source.Replace("`r`n", "`n")
    $tail = $normalized.Substring($normalized.IndexOf("try {`n    if (-not [RawKb]::Start"))
    $script:RawOrchestration = [scriptblock]::Create('param($OutputDir, $Seconds = 1, [switch]$AllKeys, [switch]$NavigationKeys)' + "`n" + $tail.Replace('[RawKb]', '[RawKbFixture]'))
    $llPath = Join-Path $PSScriptRoot '../../Tools/InputDiagnostics/Test-KeyInput.ps1'
    $llSource = Get-Content -LiteralPath $llPath -Raw
    $ast = [System.Management.Automation.Language.Parser]::ParseInput($llSource, [ref]$null, [ref]$null)
    $adds = $ast.FindAll({ param($node) $node -is [System.Management.Automation.Language.CommandAst] -and $node.GetCommandName() -eq 'Add-Type' }, $true)
    foreach ($add in ($adds | Sort-Object { $_.Extent.StartOffset } -Descending)) {
        $llSource = $llSource.Remove($add.Extent.StartOffset, $add.Extent.EndOffset - $add.Extent.StartOffset)
    }
    $script:LlOrchestration = [scriptblock]::Create($llSource.Replace('[KbMon.Hook', '[KbMonFixture.Hook'))
    function Send-Fixture {
        param([int]$VKey = 160, [int]$Make = 42, [int]$Flags = 0, [int]$Length = 0)
        $header = 8 + 2 * [IntPtr]::Size
        if (-not $Length) { $Length = $header + 16 }
        $buffer = [Runtime.InteropServices.Marshal]::AllocHGlobal($Length)
        try {
            $bytes = [byte[]]::new($Length)
            [Runtime.InteropServices.Marshal]::Copy($bytes, 0, $buffer, $Length)
            if ($Length -ge ($header + 16)) {
                [Runtime.InteropServices.Marshal]::WriteInt32($buffer, 0, 1)
                [Runtime.InteropServices.Marshal]::WriteInt32($buffer, 4, $Length)
                [Runtime.InteropServices.Marshal]::WriteInt16($buffer, $header, [int16]$Make)
                [Runtime.InteropServices.Marshal]::WriteInt16($buffer, ($header + 2), [int16]$Flags)
                [Runtime.InteropServices.Marshal]::WriteInt16($buffer, ($header + 6), [int16]$VKey)
            }
            [RawKb]::ProcessPacket($buffer, [uint32]$Length, 12345, 4242)
        } finally { [Runtime.InteropServices.Marshal]::FreeHGlobal($buffer) }
    }
}

Describe 'Collector help and dry-run suppress capture side effects' {
    It 'returns the requested preview without loading native code or creating an output directory' {
        $output = Join-Path $TestDrive 'preview-must-stay-absent'
        Mock Add-Type { throw 'Preview loaded native code' }
        Mock New-Item { throw 'Preview created a directory' }
        Mock Add-Content { throw 'Preview appended a capture' }
        Mock Set-Content { throw 'Preview wrote a capture' }
        $preview = & $script:CollectorPath -DryRun -Seconds 17 -OutputDir $output -AllKeys -NavKeys
        $preview.Mode | Should -Be 'DryRun'
        $preview.DurationSeconds | Should -Be 17
        $preview.OutputDirectory | Should -Be $output
        $preview.AllKeys | Should -BeTrue
        $preview.NavigationKeys | Should -BeTrue
        $preview.NativeRegistrationRequested | Should -BeFalse
        $preview.WritesRequested | Should -BeFalse
        Test-Path -LiteralPath $output | Should -BeFalse
        Should -Invoke Add-Type -Times 0
        Should -Invoke New-Item -Times 0
        Should -Invoke Add-Content -Times 0
        Should -Invoke Set-Content -Times 0
    }

    It 'rejects unsupported arguments before native loading or output writes' {
        $output = Join-Path $TestDrive 'invalid-must-stay-absent'
        Mock Add-Type { throw 'Invalid request loaded native code' }
        { & $script:CollectorPath --unknown -OutputDir $output } | Should -Throw '*Unsupported argument*'
        Should -Invoke Add-Type -Times 0
        Test-Path -LiteralPath $output | Should -BeFalse
    }

    It 'runs the actual script in a fresh process without loading the native collector: <Mode>' -ForEach @(
        @{ Mode = 'DryRun'; ExistingDirectory = $false }
        @{ Mode = 'DryRun'; ExistingDirectory = $true }
        @{ Mode = 'Help'; ExistingDirectory = $false }
        @{ Mode = 'ShortHelp'; ExistingDirectory = $false }
        @{ Mode = 'LongHelp'; ExistingDirectory = $false }
    ) {
        $runner = Join-Path $TestDrive 'preview-runner.ps1'
        $output = Join-Path $TestDrive ('capture-' + $Mode + '-' + $ExistingDirectory)
        if ($ExistingDirectory) {
            $null = New-Item -ItemType Directory -Path $output
            [IO.File]::WriteAllText((Join-Path $output 'existing.jsonl'), 'existing capture must be preserved')
        }
        $before = if ($ExistingDirectory) { (Get-FileHash -LiteralPath (Join-Path $output 'existing.jsonl')).Hash } else { $null }
        @'
param([string]$CollectorPath, [string]$OutputDir, [string]$Mode)
if ('RawKb' -as [type]) { throw 'Fresh preview process already has a collector type.' }
$result = switch ($Mode) {
    DryRun { & $CollectorPath -DryRun -OutputDir $OutputDir -Seconds 17 -NavigationKeys -AllKeys }
    Help { & $CollectorPath -Help -OutputDir $OutputDir }
    ShortHelp { & $CollectorPath -h -OutputDir $OutputDir }
    LongHelp { & $CollectorPath --help -OutputDir $OutputDir }
    default { throw 'Unknown fixture mode.' }
}
if ('RawKb' -as [type]) { throw 'Preview loaded the native collector type.' }
$result | ConvertTo-Json -Depth 4 -Compress
'@ | Set-Content -LiteralPath $runner
        $start = [Diagnostics.ProcessStartInfo]::new((Get-Command pwsh).Source)
        $start.UseShellExecute = $false
        $start.CreateNoWindow = $true
        $start.RedirectStandardOutput = $true
        $start.RedirectStandardError = $true
        foreach ($argument in @('-NoLogo', '-NoProfile', '-File', $runner, '-CollectorPath', $script:CollectorPath, '-OutputDir', $output, '-Mode', $Mode)) {
            $start.ArgumentList.Add($argument)
        }
        $process = [Diagnostics.Process]::Start($start)
        try {
            $stdout = $process.StandardOutput.ReadToEndAsync()
            $stderr = $process.StandardError.ReadToEndAsync()
            if (-not $process.WaitForExit(10000)) {
                $process.Kill($true)
                throw 'Collector preview exceeded ten seconds.'
            }
            $process.ExitCode | Should -Be 0
            $stderr.GetAwaiter().GetResult() | Should -BeNullOrEmpty
            $result = $stdout.GetAwaiter().GetResult() | ConvertFrom-Json
            if ($Mode -eq 'DryRun') {
                $result.Mode | Should -Be 'DryRun'
                $result.DurationSeconds | Should -Be 17
                $result.OutputDirectory | Should -Be $output
                $result.AllKeys | Should -BeTrue
                $result.NavigationKeys | Should -BeTrue
                $result.NativeRegistrationRequested | Should -BeFalse
                $result.WritesRequested | Should -BeFalse
            }
            else {
                $result | Should -Match 'Trace-ShiftKeySource.ps1.*-DryRun.*--help'
            }
            if ($ExistingDirectory) {
                @(Get-ChildItem -LiteralPath $output -Force).Count | Should -Be 1
                (Get-FileHash -LiteralPath (Join-Path $output 'existing.jsonl')).Hash | Should -Be $before
            }
            else {
                Test-Path -LiteralPath $output | Should -BeFalse
            }
        }
        finally { $process.Dispose() }
    }
}

Describe 'Collector orchestration cleanup and persistence' {
    BeforeEach { Mock Write-Host {} }
    It 'drains events from the final pump into both live JSONL and final JSON' {
        Mock Start-Sleep { [Threading.Thread]::Sleep(1100) }
        $output = Join-Path $TestDrive 'final-pump'
        & $script:RawOrchestration -OutputDir $output
        $live = Get-ChildItem -LiteralPath $output -Filter '*.jsonl' | Get-Content | ConvertFrom-Json
        $final = Get-ChildItem -LiteralPath $output -Filter '*.json' | Get-Content -Raw | ConvertFrom-Json
        @($live).Count | Should -Be 1
        $final.totalEvents | Should -Be 1
        $live.name | Should -Be $final.events[0].Name
        $live.monotonicTicks | Should -Be $final.events[0].MonotonicTicks
        [RawKbFixture]::StopCalls | Should -Be 1
    }
    It 'cleans up the raw sink when output creation fails' {
        Mock New-Item { throw 'fixture output failure' }
        { & $script:RawOrchestration -OutputDir $TestDrive } | Should -Throw '*fixture output failure*'
        [RawKbFixture]::StopCalls | Should -Be 1
    }
    It 'unhooks the merged monitor when its capture loop throws' {
        [KbMonFixture.Hook]::UnhookCalls = 0
        Mock Start-Sleep { throw 'fixture interrupted capture' }
        { & $script:LlOrchestration -Seconds 1 } | Should -Throw '*fixture interrupted capture*'
        [KbMonFixture.Hook]::UnhookCalls | Should -Be 1
    }
}

Describe 'Raw keyboard collector offline behavior' {
    BeforeEach { [RawKb]::Reset($false, $false) }
    It 'keeps the default modifier-only and limits NavKeys to arrows plus modifiers' {
        [RawKb]::ShouldCapture(160, $false, $false) | Should -BeTrue
        [RawKb]::ShouldCapture(39, $false, $false) | Should -BeFalse
        foreach ($vk in @((33..40) + 45 + 46)) { [RawKb]::ShouldCapture($vk, $false, $true) | Should -BeTrue }
        foreach ($vk in 48, 65, 90, 13, 32) { [RawKb]::ShouldCapture($vk, $false, $true) | Should -BeFalse }
        [RawKb]::ShouldCapture(65, $true, $false) | Should -BeTrue
    }
    It 'rejects truncated packets before reading the keyboard union and counts them' {
        Send-Fixture -Length 4
        [RawKb]::Events.Count | Should -Be 0
        [RawKb]::Health.MalformedPackets | Should -Be 1
    }
    It 'preserves arrow direction, extended flag, monotonic receipt and foreground PID' {
        [RawKb]::Reset($false, $true)
        Send-Fixture -VKey 39 -Make 77 -Flags 3
        $event = [RawKb]::Events[0]
        $event.Name | Should -Be 'RIGHT'
        $event.Dir | Should -Be 'UP'
        $event.E0 | Should -BeTrue
        $event.MonotonicTicks | Should -Be 12345
        $event.ForegroundPid | Should -Be 4242
    }
    It 'does not misclassify a non-modifier solely because its make code matches Shift' {
        Send-Fixture -VKey 65 -Make 42
        [RawKb]::Events.Count | Should -Be 0
        [RawKb]::Health.FilteredPackets | Should -Be 1
    }
    It 'retries failed device resolution and caches only a successful name' {
        $device = [IntPtr]123
        [RawKb]::ResolveDeviceName($device, [Func[string]]{ $null }) | Should -Be '(unknown)'
        [RawKb]::ResolveDeviceName($device, [Func[string]]{ 'fixture-device' }) | Should -Be 'fixture-device'
        [RawKb]::ResolveDeviceName($device, [Func[string]]{ throw 'cached lookup must not run' }) | Should -Be 'fixture-device'
        [RawKb]::Health.NameLookups | Should -Be 2
        [RawKb]::Health.NameFailures | Should -Be 1
    }
    It 'names generic Ctrl and Alt using the extended bit' {
        Send-Fixture -VKey 17 -Make 29 -Flags 2
        Send-Fixture -VKey 18 -Make 56
        [RawKb]::Events[0].Name | Should -Be 'RCTRL'
        [RawKb]::Events[1].Name | Should -Be 'LALT'
    }
}
