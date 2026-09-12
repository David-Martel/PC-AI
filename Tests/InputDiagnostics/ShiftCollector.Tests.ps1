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
