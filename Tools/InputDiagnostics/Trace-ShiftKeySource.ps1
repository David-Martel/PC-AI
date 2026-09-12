#Requires -Version 7.0
<#
.SYNOPSIS
    Device-aware Shift-key trace. Uses the Raw Input API (WM_INPUT, RIDEV_INPUTSINK)
    to log every key event TOGETHER WITH the physical device that produced it, so a
    single capture session distinguishes the INTERNAL ThinkPad keyboard (ACPI\LEN0071,
    i8042/PS-2) from any USB keyboard.

.DESCRIPTION
    This supplies device-attributed evidence for the "internal Shift is intermittent"
    symptom on the ThinkPad P1 Gen 7. The existing Test-KeyInput.ps1
    installs a WH_KEYBOARD_LL hook, which sees the MERGED input stream and CANNOT tell
    which keyboard sent a key. Raw Input exposes RAWINPUTHEADER.hDevice, which this script
    resolves to a device name and classifies as INTERNAL vs USB.

    Because it registers with RIDEV_INPUTSINK, it captures even when this window is NOT
    focused -- type into your normal app (editor, browser) and reproduce the Shift loss
    while this runs in the background.

    A received event proves receipt at this observer for that event only. Correlate
    the exact failed trial, device, modifier state and application outcome. Missing
    events can reflect an unperformed trial, capture limitations, drivers or hardware;
    their absence alone does not identify an EC or physical fault. A successful hold
    does not exclude an intermittent failure outside the observed interval.

    Timestamps are user-mode receipt times, not hardware interrupt timestamps.
    Use known test input; -AllKeys records key codes that can reveal typed content.

    Read-only: registers a passive raw-input sink; never blocks, injects, or remaps keys.

.PARAMETER Seconds
    Capture duration. Default 30.

.PARAMETER OutputDir
    Where to write the JSON capture. Default: PC_AI\Logs\input-diagnostics.

.PARAMETER AllKeys
    Log every key (not just Shift/modifiers). Useful to prove the internal keyboard is
    otherwise alive while Shift is dropped.

.PARAMETER NavigationKeys
    Include arrows, Home, End, Page Up/Down, Insert and Delete alongside modifiers.
    Does not include letters, digits or text. Alias: NavKeys.

.EXAMPLE
    pwsh -File .\Trace-ShiftKeySource.ps1 -Seconds 45
    # Then, in your normal app, press internal Left/Right Shift, Shift+A, and (if attached)
    # the USB keyboard's Shift. Reproduce the failure. The summary attributes each Shift
    # event to INTERNAL vs USB.

.NOTES
    Author: input-stack investigation (Claude Code) - 2026-06-19.
    Companion to Test-KeyInput.ps1 (LL hook, device-agnostic) and Watch-InputGlitch.ps1.
#>
[CmdletBinding()]
param(
    [ValidateRange(1, 600)][int]$Seconds = 30,
    [string]$OutputDir = "$PSScriptRoot\..\..\Logs\input-diagnostics",
    [switch]$AllKeys,
    [Alias('NavKeys')][switch]$NavigationKeys
)

$cs = @'
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;
using System.Windows.Forms;

public static class RawKb
{
    public const int WM_INPUT = 0x00FF;
    const uint RID_INPUT = 0x10000003;
    const uint RIDI_DEVICENAME = 0x20000007;
    const uint RIDEV_INPUTSINK = 0x00000100;

    [StructLayout(LayoutKind.Sequential)]
    struct RAWINPUTDEVICE { public ushort UsagePage; public ushort Usage; public uint Flags; public IntPtr hwndTarget; }

    [DllImport("user32.dll", SetLastError = true)]
    static extern bool RegisterRawInputDevices(RAWINPUTDEVICE[] pRawInputDevices, uint uiNumDevices, uint cbSize);
    [DllImport("user32.dll", SetLastError = true)]
    static extern uint GetRawInputData(IntPtr hRawInput, uint uiCommand, IntPtr pData, ref uint pcbSize, uint cbSizeHeader);
    [DllImport("user32.dll", SetLastError = true)]
    static extern uint GetRawInputDeviceInfoW(IntPtr hDevice, uint uiCommand, IntPtr pData, ref uint pcbSize);
    [DllImport("user32.dll")] static extern IntPtr GetForegroundWindow();
    [DllImport("user32.dll")] static extern uint GetWindowThreadProcessId(IntPtr hwnd, out uint processId);

    public class Evt
    {
        public string Time; public string Dev; public string Class; public int Make;
        public int VKey; public string Dir; public bool E0; public string Name;
        public long MonotonicTicks; public uint ForegroundPid;
    }

    public class CaptureHealth
    {
        public long InputMessages, RawReadFailures, MalformedPackets, KeyboardPackets;
        public long FilteredPackets, NameLookups, NameFailures;
        public int LastReadError, LastNameError, RegistrationError, UnregisterError;
        public bool Registered, Stopped;
    }
    public static CaptureHealth Health = new CaptureHealth();
    public static long TimestampFrequency { get { return Stopwatch.Frequency; } }

    public static List<Evt> Events = new List<Evt>();
    static Dictionary<IntPtr, string> _names = new Dictionary<IntPtr, string>();
    static bool _allKeys = false;
    static bool _navigationKeys = false;
    static bool _registered = false;

    public static void Reset(bool allKeys, bool navigationKeys)
    {
        if (_sink != null) throw new InvalidOperationException("Capture already active.");
        _allKeys = allKeys; _navigationKeys = navigationKeys;
        Events.Clear(); _names.Clear(); Health = new CaptureHealth();
    }

    public static bool ShouldCapture(int vk, bool allKeys, bool navigationKeys)
    {
        bool modifier = (vk >= 160 && vk <= 165) || vk == 16 || vk == 17 || vk == 18 || vk == 91 || vk == 92;
        bool navigation = (vk >= 33 && vk <= 40) || vk == 45 || vk == 46;
        return allKeys || modifier || (navigationKeys && navigation);
    }

    public static string ResolveDeviceName(IntPtr h, Func<string> lookup)
    {
        string name;
        if (_names.TryGetValue(h, out name)) return name;
        Health.NameLookups++;
        name = lookup();
        if (String.IsNullOrWhiteSpace(name)) { Health.NameFailures++; return "(unknown)"; }
        _names[h] = name;
        return name;
    }

    static string DeviceName(IntPtr h)
    {
        return ResolveDeviceName(h, () => ReadDeviceName(h));
    }

    static string ReadDeviceName(IntPtr h)
    {
        uint size = 0;
        if (GetRawInputDeviceInfoW(h, RIDI_DEVICENAME, IntPtr.Zero, ref size) == UInt32.MaxValue)
        { Health.LastNameError = Marshal.GetLastWin32Error(); return null; }
        if (size > 0 && size <= 32768)
        {
            IntPtr buf = Marshal.AllocHGlobal((int)(size * 2));
            try {
                uint result = GetRawInputDeviceInfoW(h, RIDI_DEVICENAME, buf, ref size);
                if (result != UInt32.MaxValue && result > 0) return Marshal.PtrToStringUni(buf, (int)result).TrimEnd('\0');
                Health.LastNameError = Marshal.GetLastWin32Error();
            }
            finally { Marshal.FreeHGlobal(buf); }
        }
        return null;
    }

    static string Classify(string devName)
    {
        if (devName == null || devName == "(unknown)") return "UNKNOWN";
        string u = devName.ToUpperInvariant();
        if (u.Contains("ACPI") || u.Contains("LEN0071")) return "INTERNAL";
        if (u.Contains("VID_") || u.Contains("USB") || u.Contains("HID")) return "USB/HID";
        return "OTHER";
    }

    static string KeyName(int vk, int make, bool e0)
    {
        switch (vk)
        {
            case 16: return make == 0x36 ? "RSHIFT" : "LSHIFT";
            case 160: return "LSHIFT"; case 161: return "RSHIFT";
            case 17: return e0 ? "RCTRL" : "LCTRL";
            case 18: return e0 ? "RALT" : "LALT";
            case 162: return "LCTRL"; case 163: return "RCTRL";
            case 164: return "LALT"; case 165: return "RALT";
            case 91: return "LWIN"; case 92: return "RWIN";
            case 33: return "PAGEUP"; case 34: return "PAGEDOWN";
            case 35: return "END"; case 36: return "HOME";
            case 37: return "LEFT"; case 38: return "UP";
            case 39: return "RIGHT"; case 40: return "DOWN";
            case 45: return "INSERT"; case 46: return "DELETE";
            default: return "VK=" + vk;
        }
    }

    public static void ProcessPacket(IntPtr buf, uint bytes, long monotonicTicks, uint foregroundPid)
    {
        int hdr = 8 + 2 * IntPtr.Size;
        if (buf == IntPtr.Zero || bytes < hdr) { Health.MalformedPackets++; return; }
        uint declared = unchecked((uint)Marshal.ReadInt32(buf, 4));
        if (declared > bytes || declared < hdr) { Health.MalformedPackets++; return; }
        if (Marshal.ReadInt32(buf, 0) != 1) return;
        if (declared < hdr + 16) { Health.MalformedPackets++; return; }
        Health.KeyboardPackets++;
        int make = (ushort)Marshal.ReadInt16(buf, hdr);
        int flags = (ushort)Marshal.ReadInt16(buf, hdr + 2);
        int vkey = (ushort)Marshal.ReadInt16(buf, hdr + 6);
        if (make == 0xFF || vkey == 0xFF || !ShouldCapture(vkey, _allKeys, _navigationKeys))
        { Health.FilteredPackets++; return; }
        bool e0 = (flags & 2) != 0;
        string dn = DeviceName(Marshal.ReadIntPtr(buf, 8));
        Events.Add(new Evt {
            Time = DateTime.Now.ToString("HH:mm:ss.fff"), MonotonicTicks = monotonicTicks,
            ForegroundPid = foregroundPid, Dev = dn, Class = Classify(dn), Make = make, VKey = vkey,
            Dir = (flags & 1) != 0 ? "UP" : "DOWN", E0 = e0, Name = KeyName(vkey, make, e0)
        });
    }

    class Sink : NativeWindow
    {
        public Sink() { CreateHandle(new CreateParams { Parent = (IntPtr)(-3) }); } // HWND_MESSAGE
        protected override void WndProc(ref Message m)
        {
            if (m.Msg == WM_INPUT) OnInput(m.LParam);
            base.WndProc(ref m);
        }
        void OnInput(IntPtr hRawInput)
        {
            Health.InputMessages++;
            long received = Stopwatch.GetTimestamp();
            uint foregroundPid;
            GetWindowThreadProcessId(GetForegroundWindow(), out foregroundPid);
            uint hdr = (uint)(8 + 2 * IntPtr.Size);
            uint sz = 0;
            if (GetRawInputData(hRawInput, RID_INPUT, IntPtr.Zero, ref sz, hdr) == UInt32.MaxValue)
            { Health.RawReadFailures++; Health.LastReadError = Marshal.GetLastWin32Error(); return; }
            if (sz < hdr || sz > 65536) { Health.MalformedPackets++; return; }
            IntPtr buf = Marshal.AllocHGlobal((int)sz);
            try
            {
                uint capacity = sz;
                uint read = GetRawInputData(hRawInput, RID_INPUT, buf, ref sz, hdr);
                if (read == UInt32.MaxValue || read != sz || read > capacity)
                { Health.RawReadFailures++; Health.LastReadError = Marshal.GetLastWin32Error(); return; }
                ProcessPacket(buf, read, received, foregroundPid);
            }
            finally { Marshal.FreeHGlobal(buf); }
        }
    }

    static Sink _sink;
    public static bool Start(bool allKeys)
    { return Start(allKeys, false); }
    public static bool Start(bool allKeys, bool navigationKeys)
    {
        Reset(allKeys, navigationKeys);
        _sink = new Sink();
        var rid = new RAWINPUTDEVICE[1];
        rid[0].UsagePage = 0x01; rid[0].Usage = 0x06;       // generic desktop / keyboard
        rid[0].Flags = RIDEV_INPUTSINK; rid[0].hwndTarget = _sink.Handle;
        Health.Registered = RegisterRawInputDevices(rid, 1, (uint)Marshal.SizeOf(typeof(RAWINPUTDEVICE)));
        _registered = Health.Registered;
        if (!Health.Registered) { Health.RegistrationError = Marshal.GetLastWin32Error(); Stop(); }
        return Health.Registered;
    }
    public static void Pump() { Application.DoEvents(); }
    public static void Stop()
    {
        try {
            if (_registered) {
                var rid = new RAWINPUTDEVICE[1];
                rid[0].UsagePage = 1; rid[0].Usage = 6; rid[0].Flags = 1; // RIDEV_REMOVE, null hwnd
                if (!RegisterRawInputDevices(rid, 1, (uint)Marshal.SizeOf(typeof(RAWINPUTDEVICE))))
                    Health.UnregisterError = Marshal.GetLastWin32Error();
            }
        } finally {
            try { if (_sink != null) _sink.DestroyHandle(); }
            finally { _sink = null; _registered = false; Health.Stopped = true; }
        }
    }
}
'@

Add-Type -AssemblyName System.Windows.Forms
Add-Type -TypeDefinition $cs -ReferencedAssemblies System.Windows.Forms, 'System.Windows.Forms.Primitives', System.Drawing, System.Collections, System.Runtime.InteropServices -ErrorAction Stop

try {
    if (-not [RawKb]::Start([bool]$AllKeys, [bool]$NavigationKeys)) {
        throw "RegisterRawInputDevices failed (LastError=$([RawKb]::Health.RegistrationError))"
    }

Write-Host "Device-aware keyboard trace running for $Seconds s. Switch to your test app." -ForegroundColor Cyan
if ($NavigationKeys) {
    Write-Host '  Compare labeled internal/USB arrow trials, including the affected directions.' -ForegroundColor Cyan
    Write-Host '  Modifiers and other navigation keys are included; ordinary typing is excluded unless AllKeys is set.' -ForegroundColor Cyan
} else {
    Write-Host '  Compare internal Left Shift, Right Shift and Shift+A with USB Shift controls.' -ForegroundColor Cyan
}

# Live JSONL log so a background capture is observable in real time (tail the file).
$null = New-Item -ItemType Directory -Path $OutputDir -Force -ErrorAction Stop
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$liveFile = Join-Path $OutputDir "shift-source-live-$stamp.jsonl"
[IO.File]::WriteAllText($liveFile, '')
Write-Host "Live log: $liveFile" -ForegroundColor DarkCyan

$sw = [System.Diagnostics.Stopwatch]::StartNew()
$idx = 0
do {
    [RawKb]::Pump()
    $finished = $sw.Elapsed.TotalSeconds -ge $Seconds
    $all = [RawKb]::Events
    while ($all.Count -gt $idx) {
        $e = $all[$idx]; $idx++
        ('{0}' -f ([pscustomobject]@{ t=$e.Time; cls=$e.Class; name=$e.Name; dir=$e.Dir; make=$e.Make; vk=$e.VKey; dev=$e.Dev; e0=$e.E0; monotonicTicks=$e.MonotonicTicks; foregroundPid=$e.ForegroundPid } | ConvertTo-Json -Compress)) |
            Add-Content -Path $liveFile -Encoding UTF8 -ErrorAction Stop
    }
    if (-not $finished) { Start-Sleep -Milliseconds 8 }
} while (-not $finished)
} finally { [RawKb]::Stop() }

$events = [RawKb]::Events
Write-Host "`n===== CAPTURED ($($events.Count) selected events) =====" -ForegroundColor Cyan
foreach ($e in $events) {
    $color = if ($e.Class -eq 'INTERNAL') { 'Yellow' } else { 'Gray' }
    Write-Host ("{0}  {1,-8} {2,-6} make=0x{3:X2} vk={4,-3} [{5}]  {6}" -f `
        $e.Time, $e.Class, "$($e.Name) $($e.Dir)", $e.Make, $e.VKey, $e.Class, $e.Dev) -ForegroundColor $color
}

# ---- Per-device Shift summary ----
$shift = $events | Where-Object { $_.Name -in 'LSHIFT','RSHIFT' }
$byDev = $shift | Group-Object Dev
Write-Host "`n===== SHIFT-BY-DEVICE SUMMARY =====" -ForegroundColor Cyan
$summary = foreach ($g in $byDev) {
    $cls = ($g.Group | Select-Object -First 1).Class
    $down = ($g.Group | Where-Object Dir -eq 'DOWN').Count
    $up   = ($g.Group | Where-Object Dir -eq 'UP').Count
    Write-Host ("  [{0,-8}] down={1} up={2}  {3}" -f $cls, $down, $up, $g.Name)
    [pscustomobject]@{ Device = $g.Name; Class = $cls; ShiftDown = $down; ShiftUp = $up }
}

$internalShift = ($shift | Where-Object Class -eq 'INTERNAL').Count
$usbShift      = ($shift | Where-Object Class -ne 'INTERNAL').Count
$internalAny   = ($events | Where-Object Class -eq 'INTERNAL').Count

Write-Host "`n===== OBSERVATION (ROOT CAUSE UNDETERMINED) =====" -ForegroundColor Cyan
if ($internalShift -gt 0) {
    Write-Host "Observed $internalShift internal Shift events in this capture."
    Write-Host "Correlate the exact failed trial and app outcome; other failures remain possible."
} elseif ($internalAny -gt 0) {
    Write-Host "Observed internal input ($internalAny events), but no internal Shift events."
    Write-Host "Verify a Shift trial occurred and the observer was healthy before localizing the fault."
} else {
    Write-Host "No internal-keyboard events captured. Re-run and ensure you press keys on the BUILT-IN keyboard" -ForegroundColor Yellow
    Write-Host "during the window (and that you actually reproduced the failure)." -ForegroundColor Yellow
}
if ($usbShift -gt 0) { Write-Host "Other-device Shift events observed: $usbShift; this alone does not validate the control trial." }

# ---- Persist JSON ----
$outFile = Join-Path $OutputDir "shift-source-trace-$stamp.json"
[pscustomobject]@{
    capturedAt    = (Get-Date).ToString('o')
    durationSec   = $Seconds
    allKeys       = [bool]$AllKeys
    navigationKeys = [bool]$NavigationKeys
    timestampFrequency = [RawKb]::TimestampFrequency
    health        = [RawKb]::Health
    machine       = $env:COMPUTERNAME
    totalEvents   = $events.Count
    internalShift = $internalShift
    usbShift      = $usbShift
    summary       = $summary
    events        = $events
} | ConvertTo-Json -Depth 5 | Set-Content -Path $outFile -Encoding UTF8
Write-Host "`nSaved: $outFile" -ForegroundColor DarkCyan
