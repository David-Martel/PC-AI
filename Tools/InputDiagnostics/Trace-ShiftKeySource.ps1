#Requires -Version 7.0
<#
.SYNOPSIS
    Device-aware Shift-key trace. Uses the Raw Input API (WM_INPUT, RIDEV_INPUTSINK)
    to log every key event TOGETHER WITH the physical device that produced it, so a
    single capture session distinguishes the INTERNAL ThinkPad keyboard (ACPI\LEN0071,
    i8042/PS-2) from any USB keyboard.

.DESCRIPTION
    This is the decisive discriminator for the "internal Shift is intermittent but USB
    Shift always works" symptom on the ThinkPad P1 Gen 7. The existing Test-KeyInput.ps1
    installs a WH_KEYBOARD_LL hook, which sees the MERGED input stream and CANNOT tell
    which keyboard sent a key. Raw Input exposes RAWINPUTHEADER.hDevice, which this script
    resolves to a device name and classifies as INTERNAL vs USB.

    Because it registers with RIDEV_INPUTSINK, it captures even when this window is NOT
    focused -- type into your normal app (editor, browser) and reproduce the Shift loss
    while this runs in the background.

    Interpretation when the internal Shift "fails" during the window:
      - Internal Shift DOWN/UP events ARE present for the internal device  -> scancodes
        reach Windows; the loss is above the driver (focused app / IME / a hook). Software.
      - Internal Shift events are ABSENT while internal letter keys still appear, AND a USB
        keyboard's Shift appears fine -> the internal Shift scancode never reaches Windows.
        Embedded-controller (EC) firmware or a physical keyboard-matrix/contact fault.
        Next step: Lenovo BIOS/EC update (Commercial Vantage) + EC reset, then warranty.

    Read-only: registers a passive raw-input sink; never blocks, injects, or remaps keys.

.PARAMETER Seconds
    Capture duration. Default 30.

.PARAMETER OutputDir
    Where to write the JSON capture. Default: PC_AI\Logs\input-diagnostics.

.PARAMETER AllKeys
    Log every key (not just Shift/modifiers). Useful to prove the internal keyboard is
    otherwise alive while Shift is dropped.

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
    [int]$Seconds = 30,
    [string]$OutputDir = "$PSScriptRoot\..\..\Logs\input-diagnostics",
    [switch]$AllKeys
)

$cs = @'
using System;
using System.Collections.Generic;
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

    public class Evt
    {
        public string Time; public string Dev; public string Class; public int Make;
        public int VKey; public string Dir; public bool E0; public string Name;
    }

    public static List<Evt> Events = new List<Evt>();
    static Dictionary<IntPtr, string> _names = new Dictionary<IntPtr, string>();
    static bool _allKeys = false;

    static string DeviceName(IntPtr h)
    {
        string n;
        if (_names.TryGetValue(h, out n)) return n;
        uint size = 0;
        GetRawInputDeviceInfoW(h, RIDI_DEVICENAME, IntPtr.Zero, ref size);
        n = "(unknown)";
        if (size > 0)
        {
            IntPtr buf = Marshal.AllocHGlobal((int)(size * 2));
            try { if (GetRawInputDeviceInfoW(h, RIDI_DEVICENAME, buf, ref size) != unchecked((uint)-1)) n = Marshal.PtrToStringUni(buf); }
            finally { Marshal.FreeHGlobal(buf); }
        }
        _names[h] = n;
        return n;
    }

    static string Classify(string devName)
    {
        if (devName == null) return "UNKNOWN";
        string u = devName.ToUpperInvariant();
        if (u.Contains("ACPI") || u.Contains("LEN0071")) return "INTERNAL";
        if (u.Contains("VID_") || u.Contains("USB") || u.Contains("HID")) return "USB/HID";
        return "OTHER";
    }

    static string KeyName(int vk, int make)
    {
        if (make == 0x2A) return "LSHIFT";
        if (make == 0x36) return "RSHIFT";
        switch (vk)
        {
            case 16: case 160: return "LSHIFT"; case 161: return "RSHIFT";
            case 162: return "LCTRL"; case 163: return "RCTRL";
            case 164: return "LALT"; case 165: return "RALT";
            default: return "VK=" + vk;
        }
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
            uint sz = 0;
            GetRawInputData(hRawInput, RID_INPUT, IntPtr.Zero, ref sz, (uint)(4 + 4 + IntPtr.Size + IntPtr.Size));
            if (sz == 0) return;
            IntPtr buf = Marshal.AllocHGlobal((int)sz);
            try
            {
                uint hdr = (uint)(4 + 4 + IntPtr.Size + IntPtr.Size);
                if (GetRawInputData(hRawInput, RID_INPUT, buf, ref sz, hdr) == unchecked((uint)-1)) return;
                int dwType = Marshal.ReadInt32(buf, 0);
                if (dwType != 1) return; // RIM_TYPEKEYBOARD
                IntPtr hDevice = Marshal.ReadIntPtr(buf, 8);
                int koff = (int)hdr;                                  // keyboard union begins after header
                ushort make = (ushort)Marshal.ReadInt16(buf, koff + 0);
                ushort flags = (ushort)Marshal.ReadInt16(buf, koff + 2);
                ushort vkey = (ushort)Marshal.ReadInt16(buf, koff + 6);
                bool brk = (flags & 0x01) != 0;                       // RI_KEY_BREAK = key up
                bool e0 = (flags & 0x02) != 0;
                if (make == 0xFF || vkey == 0xFF) return;             // overrun / fake-shift filler
                bool isMod = (vkey >= 160 && vkey <= 165) || vkey == 16 || vkey == 17 || vkey == 18 || make == 0x2A || make == 0x36;
                if (!_allKeys && !isMod) return;
                string dn = DeviceName(hDevice);
                Events.Add(new Evt {
                    Time = DateTime.Now.ToString("HH:mm:ss.fff"),
                    Dev = dn, Class = Classify(dn), Make = make, VKey = vkey,
                    Dir = brk ? "UP" : "DOWN", E0 = e0, Name = KeyName(vkey, make)
                });
            }
            finally { Marshal.FreeHGlobal(buf); }
        }
    }

    static Sink _sink;
    public static bool Start(bool allKeys)
    {
        _allKeys = allKeys;
        Events.Clear();
        _sink = new Sink();
        var rid = new RAWINPUTDEVICE[1];
        rid[0].UsagePage = 0x01; rid[0].Usage = 0x06;       // generic desktop / keyboard
        rid[0].Flags = RIDEV_INPUTSINK; rid[0].hwndTarget = _sink.Handle;
        return RegisterRawInputDevices(rid, 1, (uint)Marshal.SizeOf(typeof(RAWINPUTDEVICE)));
    }
    public static void Pump() { Application.DoEvents(); }
}
'@

Add-Type -AssemblyName System.Windows.Forms
Add-Type -TypeDefinition $cs -ReferencedAssemblies System.Windows.Forms, 'System.Windows.Forms.Primitives', System.Drawing, System.Collections, System.Runtime.InteropServices -ErrorAction Stop

if (-not [RawKb]::Start([bool]$AllKeys)) {
    Write-Error "RegisterRawInputDevices failed (LastError=$([System.Runtime.InteropServices.Marshal]::GetLastWin32Error()))"
    return
}

Write-Host "Device-aware Shift trace running for $Seconds s. Switch to your normal app and press:" -ForegroundColor Cyan
Write-Host "  internal Left Shift, internal Right Shift, internal Shift+A  -- and reproduce the failure." -ForegroundColor Cyan
Write-Host "  If a USB keyboard is attached, press its Shift too (control)." -ForegroundColor Cyan

# Live JSONL log so a background capture is observable in real time (tail the file).
$null = New-Item -ItemType Directory -Path $OutputDir -Force -ErrorAction SilentlyContinue
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$liveFile = Join-Path $OutputDir "shift-source-live-$stamp.jsonl"
Set-Content -Path $liveFile -Value '' -Encoding UTF8
Write-Host "Live log: $liveFile" -ForegroundColor DarkCyan

$sw = [System.Diagnostics.Stopwatch]::StartNew()
$idx = 0
while ($sw.Elapsed.TotalSeconds -lt $Seconds) {
    [RawKb]::Pump()
    $all = [RawKb]::Events
    while ($all.Count -gt $idx) {
        $e = $all[$idx]; $idx++
        ('{0}' -f ([pscustomobject]@{ t=$e.Time; cls=$e.Class; name=$e.Name; dir=$e.Dir; make=$e.Make; vk=$e.VKey; dev=$e.Dev } | ConvertTo-Json -Compress)) |
            Add-Content -Path $liveFile -Encoding UTF8
    }
    Start-Sleep -Milliseconds 8
}
[RawKb]::Pump()

$events = [RawKb]::Events
Write-Host "`n===== CAPTURED ($($events.Count) modifier events) =====" -ForegroundColor Cyan
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

Write-Host "`n===== VERDICT =====" -ForegroundColor Cyan
if ($internalShift -gt 0) {
    Write-Host "Internal Shift scancodes DID reach Windows ($internalShift). If Shift still 'failed'" -ForegroundColor Green
    Write-Host "in your app during this window, the loss is ABOVE the driver (focused app / IME / a hook)." -ForegroundColor Green
} elseif ($internalAny -gt 0) {
    Write-Host "Internal keyboard produced events ($internalAny modifier events) but ZERO internal Shift." -ForegroundColor Red
    Write-Host "=> The internal Shift scancode is NOT reaching Windows while the keyboard is otherwise alive." -ForegroundColor Red
    Write-Host "=> Embedded-controller (EC) firmware or physical matrix/contact fault. Update Lenovo BIOS/EC" -ForegroundColor Red
    Write-Host "   (Commercial Vantage), do an EC reset (power-drain), then pursue warranty if it persists." -ForegroundColor Red
} else {
    Write-Host "No internal-keyboard events captured. Re-run and ensure you press keys on the BUILT-IN keyboard" -ForegroundColor Yellow
    Write-Host "during the window (and that you actually reproduced the failure)." -ForegroundColor Yellow
}
if ($usbShift -gt 0) { Write-Host "USB/HID Shift events seen: $usbShift (control path healthy)." -ForegroundColor Green }

# ---- Persist JSON ----
$outFile = Join-Path $OutputDir "shift-source-trace-$stamp.json"
[pscustomobject]@{
    capturedAt    = (Get-Date).ToString('o')
    durationSec   = $Seconds
    allKeys       = [bool]$AllKeys
    machine       = $env:COMPUTERNAME
    totalEvents   = $events.Count
    internalShift = $internalShift
    usbShift      = $usbShift
    summary       = $summary
    events        = $events
} | ConvertTo-Json -Depth 5 | Set-Content -Path $outFile -Encoding UTF8
Write-Host "`nSaved: $outFile" -ForegroundColor DarkCyan
