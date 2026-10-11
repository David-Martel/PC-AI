# Lenovo keyboard driver and diagnostic research — 2026-09-12

Read-only investigation on `dtm-p1gen7`, started 21:15 UTC. No driver
installation, device restart/removal, firmware change, registry modification,
reboot, or trace recording was performed. Hardware serial numbers, UUIDs, and
device-instance identifiers are omitted.

## Finding

The laptop has a started PS/2 keyboard using Microsoft's `kbdclass → i8042prt →
ACPI` stack, alongside three started USB HID keyboard interfaces. The enumerated
effective keyboard stacks contain no additional third-party kernel filter.
This does not exclude user-mode hooks, remapping software, input-method problems,
firmware faults, or an intermittent physical connection. The PS/2 device is the
strong native-keyboard candidate; a physical input comparison is still needed to
tie a failure to that device.

Prioritize a built-in versus external-keyboard comparison during a failure, then
an out-of-Windows keyboard test. Serial IO/I2C drivers are a lower-priority lead
for ordinary native keystrokes because they do not appear in the observed PS/2
keyboard stack. They become relevant if touchpad or other I2C symptoms occur at
the same time. Microsoft's PS/2 architecture documentation identifies `kbdclass`
and `i8042prt` as the corresponding class and function drivers. [Microsoft
keyboard driver architecture](https://learn.microsoft.com/en-us/previous-versions/windows/hardware/hid/keyboard-and-mouse-class-drivers)

## Verified machine and driver inventory

Collected with selective CIM properties, registry OS-version metadata, and
`pnputil /enum-devices /class Keyboard /connected /stack /drivers`. Identification
and registration lines were filtered before output. Snapshot health is not proof
that an intermittent problem is absent.

| Item | Observed value |
| --- | --- |
| Manufacturer/model | Lenovo ThinkPad P1 Gen 7, `21KV0014US` |
| OS | Windows 11 Pro, version 25H2, build `26200.9445` |
| BIOS | `N48ET35W (1.22)`, SMBIOS release date June 23, 2026 UTC |
| Embedded controller | SMBIOS major/minor `1.15` |
| Standard PS/2 Keyboard | Microsoft `keyboard.inf`, `10.0.26100.8972`, signed, Started |
| PS/2 effective stack | `kbdclass`, `i8042prt`, `ACPI` |
| Three HID Keyboard Device interfaces | Microsoft `keyboard.inf`, `10.0.26100.8972`, signed, Started |
| USB HID keyboard effective stacks | `kbdclass`, `kbdhid`, `HidUsb` |
| Intel Serial IO GPIO INTC1083 | `30.100.2527.40`, `oem369.inf`, signed |
| Intel Serial IO I2C 7E50 / 7E78 | `30.100.2527.40`, `oem107.inf`, signed |
| Lenovo PM Device | `1.69.168.0`, `oem151.inf`, signed |
| Lenovo Power and Battery | `10.2.5.3`, `oem283.inf`, signed |
| CIM keyboard health | Four devices: `Status=OK`, `ConfigManagerErrorCode=0` |
| WPR | System32 executable present; reported version `10.0.26100`; idle |
| WPA | Not found on PATH; installation elsewhere not exhaustively searched |

PnPUtil labels the installed keyboard package Best Ranked / Installed. It also
lists `input.inf` `10.0.26100.9444` as an outranked match for the USB interfaces.
A numerically higher version in a different INF is not an instruction to force
that package onto the keyboard. The keyboard package's displayed June 2006 date
alone does not establish that the active Windows component is obsolete.

`Get-PnpDevice` was unavailable in this PowerShell session; CIM and the installed
PnPUtil worked, so no module installation was needed. PnPUtil enumeration and
stack/driver flags were checked against Microsoft's documentation through
Context7 and the official reference. [PnPUtil command
syntax](https://learn.microsoft.com/en-us/windows-hardware/drivers/devtest/pnputil-command-syntax)

## Ranked candidates and discriminating evidence

These are investigation priorities, not diagnosed causes or probabilities.

| Priority | Candidate | Best distinguishing evidence |
| --- | --- | --- |
| 1 | Native keyboard matrix, connection, EC/firmware path | Same physical key failures in Lenovo UEFI keyboard testing; external USB keyboard still reliable during the Windows failure. A UEFI pass between episodes cannot eliminate an intermittent fault. |
| 2 | Windows input processing, application focus, user-mode hook/remapper, accessibility setting | Both keyboard types fail in the same applications while firmware testing passes; compare ordinary typing in a local plain-text app with the affected app. Root's software lane examines installed hooks and settings. |
| 3 | Resume/power-transition or firmware interaction | Reproducible onset after sleep, lid/AC transitions, or docking, with accurate timestamps and recovery conditions. BIOS/EC review is warranted only against a matching Lenovo change log. |
| 4 | System contention, ISR/DPC delay, GPU/UI stall | Simultaneous delayed input and display/application responsiveness; correlate a controlled repro with WPR CPU scheduling, interrupt, and DPC evidence. A CPU spike alone does not identify a keyboard driver defect. |
| 5 | Serial IO/I2C or USB controller issue | Elevate only if the failing device's bus or concurrent touchpad/USB events support it. The observed native PS/2 path does not traverse the enumerated I2C driver. |

No exact P1 Gen 7 keyboard firmware fix or universally applicable driver update
was confirmed in the bounded official-source search. No current native-keyboard
issue appeared on Microsoft's retrieved 25H2 known-issues page. That absence is
not proof that this build cannot have an input bug. The published historical USB
keyboard issue concerns WinRE after the October 14, 2025 update, explicitly says
normal Windows input continued working, and was resolved October 20, 2025; it
does not match intermittent PS/2 typing in the running desktop. [Current 25H2
issues](https://learn.microsoft.com/en-us/windows/release-health/status-windows-11-25h2),
[resolved 25H2 issues](https://learn.microsoft.com/en-us/windows/release-health/resolved-issues-windows-11-25h2)

## Model-applicable firmware and driver review

Lenovo's BIOS package explicitly covers P1 Gen 7 types **21KV and 21KW**.
An initial US search result exposed older **1.21 / N48ET34W** metadata. An exact
BIOS-ID search then found **N48UJ21W**, whose readme was retrieved with HTTP 200.
It lists **1.22 / N48ET35W**, **EC 1.15 / N48HT28W**, issue date **2026-08-11**,
and the exact model family. Both firmware version numbers match this machine.
The release lists security/enhancement work and no problem fixes for 1.22;
no intermittent typing repair was identified. Package issue date and SMBIOS
firmware build date are distinct metadata. No newer applicable BIOS was verified.
The stale 1.21 result is not a reason to downgrade. Lenovo's main pages sometimes
return navigation shells, so the direct readme provides better evidence here.
[Exact-model BIOS package](https://support.lenovo.com/eg/en/downloads/ds569195),
[matching 1.22 package readme](https://download.lenovo.com/pccbbs/mobiles/n48uj21w.html)

For a future update review, use Lenovo Vantage/System Update or the manually
selected 21KV product support page and compare the offered package's supported
model, OS, version, and change log with the table above. Review Windows Update
history and optional driver offers separately. Lenovo documents Windows Update
before System Update, and recommends handling BIOS separately from other
updates. This lane did not launch Lenovo's updater, accept an offer, or establish that
any installed Intel/Lenovo driver is behind its model-specific supported release.
[Lenovo System Update guidance](https://support.lenovo.com/ai/en/solutions/HT003029),
[Lenovo Vantage update workflow](https://support.lenovo.com/aw/en/videos/VID500028)

A bounded Windows Update Agent COM query completed successfully (`ResultCode=2`)
and returned **zero** currently offered, uninstalled, nonhidden driver updates
for `IsInstalled=0 and Type='Driver' and IsHidden=0`. It used the machine's
configured update service and a 45-second process-job deadline. No download or
installation API was invoked. This is the WUA offer view, not proof that every
Lenovo catalog package is current; hidden, policy-excluded, or separately
published vendor packages are outside this result. The query can refresh update
scan metadata while leaving driver/device state unchanged. [WUA search
API](https://learn.microsoft.com/en-us/windows/win32/api/wuapi/nf-wuapi-iupdatesearcher-search),
[result-code definition](https://learn.microsoft.com/en-us/windows/win32/api/wuapi/ne-wuapi-operationresultcode)

## Testing options and limits

**Most useful hardware split: Lenovo UEFI.** At a user-controlled maintenance
opportunity, use the built-in keyboard to enter BIOS with F1 and verify navigation
and Caps Lock response. Lenovo's diagnostic instructions describe F10 on
ThinkPad systems, then the interactive keyboard component test; verify each key,
modifiers, repeated presses, and Fn combinations. If diagnostics is not present,
Lenovo describes a matching-architecture bootable diagnostic USB. No reboot or
USB creation was performed here. [ThinkPad keyboard
troubleshooting](https://support.lenovo.com/ls/en/solutions/ht080160),
[Lenovo UEFI keyboard diagnostic
instructions](https://support.lenovo.com/ee/en/solutions/YTV104507)

**WPR/WPA for a reproducible Windows-only stall.** The installed GeneralProfile
was inspected and includes `DPC`, `Interrupt`, `CSwitch`, `ReadyThread`, and
sampled CPU profiling. WPR is currently idle. A brief, planned capture can help
correlate a failure with scheduling or driver execution. It cannot alone prove
that a physical switch failed to generate a scan code. Available ETW providers
include `Microsoft-Windows-Input-HIDCLASS`, `Microsoft-Windows-SPB-HIDI2C`,
`Microsoft-Windows-KeyboardFilter`, and `Microsoft-Windows-Win32k`; provider
registration does not prove a provider exposes suitable per-key events. Do not
assume HID/I2C traces cover the PS/2 path. A custom input profile requires checking
its actual event schema and privacy footprint first. [WPR
overview](https://learn.microsoft.com/en-us/windows-hardware/test/wpt/introduction-to-wpr),
[WPR command-line options](https://learn.microsoft.com/en-us/windows-hardware/test/wpt/wpr-command-line-options)

**HLK is a lab escalation, not a quick field keyboard test.** Microsoft's
keyboard prerequisites require a configured test computer and suitable keyboard;
USB testing can require a specified hub. Its Device.Input troubleshooting page
says keyboard coverage comes through Device.Fundamentals rather than dedicated
keyboard tests in that category. The separately indexed “Keyboard works in UEFI”
test is explicitly for 8998 MTP / ARM64 hardware, so it is not a P1 Gen 7 test
recipe. No HLK deployment or stress test is warranted just to verify this laptop's
keys. [Keyboard HLK prerequisites](https://learn.microsoft.com/en-us/windows-hardware/test/hlk/testref/keyboard-testing-prerequisites),
[Device.Input testing scope](https://learn.microsoft.com/en-us/windows-hardware/test/hlk/testref/troubleshooting-deviceinput-testing),
[restricted UEFI test applicability](https://learn.microsoft.com/en-us/windows-hardware/test/hlk/testref/3c23ef6b-828d-4b19-af02-5b7ad95a3519)

## Repeatable read-only checks

Run these during a good interval and a failure interval. They inspect inventory
and recorder configuration; they do not initiate driver changes or recording.

```powershell
Get-CimInstance Win32_ComputerSystem |
    Select-Object Manufacturer, Model, SystemFamily
Get-CimInstance Win32_BIOS |
    Select-Object SMBIOSBIOSVersion, ReleaseDate,
        EmbeddedControllerMajorVersion, EmbeddedControllerMinorVersion
Get-ItemProperty 'HKLM:/SOFTWARE/Microsoft/Windows NT/CurrentVersion' |
    Select-Object DisplayVersion, CurrentBuild, UBR
Get-CimInstance Win32_Keyboard |
    Select-Object Name, Status, ConfigManagerErrorCode
Get-CimInstance Win32_PnPSignedDriver |
    Where-Object DeviceClass -eq 'KEYBOARD' |
    Select-Object DeviceName, DriverProviderName, DriverVersion, InfName, IsSigned

# Keep instance identifiers and GUIDs out of displayed/shared evidence.
pnputil /enum-devices /class Keyboard /connected /stack /drivers |
    Where-Object { $_ -notmatch 'Instance ID:|Class GUID:|Driver Node Strong Name:|Matching Device ID:' } |
    ForEach-Object {
        $_ -replace '\{?[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\}?', '[guid omitted]'
    }
wpr -status
wpr -profiles
wpr -profiledetails GeneralProfile
```

The display filter matches this machine's English PnPUtil labels; do not treat it
as a general-purpose sanitizer for other locales or arbitrary diagnostic output.

## Remaining evidence needed

- Exact symptom: missing presses, duplicates, swapped characters, delayed bursts,
  stuck modifiers, or total loss; affected keys and duration.
- Whether external typing, on-screen input, pointer motion, and application
  rendering continue during the same episode; local console versus remote use.
- Timestamped relation to sleep/resume, docking, AC changes, workload, and any
  recently installed Windows/Lenovo updates.
- Interactive UEFI result during or near a failure; actual user participation is
  required and has not occurred in this research lane.
- Any newer 21KV-applicable vendor offers beyond the verified matching 1.22
  readme; WUA currently offers no uninstalled/nonhidden driver packages, but this
  does not establish the state of Lenovo's separate catalog.
- A reproducible short trace and analysis of the failing interval before
  attributing the behavior to a particular driver or proposing a driver change.
