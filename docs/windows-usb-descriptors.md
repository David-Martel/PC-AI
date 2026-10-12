# Windows USB descriptor inventory

Run the profile-independent query tool with PowerShell 7:

```powershell
pwsh -NoLogo -NoProfile -File .\Tools\Get-PcaiUsbDescriptorInventory.ps1 -AsJson
pwsh -NoLogo -NoProfile -File .\Tools\Get-PcaiUsbDescriptorInventory.ps1 -DryRun -OutputPath C:\Temp\usb.json
```

An optional absolute `-OutputPath` writes JSON; existing files require `-Force`.
All reserved Windows path components, alternate data streams and device namespace
paths are rejected. `-h` and `--help` display help. Dry run loads no native code,
queries no hardware and writes no files.

The C# helper enumerates present hub interfaces and uses only node, connection,
configuration-descriptor and EX_V2 query IOCTLs. It does not reset ports, replace
drivers, set wake policy or send vendor commands. Hub indices are ordinal within
a snapshot; raw hub/device paths and serial strings are excluded. The helper
opens hub handles with the access required by these query IOCTLs; that access
does not imply a device setting write.

`DeclaredMaxBusCurrentMa` is configuration `bMaxPower` multiplied by 8 mA for
devices operating at SuperSpeed or higher, or 2 mA otherwise. It is populated
only when EX_V2 establishes speed and descriptor configuration index0 matches
the active configuration. A mismatch or query failure leaves the active budget
null. Descriptor self-power and remote-wake bits describe capability; they do
not establish actual supply current, enabled wake policy or successful wake.

A descriptor maximum is neither measured current nor a negotiated USB-C PD
contract. It cannot validate a PC's barrel-adapter capacity or infer the current
draw of an unenumerated peripheral. `measuredBusCurrentMa` and `pcInputWatts`
remain null. Use a separately qualified meter/PD analyzer for those measurements.

Per-hub/port failures are retained as bounded error codes and produce `Partial`
status. Native initialization/enumeration failures stop the command before
creating output. Synchronous driver queries have no guaranteed wall-clock timeout;
the tool is passive discovery, not a stress or wake test. Configuration descriptors
are capped at 16 KiB. Synthetic tests exercise decoding, unknown speed/configuration,
malformed chains and output safety without enumerating live hardware.

The private workstation probe predates this promoted implementation. Its earlier
device observations do not constitute a fresh run, deployment or acceptance of
this source. Review and exact-source validation remain separate from live discovery.
