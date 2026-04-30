# PC_AI Context: Driver Update Framework Complete
**Date**: 2026-03-11
**Branch**: main @ 5dc05d8
**Context ID**: ctx-pcai-20260311-driver-framework

## Summary

Built a production-quality driver update framework for Windows PC peripherals (USB hubs, ethernet adapters, Thunderbolt docks). Extended Rust PnP enumeration with driver metadata (version/date/provider via SetupDi registry API), built C# P/Invoke wrapper, created PowerShell `PC-AI.Drivers` module with full scan/compare/install pipeline, and successfully updated Realtek RTL8156/RTL8157 drivers via INF-based pnputil installation (bypassing broken setup.exe).

## Key Deliverables

### New Module: PC-AI.Drivers
- `Modules/PC-AI.Drivers/` — Full PowerShell module with manifest
- Public: `Get-DriverReport`, `Compare-DriverVersion`, `Install-DriverUpdate`, `Get-DriverRegistry`, `Get-PnpDeviceInventory`, `Update-DriverRegistry`
- Private: `Resolve-HardwareId`, `Invoke-TrustedDownload`, `Test-AdminElevation`

### Config: driver-registry.json
- `Config/driver-registry.json` — Central device-to-driver mapping
- 8 devices: Realtek RTL8156/8157, CalDigit Element Hub, Cable Matters USB4, ACASIS TB3, Intel BT/WiFi, Realtek Audio
- Match rules: `vid_pid`, `friendly_name`, `pci_class`
- `versionComparable`, `sharedDriverGroup`, `installerType` (inf, manual, zip-with-exe, windows-update, none)

### Tools
- `Tools/Update-Drivers.ps1` — Main orchestrator (scan + install workflow)
- `Tools/Update-UsbDrivers.ps1` — Standalone USB driver updater with CalDigit firmware support
- `Tools/Install-InfDriver.ps1` — **NEW** Reusable pnputil-based INF installer that extracts SFX/7z archives and installs drivers via `pnputil /add-driver /install`

### Native Extensions
- `Native/pcai_core/pcai_core_lib/src/telemetry/pnp.rs` — Added `driver_version`, `driver_date`, `driver_provider` fields via SetupDi registry API (`get_driver_metadata()`, `read_reg_sz()`)
- `Native/PcaiNative/HardwareModule.cs` — Updated P/Invoke wrapper with driver field XML docs

## Architecture Decisions

| Decision | Rationale |
|----------|-----------|
| `pnputil /add-driver /install` over setup.exe | Realtek's WinZip SFX setup.exe hangs; pnputil installs INFs directly and reliably |
| `versionComparable: false` + `ManualCheck` status | Firmware versions (CalDigit) and package-vs-INF version mismatches can't be compared meaningfully |
| `sharedDriverGroup` for deduplication | Realtek RTL8156/8157 share a unified driver package — prevents duplicate downloads/installs |
| Subdomain matching in `Invoke-TrustedDownload` | `downloads.caldigit.com` must match trusted source `caldigit.com` |
| ShouldProcess suppression during scan phase | CIM cmdlets (`Get-PnpDeviceProperty`) inherit `-WhatIf` from caller, causing false NoDriver results |
| INF driver version format: `1156.21.20.1110` | Different from package version `11.21.1110.2025`; registry now uses INF-level versions for accurate comparison |

## Driver Status (as of 2026-03-11)

| Device | Installed Version | Target | Status |
|--------|------------------|--------|--------|
| Realtek RTL8156 (2.5GbE) | 1156.21.20.1110 | 1156.21.20.1110 | **Current** |
| Realtek RTL8157 (5GbE) | 1157.21.20.1110 | 1157.21.20.1110 | **Current** |
| CalDigit Element Hub | FW (hub driver) | FW v45.1 | ManualCheck |
| Intel Wireless Bluetooth | — | Windows Update | NoUpdate |
| Cable Matters USB4 107064 | — | Inbox driver | NoUpdate |

## 5-Agent Code Review Results (Completed)

Agents: powershell-pro, csharp-pro, rust-pro, debugger, architect-reviewer

### Critical/High Fixes Applied (13 total)
1. CalDigit `installerType` corrected to `zip-with-exe`, download URL fixed
2. Realtek download URL set to null with `installerType: manual` (then later `inf`)
3. `versionComparable: false` for firmware/non-comparable versions
4. `ManualCheck` status implemented throughout pipeline
5. ShouldProcess moved before download in `Install-DriverUpdate.ps1`
6. `-WhatIf` suppression around scan phase in `Update-Drivers.ps1`
7. Exit code 3010 (reboot required) handled as success
8. `$pid` renamed to `$devPid` (shadowed automatic `$PID`)
9. `$results = @()` replaced with `List[PSCustomObject]`
10. Subdomain matching for trusted host validation
11. Shared-group skip check reordered before manual handler
12. Duplicate `-IncludeUnknown` parameter removed
13. Dead `$report` call removed (was doubling scan time)

### Deferred Items (Medium/Low Priority)
- **Rust**: RAII wrapper for HDEVINFO, dynamic buffer for SPDRP_HARDWAREID, data_type validation in read_reg_sz, catch_unwind on FFI boundary
- **Architecture**: Refactor Get-DriverReport to use Get-PnpDeviceInventory (DRY), implement pci_class matching, move Write-Host out of library functions
- **PowerShell**: Add Pester tests, add PowerShellVersion to manifest, fix CIM fallback path
- **C#**: PcaiCore._version pointer leak, GetStatusDescription pointer leak, MarshalAs on bool return

## Realtek Driver Package Structure
```
Install_USB_Win11_11021_20_11102025_01302026.exe  (WinZip SFX, 5.3MB)
  └─ extracted/
     ├── Setup.exe          (InnoSetup-based — HANGS, do not use)
     ├── Silent_Install.bat (calls Setup /verysilent)
     ├── WIN11/cx/64/       (x64 INF+SYS+CAT drivers)
     │   ├── rtu56cx22x64sta.INF  (RTL8156 — DriverVer 1156.21.20.1110)
     │   ├── rtu56cx22x64.sys
     │   ├── rtu56cx22x64.cat
     │   ├── rtu57cx22x64sta.INF  (RTL8157 — DriverVer 1157.21.20.1110)
     │   ├── rtu57cx22x64.sys
     │   ├── rtu57cx22x64.cat
     │   └── ... (rtu52/53/55/59 for other chipsets)
     ├── WIN11/cx/arm64/    (ARM64 variants)
     └── TOOL/HW_ENUM.txt   (VID/PID→device name mapping)
```

## Files Changed (Uncommitted)

### New Files (untracked)
- `Config/driver-registry.json`
- `Modules/PC-AI.Drivers/` (full module — 11 files)
- `Tools/Update-Drivers.ps1`
- `Tools/Update-UsbDrivers.ps1`
- `Tools/Install-InfDriver.ps1`

### Modified Files
- `Native/pcai_core/pcai_core_lib/src/telemetry/pnp.rs` (driver metadata)
- `Native/pcai_core/pcai_core_lib/Cargo.toml` (Win32_System_Registry feature)
- `Native/PcaiNative/HardwareModule.cs` (P/Invoke + XML docs)

## Agent Work Registry

| Agent | Task | Files Touched | Status |
|-------|------|---------------|--------|
| rust-pro | PnP driver metadata (SetupDi registry API) | pnp.rs, Cargo.toml | Complete |
| csharp-pro | C# P/Invoke wrapper review | HardwareModule.cs | Complete |
| powershell-pro | PS module review + fixes | PC-AI.Drivers/*.ps1, Update-Drivers.ps1 | Complete |
| debugger | Data/logic bug detection | driver-registry.json, Compare-DriverVersion.ps1 | Complete |
| architect-reviewer | Architecture + DRY review | Get-DriverReport.ps1, Install-DriverUpdate.ps1 | Complete |

## Recommended Next Agents

1. **test-automator**: Add Pester tests for PC-AI.Drivers module (currently 0% coverage)
2. **code-reviewer**: Review Install-InfDriver.ps1 (newly created, not yet reviewed)
3. **powershell-pro**: Implement `pci_class` match rule type (currently stub)
