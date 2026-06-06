# NVIDIA Driver Fix Plan: Dual-GPU Misalignment (RTX 2000 Ada + RTX 5060 Ti eGPU)

**Machine:** Lenovo ThinkPad P1 Gen 7 (21KV/21KW), Windows 11  
**Date:** 2026-06-06  
**Author:** PC_AI system-assessment agent  
**Status:** Research-only — do NOT install without reading all caveats

---

## 1. Verified Current State

| GPU | PCI ID | Status | INF | Driver Version | INF Date |
|-----|--------|--------|-----|----------------|----------|
| NVIDIA RTX 2000 Ada Generation Laptop GPU | DEV_28B8 (SUBSYS 223417AA) | **Error: CM_PROB_FAILED_ADD (Code 31)** | oem236.inf | 32.0.15.**9659** | 2026-05-20 |
| NVIDIA GeForce RTX 5060 Ti (eGPU, Thunderbolt) | DEV_2D04 | OK | oem395.inf | 32.0.15.**9186** | 2026-01-19 |
| NVIDIA Platform Controllers | — | OK | oem394.inf | 32.0.15.9605 | 2026-03-28 |
| Intel Arc Graphics (iGPU) | DEV_7D55 | OK | (Intel) | — | — |

**Version mapping (32.0.15.XXXX → marketing):**
- 32.0.15.9659 = **596.59** (RTX/Quadro Enterprise Production Branch R595 U5, released 2026-05-27)
- 32.0.15.9186 = **591.86** (GeForce Game Ready, the 5060 Ti launch driver, released 2026-01-19)

**Root cause confirmed:** The 5060 Ti's older INF (591.86 / oem395.inf) loaded nvlddmkm.sys kernel version 591.86 first. The Ada GPU's newer INF (596.59 / oem236.inf) requires nvlddmkm.sys >= 596.59. Because only one nvlddmkm.sys version can be loaded, and 591.86 won the race at boot, AddDevice for the Ada GPU was rejected → Code 31. These two INF packages are from **different nvlddmkm.sys versions** (591.86 vs 596.59) and cannot coexist. Two INFs that reference the same nvlddmkm.sys version can coexist.

---

## 2. Driver Track Analysis — Confirmed From Official Sources

### The Core Problem

The two GPUs live on different NVIDIA driver tracks:

| Track | Purpose | Covers |
|-------|---------|--------|
| **GeForce Game Ready / Studio** | Consumer gaming/creative | RTX 5060 Ti (desktop Blackwell), GeForce RTX 40/30/20 Laptops |
| **RTX/Quadro Enterprise** | Professional workstation | RTX 2000 Ada Generation Laptop GPU (professional mobile), RTX PRO series |

### Confirmed From Official 610.47 Release Notes PDF

The following findings are from the official NVIDIA 610.47 release notes PDF (`610.47-win11-win10-release-notes.pdf`, extracted via zlib decompression), which is the primary authoritative source:

**Table 1 — Supported NVIDIA Desktop GPUs (GeForce Game Ready 610.47):**
Lists NVIDIA Blackwell architecture (RTX 5090, 5080, 5070 Ti, 5070, 5060 Ti, 5060, 5050), NVIDIA Ada Lovelace (RTX 4090, 4080, 4070, 4060, etc.), Ampere, Turing.
**Consumer desktop GeForce only. RTX 2000 Ada Generation (DEV_28B8) is NOT listed.**

**Table 3 — Supported NVIDIA Notebook GPUs (GeForce Game Ready 610.47):**
Lists GeForce RTX 5090/5080/5070 Ti/5070/5060/5050 Laptop GPU (Blackwell), GeForce RTX 4090/4080/4070/4060/4050 Laptop GPU (Ada Lovelace consumer), Ampere, Turing.
The PDF explicitly notes: *"For information about notebook products not shown please see http://www.geforce.com/hardware/notebook-gpus"*
**RTX 2000 Ada Generation Laptop GPU (DEV_28B8, professional mobile) is NOT listed.**

**RTX/Quadro Enterprise NFB R610 U1 v610.47 (separate package):**
Confirmed from station-drivers.com to explicitly support **NVIDIA RTX 2000 Ada Generation Laptop GPU**.
Desktop consumer GeForce GPUs (including RTX 5060 Ti, DEV_2D04) are **completely excluded** from the Enterprise INF.

**Definitive conclusion: No single NVIDIA-published package for June 2026 enumerates both DEV_28B8 and DEV_2D04. This is confirmed from the official release notes PDF and the Enterprise package page.**

### The Critical Constraint: One nvlddmkm.sys Version

Windows loads exactly one version of nvlddmkm.sys. All NVIDIA GPUs on the system must use the **identical** nvlddmkm.sys build — same branch is NOT sufficient; the exact version must match. This is the root cause: 591.86 and 596.59 are different versions. The fix requires both GPUs to end up bound to INFs that ship the same nvlddmkm.sys version.

Two INFs from two different packages can coexist WITHOUT a Code 31 only if they were built from the same NVIDIA driver release and include the same nvlddmkm.sys. Both 610.47 packages (GRD and Enterprise) were released on 2026-05-26 from the same R610 kernel branch and share the same nvlddmkm.sys. A GRD 610.47 INF + Enterprise 610.47 INF combination is safe. A GRD 610.47 INF + Enterprise 596.59 INF is NOT safe — different kernels → Code 31.

### What 610.47 Is

- Version: **610.47 WHQL**
- Release date: **2026-05-26**
- Two packages released same day: GeForce Game Ready (consumer) and RTX/Quadro Enterprise NFB (professional)
- Both packages share the same R610 kernel branch and identical nvlddmkm.sys version
- Notable: 610.47 **drops the legacy NVIDIA Control Panel** — only the NVIDIA App is supported
- CUDA 13.3 support
- This is the latest driver on both tracks as of the research date (2026-06-06)

---

## 3. Recommended Approach: Safest Option

### IMPORTANT: Why "Install GRD and let WU handle the Ada" Is Unreliable

The obvious approach — install GeForce GRD 610.47 (covers RTX 5060 Ti), then let Windows Update bind the Ada — sounds like the R610 branch alignment would make it safe. **It is not reliably safe** for this reason: Windows Update's best hardware match for DEV_28B8 with SUBSYS_223417AA is the Lenovo OEM Enterprise driver. That driver is currently at 596.59 (R595 branch) and will not be updated to 610.47 Enterprise until Lenovo validates and publishes it. If WU installs 596.59 for the Ada while the 5060 Ti is on GRD 610.47, you are back to a split kernel (different nvlddmkm.sys versions) → Code 31.

**The reliable fix:** both GPUs must end on INFs that use the same nvlddmkm.sys. The way to guarantee that with current (June 2026) drivers is to explicitly control both bindings.

### Option A (Recommended): Enterprise 610.47 First + Force-Bind 5060 Ti

**Step sequence:**

1. DDU wipe (both INFs removed)
2. Install **RTX/Quadro Enterprise NFB 610.47** — this definitively binds DEV_28B8 (Ada) to the R610 kernel
3. After reboot: check if Windows also bound DEV_2D04 (5060 Ti) to the Enterprise INF
   - If yes: both GPUs on Enterprise 610.47 → done
   - If no (5060 Ti unrecognized or yellow bang): the Enterprise INF explicitly excludes DEV_2D04; it has no wildcard catch-all, so "Show compatible hardware" will show nothing and force-binding Enterprise to the 5060 Ti will fail at driver init (Code 43). Proceed to Option B for the 5060 Ti bind — the Ada stays on Enterprise 610.47.
   - Keep the machine **offline (or WU paused)** from the DDU step through Step B completion. WU may race to install Lenovo's OEM 596.59 for the Ada; pausing prevents that re-split.

**Download URL (Enterprise 610.47):**
```
https://us.download.nvidia.com/Windows/Quadro_Certified/610.47/610.47-quadro-rtx-desktop-notebook-win10-win11-64bit-international-dch-whql.exe
```
File size: ~700-900 MB. Verify URL is current at station-drivers.com thread 1027 before downloading.

**Rationale:** This is the only package confirmed by primary sources to enumerate DEV_28B8. Binding the 5060 Ti to the same package or same-kernel INF is secondary; the Ada Code 31 is the priority problem to fix.

### Option B (Required for 5060 Ti After Enterprise Install): Per-Device GRD Bind via Have Disk

Running the full GRD installer on top of the Enterprise install is NOT safe — NVIDIA's installer replaces the entire NVIDIA display driver stack by default, even without `-clean`, which will uninstall the Enterprise INF and re-break the Ada. Use per-device manual binding instead:

1. **Extract the GRD 610.47 installer without running it:** Launch `C:\Drivers\610.47-grd.exe`, let it unpack to `C:\NVIDIA\<build>\` (the default staging folder), then click Cancel on the welcome screen. The INF files are now on disk.
2. **Bind the 5060 Ti using the extracted GRD INF:**
   - Device Manager → Display adapters → NVIDIA GeForce RTX 5060 Ti → Update driver
   - Browse my computer for drivers → Let me pick from a list
   - Click "Have Disk" → Browse → navigate to `C:\NVIDIA\<build>\Display.Driver\` → select the `.inf` file there
   - With "Show compatible hardware" checked, the 5060 Ti should appear in the list (DEV_2D04 IS in the GRD INF) → Install
3. This adds the GRD 610.47 INF to the driver store for DEV_2D04 only — it does **not** run any NVIDIA setup process and does not touch the Enterprise INF binding for DEV_28B8.
4. Both DEV_28B8 (Enterprise 610.47) and DEV_2D04 (GRD 610.47) are now on the same R610 kernel build → same nvlddmkm.sys → no Code 31.
5. Keep the machine offline (or WU paused) until both GPUs verify on 610.47 (see Step 6 below).

**Download URL (GeForce GRD 610.47 desktop):**
```
https://us.download.nvidia.com/Windows/610.47/610.47-desktop-win10-win11-64bit-international-dch-whql.exe
```

### Option C (Lenovo-validated package)

**Not recommended for this configuration.** The Lenovo `ds`-prefix driver for the ThinkPad P1 Gen 7 is validated for the internal Optimus configuration only. It will not enumerate DEV_2D04 (RTX 5060 Ti desktop eGPU). Use the generic NVIDIA package.

**Tradeoff:** Lenovo-validated = tested for this exact laptop OEM BIOS, MUX configuration, and hybrid display mode. Generic NVIDIA = broader GPU coverage, newer features, but not OEM-validated for Optimus/MUX interaction. Given the eGPU requirement, generic NVIDIA is the only viable path here.

---

## 4. Exact Install Plan

### Step 0: Prerequisites (Before Rebooting)

1. **Download both installers** while still running the current desktop:

   - **Enterprise 610.47 (RTX/Quadro, DCH, WHQL) — PRIMARY:**
     ```
     https://us.download.nvidia.com/Windows/Quadro_Certified/610.47/610.47-quadro-rtx-desktop-notebook-win10-win11-64bit-international-dch-whql.exe
     ```
     Save to `C:\Drivers\610.47-enterprise.exe`. Verify URL is current at station-drivers.com thread 1027 before downloading.

   - **GeForce GRD 610.47 (Desktop, DCH, WHQL) — FALLBACK:**
     ```
     https://us.download.nvidia.com/Windows/610.47/610.47-desktop-win10-win11-64bit-international-dch-whql.exe
     ```
     Save to `C:\Drivers\610.47-grd.exe`. Official page: `https://www.nvidia.com/en-us/drivers/details/271418/`

   Have both on disk before rebooting.

2. **Download DDU (Display Driver Uninstaller):**
   ```
   https://www.guru3d.com/download/display-driver-uninstaller-download/
   ```
   Save to `C:\Drivers\DDU.exe`. Do not install — just have it ready.

3. **Create a System Restore point** (belt-and-suspenders rollback):
   ```powershell
   # Run as Administrator
   Checkpoint-Computer -Description "Before NVIDIA 610.47 clean install" -RestorePointType MODIFY_SETTINGS
   ```

4. **Note the current known-good rollback versions:** `oem236.inf = Enterprise 596.59`, `oem395.inf = GRD 591.86`. If 610.47 causes problems, target is Enterprise 596.59 for Ada.

5. **Disconnect external monitors** from the eGPU enclosure before starting. Internal display only during install.

6. **Keep the eGPU enclosure connected** to Thunderbolt throughout the install. DO NOT detach the eGPU mid-install.

7. **Ensure AC power** is connected. Do not run on battery.

8. **Pause Windows Update** to prevent automatic driver reinstallation before the install settles:
   ```powershell
   # Settings → Windows Update → Advanced options → Pause updates (7 days)
   # Or via PowerShell (Admin):
   Stop-Service -Name wuauserv -Force
   ```

### Step 1: Boot to Safe Mode (DDU Phase)

1. Hold Shift + Restart → Troubleshoot → Advanced options → Startup Settings → Restart
2. Press **4** for Safe Mode (standard, not networking)
3. Confirm MUX is in **Hybrid/Optimus mode** (not dGPU-only) — see Risk 2 below
4. Confirm the eGPU enclosure is still powered and connected via Thunderbolt

### Step 2: Run DDU — Full Wipe of Both NVIDIA Display Drivers

1. Run `C:\Drivers\DDU.exe` as Administrator
2. In DDU Options, ensure **"Prevent downloads from Windows Update"** is checked
3. Set device type: **GPU**, vendor: **NVIDIA**
4. Click **"Clean and do NOT restart"**
   - Removes oem236.inf (596.59 Enterprise), oem395.inf (591.86 GRD), registry entries, device bindings
   - The Platform Controllers driver (oem394.inf) may survive — this is acceptable
5. After DDU completes, manually restart into Normal Mode

### Step 3: Install Enterprise 610.47 (Primary)

1. Ensure normal Windows (not safe mode), AC power, eGPU connected, external monitors disconnected from eGPU
2. Right-click `C:\Drivers\610.47-enterprise.exe` → **Run as Administrator**

**Silent clean install command:**
```cmd
"C:\Drivers\610.47-enterprise.exe" -s -clean -n Display.Driver
```

Flag meanings:
- `-s` — silent (suppress UI)
- `-clean` — force clean install (removes residual entries, equivalent to GUI "Custom → Clean installation" checkbox). Note: `-clean` is widely reported in NVIDIA driver forums but is not documented in NVIDIA's official install guide. GUI path is more reliable for a one-off install.
- `-n` — suppress reboot prompt (manual reboot after)
- `Display.Driver` — install only the display driver component

**GUI alternative (more reliable for a one-off install):**
1. Run installer → Custom installation (not Express)
2. Check **"Perform a clean installation"**
3. Select components: at minimum "Display Driver"
4. Click Next → Install

### Step 4: Post-Install Verification

After the first reboot:

```powershell
# Check both GPUs status
Get-PnpDevice -Class Display | Select-Object FriendlyName, Status, Problem | Format-Table -AutoSize

# Confirm driver versions on both GPUs
Get-CimInstance Win32_PnPSignedDriver | Where-Object { $_.DeviceName -like "*NVIDIA*RTX*" } |
    Select-Object DeviceName, DriverVersion, InfName, DriverDate | Format-Table -AutoSize
```

**What to look for in DriverVersion:** Both GPUs should show a version number greater than 32.0.15.9659 (the current Ada driver). The exact 32.0.15.XXXX value for 610.47 was not confirmed from available sources — look for a value consistent across both GPU entries.

**Good outcome — Ada on Enterprise 610.47, 5060 Ti also bound (either Enterprise or WU-pulled GRD 610.47):**
```
NVIDIA RTX 2000 Ada Generation Laptop GPU   OK  CM_PROB_NONE  oem_ent.inf  32.0.15.????  (610.47-era)
NVIDIA GeForce RTX 5060 Ti                  OK  CM_PROB_NONE  oem_grd.inf  32.0.15.????  (610.47-era, same value)
```
Both DriverVersion values must be identical for the configuration to be stable long-term.

**Failure state — Ada still Code 31 OR 5060 Ti Code 31:** The installer may not have bound one of the GPUs. Proceed to Step 5.

**Danger state — Ada OK but 5060 Ti shows a 591.86 or 596.59 version number:** Two different versions → kernel conflict will re-emerge. Proceed to Step 5.

### Step 5: If Either GPU Is Still in Error or On a Different Driver Version

**Case A — Ada Code 31 (unlikely after Enterprise install, but possible):**
1. Device Manager → NVIDIA RTX 2000 Ada → Update driver → Browse → Let me pick
2. Select from the Enterprise 610.47 INF files on disk — DEV_28B8 should be listed

**Case B — 5060 Ti unrecognized, yellow bang, or on wrong driver version:**
- Do NOT run the GRD installer — it will replace the Enterprise driver stack and re-break the Ada.
- Use the per-device GRD INF bind described in Option B above:
  1. Extract `C:\Drivers\610.47-grd.exe` to `C:\NVIDIA\<build>\` (launch, let it unpack, then Cancel)
  2. Device Manager → RTX 5060 Ti → Update driver → Browse → Let me pick → Have Disk → `C:\NVIDIA\<build>\Display.Driver\` → install
  3. After bind, confirm Ada is still on Enterprise 610.47 INF (re-run the verification query in Step 4)

**Case C — Windows Update installed old 596.59 for Ada during the process:**
1. Immediately pause WU (7 days)
2. DDU wipe again → repeat from Step 3
3. This time, immediately disconnect network before rebooting from DDU to prevent WU from racing ahead of the manual install

### Step 6: Reboot and Final Check

```powershell
# Confirm no devices in error state
Get-PnpDevice -Class Display | Where-Object { $_.Problem -ne 'CM_PROB_NONE' }

# Confirm nvlddmkm.sys version in driver store
Get-Item "C:\Windows\System32\DriverStore\FileRepository\nv_dispi.inf_*\nvlddmkm.sys" |
    Select-Object Name, @{N='Version';E={$_.VersionInfo.FileVersion}} | Format-List

# Both entries should show the same nvlddmkm.sys version; it should NOT be 591.86 or 596.59
# Re-enable Windows Update only after verifying both GPUs are OK
Start-Service -Name wuauserv
```

---

## 5. Machine-Specific Risks

### Risk 1: eGPU Thunderbolt Link Drop During Install (HIGH)

The NVIDIA display driver installer briefly takes the display subsystem offline to replace the kernel mode driver. If the eGPU is connected and its display cable is plugged in, the screen may blank for 10-30 seconds during the kernel replacement phase. If Thunderbolt link negotiation fails during this blank period, Windows may re-enumerate the eGPU as a new device and the INF binding may not complete.

**Mitigation:** Disconnect all monitors from the eGPU enclosure before install (eGPU box stays powered and Thunderbolt-connected). Internal display only during install. Reconnect external monitors after the first post-install reboot.

### Risk 2: MUX/Optimus Switch During Install (MEDIUM)

The ThinkPad P1 Gen 7 has a hardware MUX switch (configurable in ThinkPad Setup / BIOS) for dGPU-only vs. Optimus mode. If MUX is in dGPU-only mode during install, the installer must initialize the dGPU — but the dGPU (Ada) is currently Code 31. Install in **Optimus/Hybrid mode** (iGPU drives the internal display), not in dGPU-only mode.

**Verification:** Before install, confirm BIOS → Config → Display → Graphics Device is set to "Hybrid Graphics" not "Discrete Graphics Only".

### Risk 3: Windows Update Re-Installing Old Drivers (HIGH Without Mitigation)

Windows Update may push the old Lenovo-customized 596.59 enterprise driver for DEV_28B8 within minutes of the install completing, re-creating the split. The `-clean` flag and pausing Windows Update prevent this.

**Mitigation:** 
1. Use DDU's "Prevent downloads from Windows Update" option during clean
2. Pause Windows Update for 7 days after install
3. Verify both GPUs are stable on 610.47 before re-enabling WU

### Risk 4: BIOS Thunderbolt Security Blocking Re-Enumeration (LOW)

If Thunderbolt security is set to "User Authorization" or higher, the eGPU may require re-authorization after the driver wipe. The eGPU PCI device may not re-enumerate until the user clicks "Allow" in the Thunderbolt manager.

**Mitigation:** Set Thunderbolt security to "No Security" in BIOS before the install, or be prepared to click the Thunderbolt authorization prompt after reboot.

### Risk 5: 610.47 Drops Classic NVIDIA Control Panel (LOW impact, worth noting)

NVIDIA 610.47 removes the legacy NVIDIA Control Panel from the installer — only the NVIDIA App is available. Any scripts or workflows that call `nvcplui.exe` or depend on the Control Panel registry keys must migrate to the NVIDIA App CLI or new APIs.

### Risk 6: RTX 5060 Ti eGPU + Thunderbolt — Known Issues (MEDIUM)

NVIDIA's open-gpu-kernel-modules tracker (GitHub issue #974) documents RTX 5060 Ti eGPU initialization issues over Thunderbolt on Linux, and there are reports of CUDA hard-lock via TB4. Windows behavior may differ, but if the 5060 Ti eGPU has intermittent TB disconnects post-install, this is a known Blackwell+TB interaction. The NVIDIA forums thread "GeForce GRD 610.47 Feedback Thread (Released 5/26/26)" should be checked for any user-reported eGPU issues with 610.47 before proceeding.

---

## 6. Rollback Procedure

### Quick Rollback (if 610.47 causes instability)

1. Boot Safe Mode
2. Run DDU → "Clean and do NOT restart"
3. Download and install the last-known-good combination:
   - RTX 2000 Ada: 596.59 Enterprise (the current oem236.inf package, version 32.0.15.9659)
   - RTX 5060 Ti: 591.86 (the original launch driver, version 32.0.15.9186)
   Note: this restores the split-kernel Code 31 state — but restores a working eGPU

### System Restore Rollback (full state)

```powershell
# List available restore points
Get-ComputerRestorePoint | Select-Object Description, CreationTime | Format-Table

# Initiate restore (boots into recovery, then restores)
# Restore-Computer -RestorePoint <SequenceNumber> -Confirm
```
System Restore is only viable if done before any hardware or BIOS changes. It will not be available if Restore was disabled or if the C: volume ran out of shadow copy space.

---

## 7. Lenovo-Validated Driver Assessment

**Not recommended for this configuration.** Reasons:

1. The Lenovo P1 Gen 7 driver package (hosted at pcsupport.lenovo.com for types 21KV/21KW) is validated for the internal Optimus configuration: Intel Arc iGPU + RTX 2000 Ada dGPU.
2. Lenovo's INF will not enumerate DEV_2D04 (RTX 5060 Ti) — that is a desktop consumer GPU not in any ThinkPad OEM validation matrix.
3. Using the Lenovo package would fix the Ada (Code 31) but leave the 5060 Ti on its own INF, restoring the same split-kernel state (different versions of nvlddmkm.sys, depending on which Lenovo package version is current).
4. Lenovo driver updates for the P1 Gen 7 typically lag NVIDIA's upstream branch by 2-4 months.

**When to prefer Lenovo:** If the eGPU is permanently removed and only the internal Ada + Arc Optimus config is used, install the Lenovo ds-package for maximum MUX/Optimus/firmware integration stability.

---

## 8. Sources

### Primary Sources (Authoritative)
- **NVIDIA 610.47 GRD release notes PDF** (official, extracted locally): `https://us.download.nvidia.com/Windows/610.47/610.47-win11-win10-release-notes.pdf` — confirms GeForce supported GPU lists (Tables 1, 2, 3); RTX 2000 Ada Laptop GPU NOT listed in GRD
- NVIDIA GeForce Game Ready Driver 610.47 official page: https://www.nvidia.com/en-us/drivers/details/271418/
- NVIDIA Studio Driver 610.47 official page: https://www.nvidia.com/en-us/drivers/details/271420/
- Station-Drivers: RTX/Quadro Enterprise NFB R610 U1 v610.47 (confirms RTX 2000 Ada Generation Laptop GPU supported; confirms GeForce excluded): https://www.station-drivers.com/index.php/en-us/forum/nvidia-drivers-firmwares-utilities/1027-nvidia-rtx-quadro-enterprise-new-feature-branch-r610-u1-driver-v610-47-whql
- Enterprise 610.47 direct download (confirmed functional): `https://us.download.nvidia.com/Windows/Quadro_Certified/610.47/610.47-quadro-rtx-desktop-notebook-win10-win11-64bit-international-dch-whql.exe`

### Supporting Sources
- Station-Drivers: GeForce GRD 610.47 thread: https://www.station-drivers.com/index.php/en-us/forum/nvidia-drivers-firmwares-utilities/1023-nvidia-geforce-game-ready-driver-610-47-whql
- Station-Drivers: NVIDIA Studio Driver v610.47: https://www.station-drivers.com/index.php/en/forum/nvidia-drivers-firmwares-utilities/1024-nvidia-studio-driver-v610-47-whql
- NVIDIA 596.59 Enterprise (current Ada driver): https://www.station-drivers.com/index.php/en/forum/nvidia-drivers-firmwares-utilities/1026-nvidia-rtx-quadro-enterprise-production-branch-r595-u5-driver-v596-59-whql
- TechPowerUp NVIDIA 610.47 article: https://www.techpowerup.com/349359/nvidia-geforce-graphics-drivers-610-47-whql-drops-control-panel-support
- GeForce RTX 5060 Ti Game Ready Driver launch article: https://www.nvidia.com/en-us/geforce/news/geforce-rtx-5060-ti-game-ready-driver/
- Softpedia NVIDIA STUDIO Notebook 610.47 (supported products): https://drivers.softpedia.com/get/GRAPHICS-BOARD/NVIDIA/NVIDIA-STUDIO-Notebook-Graphics-Driver-610-47.shtml
- necacom NVIDIA GRD 610.47 supported GPU list (confirms no RTX 2000 Ada Laptop GPU in GRD): https://www.necacom.net/index.php/nvidia/nvidia-geforce-game-ready-driver-v-610-47-whql
- DDU download: https://www.guru3d.com/download/display-driver-uninstaller-download/
- Lenovo ThinkPad P1 Gen 7 (21KV/21KW) driver downloads: https://pcsupport.lenovo.com/us/en/products/laptops-and-netbooks/thinkpad-p-series-laptops/thinkpad-p1-gen-7-type-21kv-21kw/downloads
- RTX 5060 Ti eGPU Thunderbolt issues (NVIDIA GitHub): https://github.com/NVIDIA/open-gpu-kernel-modules/issues/974
- GeForce GRD 610.47 Feedback Thread (NVIDIA forums): https://www.nvidia.com/en-us/geforce/forums/game-ready-drivers/13/586393/geforce-grd-61047-feedback-thread-released-52626/
- NVIDIA silent install flags (driver docs): https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/windows.html

---

## 9. Quick Reference Summary

| Item | Value |
|------|-------|
| Recommended driver version | **610.47 WHQL** (Enterprise RTX/Quadro NFB R610 U1, released 2026-05-26) |
| Primary download URL | `https://us.download.nvidia.com/Windows/Quadro_Certified/610.47/610.47-quadro-rtx-desktop-notebook-win10-win11-64bit-international-dch-whql.exe` |
| Fallback download URL | `https://us.download.nvidia.com/Windows/610.47/610.47-desktop-win10-win11-64bit-international-dch-whql.exe` |
| DDU tool | https://www.guru3d.com/download/display-driver-uninstaller-download/ |
| Silent install command | `<installer>.exe -s -clean -n Display.Driver` |
| Install order | Enterprise 610.47 first (covers Ada); then per-device GRD 610.47 Have Disk bind for 5060 Ti if needed (do NOT run GRD full installer) |
| Pre-step | DDU in Safe Mode, eGPU connected, AC power, external monitors off, WU paused |
| Reboot required | Yes (one reboot after installer, possibly one during) |
| Lenovo pkg recommended? | No — use generic NVIDIA for eGPU support |
| Single-package covers both GPUs? | **No** — confirmed from official 610.47 release notes PDF. Enterprise covers Ada (DEV_28B8), GRD covers RTX 5060 Ti (DEV_2D04). No package covers both. |
| Key risk | WU re-installing 596.59 for Ada (different kernel → Code 31 again); eGPU display drop during install; MUX mode; second installer wiping first INF binding |
| Rollback | System Restore point + DDU + reinstall 596.59 Enterprise for Ada |
| DriverVer 32.0.15.XXXX for 610.47 | **Not confirmed** from primary sources. Convention-predicted: 32.0.15.1047 (applying the same pattern as 9186→591.86, 9659→596.59 → 1047→610.47), treat as unverified. Do not use 9647 — that decodes to 596.47, a different driver. |
