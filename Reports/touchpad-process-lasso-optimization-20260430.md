# Touchpad Process Lasso Optimization - 2026-04-30

## Hardware And Input Stack

Machine: Lenovo ThinkPad P1 Gen 7.

Relevant discovered devices and drivers:

- Lenovo PSREF identifies the machine as using a glass multi-touch haptic
  touchpad with TrackPoint support.
- Local HID inventory shows the built-in touchpad as Sensel-class HID devices:
  - `HID\SNSL002D&Col01...` as `HID-compliant mouse`.
  - `HID\SNSL002D&Col02...` as `HID-compliant touch pad`.
  - `HID\SNSL002D&Col04...` as a vendor-defined HID device.
- Local driver inventory shows the active controller path is kernel/driver
  based:
  - `iaLPSS2_I2C_MTL.sys` running for Intel Serial IO I2C.
  - `hidi2c.sys` running for Microsoft I2C HID.
  - `mshidkmdf.sys`, `mshidumdf.sys`, `mouclass.sys`, and `mouhid.sys`
    running for HID/mouse processing.
- Local user-mode support process:
  - `SynRpcServer.exe` running as `SynHsaService` from
    `synawudfbiousbuwp.inf`.
- Local TrackPoint device inventory includes an ELAN TrackPoint device.

Process Lasso cannot directly reprioritize kernel HID/I2C interrupt handling.
The useful Process Lasso surface is therefore the user-mode input/shell path
and background workloads that compete for CPU, I/O, memory, and composition
latency.

## Internet Research Notes

- Process Lasso documentation states that ProBalance maintains responsiveness
  by dynamically lowering CPU priority of overly active background processes.
- Bitsum's own priority guidance says it is generally better to lower
  background/unimportant processes than to aggressively raise important ones.
- Process Lasso supports persistent CPU, GPU, I/O, and memory priority rules,
  and its rule notation confirms ProBalance exclusions, SmartTrim exclusions,
  priority classes, and I/O classes.
- Bitsum's I/O priority reference maps High, Normal, Low, and Very Low
  semantics, with Below Normal CPU priority associated with lower I/O priority.
- Notebookcheck and Lenovo PSREF identify the P1 Gen 7 touchpad as a haptic
  touchpad, with third-party reporting that this P1 Gen 7 generation uses a
  Sensel haptic trackpad.
- Sensel community reports describe P1 Gen 7 touchpad degradation over time on
  some systems, which supports treating this as a mixed firmware/driver/load
  sensitivity issue rather than only an application scheduling issue.

## Applied Process Lasso Policy

Script updated and applied:

- `Tools\Apply-ProcessLassoUiSyncTuning.ps1`

Backup:

- `C:\ProgramData\ProcessLasso\config\prolasso.ini.bak-20260430-145824-boot-safety`

Reports:

- `Reports\processlasso-touchpad-responsiveness-dryrun.json`
- `Reports\processlasso-touchpad-responsiveness-apply.json`
- `Reports\processlasso-touchpad-responsiveness-postapply-dryrun.json`
- `Reports\processlasso-touchpad-boot-safety-validation.json`

### Protected/Elevated User-Mode Input Path

Added or verified ProBalance and SmartTrim protection for:

- `TextInputHost.exe`
- `TabTip.exe`
- `ctfmon.exe`
- `dwm.exe`
- `explorer.exe`
- `sihost.exe`
- `ShellExperienceHost.exe`
- `StartMenuExperienceHost.exe`
- `SynRpcServer.exe`
- `Sensel*.exe`
- `SNSL*.exe`
- `SynTP*.exe`
- `Synaptics*.exe`
- `ELAN*.exe`
- `ETD*.exe`
- Lenovo service/UI support processes.

Default CPU priority:

- Above Normal for the user-mode input/shell/vendor path.

Default I/O priority:

- High I/O hint for the same user-mode input/shell/vendor path.

This intentionally avoids Real-time priority.

### De-Elevated Background Competitors

Default CPU priority set to Below Normal and default I/O priority set to Low
for competing background classes:

- OneDrive/FileSync:
  - `OneDrive.exe`
  - `OneDrive.Sync.Service.exe`
  - `FileSyncHelper.exe`
- Other cloud/sync:
  - `GoogleDriveFS.exe`
  - `Dropbox.exe`
  - `iCloudDrive.exe`
  - `iCloudCKKS.exe`
  - `ProtonDrive.exe`
  - `rclone.exe`
- Docker/WSL/Redis:
  - `Docker Desktop.exe`
  - `com.docker.backend.exe`
  - `com.docker.build.exe`
  - `docker-agent.exe`
  - `docker-sandbox.exe`
  - `wsl.exe`
  - `wslhost.exe`
  - `wslservice.exe`
  - `vmmemWSL`
  - `redis-server.exe`
  - `redis-service.exe`
- Build/archive/tooling:
  - `7z.exe`
  - `robocopy.exe`
  - `cargo.exe`
  - `rustc.exe`
  - `sccache.exe`
  - `sccache-dist.exe`
  - `link.exe`
  - `cl.exe`
  - `node.exe`
  - `npm.exe`
  - `npx.exe`
  - `winget.exe`
  - `git-cluster-analyzer.exe`

This trades some sync/build throughput for foreground responsiveness while
OneDrive remains unstable.

## Live Verification

`Tools\Test-ProcessLassoBootSafety.ps1` passed after the policy update:

- Governor running: `ProcessGovernor.exe`, PID `58624`.
- Governor responding: true.
- ProBalance exclusions: ok.
- SmartTrim exclusions: ok.
- Logging flags: ok.
- Recent Process Lasso log lines: present.

Process Lasso log evidence after apply showed live rules being enforced, for
example `node.exe` and other background helpers being adjusted to Below Normal
CPU and Low I/O priority.

WMIC base-priority snapshot after apply:

- Elevated input path at base priority `10`:
  - `dwm.exe`
  - `explorer.exe`
  - `sihost.exe`
  - `StartMenuExperienceHost.exe`
  - `ShellExperienceHost.exe`
  - `TextInputHost.exe`
  - `ctfmon.exe`
  - `SynRpcServer.exe`
  - `Lenovo.Modern.ImController.exe`
  - `LenovoVantageService.exe`
- De-elevated background competitors at base priority `6`:
  - `OneDrive.exe`
  - `FileSyncHelper.exe`
  - `GoogleDriveFS.exe`
  - `rclone.exe`
  - Docker backend/build/agent/sandbox processes.
  - Redis processes.
  - `node.exe` helper processes.
  - `git-cluster-analyzer.exe`.

## Remaining Risk

This policy reduces scheduling and I/O contention but cannot fix firmware,
static-charge, HID/I2C driver, or hardware-layer Sensel issues. If touchpad
glitches persist while OneDrive remains paused or de-elevated, the next
debugging step should capture a fresh UI glitch report and check HID/I2C,
Kernel-PnP, Lenovo/Sensel driver versions, and firmware update state.

## Sources

- Process Lasso documentation:
  <https://bitsum.com/processlasso-docs/>
- Bitsum I/O priority reference:
  <https://bitsum.com/pl_io_priority.php>
- Lenovo ThinkPad P1 Gen 7 PSREF:
  <https://psref.lenovo.com/syspool/Sys/PDF/ThinkPad/ThinkPad_P1_Gen_7/ThinkPad_P1_Gen_7_Spec.PDF>
- Notebookcheck P1 Gen 7 haptic/Sensel touchpad review context:
  <https://www.notebookcheck.net/Lenovo-ThinkPad-P1-Gen-7-review-Without-TrackPoint-buttons-with-Nvidia-GeForce-RTX-4060.901578.0.html>
- Sensel P1 Gen 7 touchpad issue report:
  <https://forum.sensel.com/t/issue-with-thinkpad-p1-gen7-touchpad/3409>
