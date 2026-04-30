# Boot Diagnostics 20260430-150703

- Generated: 2026-04-30T15:07:12.7682643-04:00
- Output: `C:\codedev\PC_AI\Reports\boot-diagnostics\20260430-150703`
- Window: last 240 minutes
- Post-reboot verifier: False

## Task Issues

| Task | State | Last Run | Last Result | Issues |
| --- | --- | --- | --- | --- |
| \OneDrive Per-Machine Standalone Update Task | Ready | 04/30/2026 14:15:20 | 2147806724 | LastTaskResult=2147806724 |
| \OneDrive Reporting Task-S-1-5-21-357761096-1057752491-2136275046-1002 | Ready | 11/30/1999 00:00:00 | 267011 | LastTaskResult=267011 |
| \OneDrive Reporting Task-S-1-5-21-357761096-1057752491-2136275046-1003 | Ready | 11/30/1999 00:00:00 | 267011 | LastTaskResult=267011 |
| \OneDrive Reporting Task-S-1-5-21-357761096-1057752491-2136275046-1007 | Ready | 11/30/1999 00:00:00 | 267011 | LastTaskResult=267011 |
| \OneDrive Reporting Task-S-1-5-21-357761096-1057752491-2136275046-1009 | Ready | 11/30/1999 00:00:00 | 267011 | LastTaskResult=267011 |
| \OneDrive Startup Task-S-1-5-21-357761096-1057752491-2136275046-1003 | Ready | 11/30/1999 00:00:00 | 267011 | LastTaskResult=267011 |
| \OneDrive Startup Task-S-1-5-21-357761096-1057752491-2136275046-1007 | Ready | 11/30/1999 00:00:00 | 267011 | LastTaskResult=267011 |
| \OneDrive Startup Task-S-1-5-21-357761096-1057752491-2136275046-1009 | Ready | 11/30/1999 00:00:00 | 267011 | LastTaskResult=267011 |
| \Session agent for Process Lasso | Running | 04/30/2026 12:50:57 | 1 | LastTaskResult=1; RunningLongerThanMinutes=30 |
| \UnifiUdmDriveStackStartup | Disabled | 04/30/2026 12:52:53 | 50 | LastTaskResult=50 |
| \WSL-Docker-Startup | Disabled | 01/25/2026 22:00:07 | 1 | LastTaskResult=1 |

## Expected VHDs

| Name | Path | Exists | Expected | Attached | Volumes |
| --- | --- | --- | --- | --- | --- |
| cloud-cache-disk | `T:\vm\cloud-cache-disk.vhdx` | True | mounted-volume F | True | F: cloud-cache-disk NTFS |
| share-ext4 | `T:\vm\share-ext4.vhdx` | True | attached-disk-only  | True |  |
| shared-dev | `T:\vm\shared-dev.vhdx` | True | mounted-volume W | True | W: WSL-Shared-Dev NTFS |

## Event Profile

- Events captured: 360
- Query errors: 0
- Optional providers skipped: 3
- Event JSON: `C:\codedev\PC_AI\Reports\boot-diagnostics\20260430-150703\boot-events.json`

## Process Lasso Validation

- Status: Ran
- Script: `C:\codedev\PC_AI\Tools\Test-ProcessLassoBootSafety.ps1`
- Output: `C:\codedev\PC_AI\Reports\boot-diagnostics\20260430-150703\processlasso-validation.txt`
