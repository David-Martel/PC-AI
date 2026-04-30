# Boot Diagnostics 20260430-132534

- Generated: 2026-04-30T13:25:53.3188572-04:00
- Output: `C:\codedev\PC_AI\Reports\boot-diagnostics\20260430-132534`
- Window: last 360 minutes
- Post-reboot verifier: True

## Task Issues

| Task | State | Last Run | Last Result | Issues |
| --- | --- | --- | --- | --- |
| \OneDrive Per-Machine Standalone Update Task | Ready | 04/29/2026 14:04:49 | 2147806724 | LastTaskResult=2147806724 |
| \OneDrive Reporting Task-S-1-5-21-357761096-1057752491-2136275046-1002 | Ready | 11/30/1999 00:00:00 | 267011 | LastTaskResult=267011 |
| \OneDrive Reporting Task-S-1-5-21-357761096-1057752491-2136275046-1003 | Ready | 11/30/1999 00:00:00 | 267011 | LastTaskResult=267011 |
| \OneDrive Reporting Task-S-1-5-21-357761096-1057752491-2136275046-1007 | Ready | 11/30/1999 00:00:00 | 267011 | LastTaskResult=267011 |
| \OneDrive Reporting Task-S-1-5-21-357761096-1057752491-2136275046-1009 | Ready | 11/30/1999 00:00:00 | 267011 | LastTaskResult=267011 |
| \OneDrive Startup Task-S-1-5-21-357761096-1057752491-2136275046-1003 | Ready | 11/30/1999 00:00:00 | 267011 | LastTaskResult=267011 |
| \OneDrive Startup Task-S-1-5-21-357761096-1057752491-2136275046-1007 | Ready | 11/30/1999 00:00:00 | 267011 | LastTaskResult=267011 |
| \OneDrive Startup Task-S-1-5-21-357761096-1057752491-2136275046-1009 | Ready | 11/30/1999 00:00:00 | 267011 | LastTaskResult=267011 |
| \Process Lasso Core Engine Only | Running | 04/30/2026 12:50:54 | 267009 | LastTaskResult=267009; RunningLongerThanMinutes=30 |
| \Process Lasso Management Console (GUI) | Running | 04/30/2026 12:50:53 | 267009 | LastTaskResult=267009; RunningLongerThanMinutes=30 |
| \Session agent for Process Lasso | Running | 04/30/2026 12:50:57 | 1 | LastTaskResult=1; RunningLongerThanMinutes=30 |
| \UnifiUdmDriveStackStartup | Ready | 04/30/2026 12:52:53 | 50 | LastTaskResult=50 |
| \WSL-Docker-Startup | Disabled | 01/25/2026 22:00:07 | 1 | LastTaskResult=1 |

## Expected VHDs

| Name | Path | Exists | Expected | Attached | Volumes |
| --- | --- | --- | --- | --- | --- |
| cloud-cache-disk | `T:\vm\cloud-cache-disk.vhdx` | True | mounted-volume F | True | F: cloud-cache-disk NTFS |
| share-ext4 | `T:\vm\share-ext4.vhdx` | True | attached-disk-only  | True |  |
| shared-dev | `T:\vm\shared-dev.vhdx` | True | mounted-volume W | True | W: WSL-Shared-Dev NTFS |

## Event Profile

- Events captured: 357
- Query errors: 0
- Optional providers skipped: 3
- Event JSON: `C:\codedev\PC_AI\Reports\boot-diagnostics\20260430-132534\boot-events.json`

## Post-Reboot Verifier

- Failure count: 1
- Mount failures: 
- Sync failures: OneDrive/FileSyncHelper WER events occurred after 2026-04-30T12:50:02.5000000-04:00

## Process Lasso Validation

- Status: Ran
- Script: `C:\codedev\PC_AI\Tools\Test-ProcessLassoBootSafety.ps1`
- Output: `C:\codedev\PC_AI\Reports\boot-diagnostics\20260430-132534\processlasso-validation.txt`
