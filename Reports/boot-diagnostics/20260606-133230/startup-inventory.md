# Boot Diagnostics 20260606-133230

- Generated: 2026-06-06T13:32:47.2594265-04:00
- Output: `C:\codedev\PC_AI\Reports\boot-diagnostics\20260606-133230`
- Window: last 240 minutes
- Post-reboot verifier: True

## Task Issues

| Task | State | Last Run | Last Result | Issues |
| --- | --- | --- | --- | --- |
| \OneDrive Per-Machine Standalone Update Task | Ready | 11/30/1999 00:00:00 | 267011 | LastTaskResult=267011 |
| \OneDrive Reporting Task-S-1-5-21-357761096-1057752491-2136275046-1003 | Ready | 11/30/1999 00:00:00 | 267011 | LastTaskResult=267011 |
| \OneDrive Reporting Task-S-1-5-21-357761096-1057752491-2136275046-1007 | Ready | 11/30/1999 00:00:00 | 267011 | LastTaskResult=267011 |
| \OneDrive Reporting Task-S-1-5-21-357761096-1057752491-2136275046-1009 | Ready | 11/30/1999 00:00:00 | 267011 | LastTaskResult=267011 |
| \OneDrive Startup Task-S-1-5-21-357761096-1057752491-2136275046-1003 | Ready | 11/30/1999 00:00:00 | 267011 | LastTaskResult=267011 |
| \OneDrive Startup Task-S-1-5-21-357761096-1057752491-2136275046-1007 | Ready | 11/30/1999 00:00:00 | 267011 | LastTaskResult=267011 |
| \OneDrive Startup Task-S-1-5-21-357761096-1057752491-2136275046-1009 | Ready | 11/30/1999 00:00:00 | 267011 | LastTaskResult=267011 |
| \Process Lasso Core Engine Only | Running | 06/05/2026 21:04:26 | 267009 | LastTaskResult=267009; RunningLongerThanMinutes=30 |
| \Session agent for Process Lasso | Ready | 06/05/2026 21:04:27 | 1073807364 | LastTaskResult=1073807364 |
| \UnifiUdmDriveStackStartup | Disabled | 04/30/2026 12:52:53 | 50 | LastTaskResult=50 |
| \WSL-Docker-Startup | Disabled | 01/25/2026 22:00:07 | 1 | LastTaskResult=1 |

## Expected VHDs

| Name | Path | Exists | Expected | Attached | Volumes |
| --- | --- | --- | --- | --- | --- |
| cloud-cache-disk | `T:\vm\cloud-cache-disk.vhdx` | True | mounted-volume F | True | F: cloud-cache-disk NTFS |
| share-ext4 | `T:\vm\share-ext4.vhdx` | True | attached-disk-only  | True |  |
| shared-dev | `T:\vm\shared-dev.vhdx` | True | mounted-volume W | True | W: WSL-Shared-Dev NTFS |

## Event Profile

- Events captured: 9
- Query errors: 0
- Optional providers skipped: 3
- Event JSON: `C:\codedev\PC_AI\Reports\boot-diagnostics\20260606-133230\boot-events.json`

## Post-Reboot Verifier

- Failure count: 0
- Mount failures: 
- Sync failures: 

## Process Lasso Validation

- Status: Ran
- Script: `C:\codedev\PC_AI\Tools\Test-ProcessLassoBootSafety.ps1`
- Output: `C:\codedev\PC_AI\Reports\boot-diagnostics\20260606-133230\processlasso-validation.txt`
