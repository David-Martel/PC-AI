# NukeNul Deployment Checklist

Production-ready deployment checklist for the NukeNul tool.

---

## Pre-Deployment

### Code Review

- [ ] Rust code reviewed for safety and correctness
- [ ] C# code reviewed for P/Invoke correctness
- [ ] No hardcoded paths or credentials
- [ ] Error handling is comprehensive
- [ ] Logging is appropriate (not excessive)

### Security Audit

- [ ] Run `cargo audit` on Rust dependencies
- [ ] Check for known vulnerabilities in .NET packages
- [ ] Verify no sensitive data in output
- [ ] Test with restricted user accounts
- [ ] Verify elevation is not required (unless intended)

### Testing

- [ ] All integration tests pass (`.\test.ps1`)
- [ ] Stress test with 1000+ files passes (`.\test.ps1 -TestCount 1000`)
- [ ] Deep nesting test passes (`.\test.ps1 -DeepNesting`)
- [ ] Manual testing on production-like data completed
- [ ] Performance benchmarks meet expectations
- [ ] No memory leaks detected during long runs

### Documentation

- [ ] README.md is up-to-date
- [ ] BUILD_AND_TEST.md is comprehensive
- [ ] QUICK_REFERENCE.md is accurate
- [ ] Code comments are clear and helpful
- [ ] Known limitations are documented

---

## Build Process

### Release Build

```powershell
# Clean release build
.\build.ps1 -Clean -Configuration Release

# Verify output
Test-Path ".\nuker_core\target\release\nuker_core.dll"
Test-Path ".\NukeNul\bin\Release\net8.0\NukeNul.exe"
```

- [ ] Release build completes without errors
- [ ] No compiler warnings in Rust
- [ ] No compiler warnings in C#
- [ ] DLL is copied to C# output directory
- [ ] Both artifacts exist in expected locations

### Self-Contained Build (Optional)

```powershell
# Self-contained executable
.\build.ps1 -Publish -Clean

# Verify output
Test-Path ".\NukeNul\bin\Release\net8.0\win-x64\publish\NukeNul.exe"
```

- [ ] Self-contained build completes successfully
- [ ] Executable size is reasonable (<50MB for self-contained)
- [ ] No external dependencies required
- [ ] Runs on clean Windows installation (tested)

### Binary Verification

```powershell
# Check file signatures
Get-AuthenticodeSignature ".\NukeNul\bin\Release\net8.0\NukeNul.exe"

# Check architecture
dumpbin /headers ".\nuker_core\target\release\nuker_core.dll"
dumpbin /headers ".\NukeNul\bin\Release\net8.0\NukeNul.exe"
```

- [ ] Both binaries are x64 architecture
- [ ] Code signing completed (if applicable)
- [ ] Antivirus scan passed
- [ ] Windows SmartScreen reputation established (if applicable)

---

## Deployment Strategies

### Strategy 1: PATH Installation

**Use Case**: System-wide availability for all users

```powershell
# Install to Windows system directory
$ExePath = ".\NukeNul\bin\Release\net8.0\win-x64\publish\NukeNul.exe"
Copy-Item $ExePath "C:\Windows\System32\NukeNul.exe"
```

**Checklist**:
- [ ] Administrative privileges obtained
- [ ] Self-contained executable used
- [ ] Executable copied to `C:\Windows\System32`
- [ ] Verified from new PowerShell session: `NukeNul.exe --help`
- [ ] Added to documentation/runbooks

### Strategy 2: User Binaries

**Use Case**: Per-user installation without admin rights

```powershell
# Install to user AppData
$UserBin = "$env:LOCALAPPDATA\Microsoft\WindowsApps"
$ExePath = ".\NukeNul\bin\Release\net8.0\win-x64\publish\NukeNul.exe"
Copy-Item $ExePath "$UserBin\NukeNul.exe"
```

**Checklist**:
- [ ] User binaries directory exists
- [ ] Executable copied successfully
- [ ] Verified from new PowerShell session: `NukeNul.exe --help`
- [ ] User documented installation location

### Strategy 3: Custom Tool Directory

**Use Case**: Centralized tools directory with version control

```powershell
# Install to custom tools directory
$ToolDir = "C:\Tools\NukeNul"
New-Item -ItemType Directory -Path $ToolDir -Force

# Copy files
Copy-Item ".\NukeNul\bin\Release\net8.0\NukeNul.exe" $ToolDir
Copy-Item ".\NukeNul\bin\Release\net8.0\nuker_core.dll" $ToolDir

# Add to PATH
$CurrentPath = [Environment]::GetEnvironmentVariable("PATH", "User")
[Environment]::SetEnvironmentVariable("PATH", "$CurrentPath;$ToolDir", "User")
```

**Checklist**:
- [ ] Tool directory created
- [ ] Both EXE and DLL copied
- [ ] Added to user PATH or system PATH
- [ ] Verified from new PowerShell session
- [ ] Documented in team wiki/documentation

### Strategy 4: PowerShell Module

**Use Case**: Integration with existing PowerShell workflows

```powershell
# Create module structure
$ModulePath = "$env:USERPROFILE\Documents\PowerShell\Modules\NukeNul"
New-Item -ItemType Directory -Path $ModulePath -Force

# Copy files
Copy-Item ".\NukeNul\bin\Release\net8.0\*" $ModulePath

# Create module manifest
New-ModuleManifest -Path "$ModulePath\NukeNul.psd1" `
  -RootModule "NukeNul.exe" `
  -ModuleVersion "1.0.0" `
  -Description "High-performance reserved filename deletion tool"
```

**Checklist**:
- [ ] Module directory created
- [ ] Files copied to module directory
- [ ] Module manifest created
- [ ] Module imports successfully: `Import-Module NukeNul`
- [ ] Cmdlet is available: `Get-Command -Module NukeNul`

### Strategy 5: Wrapper Script Deployment

**Use Case**: Gradual migration from existing PowerShell script

```powershell
# Deploy wrapper
Copy-Item ".\delete-nul-files-v2.ps1" "C:\Scripts\delete-nul-files.ps1"

# Update existing automation to use new wrapper
# (wrapper auto-detects and uses Rust version if available)
```

**Checklist**:
- [ ] Wrapper script deployed
- [ ] NukeNul.exe built and accessible
- [ ] Wrapper tested with both Rust and PowerShell fallback
- [ ] Existing automation updated to use wrapper
- [ ] Rollback plan documented

---

## Post-Deployment Verification

### Functional Testing

```powershell
# Test basic functionality
NukeNul.exe --help

# Test on safe directory
$TestDir = "$env:TEMP\NukeNul_Verify"
New-Item -ItemType Directory -Path $TestDir -Force
NukeNul.exe $TestDir
Remove-Item $TestDir -Recurse -Force
```

- [ ] Help output displays correctly
- [ ] Executes without errors on empty directory
- [ ] JSON output is well-formed
- [ ] No crashes or hangs observed

### Performance Verification

```powershell
# Run benchmark on realistic data
NukeNul.exe "C:\Projects"
# Verify ElapsedMs is reasonable for directory size
```

- [ ] Completes in expected time
- [ ] CPU usage is reasonable (not 100% indefinitely)
- [ ] Memory usage is reasonable (<500MB for large directories)
- [ ] No performance degradation over multiple runs

### Integration Testing

- [ ] Tested in target environment (dev/staging/prod)
- [ ] Tested with representative data volumes
- [ ] Tested with various file system types (NTFS, ReFS, network shares)
- [ ] Tested with long paths (>260 characters)
- [ ] Tested with special characters in paths

### Error Handling

```powershell
# Test with permission issues
NukeNul.exe "C:\Windows\System32"  # Should handle access denied gracefully

# Test with invalid paths
NukeNul.exe "Z:\NonExistent"  # Should report error clearly

# Test with locked files
# (Create and lock a "nul" file, then run tool)
```

- [ ] Access denied errors are handled gracefully
- [ ] Invalid paths are reported clearly
- [ ] Locked files are skipped with appropriate error
- [ ] No unexpected crashes or panics

---

## Monitoring and Maintenance

### Logging

```powershell
# Capture output for monitoring
NukeNul.exe "C:\Projects" | Tee-Object -FilePath "nuke-log.json"

# Parse for monitoring
$Results = Get-Content "nuke-log.json" | ConvertFrom-Json
if ($Results.Results.Errors -gt 0) {
    # Alert or log warning
}
```

- [ ] Output logging configured
- [ ] Log rotation configured (if running regularly)
- [ ] Monitoring alerts configured for errors
- [ ] Performance metrics are tracked

### Scheduled Tasks

```powershell
# Create scheduled task (if needed)
$Action = New-ScheduledTaskAction -Execute "NukeNul.exe" -Argument "C:\Projects"
$Trigger = New-ScheduledTaskTrigger -Daily -At "2:00AM"
$Principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount
Register-ScheduledTask -TaskName "NukeNul-Daily" -Action $Action -Trigger $Trigger -Principal $Principal
```

- [ ] Scheduled task created (if applicable)
- [ ] Task runs successfully on schedule
- [ ] Output is logged appropriately
- [ ] Errors are alerted appropriately

### Updates and Versioning

```powershell
# Version information
$Version = "1.0.0"
$BuildDate = Get-Date -Format "yyyy-MM-dd"

# Document in release notes
@"
Version: $Version
Build Date: $BuildDate
Changes:
- Initial release
- Rust/C# hybrid implementation
- Parallel directory walking
- JSON output
"@ | Out-File "RELEASE_NOTES.txt"
```

- [ ] Version number is tracked
- [ ] Release notes are maintained
- [ ] Changelog is updated
- [ ] Upgrade path is documented

---

## Rollback Plan

### Backup Original Files

```powershell
# Before deployment, backup original files
$BackupDir = "C:\Backups\NukeNul_$(Get-Date -Format 'yyyyMMdd')"
New-Item -ItemType Directory -Path $BackupDir -Force

# Backup original PowerShell script
Copy-Item ".\delete-nul-files.ps1" $BackupDir

# Backup wrapper (if updating)
if (Test-Path "C:\Scripts\delete-nul-files.ps1") {
    Copy-Item "C:\Scripts\delete-nul-files.ps1" "$BackupDir\delete-nul-files-wrapper.ps1"
}
```

- [ ] Original files backed up
- [ ] Backup location documented
- [ ] Backup verified to be restorable

### Rollback Procedure

```powershell
# If issues occur, restore original files
$BackupDir = "C:\Backups\NukeNul_20250123"  # Use actual backup date

# Remove new installation
Remove-Item "C:\Windows\System32\NukeNul.exe" -ErrorAction SilentlyContinue

# Restore original wrapper
Copy-Item "$BackupDir\delete-nul-files-wrapper.ps1" "C:\Scripts\delete-nul-files.ps1" -Force

# Verify rollback
& "C:\Scripts\delete-nul-files.ps1" -SearchPath $env:TEMP
```

- [ ] Rollback procedure documented
- [ ] Rollback tested in non-production environment
- [ ] Rollback can be executed quickly (<5 minutes)
- [ ] Team is trained on rollback procedure

---

## Documentation Updates

### User-Facing Documentation

- [ ] User guide updated with new tool information
- [ ] Examples updated to use NukeNul
- [ ] Performance expectations documented
- [ ] Troubleshooting guide updated

### Technical Documentation

- [ ] Architecture documentation updated
- [ ] API/interface documentation updated (if applicable)
- [ ] Integration guide updated
- [ ] Maintenance procedures documented

### Team Communication

- [ ] Deployment announced to team
- [ ] Training session conducted (if needed)
- [ ] FAQ created and shared
- [ ] Feedback mechanism established

---

## Sign-Off

### Stakeholder Approval

- [ ] Development team sign-off
- [ ] QA team sign-off
- [ ] Security team sign-off (if applicable)
- [ ] Management approval (if required)

### Final Checks

- [ ] All checklist items completed
- [ ] No critical issues outstanding
- [ ] Rollback plan is ready
- [ ] Monitoring is in place
- [ ] Documentation is complete

### Deployment Record

```
Deployed By: _______________
Date: _______________
Version: _______________
Environment: _______________
Notes: _______________
```

---

## Post-Deployment

### Week 1

- [ ] Monitor logs daily for errors
- [ ] Check performance metrics
- [ ] Gather user feedback
- [ ] Address any issues promptly

### Week 2-4

- [ ] Review performance trends
- [ ] Optimize if needed
- [ ] Update documentation based on feedback
- [ ] Plan for next iteration (if needed)

### Ongoing

- [ ] Regular security audits
- [ ] Dependency updates (Rust crates, .NET packages)
- [ ] Performance monitoring
- [ ] User satisfaction surveys

---

## Appendix: Emergency Contacts

```
Development Team Lead: _______________
Operations Team Lead: _______________
Security Team Contact: _______________
Escalation Path: _______________
```

---

## Appendix: Useful Commands

```powershell
# Check current deployment
Get-Command NukeNul.exe | Select-Object Source, Version

# View version info
[System.Diagnostics.FileVersionInfo]::GetVersionInfo("C:\Path\To\NukeNul.exe")

# Check last execution
Get-EventLog -LogName Application -Source "NukeNul" -Newest 10

# Monitor performance
Get-Process NukeNul -ErrorAction SilentlyContinue | Format-Table CPU,WS,PM -AutoSize
```

---

**Deployment Status**: [ ] Not Started  [ ] In Progress  [ ] Complete

**Deployment Date**: _______________

**Signed Off By**: _______________
