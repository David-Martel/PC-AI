function Get-DiagnosticTestCases {
    <#
    .SYNOPSIS
        Returns test cases specific to PC-AI diagnostic evaluation
    #>
    return @(
        [EvaluationTestCase]@{
            Id = "diag-001"
            Category = "device-error"
            Prompt = @"
Analyze this diagnostic report and provide recommendations:

Device Manager Errors:
- Unknown Device (Code 28): PCI\VEN_10DE&DEV_1234
- USB Controller Error (Code 43): USB\VID_0781&PID_5583

SMART Status: All disks OK
Network: Connected
"@
            ExpectedOutput = @"
The report shows two device issues requiring attention. The unknown device needs driver installation, and the USB controller may have a hardware or driver issue.
"@
            Context = @{ context = "Windows 10 diagnostics" }
            Tags = @('device', 'driver', 'usb')
        }
        [EvaluationTestCase]@{
            Id = "diag-002"
            Category = "disk-health"
            Prompt = @"
Analyze this diagnostic report:

SMART Status:
- Disk 0 (Samsung 980 Pro): GOOD
- Disk 1 (WD Blue): CAUTION - Reallocated Sector Count: 50
- Disk 2 (Seagate): GOOD

No device errors.
"@
            ExpectedOutput = @"
The WD Blue disk shows signs of wear with reallocated sectors. Recommend backup and monitoring.
"@
            Context = @{ context = "SMART disk health analysis" }
            Tags = @('disk', 'smart', 'backup')
        }
        [EvaluationTestCase]@{
            Id = "diag-003"
            Category = "network"
            Prompt = @"
Analyze network diagnostic:

Adapters:
- Intel Wi-Fi 6: Connected, 866 Mbps
- Realtek Ethernet: Disconnected
- Hyper-V Virtual Switch: Connected

DNS: 8.8.8.8 (responding)
Gateway: 192.168.1.1 (responding)
"@
            ExpectedOutput = @"
Network is healthy with Wi-Fi connected. Ethernet disconnected is normal if not plugged in.
"@
            Context = @{ context = "Network diagnostics" }
            Tags = @('network', 'wifi', 'dns')
        }
        [EvaluationTestCase]@{
            Id = "diag-004"
            Category = "wsl"
            Prompt = @"
WSL Diagnostic Report:

WSL Version: 2.0.14.0
Distributions:
- Ubuntu-22.04: Running, 2GB memory
- docker-desktop-data: Stopped

Network: vEthernet (WSL) connected
Docker: Running
"@
            ExpectedOutput = @"
WSL2 environment is healthy with Ubuntu running. Docker integration is active.
"@
            Context = @{ context = "WSL2 and Docker diagnostics" }
            Tags = @('wsl', 'docker', 'virtualization')
        }
        [EvaluationTestCase]@{
            Id = "diag-005"
            Category = "critical"
            Prompt = @"
CRITICAL DIAGNOSTIC ALERT:

SMART Status:
- Disk 0: FAILING - Pending Sector Count: 1500, Reallocated: 800

Event Log Errors:
- Disk: The device has a bad block (x15 in last hour)
- NTFS: The file system structure is corrupt

Device Errors: None
"@
            ExpectedOutput = @"
CRITICAL: Disk 0 is failing with bad sectors and filesystem corruption. Immediate backup required. Do not write additional data. Consider replacement.
"@
            Context = @{ context = "Critical disk failure" }
            Tags = @('critical', 'disk', 'backup', 'failure')
        }
    )
}
