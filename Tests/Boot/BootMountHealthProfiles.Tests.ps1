#Requires -Modules Pester
# Protects: declared expected disks survive missing files and unrelated RustCache presence.
# Detects: inferred profile selection hiding missing mounts. Needs: Pester, no Windows devices.
# Breadcrumb: Tools/Test-BootMountHealth.ps1; Test-Path/OS inventory are boundary fixtures.
BeforeAll {
    $script:HealthScript = Join-Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) 'Tools/Test-BootMountHealth.ps1'
    function Get-CimInstance { param($ClassName) }
    function Get-ScheduledTask { param($TaskName) }
    function Get-WinEvent { param($FilterHashtable) }
    function Get-VHD { param($Path) }
    function Get-Disk { param($Number) }
    function Get-Partition { param($DiskNumber) }
    function Get-Volume { param([Parameter(ValueFromPipeline)]$InputObject) process {} }
}
Describe 'Boot mount explicit health profiles' {
    BeforeEach {
        Mock Get-CimInstance { [pscustomobject]@{ LastBootUpTime = (Get-Date).AddHours(-1) } }
        Mock Get-ScheduledTask { @() }
        Mock Get-WinEvent { @() }
        Mock Test-Path { $LiteralPath -eq 'C:\DevDrives\RustCache.vhdx' }
        Mock Get-VHD { [pscustomobject]@{ Attached = $true; DiskNumber = 7 } }
        Mock Get-Disk { [pscustomobject]@{ OperationalStatus = 'Online' } }
        Mock Get-Partition { [pscustomobject]@{ DiskNumber = 7 } }
        Mock Get-Volume { [pscustomobject]@{ DriveLetter = 'T'; FileSystemLabel = 'RustCache'; FileSystem = 'ReFS'; HealthStatus = 'Healthy'; OperationalStatus = 'OK'; Path = 'fixture' } }
    }
    # Protects: the default three-disk health contract.
    # Detects: file-presence inference hiding missing disks.
    # Needs: Pester and synthetic OS inventory; no disks, mounts or tasks.
    # Breadcrumb: Tools/Test-BootMountHealth.ps1 expected profile and missing-file checks.
    It 'keeps all default disks required when only RustCache exists' {
        $report = & $script:HealthScript -PassThru -DryRun
        $report.Status | Should -Be Fail
        @($report.Vhds).Count | Should -Be 3
        @($report.Failures | Where-Object { $_ -like 'Missing expected VHD:*' }).Count | Should -Be 3
        $report.Vhds.Name | Should -Contain 'cloud-cache-disk'
    }
    # Protects: explicit RustCache-only enrollment.
    # Detects: failure to honor the selected health profile.
    # Needs: Pester and synthetic OS inventory; no disks, mounts or tasks.
    # Breadcrumb: Tools/Test-BootMountHealth.ps1 expected profile and missing-file checks.
    It 'accepts RustCache alone only with the explicit profile' {
        $report = & $script:HealthScript -Profile RustCacheDevDrive -PassThru -DryRun
        $report.Status | Should -Be Pass
        @($report.Vhds).Count | Should -Be 1
        $report.Vhds.Name | Should -Be RustCache
    }
    # Protects: required RustCache file reporting.
    # Detects: missing media incorrectly passing health checks.
    # Needs: Pester and synthetic OS inventory; no disks, mounts or tasks.
    # Breadcrumb: Tools/Test-BootMountHealth.ps1 expected profile and missing-file checks.
    It 'reports missing RustCache under the explicit profile' {
        Mock Test-Path { $false }
        $report = & $script:HealthScript -Profile RustCacheDevDrive -PassThru -DryRun
        $report.Status | Should -Be Fail
        $report.Failures | Should -Contain 'Missing expected VHD: C:\DevDrives\RustCache.vhdx'
    }
    # Protects: the combined four-disk health contract.
    # Detects: combined profiles dropping missing default disks.
    # Needs: Pester and synthetic OS inventory; no disks, mounts or tasks.
    # Breadcrumb: Tools/Test-BootMountHealth.ps1 expected profile and missing-file checks.
    It 'retains missing default disks under All' {
        $report = & $script:HealthScript -Profile All -PassThru -DryRun
        $report.Status | Should -Be Fail
        @($report.Vhds).Count | Should -Be 4
        @($report.Failures | Where-Object { $_ -like 'Missing expected VHD:*' }).Count | Should -Be 3
    }
}
