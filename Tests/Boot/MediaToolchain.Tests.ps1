#Requires -Version 7.0
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }
BeforeAll {
    $script:MediaScript = Join-Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) 'Tools\Test-PcaiMediaToolchain.ps1'
    . $script:MediaScript -DryRun | Out-Null
    function New-ProbeResult {
        param([string]$Status = 'Succeeded', [string]$Stdout = '', [string]$Stderr = '', [int]$ExitCode = 0)
        [pscustomobject]@{ status = $Status; stdout = $Stdout; stderr = $Stderr; exitCode = $ExitCode
            timedOut = ($Status -eq 'TimedOut'); error = $null }
    }
}
Describe 'Media qualification boundaries' -Skip:(-not $IsWindows) {
    It 'parses and exposes help aliases' {
        $tokens = $null; $parseErrors = $null
        [void][System.Management.Automation.Language.Parser]::ParseFile($script:MediaScript, [ref]$tokens, [ref]$parseErrors)
        $parseErrors.Count | Should -Be 0
        foreach ($alias in @('h', '-help')) {
            $parameters = @{ $alias = $true }
            (& $script:MediaScript @parameters | Out-String) | Should -Match 'synthetic'
        }
    }
    It 'does no process, provider, discovery or output work in DryRun' {
        Mock Invoke-MediaProcess { throw 'Process forbidden' }
        Mock Get-MediaGpuInventory { throw 'CIM forbidden' }
        Mock Find-MediaExecutable { throw 'Discovery forbidden' }
        $result = Get-MediaToolchainReport -DryRun -ValidateIntelQsv
        $result.status | Should -Be 'DryRun'
        $result.intelSyntheticFrames | Should -Be 16
        foreach ($name in @('Invoke-MediaProcess', 'Get-MediaGpuInventory', 'Find-MediaExecutable')) {
            Should -Invoke $name -Times 0 -Exactly
        }
        $path = Join-Path $TestDrive 'dry.json'
        & $script:MediaScript -DryRun -OutputPath $path | Out-Null
        Test-Path -LiteralPath $path | Should -BeFalse
    }
    It 'rejects reserved, relative, alternate stream, existing and missing-parent outputs' {
        foreach ($leaf in @('$null', 'NUL.json', 'AUX', 'COM1.log')) {
            { Resolve-MediaReportPath (Join-Path $TestDrive $leaf) } | Should -Throw '*reserved*'
        }
        { Resolve-MediaReportPath '.\out.json' } | Should -Throw '*absolute*'
        { Resolve-MediaReportPath (Join-Path $TestDrive 'out.json:stream') } | Should -Throw '*alternate*'
        $existing = Join-Path $TestDrive 'existing.json'
        [System.IO.File]::WriteAllText($existing, 'retained')
        { Resolve-MediaReportPath $existing } | Should -Throw '*already exists*'
        [System.IO.File]::ReadAllText($existing) | Should -Be 'retained'
        { Resolve-MediaReportPath (Join-Path $TestDrive 'absent/out.json') } | Should -Throw '*parent*'
    }
    It 'preserves stderr and a real nonzero child exit code' {
        $result = Invoke-MediaProcess (Join-Path $PSHOME 'pwsh.exe') @('-NoProfile', '-Command', '[Console]::Error.WriteLine("fixture-error"); exit 7') 5
        $result.status | Should -Be 'Failed'
        $result.exitCode | Should -Be 7
        $result.stderr | Should -Match 'fixture-error'
        $result.timedOut | Should -BeFalse
    }
    It 'terminates a hung child and retains partial output' {
        $result = Invoke-MediaProcess (Join-Path $PSHOME 'pwsh.exe') @('-NoProfile', '-Command', '[Console]::Out.WriteLine("ready"); Start-Sleep -Seconds 30') 3
        $result.status | Should -Be 'TimedOut'
        $result.timedOut | Should -BeTrue
        $result.stdout | Should -Match 'ready'
        $result.elapsedMilliseconds | Should -BeLessThan 8000
    }
    It 'rejects a system interpreter instead of using ambient packages' {
        { Get-MediaToolchainReport -PythonPath (Join-Path $PSHOME 'pwsh.exe') } | Should -Throw '*isolated venv*'
    }
    It 'retains typed GPU provider failure as unknown' {
        Mock Get-CimInstance { throw [System.InvalidOperationException]::new('fixture CIM failure') }
        $gpu = Get-MediaGpuInventory
        $gpu.status | Should -Be 'Unknown'
        $gpu.error | Should -Be 'fixture CIM failure'
        $gpu.exceptionType | Should -Be 'System.InvalidOperationException'
        Should -Invoke Get-CimInstance -Times 1 -Exactly -ParameterFilter { $OperationTimeoutSec -eq 3 }
    }
}
Describe 'Media report semantics' -Skip:(-not $IsWindows) {
    BeforeEach {
        Mock Find-MediaExecutable { param($Name) if ($Name -eq 'ffmpeg.exe') { 'C:\fixture\ffmpeg.exe' } }
        Mock Get-MediaGpuInventory { [pscustomobject]@{ status = 'Available'; devices = @([pscustomobject]@{Name = 'Intel Iris Xe'}); error = $null } }
        Mock Invoke-MediaProcess {
            param($FilePath, $Arguments)
            if ($Arguments -contains '-hwaccels') { New-ProbeResult -Stdout "Hardware acceleration methods:`nqsv`ncuda`nd3d11va`n" }
            elseif ($Arguments -contains '-encoders') { New-ProbeResult -Stdout " V..... h264_qsv QSV`n V....D h264_nvenc NVIDIA`n A....D aac AAC" }
            else { New-ProbeResult -Stdout 'fixture version' }
        }
    }
    It 'separates compiled NVIDIA support from runtime and does not encode by default' {
        $result = Get-MediaToolchainReport
        $result.ffmpeg.compiledHardwareMethods | Should -Contain 'cuda'
        $result.ffmpeg.compiledEncoders | Should -Contain 'h264_nvenc'
        $result.intelQsv.status | Should -Be 'NotRequested'
        $result.nvidiaRuntime.status | Should -Be 'NotAttempted'
        Should -Invoke Invoke-MediaProcess -Times 3 -Exactly
        Should -Invoke Invoke-MediaProcess -Times 0 -Exactly -ParameterFilter { $Arguments -contains '-i' }
    }
    It 'retains missing FFmpeg without attempting a workload or capture' {
        Mock Find-MediaExecutable { $null } -ParameterFilter { $Name -eq 'ffmpeg.exe' }
        $result = Get-MediaToolchainReport
        $result.ffmpeg.status | Should -Be 'Missing'
        $result.ffmpeg.path | Should -BeNullOrEmpty
        $result.intelQsv.status | Should -Be 'NotRequested'
        $result.captureDeviceListing.status | Should -Be 'NotRequested'
        Should -Invoke Invoke-MediaProcess -Times 0 -Exactly
    }
    It 'marks requested QSV failure and retains its exit evidence' {
        Mock Invoke-MediaProcess { New-ProbeResult -Status 'Failed' -ExitCode 9 -Stderr 'QSV fixture error' } -ParameterFilter { $Arguments -contains 'lavfi' }
        $result = Get-MediaToolchainReport -ValidateIntelQsv
        $result.status | Should -Be 'CompletedWithIssues'
        $result.intelQsv.status | Should -Be 'Failed'
        $result.intelQsv.probe.exitCode | Should -Be 9
        $result.intelQsv.probe.stderr | Should -Match 'QSV fixture'
        Should -Invoke Invoke-MediaProcess -Times 1 -Exactly -ParameterFilter {
            $Arguments -contains '16' -and $Arguments -contains 'color=c=black:s=320x240:r=30' -and
            $Arguments[-2] -eq 'null' -and $Arguments[-1] -eq '-'
        }
    }
    It 'does not initialize QSV when GPU inventory is unknown or NVIDIA only' {
        Mock Get-MediaGpuInventory { [pscustomobject]@{status = 'Unknown'; devices = @(); error = 'provider error'} }
        (Get-MediaToolchainReport -ValidateIntelQsv).intelQsv.status | Should -Be 'UnknownGpuInventory'
        Mock Get-MediaGpuInventory { [pscustomobject]@{status = 'Available'; devices = @([pscustomobject]@{Name = 'NVIDIA RTX'}); error = $null} }
        (Get-MediaToolchainReport -ValidateIntelQsv).intelQsv.status | Should -Be 'NoIntelGpu'
        Should -Invoke Invoke-MediaProcess -Times 0 -Exactly -ParameterFilter { $Arguments -contains 'lavfi' }
    }
    It 'accepts expected DirectShow listing exit while preserving its actual code' {
        Mock Invoke-MediaProcess { New-ProbeResult -Status 'Failed' -ExitCode 1 -Stderr '[dshow] Fixture Camera (video)' } -ParameterFilter { $Arguments -contains 'dshow' }
        $result = Get-MediaToolchainReport -ListCaptureDevices
        $result.captureDeviceListing.status | Should -Be 'Listed'
        $result.captureDeviceListing.probe.exitCode | Should -Be 1
        Should -Invoke Invoke-MediaProcess -Times 1 -Exactly -ParameterFilter { $Arguments[-1] -eq 'dummy' }
    }
    It 'does not present a DirectShow enumeration error as a successful device listing' {
        Mock Invoke-MediaProcess { New-ProbeResult -Status 'Failed' -ExitCode 1 -Stderr 'Could not enumerate video devices' } -ParameterFilter { $Arguments -contains 'dshow' }
        $result = Get-MediaToolchainReport -ListCaptureDevices
        $result.captureDeviceListing.status | Should -Be 'NoDevicesOrProviderFailure'
        $result.status | Should -Be 'CompletedWithIssues'
    }
    It 'rejects a venv that exposes system packages or omits the isolation setting' -ForEach @(
        @{ Setting = 'include-system-site-packages = true' }
        @{ Setting = 'home = C:\Python312' }
    ) {
        $venv = Join-Path $TestDrive 'unsafe-venv'
        [void][System.IO.Directory]::CreateDirectory((Join-Path $venv 'Scripts'))
        [System.IO.File]::WriteAllText((Join-Path $venv 'pyvenv.cfg'), $Setting)
        $python = Join-Path $venv 'Scripts/python.exe'
        [System.IO.File]::WriteAllText($python, 'fixture not executed')
        { Get-MediaToolchainReport -PythonPath $python } | Should -Throw '*system site packages*'
        Should -Invoke Invoke-MediaProcess -Times 0 -Exactly
    }
    It 'preserves incomplete qualification when GStreamer is missing' {
        Mock Find-MediaExecutable { $null } -ParameterFilter { $Name -eq 'gst-inspect-1.0.exe' }
        (Get-MediaToolchainReport).status | Should -Be 'CompletedWithIssues'
    }
    It 'reports competing cv2 wheels as a conflict instead of an available CUDA backend' {
        $venv = Join-Path $TestDrive 'venv'
        [void][System.IO.Directory]::CreateDirectory((Join-Path $venv 'Scripts'))
        [System.IO.File]::WriteAllText((Join-Path $venv 'pyvenv.cfg'), 'include-system-site-packages = false')
        $python = Join-Path $venv 'Scripts/python.exe'
        [System.IO.File]::WriteAllText($python, 'fixture not executed')
        Mock Invoke-MediaProcess { New-ProbeResult -Stdout '{"opencvConflict":true,"opencvVariants":{"opencv-python":"5","opencv-python-headless":"5"},"cv2":null,"av":null,"errors":[]}' } -ParameterFilter { $FilePath -eq $python }
        $result = Get-MediaToolchainReport -PythonPath $python
        $result.python.status | Should -Be 'OpenCvConflict'
        $result.python.data.cv2 | Should -BeNullOrEmpty
        $result.status | Should -Be 'CompletedWithIssues'
    }
    It 'retains Python malformed output and module import errors' -ForEach @(
        @{ Payload = 'not json'; Expected = 'InvalidJson' }
        @{ Payload = '{"opencvConflict":false,"cv2":null,"av":null,"errors":[{"module":"cv2","type":"ImportError","message":"fixture DLL missing"}]}'; Expected = 'ImportFailed' }
        @{ Payload = '{"opencvConflict":false,"cv2":{"version":"fixture"},"av":null,"errors":[]}'; Expected = 'Incomplete' }
        @{ Payload = '{"opencvConflict":false,"cv2":{"version":"fixture","cudaDeviceCount":0},"av":{"version":"fixture"},"errors":[]}'; Expected = 'Inspected' }
    ) {
        $venv = Join-Path $TestDrive 'python-fixture'
        [void][System.IO.Directory]::CreateDirectory((Join-Path $venv 'Scripts'))
        [System.IO.File]::WriteAllText((Join-Path $venv 'pyvenv.cfg'), 'include-system-site-packages = false')
        $python = Join-Path $venv 'Scripts/python.exe'
        [System.IO.File]::WriteAllText($python, 'fixture not executed')
        Mock Invoke-MediaProcess { New-ProbeResult -Stdout $Payload } -ParameterFilter { $FilePath -eq $python }
        $result = Get-MediaToolchainReport -PythonPath $python
        $result.python.status | Should -Be $Expected
        $result.python.probe.stdout | Should -Be $Payload
    }
}
Describe 'Installed media smoke' -Tag 'Integration' -Skip:(-not $IsWindows) {
    It 'enumerates real binaries without any capture or runtime encode' {
        $result = & $script:MediaScript
        $result.schemaVersion | Should -Be 1
        if ($result.ffmpeg.status -eq 'Missing' -and [string]::IsNullOrWhiteSpace($result.ffmpeg.path)) {
            Set-ItResult -Skipped -Because 'Installed FFmpeg is required for real-binary enumeration; absence is covered by deterministic report tests.'
            return
        }
        $result.gpuInventory.status | Should -Be 'Available'
        $result.ffmpeg.status | Should -Be 'Enumerated'
        $result.ffmpeg.compiledEncoders | Should -Contain 'h264_qsv'
        $result.intelQsv.status | Should -Be 'NotRequested'
        $result.captureDeviceListing.status | Should -Be 'NotRequested'
    }
    It 'saves a fresh JSON report and refuses to overwrite it' {
        $path = Join-Path $TestDrive 'saved-report.json'
        $result = & $script:MediaScript -OutputPath $path
        $saved = [System.IO.File]::ReadAllText($path) | ConvertFrom-Json
        $saved.schemaVersion | Should -Be 1
        ([datetimeoffset]$saved.utc).UtcDateTime | Should -Be ([datetimeoffset]$result.utc).UtcDateTime
        $saved.ffmpeg.path | Should -Be $result.ffmpeg.path
        $before = [System.IO.File]::ReadAllText($path)
        { & $script:MediaScript -OutputPath $path } | Should -Throw '*already exists*'
        [System.IO.File]::ReadAllText($path) | Should -Be $before
    }
}
