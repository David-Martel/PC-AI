#Requires -Version 5.1

<#
.SYNOPSIS
    PcaiInference — PowerShell FFI wrapper for the pcai-inference Rust library.

.DESCRIPTION
    Provides synchronous and asynchronous LLM inference from PowerShell by calling
    into the Rust inference engine via PcaiNative.dll ([PcaiNative.InferenceModule]).
    Supports both llama.cpp and mistral.rs backends with optional CUDA GPU offload.

    Exported functions (sync):
      Initialize-PcaiInference  - Load PcaiNative.dll and initialize the selected
                                  inference backend (llamacpp | mistralrs | auto).
                                  Must be called before any generate/model operation.
      Import-PcaiModel          - Load a configured model file (GGML/GGUF/SafeTensors) into the
                                  active backend with optional GPU layer offloading.
      Invoke-PcaiInference      - Run synchronous text completion for a prompt.
      Invoke-PcaiGenerate       - Alias for Invoke-PcaiInference.
      Stop-PcaiInference        - Shut down the active backend and free resources.
      Close-PcaiInference       - Alias for Stop-PcaiInference.
      Get-PcaiInferenceStatus   - Return current state: DllExists, BackendInitialized,
                                  ModelLoaded, CurrentBackend, DllPath.
                                  Mapped by FunctionGemma router as pcai_native_inference_status.
      Test-PcaiInference        - Probe whether the DLL can initialize a backend
                                  (optionally loading a model); returns $true/$false.
      Test-PcaiDllVersion       - Return file version metadata for pcai_inference.dll.

    Exported functions (async):
      Invoke-PcaiGenerateAsync  - Submit a prompt for async generation. Returns the
                                  completed text by default; use -NoWait to get a
                                  request ID and poll separately.
      Get-PcaiAsyncResult       - Poll or block on a pending async request by ID.
      Stop-PcaiGeneration       - Cancel a pending or running async request.

    Native acceleration (PcaiNative.dll — InferenceModule):
      All inference functions require bin\PcaiNative.dll built from
      Native\PcaiNative\. DLL search order:
        1. Config\llm-config.json nativeInference.dllSearchPaths
        2. bin\pcai_inference.dll (project root)
        3. bin\Release\ / bin\Debug\
        4. .pcai\build\artifacts\pcai-llamacpp\ or pcai-mistralrs\
        5. Native\pcai_core\pcai_inference\target\release\
        6. %USERPROFILE%\.local\bin\
      PcaiNative.dll is loaded from bin\ alongside pcai_inference.dll.

    GPU offload:
      Pass -GpuLayers to Import-PcaiModel (-1 = full GPU, 0 = CPU only).
      GPU assignment: RTX 2000 Ada (8 GB, SM 89) for inference,
      RTX 5060 Ti (16 GB, SM 120) for training (set via Config\llm-config.json).

    Dependencies:
      - PowerShell 5.1 or later
      - bin\PcaiNative.dll (C# P/Invoke wrapper, Native\PcaiNative\)
      - pcai_inference.dll (Rust inference engine, Native\pcai_core\pcai_inference\)
      - CUDA Toolkit 13.x (optional, for GPU offload)
      - Windows 10/11 x64
#>

#region Module Variables
$script:ModulePath = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.ScriptBlock.File }
$script:BackendInitialized = $false
$script:ModelLoaded = $false
$script:CurrentBackend = $null
$script:DllPath = $null
$script:DllExists = $false
#endregion

#region Internal Logic
function Add-EnvPath {
    param([string]$Path)
    if (-not $Path) { return }
    if (-not (Test-Path $Path)) { return }
    if ($env:PATH -notlike "*$Path*") {
        $env:PATH = "$Path;$env:PATH"
    }
}

function Get-PcaiProjectRoot {
    if ($env:PCAI_ROOT) {
        $root = (Resolve-Path -LiteralPath $env:PCAI_ROOT -ErrorAction Stop).ProviderPath
        if (-not (Test-Path -LiteralPath (Join-Path $root 'PC-AI.ps1') -PathType Leaf)) { throw 'PCAI_ROOT must select a PC-AI checkout containing PC-AI.ps1.' }
        return $root
    }
    $cursor = $script:ModulePath
    while ($cursor) {
        if (Test-Path -LiteralPath (Join-Path $cursor 'PC-AI.ps1') -PathType Leaf) { return $cursor }
        $cursor = Split-Path -Parent $cursor
    }
    if (Get-Command 'PC-AI.Common\Resolve-PcaiRepoRoot' -ErrorAction SilentlyContinue) {
        $root = PC-AI.Common\Resolve-PcaiRepoRoot -StartPath $script:ModulePath
        if ($root -and (Test-Path -LiteralPath (Join-Path $root 'PC-AI.ps1') -PathType Leaf)) { return $root }
    }
    throw 'PC-AI checkout unavailable. Set PCAI_ROOT for this machine or select PCAI_NATIVE_BUNDLE_ROOT for native loading.'
}

function Get-PcaiConfig {
    if ($env:PCAI_NATIVE_BUNDLE_ROOT) { return $null }
    $projectRoot = Get-PcaiProjectRoot
    $configPath = Join-Path $projectRoot 'Config\llm-config.json'
    if (-not (Test-Path $configPath)) { return $null }

    try {
        return (Get-Content $configPath -Raw | ConvertFrom-Json)
    } catch {
        Write-Verbose "Failed to parse ${configPath}: $_"
        return $null
    }
}

function Get-PcaiNativeProviderConfig {
    $config = Get-PcaiConfig
    if ($config -and $config.providers -and $config.providers.'pcai-native') {
        return $config.providers.'pcai-native'
    }
    return $null
}

function Resolve-PcaiModelPathFromConfig {
    $projectRoot = Get-PcaiProjectRoot
    $nativeCfg = Get-PcaiNativeProviderConfig
    $configuredPath = if ($nativeCfg) { $nativeCfg.modelPath } else { $null }
    if (-not $configuredPath) { return $null }
    if ([System.IO.Path]::IsPathRooted([string]$configuredPath)) { return [string]$configuredPath }
    return (Join-Path $projectRoot ([string]$configuredPath))
}

function Get-PcaiCudaCapability {
    try {
        $cap = nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>$null | Select-Object -First 1
        if (-not $cap) { return $null }
        $raw = ([string]$cap).Trim()
        if (-not $raw) { return $null }
        if ($raw -match '^(\d+)\.(\d+)$') {
            return ([int]$Matches[1] * 10 + [int]$Matches[2])
        }
        if ($raw -match '^(\d+)$') {
            return [int]$Matches[1]
        }
    } catch {
        Write-Verbose "Unable to detect CUDA capability via nvidia-smi: $_"
    }
    return $null
}

function Resolve-PcaiRuntimeVariantDll {
    param(
        [Parameter(Mandatory)][AllowNull()] $Config,
        [Parameter(Mandatory)][string]$ProjectRoot
    )

    $pending = Get-Variable -Name PcaiRuntimeVariantUnresolved -Scope Script -ErrorAction SilentlyContinue
    if ($null -ne $pending -and $pending.Value.Count -gt 0) { throw 'Runtime activation has retained unresolved resources; original-object recovery is required.' }
    function Save-RuntimeUnresolved($State, $Cause) {
        $existing = Get-Variable -Name PcaiRuntimeVariantUnresolved -Scope Script -ErrorAction SilentlyContinue
        if ($null -eq $existing) {
            Set-Variable -Name PcaiRuntimeVariantUnresolved -Scope Script -Value ([Collections.Generic.List[object]]::new())
        }
        $script:PcaiRuntimeVariantUnresolved.Add($State)
        $Cause.Exception.Data['RuntimeActivationUnresolved'] = $true
    }
    function Get-RuntimeOptionalValue($Value, [string]$Name) {
        if ($null -eq $Value) { return $null }
        if ($Value -is [Collections.IDictionary]) {
            $matchingKeys = @(foreach ($key in $Value.PSBase.Keys) {
                if ($key -is [string] -and [string]::Equals($key, $Name, [StringComparison]::OrdinalIgnoreCase)) { $key }
            })
            if ($matchingKeys.Count -gt 1) { throw "Ambiguous runtime configuration key: $Name" }
            if ($matchingKeys.Count -eq 1) { return ,$Value[$matchingKeys[0]] }
            return $null
        }
        $property = $Value.PSObject.Properties[$Name]
        if ($null -ne $property) { return ,$property.Value }
        return $null
    }
    function Assert-RuntimeOrdinaryPath([string]$Path, [bool]$RequireLeaf = $false) {
        $item = Get-Item -LiteralPath $Path -Force -ErrorAction SilentlyContinue
        if ($RequireLeaf -and ($null -eq $item -or $item -isnot [IO.FileInfo])) { throw "Runtime DLL is not a file: $Path" }
        $ancestor = [IO.Path]::GetDirectoryName([IO.Path]::GetFullPath($Path))
        while ($null -eq $item -and $ancestor) {
            $item = Get-Item -LiteralPath $ancestor -Force -ErrorAction SilentlyContinue
            $ancestor = [IO.Path]::GetDirectoryName($ancestor)
        }
        while ($null -ne $item) {
            if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { throw "Runtime path contains a reparse point: $Path" }
            $item = if ($item -is [IO.FileInfo]) { $item.Directory } else { $item.Parent }
        }
    }
    function Get-RuntimeStreamHash([IO.Stream]$Stream) {
        $Stream.Position = 0
        $algorithm = [Security.Cryptography.SHA256]::Create()
        $hashFailure = $null
        $hash = $null
        try { $hash = [Convert]::ToHexString($algorithm.ComputeHash($Stream)) }
        catch { $hashFailure = $_ }
        finally {
            try { $algorithm.Dispose() } catch {
                $disposeFailure = $_
                if ($null -eq $hashFailure) { $hashFailure = $disposeFailure }
                Save-RuntimeUnresolved -State ([pscustomobject]@{Kind='HashAlgorithm'; Original=$algorithm; FirstFailure=$hashFailure; DisposeFailure=$disposeFailure}) -Cause $hashFailure
            }
        }
        if ($null -ne $hashFailure) { throw $hashFailure }
        return $hash
    }
    function Install-RuntimePair([string]$SourcePath, [string]$ExpectedHash) {
        $canonical = [IO.Path]::GetFullPath((Join-Path $ProjectRoot 'bin/pcai_inference.dll'))
        $alias = [IO.Path]::GetFullPath((Join-Path $ProjectRoot 'bin/pcai_inference_lib.dll'))
        $directory = [IO.Path]::GetDirectoryName($canonical)
        Assert-RuntimeOrdinaryPath -Path $SourcePath -RequireLeaf $true
        Assert-RuntimeOrdinaryPath -Path ([IO.Path]::GetDirectoryName($SourcePath))
        Assert-RuntimeOrdinaryPath -Path $ProjectRoot
        Assert-RuntimeOrdinaryPath -Path $directory
        [void][IO.Directory]::CreateDirectory($directory)
        $records = [Collections.Generic.List[object]]::new()
        $ownedTemps = [Collections.Generic.List[string]]::new()
        $secondary = [Collections.Generic.List[object]]::new()
        $firstFailure = $null
        $sourceStream = $null
        $snapshotStream = $null
        $sourceHandle = $null
        $snapshotHandle = $null
        $snapshotPath = Join-Path $directory ('.pcai-runtime-stage-' + [Guid]::NewGuid().ToString('N') + '.tmp')
        $success = $false
        try {
            # Verify the source first; acquire both destination locks before any mutation.
            $sourceStream = [IO.File]::Open($SourcePath, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::Read)
            $sourceHandle = $sourceStream.SafeFileHandle
            $sourceHash = Get-RuntimeStreamHash $sourceStream
            if ($ExpectedHash -and $sourceHash -cne $ExpectedHash) { throw "SHA256 hash mismatch for runtime variant: $SourcePath" }
            try {
                $sourceStream.Dispose()
                if (-not $sourceHandle.IsClosed) { throw 'Original source stream handle did not close.' }
            } catch { $_.Exception.Data['RuntimeActivationSettlementFailure'] = $true; throw }
            $sourceStream = $null

            # Both existing destination locks are acquired before the first canonical mutation.
            foreach ($destination in @($canonical, $alias)) {
                Assert-RuntimeOrdinaryPath -Path $destination
                $record = [pscustomobject]@{ Path=$destination; Stream=$null; Handle=$null; Existed=(Test-Path -LiteralPath $destination); Backup=$null; BackupHandle=$null; BackupPath=$null; OldHash=$null; Touched=$false; Closed=$false }
                $records.Add($record)
                if ($record.Existed) {
                    Assert-RuntimeOrdinaryPath -Path $destination -RequireLeaf $true
                    $record.Stream = [IO.File]::Open($destination, [IO.FileMode]::Open, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
                    $record.Handle = $record.Stream.SafeFileHandle
                }
            }
            foreach ($record in $records) {
                if ($record.Existed) {
                    $record.OldHash = Get-RuntimeStreamHash $record.Stream
                    if ($record.Path.Equals([IO.Path]::GetFullPath($SourcePath), [StringComparison]::OrdinalIgnoreCase) -and $record.OldHash -cne $sourceHash) { throw 'Canonical source changed before destination lock.' }
                }
            }
            # An already matching pair requires no temporary file, backup or destination write.
            $needsActivation = @($records | Where-Object { -not $_.Existed -or $_.OldHash -cne $sourceHash }).Count -gt 0
            if ($needsActivation) {
                $snapshotStream = [IO.FileStream]::new($snapshotPath, [IO.FileMode]::CreateNew, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None, 4096, [IO.FileOptions]::DeleteOnClose)
                $snapshotHandle = $snapshotStream.SafeFileHandle
                $ownedTemps.Add($snapshotPath)
                $sourceRecords = @($records | Where-Object { $_.Path.Equals([IO.Path]::GetFullPath($SourcePath), [StringComparison]::OrdinalIgnoreCase) })
                if ($sourceRecords.Count -eq 1) { $snapshotSource = $sourceRecords[0].Stream }
                else {
                    $sourceStream = [IO.File]::Open($SourcePath, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::Read)
                    $sourceHandle = $sourceStream.SafeFileHandle
                    $snapshotSource = $sourceStream
                }
                if ((Get-RuntimeStreamHash $snapshotSource) -cne $sourceHash) { throw 'Runtime source changed before staging.' }
                $snapshotSource.Position = 0
                $snapshotSource.CopyTo($snapshotStream)
                $snapshotStream.Flush($true)
                if ((Get-RuntimeStreamHash $snapshotStream) -cne $sourceHash) { throw 'Runtime source snapshot changed bytes.' }
            foreach ($record in $records) {
                if ($record.Existed) {
                    if ($record.OldHash -ceq $sourceHash) { continue }
                    $backupPath = Join-Path $directory ('.pcai-runtime-backup-' + [Guid]::NewGuid().ToString('N') + '.tmp')
                    $record.BackupPath = $backupPath
                    $record.Backup = [IO.File]::Open($backupPath, [IO.FileMode]::CreateNew, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
                    $record.BackupHandle = $record.Backup.SafeFileHandle
                    $ownedTemps.Add($backupPath)
                    $record.Stream.Position = 0
                    $record.Stream.CopyTo($record.Backup)
                    $record.Backup.Flush($true)
                    if ((Get-RuntimeStreamHash $record.Backup) -cne $record.OldHash) { throw 'Runtime rollback snapshot changed bytes.' }
                }
            }
            foreach ($record in $records) {
                if (-not $record.Existed) {
                    # CreateNew refuses a destination created by another writer after preflight.
                    $record.Stream = [IO.File]::Open($record.Path, [IO.FileMode]::CreateNew, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
                    $record.Handle = $record.Stream.SafeFileHandle
                    $record.Touched = $true
                }
            }
            foreach ($record in $records) {
                if ($record.Existed -and $record.OldHash -ceq $sourceHash) { continue }
                $record.Touched = $true
                $record.Stream.Position = 0
                $record.Stream.SetLength(0)
                $snapshotStream.Position = 0
                $snapshotStream.CopyTo($record.Stream)
                $record.Stream.Flush($true)
                if ((Get-RuntimeStreamHash $record.Stream) -cne $sourceHash) { throw 'Canonical runtime pair failed byte verification.' }
            }
            }
            $success = $true
        } catch {
            $firstFailure = $_
            if (@($records | Where-Object Touched).Count) { $firstFailure.Exception.Data['RuntimeActivationMutated'] = $true }
            foreach ($record in $records) {
                if ($record.Existed -and $record.Touched -and $null -ne $record.Backup) {
                    try {
                        $record.Stream.Position = 0
                        $record.Stream.SetLength(0)
                        $record.Backup.Position = 0
                        $record.Backup.CopyTo($record.Stream)
                        $record.Stream.Flush($true)
                        if ((Get-RuntimeStreamHash $record.Stream) -cne $record.OldHash) { throw 'Runtime rollback byte verification failed.' }
                    } catch { $secondary.Add($_) }
                }
            }
        } finally {
            foreach ($resource in @([pscustomobject]@{Stream=$sourceStream;Handle=$sourceHandle}, [pscustomobject]@{Stream=$snapshotStream;Handle=$snapshotHandle})) {
                if ($null -ne $resource.Stream) {
                    try { $resource.Stream.Dispose(); if (-not $resource.Handle.IsClosed) { throw 'Original staging/source handle did not close.' } } catch { $secondary.Add($_) }
                }
            }
            foreach ($record in $records) {
                if ($null -ne $record.Backup) {
                    try { $record.Backup.Dispose(); if (-not $record.BackupHandle.IsClosed) { throw 'Original backup handle did not close.' } } catch { $secondary.Add($_) }
                }
                if ($null -ne $record.Stream) {
                    try { $record.Stream.Dispose(); $record.Closed = $record.Handle.IsClosed; if (-not $record.Closed) { throw 'Original destination handle did not close.' } } catch { $secondary.Add($_) }
                }
            }
            $unqualifiedNewTargets = @($records | Where-Object { -not $success -and -not $_.Existed -and $_.Touched })
            $unresolved = $secondary.Count -gt 0 -or $unqualifiedNewTargets.Count -gt 0 -or ($null -ne $firstFailure -and $firstFailure.Exception.Data.Contains('RuntimeActivationUnresolved'))
            foreach ($record in $records) {
                if ($null -ne $record.BackupPath) {
                    try { Write-Verbose "Original runtime bytes retained for operator recovery: $($record.BackupPath)" }
                    catch { $secondary.Add($_); $unresolved = $true }
                }
            }
            if ($unresolved) {
                if ($null -eq $firstFailure) { $firstFailure = $secondary[0] }
                Save-RuntimeUnresolved -State ([pscustomobject]@{
                    Kind='PairActivation'; Source=$sourceStream; SourceHandle=$sourceHandle; Snapshot=$snapshotStream; SnapshotHandle=$snapshotHandle
                    Records=$records; RetainedTemporaryPaths=$ownedTemps; UnqualifiedNewTargets=$unqualifiedNewTargets
                    FirstFailure=$firstFailure; SecondaryFailures=$secondary
                }) -Cause $firstFailure
            }
        }
        if ($null -ne $firstFailure) {
            if ($secondary.Count) { $firstFailure.Exception.Data['RuntimePairSecondaryErrors'] = @($secondary | ForEach-Object { $_.Exception.Message }) }
            throw $firstFailure
        }
        if ($secondary.Count) { throw $secondary[0] }
        return $canonical
    }

    $native = Get-RuntimeOptionalValue $Config 'nativeInference'
    $selector = Get-RuntimeOptionalValue $native 'runtimeBinarySelection'
    $variantValues = Get-RuntimeOptionalValue $selector 'variants'
    if (-not (Get-RuntimeOptionalValue $selector 'enabled') -or $null -eq $variantValues -or @($variantValues).Count -eq 0) { return $null }
    $capability = Get-PcaiCudaCapability
    $ranked = @()
    foreach ($variant in @($variantValues)) {
        $dllPath = Get-RuntimeOptionalValue $variant 'dllPath'
        if (-not $dllPath) { continue }
        $kindValue = Get-RuntimeOptionalValue $variant 'kind'
        $kind = if ($kindValue) { [string]$kindValue } else { 'cpu' }
        $minimum = Get-RuntimeOptionalValue $variant 'minCompute'
        $maximum = Get-RuntimeOptionalValue $variant 'maxCompute'
        $minCompute = if ($null -ne $minimum) { [int]$minimum } else { -1 }
        $maxCompute = if ($null -ne $maximum) { [int]$maximum } else { 9999 }
        if ($kind -eq 'cuda' -and ($null -eq $capability -or $capability -lt $minCompute -or $capability -gt $maxCompute)) { continue }
        $ranked += [pscustomobject]@{ Rank= $(if ($kind -eq 'cuda') { 1000 + $minCompute } else { $minCompute }); Variant=$variant }
    }
    $firstRefusal = $null
    foreach ($entry in @($ranked | Sort-Object -Property Rank -Descending)) {
        $selected = $entry.Variant
        $path = [string](Get-RuntimeOptionalValue $selected 'dllPath')
        if (-not [IO.Path]::IsPathRooted($path)) { $path = Join-Path $ProjectRoot $path }
        $path = [IO.Path]::GetFullPath($path)
        $downloadTemp = $null
        $reservation = $null
        $reservationHandle = $null
        $stopActivation = $false
        try {
            $digest = Get-RuntimeOptionalValue $selected 'sha256'
            $expectedHash = $null
            if ($null -ne $digest -and -not ($digest -is [string] -and $digest.Length -eq 0)) {
                if ($digest -isnot [string] -or $digest -cnotmatch '^[0-9a-fA-F]{64}$') { throw 'Runtime variant sha256 must contain exactly 64 hexadecimal characters.' }
                $expectedHash = $digest.ToUpperInvariant()
            }
            Assert-RuntimeOrdinaryPath -Path $path
            if (-not (Test-Path -LiteralPath $path)) {
                $url = Get-RuntimeOptionalValue $selected 'url'
                if (-not (Get-RuntimeOptionalValue $selector 'autoDownload') -or -not $url) { continue }
                $directory = [IO.Path]::GetDirectoryName($path)
                Assert-RuntimeOrdinaryPath -Path $directory
                Assert-RuntimeOrdinaryPath -Path $ProjectRoot
                [void][IO.Directory]::CreateDirectory($directory)
                $downloadTemp = Join-Path $directory ('.pcai-runtime-download-' + [Guid]::NewGuid().ToString('N') + '.tmp')
                # Reserve only our fresh ordinary file. Transport failure never publishes its prefix.
                $reservation = [IO.File]::Open($downloadTemp, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
                $reservationHandle = $reservation.SafeFileHandle
                $reservation.Dispose()
                if (-not $reservationHandle.IsClosed) { throw 'Original download reservation did not close.' }
                Invoke-WebRequest -Uri ([string]$url) -OutFile $downloadTemp -UseBasicParsing -TimeoutSec 600 -ErrorAction Stop | Out-Null
                Assert-RuntimeOrdinaryPath -Path $downloadTemp -RequireLeaf $true
                $downloadHash = (Get-FileHash -LiteralPath $downloadTemp -Algorithm SHA256 -ErrorAction Stop).Hash
                if ($expectedHash -and $downloadHash -cne $expectedHash) { throw 'SHA256 hash mismatch for downloaded runtime variant.' }
                if (-not $expectedHash) { Write-Warning "No sha256 hash provided for variant '$(Get-RuntimeOptionalValue $selected 'name')'. Download proceeding without verification." }
                [IO.File]::Move($downloadTemp, $path, $false)
                $downloadTemp = $null
            }
            Assert-RuntimeOrdinaryPath -Path $path -RequireLeaf $true
            return Install-RuntimePair -SourcePath $path -ExpectedHash $expectedHash
        } catch {
            if ($null -eq $firstRefusal) { $firstRefusal = $_ }
            if ($_.Exception.Data.Contains('RuntimeActivationUnresolved') -or $_.Exception.Data.Contains('RuntimeActivationMutated') -or $_.Exception.Data.Contains('RuntimeActivationSettlementFailure')) { $stopActivation = $true }
            try { Write-Verbose "Runtime variant refused: $($_.Exception.Message)" }
            catch { $firstRefusal.Exception.Data['RuntimeVariantDiagnosticError'] = $_.Exception.Message; $stopActivation = $true }
        } finally {
            if ($null -ne $downloadTemp) {
                # Transport owns its write lifetime; never delete a released path by guess.
                Save-RuntimeUnresolved -State ([pscustomobject]@{Kind='UnqualifiedDownload'; Path=$downloadTemp; Reservation=$reservation; ReservationHandle=$reservationHandle; Reason='Transport or publication did not complete; retain partial/unqualified bytes.'; FirstFailure=$firstRefusal}) -Cause $firstRefusal
                $stopActivation = $true
            }
        }
        if ($stopActivation) { throw $firstRefusal }
    }
    # Integrity/staging refusal must not become ordinary canonical fallback in the caller.
    if ($null -ne $firstRefusal) { throw $firstRefusal }
    return $null
}

function Resolve-PcaiInferenceDll {
    param([string]$OverridePath)

    if ($env:PCAI_NATIVE_BUNDLE_ROOT) {
        $bundle = Resolve-Path -LiteralPath $env:PCAI_NATIVE_BUNDLE_ROOT -ErrorAction Stop
        if ($bundle.Provider.Name -ne 'FileSystem' -or -not (Test-Path -LiteralPath $bundle.ProviderPath -PathType Container)) { throw 'Native bundle must be a filesystem directory.' }
        foreach ($leaf in @('PcaiNative.dll', 'pcai_inference.dll')) {
            if (-not (Test-Path -LiteralPath (Join-Path $bundle.ProviderPath $leaf) -PathType Leaf)) { throw "Explicit native inference bundle lacks $leaf." }
        }
        $env:PCAI_NATIVE_BUNDLE_ROOT = [IO.Path]::GetFullPath($bundle.ProviderPath)
        $selected = Join-Path $env:PCAI_NATIVE_BUNDLE_ROOT 'pcai_inference.dll'
        if ($OverridePath -and -not [IO.Path]::GetFullPath((Resolve-Path -LiteralPath $OverridePath -ErrorAction Stop).ProviderPath).Equals($selected, [StringComparison]::OrdinalIgnoreCase)) {
            throw 'DllPath conflicts with PCAI_NATIVE_BUNDLE_ROOT.'
        }
        return $selected
    }
    if ($OverridePath) {
        if (Test-Path $OverridePath) {
            return (Resolve-Path $OverridePath).Path
        }
        return $null
    }

    $projectRoot = Get-PcaiProjectRoot
    $config = Get-PcaiConfig

    $runtimeVariant = Resolve-PcaiRuntimeVariantDll -Config $config -ProjectRoot $projectRoot
    if ($runtimeVariant) {
        return $runtimeVariant
    }

    $candidates = @()
    $native = if ($null -eq $config) { $null } elseif ($config -is [Collections.IDictionary]) { $config['nativeInference'] } else {
        $property = $config.PSObject.Properties['nativeInference']
        if ($null -ne $property) { $property.Value }
    }
    $searchPaths = if ($null -eq $native) { $null } elseif ($native -is [Collections.IDictionary]) { $native['dllSearchPaths'] } else {
        $property = $native.PSObject.Properties['dllSearchPaths']
        if ($null -ne $property) { $property.Value }
    }
    if ($searchPaths) {
        foreach ($path in $searchPaths) {
            if (-not $path) { continue }
            if ([System.IO.Path]::IsPathRooted($path)) {
                $candidates += $path
            } else {
                $candidates += (Join-Path $projectRoot $path)
            }
        }
    }

    $candidates += @(
        (Join-Path $projectRoot 'bin\pcai_inference.dll'),
        (Join-Path $projectRoot 'bin\Release\pcai_inference.dll'),
        (Join-Path $projectRoot 'bin\Debug\pcai_inference.dll'),
        (Join-Path $projectRoot '.pcai\build\artifacts\pcai-llamacpp\pcai_inference.dll'),
        (Join-Path $projectRoot '.pcai\build\artifacts\pcai-mistralrs\pcai_inference.dll'),
        (Join-Path $projectRoot 'Native\pcai_core\pcai_inference\target\release\pcai_inference.dll'),
        (Join-Path ([Environment]::GetFolderPath('UserProfile')) '.local\bin\pcai_inference.dll')
    ) | Where-Object { $_ }

    $libCandidates = @(
        (Join-Path $projectRoot 'bin\pcai_inference_lib.dll'),
        (Join-Path $projectRoot 'bin\Release\pcai_inference_lib.dll'),
        (Join-Path $projectRoot 'bin\Debug\pcai_inference_lib.dll'),
        (Join-Path $projectRoot '.pcai\build\artifacts\pcai-llamacpp\pcai_inference_lib.dll'),
        (Join-Path $projectRoot '.pcai\build\artifacts\pcai-mistralrs\pcai_inference_lib.dll'),
        (Join-Path $projectRoot 'Native\pcai_core\pcai_inference\target\release\pcai_inference_lib.dll'),
        (Join-Path ([Environment]::GetFolderPath('UserProfile')) '.local\bin\pcai_inference_lib.dll')
    ) | Where-Object { $_ }

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return (Resolve-Path $candidate).Path
        }
    }

    # Compatibility fallback: if only pcai_inference_lib.dll exists, create a local
    # alias as pcai_inference.dll for P/Invoke binding and return that path.
    foreach ($libCandidate in $libCandidates) {
        if (-not (Test-Path $libCandidate)) { continue }
        $dllAlias = Join-Path (Split-Path $libCandidate -Parent) 'pcai_inference.dll'
        if (-not (Test-Path $dllAlias)) {
            try { Copy-Item $libCandidate -Destination $dllAlias -Force } catch {}
        }
        if (Test-Path $dllAlias) {
            return (Resolve-Path $dllAlias).Path
        }
    }

    return $null
}

function Initialize-PcaiFFI {
    param([string]$DllPath)

    $resolvedDll = Resolve-PcaiInferenceDll -OverridePath $DllPath
    $script:DllPath = $resolvedDll
    $script:DllExists = $null -ne $resolvedDll -and (Test-Path $resolvedDll)

    # Explicit bundles select both halves before config, PATH or variant discovery.
    $nativeDll = if ($env:PCAI_NATIVE_BUNDLE_ROOT) {
        Join-Path $env:PCAI_NATIVE_BUNDLE_ROOT 'PcaiNative.dll'
    } else { Join-Path (Join-Path (Get-PcaiProjectRoot) 'bin') 'PcaiNative.dll' }
    if (Test-Path -LiteralPath $nativeDll -PathType Leaf) {
        try {
            # Match Assembly.Location even when the selected directory uses 8.3 names.
            $nativeDll = [IO.Path]::GetFullPath((Resolve-Path -LiteralPath $nativeDll -ErrorAction Stop).ProviderPath)
            $assembly = [AppDomain]::CurrentDomain.GetAssemblies() | Where-Object { $_.GetName().Name -eq 'PcaiNative' } | Select-Object -First 1
            if ($assembly -and -not [IO.Path]::GetFullPath($assembly.Location).Equals($nativeDll, [StringComparison]::OrdinalIgnoreCase)) {
                throw 'A different PcaiNative bridge is already loaded. Select the bundle in a fresh process.'
            }
            $actualPath = [IO.Path]::GetFullPath([Reflection.Assembly]::LoadFrom($nativeDll).Location)
            if (-not $actualPath.Equals($nativeDll, [StringComparison]::OrdinalIgnoreCase)) { throw 'Managed loading reused another PcaiNative bridge. Use a fresh process.' }
            if ($script:DllExists) { Add-EnvPath (Split-Path $resolvedDll -Parent) }
            return $true
        } catch {
            Write-Warning "Failed to load $($nativeDll): $($_)"
        }
    }
    return $false
}
#endregion

#region Public Functions

function Initialize-PcaiInference {
    [CmdletBinding()]
    param(
        [Parameter()]
        [ValidateSet('auto', 'llamacpp', 'mistralrs')]
        [string]$Backend = 'auto',

        [Parameter()]
        [string]$DllPath
    )

    $config = Get-PcaiConfig
    $nativeCfg = Get-PcaiNativeProviderConfig
    $configuredBackend = if ($nativeCfg) { $nativeCfg.backend } else { $null }
    $backendChoice = if ($Backend -eq 'auto') {
        if ($configuredBackend) { [string]$configuredBackend } else { 'mistralrs' }
    } else {
        $Backend
    }
    $backendName = $backendChoice
    if ($backendName -eq 'llamacpp') {
        $backendName = 'llama_cpp'
    }

    if (-not (Initialize-PcaiFFI -DllPath $DllPath)) {
        throw 'PcaiNative.dll not found in bin. Please run build.ps1 first.'
    }

    if (-not $script:DllExists) {
        throw "DLL not found: pcai_inference.dll. Update Config/llm-config.json nativeInference.dllSearchPaths or build the native backend."
    }

    Write-Verbose "Initializing backend: $backendName"

    try {
        $result = [PcaiNative.InferenceModule]::pcai_init($backendName)
        if ($result -ne 0) {
            $nativeError = [PcaiNative.InferenceModule]::GetLastError()
            throw "Failed to initialize backend '$backendName': $nativeError"
        }

        $script:BackendInitialized = $true
        $script:CurrentBackend = $backendName
        Write-Verbose "Backend initialized successfully: $backendName"
        return [PSCustomObject]@{
            Success = $true
            Backend = $backendName
            DllPath = $script:DllPath
        }
    } catch {
        $script:BackendInitialized = $false
        throw "Backend initialization failed: $_"
    }
}

function Import-PcaiModel {
    [CmdletBinding()]
    param(
        [Parameter()]
        [string]$ModelPath,

        [Parameter()]
        [int]$GpuLayers = -1
    )

    if (-not $script:BackendInitialized) {
        throw 'Backend not initialized. Call Initialize-PcaiInference first.'
    }

    if (-not $ModelPath) {
        $ModelPath = Resolve-PcaiModelPathFromConfig
    }
    if (-not $PSBoundParameters.ContainsKey('GpuLayers')) {
        $nativeCfg = Get-PcaiNativeProviderConfig
        $configuredGpuLayers = if ($nativeCfg) { $nativeCfg.gpuLayers } else { $null }
        if ($null -ne $configuredGpuLayers) {
            $GpuLayers = [int]$configuredGpuLayers
        }
    }

    if (-not $ModelPath) {
        throw 'Model path not provided and providers.pcai-native.modelPath is not set in Config\llm-config.json.'
    }
    if (-not (Test-Path $ModelPath)) {
        throw "Model file not found: $ModelPath"
    }

    Write-Verbose "Loading model: $ModelPath"

    # ── Preflight GPU readiness check ───────────────────────────────────────
    if (Get-Command Test-PcaiGpuReadiness -ErrorAction SilentlyContinue) {
        try {
            $preflight = Test-PcaiGpuReadiness -RequiredMB 2000
            if ($preflight.Verdict -eq 'fail') {
                Write-Warning "GPU preflight failed: $($preflight.Reason)"
                foreach ($gpu in $preflight.Gpus) {
                    Write-Warning "  GPU$($gpu.index) ($($gpu.name)): $($gpu.free_mb)MB free"
                }
            } elseif ($preflight.Verdict -eq 'warn') {
                Write-Warning "GPU preflight warning: $($preflight.Reason)"
            } else {
                Write-Verbose "GPU preflight passed: $($preflight.Reason)"
            }
        } catch {
            Write-Verbose "GPU preflight check skipped (error): $_"
        }
    }

    try {
        $result = [PcaiNative.InferenceModule]::pcai_load_model($ModelPath, $GpuLayers)
        if ($result -ne 0) {
            $nativeError = [PcaiNative.InferenceModule]::GetLastError()
            throw "Failed to load model: $nativeError"
        }

        $script:ModelLoaded = $true
        Write-Verbose 'Model loaded successfully'
        return [PSCustomObject]@{
            Success   = $true
            ModelPath = $ModelPath
        }
    } catch {
        $script:ModelLoaded = $false
        throw "Model loading failed: $_"
    }
}

function Invoke-PcaiInference {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory, Position = 0)]
        [string]$Prompt,

        [Parameter()]
        [uint32]$MaxTokens = 512,

        [Parameter()]
        [ValidateRange(0.0, 2.0)]
        [float]$Temperature = 0.7
    )

    if (-not $script:ModelLoaded) {
        throw 'Model not loaded. Call Import-PcaiModel first.'
    }

    if (-not $PSBoundParameters.ContainsKey('MaxTokens') -or -not $PSBoundParameters.ContainsKey('Temperature')) {
        $nativeCfg = Get-PcaiNativeProviderConfig
        if (-not $PSBoundParameters.ContainsKey('MaxTokens') -and $nativeCfg -and $null -ne $nativeCfg.defaultMaxTokens) {
            $MaxTokens = [uint32]$nativeCfg.defaultMaxTokens
        }
        if (-not $PSBoundParameters.ContainsKey('Temperature') -and $nativeCfg -and $null -ne $nativeCfg.defaultTemperature) {
            $Temperature = [float]$nativeCfg.defaultTemperature
        }
    }

    try {
        $result = [PcaiNative.InferenceModule]::Generate($Prompt, $MaxTokens, $Temperature)
        if ($null -eq $result) {
            $nativeError = [PcaiNative.InferenceModule]::GetLastError()
            throw "Generation failed: $nativeError"
        }
        return $result
    } catch {
        throw "Inference error: $_"
    }
}

function Invoke-PcaiGenerate {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory, Position = 0)]
        [string]$Prompt,

        [Parameter()]
        [uint32]$MaxTokens = 512,

        [Parameter()]
        [ValidateRange(0.0, 2.0)]
        [float]$Temperature = 0.7
    )

    return Invoke-PcaiInference -Prompt $Prompt -MaxTokens $MaxTokens -Temperature $Temperature
}

function Invoke-PcaiGenerateAsync {
    <#
    .SYNOPSIS
        Starts an asynchronous inference request.
    .DESCRIPTION
        Submits a prompt for async generation. By default, polls until complete
        and returns the result. Use -NoWait to get the request ID immediately
        for manual polling with Get-PcaiAsyncResult.
    .PARAMETER Prompt
        The input prompt text.
    .PARAMETER MaxTokens
        Maximum tokens to generate.
    .PARAMETER Temperature
        Sampling temperature (0.0-2.0).
    .PARAMETER NoWait
        Return the request ID immediately without waiting for completion.
    .PARAMETER PollIntervalMs
        Milliseconds between poll attempts when waiting.
    .PARAMETER TimeoutSeconds
        Maximum seconds to wait before cancelling. 0 = no timeout.
    .EXAMPLE
        $result = Invoke-PcaiGenerateAsync -Prompt "Hello world"
    .EXAMPLE
        $id = Invoke-PcaiGenerateAsync -Prompt "Hello" -NoWait
        # ... do other work ...
        $result = Get-PcaiAsyncResult -RequestId $id -Wait
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory, Position = 0)]
        [string]$Prompt,

        [Parameter()]
        [uint32]$MaxTokens = 512,

        [Parameter()]
        [ValidateRange(0.0, 2.0)]
        [float]$Temperature = 0.7,

        [Parameter()]
        [switch]$NoWait,

        [Parameter()]
        [int]$PollIntervalMs = 50,

        [Parameter()]
        [int]$TimeoutSeconds = 0
    )

    if (-not $script:ModelLoaded) {
        throw 'Model not loaded. Call Import-PcaiModel first.'
    }

    if (-not $PSBoundParameters.ContainsKey('MaxTokens') -or -not $PSBoundParameters.ContainsKey('Temperature')) {
        $nativeCfg = Get-PcaiNativeProviderConfig
        if (-not $PSBoundParameters.ContainsKey('MaxTokens') -and $nativeCfg -and $null -ne $nativeCfg.defaultMaxTokens) {
            $MaxTokens = [uint32]$nativeCfg.defaultMaxTokens
        }
        if (-not $PSBoundParameters.ContainsKey('Temperature') -and $nativeCfg -and $null -ne $nativeCfg.defaultTemperature) {
            $Temperature = [float]$nativeCfg.defaultTemperature
        }
    }

    try {
        $requestId = [PcaiNative.InferenceModule]::pcai_generate_async($Prompt, $MaxTokens, $Temperature)
        if ($requestId -lt 0) {
            $nativeError = [PcaiNative.InferenceModule]::GetLastError()
            throw "Failed to start async generation: $nativeError"
        }

        if ($NoWait) {
            return [PSCustomObject]@{
                RequestId = $requestId
                Status    = 'Pending'
            }
        }

        # Poll until complete
        $deadline = if ($TimeoutSeconds -gt 0) { (Get-Date).AddSeconds($TimeoutSeconds) } else { $null }

        while ($true) {
            $poll = [PcaiNative.InferenceModule]::PollResult($requestId)
            $status = $poll.Item1
            $text = $poll.Item2

            switch ($status) {
                ([PcaiNative.InferenceModule+AsyncRequestStatus]::Complete) {
                    return $text
                }
                ([PcaiNative.InferenceModule+AsyncRequestStatus]::Failed) {
                    throw "Async generation failed: $text"
                }
                ([PcaiNative.InferenceModule+AsyncRequestStatus]::Cancelled) {
                    Write-Warning 'Async request was cancelled.'
                    return $null
                }
                ([PcaiNative.InferenceModule+AsyncRequestStatus]::Unknown) {
                    throw "Unknown request ID: $requestId"
                }
                default {
                    if ($deadline -and (Get-Date) -gt $deadline) {
                        [PcaiNative.InferenceModule]::CancelRequest($requestId) | Out-Null
                        throw "Async generation timed out after $TimeoutSeconds seconds."
                    }
                    Start-Sleep -Milliseconds $PollIntervalMs
                }
            }
        }
    } catch {
        throw "Async inference error: $_"
    }
}

function Get-PcaiAsyncResult {
    <#
    .SYNOPSIS
        Gets the result of an async inference request.
    .PARAMETER RequestId
        The request ID returned by Invoke-PcaiGenerateAsync -NoWait.
    .PARAMETER Wait
        Block until the request completes.
    .PARAMETER PollIntervalMs
        Milliseconds between poll attempts when waiting.
    .EXAMPLE
        $result = Get-PcaiAsyncResult -RequestId $id -Wait
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory, Position = 0)]
        [long]$RequestId,

        [Parameter()]
        [switch]$Wait,

        [Parameter()]
        [int]$PollIntervalMs = 50
    )

    do {
        $poll = [PcaiNative.InferenceModule]::PollResult($RequestId)
        $status = $poll.Item1
        $text = $poll.Item2

        $statusName = switch ($status) {
            ([PcaiNative.InferenceModule+AsyncRequestStatus]::Pending)   { 'Pending' }
            ([PcaiNative.InferenceModule+AsyncRequestStatus]::Running)   { 'Running' }
            ([PcaiNative.InferenceModule+AsyncRequestStatus]::Complete)  { 'Complete' }
            ([PcaiNative.InferenceModule+AsyncRequestStatus]::Failed)    { 'Failed' }
            ([PcaiNative.InferenceModule+AsyncRequestStatus]::Cancelled) { 'Cancelled' }
            default { 'Unknown' }
        }

        if ($status -in @(
            [PcaiNative.InferenceModule+AsyncRequestStatus]::Complete,
            [PcaiNative.InferenceModule+AsyncRequestStatus]::Failed,
            [PcaiNative.InferenceModule+AsyncRequestStatus]::Cancelled,
            [PcaiNative.InferenceModule+AsyncRequestStatus]::Unknown
        )) {
            return [PSCustomObject]@{
                RequestId = $RequestId
                Status    = $statusName
                Text      = $text
            }
        }

        if (-not $Wait) {
            return [PSCustomObject]@{
                RequestId = $RequestId
                Status    = $statusName
                Text      = $null
            }
        }

        Start-Sleep -Milliseconds $PollIntervalMs
    } while ($Wait)
}

function Stop-PcaiGeneration {
    <#
    .SYNOPSIS
        Cancels a pending or running async inference request.
    .PARAMETER RequestId
        The request ID to cancel.
    .EXAMPLE
        Stop-PcaiGeneration -RequestId $id
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory, Position = 0)]
        [long]$RequestId
    )

    $cancelled = [PcaiNative.InferenceModule]::CancelRequest($RequestId)
    return [PSCustomObject]@{
        RequestId = $RequestId
        Cancelled = $cancelled
    }
}

function Stop-PcaiInference {
    [CmdletBinding()]
    param()

    if ($script:BackendInitialized) {
        Write-Verbose 'Shutting down inference backend...'
        try {
            [PcaiNative.InferenceModule]::pcai_shutdown()
            $script:BackendInitialized = $false
            $script:ModelLoaded = $false
            $script:CurrentBackend = $null
        } catch {
            Write-Warning "Error during shutdown: $_"
        }
    }
}

function Close-PcaiInference {
    [CmdletBinding()]
    param()

    Stop-PcaiInference
}

function Get-PcaiInferenceStatus {
    [CmdletBinding()]
    param()

    if (-not $script:DllPath) {
        $script:DllPath = Resolve-PcaiInferenceDll
        $script:DllExists = $null -ne $script:DllPath -and (Test-Path $script:DllPath)
    }

    return [PSCustomObject]@{
        DllPath           = $script:DllPath
        DllExists         = $script:DllExists -and (Test-Path $script:DllPath)
        BackendInitialized = $script:BackendInitialized
        ModelLoaded        = $script:ModelLoaded
        CurrentBackend     = $script:CurrentBackend
    }
}

function Test-PcaiInference {
    [CmdletBinding()]
    param(
        [Parameter()]
        [ValidateSet('auto', 'llamacpp', 'mistralrs')]
        [string]$Backend = 'auto',

        [Parameter()]
        [string]$ModelPath,

        [Parameter()]
        [int]$GpuLayers = -1
    )

    try {
        $init = Initialize-PcaiInference -Backend $Backend
        if ($ModelPath) {
            $null = Import-PcaiModel -ModelPath $ModelPath -GpuLayers $GpuLayers
        }
        return $true
    } catch {
        Write-Verbose "Test-PcaiInference failed: $_"
        return $false
    } finally {
        try { Close-PcaiInference -ErrorAction SilentlyContinue } catch {}
    }
}

function Test-PcaiDllVersion {
    [CmdletBinding()]
    param(
        [Parameter()]
        [string]$DllPath
    )

    $resolved = Resolve-PcaiInferenceDll -OverridePath $DllPath
    if (-not $resolved) {
        return [PSCustomObject]@{
            Success = $false
            Message = 'pcai_inference.dll not found'
        }
    }

    $info = Get-Item $resolved -ErrorAction SilentlyContinue
    return [PSCustomObject]@{
        Success        = $true
        DllPath        = $resolved
        FileVersion    = $info.VersionInfo.FileVersion
        ProductVersion = $info.VersionInfo.ProductVersion
    }
}

#endregion

#region Module Cleanup
if ($MyInvocation.MyCommand.ScriptBlock.Module) {
    $MyInvocation.MyCommand.ScriptBlock.Module.OnRemove = {
        if ($script:BackendInitialized) {
            [PcaiNative.InferenceModule]::pcai_shutdown()
        }
    }
}
#endregion

#region Module Exports
Export-ModuleMember -Function @(
    'Initialize-PcaiInference',
    'Import-PcaiModel',
    'Invoke-PcaiInference',
    'Invoke-PcaiGenerate',
    'Invoke-PcaiGenerateAsync',
    'Get-PcaiAsyncResult',
    'Stop-PcaiGeneration',
    'Stop-PcaiInference',
    'Close-PcaiInference',
    'Get-PcaiInferenceStatus',
    'Test-PcaiInference',
    'Test-PcaiDllVersion'
)
#endregion
