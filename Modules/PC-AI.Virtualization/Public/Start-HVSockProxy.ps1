#Requires -Version 5.1
<#
.SYNOPSIS
    Starts configured proxies while preserving exact process custody.
#>
function Start-HVSockProxy {
    [CmdletBinding(SupportsShouldProcess)]
    [OutputType([PSCustomObject])]
    param(
        [string]$ConfigPath = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path 'Config\hvsock-proxy.conf'),
        [string]$StatePath = "$env:ProgramData\PC_AI\hvsock-proxy\state.json",
        [switch]$Force,
        [switch]$RegisterServices
    )
    $StatePath = Resolve-HVSockStatePath -Path $StatePath
    $stateDir = [IO.Path]::GetDirectoryName($StatePath)
    if ([IO.Directory]::Exists($stateDir)) {
        $recoveryFiles = [IO.Directory]::GetFiles($stateDir, "$([IO.Path]::GetFileName($StatePath)).pcai-recovery-*.json")
        if ($recoveryFiles.Count) { throw 'Pending proxy recovery custody must be reconciled explicitly before new launches.' }
    }
    if (-not (Test-Path -LiteralPath $ConfigPath -PathType Leaf)) { throw "HVSOCK proxy config not found: $ConfigPath" }
    if (Test-Path -LiteralPath $StatePath) {
        if (-not $Force) { throw 'Existing proxy custody must be reconciled before starting duplicates.' }
        if (-not $PSCmdlet.ShouldProcess($StatePath, 'Reconcile existing proxies and start replacements')) { return }
        $stop = Stop-HVSockProxy -StatePath $StatePath -Confirm:$false
        if ($stop.Unresolved.Count -or (Test-Path -LiteralPath $StatePath)) { throw 'Existing proxy custody remains unresolved; replacement refused.' }
    }
    $winsocat = Get-Command winsocat.exe -CommandType Application -ErrorAction SilentlyContinue
    if (-not $winsocat) { throw 'WinSocat not found. Run Install-HVSockProxy first.' }
    $executable = (Get-Item -LiteralPath $winsocat.Path -ErrorAction Stop).FullName
    $content = Get-Content -LiteralPath $ConfigPath -Raw -ErrorAction Stop
    $lines = if ([string]::IsNullOrWhiteSpace($content)) { @() } else { $content.Split("`n") | ForEach-Object { $_.Trim() } | Where-Object { $_ -and -not $_.StartsWith('#') } }
    $definitions = @(
        foreach ($line in $lines) {
            $parts = $line.Split(':')
            if ($parts.Count -lt 4) { throw 'Malformed proxy configuration; no processes launched.' }
            [pscustomobject]@{ Name = $parts[0]; ServiceId = $parts[1]; TcpTarget = "$($parts[2]):$($parts[3])" }
        }
    )
    if (-not $PSCmdlet.ShouldProcess($StatePath, 'Register requested services, launch proxies and publish custody')) {
        return [pscustomobject]@{ ConfigPath = $ConfigPath; StatePath = $StatePath; Count = 0; Proxies = @() }
    }
    if ($RegisterServices) { Register-HVSockServices -ConfigPath $ConfigPath -Force:$Force -ErrorAction Stop | Out-Null }
    $null = [IO.Directory]::CreateDirectory($stateDir)
    $entries = [Collections.Generic.List[object]]::new()
    $children = [Collections.Generic.List[object]]::new()
    try {
        foreach ($definition in $definitions) {
            $arguments = "HVSock-LISTEN:$($definition.ServiceId) TCP:$($definition.TcpTarget)"
            $process = Start-Process -FilePath $executable -ArgumentList $arguments -PassThru -WindowStyle Hidden -ErrorAction Stop
            # Register the strong reference before any identity getter can fail.
            $custody = [pscustomobject]@{ Process = $process; Pid = $null; ExecutablePath = $executable
                ProcessStartTimeUtcTicks = $null; HandleVerified = $false; Retain = $false }
            $children.Add($custody)
            $custody.Pid = $process.Id
            $handle = $process.Handle
            if ($handle -isnot [IntPtr] -or $handle -eq [IntPtr]::Zero -or $handle -eq [IntPtr]::new(-1)) { throw 'New proxy process handle could not be verified.' }
            $custody.HandleVerified = $true
            $actualExecutable = (Get-Item -LiteralPath $process.MainModule.FileName -ErrorAction Stop).FullName
            if (-not [string]::Equals($actualExecutable, $executable, [StringComparison]::OrdinalIgnoreCase)) { throw 'Launched proxy executable identity does not match.' }
            $custody.ProcessStartTimeUtcTicks = $process.StartTime.ToUniversalTime().Ticks
            $entries.Add([pscustomobject]@{ Name = $definition.Name; ServiceId = $definition.ServiceId; TcpTarget = $definition.TcpTarget
                Pid = $custody.Pid; ExecutablePath = $executable; ProcessStartTimeUtcTicks = $custody.ProcessStartTimeUtcTicks
                Command = "$executable $arguments"; Started = [DateTime]::UtcNow.ToString('o') })
        }
        $null = Set-HVSockCustodyState -Path $StatePath -Expected $null -Entries @($entries.ToArray())
    } catch {
        $originalError = $_
        $pending = [Collections.Generic.List[object]]::new()
        $retained = [Collections.Generic.List[Diagnostics.Process]]::new()
        foreach ($custody in $children) {
            try {
                if (-not $custody.HandleVerified) { throw 'Unverified new-child handle; automatic termination refused.' }
                if (-not $custody.Process.HasExited) { Stop-Process -InputObject $custody.Process -Force -ErrorAction Stop }
                if (-not $custody.Process.WaitForExit(5000)) { throw 'New proxy closure could not be confirmed.' }
            } catch {
                $custody.Retain = $true
                $retained.Add($custody.Process)
                $pending.Add([pscustomobject]@{ Name = 'RecoveredProxy'; ServiceId = $null; TcpTarget = $null; Started = [DateTime]::UtcNow.ToString('o')
                    Pid = $custody.Pid; ExecutablePath = $custody.ExecutablePath
                    ProcessStartTimeUtcTicks = $custody.ProcessStartTimeUtcTicks
                    MetadataIncomplete = (-not $custody.HandleVerified -or -not $custody.ProcessStartTimeUtcTicks)
                    RecoveryError = $_.Exception.Message })
            }
        }
        if ($pending.Count) {
            # Publish in-memory ownership before any fallible persistence step.
            # Caller owns closure/disposal of these exact unresolved references.
            $originalError.Exception.Data['ProxyOriginalErrorRecord'] = $originalError
            $originalError.Exception.Data['ProxyOriginalOperationException'] = $originalError.Exception
            $originalError.Exception.Data['ProxyRecoveryEntries'] = @($pending.ToArray())
            $originalError.Exception.Data['ProxyRecoveryProcessReferences'] = @($retained.ToArray())
            $recoveryPath = "$StatePath.pcai-recovery-$([guid]::NewGuid().ToString('N')).json"
            try {
                $null = Set-HVSockCustodyState -Path $recoveryPath -Expected $null -Entries @($pending.ToArray())
                $originalError.Exception.Data['ProxyRecoveryStatePath'] = $recoveryPath
            } catch {
                Write-Warning 'New proxy custody could not be persisted; retained entries require review.'
            }
        }
        throw $originalError
    } finally { foreach ($custody in $children) { if (-not $custody.Retain) { $custody.Process.Dispose() } } }
    return [pscustomobject]@{ ConfigPath = $ConfigPath; StatePath = $StatePath; Count = $entries.Count; Proxies = @($entries.ToArray()) }
}
