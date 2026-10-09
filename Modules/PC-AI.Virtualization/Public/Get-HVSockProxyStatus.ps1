#Requires -Version 5.1
# Shared helpers are unexported and execute only when a command is invoked.
function Resolve-HVSockStatePath {
    param([Parameter(Mandatory)][string]$Path)
    # Validate the spelling before provider resolution can map NUL to a device path.
    foreach ($component in $Path.Split([char[]]'\/')) {
        $leaf = $component.TrimEnd([char[]]'. ')
        if ($leaf -eq '$null' -or $leaf -match '^(?i:AUX|CON|NUL|PRN|COM[1-9]|LPT[1-9])(?:\..*)?$') { throw 'Reserved proxy state path refused.' }
    }
    $provider = $null; $drive = $null
    $resolved = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Path, [ref]$provider, [ref]$drive)
    if ($provider.Name -ne 'FileSystem') { throw 'Proxy state requires a FileSystem path.' }
    $resolved = [IO.Path]::GetFullPath($resolved)
    foreach ($component in $resolved.Substring([IO.Path]::GetPathRoot($resolved).Length).Split([char[]]'\/')) {
        $leaf = $component.TrimEnd([char[]]'. ')
        if ($leaf -eq '$null' -or $leaf -match '^(?i:AUX|CON|NUL|PRN|COM[1-9]|LPT[1-9])(?:\..*)?$') { throw 'Reserved proxy state path refused.' }
        if ($component.IndexOfAny([IO.Path]::GetInvalidFileNameChars()) -ge 0) { throw 'Invalid proxy state path refused.' }
    }
    $ancestor = $resolved
    while ($ancestor) {
        if (Test-Path -LiteralPath $ancestor) {
            $item = Get-Item -LiteralPath $ancestor -Force -ErrorAction Stop
            if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Linked proxy state path refused.' }
        }
        $ancestor = [IO.Path]::GetDirectoryName($ancestor)
    }
    return $resolved
}

function Get-HVSockStateIdentity {
    param([Parameter(Mandatory)][IO.FileStream]$Stream)
    if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT) { throw 'Proxy state publication requires supported Windows file identity.' }
    if (-not ('Pcai.Proxy.FileIdentityV1' -as [type])) {
        Add-Type -TypeDefinition @'
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;
namespace Pcai.Proxy {
    public static class FileIdentityV1 {
        public static int ContractVersion { get { return 1; } }
        [StructLayout(LayoutKind.Sequential)] struct Identity { public ulong Volume; public Guid FileId; }
        [DllImport("kernel32.dll", SetLastError=true)]
        static extern bool GetFileInformationByHandleEx(SafeFileHandle handle, int kind, out Identity value, uint size);
        public static string Read(SafeFileHandle handle) {
            Identity value;
            if (!GetFileInformationByHandleEx(handle, 18, out value, 24)) throw new Win32Exception(Marshal.GetLastWin32Error());
            return value.Volume.ToString("X16") + ":" + value.FileId.ToString("N");
        }
    }
}
'@ -ErrorAction Stop
    }
    if ([Pcai.Proxy.FileIdentityV1]::ContractVersion -ne 1) { throw 'Incompatible loaded proxy file identity helper.' }
    return [Pcai.Proxy.FileIdentityV1]::Read($Stream.SafeFileHandle)
}

function Get-HVSockStateSnapshot {
    param([Parameter(Mandatory)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $null }
    $stream = [IO.File]::Open($Path, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete)
    try {
        $identity = Get-HVSockStateIdentity -Stream $stream
        $memory = [IO.MemoryStream]::new()
        try { $stream.CopyTo($memory); $bytes = $memory.ToArray() } finally { $memory.Dispose() }
        $sha = [Security.Cryptography.SHA256]::Create()
        try { $hash = [BitConverter]::ToString($sha.ComputeHash($bytes)).Replace('-', '') } finally { $sha.Dispose() }
        return [pscustomobject]@{ Identity = $identity; Hash = $hash; Bytes = $bytes }
    } finally { $stream.Dispose() }
}

function Test-HVSockStateSnapshot {
    param([string]$Path, $Expected)
    $actual = Get-HVSockStateSnapshot -Path $Path
    if (-not $Expected) { return -not $actual }
    return $actual -and $actual.Identity -ceq $Expected.Identity -and $actual.Hash -ceq $Expected.Hash
}

function Set-HVSockCustodyState {
    param([string]$Path, $Expected, [object[]]$Entries)
    if (-not (Test-HVSockStateSnapshot -Path $Path -Expected $Expected)) { throw 'Proxy state changed concurrently; original custody retained.' }
    $staged = $null; $displaced = $null
    try {
        if ($Entries.Count) {
            $staged = "$Path.pcai-stage-$([guid]::NewGuid().ToString('N'))"
            $json = ConvertTo-Json -InputObject @($Entries) -Depth 8 -WarningAction Stop
            $stageStream = [IO.File]::Open($staged, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
            try {
                $encoded = [Text.UTF8Encoding]::new($false).GetBytes($json)
                $stageStream.Write($encoded, 0, $encoded.Length)
                $stageStream.Flush($true)
            } finally { $stageStream.Dispose() }
        }
        # Boundary and displaced-object checks detect bounded races, not arbitrary-writer CAS.
        if (-not (Test-HVSockStateSnapshot -Path $Path -Expected $Expected)) { throw 'Proxy state changed before publication.' }
        if ($Expected) {
            $displaced = "$Path.pcai-previous-$([guid]::NewGuid().ToString('N'))"
            [IO.File]::Move($Path, $displaced)
            if (-not (Test-HVSockStateSnapshot -Path $displaced -Expected $Expected)) {
                if (-not (Test-Path -LiteralPath $Path)) { [IO.File]::Move($displaced, $Path) }
                throw 'Displaced proxy state identity changed; recovery requires review.'
            }
        }
        if ($staged) { [IO.File]::Move($staged, $Path); $staged = $null }
        return $displaced
    } catch {
        if ($displaced -and (Test-Path -LiteralPath $displaced) -and -not (Test-Path -LiteralPath $Path)) {
            try { [IO.File]::Copy($displaced, $Path, $false) } catch { Write-Warning 'Proxy state recovery requires review; displaced custody retained.' }
        }
        throw
    } finally {
        if ($staged -and (Test-Path -LiteralPath $staged)) { Remove-Item -LiteralPath $staged -ErrorAction SilentlyContinue }
    }
}

function Get-HVSockOwnedProcess {
    param([Parameter(Mandatory)]$Entry)
    $process = $null
    try {
        $processId = 0; $ticks = [long]0
        if (-not [int]::TryParse([string]$Entry.Pid, [ref]$processId) -or $processId -le 0 -or
            -not [long]::TryParse([string]$Entry.ProcessStartTimeUtcTicks, [ref]$ticks) -or $ticks -le 0 -or
            [string]::IsNullOrWhiteSpace([string]$Entry.ExecutablePath)) { throw 'Missing or invalid process custody identity.' }
        $expected = Get-Item -LiteralPath $Entry.ExecutablePath -ErrorAction Stop
        if ($expected.PSProvider.Name -ne 'FileSystem' -or $expected.PSIsContainer) { throw 'Executable identity is not an existing file.' }
        $process = Get-Process -Id $processId -ErrorAction Stop
        $handle = $process.Handle
        if ($handle -isnot [IntPtr] -or $handle -eq [IntPtr]::Zero -or $handle -eq [IntPtr]::new(-1)) { throw 'Process custody handle could not be verified.' }
        $actual = Get-Item -LiteralPath $process.MainModule.FileName -ErrorAction Stop
        if (-not [string]::Equals($actual.FullName, $expected.FullName, [StringComparison]::OrdinalIgnoreCase) -or
            $process.StartTime.ToUniversalTime().Ticks -ne $ticks -or $process.HasExited) { throw 'Process custody identity does not match.' }
        return [pscustomobject]@{ Verified = $true; Process = $process; Reason = $null }
    } catch {
        if ($process) { $process.Dispose() }
        return [pscustomobject]@{ Verified = $false; Process = $null; Reason = $_.Exception.Message }
    }
}

<#
.SYNOPSIS
    Reports proxy status only for verified process custody.
#>
function Get-HVSockProxyStatus {
    [CmdletBinding()]
    [OutputType([PSCustomObject])]
    param([string]$StatePath = "$env:ProgramData\PC_AI\hvsock-proxy\state.json")
    $StatePath = Resolve-HVSockStatePath -Path $StatePath
    if (-not (Test-Path -LiteralPath $StatePath -PathType Leaf)) { return [pscustomobject]@{ Running = 0; Entries = @(); StatePath = $StatePath } }
    $state = Get-Content -LiteralPath $StatePath -Raw -ErrorAction Stop | ConvertFrom-Json -ErrorAction Stop
    $entries = @(
        foreach ($entry in $state) {
            $owned = Get-HVSockOwnedProcess -Entry $entry
            try {
                [pscustomobject]@{ Name = $entry.Name; Pid = $entry.Pid; Running = $owned.Verified; CustodyVerified = $owned.Verified
                    CustodyError = $owned.Reason; ServiceId = $entry.ServiceId; TcpTarget = $entry.TcpTarget; Started = $entry.Started }
            } finally { if ($owned.Process) { $owned.Process.Dispose() } }
        }
    )
    return [pscustomobject]@{ Running = @($entries | Where-Object Running).Count; Entries = $entries; StatePath = $StatePath }
}
