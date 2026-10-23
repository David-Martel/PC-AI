#Requires -Version 5.1
param([string]$VsockSourcePath, [string]$EvidenceRoot)

BeforeDiscovery {
    $vsockWindows = [Environment]::OSVersion.Platform -eq [PlatformID]::Win32NT
    $vsockAdmin = $false
    if ($vsockWindows) {
        $vsockAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    }
}

Describe 'VSock predecessor custody and native boundaries' -Tag 'Unit', 'Network', 'Windows', 'RequiresAdmin' -Skip:(-not ($vsockWindows -and $vsockAdmin)) {
    BeforeAll {
        if (-not ('PcaiVsockFixtureKey' -as [type])) {
            Add-Type -TypeDefinition @'
using System;
using System.Collections;
using System.Collections.Generic;
using Microsoft.Win32;
public sealed class PcaiVsockFixtureKey : IDisposable {
    public static int Disposals;
    private readonly Hashtable originals;
    private readonly string failure;
    public PcaiVsockFixtureKey(Hashtable originals, string failure) { this.originals = originals; this.failure = failure; }
    public string[] GetValueNames() {
        if (failure == "names") throw new InvalidOperationException("Synthetic value-name read refusal.");
        var names = new List<string>(); foreach (string name in originals.Keys) names.Add(name); return names.ToArray();
    }
    public object GetValue(string name, object fallback, RegistryValueOptions options) {
        if (failure == "values" || failure == "value-" + name) throw new InvalidOperationException("Synthetic value read refusal.");
        if (options != RegistryValueOptions.DoNotExpandEnvironmentNames) throw new InvalidOperationException("Expanded original strings are forbidden.");
        return ((Hashtable)originals[name])["Value"];
    }
    public RegistryValueKind GetValueKind(string name) {
        if (failure == "kind") throw new InvalidOperationException("Synthetic kind read refusal.");
        return (RegistryValueKind)Enum.Parse(typeof(RegistryValueKind), (string)((Hashtable)originals[name])["Kind"]);
    }
    public void Dispose() { Disposals++; }
}
'@
        }
        $source = if ($VsockSourcePath) { $VsockSourcePath } else { Join-Path $PSScriptRoot '../../Modules/PC-AI.Network/Public/Optimize-VSock.ps1' }
        . $source
        function Get-RegistryValueSafe { param($Path, $Name) throw 'Live registry reads are forbidden.' }
        function Set-RegistryValueSafe { param($Path, $Name, $Value, $PropertyType, $WhatIf) throw 'Live registry writes are forbidden.' }
        function Set-ItemProperty { [CmdletBinding()] param($Path,$LiteralPath,$Name,$Value,$Type,[switch]$Force) throw 'Live registry restore is forbidden.' }
        function Remove-ItemProperty { [CmdletBinding()] param($Path,$LiteralPath,$Name) throw 'Live registry deletion is forbidden.' }
        function netsh { param([Parameter(ValueFromRemainingArguments)][object[]]$Tokens) throw 'Live netsh is forbidden.' }
        function wsl { throw 'Live WSL is forbidden.' }
        function Invoke-OwnedVsockNative {
            param([int]$ExitCode, [string]$Text)
            $path = Join-Path $TestDrive 'vsock-native-control.cmd'
            [IO.File]::WriteAllLines($path, @('@echo off', "echo $Text", "exit /b $ExitCode"), [Text.UTF8Encoding]::new($false))
            & $env:ComSpec /d /c $path
            $global:LASTEXITCODE = $LASTEXITCODE
        }
        function Get-OwnedVsockNamespace {
            @(Get-ChildItem -LiteralPath $TestDrive -File -Recurse | ForEach-Object {
                [pscustomobject]@{ Path = $_.FullName; Hash = (Get-FileHash -LiteralPath $_.FullName).Hash }
            } | Sort-Object Path | ConvertTo-Json -Compress)
        }
        function New-OwnedRegistryKey {
            return [PcaiVsockFixtureKey]::new($script:Originals,$script:CaptureFailure)
        }
    }
    BeforeEach {
        $script:Backup = Join-Path $TestDrive 'owned-backup.json'
        if ([IO.File]::Exists($script:Backup)) { [IO.File]::Delete($script:Backup) }
        $script:ModuleRoot = Join-Path $TestDrive 'default-root/Modules/PC-AI.Network'
        $script:NativeMode = 'healthy'
        $script:WslNativeCalls = 0
        $script:CaptureFailure = ''
        [PcaiVsockFixtureKey]::Disposals = 0
        $script:Originals = @{}
        foreach ($name in @('EnableAutoTuning','Tcp1323Opts','DefaultTTL','EnableTCPChimney','RssBaseCpu','NetworkThrottlingIndex','SystemResponsiveness','TcpMaxDataRetransmissions','MaxCmds')) {
            $script:Originals[$name] = @{ Value = [int]0; Kind = 'DWord' }
        }
        Mock Get-RegistryValueSafe { if ($script:Originals.ContainsKey($Name)) { $script:Originals[$Name].Value } else { $null } }
        Mock Set-RegistryValueSafe { $true }
        Mock Get-Item { if ($script:CaptureFailure -eq 'get-item') { throw 'Synthetic key read refusal.' }; New-OwnedRegistryKey } -ParameterFilter { $Path -like 'HKLM:*' -or $LiteralPath -like 'HKLM:*' }
        Mock Set-ItemProperty { throw 'Unmocked registry restore forbidden.' }
        Mock Remove-ItemProperty { throw 'Unmocked registry deletion forbidden.' }
        Mock Test-Path {
            $target = if ($LiteralPath) { $LiteralPath } else { $Path }
            if ($target -like 'HKLM:*') { $true } else { [IO.File]::Exists($target) -or [IO.Directory]::Exists($target) }
        }
        Mock Write-Host { }
        Mock Start-Sleep { }
        Mock netsh {
            if ($script:NativeMode -eq 'netsh-throw') { throw 'Synthetic netsh refusal.' }
            $exitCode = if ($script:NativeMode -eq 'netsh-fail') { 7 } else { 0 }
            Invoke-OwnedVsockNative -ExitCode $exitCode -Text 'Plausible success'
        }
        Mock wsl {
            if ($script:NativeMode -eq 'wsl-throw') { throw 'Synthetic WSL refusal.' }
            $script:WslNativeCalls++
            $exitCode = if ($script:WslNativeCalls -eq 1 -and $script:NativeMode -eq 'shutdown-fail') { 7 } elseif ($script:WslNativeCalls -gt 1 -and $script:NativeMode -eq 'echo-fail') { 7 } else { 0 }
            Invoke-OwnedVsockNative -ExitCode $exitCode -Text 'VSock test'
        }
    }
    It 'preserves actual preexisting backup bytes on Apply and refuses all mutation' {
        [IO.File]::WriteAllText($script:Backup, 'Foreign predecessor backup sentinel')
        $before = (Get-FileHash -LiteralPath $script:Backup).Hash
        if ($EvidenceRoot) { [IO.File]::Copy($script:Backup, (Join-Path $EvidenceRoot 'apply-backup.before.json'), $false) }
        $r = Optimize-VSock -BackupPath $script:Backup -Confirm:$false
        if ($EvidenceRoot) { [IO.File]::Copy($script:Backup, (Join-Path $EvidenceRoot 'apply-backup.after.json'), $false) }
        (Get-FileHash -LiteralPath $script:Backup).Hash | Should -BeExactly $before
        $r.BackupCreated | Should -BeFalse
        @($r.Errors).Count | Should -BeGreaterThan 0
        Should -Invoke Set-RegistryValueSafe -Times 0 -Exactly
        Should -Invoke netsh -Times 0 -Exactly
        Should -Invoke wsl -Times 0 -Exactly
    }
    It 'refuses Apply when all original registry values are indeterminate' {
        Mock Get-RegistryValueSafe { $null }
        $script:CaptureFailure = 'get-item'
        $r = Optimize-VSock -BackupPath $script:Backup -Confirm:$false
        $r.BackupCreated | Should -BeFalse
        @($r.Errors).Count | Should -BeGreaterThan 0
        Should -Invoke Set-RegistryValueSafe -Times 0 -Exactly
        Should -Invoke netsh -Times 0 -Exactly
        Should -Invoke wsl -Times 0 -Exactly
    }
    It 'refuses Apply after mixed incomplete original registry capture' {
        Mock Get-RegistryValueSafe { if ($Name -eq 'DefaultTTL') { $null } else { 0 } }
        $script:CaptureFailure = 'value-DefaultTTL'
        $r = Optimize-VSock -BackupPath $script:Backup -Confirm:$false
        $r.BackupCreated | Should -BeFalse
        @($r.Errors).Count | Should -BeGreaterThan 0
        Should -Invoke Set-RegistryValueSafe -Times 0 -Exactly
        Should -Invoke netsh -Times 0 -Exactly
    }
    It 'does not create the absent default Config directory during WhatIf' {
        $config = Join-Path $TestDrive 'default-root/Config'
        [IO.Directory]::Exists($config) | Should -BeFalse
        $before = Get-OwnedVsockNamespace
        $r = Optimize-VSock -SkipWSLRestart -WhatIf
        if ($EvidenceRoot) {
            [ordered]@{ Config = $config; ExistsBefore = $false; ExistsAfter = [IO.Directory]::Exists($config); NamespaceBefore = $before; NamespaceAfter = (Get-OwnedVsockNamespace) } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $EvidenceRoot 'default-whatif.json')
        }
        [IO.Directory]::Exists($config) | Should -BeFalse
        Get-OwnedVsockNamespace | Should -BeExactly $before
        $r.BackupCreated | Should -BeFalse
        Should -Invoke Set-RegistryValueSafe -Times 0 -Exactly
        Should -Invoke netsh -Times 0 -Exactly
        Should -Invoke wsl -Times 0 -Exactly
    }
    It 'does not promote shutdown success and failed echo to restart success' {
        $script:NativeMode = 'echo-fail'
        $r = Optimize-VSock -BackupPath $script:Backup -Confirm:$false
        if ($EvidenceRoot) { $r | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $EvidenceRoot 'shutdown-echo-output.json') }
        $r.WSLRestarted | Should -BeFalse
        @($r.Errors | Where-Object { $_ -match 'WSL' }).Count | Should -BeGreaterThan 0
        Should -Invoke wsl -Times 2 -Exactly
    }
    It 'reports native netsh exceptions' {
        $script:NativeMode = 'netsh-throw'
        $r = Optimize-VSock -BackupPath $script:Backup -SkipWSLRestart -Confirm:$false
        @($r.Errors | Where-Object { $_ -match 'netsh.*Synthetic netsh refusal' }).Count | Should -Be 4
    }
    It 'reports native WSL exceptions without claiming restart' {
        $script:NativeMode = 'wsl-throw'
        $r = Optimize-VSock -BackupPath $script:Backup -Confirm:$false
        $r.WSLRestarted | Should -BeFalse
        @($r.Errors | Where-Object { $_ -match 'WSL.*Synthetic WSL refusal' }).Count | Should -Be 1
    }
    It 'reports registry apply exceptions' {
        Mock Set-RegistryValueSafe { throw 'Synthetic registry publisher refusal.' }
        $r = Optimize-VSock -BackupPath $script:Backup -SkipWSLRestart -Confirm:$false
        @($r.Errors | Where-Object { $_ -match 'Error setting.*Synthetic registry publisher refusal' }).Count | Should -Be 8
        $r.ChangesApplied.Count | Should -Be 0
    }
    It 'reports a nonterminating legacy restore write refusal as restore failure' {
        [IO.File]::WriteAllText($script:Backup, '[{"Path":"HKLM:\\OwnedFixture","Name":"FixtureValue","Value":10}]')
        $before = (Get-FileHash -LiteralPath $script:Backup).Hash
        Mock Set-ItemProperty {
            # Model the cmdlet's explicitly supplied action across module scopes.
            $action = if ($PesterBoundParameters.ContainsKey('ErrorAction')) { $PesterBoundParameters.ErrorAction } else { 'Continue' }
            Microsoft.PowerShell.Utility\Write-Error 'Synthetic restore publisher refusal.' -ErrorAction $action
        }
        $r = Optimize-VSock -RestoreBackup -BackupPath $script:Backup -Confirm:$false -ErrorAction SilentlyContinue
        @($r.Errors | Where-Object { $_ -match 'Failed to restore backup.*Synthetic restore publisher refusal' }).Count | Should -BeGreaterThan 0
        @($r.ChangesApplied).Count | Should -Be 0
        (Get-FileHash -LiteralPath $script:Backup).Hash | Should -BeExactly $before
        Should -Invoke netsh -Times 0 -Exactly
        Should -Invoke wsl -Times 0 -Exactly
    }
    It 'closes a captured key after value-name read failure' {
        $script:CaptureFailure = 'names'
        $r = Optimize-VSock -BackupPath $script:Backup -SkipWSLRestart -Confirm:$false
        @($r.Errors | Where-Object { $_ -match 'Registry backup capture' }).Count | Should -Be 1
        [PcaiVsockFixtureKey]::Disposals | Should -Be 1
        [IO.File]::Exists($script:Backup) | Should -BeFalse
        Should -Invoke Set-RegistryValueSafe -Times 0 -Exactly
        Should -Invoke netsh -Times 0 -Exactly
    }
    It 'closes a captured key after value-kind read failure' {
        $script:CaptureFailure = 'kind'
        $r = Optimize-VSock -BackupPath $script:Backup -SkipWSLRestart -Confirm:$false
        @($r.Errors | Where-Object { $_ -match 'Registry backup capture' }).Count | Should -Be 1
        [PcaiVsockFixtureKey]::Disposals | Should -Be 1
        [IO.File]::Exists($script:Backup) | Should -BeFalse
        Should -Invoke Set-RegistryValueSafe -Times 0 -Exactly
    }
    It 'rejects unsupported original None kind before backup or mutation' {
        $script:Originals['DefaultTTL'] = @{ Value = [byte[]]@(0,255); Kind = 'None' }
        $r = Optimize-VSock -BackupPath $script:Backup -SkipWSLRestart -Confirm:$false
        @($r.Errors | Where-Object { $_ -match 'Unsupported original registry kind None' }).Count | Should -Be 1
        [IO.File]::Exists($script:Backup) | Should -BeFalse
        Should -Invoke Set-RegistryValueSafe -Times 0 -Exactly
    }
    It 'retains supported typed original bytes and restores their actual types' {
        $script:Originals['EnableAutoTuning'] = @{ Value = [int]-1; Kind = 'DWord' }
        $script:Originals['Tcp1323Opts'] = @{ Value = [long]4294967296; Kind = 'QWord' }
        $script:Originals['DefaultTTL'] = @{ Value = 'retained literal'; Kind = 'String' }
        $script:Originals['EnableTCPChimney'] = @{ Value = '%OWNED_LITERAL%'; Kind = 'ExpandString' }
        $script:Originals['RssBaseCpu'] = @{ Value = [string[]]@('first','second'); Kind = 'MultiString' }
        $script:Originals['NetworkThrottlingIndex'] = @{ Value = [byte[]]@(0,127,255); Kind = 'Binary' }
        $null = Optimize-VSock -BackupPath $script:Backup -SkipWSLRestart -Confirm:$false
        $rows = Get-Content -LiteralPath $script:Backup -Raw | ConvertFrom-Json
        $rows = @($rows)
        $rows.Count | Should -Be 9
        if ($EvidenceRoot) { [IO.File]::Copy($script:Backup,(Join-Path $EvidenceRoot 'typed-roundtrip.backup.json'),$false) }
        @($rows | Where-Object { $_.Present -ne $true -or -not $_.Kind }).Count | Should -Be 0
        [PcaiVsockFixtureKey]::Disposals | Should -Be 9
        $script:Restored = @{}
        Mock Set-ItemProperty { $script:Restored[$Name] = @{ Value = $Value; Kind = $Type } }
        $r = Optimize-VSock -RestoreBackup -BackupPath $script:Backup -Confirm:$false
        if ($EvidenceRoot) { [ordered]@{ Result = $r; Restored = $script:Restored } | ConvertTo-Json -Depth 9 | Set-Content -LiteralPath (Join-Path $EvidenceRoot 'typed-restore-output.json') }
        @($r.Errors).Count | Should -Be 0
        @($r.ChangesApplied).Count | Should -Be 9
        foreach ($name in $script:Originals.Keys) {
            $script:Restored[$name].Kind | Should -BeExactly $script:Originals[$name].Kind
            $script:Restored[$name].Value.GetType().FullName | Should -BeExactly $script:Originals[$name].Value.GetType().FullName
            ($script:Restored[$name].Value | ConvertTo-Json -Compress) | Should -BeExactly ($script:Originals[$name].Value | ConvertTo-Json -Compress)
        }
        Should -Invoke Remove-ItemProperty -Times 0 -Exactly
    }
    It 'records actual absent original properties and removes only those properties on restore' {
        $script:Originals.Remove('DefaultTTL')
        $null = Optimize-VSock -BackupPath $script:Backup -SkipWSLRestart -Confirm:$false
        $rows = Get-Content -LiteralPath $script:Backup -Raw | ConvertFrom-Json
        $rows = @($rows)
        $absent = @($rows | Where-Object Name -eq 'DefaultTTL')
        $absent.Count | Should -Be 1
        $absent[0].Present | Should -BeFalse
        $absent[0].Value | Should -BeNullOrEmpty
        if ($EvidenceRoot) { [IO.File]::Copy($script:Backup,(Join-Path $EvidenceRoot 'absent-property.backup.json'),$false) }
        $before = (Get-FileHash -LiteralPath $script:Backup).Hash
        $script:Originals['DefaultTTL'] = @{ Value = 128; Kind = 'DWord' }
        Mock Set-ItemProperty { }
        Mock Remove-ItemProperty { }
        $r = Optimize-VSock -RestoreBackup -BackupPath $script:Backup -Confirm:$false
        @($r.Errors).Count | Should -Be 0
        @($r.ChangesApplied).Count | Should -Be 9
        Should -Invoke Remove-ItemProperty -Times 1 -Exactly -ParameterFilter { $Name -eq 'DefaultTTL' -and $LiteralPath -eq 'HKLM:\SYSTEM\CurrentControlSet\Services\Tcpip\Parameters' }
        Should -Invoke Set-ItemProperty -Times 8 -Exactly
        (Get-FileHash -LiteralPath $script:Backup).Hash | Should -BeExactly $before
    }
    It 'restores retained legacy backups without inventing type metadata' {
        [IO.File]::WriteAllText($script:Backup, '[{"Path":"HKLM:\\OwnedFixture","Name":"FixtureValue","Value":10}]')
        Mock Set-ItemProperty { }
        $r = Optimize-VSock -RestoreBackup -BackupPath $script:Backup -Confirm:$false
        @($r.Errors).Count | Should -Be 0
        @($r.ChangesApplied).Count | Should -Be 1
        Should -Invoke Set-ItemProperty -Times 1 -Exactly -ParameterFilter { $Path -eq 'HKLM:\OwnedFixture' -and $Name -eq 'FixtureValue' -and $Value -eq 10 -and $ErrorAction -eq 'Stop' -and -not $Type }
        Should -Invoke Remove-ItemProperty -Times 0 -Exactly
    }
    It 'reports restore WhatIf as pending and preserves actual backup bytes' {
        [IO.File]::WriteAllText($script:Backup, '[{"Path":"HKLM:\\OwnedFixture","Name":"FixtureValue","Value":10}]')
        $before = (Get-FileHash -LiteralPath $script:Backup).Hash
        $r = Optimize-VSock -RestoreBackup -BackupPath $script:Backup -WhatIf
        @($r.Errors).Count | Should -Be 0
        @($r.ChangesApplied).Count | Should -Be 0
        @($r.ChangesPending).Count | Should -Be 1
        Should -Invoke Set-ItemProperty -Times 0 -Exactly
        Should -Invoke Remove-ItemProperty -Times 0 -Exactly
        (Get-FileHash -LiteralPath $script:Backup).Hash | Should -BeExactly $before
    }
    It 'applies <Profile> policy with a complete retained backup' -ForEach @(
        @{ Profile = 'Balanced'; Rss = 1; Responsiveness = 10; MaxCommands = 50; NetshCalls = 4; Throttle = 10 },
        @{ Profile = 'Performance'; Rss = 0; Responsiveness = 0; MaxCommands = 100; NetshCalls = 5; Throttle = [uint32]::MaxValue },
        @{ Profile = 'Conservative'; Rss = 2; Responsiveness = 20; MaxCommands = 30; NetshCalls = 3; Throttle = 10 }
    ) {
        Mock Get-RegistryValueSafe { -2 }
        $r = Optimize-VSock -Profile $Profile -BackupPath $script:Backup -SkipWSLRestart -Confirm:$false
        if ($EvidenceRoot -and [IO.File]::Exists($script:Backup)) { [IO.File]::Copy($script:Backup,(Join-Path $EvidenceRoot ('profile-' + $Profile.ToLowerInvariant() + '.backup.json')),$false) }
        $r.BackupCreated | Should -BeTrue
        @($r.Errors).Count | Should -Be 0
        @($r.ChangesApplied).Count | Should -Be 9
        Should -Invoke Set-RegistryValueSafe -Times 1 -Exactly -ParameterFilter { $Name -eq 'RssBaseCpu' -and $Value -eq $Rss }
        Should -Invoke Set-RegistryValueSafe -Times 1 -Exactly -ParameterFilter { $Name -eq 'SystemResponsiveness' -and $Value -eq $Responsiveness }
        Should -Invoke Set-RegistryValueSafe -Times 1 -Exactly -ParameterFilter { $Name -eq 'MaxCmds' -and $Value -eq $MaxCommands }
        Should -Invoke Set-RegistryValueSafe -Times 1 -Exactly -ParameterFilter { $Name -eq 'NetworkThrottlingIndex' -and $Value -eq $Throttle }
        Should -Invoke netsh -Times $NetshCalls -Exactly
        Should -Invoke wsl -Times 0 -Exactly
    }
    It 'preserves actual existing backup bytes during WhatIf' {
        [IO.File]::WriteAllText($script:Backup,'Preserved WhatIf backup sentinel')
        $before = (Get-FileHash -LiteralPath $script:Backup).Hash
        $r = Optimize-VSock -BackupPath $script:Backup -WhatIf
        (Get-FileHash -LiteralPath $script:Backup).Hash | Should -BeExactly $before
        $r.BackupCreated | Should -BeFalse
        Should -Invoke Set-RegistryValueSafe -Times 0 -Exactly
        Should -Invoke netsh -Times 0 -Exactly
        Should -Invoke wsl -Times 0 -Exactly
    }
    It 'records actual native netsh exit7 despite plausible success stdout' {
        $script:NativeMode = 'netsh-fail'
        $r = Optimize-VSock -BackupPath $script:Backup -SkipWSLRestart -Confirm:$false
        @($r.Errors | Where-Object { $_ -match 'netsh.*exit code 7' }).Count | Should -Be 4
        Should -Invoke netsh -Times 4 -Exactly
    }
    It 'refuses restart after actual native shutdown exit7' {
        $script:NativeMode = 'shutdown-fail'
        $r = Optimize-VSock -BackupPath $script:Backup -Confirm:$false
        $r.WSLRestarted | Should -BeFalse
        @($r.Errors | Where-Object { $_ -match 'WSL shutdown failed with exit code 7' }).Count | Should -Be 1
        Should -Invoke wsl -Times 1 -Exactly
    }
    It 'preserves successful actual private native observations in one result' {
        $r = Optimize-VSock -BackupPath $script:Backup -Confirm:$false
        @($r).Count | Should -Be 1
        $r.BackupCreated | Should -BeTrue
        $r.WSLRestarted | Should -BeTrue
        @($r.Errors).Count | Should -Be 0
        Should -Invoke wsl -Times 2 -Exactly
    }
    It 'refuses mutation after actual directory backup write failure' {
        $null = [IO.Directory]::CreateDirectory($script:Backup)
        $r = Optimize-VSock -BackupPath $script:Backup -Confirm:$false
        $r.BackupCreated | Should -BeFalse
        @($r.Errors).Count | Should -BeGreaterThan 0
        Should -Invoke Set-RegistryValueSafe -Times 0 -Exactly
        Should -Invoke netsh -Times 0 -Exactly
        Should -Invoke wsl -Times 0 -Exactly
    }
}

Describe 'Entire VSock restore ledger preflight' -Tag 'Unit', 'Network', 'Windows', 'RequiresAdmin' -Skip:(-not ($vsockWindows -and $vsockAdmin)) {
    BeforeAll {
        $source = if ($VsockSourcePath) { $VsockSourcePath } else { Join-Path $PSScriptRoot '../../Modules/PC-AI.Network/Public/Optimize-VSock.ps1' }
        . $source
        function Set-ItemProperty { [CmdletBinding()] param($Path,$LiteralPath,$Name,$Value,$Type,[switch]$Force) throw 'Live restore forbidden.' }
        function Remove-ItemProperty { [CmdletBinding()] param($Path,$LiteralPath,$Name) throw 'Live deletion forbidden.' }
        function Get-Item { [CmdletBinding()] param($Path,$LiteralPath) throw 'Live registry reads forbidden.' }
        function Get-RegistryValueSafe { throw 'Live capture forbidden.' }
        function Set-RegistryValueSafe { throw 'Live optimization forbidden.' }
        function wsl { throw 'Live WSL forbidden.' }
        function netsh { throw 'Live netsh forbidden.' }
    }
    BeforeEach {
        $script:Ledger = Join-Path $TestDrive 'restore-ledger.json'
        $script:RestoreCalls = @()
        Mock Set-ItemProperty { $script:RestoreCalls += [pscustomobject]@{ Path=if($LiteralPath){$LiteralPath}else{$Path}; Name=$Name; Value=$Value; Kind=$Type } }
        Mock Remove-ItemProperty { throw 'Unexpected property deletion forbidden.' }
        Mock Write-Host { }
    }
    It 'rejects <Case> before publishing any earlier valid row' -ForEach @(
        @{ Case='json-null'; Json='null' },
        @{ Case='empty-array'; Json='[]' },
        @{ Case='invalid-later-presence'; Json='[{"Path":"HKLM:\\Owned","Name":"First","Value":10},{"Path":"HKLM:\\Owned","Name":"Second","Value":20,"Present":"false","Kind":"DWord"}]' },
        @{ Case='invalid-later-kind'; Json='[{"Path":"HKLM:\\Owned","Name":"First","Value":10},{"Path":"HKLM:\\Owned","Name":"Second","Value":20,"Present":true,"Kind":"NotARegistryKind"}]' },
        @{ Case='invalid-later-path'; Json='[{"Path":"HKLM:\\Owned","Name":"First","Value":10},{"Name":"Second","Value":20}]' },
        @{ Case='invalid-later-name'; Json='[{"Path":"HKLM:\\Owned","Name":"First","Value":10},{"Path":"HKLM:\\Owned","Value":20}]' },
        @{ Case='invalid-later-value'; Json='[{"Path":"HKLM:\\Owned","Name":"First","Value":10},{"Path":"HKLM:\\Owned","Name":"Second","Value":"not-a-DWord","Present":true,"Kind":"DWord"}]' }
    ) {
        [IO.File]::WriteAllText($script:Ledger,$Json,[Text.UTF8Encoding]::new($false))
        $before = (Get-FileHash -LiteralPath $script:Ledger).Hash
        $r = Optimize-VSock -RestoreBackup -BackupPath $script:Ledger -Confirm:$false
        if ($EvidenceRoot) { [ordered]@{ Case=$Case; InputHash=$before; InputJson=$Json; Calls=$script:RestoreCalls; Result=$r; OutputHash=(Get-FileHash -LiteralPath $script:Ledger).Hash } | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $EvidenceRoot ($Case + '.json')) }
        Should -Invoke Set-ItemProperty -Times 0 -Exactly
        Should -Invoke Remove-ItemProperty -Times 0 -Exactly
        Should -Invoke Write-Host -Times 0 -Exactly -ParameterFilter { $Object -eq '[+] Settings restored from backup' }
        @($r.Errors | Where-Object { $null -ne $_ }).Count | Should -BeGreaterThan 0
        (Get-FileHash -LiteralPath $script:Ledger).Hash | Should -BeExactly $before
    }
    It 'preserves valid predecessor rows and exact source backup bytes' {
        $json='[{"Path":"HKLM:\\Owned","Name":"First","Value":10},{"Path":"HKLM:\\Owned","Name":"Second","Value":20}]'
        [IO.File]::WriteAllText($script:Ledger,$json,[Text.UTF8Encoding]::new($false))
        $before=(Get-FileHash -LiteralPath $script:Ledger).Hash
        $r=Optimize-VSock -RestoreBackup -BackupPath $script:Ledger -Confirm:$false
        Should -Invoke Set-ItemProperty -Times 2 -Exactly
        if ($null -ne $r) { @($r.Errors).Count | Should -Be 0; @($r.ChangesApplied).Count | Should -Be 2 }
        (Get-FileHash -LiteralPath $script:Ledger).Hash | Should -BeExactly $before
    }
    It 'preserves valid predecessor rows under caller strict mode' {
        $json='[{"Path":"HKLM:\\Owned","Name":"First","Value":10},{"Path":"HKLM:\\Owned","Name":"Second","Value":20}]'
        [IO.File]::WriteAllText($script:Ledger,$json,[Text.UTF8Encoding]::new($false))
        $before=(Get-FileHash -LiteralPath $script:Ledger).Hash
        try {
            Set-StrictMode -Version Latest
            $r=Optimize-VSock -RestoreBackup -BackupPath $script:Ledger -Confirm:$false
        } finally { Set-StrictMode -Off }
        if ($EvidenceRoot) { [ordered]@{Case='strict-legacy';Calls=$script:RestoreCalls;Result=$r;InputHash=$before;OutputHash=(Get-FileHash -LiteralPath $script:Ledger).Hash}|ConvertTo-Json -Depth 6|Set-Content -LiteralPath (Join-Path $EvidenceRoot 'strict-legacy.json') }
        Should -Invoke Set-ItemProperty -Times 2 -Exactly
        @($r.Errors).Count|Should -Be 0
        @($r.ChangesApplied).Count|Should -Be 2
        (Get-FileHash -LiteralPath $script:Ledger).Hash|Should -BeExactly $before
    }
    It 'restores ISO text literally for <Kind>' -ForEach @(@{Kind='String'},@{Kind='ExpandString'},@{Kind='MultiString'}) {
        $literal='2026-10-09T00:00:00Z'
        $value=if($Kind -eq 'MultiString'){,([string[]]@($literal,'retained second'))}else{$literal}
        @([pscustomobject]@{Path='HKLM:\Owned';Name='Literal';Value=$value;Present=$true;Kind=$Kind})|ConvertTo-Json -Depth 5|Set-Content -LiteralPath $script:Ledger
        $r=Optimize-VSock -RestoreBackup -BackupPath $script:Ledger -Confirm:$false
        if ($EvidenceRoot) { [ordered]@{Kind=$Kind;Expected=$value;Calls=$script:RestoreCalls;Result=$r}|ConvertTo-Json -Depth 6|Set-Content -LiteralPath (Join-Path $EvidenceRoot ('iso-' + $Kind.ToLowerInvariant() + '.json')) }
        $script:RestoreCalls.Count|Should -Be 1
        $actual=$script:RestoreCalls[0].Value
        $actual.GetType().FullName|Should -BeExactly $value.GetType().FullName
        ($actual|ConvertTo-Json -Compress)|Should -BeExactly ($value|ConvertTo-Json -Compress)
    }
}
