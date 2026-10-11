<#
.SYNOPSIS
    Exercises provider-order persistence using isolated, real JSON files.
#>

BeforeAll {
    $modulePath = Join-Path $PSScriptRoot '../../Modules/PC-AI.LLM/PC-AI.LLM.psd1'
    $script:RepositoryConfigPath = Join-Path $PSScriptRoot '../../Config/llm-config.json'
    $script:RepositoryConfigHash = (Get-FileHash -LiteralPath $script:RepositoryConfigPath).Hash
    Import-Module $modulePath -Force -ErrorAction Stop
    $script:OriginalModuleConfig = InModuleScope PC-AI.LLM { $script:ModuleConfig.Clone() }
    if (-not ('Pcai.Config.AclFixtureV1' -as [type])) {
        Add-Type -TypeDefinition @'
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;
namespace Pcai.Config {
 public static class AclFixtureV1 {
  [DllImport("ntdll.dll", ExactSpelling=true)]
  public static extern int NtSetSecurityObject(SafeFileHandle handle, uint information, [In] byte[] descriptor);
 }
}
'@
    }
    function Get-ConfigFixtureReceipt {
        $pathHash = InModuleScope PC-AI.LLM -Parameters @{ Path = $script:ConfigPath } {
            $identity = if ($IsWindows) { [IO.Path]::GetFullPath($Path).ToUpperInvariant() } else { [IO.Path]::GetFullPath($Path) }
            (Get-LLMConfigBytesHash -Bytes ([Text.Encoding]::UTF8.GetBytes($identity))).ToLowerInvariant()
        }
        $root = Join-Path (Split-Path -Parent $script:ConfigPath) ('.pcai/config-write/' + $pathHash)
        return Join-Path $root 'r1/receipt.json'
    }
}

AfterAll {
    InModuleScope PC-AI.LLM -Parameters @{ Original = $script:OriginalModuleConfig } {
        $script:ModuleConfig = $Original
    }
    (Get-FileHash -LiteralPath $script:RepositoryConfigPath).Hash | Should -Be $script:RepositoryConfigHash
    Remove-Module PC-AI.LLM -Force -ErrorAction SilentlyContinue
}

Describe 'Set-LLMProviderOrder' -Tag 'Unit', 'LLM', 'Fast', 'Windows' {
    BeforeEach {
        $script:ConfigPath = Join-Path $TestDrive ('llm-config-' + [guid]::NewGuid().ToString('N') + '.json')
        @'
{"fallbackOrder":["ollama"],"providers":{"ollama":{"defaultModel":"machine-model","timeout":777}},"ollama":{"num_gpu":2,"num_ctx":16384,"tool_model":"machine-tools"},"machineSetting":{"preserve":true}}
'@ | Set-Content -LiteralPath $script:ConfigPath -Encoding utf8NoBOM
        InModuleScope PC-AI.LLM -Parameters @{ ConfigPath = $script:ConfigPath; Original = $script:OriginalModuleConfig } {
            $script:ModuleConfig = $Original.Clone()
            $script:ModuleConfig.ConfigPath = $ConfigPath
            $script:ModuleConfig.ProjectConfigPath = $ConfigPath
            $script:ModuleConfig.ProviderOrder = @('ollama')
        }
        $script:BeforeHash = (Get-FileHash -LiteralPath $script:ConfigPath).Hash
    }

    It 'updates disk and memory while preserving machine configuration and BOM-free JSON' {
        $result = Set-LLMProviderOrder -Order @('pcai-inference', 'ollama')
        $config = Get-Content -LiteralPath $script:ConfigPath -Raw | ConvertFrom-Json
        $result.Success | Should -BeTrue
        $result.Order -join ',' | Should -Be 'pcai-inference,ollama'
        $result.ConfigPath | Should -Be $script:ConfigPath
        $config.fallbackOrder -join ',' | Should -Be 'pcai-inference,ollama'
        $config.providers.ollama.defaultModel | Should -Be 'machine-model'
        $config.providers.ollama.timeout | Should -Be 777
        $config.ollama.num_gpu | Should -Be 2
        $config.ollama.num_ctx | Should -Be 16384
        $config.ollama.tool_model | Should -Be 'machine-tools'
        $config.machineSetting.preserve | Should -BeTrue
        [System.IO.File]::ReadAllBytes($script:ConfigPath)[0] | Should -Be 123
        InModuleScope PC-AI.LLM { $script:ModuleConfig.ProviderOrder -join ',' } | Should -Be 'pcai-inference,ollama'
    }

    It 'supports every runtime provider and legacy alias' {
        $order = @('ollama', 'pcai-inference', 'vllm', 'lmstudio', 'pcai-native', 'functiongemma')
        Set-LLMProviderOrder -Order $order | Out-Null
        (Get-Content -LiteralPath $script:ConfigPath -Raw | ConvertFrom-Json).fallbackOrder -join ',' | Should -Be ($order -join ',')
    }

    It 'adds fallbackOrder when absent' {
        '{"providers":{}}' | Set-Content -LiteralPath $script:ConfigPath -Encoding utf8NoBOM
        Set-LLMProviderOrder -Order @('vllm', 'ollama') | Out-Null
        (Get-Content -LiteralPath $script:ConfigPath -Raw | ConvertFrom-Json).fallbackOrder -join ',' | Should -Be 'vllm,ollama'
    }

    It 'rejects <Label> without changing disk or memory' -TestCases @(
        @{ Label = 'unknown provider'; Order = @('invalid') }
        @{ Label = 'mixed known and unknown providers'; Order = @('ollama', 'invalid') }
        @{ Label = 'whitespace provider'; Order = @('ollama', ' ') }
        @{ Label = 'empty provider'; Order = @('ollama', '') }
        @{ Label = 'null order'; Order = $null }
        @{ Label = 'empty order'; Order = @() }
    ) {
        param($Order)
        { Set-LLMProviderOrder -Order $Order -ErrorAction Stop } | Should -Throw
        (Get-FileHash -LiteralPath $script:ConfigPath).Hash | Should -Be $script:BeforeHash
        InModuleScope PC-AI.LLM { $script:ModuleConfig.ProviderOrder -join ',' } | Should -Be 'ollama'
    }

    It 'leaves disk and memory unchanged under WhatIf' {
        Set-LLMProviderOrder -Order @('vllm') -WhatIf
        (Get-FileHash -LiteralPath $script:ConfigPath).Hash | Should -Be $script:BeforeHash
        InModuleScope PC-AI.LLM { $script:ModuleConfig.ProviderOrder -join ',' } | Should -Be 'ollama'
        Test-Path (Get-ConfigFixtureReceipt) | Should -BeFalse
    }

    It 'does not change memory or create files when the config is missing' {
        $missingPath = Join-Path $TestDrive 'missing-llm-config.json'
        InModuleScope PC-AI.LLM -Parameters @{ ConfigPath = $missingPath } {
            $script:ModuleConfig.ConfigPath = $ConfigPath
            $script:ModuleConfig.ProjectConfigPath = $ConfigPath
        }
        { Set-LLMProviderOrder -Order @('pcai-inference') -ErrorAction Stop } | Should -Throw "Config file not found: $missingPath"
        Test-Path -LiteralPath $missingPath | Should -BeFalse
        InModuleScope PC-AI.LLM { $script:ModuleConfig.ProviderOrder -join ',' } | Should -Be 'ollama'
    }

    It 'leaves malformed JSON and memory unchanged' {
        '{broken' | Set-Content -LiteralPath $script:ConfigPath -Encoding utf8NoBOM
        $before = (Get-FileHash -LiteralPath $script:ConfigPath).Hash
        { Set-LLMProviderOrder -Order @('pcai-inference') -ErrorAction Stop } | Should -Throw
        (Get-FileHash -LiteralPath $script:ConfigPath).Hash | Should -Be $before
        InModuleScope PC-AI.LLM { $script:ModuleConfig.ProviderOrder -join ',' } | Should -Be 'ollama'
    }

    It 'preserves a machine extension deeper than the predecessor serializer limit at depth <Depth>' -TestCases @(@{ Depth = 26 }, @{ Depth = 96 }) {
        param($Depth)
        $deep = ('{"nested":' * $Depth) + '"provider-extension-leaf"' + ('}' * $Depth)
        ('{"fallbackOrder":["ollama"],"providers":{},"machineSetting":' + $deep + '}') |
            Set-Content -LiteralPath $script:ConfigPath -Encoding utf8NoBOM
        Set-LLMProviderOrder -Order @('pcai-inference') -ErrorAction Stop | Out-Null
        $saved = Get-Content -LiteralPath $script:ConfigPath -Raw | ConvertFrom-Json
        $value = $saved.machineSetting
        for ($level = 0; $level -lt $Depth; $level++) { $value = $value.nested }
        $value | Should -BeExactly 'provider-extension-leaf'
    }

    It 'rejects over-limit JSON before replacing original bytes or provider memory' {
        $deep = ('{"nested":' * 110) + '"must-remain-unmodified"' + ('}' * 110)
        ('{"fallbackOrder":["ollama"],"providers":{},"machineSetting":' + $deep + '}') |
            Set-Content -LiteralPath $script:ConfigPath -Encoding utf8NoBOM
        $beforeHash = (Get-FileHash -LiteralPath $script:ConfigPath).Hash
        { Set-LLMProviderOrder -Order @('pcai-inference') -ErrorAction Stop } | Should -Throw
        (Get-FileHash -LiteralPath $script:ConfigPath).Hash | Should -BeExactly $beforeHash
        InModuleScope PC-AI.LLM { $script:ModuleConfig.ProviderOrder -join ',' } | Should -Be 'ollama'
    }

    It 'does not publish the new memory order when a real file write fails' {
        # Acquire a real deny-share handle only after reading and staging, at
        # the last pre-publication hash boundary. File.Replace must then fail.
        Mock Get-LLMConfigCurrentHash {
            param($Path)
            $hash = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
            # Permit the publisher's retained read handle, but deny rename/delete.
            $script:ConfigWriteFixtureHandle = [IO.File]::Open($Path, 'Open', 'Read', 'ReadWrite')
            return $hash
        } -ModuleName PC-AI.LLM -ParameterFilter { [IO.Path]::GetFileName($Path) -like 'llm-config*.json' }
        try {
            $failure = { Set-LLMProviderOrder -Order @('pcai-inference') -ErrorAction Stop } | Should -Throw -PassThru
            $failure.Exception.GetBaseException() | Should -BeOfType ([System.IO.IOException])
            $failure.Exception.Message | Should -Match 'Replace'
            InModuleScope PC-AI.LLM { $script:ModuleConfig.ProviderOrder -join ',' } | Should -Be 'ollama'
        } finally {
            if ($script:ConfigWriteFixtureHandle) { $script:ConfigWriteFixtureHandle.Dispose(); $script:ConfigWriteFixtureHandle = $null }
        }
        (Get-FileHash -LiteralPath $script:ConfigPath).Hash | Should -Be $script:BeforeHash
        $receipt = Get-Content (Get-ConfigFixtureReceipt) -Raw | ConvertFrom-Json
        $receipt.State | Should -BeExactly 'FailedBeforePublish'
        (Get-FileHash (Join-Path (Split-Path -Parent (Get-ConfigFixtureReceipt)) 'original.bin')).Hash | Should -BeExactly $script:BeforeHash
    }

    It 'retains exact original bytes under private permissions and a nonsensitive receipt' {
        Set-LLMProviderOrder -Order @('pcai-inference') -ErrorAction Stop | Out-Null
        $receiptPath = Get-ConfigFixtureReceipt
        $receipt = Get-Content $receiptPath -Raw | ConvertFrom-Json
        $receipt.State | Should -BeExactly 'Published'
        $receipt.OriginalSHA256 | Should -BeExactly $script:BeforeHash
        $receipt.DisplacedSHA256 | Should -BeExactly $script:BeforeHash
        $receipt.PublishedObservedSHA256 | Should -BeExactly (Get-FileHash $script:ConfigPath).Hash
        foreach ($name in @('original.bin', 'displaced-original.bin')) {
            $file = Join-Path (Split-Path -Parent $receiptPath) $name
            (Get-FileHash $file).Hash | Should -BeExactly $script:BeforeHash
            if ($IsWindows) {
                $acl = Get-Acl $file
                $acl.AreAccessRulesProtected | Should -BeTrue
                $allowed = @([Security.Principal.WindowsIdentity]::GetCurrent().User.Value, 'S-1-5-18', 'S-1-5-32-544')
                foreach ($rule in $acl.Access) { $rule.IdentityReference.Translate([Security.Principal.SecurityIdentifier]).Value | Should -BeIn $allowed }
            } else {
                [IO.File]::GetUnixFileMode($file) | Should -Be ([IO.UnixFileMode]'UserRead,UserWrite')
            }
        }
        (Get-Content $receiptPath -Raw) | Should -Not -Match 'machine-model|machine-tools|num_ctx'
    }

    It 'rejects serializer depth warnings before creating custody or publishing memory' {
        InModuleScope PC-AI.LLM {
            $snapshot = Read-LLMConfigSnapshot -Path $script:ModuleConfig.ProjectConfigPath
            $deep = ('{"nested":' * 110) + '"must-not-be-stringified"' + ('}' * 110)
            $snapshot.Configuration.machineSetting = $deep | ConvertFrom-Json -Depth 200
            { Save-LLMConfigAtomically -Configuration $snapshot.Configuration -Snapshot $snapshot } | Should -Throw
        }
        (Get-FileHash $script:ConfigPath).Hash | Should -BeExactly $script:BeforeHash
        Test-Path (Get-ConfigFixtureReceipt) | Should -BeFalse
        InModuleScope PC-AI.LLM { $script:ModuleConfig.ProviderOrder -join ',' } | Should -Be 'ollama'
    }

    It 'rejects changed staged bytes before publication' {
        Mock Get-LLMConfigCurrentHash {
            param($Path)
            [IO.File]::WriteAllText($Path, '{"tampered":true}', [Text.UTF8Encoding]::new($false))
            (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
        } -ModuleName PC-AI.LLM -ParameterFilter { [IO.Path]::GetFileName($Path) -eq 'staged.json' }
        { Set-LLMProviderOrder -Order @('pcai-inference') -ErrorAction Stop } | Should -Throw '*Staged configuration hash mismatch*'
        (Get-FileHash $script:ConfigPath).Hash | Should -BeExactly $script:BeforeHash
        (Get-Content (Get-ConfigFixtureReceipt) -Raw | ConvertFrom-Json).State | Should -BeExactly 'FailedBeforePublish'
        InModuleScope PC-AI.LLM { $script:ModuleConfig.ProviderOrder -join ',' } | Should -Be 'ollama'
    }

    It 'refuses a writer that changes the original before the final guard' {
        Mock Get-LLMConfigCurrentHash {
            param($Path)
            [IO.File]::WriteAllText($Path, '{"writer":"before-guard"}', [Text.UTF8Encoding]::new($false))
            (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
        } -ModuleName PC-AI.LLM -ParameterFilter { [IO.Path]::GetFileName($Path) -like 'llm-config*.json' }
        { Set-LLMProviderOrder -Order @('pcai-inference') -ErrorAction Stop } | Should -Throw '*changed after reading*'
        (Get-Content $script:ConfigPath -Raw | ConvertFrom-Json).writer | Should -BeExactly 'before-guard'
        (Get-Content (Get-ConfigFixtureReceipt) -Raw | ConvertFrom-Json).State | Should -BeExactly 'FailedBeforePublish'
        InModuleScope PC-AI.LLM { $script:ModuleConfig.ProviderOrder -join ',' } | Should -Be 'ollama'
    }

    It 'captures and restores an intervening writer at the actual replacement boundary' {
        $script:ConfigWriteFixtureCalls = 0
        Mock Get-LLMConfigCurrentHash {
            param($Path)
            $hash = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
            $script:ConfigWriteFixtureCalls++
            if ($script:ConfigWriteFixtureCalls -eq 1) { [IO.File]::WriteAllText($Path, '{"writer":"at-boundary"}', [Text.UTF8Encoding]::new($false)) }
            return $hash
        } -ModuleName PC-AI.LLM -ParameterFilter { [IO.Path]::GetFileName($Path) -like 'llm-config*.json' }
        { Set-LLMProviderOrder -Order @('pcai-inference') -ErrorAction Stop } | Should -Throw '*replacement boundary*'
        (Get-Content $script:ConfigPath -Raw | ConvertFrom-Json).writer | Should -BeExactly 'at-boundary'
        $receiptPath = Get-ConfigFixtureReceipt
        $receipt = Get-Content $receiptPath -Raw | ConvertFrom-Json
        $receipt.State | Should -BeExactly 'RolledBack'
        $receipt.DisplacedSHA256 | Should -BeExactly (Get-FileHash $script:ConfigPath).Hash
        $receipt.Recovery.Count | Should -Be 1
        (Get-FileHash (Join-Path (Split-Path -Parent $receiptPath) 'original.bin')).Hash | Should -BeExactly $script:BeforeHash
        InModuleScope PC-AI.LLM { $script:ModuleConfig.ProviderOrder -join ',' } | Should -Be 'ollama'
    }

    It 'preserves a later current writer without attempting recovery over it' {
        $script:ConfigWriteFixtureCalls = 0
        Mock Get-LLMConfigCurrentHash {
            param($Path)
            $script:ConfigWriteFixtureCalls++
            if ($script:ConfigWriteFixtureCalls -eq 2) { [IO.File]::WriteAllText($Path, '{"writer":"after-publication"}', [Text.UTF8Encoding]::new($false)) }
            (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
        } -ModuleName PC-AI.LLM -ParameterFilter { [IO.Path]::GetFileName($Path) -like 'llm-config*.json' }
        { Set-LLMProviderOrder -Order @('pcai-inference') -ErrorAction Stop } | Should -Throw '*current writer bytes are retained*'
        (Get-Content $script:ConfigPath -Raw | ConvertFrom-Json).writer | Should -BeExactly 'after-publication'
        $receipt = Get-Content (Get-ConfigFixtureReceipt) -Raw | ConvertFrom-Json
        $receipt.State | Should -BeExactly 'LaterWriterPreserved'
        $receipt.FailureCurrentSHA256 | Should -BeExactly (Get-FileHash $script:ConfigPath).Hash
        $receipt.Recovery.Count | Should -Be 0
        InModuleScope PC-AI.LLM { $script:ModuleConfig.ProviderOrder -join ',' } | Should -Be 'ollama'
    }

    It 'refuses linked configuration paths without changing their targets' {
        $link = Join-Path $TestDrive ('linked-config-' + [guid]::NewGuid().ToString('N') + '.json')
        New-Item -ItemType SymbolicLink -Path $link -Target $script:ConfigPath -ErrorAction Stop | Out-Null
        try {
            InModuleScope PC-AI.LLM -Parameters @{ Path = $link } { $script:ModuleConfig.ProjectConfigPath = $Path }
            { Set-LLMProviderOrder -Order @('pcai-inference') -ErrorAction Stop } | Should -Throw '*linked paths*'
            (Get-FileHash $script:ConfigPath).Hash | Should -BeExactly $script:BeforeHash
        } finally { Remove-Item -LiteralPath $link -Force }
    }

    It 'refuses hardlinked configuration files without changing either alias' {
        $link = Join-Path $TestDrive ('hardlinked-config-' + [guid]::NewGuid().ToString('N') + '.json')
        New-Item -ItemType HardLink -Path $link -Target $script:ConfigPath -ErrorAction Stop | Out-Null
        (Get-Item $script:ConfigPath).LinkType | Should -BeExactly 'HardLink'
        { Set-LLMProviderOrder -Order @('pcai-inference') -ErrorAction Stop } | Should -Throw '*linked paths*'
        (Get-FileHash $link).Hash | Should -BeExactly $script:BeforeHash
        (Get-FileHash $script:ConfigPath).Hash | Should -BeExactly $script:BeforeHash
        Test-Path (Get-ConfigFixtureReceipt) | Should -BeFalse
        InModuleScope PC-AI.LLM { $script:ModuleConfig.ProviderOrder -join ',' } | Should -Be 'ollama'
    }

    It 'rejects a <Label> JSON root without changing bytes or module memory' -TestCases @(
        @{ Label='one-object array'; Json='[{"fallbackOrder":["ollama"]}]' }
        @{ Label='empty array'; Json='[]' }
        @{ Label='string'; Json='"configuration"' }
        @{ Label='null'; Json='null' }
    ) {
        param($Json)
        [IO.File]::WriteAllText($script:ConfigPath, $Json, [Text.UTF8Encoding]::new($false))
        $before = (Get-FileHash $script:ConfigPath).Hash
        { Set-LLMProviderOrder -Order @('pcai-inference') -ErrorAction Stop } | Should -Throw '*JSON object*'
        (Get-FileHash $script:ConfigPath).Hash | Should -BeExactly $before
        Test-Path (Get-ConfigFixtureReceipt) | Should -BeFalse
        InModuleScope PC-AI.LLM { $script:ModuleConfig.ProviderOrder -join ',' } | Should -Be 'ollama'
    }

    It 'refuses to publish an out-of-scope snapshot' {
        $otherPath = Join-Path $TestDrive 'unselected-config.json'
        [IO.File]::WriteAllText($otherPath, '{"unrelated":true}', [Text.UTF8Encoding]::new($false))
        $before = (Get-FileHash $otherPath).Hash
        InModuleScope PC-AI.LLM -Parameters @{ OtherPath=$otherPath } {
            $snapshot = Read-LLMConfigSnapshot -Path $OtherPath
            $snapshot.Configuration.unrelated = $false
            { Save-LLMConfigAtomically -Configuration $snapshot.Configuration -Snapshot $snapshot } | Should -Throw '*outside the selected module*'
        }
        (Get-FileHash $otherPath).Hash | Should -BeExactly $before
        (Get-FileHash $script:ConfigPath).Hash | Should -BeExactly $script:BeforeHash
    }

    It 'refuses publication while another cooperating writer holds the exclusive lock' {
        Set-LLMProviderOrder -Order @('ollama') -ErrorAction Stop | Out-Null
        $before = (Get-FileHash $script:ConfigPath).Hash
        $lockPath = Join-Path (Split-Path -Parent (Split-Path -Parent (Get-ConfigFixtureReceipt))) 'publish.lock'
        $handle = [IO.File]::Open($lockPath, 'Open', 'ReadWrite', 'None')
        try {
            $failure = { Set-LLMProviderOrder -Order @('pcai-inference') -ErrorAction Stop } | Should -Throw -PassThru
            $failure.Exception.GetBaseException() | Should -BeOfType ([IO.IOException])
            (Get-FileHash $script:ConfigPath).Hash | Should -BeExactly $before
            InModuleScope PC-AI.LLM { $script:ModuleConfig.ProviderOrder -join ',' } | Should -Be 'ollama'
            Test-Path (Join-Path (Split-Path -Parent (Split-Path -Parent (Get-ConfigFixtureReceipt))) 'r2') | Should -BeFalse
        } finally { $handle.Dispose() }
    }

    It 'captures a second writer racing during recovery and restores its actual bytes' {
        $script:ConfigWriteFixtureCalls = 0
        Mock Get-LLMConfigCurrentHash {
            param($Path)
            $hash = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
            $script:ConfigWriteFixtureCalls++
            if ($script:ConfigWriteFixtureCalls -eq 1) { [IO.File]::WriteAllText($Path, '{"writer":"boundary-first"}', [Text.UTF8Encoding]::new($false)) }
            if ($script:ConfigWriteFixtureCalls -eq 2) { [IO.File]::WriteAllText($Path, '{"writer":"recovery-second"}', [Text.UTF8Encoding]::new($false)) }
            return $hash
        } -ModuleName PC-AI.LLM -ParameterFilter { [IO.Path]::GetFileName($Path) -like 'llm-config*.json' }
        { Set-LLMProviderOrder -Order @('pcai-inference') -ErrorAction Stop } | Should -Throw '*replacement boundary*'
        (Get-Content $script:ConfigPath -Raw | ConvertFrom-Json).writer | Should -BeExactly 'recovery-second'
        $receipt = Get-Content (Get-ConfigFixtureReceipt) -Raw | ConvertFrom-Json
        $receipt.State | Should -BeExactly 'RolledBack'
        $receipt.Recovery.Count | Should -Be 2
        $receipt.Recovery[1].RestoredSHA256 | Should -BeExactly (Get-FileHash $script:ConfigPath).Hash
        InModuleScope PC-AI.LLM { $script:ModuleConfig.ProviderOrder -join ',' } | Should -Be 'ollama'
    }

    It 'restores an original actually displaced before a replacement exception, preserving its Windows ACL' {
        $originalAcl = (Get-Acl $script:ConfigPath).Sddl
        $script:ConfigWriteFixturePartialFault = $false
        Mock Get-LLMConfigCurrentHash {
            param($Path)
            if (-not $script:ConfigWriteFixturePartialFault) {
                $script:ConfigWriteFixturePartialFault = $true
                $backup = Join-Path (Split-Path -Parent (Get-ConfigFixtureReceipt)) 'displaced-original.bin'
                [IO.File]::Move($Path, $backup)
                throw [IO.IOException]::new('Fixture mimics documented ReplaceFile error1177 after actual original displacement.', -2147023719)
            }
            (Get-FileHash $Path).Hash
        } -ModuleName PC-AI.LLM -ParameterFilter { [IO.Path]::GetFileName($Path) -like 'llm-config*.json' }
        { Set-LLMProviderOrder -Order @('pcai-inference') -ErrorAction Stop } | Should -Throw '*error1177*'
        $receipt = Get-Content (Get-ConfigFixtureReceipt) -Raw | ConvertFrom-Json
        if ($receipt.State -ne 'RolledBack') { $receipt | ConvertTo-Json -Depth 6 | Write-Host }
        (Get-FileHash $script:ConfigPath).Hash | Should -BeExactly $script:BeforeHash
        (Get-Acl $script:ConfigPath).Sddl | Should -BeExactly $originalAcl
        $receiptPath = Get-ConfigFixtureReceipt
        $receipt = Get-Content $receiptPath -Raw | ConvertFrom-Json
        $receipt.PartialPublicationObserved | Should -BeTrue
        $receipt.State | Should -BeExactly 'RolledBack'
        $receipt.RecoveryState | Should -BeExactly 'Restored'
        $receipt.DisplacedSHA256 | Should -BeExactly $script:BeforeHash
        $receipt.OriginalFileIdentity | Should -BeExactly $receipt.DisplacedFileIdentity
        $receipt.OriginalWindowsDescriptorSHA256 | Should -BeExactly (Get-FileHash (Join-Path (Split-Path -Parent $receiptPath) 'original-security-descriptor.bin')).Hash
        $receipt.DisplacedWindowsDescriptorSHA256 | Should -BeExactly (Get-FileHash (Join-Path (Split-Path -Parent $receiptPath) 'displaced-security-descriptor.bin')).Hash
        (Get-FileHash (Join-Path (Split-Path -Parent $receiptPath) 'displaced-original.bin')).Hash | Should -BeExactly $script:BeforeHash
        InModuleScope PC-AI.LLM { $script:ModuleConfig.ProviderOrder -join ',' } | Should -Be 'ollama'
    }

    It 'lets a later writer win the missing-target restoration race without overwriting or moving custody' {
        $script:ConfigWriteFixturePartialFault = $false
        Mock Get-LLMConfigCurrentHash {
            param($Path)
            if (-not $script:ConfigWriteFixturePartialFault) {
                $script:ConfigWriteFixturePartialFault = $true
                [IO.File]::Move($Path, (Join-Path (Split-Path -Parent (Get-ConfigFixtureReceipt)) 'displaced-original.bin'))
                throw [IO.IOException]::new('Fixture error1177 after actual original displacement.', -2147023719)
            }
            (Get-FileHash $Path).Hash
        } -ModuleName PC-AI.LLM -ParameterFilter { [IO.Path]::GetFileName($Path) -like 'llm-config*.json' }
        Mock Get-LLMConfigCurrentHash {
            param($Path)
            $hash = (Get-FileHash $Path).Hash
            [IO.File]::WriteAllText($script:ConfigPath, '{"writer":"missing-target-race"}', [Text.UTF8Encoding]::new($false))
            return $hash
        } -ModuleName PC-AI.LLM -ParameterFilter { [IO.Path]::GetFileName($Path) -eq 'recovery-missing-target.json' }
        { Set-LLMProviderOrder -Order @('pcai-inference') -ErrorAction Stop } | Should -Throw '*error1177*'
        (Get-Content $script:ConfigPath -Raw | ConvertFrom-Json).writer | Should -BeExactly 'missing-target-race'
        $receiptPath = Get-ConfigFixtureReceipt
        $receipt = Get-Content $receiptPath -Raw | ConvertFrom-Json
        $receipt.State | Should -BeExactly 'LaterWriterPreserved'
        $receipt.FailureCurrentSHA256 | Should -BeExactly (Get-FileHash $script:ConfigPath).Hash
        (Get-FileHash (Join-Path (Split-Path -Parent $receiptPath) 'displaced-original.bin')).Hash | Should -BeExactly $script:BeforeHash
        (Get-FileHash (Join-Path (Split-Path -Parent $receiptPath) 'original.bin')).Hash | Should -BeExactly $script:BeforeHash
        InModuleScope PC-AI.LLM { $script:ModuleConfig.ProviderOrder -join ',' } | Should -Be 'ollama'
    }

    It 'keeps the original exception and review state when owned-handle ACL recovery fails' {
        $script:ConfigWriteFixturePartialFault = $false
        Mock Get-LLMConfigCurrentHash {
            param($Path)
            if (-not $script:ConfigWriteFixturePartialFault) {
                $script:ConfigWriteFixturePartialFault = $true
                [IO.File]::Move($Path, (Join-Path (Split-Path -Parent (Get-ConfigFixtureReceipt)) 'displaced-original.bin'))
                throw [IO.IOException]::new('Original fixture error1177.', -2147023719)
            }
            (Get-FileHash $Path).Hash
        } -ModuleName PC-AI.LLM -ParameterFilter { [IO.Path]::GetFileName($Path) -like 'llm-config*.json' }
        Mock Set-LLMConfigOwnedRecoveryAcl { throw [UnauthorizedAccessException]::new('Fixture ACL recovery denied.') } -ModuleName PC-AI.LLM
        $failure = { Set-LLMProviderOrder -Order @('pcai-inference') -ErrorAction Stop } | Should -Throw '*Original fixture error1177*' -PassThru
        $failure.Exception.GetBaseException().HResult | Should -Be -2147023719
        (Get-FileHash $script:ConfigPath).Hash | Should -BeExactly $script:BeforeHash
        $receipt = Get-Content (Get-ConfigFixtureReceipt) -Raw | ConvertFrom-Json
        $receipt.State | Should -BeExactly 'RecoveryRequiresReview'
        $receipt.RecoveryErrorType | Should -BeExactly 'System.UnauthorizedAccessException'
        InModuleScope PC-AI.LLM { $script:ModuleConfig.ProviderOrder -join ',' } | Should -Be 'ollama'
    }

    It 'restores ACLs only through the owned handle after a different writer replaces the target' {
        $script:ConfigWriteFixturePartialFault = $false
        $script:ConfigWriteFixtureLaterAcl = $null
        Mock Get-LLMConfigCurrentHash {
            param($Path)
            if (-not $script:ConfigWriteFixturePartialFault) {
                $script:ConfigWriteFixturePartialFault = $true
                [IO.File]::Move($Path, (Join-Path (Split-Path -Parent (Get-ConfigFixtureReceipt)) 'displaced-original.bin'))
                throw [IO.IOException]::new('Original fixture error1177.', -2147023719)
            }
            (Get-FileHash $Path).Hash
        } -ModuleName PC-AI.LLM -ParameterFilter { [IO.Path]::GetFileName($Path) -like 'llm-config*.json' }
        Mock Set-LLMConfigOwnedRecoveryAcl {
            param($Stream, $Acl)
            $aside = Join-Path (Split-Path -Parent (Get-ConfigFixtureReceipt)) 'recovery-owned-aside.bin'
            [IO.File]::Move($script:ConfigPath, $aside)
            [IO.File]::WriteAllText($script:ConfigPath, '{"writer":"after-recovery-move"}', [Text.UTF8Encoding]::new($false))
            $script:ConfigWriteFixtureLaterAcl = (Get-Acl $script:ConfigPath).Sddl
            [IO.FileSystemAclExtensions]::SetAccessControl($Stream, $Acl)
        } -ModuleName PC-AI.LLM
        { Set-LLMProviderOrder -Order @('pcai-inference') -ErrorAction Stop } | Should -Throw '*error1177*'
        (Get-Content $script:ConfigPath -Raw | ConvertFrom-Json).writer | Should -BeExactly 'after-recovery-move'
        (Get-Acl $script:ConfigPath).Sddl | Should -BeExactly $script:ConfigWriteFixtureLaterAcl
        $receiptPath = Get-ConfigFixtureReceipt
        $receipt = Get-Content $receiptPath -Raw | ConvertFrom-Json
        $receipt.State | Should -BeExactly 'LaterWriterPreserved'
        $receipt.FailureCurrentSHA256 | Should -BeExactly (Get-FileHash $script:ConfigPath).Hash
        (Get-FileHash (Join-Path (Split-Path -Parent $receiptPath) 'recovery-owned-aside.bin')).Hash | Should -BeExactly $script:BeforeHash
        (Get-FileHash (Join-Path (Split-Path -Parent $receiptPath) 'displaced-original.bin')).Hash | Should -BeExactly $script:BeforeHash
        InModuleScope PC-AI.LLM { $script:ModuleConfig.ProviderOrder -join ',' } | Should -Be 'ollama'
    }

    It 'preserves exact <Mode> ACL controls and ACEs after actual displacement (repeat <Repeat>)' -ForEach @(
        @{ Mode='legacy inherited'; AutoInherited=$false; Protected=$false; Repeat=1 }
        @{ Mode='legacy protected'; AutoInherited=$false; Protected=$true; Repeat=1 }
        @{ Mode='auto inherited'; AutoInherited=$true; Protected=$false; Repeat=1 }
        @{ Mode='auto protected'; AutoInherited=$true; Protected=$true; Repeat=1 }
        @{ Mode='legacy inherited'; AutoInherited=$false; Protected=$false; Repeat=2 }
        @{ Mode='legacy protected'; AutoInherited=$false; Protected=$true; Repeat=2 }
        @{ Mode='auto inherited'; AutoInherited=$true; Protected=$false; Repeat=2 }
        @{ Mode='auto protected'; AutoInherited=$true; Protected=$true; Repeat=2 }
    ) {
        # Establish the original descriptor through an independent native fixture.
        # The expected metadata is observed before the production recovery runs.
        $raw = [Security.AccessControl.RawSecurityDescriptor]::new((Get-Acl $script:ConfigPath).Sddl)
        $flags = $raw.ControlFlags -band (-bnot ([Security.AccessControl.ControlFlags]::DiscretionaryAclAutoInherited -bor [Security.AccessControl.ControlFlags]::DiscretionaryAclProtected))
        if ($Protected) { $flags = $flags -bor [Security.AccessControl.ControlFlags]::DiscretionaryAclProtected }
        if ($AutoInherited) { $flags = $flags -bor [Security.AccessControl.ControlFlags]::DiscretionaryAclAutoInherited -bor [Security.AccessControl.ControlFlags]::DiscretionaryAclAutoInheritRequired }
        $raw.SetFlags($flags)
        $descriptor = [byte[]]::new($raw.BinaryLength)
        $raw.GetBinaryForm($descriptor, 0)
        $fixture = [IO.FileSystemAclExtensions]::Create([IO.FileInfo]::new($script:ConfigPath), [IO.FileMode]::Open,
            [Security.AccessControl.FileSystemRights]'ReadData,ReadPermissions,ChangePermissions,TakeOwnership', [IO.FileShare]'ReadWrite,Delete', 4096, [IO.FileOptions]::None, $null)
        try { [Pcai.Config.AclFixtureV1]::NtSetSecurityObject($fixture.SafeFileHandle, 7, $descriptor) | Should -Be 0 }
        finally { $fixture.Dispose() }
        $originalAcl = (Get-Acl $script:ConfigPath).Sddl
        $expected = [Security.AccessControl.RawSecurityDescriptor]::new($originalAcl)
        [bool]($expected.ControlFlags -band [Security.AccessControl.ControlFlags]::DiscretionaryAclAutoInherited) | Should -Be $AutoInherited
        [bool]($expected.ControlFlags -band [Security.AccessControl.ControlFlags]::DiscretionaryAclProtected) | Should -Be $Protected
        $script:ConfigWriteFixturePartialFault = $false
        Mock Get-LLMConfigCurrentHash {
            param($Path)
            if (-not $script:ConfigWriteFixturePartialFault) {
                $script:ConfigWriteFixturePartialFault = $true
                [IO.File]::Move($Path, (Join-Path (Split-Path -Parent (Get-ConfigFixtureReceipt)) 'displaced-original.bin'))
                throw [IO.IOException]::new('Original fixture error1177 with controlled descriptor.', -2147023719)
            }
            (Get-FileHash $Path).Hash
        } -ModuleName PC-AI.LLM -ParameterFilter { [IO.Path]::GetFileName($Path) -like 'llm-config*.json' }
        { Set-LLMProviderOrder -Order @('pcai-inference') -ErrorAction Stop } | Should -Throw '*error1177*'
        (Get-Acl $script:ConfigPath).Sddl | Should -BeExactly $originalAcl
        (Get-FileHash $script:ConfigPath).Hash | Should -BeExactly $script:BeforeHash
        $receiptPath = Get-ConfigFixtureReceipt
        $receipt = Get-Content $receiptPath -Raw | ConvertFrom-Json
        $receipt.State | Should -BeExactly 'RolledBack'
        $receipt.RecoveryState | Should -BeExactly 'Restored'
        $receipt.OriginalFileIdentity | Should -BeExactly $receipt.DisplacedFileIdentity
        $receipt.Recovery[0].MetadataSource | Should -BeExactly 'IdentityBoundPrePublicationWindowsAcl'
        foreach ($leaf in @('original-security-descriptor.bin', 'displaced-security-descriptor.bin')) {
            $acl = Get-Acl (Join-Path (Split-Path -Parent $receiptPath) $leaf)
            $acl.AreAccessRulesProtected | Should -BeTrue
            $allowed = @([Security.Principal.WindowsIdentity]::GetCurrent().User.Value, 'S-1-5-18', 'S-1-5-32-544')
            foreach ($rule in $acl.Access) { $rule.IdentityReference.Translate([Security.Principal.SecurityIdentifier]).Value | Should -BeIn $allowed }
        }
    }

    It 'retains both descriptors and refuses stale metadata for a foreign displaced file with identical bytes' {
        $script:ConfigWriteFixturePartialFault = $false
        Mock Get-LLMConfigCurrentHash {
            param($Path)
            if (-not $script:ConfigWriteFixturePartialFault) {
                $script:ConfigWriteFixturePartialFault = $true
                $transaction = Split-Path -Parent (Get-ConfigFixtureReceipt)
                $bytes = [IO.File]::ReadAllBytes($Path)
                [IO.File]::Move($Path, (Join-Path $transaction 'fixture-retained-original.bin'))
                [IO.File]::WriteAllBytes($Path, $bytes)
                $foreignAcl = Get-Acl -LiteralPath $Path
                $foreignAcl.SetAccessRuleProtection($true, $true)
                Set-Acl -LiteralPath $Path -AclObject $foreignAcl -ErrorAction Stop
                [IO.File]::Move($Path, (Join-Path $transaction 'displaced-original.bin'))
                throw [IO.IOException]::new('Original fixture error1177 with foreign displaced writer.', -2147023719)
            }
            (Get-FileHash $Path).Hash
        } -ModuleName PC-AI.LLM -ParameterFilter { [IO.Path]::GetFileName($Path) -like 'llm-config*.json' }
        { Set-LLMProviderOrder -Order @('pcai-inference') -ErrorAction Stop } | Should -Throw '*Original fixture error1177*'
        Test-Path $script:ConfigPath | Should -BeFalse
        $receiptPath = Get-ConfigFixtureReceipt
        $transaction = Split-Path -Parent $receiptPath
        $receipt = Get-Content $receiptPath -Raw | ConvertFrom-Json
        $receipt.State | Should -BeExactly 'RecoveryRequiresReview'
        $receipt.OriginalFileIdentity | Should -Not -Be $receipt.DisplacedFileIdentity
        $receipt.DisplacedSHA256 | Should -BeExactly $script:BeforeHash
        foreach ($leaf in @('displaced-original.bin', 'fixture-retained-original.bin')) {
            (Get-FileHash (Join-Path $transaction $leaf)).Hash | Should -BeExactly $script:BeforeHash
        }
        $receipt.OriginalWindowsDescriptorSHA256 | Should -BeExactly (Get-FileHash (Join-Path $transaction 'original-security-descriptor.bin')).Hash
        $receipt.DisplacedWindowsDescriptorSHA256 | Should -BeExactly (Get-FileHash (Join-Path $transaction 'displaced-security-descriptor.bin')).Hash
        InModuleScope PC-AI.LLM { $script:ModuleConfig.ProviderOrder -join ',' } | Should -Be 'ollama'
    }

    It 'keeps review state and the original failure when exact descriptor restoration is not achieved' {
        $script:ConfigWriteFixturePartialFault = $false
        Mock Get-LLMConfigCurrentHash {
            param($Path)
            if (-not $script:ConfigWriteFixturePartialFault) {
                $script:ConfigWriteFixturePartialFault = $true
                [IO.File]::Move($Path, (Join-Path (Split-Path -Parent (Get-ConfigFixtureReceipt)) 'displaced-original.bin'))
                throw [IO.IOException]::new('Original fixture error1177 before descriptor mismatch.', -2147023719)
            }
            (Get-FileHash $Path).Hash
        } -ModuleName PC-AI.LLM -ParameterFilter { [IO.Path]::GetFileName($Path) -like 'llm-config*.json' }
        Mock Set-LLMConfigOwnedRecoveryAcl {} -ModuleName PC-AI.LLM
        $failure = { Set-LLMProviderOrder -Order @('pcai-inference') -ErrorAction Stop } | Should -Throw '*Original fixture error1177*' -PassThru
        $failure.Exception.GetBaseException().HResult | Should -Be -2147023719
        $receiptPath = Get-ConfigFixtureReceipt
        $receipt = Get-Content $receiptPath -Raw | ConvertFrom-Json
        $receipt.State | Should -BeExactly 'RecoveryRequiresReview'
        $receipt.RecoveryState | Should -BeExactly 'RequiresReview'
        (Get-FileHash (Join-Path (Split-Path -Parent $receiptPath) 'displaced-original.bin')).Hash | Should -BeExactly $script:BeforeHash
        Test-Path (Join-Path (Split-Path -Parent $receiptPath) 'original-security-descriptor.bin') | Should -BeTrue
    }

    It 'preserves a distinct later target with identical bytes and its own ACL after the owned recovery move' {
        $script:ConfigWriteFixturePartialFault = $false
        $script:ConfigWriteFixtureForeignAcl = $null
        $script:ConfigWriteFixtureRealSetter = & (Get-Module PC-AI.LLM) { (Get-Command Set-LLMConfigOwnedRecoveryAcl).ScriptBlock }
        Mock Get-LLMConfigCurrentHash {
            param($Path)
            if (-not $script:ConfigWriteFixturePartialFault) {
                $script:ConfigWriteFixturePartialFault = $true
                [IO.File]::Move($Path, (Join-Path (Split-Path -Parent (Get-ConfigFixtureReceipt)) 'displaced-original.bin'))
                throw [IO.IOException]::new('Original fixture error1177 before a same-byte later writer.', -2147023719)
            }
            (Get-FileHash $Path).Hash
        } -ModuleName PC-AI.LLM -ParameterFilter { [IO.Path]::GetFileName($Path) -like 'llm-config*.json' }
        Mock Set-LLMConfigOwnedRecoveryAcl {
            param($Stream, $Acl, $Descriptor)
            $transaction = Split-Path -Parent (Get-ConfigFixtureReceipt)
            $bytes = [IO.File]::ReadAllBytes($script:ConfigPath)
            [IO.File]::Move($script:ConfigPath, (Join-Path $transaction 'fixture-owned-recovery-aside.bin'))
            [IO.File]::WriteAllBytes($script:ConfigPath, $bytes)
            $foreignAcl = Get-Acl -LiteralPath $script:ConfigPath
            $foreignAcl.SetAccessRuleProtection($true, $true)
            Set-Acl -LiteralPath $script:ConfigPath -AclObject $foreignAcl -ErrorAction Stop
            $script:ConfigWriteFixtureForeignAcl = (Get-Acl $script:ConfigPath).Sddl
            & $script:ConfigWriteFixtureRealSetter -Stream $Stream -Acl $Acl -Descriptor $Descriptor
        } -ModuleName PC-AI.LLM
        { Set-LLMProviderOrder -Order @('pcai-inference') -ErrorAction Stop } | Should -Throw '*Original fixture error1177*'
        (Get-FileHash $script:ConfigPath).Hash | Should -BeExactly $script:BeforeHash
        (Get-Acl $script:ConfigPath).Sddl | Should -BeExactly $script:ConfigWriteFixtureForeignAcl
        $receiptPath = Get-ConfigFixtureReceipt
        $receipt = Get-Content $receiptPath -Raw | ConvertFrom-Json
        $receipt.State | Should -BeExactly 'LaterWriterPreserved'
        $receipt.RecoveryState | Should -BeExactly 'LaterWriterPreserved'
        (Get-FileHash (Join-Path (Split-Path -Parent $receiptPath) 'fixture-owned-recovery-aside.bin')).Hash | Should -BeExactly $script:BeforeHash
        InModuleScope PC-AI.LLM { $script:ModuleConfig.ProviderOrder -join ',' } | Should -Be 'ollama'
    }
}
