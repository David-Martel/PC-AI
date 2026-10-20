#Requires -Modules Pester

BeforeAll {
    $script:RepoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
    $script:ArchiveScript = Join-Path $script:RepoRoot 'Tools/Invoke-CleanupArchive.ps1'
    if (-not ('Pcai.Storage.PcaiArchiveSource' -as [type])) {
        Add-Type -Path (Join-Path $script:RepoRoot 'Tools/SystemScripts/Storage/PcaiArchiveSource.cs') -ErrorAction Stop
    }
    function rclone { throw 'External adapter must be mocked for fixture tests.' }
    Add-Type -TypeDefinition @'
using System.Collections.Generic;
public static class PcaiArchiveRcloneFixture {
    public static Dictionary<string, byte[]> Remote = new Dictionary<string, byte[]>();
    public static List<object> Calls = new List<object>();
    public static string Fault = "";
    public static string Manifest = "";
    public static string SizeJson = "";
    public static System.IO.FileStream ManifestLock;
}
'@
    function New-ArchiveFixture {
        param([string]$Name = 'source.bin')
        $path = Join-Path $script:CaseRoot $Name
        [IO.File]::WriteAllBytes($path, [Text.Encoding]::UTF8.GetBytes('original-byte-custody-fixture'))
        return $path
    }
    function Save-VerifiedFixtureReceipt {
        param($Source, [string]$Manifest)
        $json = @{ event = 'verified'; source_sha256 = $Source.Sha256; source_file_id = $Source.FileId128 } | ConvertTo-Json -Compress
        $Source.PersistVerifiedReceipt($Manifest, $json, $Source.Md5, $Source.Length, 1)
    }
    function Invoke-ArchiveFixtureProcess {
        param([string[]]$Arguments, [string]$Bin)
        $start = [Diagnostics.ProcessStartInfo]::new((Get-Process -Id $PID).Path)
        $start.UseShellExecute = $false; $start.RedirectStandardError = $true; $start.RedirectStandardOutput = $true
        foreach ($argument in @('-NoLogo', '-NoProfile') + $Arguments) { $start.ArgumentList.Add($argument) }
        if ($Bin) { $start.Environment['PATH'] = "$Bin;$($start.Environment['PATH'])" }
        $child = [Diagnostics.Process]::Start($start)
        try {
            $stdout = $child.StandardOutput.ReadToEndAsync(); $stderr = $child.StandardError.ReadToEndAsync()
            if (-not $child.WaitForExit(20000)) { throw 'Fixture CLI timed out.' }
            [pscustomobject]@{ ExitCode = $child.ExitCode; StdOut = $stdout.GetAwaiter().GetResult(); StdErr = $stderr.GetAwaiter().GetResult() }
        } finally {
            if (-not $child.HasExited) { $child.Kill($true); $child.WaitForExit() }
            $child.Dispose()
        }
    }
}

Describe 'Windows archive source byte and identity custody' -Tag 'Unit', 'Boot', 'WindowsNative' {
    BeforeEach {
        $script:CaseRoot = Join-Path $TestDrive ([guid]::NewGuid().ToString('N'))
        New-Item -ItemType Directory -Path $script:CaseRoot | Out-Null
        $script:SourcePath = New-ArchiveFixture
        $script:Manifest = Join-Path $script:CaseRoot 'receipt.jsonl'
    }

    It 'hashes held bytes exactly and permits a normal reader that does not share delete' {
        $expectedMd5 = (Get-FileHash -LiteralPath $script:SourcePath -Algorithm MD5).Hash.ToLowerInvariant()
        $expectedSha = (Get-FileHash -LiteralPath $script:SourcePath -Algorithm SHA256).Hash.ToLowerInvariant()
        $source = [Pcai.Storage.PcaiArchiveSource]::new($script:SourcePath)
        try {
            $source.Md5 | Should -BeExactly $expectedMd5
            $source.Sha256 | Should -BeExactly $expectedSha
            $source.Length | Should -Be 29
            $source.CanonicalGuidPath | Should -Match '^\\\\\?\\Volume\{'
            $reader = [IO.File]::Open($script:SourcePath, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::Read)
            try { $reader.ReadByte() | Should -Be ([byte][char]'o') } finally { $reader.Dispose() }
        } finally { $source.Dispose() }
    }

    It 'rejects an already open writer before taking source custody' {
        $writer = [IO.File]::Open($script:SourcePath, [IO.FileMode]::Open, [IO.FileAccess]::Write, [IO.FileShare]::ReadWrite)
        try { { [Pcai.Storage.PcaiArchiveSource]::new($script:SourcePath) } | Should -Throw '*custody*' } finally { $writer.Dispose() }
        [IO.File]::ReadAllText($script:SourcePath) | Should -BeExactly 'original-byte-custody-fixture'
    }

    It 'blocks new writers and renames while upload custody is held' {
        $source = [Pcai.Storage.PcaiArchiveSource]::new($script:SourcePath)
        $renamed = Join-Path $script:CaseRoot 'renamed.bin'
        try {
            { [IO.File]::OpenWrite($script:SourcePath) } | Should -Throw
            { [IO.File]::Move($script:SourcePath, $renamed) } | Should -Throw
            Test-Path -LiteralPath $renamed | Should -BeFalse
        } finally { $source.Dispose() }
        [IO.File]::ReadAllText($script:SourcePath) | Should -BeExactly 'original-byte-custody-fixture'
    }

    It 'requires a durable verified receipt before deletion' {
        $source = [Pcai.Storage.PcaiArchiveSource]::new($script:SourcePath)
        try { { $source.DeleteVerified() } | Should -Throw '*durable verified receipt*' } finally { $source.Dispose() }
        Test-Path -LiteralPath $script:SourcePath | Should -BeTrue
    }

    It 'deletes the captured file by native handle only after flushing its receipt' {
        $source = [Pcai.Storage.PcaiArchiveSource]::new($script:SourcePath)
        try {
            Save-VerifiedFixtureReceipt $source $script:Manifest
            (Get-Content -LiteralPath $script:Manifest -Raw | ConvertFrom-Json).source_sha256 | Should -BeExactly $source.Sha256
            $source.DeleteVerified()
            $source.HardLinkCountBeforeDeletion | Should -Be 1
        } finally { $source.Dispose() }
        Test-Path -LiteralPath $script:SourcePath | Should -BeFalse
    }

    It 'preserves both the replacement and renamed original during a path-swap race' {
        $sourceFolder = Join-Path $script:CaseRoot 'source-folder'
        New-Item -ItemType Directory -Path $sourceFolder | Out-Null
        $script:SourcePath = Join-Path $sourceFolder 'source.bin'
        [IO.File]::WriteAllText($script:SourcePath, 'original-byte-custody-fixture')
        $source = [Pcai.Storage.PcaiArchiveSource]::new($script:SourcePath)
        $renamed = Join-Path $sourceFolder 'retained-original.bin'
        try {
            Save-VerifiedFixtureReceipt $source $script:Manifest
            $source.PrepareForDeletion()
            [IO.File]::Move($script:SourcePath, $renamed)
            [IO.File]::WriteAllText($script:SourcePath, 'replacement-must-survive')
            { $source.DeleteVerified() } | Should -Throw '*identity changed*'
        } finally { $source.Dispose() }
        [IO.File]::ReadAllText($script:SourcePath) | Should -BeExactly 'replacement-must-survive'
        [IO.File]::ReadAllText($renamed) | Should -BeExactly 'original-byte-custody-fixture'
    }

    It 'preserves source when the verified manifest cannot be flushed' {
        [IO.File]::WriteAllText($script:Manifest, 'existing-record')
        $locked = [IO.File]::Open($script:Manifest, [IO.FileMode]::Open, [IO.FileAccess]::Write, [IO.FileShare]::None)
        $source = [Pcai.Storage.PcaiArchiveSource]::new($script:SourcePath)
        try {
            { Save-VerifiedFixtureReceipt $source $script:Manifest } | Should -Throw
            { $source.DeleteVerified() } | Should -Throw '*durable verified receipt*'
        } finally { $source.Dispose(); $locked.Dispose() }
        [IO.File]::ReadAllText($script:SourcePath) | Should -BeExactly 'original-byte-custody-fixture'
        [IO.File]::ReadAllText($script:Manifest) | Should -BeExactly 'existing-record'
    }

    It 'rejects wrong archive hash size or count without authorizing deletion' -ForEach @(
        @{ Md5 = '00000000000000000000000000000000'; Size = 29; Count = 1 }
        @{ Md5 = 'valid'; Size = 30; Count = 1 }
        @{ Md5 = 'valid'; Size = 29; Count = 2 }
    ) {
        $source = [Pcai.Storage.PcaiArchiveSource]::new($script:SourcePath)
        try {
            if ($Md5 -eq 'valid') { $Md5 = $source.Md5 }
            { $source.PersistVerifiedReceipt($script:Manifest, '{"event":"verified"}', $Md5, $Size, $Count) } | Should -Throw '*does not match*'
            { $source.DeleteVerified() } | Should -Throw '*durable verified receipt*'
        } finally { $source.Dispose() }
        Test-Path -LiteralPath $script:Manifest | Should -BeFalse
        Test-Path -LiteralPath $script:SourcePath | Should -BeTrue
    }

    It 'reports hardlinks and deletes only the opened alias' {
        $alias = Join-Path $script:CaseRoot 'alias.bin'
        New-Item -ItemType HardLink -Path $alias -Target $script:SourcePath | Out-Null
        $source = [Pcai.Storage.PcaiArchiveSource]::new($script:SourcePath)
        try {
            $source.HardLinkCount | Should -Be 2
            Save-VerifiedFixtureReceipt $source $script:Manifest
            $source.DeleteVerified()
            $source.HardLinkCountBeforeDeletion | Should -Be 2
        } finally { $source.Dispose() }
        Test-Path -LiteralPath $script:SourcePath | Should -BeFalse
        [IO.File]::ReadAllText($alias) | Should -BeExactly 'original-byte-custody-fixture'
    }

    It 'rejects reparse directory ancestry without opening redirected content' {
        $target = Join-Path $script:CaseRoot 'real'; New-Item -ItemType Directory -Path $target | Out-Null
        $file = Join-Path $target 'source.bin'; [IO.File]::WriteAllText($file, 'retained-junction-target')
        $junction = Join-Path $script:CaseRoot 'junction'; New-Item -ItemType Junction -Path $junction -Target $target | Out-Null
        { [Pcai.Storage.PcaiArchiveSource]::new((Join-Path $junction 'source.bin')) } | Should -Throw '*reparse*'
        [IO.File]::ReadAllText($file) | Should -BeExactly 'retained-junction-target'
    }

    It 'blocks ancestor renaming while source path custody is held' {
        $folder = Join-Path $script:CaseRoot 'folder'; New-Item -ItemType Directory -Path $folder | Out-Null
        $file = Join-Path $folder 'source.bin'; [IO.File]::WriteAllText($file, 'ancestor-custody')
        $source = [Pcai.Storage.PcaiArchiveSource]::new($file)
        try { { [IO.Directory]::Move($folder, (Join-Path $script:CaseRoot 'replacement-folder')) } | Should -Throw } finally { $source.Dispose() }
        [IO.File]::ReadAllText($file) | Should -BeExactly 'ancestor-custody'
    }

    It 'rejects reserved and ambiguous structured paths before opening them' -ForEach @(
        @{ Leaf = '$Null' }; @{ Leaf = 'NUL.txt' }; @{ Leaf = 'COM1.bin' }; @{ Leaf = 'trailing.' }
    ) {
        { [Pcai.Storage.PcaiArchiveSource]::ValidateLocalPath([string]::Concat($script:CaseRoot, '\', $Leaf)) } | Should -Throw
        @(Get-ChildItem -LiteralPath $script:CaseRoot -File).Count | Should -Be 1
    }

    It 'hashes and verifies a zero-byte file using the real native custody path' {
        [IO.File]::WriteAllBytes($script:SourcePath, [byte[]]@())
        $source = [Pcai.Storage.PcaiArchiveSource]::new($script:SourcePath)
        try {
            $source.Length | Should -Be 0
            $source.Md5 | Should -BeExactly 'd41d8cd98f00b204e9800998ecf8427e'
            $source.Sha256 | Should -BeExactly 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
            Save-VerifiedFixtureReceipt $source $script:Manifest
            $source.DeleteVerified()
        } finally { $source.Dispose() }
        Test-Path -LiteralPath $script:SourcePath | Should -BeFalse
    }

    It 'creates receipt descendants only under held native parents and appends durably' {
        $manifest = Join-Path $script:CaseRoot 'receipts/nested/receipt.jsonl'
        $parent = [Pcai.Storage.PcaiArchiveSource+ReceiptDirectoryCustody]::new($manifest)
        $writer = $null
        try {
            $writer = $parent.OpenWriter()
            $writer.CanonicalGuidPath | Should -BeExactly ($parent.CanonicalDirectoryGuidPath + '\receipt.jsonl')
            $writer.FileId128 | Should -Match '^[a-f0-9]{32}$'
            $writer.Append('{"event":"verified"}')
        } finally { if ($writer) { $writer.Dispose() }; $parent.Dispose() }
        [IO.File]::ReadAllText($manifest) | Should -BeExactly "{`"event`":`"verified`"}`n"
    }

    It 'blocks parent replacement in the pre-leaf-open race window' {
        $folder = Join-Path $script:CaseRoot 'receipts'
        $manifest = Join-Path $folder 'receipt.jsonl'
        $rival = Join-Path $script:CaseRoot 'rival'
        New-Item -ItemType Directory -Path $rival | Out-Null
        [IO.File]::WriteAllText((Join-Path $rival 'receipt.jsonl'), 'rival-must-survive')
        $parent = [Pcai.Storage.PcaiArchiveSource+ReceiptDirectoryCustody]::new($manifest)
        $writer = $null
        try {
            { [IO.Directory]::Move($folder, (Join-Path $script:CaseRoot 'retired-parent')) } | Should -Throw
            $writer = $parent.OpenWriter()
            $writer.Append('{"event":"verified"}')
        } finally { if ($writer) { $writer.Dispose() }; $parent.Dispose() }
        [IO.File]::ReadAllText((Join-Path $rival 'receipt.jsonl')) | Should -BeExactly 'rival-must-survive'
        [IO.File]::ReadAllText($manifest) | Should -BeExactly "{`"event`":`"verified`"}`n"
        Test-Path -LiteralPath (Join-Path $script:CaseRoot 'retired-parent') | Should -BeFalse
    }

    It 'blocks leaf renames and rival writers until source deletion custody ends' {
        $source = [Pcai.Storage.PcaiArchiveSource]::new($script:SourcePath)
        $renamed = Join-Path $script:CaseRoot 'retired-receipt.jsonl'
        try {
            Save-VerifiedFixtureReceipt $source $script:Manifest
            { [IO.File]::Move($script:Manifest, $renamed) } | Should -Throw
            { [IO.File]::OpenWrite($script:Manifest) } | Should -Throw
            $source.ReceiptFileId | Should -Match '^[a-f0-9]{32}$'
            $source.DeleteVerified()
            { [IO.File]::Move($script:Manifest, $renamed) } | Should -Throw
            $source.AppendReceipt($script:Manifest, '{"event":"deleted"}')
        } finally { $source.Dispose() }
        Test-Path -LiteralPath $script:SourcePath | Should -BeFalse
        Test-Path -LiteralPath $renamed | Should -BeFalse
        @(Get-Content -LiteralPath $script:Manifest).Count | Should -Be 2
    }

    It 'rejects a real leaf symlink installed after ancestor admission without writing its target' {
        $rival = Join-Path $script:CaseRoot 'rival.jsonl'
        [IO.File]::WriteAllText($rival, 'rival-must-survive')
        $parent = [Pcai.Storage.PcaiArchiveSource+ReceiptDirectoryCustody]::new($script:Manifest)
        try {
            New-Item -ItemType SymbolicLink -Path $script:Manifest -Target $rival -ErrorAction Stop | Out-Null
            { $parent.OpenWriter() } | Should -Throw '*reparse*'
        } finally { $parent.Dispose() }
        [IO.File]::ReadAllText($rival) | Should -BeExactly 'rival-must-survive'
        (Get-Item -LiteralPath $script:Manifest).LinkType | Should -BeExactly 'SymbolicLink'
        [IO.File]::ReadAllText($script:SourcePath) | Should -BeExactly 'original-byte-custody-fixture'
    }

    It 'rejects redirected receipt ancestry before any append and preserves source custody' {
        $rival = Join-Path $script:CaseRoot 'rival'; New-Item -ItemType Directory -Path $rival | Out-Null
        $rivalReceipt = Join-Path $rival 'receipt.jsonl'; [IO.File]::WriteAllText($rivalReceipt, 'rival-must-survive')
        $junction = Join-Path $script:CaseRoot 'receipts'; New-Item -ItemType Junction -Path $junction -Target $rival | Out-Null
        $source = [Pcai.Storage.PcaiArchiveSource]::new($script:SourcePath)
        try {
            { Save-VerifiedFixtureReceipt $source (Join-Path $junction 'receipt.jsonl') } | Should -Throw '*reparse*'
            { $source.DeleteVerified() } | Should -Throw '*durable verified receipt*'
        } finally { $source.Dispose() }
        [IO.File]::ReadAllText($rivalReceipt) | Should -BeExactly 'rival-must-survive'
        [IO.File]::ReadAllText($script:SourcePath) | Should -BeExactly 'original-byte-custody-fixture'
    }

    It 'rejects a hardlinked receipt instead of appending into rival content' {
        $rival = Join-Path $script:CaseRoot 'rival.jsonl'; [IO.File]::WriteAllText($rival, 'rival-must-survive')
        New-Item -ItemType HardLink -Path $script:Manifest -Target $rival | Out-Null
        { [Pcai.Storage.PcaiArchiveSource]::AppendDurableReceipt($script:Manifest, '{"event":"verified"}') } | Should -Throw '*unlinked file*'
        [IO.File]::ReadAllText($rival) | Should -BeExactly 'rival-must-survive'
        [IO.File]::ReadAllText($script:Manifest) | Should -BeExactly 'rival-must-survive'
    }

    It 'does not authorize deletion when strict receipt encoding fails before flushing' {
        $source = [Pcai.Storage.PcaiArchiveSource]::new($script:SourcePath)
        try {
            { $source.PersistVerifiedReceipt($script:Manifest, [string][char]0xD800, $source.Md5, $source.Length, 1) } | Should -Throw
            { $source.DeleteVerified() } | Should -Throw '*durable verified receipt*'
        } finally { $source.Dispose() }
        [IO.File]::ReadAllText($script:SourcePath) | Should -BeExactly 'original-byte-custody-fixture'
        (Get-Item -LiteralPath $script:Manifest).Length | Should -Be 0
    }
}

Describe 'Archive wrapper remote verification and preview guarantees' -Tag 'Unit', 'Boot' {
    BeforeEach {
        $script:CaseRoot = Join-Path $TestDrive ([guid]::NewGuid().ToString('N')); New-Item -ItemType Directory -Path $script:CaseRoot | Out-Null
        $script:SourcePath = New-ArchiveFixture
        $script:Manifest = Join-Path $script:CaseRoot 'receipts/receipt.jsonl'
        $script:RemoteBytes = @{}
        $script:Calls = [Collections.Generic.List[object]]::new()
        $script:Fault = ''
        [PcaiArchiveRcloneFixture]::Remote.Clear()
        [PcaiArchiveRcloneFixture]::Calls.Clear()
        [PcaiArchiveRcloneFixture]::Fault = ''
        [PcaiArchiveRcloneFixture]::Manifest = $script:Manifest
        [PcaiArchiveRcloneFixture]::SizeJson = ''
        $script:RemoteBytes = [PcaiArchiveRcloneFixture]::Remote
        $script:Calls = [PcaiArchiveRcloneFixture]::Calls
        Mock rclone {
            $arguments = @($args)
            [PcaiArchiveRcloneFixture]::Calls.Add($arguments)
            Set-Variable -Name LASTEXITCODE -Value 0 -Scope Global
            switch ($arguments[0]) {
                'mkdir' { return }
                'copyto' {
                    if ($arguments -notcontains '--ignore-existing') { throw 'Overwrite protection missing.' }
                    if (-not [PcaiArchiveRcloneFixture]::Remote.ContainsKey($arguments[2])) { [PcaiArchiveRcloneFixture]::Remote[$arguments[2]] = [IO.File]::ReadAllBytes($arguments[1]) }
                    return
                }
                'md5sum' {
                    if ([PcaiArchiveRcloneFixture]::Fault -eq 'exit') { Set-Variable -Name LASTEXITCODE -Value 9 -Scope Global; return }
                    $hash = [Convert]::ToHexString([Security.Cryptography.MD5]::HashData([PcaiArchiveRcloneFixture]::Remote[$arguments[1]])).ToLowerInvariant()
                    if ([PcaiArchiveRcloneFixture]::Fault -eq 'hash') { $hash = '00000000000000000000000000000000' }
                    if ([PcaiArchiveRcloneFixture]::Fault -eq 'empty') { return }
                    if ([PcaiArchiveRcloneFixture]::Fault -eq 'manifest') {
                        [PcaiArchiveRcloneFixture]::ManifestLock = [IO.File]::Open([PcaiArchiveRcloneFixture]::Manifest, [IO.FileMode]::Open, [IO.FileAccess]::Write, [IO.FileShare]::None)
                    }
                    "$hash  remote.bin"
                }
                'size' {
                    if ([PcaiArchiveRcloneFixture]::SizeJson) { return [PcaiArchiveRcloneFixture]::SizeJson }
                    @{ bytes = [PcaiArchiveRcloneFixture]::Remote[$arguments[1]].Length; count = $(if ([PcaiArchiveRcloneFixture]::Fault -eq 'count') { 2 } else { 1 }) } | ConvertTo-Json -Compress
                }
                default { throw "Unexpected external command: $($arguments -join ' ')" }
            }
        }
    }
    AfterEach {
        if ($null -ne [PcaiArchiveRcloneFixture]::ManifestLock) {
            [PcaiArchiveRcloneFixture]::ManifestLock.Dispose()
            [PcaiArchiveRcloneFixture]::ManifestLock = $null
        }
    }

    It 'performs no writes or external calls for forced dry run' {
        $record = & $script:ArchiveScript -SourcePath $script:SourcePath -Execute -DeleteAfterVerify -DryRun -ManifestPath $script:Manifest
        $record.event | Should -BeExactly 'planned'
        Test-Path -LiteralPath (Split-Path $script:Manifest) | Should -BeFalse
        $script:Calls.Count | Should -Be 0
        [IO.File]::ReadAllText($script:SourcePath) | Should -BeExactly 'original-byte-custody-fixture'
    }

    It 'performs no writes for WhatIf even when execution was requested' {
        & $script:ArchiveScript -SourcePath $script:SourcePath -Execute -DeleteAfterVerify -WhatIf -ManifestPath $script:Manifest | Out-Null
        Test-Path -LiteralPath (Split-Path $script:Manifest) | Should -BeFalse
        $script:Calls.Count | Should -Be 0
    }

    It 'lets WhatIf suppress even explicitly requested preview receipt writes' {
        & $script:ArchiveScript -SourcePath $script:SourcePath -RecordDryRun -WhatIf -ManifestPath $script:Manifest | Out-Null
        Test-Path -LiteralPath (Split-Path $script:Manifest) | Should -BeFalse
        $script:Calls.Count | Should -Be 0
    }

    It 'lets explicit DryRun suppress every preview receipt flag without compiling native code' -ForEach @(
        @{ Extra = @('-RecordDryRun', '-DryRun') }
        @{ Extra = @('-RecordDryRun', '-DryRun', '-Execute') }
        @{ Extra = @('-RecordDryRun', '--DryRun', '-Execute') }
    ) {
        $probe = Join-Path $script:CaseRoot 'dryrun-probe.ps1'
        [IO.File]::WriteAllText($probe, @'
param([string]$ArchiveScript, [string]$Source, [string]$Manifest)
& $ArchiveScript -SourcePath $Source -ManifestPath $Manifest @args | Out-Null
'CUSTODY_COMPILED=' + [bool]('Pcai.Storage.PcaiArchiveSource' -as [type])
'@
        )
        $result = Invoke-ArchiveFixtureProcess -Arguments (@('-File', $probe, '-ArchiveScript', $script:ArchiveScript, '-Source', $script:SourcePath, '-Manifest', $script:Manifest) + $Extra)
        $result.ExitCode | Should -Be 0
        $result.StdOut.Trim() | Should -BeExactly 'CUSTODY_COMPILED=False'
        Test-Path -LiteralPath (Split-Path $script:Manifest) | Should -BeFalse
        $script:Calls.Count | Should -Be 0
        [IO.File]::ReadAllText($script:SourcePath) | Should -BeExactly 'original-byte-custody-fixture'
    }

    It 'retains help compatibility without loading native code or requiring rclone' -ForEach @(
        @{ Flag = '--help' }; @{ Flag = '-h' }; @{ Flag = '-Help' }
    ) {
        $result = Invoke-ArchiveFixtureProcess -Arguments @('-File', $script:ArchiveScript, $Flag)
        $result.ExitCode | Should -Be 0
        $result.StdOut | Should -Match 'Suppress all writes including RecordDryRun'
        $result.StdOut | Should -Match 'Archive held Windows file bytes'
        Test-Path -LiteralPath (Split-Path $script:Manifest) | Should -BeFalse
    }

    It 'uploads held bytes verifies remote and deletes only after a durable record' {
        $record = & $script:ArchiveScript -SourcePath $script:SourcePath -Execute -DeleteAfterVerify -Confirm:$false -ManifestPath $script:Manifest
        $record.deleted | Should -BeTrue
        Test-Path -LiteralPath $script:SourcePath | Should -BeFalse
        $script:RemoteBytes.Count | Should -Be 1
        [Text.Encoding]::UTF8.GetString($script:RemoteBytes[$record.archive_remote]) | Should -BeExactly 'original-byte-custody-fixture'
        $events = @(Get-Content -LiteralPath $script:Manifest | ForEach-Object { ($_ | ConvertFrom-Json).event })
        $events -join ',' | Should -BeExactly 'attempted,verified,deleted'
        $record.hardlink_count_before | Should -Be 1
        $record.reclaimed_bytes_estimate | Should -BeGreaterThan 0
        $record.uploaded | Should -BeNullOrEmpty
        $record.copy_completed | Should -BeTrue
        $record.upload_outcome | Should -BeExactly 'copied-or-existing'
        $record.receipt_file_id | Should -Match '^[a-f0-9]{32}$'
    }

    It 'fails closed on external hash count empty-output and exit failures' -ForEach @(
        @{ Fault = 'hash' }; @{ Fault = 'count' }; @{ Fault = 'empty' }; @{ Fault = 'exit' }
    ) {
        [PcaiArchiveRcloneFixture]::Fault = $Fault
        { & $script:ArchiveScript -SourcePath $script:SourcePath -Execute -DeleteAfterVerify -Confirm:$false -ManifestPath $script:Manifest } | Should -Throw
        [IO.File]::ReadAllText($script:SourcePath) | Should -BeExactly 'original-byte-custody-fixture'
        $events = @(Get-Content -LiteralPath $script:Manifest | ForEach-Object { ($_ | ConvertFrom-Json).event })
        $events | Should -Not -Contain 'deleted'
        $events[-1] | Should -BeExactly 'error'
    }

    It 'rejects malformed remote metadata <SizeJson> without coercion or source deletion' -ForEach @(
        @{ SizeJson = '{"bytes":29.4,"count":1.4}'; Text = 'original-byte-custody-fixture' }
        @{ SizeJson = '{"bytes":29.4,"count":1}'; Text = 'original-byte-custody-fixture' }
        @{ SizeJson = '{"bytes":29,"count":1.4}'; Text = 'original-byte-custody-fixture' }
        @{ SizeJson = '{"bytes":true,"count":true}'; Text = 'x' }
        @{ SizeJson = '{"bytes":29,"count":true}'; Text = 'original-byte-custody-fixture' }
        @{ SizeJson = '[{"bytes":29,"count":1}]'; Text = 'original-byte-custody-fixture' }
        @{ SizeJson = '{"bytes":9223372036854775808,"count":1}'; Text = 'original-byte-custody-fixture' }
        @{ SizeJson = '{"bytes":[29],"count":1}'; Text = 'original-byte-custody-fixture' }
        @{ SizeJson = '{"bytes":29,"count":{"value":1}}'; Text = 'original-byte-custody-fixture' }
        @{ SizeJson = '{"bytes":"29","count":1}'; Text = 'original-byte-custody-fixture' }
        @{ SizeJson = '{"bytes":-0.4,"count":1}'; Text = '' }
        @{ SizeJson = '{"bytes":29,"count":-1}'; Text = 'original-byte-custody-fixture' }
    ) {
        [IO.File]::WriteAllText($script:SourcePath, $Text)
        [PcaiArchiveRcloneFixture]::SizeJson = $SizeJson
        { & $script:ArchiveScript -SourcePath $script:SourcePath -Execute -DeleteAfterVerify -Confirm:$false -ManifestPath $script:Manifest } | Should -Throw 'Remote size*'
        [IO.File]::ReadAllText($script:SourcePath) | Should -BeExactly $Text
        $script:RemoteBytes.Count | Should -Be 1
        $events = @(Get-Content -LiteralPath $script:Manifest | ForEach-Object { ($_ | ConvertFrom-Json).event })
        $events -join ',' | Should -BeExactly 'attempted,error'
        $script:Calls.Count | Should -Be 4
    }

    It 'accepts exact integral floating-point JSON values without rounding' {
        [PcaiArchiveRcloneFixture]::SizeJson = '{"bytes":29.0,"count":1.0}'
        $record = & $script:ArchiveScript -SourcePath $script:SourcePath -Execute -DeleteAfterVerify -Confirm:$false -ManifestPath $script:Manifest
        $record.deleted | Should -BeTrue
        $record.remote_size_bytes | Should -Be 29
        $record.remote_count | Should -Be 1
        Test-Path -LiteralPath $script:SourcePath | Should -BeFalse
    }

    It 'returns nonzero actual CLI exit for <SizeJson> without deleting sources' -ForEach @(
        @{ SizeJson = '{"bytes":29.4,"count":1.4}'; Text = 'original-byte-custody-fixture' }
        @{ SizeJson = '{"bytes":true,"count":true}'; Text = 'x' }
        @{ SizeJson = '[{"bytes":29,"count":1}]'; Text = 'original-byte-custody-fixture' }
        @{ SizeJson = '{"bytes":9223372036854775808,"count":1}'; Text = 'original-byte-custody-fixture' }
    ) {
        [IO.File]::WriteAllText($script:SourcePath, $Text)
        $hash = (Get-FileHash -LiteralPath $script:SourcePath -Algorithm MD5).Hash.ToLowerInvariant()
        $bin = Join-Path $script:CaseRoot 'bin'; New-Item -ItemType Directory -Path $bin | Out-Null
        $cmd = "@echo off`r`nif `"%~1`"==`"md5sum`" echo $hash  remote.bin`r`nif `"%~1`"==`"size`" echo $SizeJson`r`nexit /b 0`r`n"
        [IO.File]::WriteAllText((Join-Path $bin 'rclone.cmd'), $cmd)
        $result = Invoke-ArchiveFixtureProcess -Bin $bin -Arguments @('-File', $script:ArchiveScript, '-SourcePath', $script:SourcePath, '-Execute', '-DeleteAfterVerify', '-ManifestPath', $script:Manifest)
        $result.ExitCode | Should -Not -Be 0
        $result.StdErr | Should -Match 'Remote size'
        [IO.File]::ReadAllText($script:SourcePath) | Should -BeExactly $Text
        $events = @(Get-Content -LiteralPath $script:Manifest | ForEach-Object { ($_ | ConvertFrom-Json).event })
        $events -join ',' | Should -BeExactly 'attempted,error'
    }

    It 'uses separate content revisions without overwriting prior remote bytes' {
        $first = & $script:ArchiveScript -SourcePath $script:SourcePath -Execute -Confirm:$false -ManifestPath $script:Manifest
        [IO.File]::WriteAllText($script:SourcePath, 'changed-source-version')
        $second = & $script:ArchiveScript -SourcePath $script:SourcePath -Execute -Confirm:$false -ManifestPath $script:Manifest
        $second.archive_remote | Should -Not -BeExactly $first.archive_remote
        $script:RemoteBytes.Count | Should -Be 2
        [Text.Encoding]::UTF8.GetString($script:RemoteBytes[$first.archive_remote]) | Should -BeExactly 'original-byte-custody-fixture'
        [Text.Encoding]::UTF8.GetString($script:RemoteBytes[$second.archive_remote]) | Should -BeExactly 'changed-source-version'
        [IO.Path]::GetFileName($first.rclone_log) | Should -Match '^rclone-[a-f0-9]{16}-[a-f0-9]{64}\.log$'
    }

    It 'keeps date-named source identity in receipts while authoring a stable remote leaf' {
        $dated = New-ArchiveFixture 'capture-20260909T123152-cardiac_0.db3'
        $record = & $script:ArchiveScript -SourcePath $dated -Execute -Confirm:$false -ManifestPath $script:Manifest
        $record.source_leaf | Should -BeExactly 'capture-20260909T123152-cardiac_0.db3'
        ($record.archive_remote -split '/')[-1] | Should -Match '^payload-[a-f0-9]{16}-[a-f0-9]{64}\.db3$'
        [IO.File]::ReadAllText($dated) | Should -BeExactly 'original-byte-custody-fixture'
    }

    It 'preserves source and uploaded bytes when a rival attempts to lock the held receipt' {
        [PcaiArchiveRcloneFixture]::Fault = 'manifest'
        { & $script:ArchiveScript -SourcePath $script:SourcePath -Execute -DeleteAfterVerify -Confirm:$false -ManifestPath $script:Manifest } | Should -Throw '*being used*'
        [IO.File]::ReadAllText($script:SourcePath) | Should -BeExactly 'original-byte-custody-fixture'
        $script:RemoteBytes.Count | Should -Be 1
        $script:Calls.Count | Should -Be 3
        $events = @(Get-Content -LiteralPath $script:Manifest | ForEach-Object { ($_ | ConvertFrom-Json).event })
        $events -join ',' | Should -BeExactly 'attempted,error'
    }

    It 'rejects active writers without attempting any remote operation' {
        $writer = [IO.File]::Open($script:SourcePath, [IO.FileMode]::Open, [IO.FileAccess]::Write, [IO.FileShare]::ReadWrite)
        try {
            { & $script:ArchiveScript -SourcePath $script:SourcePath -Execute -DeleteAfterVerify -Confirm:$false -ManifestPath $script:Manifest } | Should -Throw '*custody*'
            $script:Calls.Count | Should -Be 0
        } finally { $writer.Dispose() }
        [IO.File]::ReadAllText($script:SourcePath) | Should -BeExactly 'original-byte-custody-fixture'
    }

    It 'leaves existing incorrect remote bytes untouched and preserves source' {
        $first = & $script:ArchiveScript -SourcePath $script:SourcePath -Execute -Confirm:$false -ManifestPath $script:Manifest
        $script:RemoteBytes[$first.archive_remote] = [Text.Encoding]::UTF8.GetBytes('foreign-existing-remote')
        { & $script:ArchiveScript -SourcePath $script:SourcePath -Execute -DeleteAfterVerify -Confirm:$false -ManifestPath $script:Manifest } | Should -Throw '*Remote hash*'
        [Text.Encoding]::UTF8.GetString($script:RemoteBytes[$first.archive_remote]) | Should -BeExactly 'foreign-existing-remote'
        [IO.File]::ReadAllText($script:SourcePath) | Should -BeExactly 'original-byte-custody-fixture'
    }

    It 'records a removed hardlink alias without claiming allocation reclamation' {
        $alias = Join-Path $script:CaseRoot 'alias.bin'; New-Item -ItemType HardLink -Path $alias -Target $script:SourcePath | Out-Null
        $record = & $script:ArchiveScript -SourcePath $script:SourcePath -Execute -DeleteAfterVerify -Confirm:$false -ManifestPath $script:Manifest
        $record.deleted | Should -BeTrue
        $record.hardlink_count_before_delete | Should -Be 2
        $record.reclaimed_bytes_estimate | Should -BeNullOrEmpty
        [IO.File]::ReadAllText($alias) | Should -BeExactly 'original-byte-custody-fixture'
    }

    It 'returns a nonzero actual CLI process status for an external failure' {
        $bin = Join-Path $script:CaseRoot 'bin'; New-Item -ItemType Directory -Path $bin | Out-Null
        [IO.File]::WriteAllText((Join-Path $bin 'rclone.cmd'), "@echo off`r`nexit /b 9`r`n")
        $start = [Diagnostics.ProcessStartInfo]::new((Get-Process -Id $PID).Path)
        $start.UseShellExecute = $false; $start.RedirectStandardError = $true; $start.RedirectStandardOutput = $true
        foreach ($argument in @('-NoLogo', '-NoProfile', '-File', $script:ArchiveScript, '-SourcePath', $script:SourcePath, '-Execute', '-DeleteAfterVerify', '-ManifestPath', $script:Manifest)) { $start.ArgumentList.Add($argument) }
        $start.Environment['PATH'] = "$bin;$($start.Environment['PATH'])"
        $child = [Diagnostics.Process]::Start($start)
        try {
            $stdout = $child.StandardOutput.ReadToEndAsync(); $stderr = $child.StandardError.ReadToEndAsync()
            $child.WaitForExit(20000) | Should -BeTrue
            $child.ExitCode | Should -Not -Be 0
            $stderr.GetAwaiter().GetResult() | Should -Match 'rclone mkdir failed'
            $stdout.GetAwaiter().GetResult() | Should -Not -Match 'verified|deleted'
        } finally {
            if (-not $child.HasExited) { $child.Kill($true); $child.WaitForExit() }
            $child.Dispose()
        }
        [IO.File]::ReadAllText($script:SourcePath) | Should -BeExactly 'original-byte-custody-fixture'
    }
}
