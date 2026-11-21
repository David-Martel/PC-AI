# Controls for Install-PcaiProfileShim.ps1, the repair for the tooling-health "shim=DRIFT" finding
# (Config/PowerShell/Profile/README.md). The shim sits in two Documents roots and the validator
# flags either one that differs from the Git source. These tests touch only TestDrive.

BeforeAll {
    $repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
    . (Join-Path $repoRoot 'Tools/SystemScripts/Machine/Install-PcaiProfileShim.ps1')
    function New-ShimFixture {
        param([string]$Name)
        $root = Join-Path $TestDrive $Name
        $null = New-Item -ItemType Directory -Path $root
        $source = Join-Path $root 'source.ps1'
        [IO.File]::WriteAllText($source, "# shim v2`r`n`$script:ShimVersion = 2`r`n")
        [pscustomobject]@{ Root = $root; Source = $source; Backup = (Join-Path $root 'backups') }
    }
}

Describe 'Install-PcaiProfileShim' -Tag 'Unit', 'Portable' {
    It 'replaces a stale copy, keeps its bytes in a hash-named backup and writes a timestamped receipt' {
        $f = New-ShimFixture 'replace'
        $target = Join-Path $f.Root 'Documents\PowerShell\Microsoft.PowerShell_profile.ps1'
        $null = New-Item -ItemType Directory -Path (Split-Path -Parent $target)
        [IO.File]::WriteAllText($target, "# shim v1 (pre-fast-path)`r`n")
        $oldHash = (Get-FileHash -LiteralPath $target).Hash.ToLowerInvariant()
        $r = @(Install-PcaiProfileShim -Source $f.Source -Path $target -Backup $f.Backup)
        $r[0].Action | Should -Be 'Replaced'
        (Get-FileHash -LiteralPath $target).Hash | Should -Be (Get-FileHash -LiteralPath $f.Source).Hash
        $r[0].Backup | Should -BeLike "*.bak-$($oldHash.Substring(0, 12))"
        [IO.File]::ReadAllText($r[0].Backup) | Should -Be "# shim v1 (pre-fast-path)`r`n"
        $receipt = Get-Content -LiteralPath (Join-Path $f.Backup 'receipts.jsonl') | ConvertFrom-Json
        $receipt.oldSha256 | Should -Be $oldHash
        $receipt.newSha256 | Should -Be (Get-FileHash -LiteralPath $f.Source).Hash.ToLowerInvariant()
        [DateTime]::Parse($receipt.utc).Kind | Should -Not -BeNullOrEmpty
        # The backup name carries a hash, not a date (authored names stay free of datetime strings).
        (Split-Path -Leaf $r[0].Backup) | Should -Match '^[A-Za-z0-9.-]+\.Documents\.PowerShell\.Microsoft\.PowerShell_profile\.ps1\.bak-[0-9a-f]{12}$'
        Get-ChildItem -LiteralPath (Split-Path -Parent $target) -Force | Where-Object Name -like '.pcai-shim-*' | Should -BeNullOrEmpty
    }
    It 'leaves a copy that differs only in line endings alone, and is idempotent' {
        $f = New-ShimFixture 'same'
        $target = Join-Path $f.Root 'Documents\PowerShell\Microsoft.PowerShell_profile.ps1'
        $null = New-Item -ItemType Directory -Path (Split-Path -Parent $target)
        [IO.File]::WriteAllText($target, "# shim v2`n`$script:ShimVersion = 2`n")
        $before = (Get-FileHash -LiteralPath $target).Hash
        (Install-PcaiProfileShim -Source $f.Source -Path $target -Backup $f.Backup).Action | Should -Be 'Unchanged'
        (Get-FileHash -LiteralPath $target).Hash | Should -Be $before
        Test-Path -LiteralPath $f.Backup | Should -BeFalse
        $stale = Join-Path $f.Root 'stale.ps1'
        [IO.File]::WriteAllText($stale, '# old')
        $null = Install-PcaiProfileShim -Source $f.Source -Path $stale -Backup $f.Backup
        (Install-PcaiProfileShim -Source $f.Source -Path $stale -Backup $f.Backup).Action | Should -Be 'Unchanged'
        @(Get-Content -LiteralPath (Join-Path $f.Backup 'receipts.jsonl')).Count | Should -Be 1
    }
    It 'creates a missing file when its PowerShell folder is missing but Documents exists' {
        $f = New-ShimFixture 'create'
        $null = New-Item -ItemType Directory -Path (Join-Path $f.Root 'Documents')
        $target = Join-Path $f.Root 'Documents\PowerShell\Microsoft.PowerShell_profile.ps1'
        $r = Install-PcaiProfileShim -Source $f.Source -Path $target -Backup $f.Backup
        $r.Action | Should -Be 'Created'
        $r.Backup | Should -BeNullOrEmpty
        Test-Path -LiteralPath $target | Should -BeTrue
    }
    It 'skips a root that does not exist on this host instead of creating OneDrive folders' {
        $f = New-ShimFixture 'skip'
        $target = Join-Path $f.Root 'OneDrive\Documents\PowerShell\Microsoft.PowerShell_profile.ps1'
        (Install-PcaiProfileShim -Source $f.Source -Path $target -Backup $f.Backup).Action | Should -Be 'Skipped'
        Test-Path -LiteralPath (Join-Path $f.Root 'OneDrive') | Should -BeFalse
    }
    It 'changes nothing with -WhatIf' {
        $f = New-ShimFixture 'whatif'
        $target = Join-Path $f.Root 'Documents\PowerShell\Microsoft.PowerShell_profile.ps1'
        $null = New-Item -ItemType Directory -Path (Split-Path -Parent $target)
        [IO.File]::WriteAllText($target, '# old')
        (Install-PcaiProfileShim -Source $f.Source -Path $target -Backup $f.Backup -WhatIf).Action | Should -Be 'WouldChange'
        [IO.File]::ReadAllText($target) | Should -Be '# old'
        Test-Path -LiteralPath $f.Backup | Should -BeFalse
    }
    It 'refuses a source that does not parse, before touching any target' {
        $f = New-ShimFixture 'badsource'
        [IO.File]::WriteAllText($f.Source, 'if ($true) { ')
        $target = Join-Path $f.Root 'Documents\PowerShell\Microsoft.PowerShell_profile.ps1'
        $null = New-Item -ItemType Directory -Path (Split-Path -Parent $target)
        [IO.File]::WriteAllText($target, '# keep me')
        { Install-PcaiProfileShim -Source $f.Source -Path $target -Backup $f.Backup } | Should -Throw '*does not parse*'
        [IO.File]::ReadAllText($target) | Should -Be '# keep me'
    }
    It 'refuses a missing source' {
        $f = New-ShimFixture 'nosource'
        { Install-PcaiProfileShim -Source (Join-Path $f.Root 'nope.ps1') -Path (Join-Path $f.Root 'x.ps1') -Backup $f.Backup } | Should -Throw '*not found*'
    }
    It 'defaults to the Documents and OneDrive Documents shim locations' {
        $t = Get-PcaiShimDefaultTarget
        $t | Should -Contain (Join-Path $HOME 'Documents\PowerShell\Microsoft.PowerShell_profile.ps1')
        $t | Should -Contain (Join-Path $HOME 'OneDrive\Documents\PowerShell\Microsoft.PowerShell_profile.ps1')
    }
    It 'deploys a shim the validator then reports as matching' {
        $f = New-ShimFixture 'validator'
        $source = Join-Path $repoRoot 'Config/PowerShell/Profile/Microsoft.PowerShell_profile.ps1'
        $target = Join-Path $f.Root 'Documents\PowerShell\Microsoft.PowerShell_profile.ps1'
        $null = New-Item -ItemType Directory -Path (Split-Path -Parent $target)
        [IO.File]::WriteAllText($target, '# pre-333 copy')
        $null = Install-PcaiProfileShim -Source $source -Path $target -Backup $f.Backup
        (Get-PcaiShimNormalizedText -Path $target) | Should -Be (Get-PcaiShimNormalizedText -Path $source)
    }
}

Describe 'Install-PcaiProfileShim script entry point' -Tag 'Unit', 'Portable' {
    It 'prints the comment-based help for <_>' -ForEach @('-h', '--help') {
        $path = Join-Path $repoRoot 'Tools/SystemScripts/Machine/Install-PcaiProfileShim.ps1'
        $out = & (Join-Path $PSHOME 'pwsh') -NoProfile -NonInteractive -File $path $_ 2>&1 | Out-String
        $LASTEXITCODE | Should -Be 0
        $out | Should -Match 'SYNOPSIS'
    }
}
