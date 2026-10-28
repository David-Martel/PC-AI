BeforeAll {
    $script:Relocator = Join-Path $PSScriptRoot '../../Tools/SystemScripts/Storage/Move-PcaiIdleBuildCache.ps1'
}

Describe 'Idle build-cache relocation' -Skip:(-not $IsWindows) {
    BeforeEach {
        $script:Workspace = Join-Path $TestDrive ([guid]::NewGuid().ToString('N'))
        $script:Source = Join-Path $script:Workspace '.pcai/cache'
        $script:Destination = Join-Path $TestDrive ([guid]::NewGuid().ToString('N'))
        [void](New-Item -ItemType Directory -Path (Join-Path $script:Source 'nested'))
        'preserve original' | Set-Content -LiteralPath (Join-Path $script:Source 'one.txt')
        [IO.File]::WriteAllBytes((Join-Path $script:Source 'nested/two.bin'), [byte[]](0, 255, 0, 17))
    }

    It 'does not create custody or move any bytes with <Mode>' -ForEach @(
        @{ Mode = 'DryRun' }, @{ Mode = 'WhatIf' }
    ) {
        $before = @(Get-FileHash -LiteralPath (Join-Path $script:Source 'one.txt'), (Join-Path $script:Source 'nested/two.bin'))
        $options = @{ WorkspaceRoot=$script:Workspace; SourceDirectory=$script:Source; DestinationDirectory=$script:Destination }
        $options[$Mode] = $true
        $result = & $script:Relocator @options
        $result.State | Should -Be 'Planned'
        $result.FileCount | Should -Be 2
        Test-Path -LiteralPath $script:Destination | Should -BeFalse
        Test-Path -LiteralPath ($script:Destination + '.pending') | Should -BeFalse
        Test-Path -LiteralPath ($script:Destination + '.relocation.json') | Should -BeFalse
        @(Get-FileHash -LiteralPath $before.Path).Hash | Should -Be $before.Hash
    }

    It 'retains each original SHA in the published manifest and removes only copied source bytes' {
        $hash = (Get-FileHash -LiteralPath (Join-Path $script:Source 'one.txt')).Hash
        $result = & $script:Relocator -WorkspaceRoot $script:Workspace -SourceDirectory $script:Source -DestinationDirectory $script:Destination
        $result.State | Should -Be 'Relocated'
        Test-Path -LiteralPath $script:Source | Should -BeFalse
        (Get-FileHash -LiteralPath (Join-Path $script:Destination 'one.txt')).Hash | Should -Be $hash
        [IO.File]::ReadAllBytes((Join-Path $script:Destination 'nested/two.bin')) | Should -Be ([byte[]](0, 255, 0, 17))
        $receipt = Get-Content -LiteralPath $result.Receipt -Raw | ConvertFrom-Json
        $receipt.State | Should -Be 'Relocated'
        $receipt.Files.Count | Should -Be 2
        ($receipt.Files | Where-Object RelativePath -eq 'one.txt').SHA256 | Should -Be $hash
    }

    It 'preserves prior destination custody instead of merging or overwriting it' {
        [void](New-Item -ItemType Directory -Path $script:Destination)
        'unrelated destination' | Set-Content -LiteralPath (Join-Path $script:Destination 'keep.txt')
        { & $script:Relocator -WorkspaceRoot $script:Workspace -SourceDirectory $script:Source -DestinationDirectory $script:Destination } |
            Should -Throw '*Existing destination custody*'
        Get-Content -LiteralPath (Join-Path $script:Destination 'keep.txt') | Should -Be 'unrelated destination'
        Test-Path -LiteralPath (Join-Path $script:Source 'one.txt') | Should -BeTrue
    }

    It 'refuses a Git store selected at <Relative>' -ForEach @(
        @{ Relative = '.git' }, @{ Relative = '.git/objects' }
    ) {
        $gitSource = Join-Path $script:Workspace $Relative
        [void](New-Item -ItemType Directory -Path $gitSource -Force)
        'git custody' | Set-Content -LiteralPath (Join-Path $gitSource 'keep')
        { & $script:Relocator -WorkspaceRoot $script:Workspace -SourceDirectory $gitSource -DestinationDirectory $script:Destination } |
            Should -Throw '*Git store*'
        Get-Content -LiteralPath (Join-Path $gitSource 'keep') | Should -Be 'git custody'
        Test-Path -LiteralPath $script:Destination | Should -BeFalse
    }

    It 'refuses a real child referencing the cache with forward-slash argv' {
        $child = [Diagnostics.Process]::new()
        $child.StartInfo.FileName = (Get-Process -Id $PID).Path
        $child.StartInfo.UseShellExecute = $false
        foreach ($arg in @('-NoLogo', '-NoProfile', '-Command', 'Start-Sleep -Seconds 60; # ' + $script:Source.Replace('\', '/'))) {
            [void]$child.StartInfo.ArgumentList.Add($arg)
        }
        try {
            [void]$child.Start()
            { & $script:Relocator -WorkspaceRoot $script:Workspace -SourceDirectory $script:Source -DestinationDirectory $script:Destination -DryRun } |
                Should -Throw '*active process*'
            Test-Path -LiteralPath $script:Destination | Should -BeFalse
            Test-Path -LiteralPath (Join-Path $script:Source 'one.txt') | Should -BeTrue
        } finally {
            if (-not $child.HasExited) { $child.Kill(); $child.WaitForExit() }
            $child.Dispose()
        }
    }

    It 'preserves a receipt that appears during copying and retains all originals' {
        $raceReceipt = $script:Destination + '.relocation.json'
        Mock Copy-Item {
            param($LiteralPath, $Destination)
            [IO.File]::Copy($LiteralPath, $Destination)
            'foreign receipt' | Set-Content -LiteralPath $raceReceipt
        }
        { & $script:Relocator -WorkspaceRoot $script:Workspace -SourceDirectory $script:Source -DestinationDirectory $script:Destination } |
            Should -Throw
        Get-Content -LiteralPath $raceReceipt | Should -Be 'foreign receipt'
        Test-Path -LiteralPath (Join-Path $script:Source 'one.txt') | Should -BeTrue
        Test-Path -LiteralPath (Join-Path $script:Destination 'nested/two.bin') | Should -BeTrue
    }

    It 'refuses linked payloads and preserves their outside targets' {
        $outside = Join-Path $TestDrive ([guid]::NewGuid().ToString('N'))
        [void](New-Item -ItemType Directory -Path $outside)
        'outside content' | Set-Content -LiteralPath (Join-Path $outside 'keep.txt')
        [void](New-Item -ItemType Junction -Path (Join-Path $script:Source 'linked') -Target $outside)
        { & $script:Relocator -WorkspaceRoot $script:Workspace -SourceDirectory $script:Source -DestinationDirectory $script:Destination } |
            Should -Throw '*Linked relocation path*'
        Get-Content -LiteralPath (Join-Path $outside 'keep.txt') | Should -Be 'outside content'
        Test-Path -LiteralPath $script:Destination | Should -BeFalse
    }

    It 'copies regular hard-linked files while preserving an outside alias unchanged' {
        $outside = Join-Path $TestDrive ([guid]::NewGuid().ToString('N'))
        'outside hardlink bytes' | Set-Content -LiteralPath $outside
        $originalHash = (Get-FileHash -LiteralPath $outside).Hash
        [void](New-Item -ItemType HardLink -Path (Join-Path $script:Source 'aliased.bin') -Target $outside)
        $result = & $script:Relocator -WorkspaceRoot $script:Workspace -SourceDirectory $script:Source -DestinationDirectory $script:Destination
        $result.State | Should -Be 'Relocated'
        (Get-FileHash -LiteralPath $outside).Hash | Should -Be $originalHash
        (Get-FileHash -LiteralPath (Join-Path $script:Destination 'aliased.bin')).Hash | Should -Be $originalHash
        $receipt = Get-Content -LiteralPath $result.Receipt -Raw | ConvertFrom-Json
        ($receipt.Files | Where-Object RelativePath -eq 'aliased.bin').SourceLinkType | Should -Be 'HardLink'
        Test-Path -LiteralPath $script:Source | Should -BeFalse
    }

    It 'retains complete destination custody when an owned open source handle prevents retirement' {
        $file = Join-Path $script:Source 'one.txt'
        $stream = [IO.File]::Open($file, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::Read)
        try {
            { & $script:Relocator -WorkspaceRoot $script:Workspace -SourceDirectory $script:Source -DestinationDirectory $script:Destination } |
                Should -Throw
            Test-Path -LiteralPath $file | Should -BeTrue
            Get-Content -LiteralPath (Join-Path $script:Destination 'one.txt') | Should -Be 'preserve original'
            $receipt = Get-Content -LiteralPath ($script:Destination + '.relocation.json') -Raw | ConvertFrom-Json
            $receipt.State | Should -Be 'VerifiedCopy'
        } finally { $stream.Dispose() }
    }

    It 'shows help without touching any paths with <Mode>' -ForEach @(
        @{ Mode = '-h' }, @{ Mode = '--help' }
    ) {
        $output = if ($Mode -eq '-h') { & $script:Relocator -h } else { & $script:Relocator --help }
        $output | Should -Match '^Usage:'
        Test-Path -LiteralPath $script:Destination | Should -BeFalse
    }
}
