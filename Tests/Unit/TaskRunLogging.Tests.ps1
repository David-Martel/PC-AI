#Requires -Version 7.0
Describe 'Bounded task diagnostics retain safe source-only logging' -Tag 'Unit','Boot' {
    BeforeAll {
        $root=Join-Path $PSScriptRoot '../..'
        foreach($entry in @(
            @{Path='Tools/SystemScripts/Machine/Unlock-BwVault.ps1';Name='Write-BwUnlockLog'},
            @{Path='Tools/SystemScripts/Networking/Mount-VigilFleetShare.ps1';Name='Write-VigilFleetShareLog'}
        )) {
            $tokens=$null;$errors=$null
            $ast=[Management.Automation.Language.Parser]::ParseFile((Join-Path $root $entry.Path),[ref]$tokens,[ref]$errors)
            if($errors.Count){throw 'Task source did not parse.'}
            $name=$entry.Name
            $definition=$ast.Find({param($node)$node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq $name},$false)
            if(-not $definition){throw "Missing production helper $name"}
            . ([scriptblock]::Create($definition.Extent.Text))
        }
    }
    BeforeEach {
        $script:PriorLocalAppData=$env:LOCALAPPDATA
        $env:LOCALAPPDATA=Join-Path $TestDrive ([guid]::NewGuid().ToString('N'))
    }
    AfterEach { $env:LOCALAPPDATA=$script:PriorLocalAppData }
    It 'writes and rolls a finite <Name> task log in the owned profile' -ForEach @(
        @{Name='Write-BwUnlockLog';Leaf='bw-auto-unlock.log'},
        @{Name='Write-VigilFleetShareLog';Leaf='vigil-public-asus-nfs.log'}
    ) {
        & $Name -Message 'owned fixed diagnostic'
        $log=Join-Path $env:LOCALAPPDATA "dtm-tasks/logs/$Leaf"
        [IO.File]::ReadAllText($log) | Should -Match 'owned fixed diagnostic'
        [IO.File]::WriteAllText($log,('x'*262145))
        & $Name -Message 'owned successor diagnostic'
        (Get-Item -LiteralPath "$log.1").Length | Should -Be 262145
        (Get-Item -LiteralPath $log).Length | Should -BeLessThan 1024
        [IO.File]::ReadAllText($log) | Should -Match 'owned successor diagnostic'
    }
    It 'does not turn a <Name> log path failure into task failure' -ForEach @(
        @{Name='Write-BwUnlockLog'},@{Name='Write-VigilFleetShareLog'}
    ) {
        [IO.Directory]::CreateDirectory($env:LOCALAPPDATA) | Out-Null
        [IO.File]::WriteAllText((Join-Path $env:LOCALAPPDATA 'dtm-tasks'),'owned blocking file')
        { & $Name -Message 'owned fixed diagnostic' -ErrorAction Stop } | Should -Not -Throw
    }
}
