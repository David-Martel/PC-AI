#Requires -Version 5.1

# These two real producer witnesses require an actual Windows User PATH; no environment is fabricated.
$script:PathWitnessAvailable = [Environment]::OSVersion.Platform -eq [PlatformID]::Win32NT -and -not [string]::IsNullOrEmpty([Environment]::GetEnvironmentVariable('PATH', 'User'))

BeforeAll {
    $script:WitnessRoot = if ($env:PCAI_PATH_WITNESS_ROOT) { $env:PCAI_PATH_WITNESS_ROOT } else { $TestDrive }
    $script:CompressionSource = if ($env:PCAI_PATH_CONTRACT_SOURCE) {
        $env:PCAI_PATH_CONTRACT_SOURCE
    } else { Join-Path $PSScriptRoot '../../Modules/PC-AI.Cleanup/Public/Optimize-PathCompression.ps1' }
    $script:CleanupHelperSource = Join-Path $PSScriptRoot '../../Modules/PC-AI.Cleanup/Private/Cleanup-Helpers.ps1'
    # Avoid the module loader's live log-directory initialization. Invoke the exact maintained producer/helpers.
    . $script:CleanupHelperSource
    . $script:CompressionSource
    $tokens = $null; $errors = $null
    $ast = [Management.Automation.Language.Parser]::ParseFile($script:CompressionSource, [ref]$tokens, [ref]$errors)
    $substitute = $ast.FindAll({
        param($node)
        $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Invoke-Substitute'
    }, $true)
    if (@($substitute).Count -ne 1 -or $errors.Count) { throw 'Unable to bind the maintained substitution helper.' }
    . ([scriptblock]::Create($substitute[0].Extent.Text))

    function Get-PathContractEnvironment {
        $snapshot = foreach ($scope in @('Process', 'User', 'Machine')) {
            $values = [Environment]::GetEnvironmentVariables($scope)
            $text = @($values.Keys | Sort-Object | ForEach-Object { "$_=$($values[$_])" }) -join "`n"
            $sha = [Security.Cryptography.SHA256]::Create()
            try { $hash = [BitConverter]::ToString($sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($text))).Replace('-', '') }
            finally { $sha.Dispose() }
            [pscustomobject]@{ Scope = $scope; Count = $values.Count; Hash = $hash }
        }
        @($snapshot)
    }
    function Get-PathContractNamespace {
        param([string]$Root)
        $ownedPrefix = [IO.Path]::GetFullPath($Root).TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
        @(Get-ChildItem -LiteralPath $Root -File -Recurse | ForEach-Object {
            if (-not $_.FullName.StartsWith($ownedPrefix, [StringComparison]::OrdinalIgnoreCase)) { throw 'Namespace entry escaped its owned root.' }
            [pscustomobject]@{ RelativePath = $_.FullName.Substring($ownedPrefix.Length); Bytes = $_.Length; Hash = (Get-FileHash -LiteralPath $_.FullName).Hash }
        })
    }
}

Describe 'PATH compression dry-run custody' -Tag 'Unit', 'Cleanup', 'Portable' {
    BeforeEach {
        $script:LogPath = $TestDrive
        $script:PathWitness = Join-Path $TestDrive 'path-backup.txt'
        $script:EnvironmentBefore = Get-PathContractEnvironment
        Mock Get-Item {
            $key = [pscustomobject]@{}
            $key | Add-Member -MemberType ScriptMethod -Name GetValueKind -Value { param($Name) [Microsoft.Win32.RegistryValueKind]::String }
            $key
        } -ParameterFilter { $Path -like 'Registry::*' }
        Mock Set-ItemProperty { throw 'Real registry writes are forbidden.' }
        Mock Add-Type { throw 'Real WM_SETTINGCHANGE interop is forbidden.' }
        Mock Write-Host { }
    }
    AfterEach {
        $after = Get-PathContractEnvironment
        ($after | ConvertTo-Json -Compress) | Should -BeExactly ($script:EnvironmentBefore | ConvertTo-Json -Compress)
        Should -Invoke Set-ItemProperty -Exactly -Times 0
        Should -Invoke Add-Type -Exactly -Times 0
    }
    It 'does not create a backup or log file during Force WhatIf' -Skip:(-not $script:PathWitnessAvailable) {
        # Only read the actual User PATH. Every write-capable helper is directed to this owned namespace.
        [string]::IsNullOrEmpty([Environment]::GetEnvironmentVariable('PATH', 'User')) | Should -BeFalse
        $before = Get-PathContractNamespace -Root $TestDrive
        $result = Optimize-PathCompression -Target User -Force -WhatIf -BackupPath $script:PathWitness
        $after = Get-PathContractNamespace -Root $TestDrive
        [ordered]@{ Before = $before; After = $after; EnvironmentBefore = $script:EnvironmentBefore; EnvironmentAfter = (Get-PathContractEnvironment); Success = $result.Success } |
            ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $script:WitnessRoot 'whatif-observation.json')
        $result.Success | Should -BeTrue
        ($after | ConvertTo-Json -Compress) | Should -BeExactly ($before | ConvertTo-Json -Compress)
    }
    It 'does not overwrite an existing backup during Force WhatIf' -Skip:(-not $script:PathWitnessAvailable) {
        [IO.File]::WriteAllText($script:PathWitness, 'Retained synthetic predecessor bytes', [Text.UTF8Encoding]::new($false))
        $before = Get-PathContractNamespace -Root $TestDrive
        $result = Optimize-PathCompression -Target User -Force -WhatIf -BackupPath $script:PathWitness
        $after = Get-PathContractNamespace -Root $TestDrive
        [ordered]@{ Before = $before; After = $after; EnvironmentBefore = $script:EnvironmentBefore; EnvironmentAfter = (Get-PathContractEnvironment); Success = $result.Success } |
            ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $script:WitnessRoot 'whatif-existing-observation.json')
        ($after | ConvertTo-Json -Compress) | Should -BeExactly ($before | ConvertTo-Json -Compress)
    }
}

Describe 'PATH substitution respects literal path components' -Tag 'Unit', 'Cleanup', 'Portable' {
    It 'substitutes an exact literal directory' {
        Invoke-Substitute -Entry 'C:\Program Files' -Substitutions @([pscustomobject]@{ Literal = 'C:\Program Files'; Token = '%ProgramFiles%' }) |
            Should -Be '%ProgramFiles%'
    }
    It 'substitutes a child directory preserving the separator and spelling' {
        Invoke-Substitute -Entry 'c:\PROGRAM FILES\Git\cmd' -Substitutions @([pscustomobject]@{ Literal = 'C:\Program Files'; Token = '%ProgramFiles%' }) |
            Should -Be '%ProgramFiles%\Git\cmd'
    }
    It 'does not substitute a sibling that merely shares a text prefix' {
        Invoke-Substitute -Entry 'C:\Program FilesElse\bin' -Substitutions @([pscustomobject]@{ Literal = 'C:\Program Files'; Token = '%ProgramFiles%' }) |
            Should -Be 'C:\Program FilesElse\bin'
    }
    It 'does not treat literal square brackets as wildcard syntax' {
        Invoke-Substitute -Entry 'C:\Users\fixture[one]\bin' -Substitutions @([pscustomobject]@{ Literal = 'C:\Users\fixture[one]'; Token = '%USERPROFILE%' }) |
            Should -Be '%USERPROFILE%\bin'
    }
    It 'does not substitute a wildcard-looking literal for another directory' {
        Invoke-Substitute -Entry 'C:\Users\fixtureo\bin' -Substitutions @([pscustomobject]@{ Literal = 'C:\Users\fixture[one]'; Token = '%USERPROFILE%' }) |
            Should -Be 'C:\Users\fixtureo\bin'
    }
    It 'preserves an unrelated directory' {
        Invoke-Substitute -Entry 'D:\Tools\bin' -Substitutions @([pscustomobject]@{ Literal = 'C:\Program Files'; Token = '%ProgramFiles%' }) |
            Should -Be 'D:\Tools\bin'
    }
}
