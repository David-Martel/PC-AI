#Requires -Version 7.0

BeforeAll {
    $script:RepoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
    $script:Probe = Join-Path $TestDrive 'copied-runtime-consumer.ps1'
    @'
param($Installed, $Root, $Mode, $SourceRoot)
$ErrorActionPreference = 'Stop'
$env:PCAI_CACHE_PROVIDER = 'memory'
$env:PCAI_ROOT = if ($Mode -eq 'Source') { $null } else { $Root }
if ($Mode -eq 'Source') {
    . (Join-Path $SourceRoot 'Modules/PC-AI.Common/Public/Get-PcaiRuntimeConfig.ps1')
    if ((Resolve-PcaiRepoRoot -StartPath (Join-Path $SourceRoot 'Modules/PC-AI.LLM')) -ne $SourceRoot) { throw 'No-override source discovery changed.' }
    'source ancestry preserved'
    exit 0
}
$module = Import-Module (Join-Path $Installed 'PC-AI.LLM/PC-AI.LLM.psd1') -Force -PassThru -ErrorAction Stop
$configuration = & $module { $script:ModuleConfig.Clone() }
[ordered]@{ Root=$configuration.ProjectRoot; Config=$configuration.ConfigPath; Model=$configuration.DefaultModel; Endpoint=$configuration.OllamaApiUrl; Tools=$configuration.ToolsPath } | ConvertTo-Json -Compress
if ($Mode -eq 'Invalid') { throw 'Invalid explicit root was accepted.' }
if ($configuration.ProjectRoot -ne $Root -or $configuration.ConfigPath -ne (Join-Path $Root 'Config/llm-config.json') -or
    $configuration.DefaultModel -ne 'copied-consumer-unique-model' -or $configuration.OllamaApiUrl -ne 'http://copied-consumer.invalid:19431' -or
    $configuration.ToolsPath -ne (Join-Path $Root 'Config/pcai-tools.json')) { throw 'Copied LLM consumer did not use its selected per-machine configuration.' }
if ($Mode -eq 'Changed') {
    $env:PCAI_ROOT = Join-Path $Root 'absent-after-valid-discovery'
    try { & $module { Resolve-PcaiRepoRoot -StartPath $PSScriptRoot }; throw 'Cached discovery hid an invalid explicit root.' }
    catch { if ($_.Exception.Message -notlike '*Explicit PCAI_ROOT*') { throw } }
    'changed explicit root rejected before cached discovery'
}
'copied runtime model endpoint and paths selected'
'@ | Set-Content -LiteralPath $script:Probe
}

Describe 'Copied modules retain per-machine runtime configuration' {
    BeforeEach {
        $script:Installed = Join-Path $TestDrive ([guid]::NewGuid().ToString('N'))
        [void](New-Item -ItemType Directory -Path $script:Installed)
        foreach ($name in @('PC-AI.Common', 'PC-AI.LLM')) {
            Copy-Item -LiteralPath (Join-Path $script:RepoRoot "Modules/$name") -Destination $script:Installed -Recurse
        }
        $script:SelectedRoot = Join-Path $TestDrive ([guid]::NewGuid().ToString('N'))
        [void](New-Item -ItemType Directory -Path (Join-Path $script:SelectedRoot 'Config'))
        '# Selected PC-AI configuration root marker' | Set-Content -LiteralPath (Join-Path $script:SelectedRoot 'PC-AI.ps1')
        '{"ollama":{"model":"copied-consumer-unique-model","base_url":"http://copied-consumer.invalid:19431"},"fallbackOrder":["ollama"]}' |
            Set-Content -LiteralPath (Join-Path $script:SelectedRoot 'Config/llm-config.json') -Encoding utf8NoBOM
        '{}' | Set-Content -LiteralPath (Join-Path $script:SelectedRoot 'Config/pcai-tools.json') -Encoding utf8NoBOM
    }

    It 'uses the selected config root in a fresh copied LLM consumer without a profile' {
        $output = & (Get-Command pwsh).Source -NoLogo -NoProfile -File $script:Probe $script:Installed $script:SelectedRoot Valid $script:RepoRoot 2>&1
        if ($LASTEXITCODE) { $output | ForEach-Object { Write-Host $_ } }
        $LASTEXITCODE | Should -Be 0
        $output | Where-Object { $_ -is [string] -and $_.StartsWith('{') } | ForEach-Object { Write-Host $_ }
        $output | Should -Contain 'copied runtime model endpoint and paths selected'
    }

    It 'does not let an earlier successful discovery hide an invalid changed override' {
        $output = & (Get-Command pwsh).Source -NoLogo -NoProfile -File $script:Probe $script:Installed $script:SelectedRoot Changed $script:RepoRoot 2>&1
        if ($LASTEXITCODE) { $output | ForEach-Object { Write-Host $_ } }
        $LASTEXITCODE | Should -Be 0
        $output | Should -Contain 'changed explicit root rejected before cached discovery'
    }

    It 'rejects an explicit invalid config root: <Kind>' -ForEach @(
        @{ Kind='missing directory' }, @{ Kind='missing marker' }, @{ Kind='missing configuration' }, @{ Kind='non-filesystem provider' }, @{ Kind='whitespace' }
    ) {
        $invalid = switch ($Kind) {
            'missing directory' { Join-Path $TestDrive 'absent-root' }
            'missing marker' { Remove-Item -LiteralPath (Join-Path $script:SelectedRoot 'PC-AI.ps1'); $script:SelectedRoot }
            'missing configuration' { Remove-Item -LiteralPath (Join-Path $script:SelectedRoot 'Config/llm-config.json'); $script:SelectedRoot }
            'non-filesystem provider' { 'Env:PCAI_ROOT' }
            'whitespace' { ' ' }
        }
        $output = & (Get-Command pwsh).Source -NoLogo -NoProfile -File $script:Probe $script:Installed $invalid Invalid $script:RepoRoot 2>&1
        $LASTEXITCODE | Should -Not -Be 0
        ($output -join "`n") | Should -Match 'Explicit PCAI_ROOT'
        ($output -join "`n") | Should -Not -Match 'Invalid explicit root was accepted'
    }

    It 'preserves source ancestry when no explicit root is configured' {
        $output = & (Get-Command pwsh).Source -NoLogo -NoProfile -File $script:Probe $script:Installed $script:SelectedRoot Source $script:RepoRoot 2>&1
        if ($LASTEXITCODE) { $output | ForEach-Object { Write-Host $_ } }
        $LASTEXITCODE | Should -Be 0
        $output | Should -Contain 'source ancestry preserved'
    }
}
