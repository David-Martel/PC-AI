#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }

BeforeAll {
    $script:ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    $script:ScriptPath = Join-Path $script:ProjectRoot 'Tools\Sync-NvidiaDriverVersion.ps1'
    $script:TempDir = Join-Path $env:TEMP "pcai_sync_nvidia_driver_tests_$(New-Guid)"

    New-Item -ItemType Directory -Path $script:TempDir -Force | Out-Null

    $script:TempRegistryPath = Join-Path $script:TempDir 'driver-registry.json'
    @'
{
  "version": "1.0.0-test",
  "lastUpdated": "2026-03-19T00:00:00Z",
  "devices": [
    {
      "id": "nvidia-rtx-2000-ada",
      "name": "NVIDIA RTX 2000 Ada Generation Laptop GPU",
      "category": "gpu",
      "matchRules": [
        { "type": "friendly_name", "pattern": "*RTX 2000 Ada*" }
      ],
      "driver": {
        "latestVersion": "591.55",
        "downloadUrl": "https://www.nvidia.com/download/index.aspx"
      },
      "sharedDriverGroup": "nvidia-gpu-driver"
    },
    {
      "id": "nvidia-rtx-5060-ti",
      "name": "NVIDIA GeForce RTX 5060 Ti",
      "category": "gpu",
      "matchRules": [
        { "type": "friendly_name", "pattern": "*RTX 5060 Ti*" }
      ],
      "driver": {
        "latestVersion": "591.55",
        "downloadUrl": "https://www.nvidia.com/download/index.aspx"
      },
      "sharedDriverGroup": "nvidia-gpu-driver"
    }
  ]
}
'@ | Set-Content -Path $script:TempRegistryPath -Encoding UTF8

    $script:FakeSmiPath = Join-Path $script:TempDir 'nvidia-smi.cmd'
    @'
@echo off
echo 0, NVIDIA RTX 2000 Ada Generation Laptop GPU, 582.41
echo 1, NVIDIA GeForce RTX 5060 Ti, 582.41
'@ | Set-Content -Path $script:FakeSmiPath -Encoding ASCII
}

AfterAll {
    if (Test-Path $script:TempDir) {
        Remove-Item -Path $script:TempDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}

Describe 'Sync-NvidiaDriverVersion utility' -Tag 'Unit', 'Drivers', 'Gpu', 'Fast', 'Portable' {
    It 'resolves the shared NVIDIA driver version for both GPUs and renders a formatted status table' {
        $rawRows = @(
            & $script:ScriptPath `
                -RegistryPath $script:TempRegistryPath `
                -NvidiaSmiPath $script:FakeSmiPath
        )
        $rows = @(
            foreach ($item in $rawRows) {
                if ($item -is [System.Array]) {
                    foreach ($nested in $item) {
                        $nested
                    }
                }
                else {
                    $item
                }
            }
        )
        $text = (
            & $script:ScriptPath `
                -RegistryPath $script:TempRegistryPath `
                -NvidiaSmiPath $script:FakeSmiPath 6>&1 |
                Out-String
        )

        $rows.Count | Should -Be 2
        ($rows | Select-Object -ExpandProperty RegistryLatest -Unique) | Should -Be @('591.55')
        ($rows | Select-Object -ExpandProperty RegistryId) | Should -Contain 'nvidia-rtx-2000-ada'
        ($rows | Select-Object -ExpandProperty RegistryId) | Should -Contain 'nvidia-rtx-5060-ti'

        $text | Should -Match 'Index\s+GPU Name\s+Installed\s+Latest\s+Status\s+Download URL'
        $text | Should -Not -Match '\{0,-'
        $text | Should -Match 'sharedDriverGroup: nvidia-gpu-driver'
        $text | Should -Match 'Latest registry version: 591\.55'
    }
}
