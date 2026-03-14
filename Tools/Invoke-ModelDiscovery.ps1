#Requires -Version 7.0
<#
.SYNOPSIS
    Discover local LLM model files and generate MODELS.md.

.DESCRIPTION
    Uses PC-AI search tooling (Find-FilesFast) to scan common locations
    and/or fixed drives for model files (GGUF, SafeTensors, etc).
    Produces a markdown inventory suitable for LLM agents.

.PARAMETER Roots
    Optional list of root paths to scan. Defaults to common user/model locations.

.PARAMETER OutputPath
    Path to write MODELS.md (default: repo root).

.PARAMETER DeepScan
    When set, scans all fixed/removable drives (excluding system folders).

.PARAMETER MaxPerRoot
    Optional limit per root path (0 = unlimited).

.EXAMPLE
    .\Tools\Invoke-ModelDiscovery.ps1

.EXAMPLE
    .\Tools\Invoke-ModelDiscovery.ps1 -DeepScan -OutputPath .\MODELS.md
#>
[CmdletBinding()]
param(
    [string[]]$Roots,
    [string]$OutputPath = (Join-Path (Resolve-Path .) 'MODELS.md'),
    [switch]$DeepScan,
    [int]$MaxPerRoot = 0,
    [switch]$ContentScan,
    [switch]$IncludeOllamaBlobs
)

$repoRoot = Resolve-Path .
$userHome = $env:USERPROFILE
if (-not $PSBoundParameters.ContainsKey('ContentScan')) {
    $ContentScan = $true
}
if (-not $PSBoundParameters.ContainsKey('IncludeOllamaBlobs')) {
    $IncludeOllamaBlobs = $true
}

$nativeAvailable = $false
if (Get-Command -Name 'Test-PcaiNativeAvailable' -ErrorAction SilentlyContinue) {
    try {
        $nativeAvailable = Test-PcaiNativeAvailable
    }
    catch {
        $nativeAvailable = $false
    }
}

# Import common bootstrap and requested acceleration modules for search/content scan.
$commonModulePath = Join-Path $repoRoot 'Modules\PC-AI.Common\PC-AI.Common.psm1'
if (Test-Path -LiteralPath $commonModulePath) {
    Import-Module $commonModulePath -Force -ErrorAction SilentlyContinue | Out-Null
}

if (Get-Command Import-PcaiAccelerationStack -ErrorAction SilentlyContinue) {
    $requestedModules = @('PC-AI.Acceleration')
    if ($ContentScan) {
        $requestedModules += 'ProfileAccelerator'
    }
    $null = Import-PcaiAccelerationStack -Modules $requestedModules -RepoRoot $repoRoot
}

if (-not (Get-Command Find-FilesFast -ErrorAction SilentlyContinue)) {
    throw "Find-FilesFast not available. Ensure the acceleration stack is installed and importable."
}

$extensions = @('gguf', 'ggml', 'ggjt', 'bin', 'safetensors', 'pt', 'pth', 'onnx', 'ckpt')

$excludePatterns = @(
    'Windows',
    'Program Files',
    'Program Files (x86)',
    'ProgramData',
    '$Recycle.Bin',
    'System Volume Information',
    '.git',
    '.pcai',
    'node_modules'
)

if (-not $Roots -or $Roots.Count -eq 0) {
    $Roots = @()
    if ($userHome) {
        $Roots += $userHome
        $Roots += (Join-Path $userHome '.cache')
        $Roots += (Join-Path $userHome '.local')
        $Roots += (Join-Path $userHome '.ollama')
        $Roots += (Join-Path $userHome '.ollama\models')
        $Roots += (Join-Path $userHome '.lmstudio')
        $Roots += (Join-Path $userHome '.lmstudio\models')
        $Roots += (Join-Path $userHome '.cache\huggingface\hub')
        $Roots += (Join-Path $userHome 'Downloads')
        $Roots += (Join-Path $userHome 'Documents')
        $Roots += (Join-Path $userHome 'Models')
    }
    if ($env:LOCALAPPDATA) {
        $Roots += (Join-Path $env:LOCALAPPDATA 'LMStudio')
        $Roots += (Join-Path $env:LOCALAPPDATA 'LMStudio\models')
        $Roots += (Join-Path $env:LOCALAPPDATA 'lmstudio')
        $Roots += (Join-Path $env:LOCALAPPDATA 'lmstudio\models')
        $Roots += (Join-Path $env:LOCALAPPDATA 'huggingface\hub')
        $Roots += (Join-Path $env:LOCALAPPDATA 'Packages')
        $Roots += (Join-Path $env:LOCALAPPDATA 'Docker')
    }
    if ($env:APPDATA) {
        $Roots += (Join-Path $env:APPDATA 'LMStudio')
        $Roots += (Join-Path $env:APPDATA 'lmstudio')
    }

    $Roots += @(
        'C:\Models', 'C:\AI', 'C:\LLM', 'C:\HuggingFace',
        'C:\codedev', 'C:\code', 'C:\dev', 'C:\projects',
        'D:\Models', 'D:\AI', 'D:\LLM', 'D:\HuggingFace',
        'D:\codedev', 'D:\code', 'D:\dev', 'D:\projects',
        'E:\Models', 'E:\AI', 'E:\LLM', 'E:\HuggingFace',
        'E:\codedev', 'E:\code', 'E:\dev', 'E:\projects',
        'T:\projects', 'T:\models'
    )

    if ($DeepScan) {
        $fixedDrives = Get-CimInstance Win32_LogicalDisk |
            Where-Object { $_.DriveType -in 2, 3 } |
            Select-Object -ExpandProperty DeviceID
        foreach ($drive in $fixedDrives) {
            $Roots += ($drive + '\')
        }
        $Roots += @(
            'C:\ProgramData\Docker',
            'C:\ProgramData\DockerDesktop'
        )
    }
}

$repoModelDir = Join-Path $repoRoot 'Models'
if (Test-Path $repoModelDir) {
    $Roots += $repoModelDir
}
$repoAtModelDir = Join-Path $repoRoot '@Models'
if (Test-Path $repoAtModelDir) {
    $Roots += $repoAtModelDir
}

$Roots = $Roots | Where-Object { $_ -and (Test-Path $_) } | Select-Object -Unique

Write-Host "Model discovery roots:" -ForegroundColor Cyan
$Roots | ForEach-Object { Write-Host "  - $_" }

$found = @()
foreach ($root in $Roots) {
    if ($nativeAvailable) {
        foreach ($ext in $extensions) {
            $nativeResult = Invoke-PcaiNativeFileSearch -Pattern "*.$ext" -Path $root -MaxResults 0
            if ($nativeResult -and $nativeResult.Files) {
                foreach ($item in $nativeResult.Files) {
                    if ($item.Path -and (Test-Path $item.Path)) {
                        $found += Get-Item $item.Path -ErrorAction SilentlyContinue
                    }
                }
            }
        }
    }

    $files = Find-FilesFast -Path $root -Extension $extensions -Type file -Hidden -Exclude $excludePatterns -FullPath -NoIgnore -PreferNative:$false
    if ($MaxPerRoot -gt 0) {
        $files = $files | Select-Object -First $MaxPerRoot
    }
    foreach ($file in $files) {
        $found += $file
    }
}

# De-duplicate by full path
$found = $found | Sort-Object -Property FullName -Unique

function Get-BackendHint {
    param([string]$Ext)
    switch ($Ext.ToLowerInvariant()) {
        'gguf' { 'llama.cpp, mistralrs' }
        'ggml' { 'llama.cpp (legacy)' }
        'ggjt' { 'llama.cpp (legacy)' }
        'safetensors' { 'mistralrs (HF), candle' }
        'pt' { 'mistralrs (HF), candle (convert to safetensors recommended)' }
        'pth' { 'mistralrs (HF), candle (convert to safetensors recommended)' }
        'onnx' { 'onnxruntime (not wired), candidate for future backend' }
        'bin' { 'HF/PyTorch (convert to safetensors or GGUF)' }
        'ckpt' { 'PyTorch (convert to safetensors or GGUF)' }
        Default { 'unknown' }
    }
}

$rows = $found | ForEach-Object {
    $ext = $_.Extension.TrimStart('.').ToLowerInvariant()
    [PSCustomObject]@{
        Name = $_.Name
        Extension = $ext
        SizeMB = [Math]::Round($_.Length / 1MB, 2)
        LastWriteTime = $_.LastWriteTime
        Path = $_.FullName
        Backend = (Get-BackendHint -Ext $ext)
    }
}

$rows = @($rows)

$ollamaEntries = @()
$ollamaBlobRows = @()
if ($userHome) {
    $ollamaRoot = Join-Path $userHome '.ollama\models'
    $ollamaManifestRoot = Join-Path $ollamaRoot 'manifests'
    $ollamaBlobRoot = Join-Path $ollamaRoot 'blobs'
    if (Test-Path $ollamaManifestRoot) {
        try {
            $manifestFiles = Get-ChildItem -Path $ollamaManifestRoot -Recurse -File -ErrorAction SilentlyContinue
            Write-Verbose "Ollama manifest root: $ollamaManifestRoot (files: $($manifestFiles.Count))"
            foreach ($file in $manifestFiles) {
                try {
                    $json = Get-Content $file.FullName -Raw -ErrorAction Stop | ConvertFrom-Json
                    if ($json -and $json.layers) {
                        if (-not $script:OllamaSampleLogged) {
                            $sample = $json.layers | Select-Object -First 1 | ConvertTo-Json -Compress
                            Write-Verbose "Ollama manifest sample layer: $sample"
                            $script:OllamaSampleLogged = $true
                        }
                        $modelLayer = $json.layers | Where-Object { $_.mediaType -match 'ollama\.image\.model' } | Select-Object -First 1
                        if ($modelLayer -and $modelLayer.digest) {
                            $digest = $modelLayer.digest
                            $digestId = $digest -replace '^sha256:', ''
                            $blobPath = if ($digestId) { Join-Path $ollamaBlobRoot ("sha256-" + $digestId) } else { $null }
                            $modelName = $file.FullName.Replace($ollamaManifestRoot, '').TrimStart('\')
                            $ollamaEntries += [PSCustomObject]@{
                                Model = $modelName
                                Digest = $digest
                                Path = $file.FullName
                                BlobPath = $blobPath
                            }
                            if ($IncludeOllamaBlobs -and $blobPath -and (Test-Path $blobPath)) {
                                $isGguf = $false
                                try {
                                    $stream = [System.IO.File]::OpenRead($blobPath)
                                    try {
                                        $buffer = New-Object byte[] 4
                                        $read = $stream.Read($buffer, 0, 4)
                                        if ($read -eq 4) {
                                            $magicStr = [System.Text.Encoding]::ASCII.GetString($buffer)
                                            if ($magicStr -eq 'GGUF') {
                                                $isGguf = $true
                                            }
                                        }
                                    }
                                    finally {
                                        $stream.Dispose()
                                    }
                                }
                                catch { }
                                $item = Get-Item $blobPath -ErrorAction SilentlyContinue
                                if ($item) {
                                    $ollamaBlobRows += [PSCustomObject]@{
                                        Name = $modelName
                                        Extension = if ($isGguf) { 'gguf (ollama)' } else { 'ollama-blob' }
                                        SizeMB = [Math]::Round($item.Length / 1MB, 2)
                                        LastWriteTime = $item.LastWriteTime
                                        Path = $item.FullName
                                        Backend = if ($isGguf) { 'llama.cpp, mistralrs' } else { 'ollama blob (inspect)' }
                                    }
                                }
                            }
                        }
                    }
                }
                catch {
                    Write-Verbose "Ollama manifest parse failed: $($file.FullName) -> $($_.Exception.Message)"
                }
            }
        }
        catch {
            Write-Verbose "Ollama manifest scan failed: $($_.Exception.Message)"
        }
    }
}

if ($ollamaBlobRows.Count -gt 0) {
    if (-not $rows -or $rows.Count -eq 0) {
        $rows = @($ollamaBlobRows)
    }
    else {
        $rows += $ollamaBlobRows
    }
}

Write-Verbose "Ollama manifests found: $($ollamaEntries.Count); blob rows: $($ollamaBlobRows.Count)"

$summaryByExt = $rows | Group-Object Extension | Sort-Object Count -Descending
$summaryByBackend = $rows | Group-Object Backend | Sort-Object Count -Descending

$contentLines = @()
$contentLines += "# MODELS"
$contentLines += ""
$contentLines += "Generated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
$contentLines += ""
$contentLines += "## Summary"
$contentLines += ""
if (-not $rows -or $rows.Count -eq 0) {
    $contentLines += "No model files found with extensions: $($extensions -join ', ')"
    $contentLines += ""
    $contentLines += "Searched roots:"
    $contentLines += ($Roots | ForEach-Object { "- $_" })
}
else {
$contentLines += "| Extension | Count | Total Size (MB) |"
$contentLines += "| --- | --- | ---: |"
foreach ($group in $summaryByExt) {
    $totalSize = ($rows | Where-Object { $_.Extension -eq $group.Name } | Measure-Object -Property SizeMB -Sum).Sum
    $contentLines += "| $($group.Name) | $($group.Count) | $([Math]::Round($totalSize, 2)) |"
}
$contentLines += ""
$contentLines += "## Backend Mapping"
$contentLines += ""
$contentLines += "| Backend | Count |"
$contentLines += "| --- | ---: |"
foreach ($group in $summaryByBackend) {
    $contentLines += "| $($group.Name) | $($group.Count) |"
}
$contentLines += ""
$contentLines += "## Discovered Models"
$contentLines += ""
$contentLines += "| Name | Ext | Size (MB) | Modified | Backend | Path |"
$contentLines += "| --- | --- | ---: | --- | --- | --- |"
foreach ($row in ($rows | Sort-Object SizeMB -Descending)) {
    $contentLines += "| $($row.Name) | $($row.Extension) | $($row.SizeMB) | $($row.LastWriteTime.ToString('yyyy-MM-dd HH:mm')) | $($row.Backend) | $($row.Path) |"
}
}

$contentLines += ""
$contentLines += "## Notes"
$contentLines += ""
$contentLines += "- GGUF works with llama.cpp and mistralrs backends."
$contentLines += "- SafeTensors is the preferred HuggingFace format for mistralrs/candle."
$contentLines += "- Legacy GGML/GGJT can be converted to GGUF for best compatibility."
$contentLines += "- PyTorch `.bin`, `.pt`, `.pth`, `.ckpt` should be converted to SafeTensors or GGUF before use."

if ($ollamaEntries.Count -gt 0) {
    $contentLines += ""
    $contentLines += "## Ollama Manifests"
    $contentLines += ""
    $contentLines += "| Model | Digest | Manifest Path |"
    $contentLines += "| --- | --- | --- |"
    foreach ($entry in ($ollamaEntries | Sort-Object Model)) {
        $contentLines += "| $($entry.Model) | $($entry.Digest) | $($entry.Path) |"
    }
}

if ($ContentScan) {
    $contentLines += ""
    $contentLines += "## Model-Related References"
    $contentLines += ""
    $contentLines += "Searching for model references in text/code files..."

    $contentPatterns = @(
        'gguf',
        'safetensors',
        'llama\\.cpp',
        'mistralrs',
        'model_path',
        'pytorch_model\\.bin',
        'onnx',
        'huggingface',
        'hf\\.co',
        'ollama',
        'lmstudio'
    )

    $contentMatches = @()
    $textPatterns = @('*.md', '*.txt', '*.json', '*.yaml', '*.yml', '*.toml', '*.ps1', '*.cs', '*.rs')
    $contentRoots = $Roots | Select-Object -First 10

    foreach ($root in $contentRoots) {
        foreach ($patternItem in $contentPatterns) {
            try {
                $results = Search-ContentFast -Path $root -Pattern $patternItem -FilePattern $textPatterns -MaxResults 200 -Context 0 -NoIgnore
                $contentMatches += $results
            }
            catch {
                # ignore content errors per root
            }
        }
    }

    $contentFiles = $contentMatches | Where-Object { $_.Path } |
        Select-Object -ExpandProperty Path -Unique |
        Sort-Object

    if ($contentFiles.Count -eq 0) {
        $contentLines += "- No model-related references found in scanned text/code files."
    }
    else {
        $contentLines += ""
        $contentLines += "| File |"
        $contentLines += "| --- |"
        foreach ($path in $contentFiles) {
            $contentLines += "| $path |"
        }
    }
}

$content = $contentLines -join "`n"
Set-Content -Path $OutputPath -Value $content -Encoding UTF8

Write-Host "Wrote model inventory to $OutputPath"
