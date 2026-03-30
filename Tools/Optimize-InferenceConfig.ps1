#Requires -Version 5.1
<#
.SYNOPSIS
    Analyzes GPU state and recommends optimal inference configuration settings.

.DESCRIPTION
    Reads the current GPU VRAM state via Test-PcaiGpuReadiness and compares
    the active Config/llm-config.json against recommended settings for the
    available hardware.  Outputs a report showing current versus recommended
    values, with specific guidance for model selection, GPU offload, and
    context window sizing.

    By default the script is advisory-only and does not modify any files.
    Pass -Apply to write the recommended changes to the config file.

    Recommendation tiers (based on best GPU free VRAM):
      - >12 GB free: 7B models, full GPU offload, up to 8192 context
      -  6-12 GB free: 3B-4B models, full GPU offload, up to 4096 context
      -  <6 GB free: 3B models, partial GPU offload, 2048 context

.PARAMETER ConfigPath
    Path to the LLM configuration file.  Defaults to Config/llm-config.json
    relative to the repository root.

.PARAMETER Apply
    When specified, writes recommended changes to the configuration file.
    Without this switch the script only prints the recommendations report.

.PARAMETER OutputFormat
    Output format for the recommendations report.
    - Table (default): human-readable aligned table.
    - Json: machine-readable JSON object.

.EXAMPLE
    .\Tools\Optimize-InferenceConfig.ps1
    Prints a table comparing current config to recommended settings.

.EXAMPLE
    .\Tools\Optimize-InferenceConfig.ps1 -OutputFormat Json
    Prints the recommendations report as JSON for programmatic consumption.

.EXAMPLE
    .\Tools\Optimize-InferenceConfig.ps1 -Apply
    Writes recommended changes to Config/llm-config.json.

.EXAMPLE
    .\Tools\Optimize-InferenceConfig.ps1 -ConfigPath D:\custom\llm-config.json -Apply
    Reads and updates a config file at a custom path.

.NOTES
    Requires the PC-AI.Gpu module (or pcai_core_lib.dll / pcai-perf.exe) for
    GPU state queries.  When no GPU backend is available, falls back to
    conservative CPU-only recommendations.
#>
[CmdletBinding(SupportsShouldProcess)]
param(
    [Parameter()]
    [string]$ConfigPath = '',

    [Parameter()]
    [switch]$Apply,

    [Parameter()]
    [ValidateSet('Table', 'Json')]
    [string]$OutputFormat = 'Table'
)

$ErrorActionPreference = 'Stop'

#region Resolve paths and load modules
$scriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
$repoRoot  = Split-Path -Parent $scriptDir

if (-not $ConfigPath) {
    $ConfigPath = Join-Path $repoRoot 'Config\llm-config.json'
}

if (-not (Test-Path $ConfigPath)) {
    Write-Error "Configuration file not found: $ConfigPath"
    return
}

# Import GPU module if available
$gpuModulePath = Join-Path $repoRoot 'Modules\PC-AI.Gpu\PC-AI.Gpu.psd1'
$gpuModuleLoaded = $false
if (Test-Path $gpuModulePath) {
    try {
        Import-Module $gpuModulePath -Force -ErrorAction Stop
        $gpuModuleLoaded = $true
    }
    catch {
        Write-Warning "Could not load PC-AI.Gpu module: $($_.Exception.Message)"
    }
}
#endregion

#region Read current config
$configRaw  = Get-Content -Path $ConfigPath -Raw -Encoding UTF8
# Strip BOM if present
if ($configRaw.StartsWith([char]0xFEFF)) {
    $configRaw = $configRaw.Substring(1)
}
$config = $configRaw | ConvertFrom-Json

# Extract the ollama section (primary optimization target)
$ollamaConfig = $config.ollama
if (-not $ollamaConfig) {
    Write-Warning "No 'ollama' section found in $ConfigPath. Creating recommendations based on defaults."
    $ollamaConfig = [PSCustomObject]@{}
}

# Extract the backend section (pcai-inference)
$backendConfig = $config.backend
if (-not $backendConfig) {
    $backendConfig = [PSCustomObject]@{}
}
#endregion

#region Query GPU state
$gpuInfo = $null
$bestGpu = $null
$bestGpuFreeMB = 0
$gpuCount = 0

if ($gpuModuleLoaded) {
    try {
        $preflight = Test-PcaiGpuReadiness
        $gpuCount  = ($preflight.Gpus | Measure-Object).Count

        if ($gpuCount -gt 0) {
            $gpuInfo = $preflight.Gpus

            # Find the GPU with the most free VRAM
            $bestFree = 0
            foreach ($gpu in $gpuInfo) {
                $totalMB = [int]($gpu.total_mb)
                $usedMB  = [int]($gpu.used_mb)
                $freeMB  = $totalMB - $usedMB
                if ($freeMB -gt $bestFree) {
                    $bestFree  = $freeMB
                    $bestGpu   = $gpu
                }
            }
            $bestGpuFreeMB = $bestFree

            Write-Verbose "GPU query successful: $gpuCount GPU(s), best has ${bestGpuFreeMB}MB free VRAM."
        }
        else {
            Write-Verbose 'No GPUs detected by preflight check.'
        }
    }
    catch {
        Write-Warning "GPU query failed: $($_.Exception.Message). Using conservative defaults."
    }
}
else {
    # Try nvidia-smi directly as a fallback
    try {
        $smiPath = $null
        foreach ($candidate in @('nvidia-smi.exe', 'C:\Windows\System32\nvidia-smi.exe')) {
            if (Get-Command -Name $candidate -ErrorAction SilentlyContinue) {
                $smiPath = $candidate
                break
            }
        }

        if ($smiPath) {
            $smiOutput = & $smiPath --query-gpu=index,name,memory.total,memory.used --format=csv,noheader,nounits 2>&1
            if ($LASTEXITCODE -eq 0) {
                $headerFields = @('Index', 'Name', 'MemoryTotal', 'MemoryUsed')
                $csvData = @($smiOutput | Where-Object { "$_".Trim() -ne '' }) | ConvertFrom-Csv -Header $headerFields
                $gpuCount = $csvData.Count

                $bestFree = 0
                foreach ($row in $csvData) {
                    $totalMB = [int]$row.MemoryTotal
                    $usedMB  = [int]$row.MemoryUsed
                    $freeMB  = $totalMB - $usedMB
                    if ($freeMB -gt $bestFree) {
                        $bestFree = $freeMB
                        $bestGpu  = [PSCustomObject]@{
                            name     = $row.Name.Trim()
                            total_mb = $totalMB
                            used_mb  = $usedMB
                            index    = [int]$row.Index
                        }
                    }
                }
                $bestGpuFreeMB = $bestFree
                Write-Verbose "nvidia-smi fallback: $gpuCount GPU(s), best has ${bestGpuFreeMB}MB free."
            }
        }
    }
    catch {
        Write-Verbose "nvidia-smi fallback failed: $($_.Exception.Message)"
    }
}
#endregion

#region Build recommendations
$recommendations = [ordered]@{}

# Determine VRAM tier
$vramTier = 'cpu-only'
if ($bestGpuFreeMB -gt 12288) {
    $vramTier = 'high'       # >12GB free
}
elseif ($bestGpuFreeMB -ge 6144) {
    $vramTier = 'medium'     # 6-12GB free
}
elseif ($bestGpuFreeMB -gt 0) {
    $vramTier = 'low'        # <6GB free
}

$tierDescriptions = @{
    'high'     = ">12GB free VRAM: 7B models, full GPU offload, large context"
    'medium'   = "6-12GB free VRAM: 3B-4B models, full GPU offload, moderate context"
    'low'      = "<6GB free VRAM: 3B models, partial GPU offload, conservative context"
    'cpu-only' = "No GPU detected: CPU-only inference, smallest models"
}

# Model recommendations by tier
$modelRecommendations = @{
    'high'     = @{ model = 'qwen2.5-coder:7b';  quantization = 'Q4_K_M'; size_gb = '~4.5' }
    'medium'   = @{ model = 'qwen2.5-coder:3b';  quantization = 'Q4_K_M'; size_gb = '~2.0' }
    'low'      = @{ model = 'qwen2.5-coder:3b';  quantization = 'Q4_K_M'; size_gb = '~2.0' }
    'cpu-only' = @{ model = 'qwen2.5-coder:0.5b'; quantization = 'Q8_0';  size_gb = '~0.5' }
}

# Context window recommendations by tier
# Rule: min(model_default, vram_mb / 2) — conservative to leave room for KV cache
$ctxRecommendations = @{
    'high'     = 8192
    'medium'   = 4096
    'low'      = 2048
    'cpu-only' = 2048
}

# GPU layers recommendations
$gpuLayerRecommendations = @{
    'high'     = -1    # all layers on GPU
    'medium'   = -1    # all layers on GPU
    'low'      = 20    # partial offload
    'cpu-only' = 0     # no GPU
}

# num_predict recommendations
$predictRecommendations = @{
    'high'     = 1024
    'medium'   = 512
    'low'      = 512
    'cpu-only' = 256
}

# Build the comparison table
$currentModel     = if ($ollamaConfig.model)       { $ollamaConfig.model }       else { '(not set)' }
$currentNumGpu    = if ($null -ne $ollamaConfig.num_gpu)  { $ollamaConfig.num_gpu }  else { '(not set)' }
$currentNumCtx    = if ($ollamaConfig.num_ctx)     { $ollamaConfig.num_ctx }     else { '(not set)' }
$currentNumPredict = if ($ollamaConfig.num_predict) { $ollamaConfig.num_predict } else { '(not set)' }
$currentNumThread = if ($ollamaConfig.num_thread)  { $ollamaConfig.num_thread }  else { '(not set)' }
$currentTemp      = if ($ollamaConfig.temperature) { $ollamaConfig.temperature } else { '(not set)' }
$currentRepeatN   = if ($ollamaConfig.repeat_last_n) { $ollamaConfig.repeat_last_n } else { '(not set)' }
$currentRepeatP   = if ($ollamaConfig.repeat_penalty) { $ollamaConfig.repeat_penalty } else { '(not set)' }

$recommendedModel   = $modelRecommendations[$vramTier].model
$recommendedCtx     = $ctxRecommendations[$vramTier]
$recommendedGpu     = $gpuLayerRecommendations[$vramTier]
$recommendedPredict = $predictRecommendations[$vramTier]

# Compute context recommendation using min(model_default, vram_mb / 2)
if ($bestGpuFreeMB -gt 0) {
    $vramBasedCtx = [Math]::Floor($bestGpuFreeMB / 2)
    # Round down to nearest 1024
    $vramBasedCtx = [Math]::Floor($vramBasedCtx / 1024) * 1024
    $vramBasedCtx = [Math]::Max($vramBasedCtx, 2048)
    $recommendedCtx = [Math]::Min($recommendedCtx, $vramBasedCtx)
}

# Processor count for thread recommendation
$cpuCores = 0
try {
    $cpuCores = (Get-CimInstance -ClassName Win32_Processor -ErrorAction SilentlyContinue |
        Measure-Object -Property NumberOfCores -Sum).Sum
}
catch {
    try {
        $cpuCores = [Environment]::ProcessorCount
    }
    catch {
        $cpuCores = 8
    }
}
$recommendedThreads = [Math]::Min($cpuCores, 8)  # Cap at 8; beyond that diminishing returns

$comparisonItems = @(
    [PSCustomObject]@{ Setting = 'model';          Current = $currentModel;     Recommended = $recommendedModel;   Match = ($currentModel -eq $recommendedModel) }
    [PSCustomObject]@{ Setting = 'num_gpu';         Current = $currentNumGpu;    Recommended = $recommendedGpu;     Match = ($currentNumGpu -eq $recommendedGpu) }
    [PSCustomObject]@{ Setting = 'num_thread';      Current = $currentNumThread; Recommended = $recommendedThreads; Match = ($currentNumThread -eq $recommendedThreads) }
    [PSCustomObject]@{ Setting = 'num_ctx';          Current = $currentNumCtx;    Recommended = $recommendedCtx;     Match = ($currentNumCtx -eq $recommendedCtx) }
    [PSCustomObject]@{ Setting = 'num_predict';      Current = $currentNumPredict; Recommended = $recommendedPredict; Match = ($currentNumPredict -eq $recommendedPredict) }
    [PSCustomObject]@{ Setting = 'repeat_last_n';    Current = $currentRepeatN;   Recommended = 64;                  Match = ($currentRepeatN -eq 64) }
    [PSCustomObject]@{ Setting = 'repeat_penalty';   Current = $currentRepeatP;   Recommended = 1.1;                 Match = ($currentRepeatP -eq 1.1) }
    [PSCustomObject]@{ Setting = 'temperature';      Current = $currentTemp;      Recommended = 0.15;                Match = ($currentTemp -eq 0.15) }
)

$matchCount    = ($comparisonItems | Where-Object { $_.Match }).Count
$mismatchCount = $comparisonItems.Count - $matchCount
#endregion

#region Output report
$report = [ordered]@{
    timestamp       = (Get-Date -Format 'o')
    config_path     = $ConfigPath
    gpu_count       = $gpuCount
    best_gpu        = if ($bestGpu) {
        [ordered]@{
            name     = [string]$bestGpu.name
            total_mb = [int]$bestGpu.total_mb
            used_mb  = [int]$bestGpu.used_mb
            free_mb  = $bestGpuFreeMB
            index    = if ($null -ne $bestGpu.index) { [int]$bestGpu.index } else { $null }
        }
    } else { $null }
    vram_tier       = $vramTier
    tier_description = $tierDescriptions[$vramTier]
    cpu_cores       = $cpuCores
    model_recommendation = $modelRecommendations[$vramTier]
    settings        = $comparisonItems
    matches         = $matchCount
    mismatches      = $mismatchCount
    env_recommendations = @(
        'OLLAMA_NUM_PARALLEL=1 (single-user, reduces scheduling overhead)'
        'OLLAMA_SCHED_SPREAD=true (distribute layers across multiple GPUs)'
        'OLLAMA_MAX_LOADED_MODELS=1 (free VRAM from idle models)'
        'OLLAMA_FLASH_ATTENTION=1 (halve KV cache VRAM on SM>=80 GPUs)'
    )
}

if ($OutputFormat -eq 'Json') {
    $report | ConvertTo-Json -Depth 5
}
else {
    # Table output
    Write-Host ''
    Write-Host '=== PC-AI Inference Configuration Optimizer ===' -ForegroundColor Cyan
    Write-Host ''

    # GPU summary
    if ($gpuCount -gt 0 -and $bestGpu) {
        Write-Host "  GPUs detected:   $gpuCount" -ForegroundColor Green
        Write-Host "  Best GPU:        $([string]$bestGpu.name)" -ForegroundColor Green
        Write-Host "  VRAM (total):    $([int]$bestGpu.total_mb) MB" -ForegroundColor Green
        Write-Host "  VRAM (free):     $bestGpuFreeMB MB" -ForegroundColor Green
    }
    else {
        Write-Host '  GPUs detected:   0 (no NVIDIA GPU found)' -ForegroundColor Yellow
    }

    Write-Host "  CPU cores:       $cpuCores"
    Write-Host "  VRAM tier:       $vramTier - $($tierDescriptions[$vramTier])" -ForegroundColor White
    Write-Host ''

    # Model recommendation
    $modelRec = $modelRecommendations[$vramTier]
    Write-Host "  Recommended model: $($modelRec.model) ($($modelRec.quantization), ~$($modelRec.size_gb) GB)" -ForegroundColor White
    Write-Host ''

    # Settings comparison table
    Write-Host '  Setting             Current          Recommended      Status' -ForegroundColor White
    Write-Host '  -------             -------          -----------      ------' -ForegroundColor DarkGray

    foreach ($item in $comparisonItems) {
        $status = if ($item.Match) { 'OK' } else { 'CHANGE' }
        $color  = if ($item.Match) { 'Green' } else { 'Yellow' }
        $line   = '  {0,-20}{1,-17}{2,-17}{3}' -f $item.Setting, $item.Current, $item.Recommended, $status
        Write-Host $line -ForegroundColor $color
    }

    Write-Host ''
    Write-Host "  Score: $matchCount/$($comparisonItems.Count) settings match recommendations" -ForegroundColor $(if ($mismatchCount -eq 0) { 'Green' } else { 'Yellow' })
    Write-Host ''

    # Environment variable recommendations
    Write-Host '  Recommended Ollama environment variables:' -ForegroundColor White
    foreach ($envRec in $report.env_recommendations) {
        Write-Host "    - $envRec" -ForegroundColor DarkGray
    }
    Write-Host ''

    if ($mismatchCount -gt 0 -and -not $Apply) {
        Write-Host "  Run with -Apply to write recommended changes to $ConfigPath" -ForegroundColor Cyan
    }
}
#endregion

#region Apply changes
if ($Apply -and $mismatchCount -gt 0) {
    if ($PSCmdlet.ShouldProcess($ConfigPath, 'Write optimized inference configuration')) {
        # Update the ollama section with recommended values
        $config.ollama.num_gpu        = $recommendedGpu
        $config.ollama.num_thread     = $recommendedThreads
        $config.ollama.num_ctx        = $recommendedCtx
        $config.ollama.num_predict    = $recommendedPredict
        $config.ollama.repeat_last_n  = 64
        $config.ollama.repeat_penalty = 1.1
        $config.ollama.temperature    = 0.15

        # Write back with pretty formatting
        $updatedJson = $config | ConvertTo-Json -Depth 10
        Set-Content -Path $ConfigPath -Value $updatedJson -Encoding UTF8 -NoNewline

        Write-Host "  Configuration updated: $ConfigPath" -ForegroundColor Green
        Write-Host "  $mismatchCount setting(s) changed to recommended values." -ForegroundColor Green
        Write-Host ''
        Write-Host '  NOTE: Model change must be applied manually via Ollama:' -ForegroundColor Yellow
        Write-Host "    ollama pull $recommendedModel" -ForegroundColor Yellow
        Write-Host ''
    }
}
elseif ($Apply -and $mismatchCount -eq 0) {
    Write-Host '  All settings already match recommendations. No changes needed.' -ForegroundColor Green
}
#endregion
