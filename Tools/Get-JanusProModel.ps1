#Requires -Version 5.1
<#
.SYNOPSIS
    Download the Janus-Pro model (and optional RealESRGAN upscaler) for the PC-AI media pipeline.

.DESCRIPTION
    Retrieves a HuggingFace API token from the Bitwarden CLI vault, lists the files in the
    requested HuggingFace model repository via the Hub API, downloads each required file with
    progress reporting, and verifies each download against the size reported by the API.

    After a successful Janus-Pro download the script updates Config/pcai-media.json so that
    the `model` field points to the downloaded directory.

    RealESRGAN upscaling weights (RealESRGAN_x4plus.onnx) are also downloaded from the
    xinntao/Real-ESRGAN GitHub release if they are not already present.

    Required files downloaded for Janus-Pro:
      config.json, tokenizer.json, tokenizer_config.json, generation_config.json,
      and all *.safetensors shards.

    Bitwarden unlock:
      If `bw` reports the vault is locked the script will prompt for the master password,
      unlock the vault, and set BW_SESSION for the current process automatically.

.PARAMETER ModelId
    HuggingFace model identifier in the form <owner>/<name>.
    Default: 'deepseek-ai/Janus-Pro-1B'

.PARAMETER OutputDir
    Override the output directory. When omitted the model is placed under
    <ProjectRoot>/Models/<model-name>/ where <model-name> is the basename of -ModelId.

.PARAMETER BitwardenSearch
    Search string passed to `bw list items --search`. Default: 'hugging'

.PARAMETER Force
    Re-download files even if they already exist and their size matches the API metadata.

.PARAMETER SkipAuth
    Skip Bitwarden lookup and perform anonymous downloads. Useful for public models that
    do not require authentication.

.EXAMPLE
    .\Tools\Get-JanusProModel.ps1
    Downloads deepseek-ai/Janus-Pro-1B to Models/Janus-Pro-1B/ using the HF token from Bitwarden.

.EXAMPLE
    .\Tools\Get-JanusProModel.ps1 -ModelId 'deepseek-ai/Janus-Pro-7B' -OutputDir 'D:\Models\Janus-7B'
    Downloads an alternate model size to a custom path.

.EXAMPLE
    .\Tools\Get-JanusProModel.ps1 -SkipAuth -Force
    Re-downloads all files without authentication (public access).
#>

[CmdletBinding(SupportsShouldProcess)]
param(
    [string]$ModelId = 'deepseek-ai/Janus-Pro-1B',

    [string]$OutputDir = '',

    [string]$BitwardenSearch = 'hugging',

    [switch]$Force,

    [switch]$SkipAuth
)

$ErrorActionPreference = 'Stop'

# ---------------------------------------------------------------------------
# Helper: Format bytes as a human-readable string
# ---------------------------------------------------------------------------
function Format-Bytes {
    param([long]$Bytes)
    if ($Bytes -ge 1GB) { return '{0:N2} GB' -f ($Bytes / 1GB) }
    if ($Bytes -ge 1MB) { return '{0:N2} MB' -f ($Bytes / 1MB) }
    if ($Bytes -ge 1KB) { return '{0:N1} KB' -f ($Bytes / 1KB) }
    return "$Bytes B"
}

# ---------------------------------------------------------------------------
# Helper: Locate the project root by walking up from $PSScriptRoot
# ---------------------------------------------------------------------------
function Find-ProjectRoot {
    $dir = $PSScriptRoot
    while ($dir) {
        if (Test-Path (Join-Path $dir 'PC-AI.ps1')) {
            return $dir
        }
        $parent = Split-Path $dir -Parent
        if (-not $parent -or $parent -eq $dir) { break }
        $dir = $parent
    }
    # Fallback: use the parent of Tools/
    return (Split-Path $PSScriptRoot -Parent)
}

# ---------------------------------------------------------------------------
# Helper: Obtain a Bitwarden session token, unlocking if needed
# ---------------------------------------------------------------------------
function Get-Bitwarden-Session {
    # Check for an already-unlocked session in the environment
    if ($env:BW_SESSION) {
        Write-Verbose 'Using existing BW_SESSION from environment.'
        return $env:BW_SESSION
    }

    # Check vault status
    $statusJson = & bw status 2>&1
    try {
        $status = $statusJson | ConvertFrom-Json
    }
    catch {
        Write-Warning "Could not parse bw status output. Output was: $statusJson"
        $status = $null
    }

    if ($status -and $status.status -eq 'unlocked') {
        Write-Verbose 'Bitwarden vault is already unlocked.'
        return $null   # bw commands will work without an explicit session
    }

    Write-Host 'Bitwarden vault is locked. Enter your master password to unlock:' -ForegroundColor Yellow
    $password = Read-Host -AsSecureString 'Master password'
    $plainPassword = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
        [Runtime.InteropServices.Marshal]::SecureStringToBSTR($password)
    )

    $session = & bw unlock $plainPassword --raw 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "bw unlock failed (exit $LASTEXITCODE): $session"
    }

    # Persist for this process
    $env:BW_SESSION = $session
    Write-Host 'Bitwarden vault unlocked.' -ForegroundColor Green
    return $session
}

# ---------------------------------------------------------------------------
# Helper: Retrieve the HuggingFace API token from Bitwarden
# ---------------------------------------------------------------------------
function Get-HuggingFaceToken {
    param([string]$SearchTerm)

    Write-Host "Searching Bitwarden vault for '$SearchTerm'..." -ForegroundColor Cyan

    $sessionArgs = @()
    if ($env:BW_SESSION) {
        $sessionArgs = @('--session', $env:BW_SESSION)
    }

    $itemsJson = & bw list items --search $SearchTerm @sessionArgs 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "bw list items failed (exit $LASTEXITCODE): $itemsJson"
    }

    $items = $itemsJson | ConvertFrom-Json
    if (-not $items -or $items.Count -eq 0) {
        throw "No Bitwarden items found matching '$SearchTerm'. Add your HuggingFace token to Bitwarden first."
    }

    # Prefer items whose name mentions huggingface / hf / token
    $preferred = $items | Where-Object {
        $_.name -imatch '(hugging|huggingface|hf|token)' -or
        ($_.login -and $_.login.username -imatch '(hugging|hf)')
    } | Select-Object -First 1

    if (-not $preferred) {
        $preferred = $items | Select-Object -First 1
        Write-Warning "Could not find an obvious HuggingFace item; using first match: '$($preferred.name)'"
    }
    else {
        Write-Host "  Found: '$($preferred.name)'" -ForegroundColor Green
    }

    # Try login password field first, then notes field
    $token = $null
    if ($preferred.login -and $preferred.login.password) {
        $token = $preferred.login.password
    }
    elseif ($preferred.notes) {
        # Notes may contain the raw token or "hf_..." token on a line
        $match = [regex]::Match($preferred.notes, '(hf_[A-Za-z0-9]{20,})')
        if ($match.Success) {
            $token = $match.Value
        }
        else {
            $token = $preferred.notes.Trim()
        }
    }
    elseif ($preferred.fields) {
        # Custom fields
        $tokenField = $preferred.fields | Where-Object { $_.name -imatch 'token|key|api' } | Select-Object -First 1
        if ($tokenField) { $token = $tokenField.value }
    }

    if (-not $token) {
        throw "Found Bitwarden item '$($preferred.name)' but could not extract a token value. Check the item structure."
    }

    return $token
}

# ---------------------------------------------------------------------------
# Helper: Build common HTTP request headers
# ---------------------------------------------------------------------------
function New-HfHeaders {
    param([string]$Token)
    $headers = @{ 'User-Agent' = 'PC-AI-ModelDownloader/1.0' }
    if ($Token) {
        $headers['Authorization'] = "Bearer $Token"
    }
    return $headers
}

# ---------------------------------------------------------------------------
# Helper: Invoke-WebRequest with a single retry on transient failure
# ---------------------------------------------------------------------------
function Invoke-WebRequestWithRetry {
    param(
        [string]$Uri,
        [hashtable]$Headers,
        [string]$OutFile = $null,
        [switch]$UseBasicParsing
    )

    $attempt = 0
    $maxAttempts = 2
    while ($attempt -lt $maxAttempts) {
        $attempt++
        try {
            $iwrArgs = @{
                Uri             = $Uri
                Headers         = $Headers
                UseBasicParsing = $true
            }
            if ($OutFile) {
                $iwrArgs['OutFile'] = $OutFile
            }
            return Invoke-WebRequest @iwrArgs
        }
        catch {
            if ($attempt -ge $maxAttempts) {
                throw
            }
            $code = $_.Exception.Response?.StatusCode.value__
            Write-Warning "  Attempt $attempt failed (HTTP $code). Retrying..."
            Start-Sleep -Seconds 3
        }
    }
}

# ---------------------------------------------------------------------------
# Helper: Download a single file from HuggingFace
# ---------------------------------------------------------------------------
function Invoke-HfFileDownload {
    param(
        [string]$ModelId,
        [string]$FileName,
        [string]$DestPath,
        [hashtable]$Headers,
        [long]$ExpectedBytes = -1,
        [switch]$Force
    )

    $url = "https://huggingface.co/$ModelId/resolve/main/$FileName"

    # Skip if the file already exists with the right size
    if ((Test-Path $DestPath) -and -not $Force) {
        if ($ExpectedBytes -gt 0) {
            $existingSize = (Get-Item $DestPath).Length
            if ($existingSize -eq $ExpectedBytes) {
                Write-Host "  [SKIP] $FileName ($(Format-Bytes $existingSize), already present)" -ForegroundColor DarkGray
                return $true
            }
            Write-Warning "  [SIZE MISMATCH] $FileName exists but is $(Format-Bytes $existingSize), expected $(Format-Bytes $ExpectedBytes). Re-downloading."
        }
        else {
            Write-Host "  [SKIP] $FileName (already present, no size metadata)" -ForegroundColor DarkGray
            return $true
        }
    }

    $sizeStr = if ($ExpectedBytes -gt 0) { " ($(Format-Bytes $ExpectedBytes))" } else { '' }
    Write-Host "  [DOWN] $FileName$sizeStr" -ForegroundColor Cyan

    $tmpPath = "$DestPath.tmp"
    try {
        Invoke-WebRequestWithRetry -Uri $url -Headers $Headers -OutFile $tmpPath
    }
    catch {
        $code = $_.Exception.Response?.StatusCode.value__
        Write-Error "  Failed to download $FileName from $url (HTTP $code): $($_.Exception.Message)"
        if (Test-Path $tmpPath) { Remove-Item $tmpPath -Force -ErrorAction SilentlyContinue }
        return $false
    }

    # Verify size
    if ($ExpectedBytes -gt 0) {
        $actualSize = (Get-Item $tmpPath).Length
        if ($actualSize -ne $ExpectedBytes) {
            Write-Warning "  Size mismatch for $FileName`: got $(Format-Bytes $actualSize), expected $(Format-Bytes $ExpectedBytes)."
        }
    }

    Move-Item -Path $tmpPath -Destination $DestPath -Force
    $finalSize = (Get-Item $DestPath).Length
    Write-Host "  [OK]   $FileName ($(Format-Bytes $finalSize))" -ForegroundColor Green
    return $true
}

# ---------------------------------------------------------------------------
# Helper: Download RealESRGAN ONNX weights from GitHub
# ---------------------------------------------------------------------------
function Invoke-RealEsrganDownload {
    param(
        [string]$DestDir,
        [hashtable]$Headers,
        [switch]$Force
    )

    $onnxDest = Join-Path $DestDir 'RealESRGAN_x4.onnx'

    if ((Test-Path $onnxDest) -and -not $Force) {
        $sz = (Get-Item $onnxDest).Length
        Write-Host "  [SKIP] RealESRGAN_x4.onnx ($(Format-Bytes $sz), already present)" -ForegroundColor DarkGray
        return $true
    }

    # Canonical GitHub Releases source for the ONNX export
    # xinntao/Real-ESRGAN releases contain a dedicated ONNX asset
    $releaseApiUrl = 'https://api.github.com/repos/xinntao/Real-ESRGAN/releases'
    Write-Host 'Querying GitHub API for latest Real-ESRGAN release...' -ForegroundColor Cyan

    $ghHeaders = @{ 'User-Agent' = 'PC-AI-ModelDownloader/1.0' }

    try {
        $releasesJson = Invoke-WebRequest -Uri $releaseApiUrl -Headers $ghHeaders -UseBasicParsing
        $releases = $releasesJson.Content | ConvertFrom-Json
    }
    catch {
        Write-Warning "  GitHub API call failed: $($_.Exception.Message). Trying direct URL fallback."
        $releases = $null
    }

    $onnxUrl = $null
    if ($releases) {
        foreach ($release in $releases) {
            $asset = $release.assets | Where-Object {
                $_.name -imatch 'RealESRGAN_x4plus.*\.onnx'
            } | Select-Object -First 1
            if ($asset) {
                $onnxUrl = $asset.browser_download_url
                Write-Host "  Found ONNX asset in release '$($release.tag_name)': $($asset.name)" -ForegroundColor Green
                break
            }
        }
    }

    # Hard fallback to the well-known v0.3.0 ONNX asset if API search found nothing
    if (-not $onnxUrl) {
        $onnxUrl = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.3.0/RealESRGAN_x4plus.onnx'
        Write-Warning "  Could not locate ONNX asset via API; using fixed fallback URL."
    }

    Write-Host "  [DOWN] RealESRGAN_x4.onnx from $onnxUrl" -ForegroundColor Cyan
    $tmpPath = "$onnxDest.tmp"
    try {
        Invoke-WebRequestWithRetry -Uri $onnxUrl -Headers $ghHeaders -OutFile $tmpPath
    }
    catch {
        Write-Error "  Failed to download RealESRGAN ONNX: $($_.Exception.Message)"
        if (Test-Path $tmpPath) { Remove-Item $tmpPath -Force -ErrorAction SilentlyContinue }
        return $false
    }

    Move-Item -Path $tmpPath -Destination $onnxDest -Force
    $finalSize = (Get-Item $onnxDest).Length
    Write-Host "  [OK]   RealESRGAN_x4.onnx ($(Format-Bytes $finalSize))" -ForegroundColor Green
    return $true
}

# ---------------------------------------------------------------------------
# Helper: Update pcai-media.json with the downloaded model path
# ---------------------------------------------------------------------------
function Update-MediaConfig {
    param(
        [string]$ConfigPath,
        [string]$ModelRelPath,
        [string]$EsrganRelPath
    )

    if (-not (Test-Path $ConfigPath)) {
        Write-Warning "Config not found at $ConfigPath — skipping update."
        return
    }

    $json = Get-Content $ConfigPath -Raw | ConvertFrom-Json

    # Update model path (use forward slashes to match existing convention)
    $json.model = $ModelRelPath -replace '\\', '/'

    # Update upscale model path if the upscale block exists
    if ($json.upscale -and $EsrganRelPath) {
        $json.upscale.model_path = $EsrganRelPath -replace '\\', '/'
    }

    $json | ConvertTo-Json -Depth 10 | Set-Content -Path $ConfigPath -Encoding UTF8
    Write-Host "Updated $ConfigPath" -ForegroundColor Green
    Write-Host "  model       = $($json.model)" -ForegroundColor DarkGray
    if ($json.upscale) {
        Write-Host "  upscale.model_path = $($json.upscale.model_path)" -ForegroundColor DarkGray
    }
}

# ===========================================================================
# MAIN
# ===========================================================================

# --- Resolve project root and output paths ----------------------------------
$projectRoot = Find-ProjectRoot
Write-Verbose "Project root: $projectRoot"

$modelName = Split-Path $ModelId -Leaf   # e.g. 'Janus-Pro-1B'

if ($OutputDir) {
    $janusDestDir = $OutputDir
}
else {
    $janusDestDir = Join-Path $projectRoot "Models\$modelName"
}

$esrganDestDir = Join-Path $projectRoot 'Models\RealESRGAN'
$configPath    = Join-Path $projectRoot 'Config\pcai-media.json'

Write-Host ''
Write-Host 'PC-AI Janus-Pro Model Downloader' -ForegroundColor White
Write-Host '=================================' -ForegroundColor White
Write-Host "  Model ID   : $ModelId" -ForegroundColor DarkCyan
Write-Host "  Destination: $janusDestDir" -ForegroundColor DarkCyan
Write-Host "  Config     : $configPath" -ForegroundColor DarkCyan
Write-Host ''

# --- Obtain HuggingFace token -----------------------------------------------
$hfToken = $null
if (-not $SkipAuth) {
    try {
        $null = Get-Bitwarden-Session
    }
    catch {
        Write-Error "Bitwarden session error: $($_.Exception.Message)"
        exit 1
    }

    try {
        $hfToken = Get-HuggingFaceToken -SearchTerm $BitwardenSearch
    }
    catch {
        Write-Error "Failed to retrieve HuggingFace token: $($_.Exception.Message)"
        exit 1
    }
    Write-Host "HuggingFace token retrieved (length: $($hfToken.Length))." -ForegroundColor Green
}
else {
    Write-Warning 'SkipAuth is set — downloading without authentication.'
}

$hfHeaders = New-HfHeaders -Token $hfToken

# --- Query HuggingFace Hub API for repository file listing ------------------
$repoApiUrl = "https://huggingface.co/api/models/$ModelId"
Write-Host "Querying HuggingFace Hub API: $repoApiUrl" -ForegroundColor Cyan

try {
    $repoResponse = Invoke-WebRequest -Uri $repoApiUrl -Headers $hfHeaders -UseBasicParsing
    $repoInfo = $repoResponse.Content | ConvertFrom-Json
}
catch {
    $code = $_.Exception.Response?.StatusCode.value__
    Write-Error "HuggingFace API query failed (HTTP $code): $($_.Exception.Message)"
    exit 1
}

# The siblings array contains { rfilename, size } entries
$allFiles = $repoInfo.siblings
if (-not $allFiles) {
    Write-Error "No files listed in HuggingFace API response for '$ModelId'. Is the model ID correct?"
    exit 1
}

Write-Host "  Repository contains $($allFiles.Count) file(s)." -ForegroundColor DarkGray

# Required filename patterns
$requiredExact = @(
    'config.json',
    'tokenizer.json',
    'tokenizer_config.json',
    'generation_config.json'
)

$wantedFiles = $allFiles | Where-Object {
    $n = $_.rfilename
    ($requiredExact -contains $n) -or ($n -match '\.safetensors$')
}

if ($wantedFiles.Count -eq 0) {
    Write-Error "No required files matched in the repository listing. Check that $ModelId contains .safetensors files."
    exit 1
}

$totalBytes = ($wantedFiles | Measure-Object -Property size -Sum).Sum
Write-Host "  Files to download: $($wantedFiles.Count)  (~$(Format-Bytes $totalBytes) total)" -ForegroundColor White
Write-Host ''

# --- Ensure destination directory exists ------------------------------------
if (-not (Test-Path $janusDestDir)) {
    if ($PSCmdlet.ShouldProcess($janusDestDir, 'Create directory')) {
        New-Item -ItemType Directory -Path $janusDestDir -Force | Out-Null
        Write-Host "Created directory: $janusDestDir" -ForegroundColor DarkGray
    }
}

# --- Download each required file --------------------------------------------
Write-Host 'Downloading Janus-Pro model files...' -ForegroundColor White
$failed = @()
foreach ($fileInfo in $wantedFiles) {
    $fname    = $fileInfo.rfilename
    $destPath = Join-Path $janusDestDir $fname

    # Handle sub-directories within the repo (e.g. tokenizer/vocab.txt)
    $destSubDir = Split-Path $destPath -Parent
    if (-not (Test-Path $destSubDir)) {
        New-Item -ItemType Directory -Path $destSubDir -Force | Out-Null
    }

    $ok = Invoke-HfFileDownload `
        -ModelId        $ModelId `
        -FileName       $fname `
        -DestPath       $destPath `
        -Headers        $hfHeaders `
        -ExpectedBytes  ($fileInfo.size ?? -1) `
        -Force:$Force

    if (-not $ok) {
        $failed += $fname
    }
}

if ($failed.Count -gt 0) {
    Write-Warning "The following files failed to download:"
    $failed | ForEach-Object { Write-Warning "  - $_" }
}
else {
    Write-Host ''
    Write-Host "All Janus-Pro model files downloaded successfully." -ForegroundColor Green
}

# --- Download RealESRGAN ONNX -----------------------------------------------
Write-Host ''
Write-Host 'Checking RealESRGAN upscaling weights...' -ForegroundColor White

if (-not (Test-Path $esrganDestDir)) {
    if ($PSCmdlet.ShouldProcess($esrganDestDir, 'Create directory')) {
        New-Item -ItemType Directory -Path $esrganDestDir -Force | Out-Null
        Write-Host "Created directory: $esrganDestDir" -ForegroundColor DarkGray
    }
}

$esrganOk = Invoke-RealEsrganDownload `
    -DestDir $esrganDestDir `
    -Headers $hfHeaders `
    -Force:$Force

if (-not $esrganOk) {
    Write-Warning 'RealESRGAN download failed. Upscaling will be unavailable until the file is present.'
}

# --- Update pcai-media.json -------------------------------------------------
Write-Host ''
Write-Host 'Updating Config/pcai-media.json...' -ForegroundColor White

# Build relative paths (relative to project root, matching existing config convention)
$janusRelPath  = "Models/$modelName"
$esrganRelPath = 'Models/RealESRGAN/RealESRGAN_x4.onnx'

if ($PSCmdlet.ShouldProcess($configPath, 'Update model paths')) {
    Update-MediaConfig `
        -ConfigPath    $configPath `
        -ModelRelPath  $janusRelPath `
        -EsrganRelPath $esrganRelPath
}

# --- Final summary ----------------------------------------------------------
Write-Host ''
Write-Host 'Done.' -ForegroundColor White
Write-Host "  Janus-Pro : $janusDestDir" -ForegroundColor DarkGray
Write-Host "  RealESRGAN: $(Join-Path $esrganDestDir 'RealESRGAN_x4.onnx')" -ForegroundColor DarkGray
if ($failed.Count -gt 0) {
    Write-Host "  Warnings  : $($failed.Count) file(s) failed — see above." -ForegroundColor Yellow
    exit 1
}
