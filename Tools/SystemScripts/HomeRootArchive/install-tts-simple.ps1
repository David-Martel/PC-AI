# Simple TTS Toolset Installer
# Installs TTS with separate Python helper scripts

param(
    [switch]$SkipVenv,
    [switch]$SkipModels,
    [switch]$TestOnly
)

# Configuration
$PROJECT_DIR = "C:\users\david\tts-project"
$MODELS_DIR = "T:\models"
$VENV_NAME = "tts-env"
$VENV_PATH = "$PROJECT_DIR\$VENV_NAME"

function Write-Success($message) { Write-Host "✓ $message" -ForegroundColor Green }
function Write-Info($message) { Write-Host "→ $message" -ForegroundColor Cyan }
function Write-Warning($message) { Write-Host "⚠ $message" -ForegroundColor Yellow }
function Write-Error($message) { Write-Host "✗ $message" -ForegroundColor Red }

function Test-Prerequisites {
    Write-Info "Checking prerequisites..."

    try {
        $uvVersion = uv --version 2>$null
        Write-Success "uv found: $uvVersion"
    }
    catch {
        Write-Error "uv not found. Install from: https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    }

    try {
        $gpuInfo = nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>$null
        Write-Success "NVIDIA GPU(s) detected:"
        $gpuInfo | ForEach-Object { Write-Info "  $_" }
    }
    catch {
        Write-Warning "NVIDIA GPU not detected"
    }

    try {
        $cudaVersion = nvcc --version 2>$null | Select-String "release"
        Write-Success "CUDA toolkit found"
    }
    catch {
        Write-Warning "CUDA toolkit not found"
    }
}

function Initialize-Project {
    Write-Info "Creating project structure..."

    if (!(Test-Path $PROJECT_DIR)) {
        New-Item -ItemType Directory -Path $PROJECT_DIR -Force | Out-Null
        Write-Success "Created project directory: $PROJECT_DIR"
    }

    if (!(Test-Path $MODELS_DIR)) {
        New-Item -ItemType Directory -Path $MODELS_DIR -Force | Out-Null
        Write-Success "Created models directory: $MODELS_DIR"
    }

    Set-Location $PROJECT_DIR
    Write-Success "Changed to project directory"
}

function New-VirtualEnvironment {
    if ($SkipVenv) {
        Write-Warning "Skipping virtual environment creation"
        return
    }

    Write-Info "Creating virtual environment with uv..."

    if (Test-Path $VENV_PATH) {
        Write-Warning "Virtual environment already exists, removing..."
        Remove-Item -Recurse -Force $VENV_PATH
    }

    uv venv $VENV_NAME
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Virtual environment created: $VENV_NAME"
    }
    else {
        Write-Error "Failed to create virtual environment"
        exit 1
    }
}

function Enable-VirtualEnvironment {
    if ($SkipVenv) {
        Write-Warning "Skipping virtual environment activation"
        return
    }

    $activateScript = "$VENV_PATH\Scripts\Activate.ps1"
    if (Test-Path $activateScript) {
        & $activateScript
        Write-Success "Virtual environment activated"
    }
    else {
        Write-Error "Activation script not found"
        exit 1
    }
}

function Install-Packages {
    Write-Info "Installing PyTorch with CUDA 12.1 support..."

    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to install PyTorch"
        exit 1
    }
    Write-Success "PyTorch with CUDA support installed"

    Write-Info "Installing Coqui TTS..."
    uv pip install TTS
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to install TTS"
        exit 1
    }
    Write-Success "Coqui TTS installed"

    Write-Info "Installing additional packages..."
    uv pip install librosa soundfile numpy scipy
    Write-Success "Additional packages installed"
}

function Test-Installation {
    Write-Info "Testing installation..."
    python C:\users\david\test_installation.py
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Installation test passed!"
    }
    else {
        Write-Error "Installation test failed"
        exit 1
    }
}

function Get-TTSModels {
    if ($SkipModels) {
        Write-Warning "Skipping model downloads"
        return
    }

    Write-Info "Downloading TTS models..."
    python C:\users\david\download_models.py $MODELS_DIR
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Models downloaded successfully"
    }
    else {
        Write-Error "Model download failed"
        exit 1
    }
}

function New-SampleTTS {
    Write-Info "Generating sample TTS audio..."
    python C:\users\david\generate_sample.py $MODELS_DIR "$PROJECT_DIR\sample_output.wav"
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Sample TTS generated successfully!"
        Write-Info "Audio file: $PROJECT_DIR\sample_output.wav"
    }
    else {
        Write-Warning "Sample TTS generation failed"
    }
}

function New-UsageScript {
    Write-Info "Creating activation script..."

    $activationScript = @"
# TTS Usage Script
& "$VENV_PATH\Scripts\Activate.ps1"
`$env:TTS_HOME = "$MODELS_DIR"
`$env:CUDA_VISIBLE_DEVICES = "0"

Write-Host "TTS Environment Ready!" -ForegroundColor Green
Write-Host "Project Directory: $PROJECT_DIR" -ForegroundColor Cyan
Write-Host "Models Directory: $MODELS_DIR" -ForegroundColor Cyan
Write-Host ""
Write-Host "Example usage:" -ForegroundColor Yellow
Write-Host 'python C:\users\david\tts_helper.py "Hello world" -o test.wav' -ForegroundColor Gray
Write-Host ""
"@

    $activationScript | Out-File -FilePath "$PROJECT_DIR\activate-tts.ps1" -Encoding UTF8
    Write-Success "Activation script created: activate-tts.ps1"
}

function Install-TTSToolset {
    Write-Info "Starting TTS Toolset installation..."
    Write-Info "Project: $PROJECT_DIR"
    Write-Info "Models: $MODELS_DIR"
    Write-Info ""

    Test-Prerequisites
    Initialize-Project

    if (!$TestOnly) {
        New-VirtualEnvironment
        Enable-VirtualEnvironment
        Install-Packages
        Get-TTSModels
        New-UsageScript
    }

    Test-Installation

    if (!$TestOnly -and !$SkipModels) {
        New-SampleTTS
    }

    Write-Success ""
    Write-Success "TTS Toolset installation complete!"
    Write-Info ""
    Write-Info "To use TTS:"
    Write-Info "1. Run: .\activate-tts.ps1"
    Write-Info "2. Use: python C:\users\david\tts_helper.py `"Hello world`" -o test.wav"
    Write-Info ""
}

# Run installation
Install-TTSToolset
