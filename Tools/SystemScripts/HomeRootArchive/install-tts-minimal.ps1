# Very Simple TTS Installer
param(
    [switch]$SkipVenv,
    [switch]$SkipModels,
    [switch]$TestOnly
)

$PROJECT_DIR = "C:\users\david\tts-project"
$MODELS_DIR = "T:\models"
$VENV_NAME = "tts-env"
$VENV_PATH = "$PROJECT_DIR\$VENV_NAME"

Write-Host "Starting TTS installation..." -ForegroundColor Green

# Check uv
try {
    $uvVersion = uv --version 2>$null
    Write-Host "uv found: $uvVersion" -ForegroundColor Green
}
catch {
    Write-Host "uv not found. Please install uv first." -ForegroundColor Red
    exit 1
}

# Create directories
Write-Host "Creating directories..." -ForegroundColor Cyan
if (!(Test-Path $PROJECT_DIR)) {
    New-Item -ItemType Directory -Path $PROJECT_DIR -Force | Out-Null
}
if (!(Test-Path $MODELS_DIR)) {
    New-Item -ItemType Directory -Path $MODELS_DIR -Force | Out-Null
}

Set-Location $PROJECT_DIR

if (!$TestOnly) {
    # Create virtual environment
    if (!$SkipVenv) {
        Write-Host "Creating virtual environment..." -ForegroundColor Cyan
        if (Test-Path $VENV_PATH) {
            Remove-Item -Recurse -Force $VENV_PATH
        }
        uv venv $VENV_NAME

        # Activate virtual environment
        & "$VENV_PATH\Scripts\Activate.ps1"
    }

    # Install packages
    Write-Host "Installing PyTorch..." -ForegroundColor Cyan
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

    Write-Host "Installing TTS..." -ForegroundColor Cyan
    uv pip install TTS

    Write-Host "Installing additional packages..." -ForegroundColor Cyan
    uv pip install librosa soundfile numpy scipy

    # Download models
    if (!$SkipModels) {
        Write-Host "Downloading models..." -ForegroundColor Cyan
        python C:\users\david\download_models.py $MODELS_DIR
    }
}

# Test installation
Write-Host "Testing installation..." -ForegroundColor Cyan
python C:\users\david\test_installation.py

if (!$TestOnly -and !$SkipModels) {
    # Generate sample
    Write-Host "Generating sample audio..." -ForegroundColor Cyan
    python C:\users\david\generate_sample.py $MODELS_DIR "$PROJECT_DIR\sample_output.wav"
}

Write-Host ""
Write-Host "Installation complete!" -ForegroundColor Green
Write-Host "To use TTS:" -ForegroundColor Yellow
Write-Host "  python C:\users\david\tts_helper.py 'Hello world' -o test.wav" -ForegroundColor Gray
