# TTS Installer - Python 3.12 Compatible
param(
    [switch]$SkipVenv,
    [switch]$SkipModels,
    [switch]$TestOnly
)

$PROJECT_DIR = "C:\users\david\tts-project"
$MODELS_DIR = "T:\models"
$VENV_NAME = "tts-env-312"
$VENV_PATH = "$PROJECT_DIR\$VENV_NAME"

Write-Host "TTS Installation with Python 3.12..." -ForegroundColor Green

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
    # Create virtual environment with Python 3.12
    if (!$SkipVenv) {
        Write-Host "Creating virtual environment with Python 3.12..." -ForegroundColor Cyan
        if (Test-Path $VENV_PATH) {
            Remove-Item -Recurse -Force $VENV_PATH
        }

        # Use uv to create venv with specific Python version
        uv venv $VENV_NAME --python 3.12

        if ($LASTEXITCODE -ne 0) {
            Write-Host "Failed to create Python 3.12 environment. Trying system Python..." -ForegroundColor Yellow
            uv venv $VENV_NAME
        }

        # Activate virtual environment
        & "$VENV_PATH\Scripts\Activate.ps1"
    }

    # Install packages
    Write-Host "Installing PyTorch with CUDA 12.1..." -ForegroundColor Cyan
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

    if ($LASTEXITCODE -ne 0) {
        Write-Host "PyTorch installation failed. Continuing anyway..." -ForegroundColor Yellow
    }

    Write-Host "Installing TTS..." -ForegroundColor Cyan
    uv pip install TTS

    if ($LASTEXITCODE -ne 0) {
        Write-Host "TTS installation failed." -ForegroundColor Red
        exit 1
    }

    Write-Host "Installing additional packages..." -ForegroundColor Cyan
    uv pip install librosa soundfile numpy scipy

    # Download models
    if (!$SkipModels) {
        Write-Host "Downloading models..." -ForegroundColor Cyan
        python C:\users\david\download_models.py $MODELS_DIR
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Model download failed, but continuing..." -ForegroundColor Yellow
        }
    }

    # Create activation script
    Write-Host "Creating activation script..." -ForegroundColor Cyan
    $activationScript = @"
# TTS Activation Script
& "$VENV_PATH\Scripts\Activate.ps1"
`$env:TTS_HOME = "$MODELS_DIR"
`$env:CUDA_VISIBLE_DEVICES = "0"

Write-Host "TTS Environment Ready!" -ForegroundColor Green
Write-Host "Project Directory: $PROJECT_DIR" -ForegroundColor Cyan
Write-Host "Models Directory: $MODELS_DIR" -ForegroundColor Cyan
Write-Host ""
Write-Host "Example usage:" -ForegroundColor Yellow
Write-Host "python C:\users\david\tts_helper.py 'Hello world' -o test.wav" -ForegroundColor Gray
Write-Host ""
"@

    $activationScript | Out-File -FilePath "$PROJECT_DIR\activate-tts.ps1" -Encoding UTF8
}

# Test installation
Write-Host "Testing installation..." -ForegroundColor Cyan
python C:\users\david\test_installation.py

if ($LASTEXITCODE -eq 0) {
    Write-Host "Installation test PASSED!" -ForegroundColor Green

    if (!$TestOnly -and !$SkipModels) {
        # Generate sample
        Write-Host "Generating sample audio..." -ForegroundColor Cyan
        python C:\users\david\generate_sample.py $MODELS_DIR "$PROJECT_DIR\sample_output.wav"
    }

    Write-Host ""
    Write-Host "Installation complete!" -ForegroundColor Green
    Write-Host "To use TTS:" -ForegroundColor Yellow
    Write-Host "1. Run: .\activate-tts.ps1" -ForegroundColor Gray
    Write-Host "2. Use: python C:\users\david\tts_helper.py 'Hello world' -o test.wav" -ForegroundColor Gray

} else {
    Write-Host "Installation test FAILED. Check the errors above." -ForegroundColor Red
}
