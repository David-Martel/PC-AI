# TTS Toolset Installer Script
# Run from: C:\users\david\
# Models install to: T:\models\

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

# Colors for output
function Write-ColorOutput($ForegroundColor) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) {
        Write-Output $args
    }
    $host.UI.RawUI.ForegroundColor = $fc
}

function Write-Success($message) { Write-ColorOutput Green "✓ $message" }
function Write-Info($message) { Write-ColorOutput Cyan "→ $message" }
function Write-Warning($message) { Write-ColorOutput Yellow "⚠ $message" }
function Write-Error($message) { Write-ColorOutput Red "✗ $message" }
# Check prerequisites
function Test-Prerequisites {
    Write-Info "Checking prerequisites..."

    # Check if uv is installed
    try {
        $uvVersion = uv --version 2>$null
        Write-Success "uv found: $uvVersion"
    }
    catch {
        Write-Error "uv not found. Install from: https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    }

    # Check NVIDIA GPU
    try {
        $gpuInfo = nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>$null
        Write-Success "NVIDIA GPU(s) detected:"
        $gpuInfo | ForEach-Object { Write-Info "  $_" }
    }
    catch {
        Write-Warning "NVIDIA GPU not detected or nvidia-smi not available"
    }

    # Check CUDA
    try {
        $cudaVersion = nvcc --version 2>$null | Select-String "release"
        Write-Success "CUDA toolkit: $($cudaVersion.Line.Trim())"
    }
    catch {
        Write-Warning "CUDA toolkit not found"
    }
}
# Create project structure
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

# Create virtual environment
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
# Activate virtual environment
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
        Write-Error "Activation script not found: $activateScript"
        exit 1
    }
}

# Install Python packages
function Install-Packages {
    Write-Info "Installing PyTorch with CUDA 12.1 support..."

    # Install PyTorch with CUDA support
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to install PyTorch"
        exit 1
    }
    Write-Success "PyTorch with CUDA support installed"

    # Install Coqui TTS
    Write-Info "Installing Coqui TTS..."
    uv pip install TTS
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to install TTS"
        exit 1
    }
    Write-Success "Coqui TTS installed"

    # Install additional useful packages
    Write-Info "Installing additional packages..."
    uv pip install librosa soundfile numpy scipy
    Write-Success "Additional packages installed"
}
# Set environment variables
function Set-EnvironmentVariables {
    Write-Info "Setting environment variables..."

    # Set TTS_HOME for current session
    $env:TTS_HOME = $MODELS_DIR
    Write-Success "TTS_HOME set to: $MODELS_DIR"

    # Create a .env file for persistence
    $envFile = "$PROJECT_DIR\.env"
    @"
# TTS Environment Variables
TTS_HOME=$MODELS_DIR
CUDA_VISIBLE_DEVICES=0
"@ | Out-File -FilePath $envFile -Encoding UTF8
    Write-Success "Environment file created: $envFile"
}

# Download models
function Get-TTSModels {
    if ($SkipModels) {
        Write-Warning "Skipping model downloads"
        return
    }

    Write-Info "Downloading TTS models to $MODELS_DIR..."

    # Essential models to download
    $models = @(
        "tts_models/multilingual/multi-dataset/xtts_v2",
        "tts_models/en/ljspeech/tacotron2-DDC",
        "vocoder_models/en/ljspeech/hifigan_v2",
        "tts_models/en/ljspeech/glow-tts"
    )

    foreach ($model in $models) {
        Write-Info "Downloading model: $model"

        # Use Python to download via TTS API
        $pythonScript = @"
import os
os.environ['TTS_HOME'] = r'$MODELS_DIR'
from TTS.utils.manage import ModelManager
manager = ModelManager()
try:
    manager.download_model('$model')
    print('✓ Downloaded: $model')
except Exception as e:
    print('✗ Failed to download ' + '$model' + ': ' + str(e))
    exit(1)
"@

        $pythonScript | python
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Downloaded: $model"
        }
        else {
            Write-Error "Failed to download: $model"
        }
    }
}
# Test installation
function Test-Installation {
    Write-Info "Testing installation..."

    # Test PyTorch CUDA
    $testScript = @"
import torch
import TTS

print('PyTorch version: ' + torch.__version__)
print('CUDA available: ' + str(torch.cuda.is_available()))
if torch.cuda.is_available():
    print('CUDA version: ' + torch.version.cuda)
    print('GPU count: ' + str(torch.cuda.device_count()))
    for i in range(torch.cuda.device_count()):
        print('GPU ' + str(i) + ': ' + torch.cuda.get_device_name(i))

print('TTS version: ' + TTS.__version__)

# Test model loading
try:
    from TTS.api import TTS
    print('✓ TTS API import successful')
except Exception as e:
    print('✗ TTS API import failed: ' + str(e))
"@

    $testScript | python

    if ($LASTEXITCODE -eq 0) {
        Write-Success "Installation test passed!"
    }
    else {
        Write-Error "Installation test failed"
        exit 1
    }
}
# Generate sample TTS
function New-SampleTTS {
    Write-Info "Generating sample TTS audio..."

    $sampleScript = @"
import os
os.environ['TTS_HOME'] = r'$MODELS_DIR'
from TTS.api import TTS

# Initialize TTS with XTTS model
tts = TTS(model_name='tts_models/multilingual/multi-dataset/xtts_v2', gpu=True)

# Generate sample
text = "Hello! This is a test of the Coqui TTS system running on your RTX 2000 Ada Generation GPU."
output_path = r'$PROJECT_DIR\sample_output.wav'

tts.tts_to_file(text=text, file_path=output_path)
print('✓ Sample audio saved to: ' + output_path)
"@

    $sampleScript | python

    if ($LASTEXITCODE -eq 0) {
        Write-Success "Sample TTS generated successfully!"
        Write-Info "Audio file: $PROJECT_DIR\sample_output.wav"
    }
    else {
        Write-Warning "Sample TTS generation failed (models may need time to download)"
    }
}
# Create usage script
function New-UsageScript {
    Write-Info "Creating usage scripts..."

    # PowerShell wrapper script
    $wrapperScript = @"
# TTS Usage Script
# Activate environment and set variables

# Activate virtual environment
& "$VENV_PATH\Scripts\Activate.ps1"

# Set environment variables
`$env:TTS_HOME = "$MODELS_DIR"
`$env:CUDA_VISIBLE_DEVICES = "0"

Write-Host "TTS Environment Ready!" -ForegroundColor Green
Write-Host "Project Directory: $PROJECT_DIR" -ForegroundColor Cyan
Write-Host "Models Directory: $MODELS_DIR" -ForegroundColor Cyan
Write-Host ""
Write-Host "Example usage:" -ForegroundColor Yellow
Write-Host 'tts --text "Hello world" --model_name tts_models/multilingual/multi-dataset/xtts_v2 --out_path output.wav' -ForegroundColor Gray
Write-Host ""
"@

    $wrapperScript | Out-File -FilePath "$PROJECT_DIR\activate-tts.ps1" -Encoding UTF8
    # Python helper script
    $helperScript = @"
#!/usr/bin/env python3
"""
TTS Helper Script
Quick TTS generation with your setup
"""
import os
import argparse
from pathlib import Path

# Set environment
os.environ['TTS_HOME'] = r'$MODELS_DIR'

def main():
    parser = argparse.ArgumentParser(description='Generate TTS audio')
    parser.add_argument('text', help='Text to synthesize')
    parser.add_argument('-o', '--output', default='output.wav', help='Output file path')
    parser.add_argument('-m', '--model', default='tts_models/multilingual/multi-dataset/xtts_v2', help='TTS model')
    parser.add_argument('--voice', help='Voice cloning reference file')

    args = parser.parse_args()

    from TTS.api import TTS

    print('Loading model: ' + args.model)
    tts = TTS(model_name=args.model, gpu=True)

    if args.voice:
        print('Using voice reference: ' + args.voice)
        tts.tts_to_file(text=args.text, speaker_wav=args.voice, file_path=args.output)
    else:
        tts.tts_to_file(text=args.text, file_path=args.output)

    print('✓ Audio saved to: ' + args.output)

if __name__ == '__main__':
    main()
"@

    $helperScript | Out-File -FilePath "$PROJECT_DIR\tts_helper.py" -Encoding UTF8

    Write-Success "Usage scripts created:"
    Write-Info "  PowerShell: $PROJECT_DIR\activate-tts.ps1"
    Write-Info "  Python Helper: $PROJECT_DIR\tts_helper.py"
}
# Main installation function
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
        Set-EnvironmentVariables
        Get-TTSModels
        New-UsageScript
    }

    Test-Installation

    if (!$TestOnly -and !$SkipModels) {
        New-SampleTTS
    }

    Write-Success ""
    Write-Success "🎉 TTS Toolset installation complete!"
    Write-Info ""
    Write-Info "To use TTS:"
    Write-Info "1. Run: .\activate-tts.ps1"
    Write-Info "2. Use: python tts_helper.py 'Hello world' -o test.wav"
    Write-Info "3. Or: tts --text 'Hello' --model_name tts_models/multilingual/multi-dataset/xtts_v2 --out_path out.wav"
    Write-Info ""
}

# Run installation
if ($MyInvocation.InvocationName -ne '.') {
    Install-TTSToolset
}