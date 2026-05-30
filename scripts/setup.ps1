# ---------------------------------------------------------------------------
# Crop Recommendation System — initial workspace setup (Windows PowerShell)
# ---------------------------------------------------------------------------
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvDir = Join-Path $ProjectRoot ".venv"
$Requirements = Join-Path $ProjectRoot "requirements.txt"

Write-Host "==> Project root: $ProjectRoot"

# 1. Create virtual environment
if (-not (Test-Path $VenvDir)) {
    Write-Host "==> Creating virtual environment at $VenvDir"
    python -m venv $VenvDir
} else {
    Write-Host "==> Virtual environment already exists at $VenvDir"
}

# 2. Activate and install dependencies
Write-Host "==> Activating virtual environment"
& "$VenvDir\Scripts\Activate.ps1"

Write-Host "==> Upgrading pip"
python -m pip install --upgrade pip

Write-Host "==> Installing requirements"
pip install -r $Requirements

Write-Host ""
Write-Host "Setup complete. Next steps:"
Write-Host "  1. .\.venv\Scripts\Activate.ps1"
Write-Host "  2. Place raw CSV in data\raw\crop_recommendation.csv"
Write-Host "  3. python src\train.py"
Write-Host "  4. streamlit run app\app.py"
