# Shock Lab environment setup (Windows / PowerShell).
#
# Creates a virtual environment, installs the Python deps, and installs PyTorch
# with the correct CUDA wheel for this machine's GPU.
#
#   This machine: NVIDIA GeForce RTX 5070 (Blackwell, compute capability sm_120).
#   Blackwell requires a CUDA 12.8+ PyTorch wheel. We use the cu128 index; the
#   driver (CUDA 13.x) is backward compatible with it.
#
# Usage:  .\scripts\setup.ps1

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

Write-Host "== Creating virtual environment (.venv) ==" -ForegroundColor Cyan
if (-not (Test-Path ".venv")) { python -m venv .venv }
& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip

Write-Host "== Installing Python dependencies ==" -ForegroundColor Cyan
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt

Write-Host "== Installing PyTorch (CUDA 12.8 wheel, for Blackwell sm_120) ==" -ForegroundColor Cyan
& ".\.venv\Scripts\python.exe" -m pip install torch --index-url https://download.pytorch.org/whl/cu128

Write-Host "== Verifying CUDA ==" -ForegroundColor Cyan
& ".\.venv\Scripts\python.exe" -c "import torch; print('torch', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

Write-Host "== Frontend deps (web/) ==" -ForegroundColor Cyan
if (Test-Path "web\package.json") {
    Push-Location web
    npm install
    Pop-Location
} else {
    Write-Host "web/ not scaffolded yet (arrives in M6); skipping npm install." -ForegroundColor Yellow
}

Write-Host "Setup complete." -ForegroundColor Green
