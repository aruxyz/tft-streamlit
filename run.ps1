# Launcher for the TFT-H1 Streamlit app.
#
# WHY THIS SCRIPT EXISTS:
# The model checkpoint + dataset_metadata.pkl were created with pandas 1.5.3 +
# pytorch-forecasting 0.10.3 (the training environment, .venv-tft at the repo
# root). Running `streamlit run app.py` with a DIFFERENT Python (e.g. the
# global pyenv install with pandas 2.3.3) fails to unpickle the metadata with:
#   ModuleNotFoundError: No module named 'pandas.core.indexes.numeric'
# This script always uses .venv-tft's python, guaranteeing version parity.
#
# Usage: run this from anywhere, or double-click if .ps1 is associated:
#   .\run.ps1

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $scriptDir "..\.venv-tft\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Error "Tidak ketemu .venv-tft di $venvPython. Pastikan .venv-tft ada di root project (sejajar dengan tft-streamlit/)."
    exit 1
}

Write-Host "Menjalankan Streamlit pakai $venvPython (version-matched dengan checkpoint TFT-H1)..." -ForegroundColor Cyan
Set-Location $scriptDir
& $venvPython -m streamlit run app.py
