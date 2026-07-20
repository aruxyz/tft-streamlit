# Launcher for the TFT-H1 Streamlit app.
#
# .venv-tft (repo root, sibling of tft-streamlit/) is kept in sync with
# requirements.txt (pytorch-forecasting>=1.8.0, torch>=2.1.0, pandas>=2.1.0).
# This script always uses .venv-tft's python for a consistent environment.
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

Write-Host "Menjalankan Streamlit pakai $venvPython..." -ForegroundColor Cyan
Set-Location $scriptDir
& $venvPython -m streamlit run app.py
