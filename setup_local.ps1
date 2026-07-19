# Setup lokal untuk test_model.ipynb
# Buat venv khusus tft-streamlit (numpy<2.0 + pytorch-forecasting==0.10.3)
# supaya tidak bentrok dengan env global (numpy 2.4.6 + pytorch-forecasting 1.7.0).
#
# Cara pakai:
#   cd ke folder tft-streamlit
#   .\setup_local.ps1
#
# Setelah selesai, jalanin notebook:
#   .\venv\Scripts\Activate.ps1
#   jupyter lab test_model.ipynb
# Lalu pilih kernel "Python (tft-streamlit)" di Jupyter.

$ErrorActionPreference = "Stop"

Write-Host "[1/6] Membuat venv di ./venv ..." -ForegroundColor Cyan
if (Test-Path "venv") {
    Write-Host "venv sudah ada. Hapus dulu kalau mau setup ulang." -ForegroundColor Yellow
} else {
    python -m venv venv
}

Write-Host "[2/6] Activate venv + upgrade pip ..." -ForegroundColor Cyan
& ".\venv\Scripts\Activate.ps1"
python -m pip install --upgrade pip

Write-Host "[3/6] Install dependencies dari requirements.txt ..." -ForegroundColor Cyan
pip install -r requirements.txt

Write-Host "[4/6] Install jupyterlab + ipykernel ..." -ForegroundColor Cyan
pip install jupyterlab ipykernel

Write-Host "[5/6] Register kernel 'Python (tft-streamlit)' ..." -ForegroundColor Cyan
python -m ipykernel install --user --name=tft-streamlit --display-name="Python (tft-streamlit)"

Write-Host "[6/6] Verifikasi ..." -ForegroundColor Cyan
python -c "import numpy, torch, pytorch_forecasting, lightning, pandas, matplotlib; print(f'numpy={numpy.__version__} | torch={torch.__version__} | pytorch_forecasting={pytorch_forecasting.__version__} | lightning={lightning.__version__} | pandas={pandas.__version__}')"

Write-Host ""
Write-Host "Setup selesai. Untuk jalanin notebook:" -ForegroundColor Green
Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor Green
Write-Host "  jupyter lab test_model.ipynb" -ForegroundColor Green
Write-Host "Pilih kernel 'Python (tft-streamlit)' di Jupyter." -ForegroundColor Green
