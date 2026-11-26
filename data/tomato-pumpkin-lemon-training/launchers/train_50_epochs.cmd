@echo off
setlocal
cd /d "%~dp0.."
set "PYTHON=py"
if exist ".venv\Scripts\python.exe" set "PYTHON=.venv\Scripts\python.exe"
"%PYTHON%" --version >nul 2>&1
if errorlevel 1 (
  echo Python tidak ditemukan. Pasang Python 3.10+ dan pastikan perintah py tersedia.
  pause
  exit /b 1
)
"%PYTHON%" -c "import yaml, ultralytics" >nul 2>&1
if errorlevel 1 (
  echo Dependency belum tersedia. Jalankan: %PYTHON% -m pip install -r requirements-training.txt
  pause
  exit /b 1
)
"%PYTHON%" training\train_tomato_pumpkin_lemon.py --data dataset\data.yaml --epochs 50
pause
