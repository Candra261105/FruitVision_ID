@echo off
setlocal
cd /d "%~dp0"

set "PYTHON=py"
if defined FRUITVISION_PYTHON set "PYTHON=%FRUITVISION_PYTHON%"
if not defined FRUITVISION_PYTHON if exist ".venv\Scripts\python.exe" set "PYTHON=.venv\Scripts\python.exe"

echo Menjalankan training adaptive. Hardware dan konfigurasi dipilih otomatis.
"%PYTHON%" "training\train_auto.py" %*
set "EXIT_CODE=%ERRORLEVEL%"
echo.
if not "%EXIT_CODE%"=="0" echo Training berhenti dengan exit code %EXIT_CODE%.
pause
exit /b %EXIT_CODE%
