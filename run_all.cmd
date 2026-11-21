@echo off
setlocal
cd /d "%~dp0"
start "FruitVision backend" cmd /k "call run_backend.cmd"
start "FruitVision frontend" cmd /k "call run_frontend.cmd"
echo Backend: http://127.0.0.1:8000/docs
echo Frontend: http://127.0.0.1:5173
