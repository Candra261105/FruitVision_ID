@echo off
cd /d "%~dp0.."
call .venv\Scripts\python.exe runs\loop\train_targeted_epochs.py --epochs 70
pause
