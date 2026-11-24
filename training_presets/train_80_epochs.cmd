@echo off
cd /d "%~dp0.."
call .venv\Scripts\python.exe runs\loop\train_tomato_pumpkin_lemon.py --epochs 80
pause
