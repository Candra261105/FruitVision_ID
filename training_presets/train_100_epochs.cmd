@echo off
cd /d "%~dp0.."
call .venv\Scripts\python.exe runs\loop\train_tomato_pumpkin_lemon.py --data "%~dp0..\data\processed\tomato-pumpkin-lemon\data.yaml" --epochs 100
pause
