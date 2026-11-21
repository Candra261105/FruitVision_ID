@echo off
setlocal
cd /d "%~dp0frontend"
if not exist "node_modules" (
  echo node_modules belum ada. Jalankan: cmd /c npm install
  exit /b 1
)
cmd /c npm run dev -- --host 127.0.0.1 --port 5173
