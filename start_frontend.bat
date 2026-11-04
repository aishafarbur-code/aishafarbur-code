@echo off
title Smart Weather Station - Frontend (React)
echo ==========================================
echo   Smart Weather Station - Frontend
echo   http://127.0.0.1:3000
echo ==========================================
cd /d "%~dp0frontend"
npx vite preview --host 127.0.0.1 --port 3000
pause
