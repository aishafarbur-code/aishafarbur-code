@echo off
title Smart Weather Station - Backend (FastAPI)
echo ==========================================
echo   Smart Weather Station - Backend Server
echo   http://127.0.0.1:8000
echo ==========================================
cd /d "%~dp0backend"
"%LOCALAPPDATA%\Programs\Python\Python311\python.exe" -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
pause
