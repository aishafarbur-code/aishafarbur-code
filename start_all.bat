
@echo off
title Smart Weather Station - Bilow Labada Server
echo ==============================================
echo   SMART WEATHER STATION & MACHINE LEARNING
echo   Bilow labada server hal mar
echo ==============================================
echo.
echo [1/2] Bilaabaya Backend Server (Port 8000)...
start "Backend - FastAPI ML Server" /D "%~dp0backend" "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload

timeout /t 3 /nobreak >nul

echo [2/2] Bilaabaya Frontend Server (Port 3000)...
start "Frontend - React Dashboard" /D "%~dp0frontend" cmd /c "npx vite preview --host 127.0.0.1 --port 3000"

timeout /t 3 /nobreak >nul

echo.
echo ==============================================
echo   SERVERS BILOWDAY GUUL LAHA!
echo.
echo   Backend API:   http://127.0.0.1:8000
echo   Frontend:      http://127.0.0.1:3000
echo   API Docs:      http://127.0.0.1:8000/docs
echo ==============================================
echo.
echo Fur browser-ka oo gal: http://127.0.0.1:3000
echo.
start "" "http://127.0.0.1:3000"
pause
