@echo off
title Trading System - Unified Server
echo.
echo ========================================
echo    Trading System - Unified Startup
echo ========================================
echo.
echo Starting unified trading system...
echo - Auto Trader Engine (background)
echo - FastAPI Server (port 8000)
echo - All endpoints consolidated
echo.
echo Dashboard: http://localhost:8000/dashboard
echo Status API: http://localhost:8000/status
echo Analytics: http://localhost:8000/analytics
echo.
echo Press Ctrl+C to stop the system
echo ========================================
echo.

cd /d "%~dp0"
python unified_startup.py

echo.
echo Trading system stopped.
pause