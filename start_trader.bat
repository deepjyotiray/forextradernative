@echo off
cd /d "%~dp0"

:: Kill existing instances (both old port 8899 and new port 8000)
curl -s -X POST http://127.0.0.1:8899/shutdown >nul 2>&1
curl -s -X POST http://127.0.0.1:8000/shutdown >nul 2>&1
timeout /t 2 /nobreak >nul

:: Kill processes on both ports
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8899 ^| findstr LISTENING 2^>nul') do taskkill /PID %%a /F >nul 2>&1
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000 ^| findstr LISTENING 2^>nul') do taskkill /PID %%a /F >nul 2>&1
timeout /t 1 /nobreak >nul

:: Start unified trading system
echo Starting Unified Trading System...
echo - Auto Trader Engine + FastAPI Server
echo - All endpoints on port 8000
echo.
start "Trading System" cmd /k "title Trading System && python unified_startup.py"
timeout /t 3 /nobreak >nul
echo.
echo Trading System started!
echo Dashboard: http://127.0.0.1:8000/dashboard
echo Status API: http://127.0.0.1:8000/status
echo Analytics: http://127.0.0.1:8000/analytics
echo.

:: Start cloudflared tunnel (update port to 8000)
taskkill /IM cloudflared.exe /F >nul 2>&1
echo Starting Cloudflare tunnel for port 8000...
start "Cloudflare Tunnel" cmd /c "cloudflared tunnel --url http://localhost:8000 > cf_log.txt 2>&1"
timeout /t 3 /nobreak >nul
echo Cloudflare tunnel started in background.
echo.