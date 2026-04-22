@echo off
cd /d "%~dp0"

:: Kill existing instance
curl -s -X POST http://127.0.0.1:8899/shutdown >nul 2>&1
timeout /t 2 /nobreak >nul
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8899 ^| findstr LISTENING 2^>nul') do taskkill /PID %%a /F >nul 2>&1
timeout /t 1 /nobreak >nul

:: Start fresh
wscript "%~dp0start_trader.vbs"
echo Auto Trader started. Dashboard opening in browser.
