@echo off
echo Shutting down Auto Trader...
curl -s -X POST http://127.0.0.1:8899/shutdown >nul 2>&1
echo Done.
timeout /t 2 >nul
