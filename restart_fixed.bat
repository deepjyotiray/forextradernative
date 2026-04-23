@echo off
echo Stopping existing trader...
taskkill /f /im python.exe >nul 2>&1
taskkill /f /im py.exe >nul 2>&1
timeout /t 3 /nobreak >nul

echo Starting trader with fixes...
start /min py auto_trader.py
timeout /t 2 /nobreak >nul

echo Trader restarted! Dashboard: http://127.0.0.1:8899
echo.
echo The database errors should now be fixed.
echo Check the trader.log file for any remaining issues.
pause