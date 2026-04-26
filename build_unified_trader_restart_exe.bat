@echo off
setlocal
cd /d "%~dp0"

echo Installing/Updating PyInstaller...
py -m pip install --upgrade pyinstaller
if errorlevel 1 (
  echo Failed to install PyInstaller.
  exit /b 1
)

echo Building UnifiedTraderRestart.exe ...
py -m PyInstaller --noconfirm --clean --onefile --windowed --icon UnifiedTraderRestart.ico --name UnifiedTraderRestart unified_trader_restart_launcher.py
if errorlevel 1 (
  echo Build failed.
  exit /b 1
)

copy /Y "dist\UnifiedTraderRestart.exe" "UnifiedTraderRestart.exe" >nul

echo.
echo Build complete:
echo   UnifiedTraderRestart.exe
echo.
echo You can now pin UnifiedTraderRestart.exe to taskbar.
endlocal
