@echo off
echo Starting deploy watcher (background)...
wscript "%~dp0deploy_watcher.vbs"
echo Deploy watcher running. Check deploy_watcher.log for output.
