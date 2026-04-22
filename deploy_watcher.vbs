Set fso = CreateObject("Scripting.FileSystemObject")
Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = fso.GetParentFolderName(WScript.ScriptFullName)
WshShell.Run "cmd /c .venv\Scripts\python.exe deploy_watcher.py >> deploy_watcher.log 2>&1", 0, False
