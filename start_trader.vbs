Set fso = CreateObject("Scripting.FileSystemObject")
Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = fso.GetParentFolderName(WScript.ScriptFullName)
WshShell.Run "cmd /c .venv\Scripts\python.exe auto_trader.py >> trader.log 2>&1", 0, False
WScript.Sleep 4000
WshShell.Run "http://127.0.0.1:8899", 1, False
