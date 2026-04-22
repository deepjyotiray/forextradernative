Set fso = CreateObject("Scripting.FileSystemObject")
Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = fso.GetParentFolderName(WScript.ScriptFullName)
WshShell.Run "cmd /c .venv\Scripts\python.exe auto_trader.py >> trader.log 2>&1", 0, False
WScript.Sleep 4000
