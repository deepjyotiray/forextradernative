Set fso = CreateObject("Scripting.FileSystemObject")
Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = fso.GetParentFolderName(WScript.ScriptFullName)
pythonLauncher = WshShell.ExpandEnvironmentStrings("%LOCALAPPDATA%") & "\Programs\Python\Launcher\py.exe"
If fso.FileExists(pythonLauncher) Then
  WshShell.Run "cmd /c """ & pythonLauncher & """ -3 deploy_watcher.py >> deploy_watcher.log 2>&1", 0, False
Else
  WshShell.Run "cmd /c py -3 deploy_watcher.py >> deploy_watcher.log 2>&1", 0, False
End If
