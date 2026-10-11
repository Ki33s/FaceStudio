@echo off
cd /d "%~dp0"
powershell.exe -NoProfile -Command "$studioExe = Join-Path (Get-Location) 'FaceStudio.exe'; Get-Process -Name FaceStudio -ErrorAction SilentlyContinue | Where-Object { $_.Path -eq $studioExe } | ForEach-Object { [void]$_.CloseMainWindow() }"
