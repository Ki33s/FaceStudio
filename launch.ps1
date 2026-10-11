param([switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
$studioExecutable = Join-Path $PSScriptRoot 'FaceStudio.exe'
if (-not (Test-Path -LiteralPath $studioExecutable)) { throw 'FaceStudio.exe is missing.' }
Start-Process -FilePath $studioExecutable -WorkingDirectory $PSScriptRoot
