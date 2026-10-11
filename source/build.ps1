$ErrorActionPreference = 'Stop'
$studioWorkspace = Split-Path -Parent $PSScriptRoot
$studioFolder = $studioWorkspace
$studioCompiler = 'C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe'
$studioWpf = 'C:\Windows\Microsoft.NET\Framework64\v4.0.30319\WPF'
$studioArguments = @('/nologo', '/target:winexe', '/platform:anycpu', '/optimize+', '/reference:System.Windows.Forms.dll', '/reference:System.Drawing.dll', '/reference:System.Web.Extensions.dll', '/reference:System.Xaml.dll')
foreach ($studioAssembly in @('PresentationFramework.dll', 'PresentationCore.dll', 'WindowsBase.dll', 'WindowsFormsIntegration.dll')) { $studioArguments += ('/reference:' + (Join-Path $studioWpf $studioAssembly)) }
$studioArguments += ('/out:' + (Join-Path $studioFolder 'FaceStudio.exe'))
$studioArguments += ('/win32manifest:' + (Join-Path $PSScriptRoot 'FaceStudioDesktop.manifest'))
$studioArguments += (Join-Path $PSScriptRoot 'FaceStudioDesktop.cs')
$studioArguments += (Join-Path $PSScriptRoot 'FaceStudioWindow.cs')
$studioArguments += (Join-Path $PSScriptRoot 'AssignmentWindow.cs')
$studioArguments += ('/resource:' + (Join-Path $PSScriptRoot 'FaceStudioWindow.xaml') + ',FaceStudio.Window.xaml')
$studioArguments += ('/resource:' + (Join-Path $PSScriptRoot 'ready.wav') + ',FaceStudio.Ready.wav')
& $studioCompiler @studioArguments
if ($LASTEXITCODE -ne 0) { throw 'Native application compilation failed.' }
Get-Item -LiteralPath (Join-Path $studioFolder 'FaceStudio.exe') | Select-Object FullName, Length

