$ErrorActionPreference = 'Stop'
$studioFolder = Split-Path -Parent $PSScriptRoot
if ((Split-Path -Leaf $studioFolder) -eq 'source') { $studioFolder = Split-Path -Parent $studioFolder }
$studioCompiler = 'C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe'
$studioWpf = 'C:\Windows\Microsoft.NET\Framework64\v4.0.30319\WPF'
$studioArguments = @('/nologo', '/target:winexe', '/platform:anycpu', '/optimize+', '/reference:System.Windows.Forms.dll', '/reference:System.Drawing.dll', '/reference:System.Web.Extensions.dll', '/reference:System.Xaml.dll')
foreach ($studioAssembly in @('PresentationFramework.dll','PresentationCore.dll','WindowsBase.dll','WindowsFormsIntegration.dll')) { $studioArguments += ('/reference:' + (Join-Path $studioWpf $studioAssembly)) }
$studioArguments += ('/out:' + (Join-Path $studioFolder 'VideoStudio.exe'))
$studioArguments += ('/win32manifest:' + (Join-Path $PSScriptRoot 'FaceStudioDesktop.manifest'))
foreach ($studioSource in @('FaceStudioDesktop.cs','FaceStudioWindow.cs','NativeCutter.cs','RangeTimeline.cs','ProtectedMoments.cs')) { $studioArguments += (Join-Path $PSScriptRoot $studioSource) }
$studioArguments += (Join-Path $studioFolder 'source\AssignmentWindow.cs')
$studioArguments += ('/resource:' + (Join-Path $PSScriptRoot 'FaceStudioWindow.xaml') + ',FaceStudio.Window.xaml')
$studioArguments += ('/resource:' + (Join-Path $PSScriptRoot 'ready.wav') + ',FaceStudio.Ready.wav')
& $studioCompiler @studioArguments
if ($LASTEXITCODE -ne 0) { throw 'Application compilation failed.' }
