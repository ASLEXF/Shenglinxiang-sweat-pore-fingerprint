# Fetches the parts of the official OpenCV 4.5.4 Windows pack that build.bat needs:
# headers, the vc15 x64 import library and the runtime DLL, into third_party\opencv\build.
# Pass -Installer / -SevenZip to reuse files already on disk instead of downloading.
param(
    [string]$Installer = "",
    [string]$SevenZip = ""
)
$ErrorActionPreference = "Stop"
# Windows PowerShell's progress bar slows Invoke-WebRequest down by an order of magnitude.
$ProgressPreference = "SilentlyContinue"

$root = $PSScriptRoot
$thirdParty = Join-Path $root "third_party"
$cache = Join-Path $thirdParty "_download"
$target = Join-Path $thirdParty "opencv\build"
New-Item -ItemType Directory -Force $cache | Out-Null

if (-not $SevenZip) {
    $SevenZip = Join-Path $cache "7zr.exe"
    if (-not (Test-Path $SevenZip)) {
        Write-Host "downloading 7zr.exe"
        Invoke-WebRequest "https://www.7-zip.org/a/7zr.exe" -OutFile $SevenZip
    }
}
if (-not $Installer) {
    $Installer = Join-Path $cache "opencv-4.5.4-vc14_vc15.exe"
    if (-not (Test-Path $Installer)) {
        Write-Host "downloading OpenCV 4.5.4 Windows pack (about 220 MB)"
        Invoke-WebRequest "https://github.com/opencv/opencv/releases/download/4.5.4/opencv-4.5.4-vc14_vc15.exe" -OutFile $Installer
    }
}

# The pack is a 7z self-extractor; take only what the build uses.
& $SevenZip x $Installer "-o$thirdParty" -y `
    "opencv\build\include" `
    "opencv\build\x64\vc15\lib\opencv_world454.lib" `
    "opencv\build\x64\vc15\bin\opencv_world454.dll" | Out-Null
if ($LASTEXITCODE -ne 0) { throw "7zr failed with exit code $LASTEXITCODE" }

foreach ($f in @("include\opencv2\core.hpp", "x64\vc15\lib\opencv_world454.lib", "x64\vc15\bin\opencv_world454.dll")) {
    if (-not (Test-Path (Join-Path $target $f))) { throw "missing after extraction: $f" }
}
Write-Host "OpenCV ready in $target"
