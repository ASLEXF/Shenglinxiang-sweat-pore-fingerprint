@echo off
rem Builds deployment\palmprint.exe with MSVC against OpenCV 4.5.4 (vc15 x64).
rem OpenCV comes from %OPENCV_DIR% (the "build" folder of the official Windows pack),
rem default third_party\opencv\build. Run fetch_opencv.ps1 once to fill it.
setlocal
set "ROOT=%~dp0"
if not defined OPENCV_DIR set "OPENCV_DIR=%ROOT%third_party\opencv\build"
set "OPENCV_INC=%OPENCV_DIR%\include"
set "OPENCV_LIB=%OPENCV_DIR%\x64\vc15\lib"
set "OPENCV_DLL=%OPENCV_DIR%\x64\vc15\bin\opencv_world454.dll"
set "OUT=%ROOT%deployment"

if not exist "%OPENCV_INC%\opencv2\core.hpp" (
    echo OpenCV headers not found under "%OPENCV_INC%".
    echo Run: powershell -ExecutionPolicy Bypass -File "%ROOT%fetch_opencv.ps1"
    exit /b 1
)
if not exist "%OPENCV_LIB%\opencv_world454.lib" (
    echo opencv_world454.lib not found under "%OPENCV_LIB%".
    exit /b 1
)

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
    echo vswhere.exe not found; install Visual Studio with the C++ workload.
    exit /b 1
)
set "VSDIR="
for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSDIR=%%i"
if not defined VSDIR (
    echo No Visual Studio with the x64 C++ tools found.
    exit /b 1
)
rem vcvars looks for vswhere on PATH and complains harmlessly otherwise.
set "PATH=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer;%PATH%"
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1

if not exist "%OUT%" mkdir "%OUT%"
pushd "%ROOT%SourceCode\C++"
cl /nologo /EHsc /MD /O2 /std:c++17 /utf-8 /I"%OPENCV_INC%" main.cpp palm.cpp ^
    /Fo"%OUT%\\" /Fe"%OUT%\palmprint.exe" ^
    /link /LIBPATH:"%OPENCV_LIB%" opencv_world454.lib
set "ERR=%ERRORLEVEL%"
popd
del /q "%OUT%\*.obj" 2>nul
if not "%ERR%"=="0" exit /b %ERR%

if not exist "%OUT%\opencv_world454.dll" (
    if exist "%OPENCV_DLL%" (
        rem One copy of the DLL. The exe directory gets a symlink; copy only if that is not allowed.
        if /I "%OPENCV_DLL%"=="%ROOT%third_party\opencv\build\x64\vc15\bin\opencv_world454.dll" (
            mklink "%OUT%\opencv_world454.dll" "..\third_party\opencv\build\x64\vc15\bin\opencv_world454.dll" >nul 2>&1
        ) else (
            mklink "%OUT%\opencv_world454.dll" "%OPENCV_DLL%" >nul 2>&1
        )
        if errorlevel 1 copy /y "%OPENCV_DLL%" "%OUT%\" >nul
    ) else (
        echo warning: opencv_world454.dll not found; copy it next to palmprint.exe before running.
    )
)
echo BUILD_OK
