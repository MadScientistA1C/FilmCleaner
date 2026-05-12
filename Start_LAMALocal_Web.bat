@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "APP_EXE="
set "PYTHON_EXE="

if exist "%~dp0LAMALocalWeb.exe" (
  set "APP_EXE=%~dp0LAMALocalWeb.exe"
) else if exist "%~dp0dist\LAMALocalWeb\LAMALocalWeb.exe" (
  set "APP_EXE=%~dp0dist\LAMALocalWeb\LAMALocalWeb.exe"
) else if exist "%~dp0.venv\Scripts\python.exe" (
  set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
) else (
  set "PYTHON_EXE=python"
)

set "GPU_AVAILABLE=0"
where nvidia-smi >nul 2>nul
if not errorlevel 1 (
  nvidia-smi -L >nul 2>nul
  if not errorlevel 1 set "GPU_AVAILABLE=1"
)

echo.
echo LAMALocal Web Launcher
echo.
if "%GPU_AVAILABLE%"=="1" (
  echo NVIDIA GPU detected.
  echo.
  echo   1. Auto select device - recommended
  echo   2. Force GPU / CUDA
  echo   3. Force CPU
  echo.
  set "DEFAULT_CHOICE=1"
  set /p "DEVICE_CHOICE=Choose compute mode [1]: "
) else (
  echo No NVIDIA GPU was detected by nvidia-smi.
  echo.
  echo   1. CPU - recommended
  echo   2. Auto select device
  echo   3. Force GPU / CUDA
  echo.
  set "DEFAULT_CHOICE=1"
  set /p "DEVICE_CHOICE=Choose compute mode [1]: "
)

if "%DEVICE_CHOICE%"=="" set "DEVICE_CHOICE=%DEFAULT_CHOICE%"

if "%GPU_AVAILABLE%"=="1" (
  if "%DEVICE_CHOICE%"=="2" (
    set "DEVICE=cuda"
  ) else if "%DEVICE_CHOICE%"=="3" (
    set "DEVICE=cpu"
  ) else (
    set "DEVICE=auto"
  )
) else (
  if "%DEVICE_CHOICE%"=="2" (
    set "DEVICE=auto"
  ) else if "%DEVICE_CHOICE%"=="3" (
    set "DEVICE=cuda"
  ) else (
    set "DEVICE=cpu"
  )
)

echo.
echo Starting LAMALocal Web with device: %DEVICE%
echo.

if defined APP_EXE (
  start "" "%APP_EXE%" --open-browser --port 0 --device %DEVICE% %*
  exit /b 0
)

"%PYTHON_EXE%" -m scripts.web_dust_server --open-browser --port 0 --device %DEVICE% %*
exit /b %ERRORLEVEL%
