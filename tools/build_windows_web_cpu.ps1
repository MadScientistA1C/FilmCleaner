param(
    [switch]$Clean,
    [switch]$Zip,
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"

$toolsRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Split-Path -Parent $toolsRoot
$venvRoot = Join-Path $root ".venv_cpu"
$python = Join-Path $venvRoot "Scripts\python.exe"
$pyInstaller = Join-Path $venvRoot "Scripts\pyinstaller.exe"
$entry = Join-Path $root "scripts\web_dust_server.py"
$distRoot = Join-Path $root "dist"
$buildRoot = Join-Path $root "build\pyinstaller_web_cpu"
$packageRoot = Join-Path $distRoot "LAMALocalWeb-CPU"
$pipTemp = Join-Path $root ".tmp\pip_cpu"

function Assert-PathExists {
    param(
        [string]$Path,
        [string]$Label
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Missing ${Label}: $Path"
    }
}

Assert-PathExists -Path $entry -Label "web app entry"
Assert-PathExists -Path (Join-Path $root "scripts\infer_deeplab_lama.py") -Label "inference script"
Assert-PathExists -Path (Join-Path $root "checkpoints\fakefilmcolor_deeplab_best.pth") -Label "color DeepLab checkpoint"
Assert-PathExists -Path (Join-Path $root "checkpoints\fakefilmbw_deeplab_best.pth") -Label "black-and-white DeepLab checkpoint"
Assert-PathExists -Path (Join-Path $root "model_cache\hub\checkpoints\big-lama.pt") -Label "LaMa checkpoint"

if ($Clean) {
    if (Test-Path -LiteralPath $packageRoot) {
        Remove-Item -LiteralPath $packageRoot -Recurse -Force
    }
    if (Test-Path -LiteralPath $buildRoot) {
        Remove-Item -LiteralPath $buildRoot -Recurse -Force
    }
    $zipPath = Join-Path $distRoot "LAMALocalWeb-CPU.zip"
    if (Test-Path -LiteralPath $zipPath) {
        Remove-Item -LiteralPath $zipPath -Force
    }
}

if (-not (Test-Path -LiteralPath $python)) {
    py -3.10 -m venv $venvRoot
    if (-not (Test-Path -LiteralPath $python)) {
        python -m venv $venvRoot
    }
}

if (-not $SkipInstall) {
    New-Item -ItemType Directory -Force $pipTemp | Out-Null
    $env:TEMP = $pipTemp
    $env:TMP = $pipTemp
    $env:TMPDIR = $pipTemp
    $env:PYTHONUTF8 = "1"
    $env:PYTHONIOENCODING = "utf-8"
    $env:PIP_DISABLE_PIP_VERSION_CHECK = "1"
    $env:PIP_CACHE_DIR = $pipTemp
    & $python -m pip install --no-cache-dir --upgrade pip
    & $python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision
    & $python -m pip install --no-cache-dir wheel setuptools
    & $python -m pip install --no-cache-dir antlr4-python3-runtime==4.9.3
    & $python -m pip install --no-cache-dir `
        pyinstaller `
        opencv-python `
        segmentation-models-pytorch `
        Pillow==9.5.0 `
        numpy `
        tqdm `
        iopaint==1.6.0 `
        gradio==4.21.0 `
        diffusers==0.27.2 `
        transformers==4.48.3 `
        huggingface-hub==0.25.2
}

Assert-PathExists -Path $pyInstaller -Label "PyInstaller executable"

New-Item -ItemType Directory -Force $distRoot, $buildRoot | Out-Null

& $pyInstaller `
    --noconfirm `
    --clean `
    --onedir `
    --console `
    --name LAMALocalWebCPU `
    --distpath $distRoot `
    --workpath $buildRoot `
    --specpath $buildRoot `
    --paths $root `
    --paths (Join-Path $root "scripts") `
    --add-data "$root\scripts\infer_deeplab_lama.py;scripts" `
    --add-data "$root\checkpoints\fakefilmcolor_deeplab_best.pth;checkpoints" `
    --add-data "$root\checkpoints\fakefilmbw_deeplab_best.pth;checkpoints" `
    --add-data "$root\model_cache\hub\checkpoints\big-lama.pt;model_cache\hub\checkpoints" `
    --collect-all iopaint `
    --collect-all segmentation_models_pytorch `
    --collect-all timm `
    --hidden-import cv2 `
    --hidden-import torch `
    --hidden-import torchvision `
    --hidden-import infer_deeplab_lama `
    $entry

if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed with exit code $LASTEXITCODE"
}

$sourceCollect = Join-Path $distRoot "LAMALocalWebCPU"
if (Test-Path -LiteralPath $packageRoot) {
    Remove-Item -LiteralPath $packageRoot -Recurse -Force
}
Move-Item -LiteralPath $sourceCollect -Destination $packageRoot

$launcherPath = Join-Path $packageRoot "Start LAMALocal Web CPU.bat"
$launcher = @'
@echo off
setlocal
cd /d "%~dp0"
start "" "%~dp0LAMALocalWebCPU.exe" --open-browser --port 0 --device cpu %*
exit /b 0
'@
Set-Content -LiteralPath $launcherPath -Value $launcher -Encoding ASCII

$readmePath = Join-Path $packageRoot "README.txt"
$readme = @'
LAMALocal Web CPU-only package for Windows

Double-click "Start LAMALocal Web CPU.bat" to start the local web service and open the browser.
This package always runs with --device cpu and does not include NVIDIA CUDA runtime files.
The browser page runs locally on 127.0.0.1 and stores temporary output under %LOCALAPPDATA%\LAMALocal\web_outputs.

If Windows SmartScreen appears, choose More info, then Run anyway.
'@
Set-Content -LiteralPath $readmePath -Value $readme -Encoding ASCII

if ($Zip) {
    $zipPath = Join-Path $distRoot "LAMALocalWeb-CPU.zip"
    if (Test-Path -LiteralPath $zipPath) {
        Remove-Item -LiteralPath $zipPath -Force
    }
    Compress-Archive -Path (Join-Path $packageRoot "*") -DestinationPath $zipPath -Force
    Write-Host "Created CPU zip package:"
    Write-Host $zipPath
}

Write-Host "Built CPU-only Windows web app:"
Write-Host (Join-Path $packageRoot "LAMALocalWebCPU.exe")
Write-Host "One-click CPU launcher:"
Write-Host $launcherPath
