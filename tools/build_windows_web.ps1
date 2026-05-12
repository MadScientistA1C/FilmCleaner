param(
    [switch]$Clean,
    [switch]$Zip
)

$ErrorActionPreference = "Stop"

$toolsRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Split-Path -Parent $toolsRoot
$pyInstaller = Join-Path $root ".venv\Scripts\pyinstaller.exe"
$entry = Join-Path $root "scripts\web_dust_server.py"
$launcherSource = Join-Path $root "Start_LAMALocal_Web.bat"
$distRoot = Join-Path $root "dist"
$buildRoot = Join-Path $root "build\pyinstaller_web"
$packageRoot = Join-Path $distRoot "LAMALocalWeb"

function Assert-PathExists {
    param(
        [string]$Path,
        [string]$Label
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Missing ${Label}: $Path"
    }
}

if (-not (Test-Path -LiteralPath $pyInstaller)) {
    throw "Missing PyInstaller executable: $pyInstaller`nInstall it with: .\.venv\Scripts\python.exe -m pip install pyinstaller"
}
Assert-PathExists -Path $entry -Label "web app entry"
Assert-PathExists -Path $launcherSource -Label "Windows launcher"
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
    $zipPath = Join-Path $distRoot "LAMALocalWeb.zip"
    if (Test-Path -LiteralPath $zipPath) {
        Remove-Item -LiteralPath $zipPath -Force
    }
}

New-Item -ItemType Directory -Force $distRoot, $buildRoot | Out-Null

& $pyInstaller `
    --noconfirm `
    --clean `
    --onedir `
    --console `
    --name LAMALocalWeb `
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

$launcherPath = Join-Path $packageRoot "Start LAMALocal Web.bat"
Copy-Item -LiteralPath $launcherSource -Destination $launcherPath -Force

$readmePath = Join-Path $packageRoot "README.txt"
$readme = @'
LAMALocal Web for Windows

Double-click "Start LAMALocal Web.bat" to choose Auto, GPU/CUDA, or CPU mode, then start the local web service and open the browser.
The browser page runs locally on 127.0.0.1 and stores temporary output under %LOCALAPPDATA%\LAMALocal\web_outputs.

GPU mode requires an NVIDIA GPU, working NVIDIA driver, and a CUDA-capable PyTorch build inside the packaged app.
If Windows SmartScreen appears, choose More info, then Run anyway.
'@
Set-Content -LiteralPath $readmePath -Value $readme -Encoding ASCII

if ($Zip) {
    $zipPath = Join-Path $distRoot "LAMALocalWeb.zip"
    if (Test-Path -LiteralPath $zipPath) {
        Remove-Item -LiteralPath $zipPath -Force
    }
    Compress-Archive -Path (Join-Path $packageRoot "*") -DestinationPath $zipPath -Force
    Write-Host "Created zip package:"
    Write-Host $zipPath
}

Write-Host "Built Windows web app:"
Write-Host (Join-Path $packageRoot "LAMALocalWeb.exe")
Write-Host "One-click launcher:"
Write-Host $launcherPath
