$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$plugin = Join-Path $root "dist\LAMALocalDust.lrplugin"
$entry = Join-Path $plugin "lr_dust_cli.py"
$buildRoot = Join-Path $root "build\pyinstaller"
$exeDist = Join-Path $plugin "bin"

if (-not (Test-Path $entry)) {
    throw "Missing entry file: $entry"
}

New-Item -ItemType Directory -Force $buildRoot, $exeDist | Out-Null

& (Join-Path $root ".venv\Scripts\python.exe") -m PyInstaller `
    --noconfirm `
    --clean `
    --onedir `
    --console `
    --name LAMALocalDust `
    --paths $plugin `
    --distpath $exeDist `
    --workpath $buildRoot `
    --specpath $buildRoot `
    --collect-all iopaint `
    --collect-all segmentation_models_pytorch `
    --collect-all timm `
    --hidden-import auto_dust `
    --hidden-import manual_dust_server `
    --hidden-import infer_deeplab_lama `
    $entry

Write-Host "Built:"
Write-Host (Join-Path $exeDist "LAMALocalDust\LAMALocalDust.exe")
