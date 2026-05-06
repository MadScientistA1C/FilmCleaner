param(
    [switch]$Zip,
    [switch]$BuildExe
)

$ErrorActionPreference = "Stop"

$toolsRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Split-Path -Parent $toolsRoot
$sourcePlugin = Join-Path $root "LAMALocalDust.lrplugin"
$distRoot = Join-Path $root "dist"
$package = Join-Path $distRoot "LAMALocalDust.lrplugin"
$zipPath = Join-Path $distRoot "LAMALocalDust.lrplugin.zip"
$buildRoot = Join-Path $root "build\pyinstaller"
$exeDist = Join-Path $package "bin"
$pyInstaller = Join-Path $root ".venv\Scripts\pyinstaller.exe"

function Assert-PathExists {
    param(
        [string]$Path,
        [string]$Label
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Missing $Label: $Path"
    }
}

function Build-PluginPackage {
    Assert-PathExists -Path $sourcePlugin -Label "plugin source directory"
    Assert-PathExists -Path (Join-Path $root "scripts\infer_deeplab_lama.py") -Label "inference script"
    Assert-PathExists -Path (Join-Path $root "requirements.txt") -Label "requirements file"
    Assert-PathExists -Path (Join-Path $root "checkpoints\best_model.pth") -Label "DeepLab checkpoint"
    Assert-PathExists -Path (Join-Path $root "model_cache\hub\checkpoints\big-lama.pt") -Label "LaMa checkpoint"

    if (Test-Path -LiteralPath $package) {
        Remove-Item -LiteralPath $package -Recurse -Force
    }

    New-Item -ItemType Directory -Force $distRoot | Out-Null
    Copy-Item -LiteralPath $sourcePlugin -Destination $package -Recurse
    Copy-Item -LiteralPath (Join-Path $root "scripts\infer_deeplab_lama.py") -Destination (Join-Path $package "infer_deeplab_lama.py")
    Copy-Item -LiteralPath (Join-Path $root "requirements.txt") -Destination (Join-Path $package "requirements.txt")

    $deeplabDir = Join-Path $package "models\deeplab"
    $lamaDir = Join-Path $package "models\lama"
    New-Item -ItemType Directory -Force $deeplabDir, $lamaDir | Out-Null
    Copy-Item -LiteralPath (Join-Path $root "checkpoints\best_model.pth") -Destination (Join-Path $deeplabDir "best_model.pth")
    Copy-Item -LiteralPath (Join-Path $root "model_cache\hub\checkpoints\big-lama.pt") -Destination (Join-Path $lamaDir "big-lama.pt")

    Get-ChildItem -Path $package -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force

    Write-Host "Built Lightroom plugin package:"
    Write-Host $package
}

function Build-PluginExe {
    Assert-PathExists -Path $package -Label "plugin package"
    Assert-PathExists -Path (Join-Path $package "lr_dust_cli.py") -Label "plugin entry file"
    Assert-PathExists -Path $pyInstaller -Label "PyInstaller executable"

    New-Item -ItemType Directory -Force $buildRoot, $exeDist | Out-Null

    & $pyInstaller `
        --noconfirm `
        --clean `
        --onedir `
        --console `
        --name LAMALocalDust `
        --paths $package `
        --distpath $exeDist `
        --workpath $buildRoot `
        --specpath $buildRoot `
        --collect-all iopaint `
        --collect-all segmentation_models_pytorch `
        --collect-all timm `
        --hidden-import auto_dust `
        --hidden-import manual_dust_server `
        --hidden-import infer_deeplab_lama `
        (Join-Path $package "lr_dust_cli.py")

    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed with exit code $LASTEXITCODE"
    }

    Write-Host "Built plugin executable:"
    Write-Host (Join-Path $exeDist "LAMALocalDust\LAMALocalDust.exe")
}

function Build-PluginZip {
    Assert-PathExists -Path $package -Label "plugin package"

    if (Test-Path -LiteralPath $zipPath) {
        Remove-Item -LiteralPath $zipPath -Force
    }

    Compress-Archive -Path $package -DestinationPath $zipPath
    Write-Host "Built plugin archive:"
    Write-Host $zipPath
}

Build-PluginPackage

if ($BuildExe) {
    Build-PluginExe
}

if ($Zip) {
    Build-PluginZip
}
