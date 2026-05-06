param(
    [switch]$Zip
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$sourcePlugin = Join-Path $root "LAMALocalDust.lrplugin"
$distRoot = Join-Path $root "dist"
$package = Join-Path $distRoot "LAMALocalDust.lrplugin"

if (Test-Path $package) {
    Remove-Item -Recurse -Force $package
}

New-Item -ItemType Directory -Force $distRoot | Out-Null
Copy-Item -Recurse $sourcePlugin $package

Copy-Item (Join-Path $root "infer_deeplab_lama.py") (Join-Path $package "infer_deeplab_lama.py")
Copy-Item (Join-Path $root "requirements.txt") (Join-Path $package "requirements.txt")

$deeplabDir = Join-Path $package "models\deeplab"
$lamaDir = Join-Path $package "models\lama"
New-Item -ItemType Directory -Force $deeplabDir, $lamaDir | Out-Null
Copy-Item (Join-Path $root "checkpoints\best_model.pth") (Join-Path $deeplabDir "best_model.pth")
Copy-Item (Join-Path $root "model_cache\hub\checkpoints\big-lama.pt") (Join-Path $lamaDir "big-lama.pt")

Get-ChildItem -Recurse -Directory $package -Filter "__pycache__" | Remove-Item -Recurse -Force

if ($Zip) {
    $zip = Join-Path $distRoot "LAMALocalDust.lrplugin.zip"
    if (Test-Path $zip) {
        Remove-Item -Force $zip
    }
    Compress-Archive -Path $package -DestinationPath $zip
}

Write-Host "Built Lightroom plugin package:"
Write-Host $package
if ($Zip) {
    Write-Host $zip
}
