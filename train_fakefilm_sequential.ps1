$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$Python = Join-Path $Root ".venv\Scripts\python.exe"
$TrainScript = Join-Path $Root "train_deeplab_dataset.py"
$LogDir = Join-Path $Root "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Invoke-Training {
    param(
        [string]$Name,
        [string]$ImageDir,
        [string]$MaskDir,
        [string]$OutputPrefix
    )

    $LogPath = Join-Path $LogDir "$Name.log"
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Starting $Name" | Set-Content -Path $LogPath

    $Command = @(
        "`"$Python`"",
        "`"$TrainScript`"",
        "--image-dir", "`"$ImageDir`"",
        "--mask-dir", "`"$MaskDir`"",
        "--output-prefix", "`"$OutputPrefix`"",
        "--epochs", "40",
        "--batch-size", "8",
        "--image-size", "512",
        "--num-workers", "4"
    ) -join " "

    & cmd.exe /c "$Command >> `"$LogPath`" 2>&1"

    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }

    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Finished $Name" | Add-Content -Path $LogPath
}

Invoke-Training `
    -Name "fakefilmbw" `
    -ImageDir (Join-Path $Root "Dataset\FakeFilmBW\patches\images") `
    -MaskDir (Join-Path $Root "Dataset\FakeFilmBW\patches\masks") `
    -OutputPrefix (Join-Path $Root "checkpoints\fakefilmbw_deeplab")

Invoke-Training `
    -Name "fakefilmcolor" `
    -ImageDir (Join-Path $Root "Dataset\FakeFilmColor\patches\images") `
    -MaskDir (Join-Path $Root "Dataset\FakeFilmColor\patches\masks") `
    -OutputPrefix (Join-Path $Root "checkpoints\fakefilmcolor_deeplab")
