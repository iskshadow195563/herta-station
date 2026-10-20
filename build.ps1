# Build the one-file HertaStation.exe
#
#   .\build.ps1
#
# Output: dist\HertaStation.exe
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
& $py -m pip install --upgrade pip *> $null
& $py -m pip install -r requirements.txt

& (Join-Path $PSScriptRoot ".venv\Scripts\pyinstaller.exe") `
    --noconfirm --onefile --windowed --name HertaStation `
    --icon "assets\herta.ico" `
    --add-data "ui.html;." `
    --collect-all webview --collect-all clr_loader `
    --paths $PSScriptRoot `
    --distpath "dist" --workpath "build\work" --specpath "build" `
    "herta_app.py"

Write-Host ""
Write-Host "built: dist\HertaStation.exe"
