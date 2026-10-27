# Build the one-file HertaStation.exe
#
#   .\build.ps1
#   .\build.ps1 -SkipInstall        # reuse the existing virtualenv
#
# Output: dist\HertaStation.exe
param(
    [string]$Python = "",
    [switch]$SkipInstall
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not $Python) {
    if (-not (Test-Path ".venv")) {
        python -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw "Virtual environment creation failed" }
    }
    $Python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
}
if (-not $SkipInstall) {
    & $Python -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed" }
}

# Prefer an explicitly supplied icon, then the packaged one, then none at all.
$icon = ""
foreach ($candidate in @("$PSScriptRoot\assets\herta.ico", "$PSScriptRoot\herta.ico")) {
    if (Test-Path $candidate) { $icon = $candidate; break }
}
$iconArgs = @()
if ($icon) { $iconArgs = @("--icon", $icon) }

& $Python -m PyInstaller `
    --noconfirm --onefile --windowed --name HertaStation `
    @iconArgs `
    --add-data "$PSScriptRoot\ui.html;." `
    --collect-all webview --collect-all clr_loader `
    --paths $PSScriptRoot `
    --distpath "$PSScriptRoot\dist" --workpath "$PSScriptRoot\build\work" --specpath "$PSScriptRoot\build" `
    "$PSScriptRoot\herta_app.py"
if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed" }

Write-Host ""
Write-Host "built: dist\HertaStation.exe"
