Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$venvPath = Join-Path $PSScriptRoot ".venv-gpu-build"
$pythonPath = Join-Path $venvPath "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonPath)) {
    python -m venv $venvPath
}
& $pythonPath -m pip install --upgrade pip
& $pythonPath -m pip install -r requirements-gpu.txt
& $pythonPath -m PyInstaller --noconfirm --clean LaoCaoMirrorGPU.spec
