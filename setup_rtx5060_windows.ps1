$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$venvPython = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) { python -m venv .venv }
& $venvPython -m pip install --upgrade pip

# PyTorch 2.14.1 CUDA 13.2 build; RTX 5060 is Blackwell-class.
& $venvPython -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu132
& $venvPython -m pip install -r requirements.txt
& $venvPython check_gpu.py
Write-Host "Setup complete. Run: .\.venv\Scripts\python.exe app.py"
