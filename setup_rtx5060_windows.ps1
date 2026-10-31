$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv")) { python -m venv .venv }
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# PyTorch 2.14.1 CUDA 13.2 build; RTX 5060 is Blackwell-class.
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu132
pip install -r requirements.txt
python check_gpu.py
Write-Host "Setup complete. Run: python app.py"
