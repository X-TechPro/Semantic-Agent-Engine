#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

python_bin="${PYTHON:-python}"
if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "Python not found. Install Python 3.11+ first." >&2
  exit 1
fi

if [ ! -x .venv/bin/python ]; then
  "$python_bin" -m venv .venv
fi
venv_python=.venv/bin/python
"$venv_python" -m pip install --upgrade pip

# PyTorch 2.14.1 CUDA 13.2 wheels support NVIDIA Blackwell (compute capability 12.0+).
# Install Torch first so requirements.txt does not replace the CUDA build.
"$venv_python" -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu132
"$venv_python" -m pip install -r requirements.txt
"$venv_python" check_gpu.py

echo
echo "Setup complete. Start the app with:"
echo "  source .venv/bin/activate"
echo "  python app.py"
