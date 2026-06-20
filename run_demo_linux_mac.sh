#!/usr/bin/env bash
# Demo ALPR + Super-Resolution bien so xe Viet Nam
set -e
cd "$(dirname "$0")"

echo "============================================================"
echo "  DEMO ALPR + Super-Resolution bien so xe Viet Nam"
echo "============================================================"

PY="$(command -v python3.11 || command -v python3.10 || command -v python3 || command -v python || true)"
if [ -z "$PY" ]; then
  echo "[LOI] Khong tim thay Python. Hay cai Python 3.10 hoac 3.11 roi chay lai."
  exit 1
fi
echo "[i] Dung Python: $PY"

if [ ! -x ".venv/bin/python" ]; then
  echo "[1/3] Tao moi truong ao .venv ..."
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

if [ ! -f ".venv/.installed" ]; then
  echo "[2/3] Cai thu vien lan dau (can Internet, mat vai phut)..."
  python -m pip install --upgrade pip
  echo "    - Cai PyTorch (ban CPU)..."
  pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu
  echo "    - Cai cac thu vien con lai..."
  pip install -r requirements.txt
  touch .venv/.installed
else
  echo "[2/3] Thu vien da cai san - bo qua."
fi

export GIT_PYTHON_REFRESH=quiet
echo "[3/3] Khoi dong demo web tai http://127.0.0.1:7860 (Ctrl+C de dung)"
python src/demo_web.py
