@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
title Demo ALPR - Super Resolution bien so xe

echo ============================================================
echo   DEMO ALPR + Super-Resolution bien so xe Viet Nam
echo ============================================================

rem --- Tim Python (uu tien 3.11 / 3.10) ---
set "PY="
py -3.11 --version >nul 2>nul && set "PY=py -3.11"
if not defined PY ( py -3.10 --version >nul 2>nul && set "PY=py -3.10" )
if not defined PY ( python --version >nul 2>nul && set "PY=python" )
if not defined PY (
  echo [LOI] Khong tim thay Python. Hay cai Python 3.10 hoac 3.11 tu python.org
  echo       roi chay lai file nay.
  pause & exit /b 1
)
echo [i] Dung Python: %PY%

rem --- Tao moi truong ao ---
if not exist ".venv\Scripts\python.exe" (
  echo [1/3] Tao moi truong ao .venv ...
  %PY% -m venv .venv || ( echo [LOI] Khong tao duoc venv & pause & exit /b 1 )
)
call ".venv\Scripts\activate.bat"

rem --- Cai thu vien (chi lan dau) ---
if not exist ".venv\.installed" (
  echo [2/3] Cai thu vien lan dau ^(can Internet, mat vai phut^)...
  python -m pip install --upgrade pip
  echo     - Cai PyTorch ^(ban CPU^)...
  pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu || ( echo [LOI] Cai torch that bai & pause & exit /b 1 )
  echo     - Cai cac thu vien con lai...
  pip install -r requirements.txt || ( echo [LOI] Cai requirements that bai & pause & exit /b 1 )
  echo done> ".venv\.installed"
) else (
  echo [2/3] Thu vien da cai san - bo qua.
)

set GIT_PYTHON_REFRESH=quiet
echo [3/3] Khoi dong demo web... Trinh duyet se tu mo http://127.0.0.1:7860
echo     ^(Neu khong tu mo, hay mo trinh duyet va vao dia chi tren^)
echo     Nhan Ctrl+C trong cua so nay de dung demo.
python src\demo_web.py

pause
