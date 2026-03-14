@echo off
setlocal
cd /d "%~dp0"

echo ========================================
echo   GPU Setup for PyQt Object Tracking
echo ========================================
echo.

set "VENV_PYTHON="
if exist ".venv\Scripts\python.exe" (
    set "VENV_PYTHON=.venv\Scripts\python.exe"
) else if exist "venv\Scripts\python.exe" (
    set "VENV_PYTHON=venv\Scripts\python.exe"
)

if not defined VENV_PYTHON (
    echo [ERROR] No virtual environment found: .venv or venv.
    echo [INFO] Run RUN_PYQT_APP.bat once to create one, then run this script again.
    exit /b 1
)

echo [INFO] Using Python: %VENV_PYTHON%
echo.

where nvidia-smi >nul 2>nul
if errorlevel 1 (
    echo [WARN] nvidia-smi not found. NVIDIA driver may be missing.
    echo [WARN] GPU setup may fail without NVIDIA driver.
    echo.
)

echo [1/5] Upgrading pip tools...
"%VENV_PYTHON%" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo [ERROR] Failed to upgrade pip tools.
    exit /b 1
)

echo.
echo [2/5] Removing CPU-only torch packages...
"%VENV_PYTHON%" -m pip uninstall -y torch torchvision torchaudio >nul 2>nul

echo.
echo [3/5] Installing PyTorch CUDA (cu121)...
"%VENV_PYTHON%" -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1+cu121 torchvision==0.20.1+cu121
if errorlevel 1 (
    echo [ERROR] Failed to install torch CUDA build.
    exit /b 1
)

echo.
echo [4/5] Installing ONNX Runtime GPU...
"%VENV_PYTHON%" -m pip install onnxruntime-gpu==1.23.2
if errorlevel 1 (
    echo [ERROR] Failed to install onnxruntime-gpu.
    exit /b 1
)

echo.
echo [5/5] Verifying GPU runtime...
"%VENV_PYTHON%" -c "import torch; import onnxruntime as ort; print('torch=', torch.__version__); print('cuda_available=', torch.cuda.is_available()); print('device=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'); print('onnx_providers=', ort.get_available_providers())"
if errorlevel 1 (
    echo [ERROR] Verification failed.
    exit /b 1
)

echo.
echo [OK] GPU environment is ready.
echo [INFO] Start app with RUN_PYQT_APP.bat
exit /b 0
