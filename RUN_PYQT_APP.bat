@echo off
REM Batch script to run PyQt5 Desktop App on Windows

echo ========================================
echo   Real-time Object Tracking - PyQt5
echo ========================================
echo.

set "VENV_DIR="

REM Reuse the existing workspace environment when available
if exist ".venv\Scripts\python.exe" (
    set "VENV_DIR=.venv"
) else if exist "venv\Scripts\python.exe" (
    set "VENV_DIR=venv"
)

REM Create a default environment only when neither exists
if not defined VENV_DIR (
    set "VENV_DIR=.venv"
    echo [INFO] Virtual environment not found. Creating %VENV_DIR%...
    python -m venv %VENV_DIR%
    echo [OK] Virtual environment created.
    echo.
)

set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"

REM Check if PyQt5 is installed
echo [INFO] Using Python interpreter: %VENV_PYTHON%
%VENV_PYTHON% -c "import PyQt5" 2>nul
if errorlevel 1 (
    echo [INFO] PyQt5 not found. Installing dependencies...
    %VENV_PYTHON% -m pip install -r requirements_pyqt.txt
    echo [OK] Dependencies installed.
    echo.
)

REM Run the application
echo [INFO] Starting PyQt5 application...
echo.
%VENV_PYTHON% pyqt_app.py

pause
