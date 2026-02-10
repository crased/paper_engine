@echo off
REM Paper Engine v1.0 - Quick Start Script (Windows)

echo ============================================================
echo           Paper Engine v1.0 - Pre-Release
echo ============================================================
echo.

REM Check if virtual environment exists
if not exist "env\" (
    echo Virtual environment not found. Creating one...
    python -m venv env
    echo Virtual environment created
    echo.
)

REM Activate virtual environment
echo Activating virtual environment...
call env\Scripts\activate.bat
echo Virtual environment activated
echo.

REM Check if dependencies are installed
python -c "import pynput" 2>nul
if errorlevel 1 (
    echo Dependencies not detected.
    set /p response="Would you like to install them now? (Y/N): "
    if /i "%response%"=="Y" (
        echo Installing dependencies...
        pip install -r requirements.txt
        echo Dependencies installed
        echo.
    ) else (
        echo Skipping dependency installation.
        echo Note: main.py will offer to install missing packages automatically.
        echo.
    )
)

REM Run Paper Engine
echo Starting Paper Engine...
echo.
python main.py

REM Deactivate on exit
call deactivate 2>nul

pause
