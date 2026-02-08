@echo off
REM ============================================
REM EDDS AI Engine Training Quick Start
REM ============================================
REM This script runs the complete training workflow
REM 
REM Prerequisites:
REM   - Dataset downloaded to data/ folder
REM   - Virtual environment activated
REM ============================================

echo.
echo ============================================
echo   EDDS AI Engine - Training Quick Start  
echo ============================================
echo.

REM Check if venv exists
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found!
    echo Please run: python -m venv venv
    echo Then: pip install -r requirements.txt
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Check if dataset exists
if not exist "data\140k-real-and-fake-faces.zip" (
    echo [WARNING] Dataset ZIP not found at data\140k-real-and-fake-faces.zip
    echo.
    echo If download is still in progress, wait for it to complete.
    echo If not downloaded, run:
    echo   kaggle datasets download -d xhlulu/140k-real-and-fake-faces -p data/
    echo.
    set /p CONTINUE="Continue anyway? (y/n): "
    if /i not "%CONTINUE%"=="y" exit /b 1
)

echo.
echo [INFO] Starting training workflow...
echo.

REM Run training with default settings
python training\run_training.py --epochs 20 --batch-size 16

echo.
echo ============================================
echo   Training workflow complete!
echo ============================================
echo.

pause
