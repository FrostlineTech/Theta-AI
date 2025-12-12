@echo off
echo Human-Like DPO Dataset Downloader for Theta AI
echo ===========================================
echo This script will download the Human-Like DPO Dataset from HuggingFace,
echo process it into a format compatible with Theta AI, and integrate it
echo with your data processing pipeline.
echo.

REM Check if Python is available
python --version > nul 2>&1
if errorlevel 1 (
    echo Python is not found. Please make sure Python is installed and in your PATH.
    exit /b 1
)

REM Set environment variables
set HF_HOME=G:\Theta AI\cache

REM Make sure the datasets library is installed
echo Checking for required libraries...
python -c "import datasets" > nul 2>&1
if errorlevel 1 (
    echo Installing the datasets library...
    pip install datasets
)

echo.
echo Starting download and integration process...
echo.

REM Run the Python script
python download_human_like_dpo_dataset.py

echo.
echo If successful, you can now run your data processor and training scripts:
echo 1. python run_data_processing.py
echo 2. train_overnight_enhanced.bat
echo.
pause
