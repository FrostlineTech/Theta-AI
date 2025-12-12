@echo off
REM Change to the script directory
cd /d "G:\Theta AI"

echo ================================================
echo NVIDIA OpenMathInstruct-1 Dataset Downloader
echo ================================================
echo.
echo This script will download the NVIDIA OpenMathInstruct-1 dataset
echo from HuggingFace to G:\Theta AI\Datasets
echo.
echo Dataset URL: https://huggingface.co/datasets/nvidia/OpenMathInstruct-1
echo.

REM Set environment variables
set HF_HOME=G:\Theta AI\cache
set PYTHONPATH=G:\Theta AI

REM Create cache directory if it doesn't exist
if not exist "G:\Theta AI\cache" mkdir "G:\Theta AI\cache"

echo Starting download...
echo Note: First download may take a while depending on your internet speed.
echo.

REM Run the download script
REM Use --sample_size to limit the number of examples if needed
REM Example: python download_openmath_instruct.py --sample_size 10000
python download_openmath_instruct.py %*

if errorlevel 1 (
    echo.
    echo Download failed. Please check the error messages above.
    echo.
) else (
    echo.
    echo Download completed successfully!
    echo Dataset saved to: G:\Theta AI\Datasets\openmath_instruct_1.json
    echo.
    echo You can now run data_processor.py or prepare_data_for_training.py
    echo to include this dataset in your training pipeline.
    echo.
)

pause
