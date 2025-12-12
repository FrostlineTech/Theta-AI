@echo off
echo ============================================================
echo OpenAssistant OASST1 Dataset Downloader
echo ============================================================
echo.
echo This script will download the OpenAssistant/oasst1 dataset
echo from HuggingFace and process it for Theta AI training.
echo.
echo Dataset: https://huggingface.co/datasets/OpenAssistant/oasst1
echo - ~161k high-quality human-assistant conversation messages
echo - Filtered for English by default
echo - Converted to Q&A format for training pipeline
echo.

REM Set environment variables
set PYTHONPATH=.
set HF_HOME=G:\Theta AI\cache

REM Create cache directory if needed
if not exist "G:\Theta AI\cache" mkdir "G:\Theta AI\cache"

echo Starting download...
echo.

python download_openassistant_oasst1.py ^
    --output-dir "G:\Theta AI\Datasets" ^
    --cache-dir "G:\Theta AI\cache" ^
    --language en ^
    --enhanced ^
    --split-domains

echo.
echo ============================================================
echo Download complete!
echo ============================================================
echo.
echo Output files:
echo - G:\Theta AI\Datasets\openassistant_oasst1.json
echo - G:\Theta AI\Datasets\openassistant_oasst1_enhanced.json
echo - G:\Theta AI\Datasets\openassistant_domains\ (domain-specific files)
echo.
echo The dataset is now integrated with:
echo - src\data_processor.py
echo - prepare_data_for_training.py
echo.
echo Run prepare_data_for_training.py to include in your training data.
echo.

pause
