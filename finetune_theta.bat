@echo off
echo Theta AI - Targeted Fine-tuning Script
echo ========================================
echo.
echo This script runs fine-tuning with reduced regularization settings.
echo Use this when initial training stalls (validation loss stuck, early stopping).
echo.
echo Started at: %date% %time%
echo.

REM Set environment variables
set CUDA_VISIBLE_DEVICES=0
set PYTHONPATH=.

REM CPU optimizations for Ryzen 5-5500
set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=4
set NUMEXPR_MAX_THREADS=4
set PYTORCH_DATALOADER_WORKERS=2

REM PyTorch optimizations for RTX 3060
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True
set NVIDIA_TF32_OVERRIDE=1

REM Custom cache directory
set HF_HOME=G:\Theta AI\cache

REM Create output directory
if not exist "models" mkdir models

REM Check for existing fine-tune config
set CONFIG_PATH=models\theta_enhanced_20251212\finetune_config.json

if exist "%CONFIG_PATH%" (
    echo Found config: %CONFIG_PATH%
    echo.
) else (
    echo No config found at %CONFIG_PATH%
    echo.
    echo Please create a fine-tuning config JSON file with these settings:
    echo {
    echo   "model_name": "models/theta_enhanced_YYYYMMDD/theta_final",
    echo   "output_dir": "models/theta_finetune_YYYYMMDD",
    echo   "learning_rate": 5e-06,
    echo   "epochs": 4,
    echo   "label_smoothing": 0.03,
    echo   "rdrop_alpha": 0.02,
    echo   "ema_decay": 0.995
    echo }
    echo.
    echo Or specify a config path as argument: finetune_theta.bat path\to\config.json
    pause
    exit /b 1
)

REM Allow override via command line argument
if not "%~1"=="" (
    set CONFIG_PATH=%~1
    echo Using custom config: %CONFIG_PATH%
)

echo.
echo Starting fine-tuning with config: %CONFIG_PATH%
echo.

python src/training/run_finetune.py --config "%CONFIG_PATH%"

if errorlevel 1 (
    echo.
    echo ============================================================
    echo FINE-TUNING FAILED - Check console output for details
    echo ============================================================
    pause
    exit /b 1
)

echo.
echo Fine-tuning completed successfully at: %date% %time%
echo.
echo Output saved to the directory specified in your config file.
echo.

pause
