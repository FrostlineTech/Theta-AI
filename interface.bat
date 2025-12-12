@echo off
setlocal EnableDelayedExpansion
echo Theta AI - Interactive Interface
echo ==============================
echo.

REM Use best performing model (epoch 26) based on validation loss
set "best_checkpoint=models\theta_checkpoint_epoch_26"

REM Check if best model exists, otherwise fall back to finding latest
if exist "%best_checkpoint%" (
    echo Using best performing model: %best_checkpoint% (Epoch 26)
) else (
    echo Best model not found, searching for alternatives...
    
    REM Find the latest checkpoint folder as fallback
    set "latest_checkpoint="
    set "latest_epoch=0"
    
    for /d %%i in (models\theta_checkpoint_epoch_*) do (
        set "folder=%%i"
        set "epoch=!folder:*_epoch_=!"
        if !epoch! GTR !latest_epoch! (
            set "latest_epoch=!epoch!"
            set "latest_checkpoint=%%i"
        )
    )
    
    REM If no checkpoint found, check if final model exists
    if "%latest_checkpoint%"=="" (
        if exist "models\theta_final" (
            set "best_checkpoint=models\theta_final"
            echo Using final model
        ) else (
            echo No model checkpoints found.
            echo Please run train_overnight.bat first.
            pause
            exit /b
        )
    ) else (
        set "best_checkpoint=%latest_checkpoint%"
        echo Found latest checkpoint: %latest_checkpoint% (Epoch %latest_epoch%)
    )
)

echo.
echo Starting Theta AI interface with the best performing model...
echo Using model with lowest validation loss for optimal responses
echo.

REM Run the interface with the best checkpoint and dataset
python src/interface.py --model_path "%best_checkpoint%" --dataset_path "Datasets/processed_data.json"

echo.
echo Theta AI interface closed.
echo.

pause