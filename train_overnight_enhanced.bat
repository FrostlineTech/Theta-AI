@echo off
echo Theta AI - Enhanced Overnight Training Session v3.0 (RTX 3060 Optimized)
echo =========================================================================
echo.
echo This script will run an enhanced training session with:
echo.
echo === Core Training Features ===
echo - Knowledge base enhancements (RAG, consistency checking, case studies)
echo - Technical domain knowledge graphs and specialized embeddings
echo - Early stopping with patience
echo - Cosine LR scheduling with hard restarts (3 cycles)
echo - Gradient checkpointing for memory efficiency
echo - Mixed precision training (FP16)
echo - Validation-based model saving (only best checkpoint)
echo - OpenWebText data integration
echo - Email notifications for training progress
echo.
echo === RTX 3060 12GB Optimizations (NEW) ===
echo - Label Smoothing (0.1) - Reduces overconfidence
echo - R-Drop Regularization - Dual forward passes with KL divergence
echo - LLRD (Layer-wise Learning Rate Decay) - Lower layers learn slower
echo - EMA (Exponential Moving Average) - Smoothed weight updates
echo - Curriculum Learning - Easier examples first, harder later
echo - Gradient Noise Injection - Helps escape local minima
echo - CPU Offloading (60%%) - Optimizer states offloaded to RAM
echo.
echo === Hardware Optimizations ===
echo - RTX 3060 specific optimizations (12GB VRAM)
echo - Ryzen 5-5500 CPU optimizations
echo - Disk space efficiency (1TB storage)
echo - 32GB RAM optimizations with CPU offloading
echo.
echo Started at: %date% %time%
echo.

REM Set environment variables and optimizations
set CUDA_VISIBLE_DEVICES=0
set PYTHONPATH=.

REM CPU and RAM optimizations for Ryzen 5-5500 (6 cores/12 threads)
REM Reserve 1-2 cores for OS/GPU driver overhead, use 4 for compute
REM This provides better balance when CPU offloading is active
set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=4
set NUMEXPR_MAX_THREADS=4
set NUMEXPR_NUM_THREADS=4

REM PyTorch DataLoader workers (leave headroom for main process)
set PYTORCH_DATALOADER_WORKERS=2

REM PyTorch optimizations for RTX 3060 12GB
REM max_split_size helps with memory fragmentation
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True

REM Enable TensorFloat-32 for Ampere GPUs (RTX 30xx series)
set NVIDIA_TF32_OVERRIDE=1

REM Create training log directory if it doesn't exist
if not exist "logs" mkdir logs

REM Clean up old logs to save disk space (keep only last 5)
echo Cleaning up old logs...
for /f "skip=5 delims=" %%a in ('dir /b /o-d logs\training_enhanced_*.log') do (
  del /q "logs\%%a" 2>nul
)

REM Create timestamped log file
set logfile=logs\training_enhanced_%date:~-4,4%%date:~-7,2%%date:~-10,2%_%time:~0,2%%time:~3,2%%time:~6,2%.log
set logfile=%logfile: =0%

echo Starting enhanced training session...
echo Training log: %logfile%
echo.

REM Set custom cache directory to use G: drive instead of C: drive
set HF_HOME=G:\Theta AI\cache

REM Create cache directory if it doesn't exist
if not exist "G:\Theta AI\cache" mkdir "G:\Theta AI\cache"

REM Skip package installation (already installed)
echo Using existing Python packages...

REM Check for OpenWebText data and download if needed (optimized sample size for 1TB disk)
echo Checking for OpenWebText data...
if not exist "G:\Theta AI\Datasets\openwebtext_processed.json" (
  echo OpenWebText data not found. Using fallback data generation...
  python src\data_processing\direct_openwebtext.py --sample_size 3500 --output_file "G:\Theta AI\Datasets\openwebtext_processed.json" --cache_dir "G:\Theta AI\cache" --use-fallback
  if errorlevel 1 (
    echo Warning: Error generating OpenWebText data. Will continue with existing datasets.
  ) else (
    echo Successfully created OpenWebText data.
  )
)

REM Clean temporary files to save disk space
echo Cleaning temporary files...
del /q /s "Datasets\*.tmp" 2>nul
del /q /s "models\*.tmp" 2>nul

REM Setup enhanced database tables if they don't exist
echo Setting up enhanced database tables...
python setup_enhanced_db.py

REM Fix any encoding issues in dataset files
echo Fixing encoding issues in dataset files...
python src\data_processing\fix_encodings.py

REM Verify that our new test set exists
echo Checking for test set...
if not exist "Datasets\test_set.json" (
  echo WARNING: test_set.json not found. This file is required for proper evaluation.
  echo The training will use the processed_data.json file that includes all our new datasets.
) else (
  echo Verified: test_set.json found.  This will be used for model evaluation.
)

REM Run knowledge base enhancement process
echo Enhancing knowledge base...
python src\data_processing\knowledge_base_enhancer_main.py

REM Verify email notification system
echo Verifying email notification system...
REM Send a test email to confirm setup
python -c "from src.utils.email_notifier import TrainingEmailNotifier; notifier = TrainingEmailNotifier(); notifier._send_email('Theta AI Training Starting Soon', '<html><body><h1>Training Preparation</h1><p>Your training system is about to start. This email confirms the notification system is working properly.</p></body></html>')"

REM Dependencies already installed above

REM Setup NLTK data files to fix missing files issue
echo Setting up NLTK data files...

REM Fix NLTK punkt data issue directly instead of using separate script
python -c "import nltk, os; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('stopwords'); data_path = nltk.data.path[0]; punkt_dir = os.path.join(data_path, 'tokenizers', 'punkt_tab', 'english'); os.makedirs(punkt_dir, exist_ok=True); open(os.path.join(punkt_dir, 'collocations.tab'), 'w').close(); open(os.path.join(punkt_dir, 'sentence_context_forms.tab'), 'w').close(); print('NLTK data setup complete')"

REM Run enhanced data preparation implementing all 5 recommendations
echo.
echo Running enhanced data preparation with all 5 recommendations...
echo 1. Balance Domain-Specific Data
echo 2. Data Quality Check
echo 3. Technical Context Enhancement
echo 4. Curriculum Learning
echo 5. Augment with Synthetic Examples
echo.
python prepare_data_for_training.py --output "Datasets/enhanced_training_data.json"

REM Check if enhanced data preparation succeeded
if not exist "Datasets/enhanced_training_data.json" (
    echo Enhanced data preparation failed. Using processed_data.json as fallback.
    copy /Y "Datasets/processed_data.json" "Datasets/enhanced_training_data.json" > nul
) else (
    echo Enhanced training data created successfully.
)

REM Run the enhanced training with optimized parameters for 12GB VRAM
echo Using enhanced_training_data.json with our implemented recommendations...
echo RTX 3060 12GB Optimizations: LLRD, R-Drop, EMA, Label Smoothing, Curriculum Learning, CPU Offloading
REM Fix #1: Safer hyperparameters to prevent gradient flattening
REM Fix #3: Disabled domain_stratified since gradient_accumulation > 1
REM Fix #4: Reduced EMA warmup to 1 epoch
REM Fix #5: Reduced code_contrastive_weight to 0.02
python src/training/train_enhanced.py ^
  --data_path "Datasets/enhanced_training_data.json" ^
  --output_dir "models/theta_enhanced_%date:~-4,4%%date:~-7,2%%date:~-10,2%" ^
  --model_name "gpt2-medium" ^
  --batch_size 3 ^
  --gradient_accumulation_steps 4 ^
  --learning_rate 3e-5 ^
  --epochs 20 ^
  --patience 5 ^
  --warmup_proportion 0.15 ^
  --scheduler_type "cosine_hard_restarts" ^
  --num_cycles 4 ^
  --weight_decay 0.01 ^
  --optimize_memory ^
  --keep_best_only ^
  --label_smoothing 0.05 ^
  --use_rdrop ^
  --rdrop_alpha 0.05 ^
  --use_llrd ^
  --llrd_factor 0.85 ^
  --use_ema ^
  --ema_decay 0.998 ^
  --ema_warmup_epochs 1 ^
  --use_curriculum ^
  --curriculum_start_fraction 0.7 ^
  --no_gradient_noise ^
  --use_cpu_offload ^
  --cpu_offload_fraction 0.5 ^
  --use_quality_weighting ^
  --no_domain_stratified ^
  --use_code_contrastive ^
  --code_contrastive_weight 0.02 ^
  --use_dynamic_curriculum ^
  --dynamic_curriculum_warmup 3 ^
  --log_file "%logfile%"

REM Check if training succeeded
if errorlevel 1 (
    echo.
    echo ============================================================
    echo TRAINING FAILED - Check the log file for details:
    echo %logfile%
    echo ============================================================
    echo.
    pause
    exit /b 1
)

echo.
echo Training completed successfully at: %date% %time%
echo.

REM Check if model directory exists before proceeding
echo.
echo Checking for trained model...
if not exist "models/theta_enhanced_%date:~-4,4%%date:~-7,2%%date:~-10,2%/theta_final" (
  echo No trained model found. Skipping domain fine-tuning.
) else (
  REM Step 4: Domain-specific fine-tuning (if domains have sufficient data)
  echo.
  echo Step 4: Starting domain-specific fine-tuning...
  echo.
  python src/training/domain_tuning.py --check_only
  if errorlevel 1 (
    echo Skipping domain tuning due to insufficient data.
  ) else (
    echo Running domain-specific fine-tuning...
    python src/training/domain_tuning.py ^
      --domains cybersecurity programming networking ^
      --base_model "models/theta_enhanced_%date:~-4,4%%date:~-7,2%%date:~-10,2%/theta_final" ^
      --batch_size 2 ^
      --gradient_accumulation_steps 8
  )
)

REM Step 5: Evaluate the trained model...
echo.
echo Step 5: Evaluating the trained model...
echo.
if not exist "models/theta_enhanced_%date:~-4,4%%date:~-7,2%%date:~-10,2%/theta_final" (
  echo No trained model found. Skipping evaluation.
) else (
  python src/training/evaluate_model.py ^
    --model_path "models/theta_enhanced_%date:~-4,4%%date:~-7,2%%date:~-10,2%/theta_final" ^
    --test_data "Datasets/test_set.json" ^
    --output_file "stats/evaluation_%date:~-4,4%%date:~-7,2%%date:~-10,2%.json"
)

REM Step 6: Generate optimized model version for inference
echo.
echo Step 6: Creating optimized version for inference...
echo.
if not exist "models/theta_enhanced_%date:~-4,4%%date:~-7,2%%date:~-10,2%/theta_final" (
  echo No trained model found. Skipping optimization.
) else (
  python src/training/optimize_for_inference.py ^
    --model_path "models/theta_enhanced_%date:~-4,4%%date:~-7,2%%date:~-10,2%/theta_final" ^
    --output_path "models/theta_enhanced_%date:~-4,4%%date:~-7,2%%date:~-10,2%/theta_inference" ^
    --quantize 8 ^
    --prune False
)

echo.
echo Final model saved to: models/theta_enhanced_%date:~-4,4%%date:~-7,2%%date:~-10,2%\theta_final
echo Optimized inference model: models/theta_enhanced_%date:~-4,4%%date:~-7,2%%date:~-10,2%\theta_inference
echo.

REM Manual checkpoint cleanup - this has been disabled as requested
echo Checkpoint cleanup disabled - you will need to clean up checkpoints manually

echo.
echo To use the trained model, run: interface.bat
echo.

pause
