# Quick Start Training Guide

Get Theta AI training in under 5 minutes.

## Prerequisites

- Completed [Installation](INSTALLATION.md)
- GPU with 12GB+ VRAM
- Training dataset (see [Datasets](DATASETS.md))

## Quick Start (Windows)

### Option 1: Full Enhanced Training (Recommended)

```batch
train_overnight_enhanced.bat
```

This script:
- Prepares and validates datasets
- Runs 20 epochs with RTX 3060 optimizations
- Sends email notifications on progress
- Saves best checkpoint automatically
- Runs evaluation after training

### Option 2: Manual Training

```bash
python src/training/train_enhanced.py ^
  --data_path "Datasets/your_data.json" ^
  --output_dir "models/theta_custom" ^
  --model_name "gpt2-medium" ^
  --batch_size 3 ^
  --epochs 10 ^
  --learning_rate 3e-5
```

## Minimal Training Example

For testing with minimal resources:

```bash
python src/training/train_enhanced.py ^
  --data_path "Datasets/basic_conversation.json" ^
  --output_dir "models/theta_test" ^
  --model_name "gpt2" ^
  --batch_size 2 ^
  --epochs 3 ^
  --no_rdrop ^
  --no_ema ^
  --no_curriculum
```

## Training Output

After training completes:
```
models/
├── theta_enhanced_YYYYMMDD/
│   ├── theta_final/          # Final model
│   ├── best_model.pt         # Best checkpoint
│   ├── loss_history.json     # Training metrics
│   └── training_config.json  # Saved configuration
```

## Monitoring Training

### Console Output
The training script displays real-time metrics:
- Loss (training and validation)
- Perplexity
- Token accuracy
- EMA status
- Curriculum progress

### Email Notifications
If configured in `.env`, you'll receive:
- Training start notification
- Epoch completion updates
- 10-minute status updates with alerts
- Training completion summary

### Log Files
Check `logs/training_enhanced_*.log` for detailed logs.

## Resume Training

Training automatically saves checkpoints. To resume:

```bash
python src/training/train_enhanced.py ^
  --data_path "Datasets/your_data.json" ^
  --output_dir "models/theta_enhanced_YYYYMMDD" ^
  --resume_from_checkpoint
```

## Common Issues

| Issue | Solution |
|-------|----------|
| OOM Error | Reduce batch_size to 2 |
| Slow training | Enable mixed precision (default) |
| High val loss | Increase epochs, check data quality |

## Next Steps

- [Training Pipeline Details](TRAINING_PIPELINE.md)
- [Hyperparameter Tuning](HYPERPARAMETERS.md)
- [RTX 3060 Optimizations](RTX3060_OPTIMIZATIONS.md)
