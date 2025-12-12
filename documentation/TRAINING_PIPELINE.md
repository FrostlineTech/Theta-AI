# Training Pipeline Documentation

Complete documentation for the Theta AI training system.

## Overview

The Theta AI training pipeline is built on PyTorch and HuggingFace Transformers, optimized for RTX 3060 12GB GPUs. It includes advanced features like R-Drop regularization, EMA, curriculum learning, and dynamic curriculum tracking.

## Architecture

```text
train_overnight_enhanced.bat
    │
    ├── Data Preparation
    │   └── prepare_data_for_training.py
    │
    ├── Training
    │   └── src/training/train_enhanced.py
    │       ├── ThetaModel (src/model/theta_model.py)
    │       ├── ValidationMetrics
    │       ├── Email Notifications
    │       └── Checkpoint Management
    │
    └── Evaluation
        └── src/training/evaluate_model.py
```

## Core Components

### train_enhanced.py

Main training script with these key classes:

| Class | Purpose |
|-------|---------|
| `ExponentialMovingAverage` | Smooths weight updates for better generalization |
| `LabelSmoothingLoss` | Reduces overconfidence in predictions |
| `CurriculumSampler` | Orders samples from easy to hard |
| `DynamicCurriculumTracker` | Adjusts sampling based on per-sample loss |
| `DomainStratifiedSampler` | Balances domain representation in batches |
| `CodeContrastiveLoss` | Auxiliary loss for code detection |
| `CPUOffloadOptimizer` | Offloads optimizer states to RAM |
| `ValidationMetrics` | Comprehensive validation metrics |

### Training Flow

1. **Initialization**
   - Load model (GPT-2 base/medium)
   - Setup tokenizer with special tokens
   - Initialize optimizers with LLRD

2. **Data Loading**
   - Load JSON dataset
   - Apply curriculum sampler
   - Create train/validation split (90/10)

3. **Training Loop**
   - Forward pass with mixed precision
   - R-Drop: dual forward passes + KL divergence
   - Label smoothing loss
   - Gradient accumulation
   - CPU offloading for optimizer states
   - EMA weight updates

4. **Validation**
   - Compute validation loss
   - Token accuracy
   - Domain-specific metrics
   - Perplexity

5. **Checkpointing**
   - Save best model based on val_loss
   - Early stopping with patience
   - Cleanup old checkpoints

## Training Features

### R-Drop Regularization

Runs two forward passes with different dropout masks and minimizes KL divergence between outputs.

```python
# Enabled by default
--use_rdrop
--rdrop_alpha 0.05  # KL divergence weight
```

### Layer-wise Learning Rate Decay (LLRD)

Lower layers learn slower (more general features), higher layers learn faster.

```python
--use_llrd
--llrd_factor 0.85  # 15% reduction per layer
```

### Exponential Moving Average (EMA)

Maintains smoothed copy of weights for better final model.

```python
--use_ema
--ema_decay 0.998
--ema_warmup_epochs 1  # Disable EMA for first epoch
```

### Curriculum Learning

Starts with easier (shorter) examples, gradually introduces harder ones.

```python
--use_curriculum
--curriculum_start_fraction 0.7  # Start with 70% of data
```

### Dynamic Curriculum

Tracks per-sample loss and upweights difficult samples.

```python
--use_dynamic_curriculum
--dynamic_curriculum_warmup 3  # Activate after 3 epochs
```

### Quality Weighting

Weights samples by quality scores from data preparation.

```python
--use_quality_weighting
```

### CPU Offloading

Offloads optimizer states to CPU RAM to save GPU memory.

```python
--use_cpu_offload
--cpu_offload_fraction 0.5  # 50% of states on CPU
```

## Email Notifications

Training sends email updates (if configured):

- **Training Start**: Configuration summary
- **Epoch Updates**: Loss, accuracy, ETA
- **10-Minute Status**: GPU status + training metrics with alerts
- **Completion**: Final results summary

### Alert Thresholds

| Metric | Alert Condition |
|--------|-----------------|
| Perplexity | > 50 (critical) |
| Token Accuracy | < 50% |
| KL Loss | > 1.0 |
| Difficult Ratio | > 70% |
| Gradient Clip Ratio | > 80% |
| Learning Rate | < 1e-8 |

## Output Files

After training:

```text
models/theta_enhanced_YYYYMMDD/
├── theta_final/           # Final model weights
│   ├── config.json
│   ├── pytorch_model.bin
│   └── tokenizer files
├── best_model.pt          # Best checkpoint (state_dict)
├── loss_history.json      # All training metrics
├── training_config.json   # Saved hyperparameters
└── training_script_backup.py  # Script copy for reproducibility
```

## Recommended Configurations

### RTX 3060 12GB (Default)

```batch
--batch_size 3
--gradient_accumulation_steps 4
--use_cpu_offload
--cpu_offload_fraction 0.5
```

### RTX 3090 24GB

```batch
--batch_size 8
--gradient_accumulation_steps 2
--no_cpu_offload
```

### CPU Only (Not Recommended)

```batch
--batch_size 1
--gradient_accumulation_steps 8
--no_rdrop
--no_ema
```

## Troubleshooting

### CUDA Out of Memory

1. Reduce `batch_size`
2. Increase `gradient_accumulation_steps`
3. Enable `--use_cpu_offload`
4. Disable `--no_rdrop` (removes dual forward pass)

### Training Divergence

1. Reduce `learning_rate`
2. Increase `warmup_proportion`
3. Check data quality
4. Enable `--ablation_mode` to test baseline

### Slow Training

1. Ensure CUDA is detected
2. Check GPU utilization with `nvidia-smi`
3. Reduce `num_workers` if CPU bottlenecked
