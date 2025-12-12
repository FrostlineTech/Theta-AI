# Hyperparameter Reference

Complete reference for all training hyperparameters.

## Core Training Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `--model_name` | gpt2 | gpt2, gpt2-medium, gpt2-large | Base model |
| `--batch_size` | 3 | 1-8 | Samples per GPU batch |
| `--gradient_accumulation_steps` | 4 | 1-16 | Steps before weight update |
| `--learning_rate` | 3e-5 | 1e-6 to 1e-4 | Peak learning rate |
| `--epochs` | 20 | 1-100 | Training epochs |
| `--patience` | 5 | 1-20 | Early stopping patience |
| `--warmup_proportion` | 0.15 | 0.0-0.3 | LR warmup fraction |
| `--weight_decay` | 0.01 | 0.0-0.1 | L2 regularization |

## Learning Rate Schedulers

| Parameter | Default | Options | Description |
|-----------|---------|---------|-------------|
| `--scheduler_type` | cosine_hard_restarts | linear, cosine, cosine_hard_restarts | LR schedule |
| `--num_cycles` | 4 | 1-10 | Restart cycles (hard restarts only) |

## RTX 3060 Optimizations

### Label Smoothing

Reduces overconfidence by distributing probability mass.

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `--label_smoothing` | 0.05 | 0.0-0.2 | Smoothing factor (0 = disabled) |

**Effect**: Lower values (0.05) for stable training, higher (0.1-0.2) for more regularization.

### R-Drop Regularization

Dual forward passes with KL divergence consistency loss.

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `--use_rdrop` | True | - | Enable R-Drop |
| `--rdrop_alpha` | 0.05 | 0.01-0.5 | KL divergence weight |

**Effect**: Higher alpha = stronger regularization, may slow convergence.

### Layer-wise Learning Rate Decay (LLRD)

Lower layers learn slower to preserve pretrained features.

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `--use_llrd` | True | - | Enable LLRD |
| `--llrd_factor` | 0.85 | 0.7-0.99 | Decay per layer |

**Effect**: 0.85 means each lower layer has 85% of the upper layer's LR.

### Exponential Moving Average (EMA)

Maintains smoothed weight copy for better generalization.

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `--use_ema` | True | - | Enable EMA |
| `--ema_decay` | 0.998 | 0.99-0.9999 | Decay rate |
| `--ema_warmup_epochs` | 1 | 0-5 | Epochs before EMA activates |

**Effect**: Higher decay = smoother updates, slower adaptation.

### Curriculum Learning

Orders training samples from easy to hard.

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `--use_curriculum` | True | - | Enable curriculum |
| `--curriculum_start_fraction` | 0.7 | 0.3-1.0 | Initial data fraction |

**Effect**: 0.7 = start with easiest 70%, gradually add harder samples.

### Dynamic Curriculum

Upweights difficult samples based on loss tracking.

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `--use_dynamic_curriculum` | True | - | Enable dynamic curriculum |
| `--dynamic_curriculum_warmup` | 3 | 0-10 | Warmup epochs |

**Effect**: After warmup, high-loss samples appear more frequently.

### CPU Offloading

Offloads optimizer states to CPU RAM.

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `--use_cpu_offload` | True | - | Enable offloading |
| `--cpu_offload_fraction` | 0.5 | 0.0-0.8 | Fraction to offload |

**Effect**: Saves GPU memory, slight speed overhead.

### Quality Weighting

Weights samples by quality score from data preparation.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--use_quality_weighting` | True | Enable quality weighting |

### Code Contrastive Loss

Auxiliary loss for code block detection.

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `--use_code_contrastive` | True | - | Enable code loss |
| `--code_contrastive_weight` | 0.02 | 0.01-0.1 | Loss weight |

### Gradient Noise

Injects noise to escape local minima.

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `--use_gradient_noise` | False | - | Enable noise |
| `--gradient_noise_scale` | 0.01 | 0.001-0.1 | Noise scale |

## Memory Optimization

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--optimize_memory` | True | Enable memory cleanup |
| `--optimize_disk` | True | Enable disk space management |
| `--keep_best_only` | True | Keep only best checkpoint |
| `--keep_last_n_epochs` | 3 | Checkpoints to retain |

## Recommended Configurations

### Conservative (Stable)

```batch
--learning_rate 2e-5
--label_smoothing 0.05
--rdrop_alpha 0.03
--llrd_factor 0.9
--ema_decay 0.999
--no_gradient_noise
```

### Aggressive (Faster Learning)

```batch
--learning_rate 5e-5
--label_smoothing 0.1
--rdrop_alpha 0.1
--llrd_factor 0.8
--ema_decay 0.995
--use_gradient_noise
```

### Ablation (Baseline Testing)

```batch
--ablation_mode
# Disables: R-Drop, gradient noise, dynamic curriculum, code contrastive
```

### Warm Start (Gradual Feature Enable)

```batch
--warm_start
# Epochs 0-1: vanilla CE + LLRD
# Epochs 2-3: + label smoothing
# Epochs 4+: + R-Drop + EMA
```
