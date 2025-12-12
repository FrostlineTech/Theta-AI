# RTX 3060 12GB Optimization Guide

This document details the GPU-specific optimizations implemented for the NVIDIA RTX 3060 with 12GB VRAM.

## Hardware Profile

| Component | Specification |
|-----------|--------------|
| GPU | NVIDIA RTX 3060 12GB |
| Architecture | Ampere (GA106) |
| CUDA Cores | 3584 |
| Tensor Cores | 112 |
| Memory | 12GB GDDR6 |
| Memory Bandwidth | 360 GB/s |
| TDP | 170W |

## Memory Management

### Problem

GPT-2 Medium (355M params) + optimizer states + activations can exceed 12GB during training.

### Solutions Implemented

#### 1. CPU Offloading (Primary)

Offloads 50% of optimizer states to CPU RAM.

```python
--use_cpu_offload
--cpu_offload_fraction 0.5
```

**Memory savings**: ~2-3GB VRAM

#### 2. Gradient Accumulation

Simulates larger batches without memory increase.

```python
--batch_size 3
--gradient_accumulation_steps 4
# Effective batch size: 12
```

#### 3. Mixed Precision (FP16)

Uses 16-bit floats for forward/backward passes.

```python
# Enabled automatically when CUDA is available
scaler = torch.amp.GradScaler('cuda')
```

**Memory savings**: ~40% reduction

#### 4. Gradient Checkpointing

Recomputes activations during backward pass (trades compute for memory).

```python
model.gradient_checkpointing_enable()
```

## Environment Variables

Set in `train_overnight_enhanced.bat`:

```batch
# CUDA device
set CUDA_VISIBLE_DEVICES=0

# Memory allocation
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True

# TensorFloat-32 (Ampere optimization)
set NVIDIA_TF32_OVERRIDE=1

# CPU threads (for Ryzen 5-5500)
set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=4
```

## Optimal Batch Configuration

### Memory Usage Breakdown

| Component | Memory |
|-----------|--------|
| Model weights (FP16) | ~700MB |
| Optimizer states | ~1.4GB |
| Gradients | ~700MB |
| Activations (batch=3) | ~2GB |
| CUDA overhead | ~1GB |
| **Total** | **~6GB** |

With CPU offloading, actual VRAM usage: **~4-5GB**

### Recommended Settings

```batch
--batch_size 3
--gradient_accumulation_steps 4
--use_cpu_offload
--cpu_offload_fraction 0.5
```

## Training Speed Optimizations

### 1. TensorFloat-32 (TF32)

RTX 30-series supports TF32 for faster matrix operations.

```batch
set NVIDIA_TF32_OVERRIDE=1
```

**Speedup**: 2-3x for matmul operations

### 2. Non-Blocking Transfers

CPU offloader uses non-blocking CUDA transfers.

```python
state[key] = value.cuda(non_blocking=True)
```

### 3. Pinned Memory

DataLoader uses pinned memory for faster CPU→GPU transfers.

```python
DataLoader(..., pin_memory=True)
```

### 4. Persistent Workers

Keeps DataLoader workers alive between epochs.

```python
DataLoader(..., persistent_workers=True, num_workers=2)
```

## Regularization Optimizations

### Why These Matter for 12GB

Limited batch size (3) can cause noisy gradients. These techniques compensate:

| Technique | Purpose | Memory Cost |
|-----------|---------|-------------|
| R-Drop | Consistency regularization | 2x forward pass |
| Label Smoothing | Prevent overconfidence | None |
| EMA | Smooth weight updates | CPU only |
| LLRD | Layer-specific learning | None |

### R-Drop Memory Consideration

R-Drop runs 2 forward passes. Memory is managed by:
- Reusing same batch (no extra data loading)
- Gradient accumulation (smaller per-step memory)

## Monitoring GPU Health

### In Training

10-minute status emails include:
- GPU temperature
- Memory usage
- Utilization %
- Power draw

### Manual Check

```bash
nvidia-smi --query-gpu=temperature.gpu,memory.used,memory.total,utilization.gpu --format=csv -l 5
```

### Temperature Thresholds

| Temp | Status |
|------|--------|
| < 70°C | Normal |
| 70-80°C | Warm (OK) |
| > 80°C | Hot (check cooling) |
| > 85°C | Throttling likely |

## Troubleshooting

### CUDA Out of Memory

1. Reduce batch_size to 2
2. Increase gradient_accumulation_steps to 6
3. Increase cpu_offload_fraction to 0.6
4. Disable R-Drop: `--no_rdrop`

### Slow Training

1. Check TF32 is enabled
2. Verify GPU utilization > 80%
3. Check for thermal throttling
4. Reduce num_workers if CPU bottlenecked

### Memory Fragmentation

If OOM occurs mid-training:

```batch
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64,expandable_segments:True
```

## Comparison: RTX 3060 vs Other GPUs

| Setting | RTX 3060 (12GB) | RTX 3090 (24GB) | RTX 4090 (24GB) |
|---------|-----------------|-----------------|-----------------|
| batch_size | 3 | 8 | 12 |
| gradient_accumulation | 4 | 2 | 1 |
| cpu_offload | Yes (50%) | Optional | No |
| Training time (20 epochs) | ~8-10 hours | ~4-5 hours | ~2-3 hours |
