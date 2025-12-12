# API Reference

Code documentation for key classes and functions.

## Model Layer

### ThetaModel

```python
from src.model.theta_model import ThetaModel
```

#### Constructor

```python
ThetaModel(model_name: str = "gpt2", device: str = None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | "gpt2" | HuggingFace model name |
| `device` | str | None | Device (auto-detected if None) |

#### Methods

**generate(prompt, max_length, temperature, top_p, top_k)**

Generate text continuation.

```python
response = model.generate(
    prompt="What is AI?",
    max_length=100,
    temperature=0.8,
    top_p=0.9,
    top_k=50
)
```

**save(path)**

Save model and tokenizer.

```python
model.save("models/my_model")
```

**load(path)** (classmethod)

Load saved model.

```python
model = ThetaModel.load("models/my_model")
```

---

## Training Layer

### train_enhanced.train()

Main training function.

```python
from src.training.train_enhanced import train
```

```python
train(args: argparse.Namespace) -> str
```

Returns path to final model directory.

### ExponentialMovingAverage

```python
from src.training.train_enhanced import ExponentialMovingAverage
```

```python
ema = ExponentialMovingAverage(
    model,
    decay=0.999,
    cpu_offload=True,
    warmup_epochs=0
)
```

| Method | Description |
|--------|-------------|
| `set_epoch(epoch)` | Update current epoch |
| `is_active()` | Check if past warmup |
| `update(model)` | Update shadow weights |
| `apply_shadow(model)` | Apply EMA weights |
| `restore(model)` | Restore original weights |

### LabelSmoothingLoss

```python
from src.training.train_enhanced import LabelSmoothingLoss
```

```python
loss_fn = LabelSmoothingLoss(smoothing=0.1, ignore_index=-100)
loss = loss_fn(logits, labels)
```

### CurriculumSampler

```python
from src.training.train_enhanced import CurriculumSampler
```

```python
sampler = CurriculumSampler(
    dataset,
    tokenizer,
    num_epochs=20,
    current_epoch=0,
    start_fraction=0.7
)
```

| Method | Description |
|--------|-------------|
| `set_epoch(epoch)` | Update curriculum progress |
| `__iter__()` | Yield sample indices |
| `__len__()` | Current sample count |

### DynamicCurriculumTracker

```python
from src.training.train_enhanced import DynamicCurriculumTracker
```

```python
tracker = DynamicCurriculumTracker(
    dataset_size=10000,
    warmup_epochs=2
)
```

| Method | Description |
|--------|-------------|
| `update_loss(indices, losses)` | Track per-sample losses |
| `set_epoch(epoch)` | Update current epoch |
| `get_sample_weights()` | Get sampling weights |
| `get_stats()` | Get tracking statistics |

### CPUOffloadOptimizer

```python
from src.training.train_enhanced import CPUOffloadOptimizer
```

```python
optimizer = CPUOffloadOptimizer(
    base_optimizer,
    offload_fraction=0.5
)
```

| Method | Description |
|--------|-------------|
| `pre_step()` | Move states to GPU |
| `post_step()` | Move states to CPU |
| `step()` | Full step with offloading |
| `zero_grad()` | Clear gradients |

### ValidationMetrics

```python
from src.training.train_enhanced import ValidationMetrics
```

```python
metrics = ValidationMetrics(tokenizer)
```

| Method | Description |
|--------|-------------|
| `compute_perplexity(loss)` | Calculate perplexity |
| `compute_token_accuracy(logits, labels)` | Token-level accuracy |
| `update(loss, logits, labels, epoch, domain)` | Update metrics |
| `get_summary()` | Get all metrics |
| `run_full_domain_evaluation(model, device)` | Domain evaluation |
| `reset()` | Reset for new epoch |

---

## Email Notifications

### TrainingEmailNotifier

```python
from src.utils.email_notifier import TrainingEmailNotifier
```

```python
notifier = TrainingEmailNotifier(
    sender="email@example.com",
    recipient="email@example.com",
    model_name="Theta AI"
)
```

| Method | Description |
|--------|-------------|
| `start_training_notification(epochs, batch_size, learning_rate)` | Send start email |
| `epoch_update(epoch, total_epochs, train_loss, val_loss, perplexity, duration)` | Send epoch email |
| `gpu_status_update()` | Send 10-min status |
| `training_completed(total_epochs, best_val_loss, best_epoch, total_duration)` | Send completion |
| `update_metrics(metrics)` | Update current metrics |

### TrainingNotifier (Wrapper)

```python
from src.training.email_integration import get_notifier
```

```python
notifier = get_notifier(model_name="Theta AI")
notifier.start_notification(args)
notifier.update_metrics(metrics_dict)
notifier.epoch_notification(...)
notifier.completion_notification(...)
```

---

## Data Processing

### prepare_data_for_training.py

Main data preparation script.

```bash
python prepare_data_for_training.py --output "Datasets/enhanced_training_data.json"
```

#### Key Functions

```python
load_all_datasets(datasets_dir: str) -> List[Dict]
balance_domains(data: List[Dict], target_distribution: Dict) -> List[Dict]
compute_quality_scores(data: List[Dict]) -> List[Dict]
apply_curriculum_ordering(data: List[Dict]) -> List[Dict]
```

---

## Utility Functions

### get_llrd_optimizer_groups()

Create optimizer groups with layer-wise learning rate decay.

```python
from src.training.train_enhanced import get_llrd_optimizer_groups

groups = get_llrd_optimizer_groups(
    model,
    base_lr=2e-5,
    weight_decay=0.03,
    llrd_factor=0.95
)
optimizer = AdamW(groups)
```

### compute_rdrop_loss()

Compute R-Drop regularization loss.

```python
from src.training.train_enhanced import compute_rdrop_loss

result = compute_rdrop_loss(
    model, input_ids, attention_mask, labels,
    alpha=0.1,
    label_smoothing_fn=None,
    scaler=None,
    device_type="cuda"
)
# result['total_loss'], result['kl_loss'], result['logits']
```

### add_gradient_noise()

Add gradient noise for regularization.

```python
from src.training.train_enhanced import add_gradient_noise

add_gradient_noise(model, noise_scale=0.01)
```

---

## GPU Utilities

### get_best_gpu_info()

Get GPU information using multiple methods.

```python
from src.utils.gpu_info import get_best_gpu_info

info = get_best_gpu_info()
# {
#   'name': 'NVIDIA GeForce RTX 3060',
#   'temperature': 65,
#   'memory_used': 4096,
#   'memory_total': 12288,
#   'utilization': 85,
#   'power_draw': 120
# }
```

---

## Cleanup Utilities

```python
from src.training.cleanup_utils import (
    cleanup_old_checkpoints,
    cleanup_temp_files,
    check_disk_space,
    optimize_memory_usage,
    keep_only_best_checkpoint
)
```

| Function | Description |
|----------|-------------|
| `cleanup_old_checkpoints(dir, keep_last_n)` | Remove old checkpoints |
| `cleanup_temp_files()` | Remove temp files |
| `check_disk_space(min_gb)` | Verify disk space |
| `optimize_memory_usage()` | Force garbage collection |
| `keep_only_best_checkpoint(dir, best_epoch)` | Keep only best |
