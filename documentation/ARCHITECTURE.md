# System Architecture

Technical architecture of the Theta AI training system.

## High-Level Overview

```text
┌─────────────────────────────────────────────────────────────────┐
│                    train_overnight_enhanced.bat                  │
│                      (Orchestration Layer)                       │
└─────────────────────────────────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌───────────────┐     ┌─────────────────┐     ┌─────────────────┐
│     Data      │     │    Training     │     │   Evaluation    │
│  Preparation  │     │    Pipeline     │     │    Pipeline     │
└───────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Datasets/   │     │    models/      │     │     stats/      │
│   (JSON)      │     │  (checkpoints)  │     │   (metrics)     │
└───────────────┘     └─────────────────┘     └─────────────────┘
```

## Core Components

### 1. Model Layer (`src/model/`)

```text
theta_model.py
├── ThetaModel
│   ├── __init__(model_name, device)
│   ├── model: GPT2LMHeadModel
│   ├── tokenizer: GPT2TokenizerFast
│   ├── generate(prompt, max_length, temperature)
│   ├── save(path)
│   └── load(path)
```

**Base Model**: GPT-2 (124M) or GPT-2 Medium (355M)

**Key Features**:

- Custom tokenizer with special tokens
- Generation with temperature control
- Checkpoint save/load

### 2. Training Layer (`src/training/`)

```text
train_enhanced.py
├── Optimization Classes
│   ├── ExponentialMovingAverage
│   ├── LabelSmoothingLoss
│   ├── CPUOffloadOptimizer
│   └── CodeContrastiveLoss
│
├── Sampling Classes
│   ├── CurriculumSampler
│   ├── DynamicCurriculumTracker
│   ├── DynamicWeightedSampler
│   └── DomainStratifiedSampler
│
├── Metrics Classes
│   ├── DomainMetrics
│   └── ValidationMetrics
│
└── Functions
    ├── train(args)
    ├── compute_rdrop_loss()
    ├── get_llrd_optimizer_groups()
    └── add_gradient_noise()
```

### 3. Data Processing Layer (`src/data_processing/`)

```text
prepare_data_for_training.py
├── load_all_datasets()
├── balance_domains()
├── compute_quality_scores()
├── apply_curriculum_ordering()
└── generate_synthetic_examples()

knowledge_base_enhancer_main.py
├── RAG integration
├── Consistency checking
└── Technical embeddings
```

### 4. Utilities Layer (`src/utils/`)

```text
email_notifier.py
├── TrainingEmailNotifier
│   ├── start_training_notification()
│   ├── epoch_update()
│   ├── gpu_status_update()
│   ├── training_completed()
│   └── update_metrics()

gpu_info.py
├── get_best_gpu_info()
└── nvidia-smi parsing
```

## Data Flow

### Training Data Flow

```text
Datasets/*.json
    │
    ▼
prepare_data_for_training.py
    │
    ├── Load all JSON files
    ├── Merge and deduplicate
    ├── Balance domain distribution
    ├── Compute quality scores
    ├── Add difficulty ratings
    │
    ▼
enhanced_training_data.json
    │
    ▼
train_enhanced.py
    │
    ├── TokenizedDataset
    ├── CurriculumSampler
    ├── DataLoader
    │
    ▼
ThetaModel.forward()
```

### Training Loop Flow

```text
for epoch in epochs:
    │
    ├── Update curriculum sampler
    ├── Update EMA epoch
    │
    ├── for batch in train_dataloader:
    │   ├── Forward pass (with R-Drop if enabled)
    │   ├── Compute loss (with label smoothing)
    │   ├── Backward pass (scaled for mixed precision)
    │   ├── Gradient accumulation
    │   ├── CPU offload: pre_step()
    │   ├── Optimizer step
    │   ├── CPU offload: post_step()
    │   ├── Update EMA weights
    │   └── Update dynamic curriculum tracker
    │
    ├── Validation phase
    │   ├── Apply EMA weights
    │   ├── Compute val_loss
    │   ├── Compute token accuracy
    │   └── Domain evaluation
    │
    ├── Update notifier metrics
    ├── Send epoch notification
    ├── Save checkpoint
    └── Check early stopping
```

## Memory Architecture

### GPU Memory Layout (RTX 3060 12GB)

```text
┌────────────────────────────────────────┐
│          GPU Memory (12GB)              │
├────────────────────────────────────────┤
│  Model Weights (FP16)      ~700MB      │
├────────────────────────────────────────┤
│  Gradients (FP16)          ~700MB      │
├────────────────────────────────────────┤
│  Activations (batch=3)     ~2GB        │
├────────────────────────────────────────┤
│  Optimizer (50% on GPU)    ~700MB      │
├────────────────────────────────────────┤
│  CUDA Overhead             ~1GB        │
├────────────────────────────────────────┤
│  Free                      ~7GB        │
└────────────────────────────────────────┘
```

### CPU Memory Layout

```text
┌────────────────────────────────────────┐
│          CPU Memory (32GB)              │
├────────────────────────────────────────┤
│  Optimizer States (50%)    ~700MB      │
├────────────────────────────────────────┤
│  EMA Shadow Weights        ~700MB      │
├────────────────────────────────────────┤
│  DataLoader Workers        ~2GB        │
├────────────────────────────────────────┤
│  Dataset (in memory)       ~1GB        │
├────────────────────────────────────────┤
│  OS + Other                ~5GB        │
├────────────────────────────────────────┤
│  Free                      ~22GB       │
└────────────────────────────────────────┘
```

## File Structure

```text
Theta AI/
├── src/
│   ├── __init__.py
│   ├── model/
│   │   ├── __init__.py
│   │   └── theta_model.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train_enhanced.py
│   │   ├── email_integration.py
│   │   ├── evaluate_model.py
│   │   └── optimize_for_inference.py
│   ├── data_processing/
│   │   ├── __init__.py
│   │   ├── knowledge_base_enhancer_main.py
│   │   └── [30+ processing scripts]
│   └── utils/
│       ├── __init__.py
│       ├── email_notifier.py
│       └── gpu_info.py
├── Datasets/
│   ├── [domain datasets]
│   └── enhanced_training_data.json
├── models/
│   └── theta_enhanced_YYYYMMDD/
├── documentation/
│   └── [this documentation]
├── train_overnight_enhanced.bat
├── requirements.txt
└── .env
```

## Dependencies

### Core

| Package | Version | Purpose |
|---------|---------|---------|
| torch | 2.0.1 | Deep learning framework |
| transformers | 4.30.2 | GPT-2 model and tokenizer |
| datasets | 2.13.0 | HuggingFace datasets |

### Training

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | 1.24.3 | Numerical operations |
| tqdm | 4.65.0 | Progress bars |
| colorama | 0.4.6 | Colored console output |
| tensorboard | 2.13.0 | Training visualization |

### Utilities

| Package | Version | Purpose |
|---------|---------|---------|
| python-dotenv | 1.0.0 | Environment variables |
| psutil | 5.9.5 | System monitoring |
| nltk | 3.8.1 | Text processing |
