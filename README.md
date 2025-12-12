# Theta AI

A GPT-2 based conversational AI training framework optimized for NVIDIA RTX 3060 (12GB VRAM). This repository contains everything needed to train, fine-tune, and run inference on Theta AI models.

## Features

- **Optimized Training Pipeline**: Gradient checkpointing, mixed precision (FP16), CPU offloading
- **Advanced Techniques**: Curriculum learning, R-Drop regularization, EMA, label smoothing
- **RTX 3060 Optimized**: Configured for 12GB VRAM with memory-efficient settings
- **Email Notifications**: Real-time training alerts with GPU stats and metric monitoring
- **Multi-domain Training**: Cybersecurity, programming, networking, data science, and more

## Quick Start

### Requirements

- **GPU**: NVIDIA RTX 3060 12GB (or similar)
- **CPU**: AMD Ryzen 5-5500 or equivalent
- **CUDA**: 11.8+
- **Python**: 3.8+

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/theta-ai.git
cd theta-ai

# Install dependencies (CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('stopwords')"

# Setup environment
cp .env.example .env
# Edit .env with your settings
```

### Download Datasets

```bash
# Human-like conversational data (28MB)
download_human_like_dpo.bat

# OpenAssistant dataset (6GB)
download_openassistant.bat

# OpenMath dataset (9GB, optional)
download_openmath_instruct.bat
```

### Training

```bash
# Full training pipeline (overnight recommended)
train_overnight_enhanced.bat
```

### Inference

```python
from src.model.theta_model import ThetaModel

model = ThetaModel.load("models/theta_enhanced_YYYYMMDD/theta_final")
response = model.generate("What is machine learning?", max_length=200)
print(response)
```

## Documentation

Full documentation is available in the [`documentation/`](documentation/) folder:

| Guide | Description |
|-------|-------------|
| [Installation](documentation/INSTALLATION.md) | Detailed setup instructions |
| [Quick Start](documentation/QUICKSTART.md) | Get training in 5 minutes |
| [Training Pipeline](documentation/TRAINING_PIPELINE.md) | Complete training system guide |
| [Datasets](documentation/DATASETS.md) | Dataset formats and creation |
| [Hyperparameters](documentation/HYPERPARAMETERS.md) | All configuration options |
| [RTX 3060 Optimizations](documentation/RTX3060_OPTIMIZATIONS.md) | GPU-specific tuning |
| [Email Notifications](documentation/EMAIL_NOTIFICATIONS.md) | Alert system setup |
| [Architecture](documentation/ARCHITECTURE.md) | System design overview |
| [API Reference](documentation/API_REFERENCE.md) | Code documentation |
| [Data Processing](documentation/DATA_PROCESSING.md) | Data preparation guide |
| [Model Config](documentation/MODEL_CONFIG.md) | Model settings |
| [Troubleshooting](documentation/TROUBLESHOOTING.md) | Common issues & fixes |

## Project Structure

```text
theta-ai/
├── src/
│   ├── model/              # Model architecture
│   ├── training/           # Training pipeline
│   ├── inference/          # Inference utilities
│   ├── data_processing/    # Dataset processing
│   └── utils/              # Email notifier, GPU info
├── Datasets/               # Training data (JSON)
├── models/                 # Saved checkpoints
├── documentation/          # Full documentation
├── train_overnight_enhanced.bat  # Main training script
├── prepare_data_for_training.py  # Data preparation
└── requirements.txt        # Dependencies
```

## Key Files

| File | Purpose |
|------|---------|
| `train_overnight_enhanced.bat` | Main training orchestration |
| `prepare_data_for_training.py` | Data preparation pipeline |
| `src/training/train_enhanced.py` | Core training logic |
| `src/model/theta_model.py` | Model architecture |
| `src/utils/email_notifier.py` | Training notifications |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.

## License

This project is for educational and research purposes.
