# Theta AI Documentation

Welcome to the Theta AI documentation. This folder contains comprehensive guides for training, deploying, and extending the Theta AI language model.

## Quick Start

1. [Installation Guide](INSTALLATION.md) - Set up your environment
2. [Quick Start Training](QUICKSTART.md) - Start training in 5 minutes
3. [Dataset Guide](DATASETS.md) - Understanding and preparing data

## Core Documentation

| Document | Description |
|----------|-------------|
| [Architecture](ARCHITECTURE.md) | System design and components |
| [Training Pipeline](TRAINING_PIPELINE.md) | Complete training documentation |
| [Model Configuration](MODEL_CONFIG.md) | Hyperparameters and optimization |
| [Email Notifications](EMAIL_NOTIFICATIONS.md) | Training alerts and monitoring |
| [Data Processing](DATA_PROCESSING.md) | Dataset creation and preprocessing |

## Advanced Topics

| Document | Description |
|----------|-------------|
| [RTX 3060 Optimizations](RTX3060_OPTIMIZATIONS.md) | GPU-specific tuning |
| [Hyperparameter Reference](HYPERPARAMETERS.md) | All configurable parameters |
| [Troubleshooting](TROUBLESHOOTING.md) | Common issues and solutions |
| [API Reference](API_REFERENCE.md) | Code documentation |

## Project Structure

```
Theta AI/
├── src/                          # Source code
│   ├── training/                 # Training pipeline
│   │   ├── train_enhanced.py     # Main training script
│   │   ├── email_integration.py  # Email notifications
│   │   ├── cleanup_utils.py      # Checkpoint management
│   │   └── evaluate_model.py     # Model evaluation
│   ├── model/                    # Model architecture
│   │   └── theta_model.py        # ThetaModel class
│   ├── data_processing/          # Data preparation
│   └── utils/                    # Utilities
├── Datasets/                     # Training data (JSON)
├── documentation/                # This folder
├── train_overnight_enhanced.bat  # Main training script
├── requirements.txt              # Python dependencies
└── .env.example                  # Environment template
```

## License

This project is provided for educational and research purposes.

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) in the root directory.
