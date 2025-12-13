#!/usr/bin/env python
"""
Targeted fine-tuning script for Theta AI.
Loads configuration from a JSON file and runs training with reduced regularization.

Usage:
    python src/training/run_finetune.py --config models/theta_enhanced_20251212/finetune_config.json
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.training.train_enhanced import train


class ConfigNamespace:
    """Convert dict to namespace for argparse compatibility."""
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)


def main():
    parser = argparse.ArgumentParser(description="Run fine-tuning with config file")
    parser.add_argument("--config", type=str, required=True, 
                        help="Path to JSON config file")
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print("=" * 60)
    print("Theta AI - Targeted Fine-tuning")
    print("=" * 60)
    print(f"\nConfig: {config_path}")
    print(f"Model: {config.get('model_name', 'N/A')}")
    print(f"Output: {config.get('output_dir', 'N/A')}")
    print(f"\nKey settings (reduced regularization):")
    print(f"  - Learning rate: {config.get('learning_rate', 'N/A')}")
    print(f"  - Label smoothing: {config.get('label_smoothing', 'N/A')}")
    print(f"  - R-Drop alpha: {config.get('rdrop_alpha', 'N/A')}")
    print(f"  - EMA decay: {config.get('ema_decay', 'N/A')}")
    print(f"  - Epochs: {config.get('epochs', 'N/A')}")
    print("=" * 60)
    
    # Convert to namespace and run training
    training_args = ConfigNamespace(config)
    train(training_args)


if __name__ == "__main__":
    main()
