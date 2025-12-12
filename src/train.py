"""
Training Script for Theta AI

This script trains the Theta AI model using the processed data.
"""

import os
import sys
import argparse
from pathlib import Path

# Add the src directory to the path to import modules
sys.path.append(str(Path(__file__).parent))
from training.train_theta import main

if __name__ == "__main__":
    # Set default paths for convenience
    parser = argparse.ArgumentParser(description="Train the Theta AI model")
    
    # Add arguments from the original parser
    parser.add_argument("--model_type", type=str, default="gpt2", help="Model type (gpt2, bert-qa)")
    parser.add_argument("--model_name", type=str, default="gpt2", help="Specific model name/version")
    parser.add_argument("--data_path", type=str, default=str(Path(__file__).parent.parent / "Datasets" / "processed_data.json"), 
                        help="Path to the processed data")
    parser.add_argument("--output_dir", type=str, default=str(Path(__file__).parent.parent / "models"), 
                        help="Directory to save model checkpoints")
    parser.add_argument("--batch_size", type=int, default=2, help="Training batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, 
                      help="Number of steps to accumulate gradients before performing a backward/update pass")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    
    # Parse arguments and pass to main
    args = parser.parse_args()
    main(args)
