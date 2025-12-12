"""
Interface Script for Theta AI

This script provides an interactive interface to the Theta AI model.
"""

import sys
import argparse
from pathlib import Path

# Add the src directory to the path to import modules
sys.path.append(str(Path(__file__).parent))
from interface.theta_interface import main

if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description="Run the Theta AI interface")
    
    # Add arguments
    parser.add_argument("--model_path", type=str, 
                        default=str(Path(__file__).parent.parent / "models" / "theta_final"),
                        help="Path to the trained model directory")
    parser.add_argument("--model_type", type=str, default="gpt2",
                        help="Model type (gpt2, bert-qa)")
    parser.add_argument("--model_name", type=str, default="gpt2",
                        help="Specific model name/version")
    parser.add_argument("--dataset_path", type=str,
                        default=str(Path(__file__).parent.parent / "Datasets" / "processed_data.json"),
                        help="Path to the dataset for retrieval-based answers")
    
    # Parse arguments and run main
    args = parser.parse_args()
    main()
