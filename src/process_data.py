"""
Data Processing Script for Theta AI

This script processes the raw Frostline data and prepares it for training.
"""

import sys
from pathlib import Path

# Add the src directory to path to fix imports
sys.path.insert(0, str(Path(__file__).parent))
from data_processing.process_data import main, process_data

if __name__ == "__main__":
    main()
