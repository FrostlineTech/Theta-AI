"""
Generate all enhanced datasets and process them for training.

This script runs all data generation modules to create a full dataset
of at least 10,000 examples for training Theta AI.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
project_root = str(Path(__file__).resolve().parent.parent)
sys.path.append(project_root)

# Import data generation modules
from src.data_processing.create_conversation_dataset import main as create_conversations
from src.data_processing.expand_conversations import main as expand_conversations
from src.data_processing.generate_technical_qa import main as generate_technical_qa
from src.data_processing.generate_tutorials import main as generate_tutorials
from src.data_processing.generate_problem_solutions import main as generate_problem_solutions
from src.data_processing.generate_multi_turn_dataset import generate_multi_turn_dataset
from src.data_processing.process_data import main as process_data

def run_generation_task(name, function):
    """Run a generation task and handle exceptions"""
    logger.info(f"Starting {name} generation...")
    try:
        result = function()
        logger.info(f"Completed {name} generation: {result}")
        return result
    except Exception as e:
        logger.error(f"Error in {name} generation: {str(e)}")
        return None

def run_openwebtext_generation():
    """Run the OpenWebText generation script as a separate process"""
    logger.info("Starting OpenWebText data generation...")
    try:
        cmd = [sys.executable, "-m", "src.data_processing.direct_openwebtext", "--sample_size", "7000"]
        result = subprocess.run(cmd, cwd=project_root, check=True)
        logger.info(f"OpenWebText generation completed with exit code {result.returncode}")
        return "Datasets/openwebtext_processed.json"
    except subprocess.CalledProcessError as e:
        logger.error(f"OpenWebText generation failed with exit code {e.returncode}")
        return None
    except Exception as e:
        logger.error(f"Error running OpenWebText generation: {str(e)}")
        return None

def main():
    """Generate all datasets and process them"""
    print("\n======== THETA AI DATASET ENHANCEMENT ========")
    print("Generating enhanced datasets for training...\n")
    
    # First, ensure we have the basic conversation datasets
    run_generation_task("Conversation flows", create_conversations)
    
    # Generate all enhanced datasets in parallel
    print("\nGenerating enhanced datasets (this may take a few minutes)...\n")
    
    # Run data generation scripts
    tasks = [
        ("Expanded conversations", expand_conversations),
        ("Technical Q&A", generate_technical_qa),
        ("Tutorials", generate_tutorials),
        ("Problem-solution pairs", generate_problem_solutions),
        ("Multi-turn conversations", lambda: generate_multi_turn_dataset(1000)),
        ("OpenWebText", run_openwebtext_generation)
    ]
    
    results = []
    for name, function in tasks:
        result = run_generation_task(name, function)
        results.append((name, result))
    
    # Process all datasets into final combined file
    print("\nProcessing all datasets into combined training data...")
    process_data()
    
    # Print summary
    print("\n======== DATASET ENHANCEMENT COMPLETE ========")
    print("Summary of generated datasets:")
    for name, result in results:
        status = "SUCCESS" if result else "FAILED"
        print(f"- {name}: {status}")
    print("\nAll datasets have been combined into Datasets/processed_data.json")
    print("You can now run training with train_overnight_enhanced.bat")
    
if __name__ == "__main__":
    main()
