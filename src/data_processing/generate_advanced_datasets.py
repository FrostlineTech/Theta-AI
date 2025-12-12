"""
Generate advanced datasets for Theta AI training.

This module combines various data sources to create comprehensive datasets
for training Theta AI with advanced conversational capabilities.
"""

import os
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import dataset generators
from src.data_processing.stack_exchange_processor import main as generate_stack_exchange
from src.data_processing.github_issue_conversations import main as generate_github_conversations
from src.data_processing.technical_documentation import main as generate_technical_documentation
from src.data_processing.tutorial_dialogues import main as generate_tutorial_dialogues

def generate_advanced_datasets(output_dir="./Datasets", sample_sizes=None):
    """
    Generate all advanced datasets.
    
    Args:
        output_dir (str): Output directory for datasets
        sample_sizes (dict): Sample sizes for each dataset
        
    Returns:
        dict: Paths to generated datasets
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set default sample sizes if not provided
    if sample_sizes is None:
        sample_sizes = {
            "stack_exchange": 10000,
            "github_conversations": 2000,
            "technical_documentation": 5000,
            "tutorial_dialogues": 1000
        }
    
    results = {}
    
    # Generate Stack Exchange dataset
    logger.info("Generating Stack Exchange dataset...")
    try:
        output_file = generate_stack_exchange(
            sample_size=sample_sizes["stack_exchange"],
            output_dir=output_dir
        )
        results["stack_exchange"] = output_file
        logger.info(f"Stack Exchange dataset generated successfully: {output_file}")
    except Exception as e:
        logger.error(f"Error generating Stack Exchange dataset: {str(e)}")
    
    # Generate GitHub Conversations dataset
    logger.info("Generating GitHub Conversations dataset...")
    try:
        output_file = generate_github_conversations(
            sample_size=sample_sizes["github_conversations"],
            output_dir=output_dir
        )
        results["github_conversations"] = output_file
        logger.info(f"GitHub Conversations dataset generated successfully: {output_file}")
    except Exception as e:
        logger.error(f"Error generating GitHub Conversations dataset: {str(e)}")
    
    # Generate Technical Documentation dataset
    logger.info("Generating Technical Documentation dataset...")
    try:
        output_file = generate_technical_documentation(
            sample_size=sample_sizes["technical_documentation"],
            output_dir=output_dir
        )
        results["technical_documentation"] = output_file
        logger.info(f"Technical Documentation dataset generated successfully: {output_file}")
    except Exception as e:
        logger.error(f"Error generating Technical Documentation dataset: {str(e)}")
    
    # Generate Tutorial Dialogues dataset
    logger.info("Generating Tutorial Dialogues dataset...")
    try:
        output_file = generate_tutorial_dialogues(
            sample_size=sample_sizes["tutorial_dialogues"],
            output_dir=output_dir
        )
        results["tutorial_dialogues"] = output_file
        logger.info(f"Tutorial Dialogues dataset generated successfully: {output_file}")
    except Exception as e:
        logger.error(f"Error generating Tutorial Dialogues dataset: {str(e)}")
    
    return results

def main():
    """Main entry point for generating all advanced datasets."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate advanced datasets for Theta AI")
    parser.add_argument("--output-dir", default="./Datasets", help="Output directory for datasets")
    parser.add_argument("--stack-exchange-size", type=int, default=10000, help="Sample size for Stack Exchange dataset")
    parser.add_argument("--github-size", type=int, default=2000, help="Sample size for GitHub Conversations dataset")
    parser.add_argument("--docs-size", type=int, default=5000, help="Sample size for Technical Documentation dataset")
    parser.add_argument("--tutorial-size", type=int, default=1000, help="Sample size for Tutorial Dialogues dataset")
    args = parser.parse_args()
    
    # Set sample sizes
    sample_sizes = {
        "stack_exchange": args.stack_exchange_size,
        "github_conversations": args.github_size,
        "technical_documentation": args.docs_size,
        "tutorial_dialogues": args.tutorial_size
    }
    
    # Generate datasets
    results = generate_advanced_datasets(args.output_dir, sample_sizes)
    
    # Print summary
    print("\n===== Advanced Dataset Generation Summary =====")
    for dataset_name, output_file in results.items():
        if output_file:
            print(f"✓ {dataset_name}: {output_file}")
        else:
            print(f"✗ {dataset_name}: Failed")
    
    print("\nAll datasets have been saved to:", args.output_dir)
    print("You can now include these in your training pipeline.")

if __name__ == "__main__":
    main()
