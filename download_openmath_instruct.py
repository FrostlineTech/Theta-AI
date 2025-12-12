"""
Download script for NVIDIA OpenMathInstruct-1 dataset from HuggingFace.
This dataset provides high-quality math instruction data for training.

Dataset URL: https://huggingface.co/datasets/nvidia/OpenMathInstruct-1

Usage:
    python download_openmath_instruct.py [--sample_size N] [--output_dir PATH]
"""

import os
import json
import argparse
import logging
from pathlib import Path
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_openmath_instruct(output_dir: str = "G:/Theta AI/Datasets", 
                                sample_size: int = None,
                                cache_dir: str = "G:/Theta AI/cache"):
    """
    Download and save the NVIDIA OpenMathInstruct-1 dataset.
    
    Args:
        output_dir: Directory to save the processed dataset
        sample_size: Number of samples to download (None = full dataset)
        cache_dir: Directory to cache the HuggingFace dataset
        
    Returns:
        Path to the saved dataset file
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Please install the 'datasets' library: pip install datasets")
        raise ImportError("datasets library is required. Install with: pip install datasets")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_file = output_path / "openmath_instruct_1.json"
    
    logger.info("=" * 60)
    logger.info("NVIDIA OpenMathInstruct-1 Dataset Downloader")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Cache directory: {cache_dir}")
    logger.info(f"Sample size: {'Full dataset' if sample_size is None else sample_size}")
    logger.info("=" * 60)
    
    # Set cache directory
    os.environ['HF_HOME'] = cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    
    try:
        logger.info("Loading NVIDIA OpenMathInstruct-1 dataset from HuggingFace...")
        logger.info("This may take a while for the first download...")
        
        # Load the dataset
        # The OpenMathInstruct-1 dataset has 'train' split
        dataset = load_dataset(
            "nvidia/OpenMathInstruct-1",
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        
        logger.info(f"Dataset loaded successfully!")
        logger.info(f"Available splits: {list(dataset.keys())}")
        
        # Get the training data
        train_data = dataset['train']
        total_examples = len(train_data)
        logger.info(f"Total examples in dataset: {total_examples}")
        
        # Sample if requested
        if sample_size is not None and sample_size < total_examples:
            logger.info(f"Sampling {sample_size} examples from {total_examples} total...")
            indices = list(range(sample_size))
            train_data = train_data.select(indices)
        
        # Convert to our format while preserving original structure
        logger.info("Processing dataset entries...")
        processed_entries = []
        
        # Get column names to understand the structure
        column_names = train_data.column_names
        logger.info(f"Dataset columns: {column_names}")
        
        for idx, example in enumerate(tqdm(train_data, desc="Processing")):
            # Preserve the original structure as much as possible
            # OpenMathInstruct-1 typically has: problem, generated_solution, expected_answer
            entry = {
                "problem": example.get("problem", example.get("question", "")),
                "solution": example.get("generated_solution", example.get("solution", "")),
                "expected_answer": example.get("expected_answer", example.get("answer", "")),
                "source": "nvidia/OpenMathInstruct-1",
                "domain": "mathematics",
                "idx": idx
            }
            
            # Add any other fields from the original dataset
            for col in column_names:
                if col not in ["problem", "generated_solution", "expected_answer", "question", "solution", "answer"]:
                    if col in example:
                        entry[col] = example[col]
            
            processed_entries.append(entry)
        
        # Save the dataset
        dataset_output = {
            "metadata": {
                "source": "nvidia/OpenMathInstruct-1",
                "url": "https://huggingface.co/datasets/nvidia/OpenMathInstruct-1",
                "description": "High-quality math instruction dataset from NVIDIA",
                "total_examples": len(processed_entries),
                "format": "problem-solution pairs",
                "domain": "mathematics",
                "columns": column_names
            },
            "entries": processed_entries
        }
        
        logger.info(f"Saving dataset to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset_output, f, indent=2, ensure_ascii=False)
        
        # Calculate file size
        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        
        logger.info("=" * 60)
        logger.info("Download complete!")
        logger.info(f"Saved {len(processed_entries)} examples to {output_file}")
        logger.info(f"File size: {file_size_mb:.2f} MB")
        logger.info("=" * 60)
        
        # Print sample entry for verification
        if processed_entries:
            logger.info("\nSample entry:")
            sample = processed_entries[0]
            for key, value in sample.items():
                if isinstance(value, str) and len(value) > 200:
                    value = value[:200] + "..."
                logger.info(f"  {key}: {value}")
        
        return str(output_file)
        
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Download NVIDIA OpenMathInstruct-1 dataset from HuggingFace"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="G:/Theta AI/Datasets",
        help="Directory to save the dataset"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Number of samples to download (default: full dataset)"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="G:/Theta AI/cache",
        help="Directory to cache HuggingFace downloads"
    )
    
    args = parser.parse_args()
    
    try:
        output_path = download_openmath_instruct(
            output_dir=args.output_dir,
            sample_size=args.sample_size,
            cache_dir=args.cache_dir
        )
        print(f"\nDataset saved to: {output_path}")
        print("You can now run data_processor.py to include this dataset in training.")
    except Exception as e:
        print(f"\nError: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure you have internet connection")
        print("2. Install required packages: pip install datasets tqdm")
        print("3. Check if HuggingFace is accessible")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
