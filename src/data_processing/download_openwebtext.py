"""
Download and process the OpenWebText dataset for Theta AI training.

This script downloads the OpenWebText dataset from Hugging Face,
processes it into chunks suitable for Theta AI training, and
creates a JSON file compatible with the existing data pipeline.
"""

import os
import json
from pathlib import Path
import argparse
import logging
import random
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets
import torch.multiprocessing as mp

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OpenWebTextProcessor:
    """Class for downloading and processing OpenWebText dataset."""
    
    def __init__(self, output_dir=None, sample_size=None, chunk_size=512, seed=42):
        """
        Initialize the processor.
        
        Args:
            output_dir: Directory to save processed data
            sample_size: Number of examples to sample (None = all)
            chunk_size: Size of text chunks
            seed: Random seed for reproducibility
        """
        # Get project root
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.datasets_dir = self.project_root / "Datasets"
        
        # Set output directory
        if output_dir is None:
            self.output_dir = self.datasets_dir
        else:
            self.output_dir = Path(output_dir)
            
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.sample_size = sample_size
        self.chunk_size = chunk_size
        self.seed = seed
        random.seed(seed)
        
    def download_dataset(self, cache_dir=None):
        """
        Download the OpenWebText dataset from Hugging Face.
        
        Args:
            cache_dir: Directory to cache the downloaded dataset
            
        Returns:
            The downloaded dataset
        """
        logger.info("Downloading OpenWebText dataset from Hugging Face...")
        try:
            # If cache_dir is None, Hugging Face will use its default cache location
            # Adding trust_remote_code=True as required by this dataset
            dataset = load_dataset("Skylion007/openwebtext", cache_dir=cache_dir, trust_remote_code=True)
            logger.info(f"Dataset downloaded successfully. Contains {len(dataset['train'])} examples.")
            return dataset
        except Exception as e:
            logger.error(f"Error downloading dataset: {e}")
            raise
            
    def sample_dataset(self, dataset, sample_size=None):
        """
        Sample from the dataset to reduce size if needed.
        
        Args:
            dataset: The full dataset
            sample_size: Number of examples to sample
            
        Returns:
            Sampled dataset
        """
        if sample_size is None or sample_size >= len(dataset["train"]):
            logger.info("Using full dataset")
            return dataset
            
        logger.info(f"Sampling {sample_size} examples from dataset...")
        
        # Ensure deterministic sampling with fixed seed
        sampled_indices = random.sample(range(len(dataset["train"])), sample_size)
        sampled_dataset = dataset["train"].select(sampled_indices)
        
        # Return as dataset dict for consistency
        return {"train": sampled_dataset}
        
    def process_text_to_qa_format(self, dataset, output_file=None, max_examples_per_file=10000):
        """
        Process raw text into question-answer format suitable for Theta AI.
        
        Args:
            dataset: The dataset to process
            output_file: Path to save processed data
            max_examples_per_file: Maximum examples per output file
            
        Returns:
            List of processed QA pairs
        """
        if output_file is None:
            output_file = self.output_dir / "openwebtext_processed.json"
            
        logger.info("Processing OpenWebText into QA format...")
        
        # Get the texts
        texts = dataset["train"]["text"]
        
        # Container for QA pairs
        qa_pairs = []
        
        # File counter for multiple files
        file_counter = 1
        examples_in_current_file = 0
        current_output_file = output_file
        
        # Ensure the file extension is properly handled
        base_name = output_file.stem
        extension = output_file.suffix
        
        # Process texts with progress bar
        for i, text in enumerate(tqdm(texts, desc="Processing texts")):
            # Skip very short texts
            if len(text) < 100:
                continue
                
            # Split text into paragraphs
            paragraphs = [p for p in text.split("\n\n") if len(p.strip()) > 50]
            
            if len(paragraphs) < 2:
                continue
                
            # Create QA pair
            # Use first paragraph as context/question, rest as answer
            question = f"Can you elaborate on this topic: {paragraphs[0].strip()}"
            answer = "\n\n".join(paragraphs[1:]).strip()
            
            # Limit answer length to maintain quality
            if len(answer) > 2000:
                answer = answer[:2000] + "..."
                
            qa_pair = {
                "question": question,
                "answer": answer
            }
            
            qa_pairs.append(qa_pair)
            examples_in_current_file += 1
            
            # If we've reached the maximum examples per file, save and reset
            if max_examples_per_file is not None and examples_in_current_file >= max_examples_per_file:
                self._save_qa_pairs(qa_pairs, current_output_file)
                qa_pairs = []
                
                # Prepare for next file
                file_counter += 1
                examples_in_current_file = 0
                current_output_file = output_file.parent / f"{base_name}_{file_counter}{extension}"
                
            # Status update every 5000 examples
            if (i + 1) % 5000 == 0:
                logger.info(f"Processed {i + 1}/{len(texts)} texts, created {len(qa_pairs)} QA pairs")
                
        # Save any remaining QA pairs
        if qa_pairs:
            self._save_qa_pairs(qa_pairs, current_output_file)
            
        logger.info(f"Processing complete. Created {file_counter} files with {len(qa_pairs)} QA pairs in the last file.")
        return current_output_file
        
    def _save_qa_pairs(self, qa_pairs, output_file):
        """Save QA pairs to a JSON file."""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(qa_pairs)} QA pairs to {output_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving QA pairs: {e}")
            return False
            
    def process_dataset(self, cache_dir=None, output_file=None, max_examples=None):
        """
        Main method to download and process the dataset.
        
        Args:
            cache_dir: Directory to cache the downloaded dataset
            output_file: Path to save processed data
            max_examples: Maximum number of examples to process
            
        Returns:
            Path to the processed data file
        """
        # If max_examples is provided, override sample_size
        sample_size = max_examples if max_examples is not None else self.sample_size
        
        # Download dataset
        dataset = self.download_dataset(cache_dir)
        
        # Sample dataset if needed
        if sample_size is not None:
            dataset = self.sample_dataset(dataset, sample_size)
            
        # Process dataset
        processed_file = self.process_text_to_qa_format(dataset, output_file)
        
        return processed_file
        
def main():
    """Main function to download and process OpenWebText."""
    parser = argparse.ArgumentParser(description="Download and process OpenWebText dataset")
    
    parser.add_argument("--output_dir", type=str, default=None, 
                        help="Directory to save processed data")
    parser.add_argument("--cache_dir", type=str, default=None, 
                        help="Directory to cache the downloaded dataset")
    parser.add_argument("--output_file", type=str, default=None, 
                        help="Path to save processed data")
    parser.add_argument("--sample_size", type=int, default=10000, 
                        help="Number of examples to sample (default: 10000, use -1 for all)")
    parser.add_argument("--chunk_size", type=int, default=512, 
                        help="Size of text chunks")
    parser.add_argument("--max_examples_per_file", type=int, default=5000, 
                        help="Maximum examples per output file")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed")
                        
    args = parser.parse_args()
    
    # Convert -1 to None for sample_size
    sample_size = None if args.sample_size == -1 else args.sample_size
    
    # Initialize processor
    processor = OpenWebTextProcessor(
        output_dir=args.output_dir,
        sample_size=sample_size,
        chunk_size=args.chunk_size,
        seed=args.seed
    )
    
    # Process dataset
    processor.process_dataset(
        cache_dir=args.cache_dir,
        output_file=args.output_file,
        max_examples=sample_size
    )
    
if __name__ == "__main__":
    # Set the start method to 'spawn' to avoid issues with multiprocessing
    if __name__ == '__main__':
        mp.set_start_method('spawn')
    main()
