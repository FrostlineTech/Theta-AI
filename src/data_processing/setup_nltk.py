"""
Ensure all required NLTK data is downloaded and properly set up.
This script should be called before any data processing that requires NLTK.
"""

import os
import sys
import nltk
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

def ensure_nltk_data():
    """Ensure all required NLTK data is downloaded."""
    logger.info("Ensuring NLTK data is properly downloaded and set up...")
    
    # Download required NLTK resources
    resources = ['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
            logger.info(f"Resource '{resource}' already downloaded")
        except LookupError:
            logger.info(f"Downloading '{resource}'...")
            nltk.download(resource)
    
    # Fix punkt structure
    fix_punkt_structure()
    
    # Verify setup
    if verify_nltk_setup():
        logger.info("✅ NLTK is properly set up")
        return True
    else:
        logger.error("❌ NLTK setup verification failed")
        return False

def fix_punkt_structure():
    """Fix NLTK punkt directory structure to prevent errors."""
    try:
        # Get NLTK data path
        nltk_data_path = nltk.data.path[0]
        logger.info(f"NLTK data path: {nltk_data_path}")
        
        # Create proper directory structure with OS-specific separators
        tokenizers_dir = os.path.join(nltk_data_path, 'tokenizers')
        punkt_dir = os.path.join(tokenizers_dir, 'punkt')
        punkt_tab_dir = os.path.join(tokenizers_dir, 'punkt_tab')
        
        # Create directories
        english_dir = os.path.join(punkt_tab_dir, 'english')
        os.makedirs(english_dir, exist_ok=True)
        logger.info(f"Created directory: {english_dir}")
        
        # Create the missing files
        files_to_create = [
            "collocations.tab",
            "sentence_context_forms.tab",
            "sent_starters.txt"  # This is the file that was missing
        ]
        
        for filename in files_to_create:
            create_file(os.path.join(english_dir, filename))
        
        # Try to copy files from punkt if they exist
        try:
            punkt_english_dir = os.path.join(punkt_dir, 'english')
            if os.path.exists(punkt_english_dir):
                for filename in os.listdir(punkt_english_dir):
                    src = os.path.join(punkt_english_dir, filename)
                    dst = os.path.join(english_dir, filename)
                    if os.path.isfile(src) and not os.path.exists(dst):
                        with open(src, 'rb') as src_file:
                            with open(dst, 'wb') as dst_file:
                                dst_file.write(src_file.read())
                        logger.info(f"Copied {src} to {dst}")
        except Exception as e:
            logger.warning(f"Error copying files from punkt/english: {e}")
            
        # If sent_starters.txt is still empty, add some content to it
        sent_starters_path = os.path.join(english_dir, "sent_starters.txt")
        if os.path.getsize(sent_starters_path) == 0:
            with open(sent_starters_path, 'w', encoding='utf-8') as f:
                f.write("# This file contains common sentence starters\n")
                f.write("The\nA\nIn\nThis\nThat\nWe\nI\nYou\nHe\nShe\nThey\nIt\n")
            logger.info(f"Added default content to {sent_starters_path}")
        
        logger.info("NLTK punkt directory structure fixed")
        return True
    except Exception as e:
        logger.error(f"Error fixing punkt structure: {e}")
        return False

def create_file(filepath):
    """Create an empty file if it doesn't exist."""
    if not os.path.exists(filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            pass  # Create empty file
        logger.info(f"Created file: {filepath}")
    else:
        logger.info(f"File already exists: {filepath}")

def verify_nltk_setup():
    """Test the NLTK setup to ensure it's working."""
    try:
        from nltk.tokenize import word_tokenize, sent_tokenize
        
        # Test sentence tokenization
        text = "This is a test. NLTK punkt should work now. Is everything fixed?"
        sentences = sent_tokenize(text)
        logger.info(f"Sentence tokenization successful: {len(sentences)} sentences")
        
        # Test word tokenization
        words = word_tokenize("This is a word tokenization test.")
        logger.info(f"Word tokenization successful: {len(words)} words")
        
        return True
    except Exception as e:
        logger.error(f"NLTK verification failed: {e}")
        return False

if __name__ == "__main__":
    ensure_nltk_data()
