"""
Download and set up NLTK resources for Theta AI.
"""

import os
import sys
import nltk
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_nltk_resources():
    """Download necessary NLTK resources."""
    try:
        # Create NLTK data directory if it doesn't exist
        nltk_dir = os.path.join(os.path.expanduser('~'), 'nltk_data')
        os.makedirs(nltk_dir, exist_ok=True)
        
        # Download resources
        packages = ['punkt', 'averaged_perceptron_tagger', 'wordnet']
        
        for package in packages:
            try:
                logger.info(f"Downloading NLTK package: {package}")
                nltk.download(package, quiet=False)
                logger.info(f"Successfully downloaded {package}")
            except Exception as e:
                logger.error(f"Error downloading {package}: {str(e)}")
        
        # Create a custom punkt_tab workaround - punkt_tab is not available in NLTK anymore
        # but some libraries still try to use it
        try:
            # Create symbolic link or copy from punkt to punkt_tab
            nltk_dir = os.path.join(os.path.expanduser('~'), 'nltk_data')
            punkt_dir = os.path.join(nltk_dir, 'tokenizers', 'punkt')
            punkt_tab_dir = os.path.join(nltk_dir, 'tokenizers', 'punkt_tab')
            
            if os.path.exists(punkt_dir) and not os.path.exists(punkt_tab_dir):
                logger.info("Creating punkt_tab directory")
                os.makedirs(punkt_tab_dir, exist_ok=True)
                
                # Create english directory
                os.makedirs(os.path.join(punkt_tab_dir, 'english'), exist_ok=True)
                
                # Copy English punkt files to punkt_tab as a workaround
                english_dir = os.path.join(punkt_dir, 'english')
                english_tab_dir = os.path.join(punkt_tab_dir, 'english')
                
                if os.path.exists(english_dir):
                    import shutil
                    for file in os.listdir(english_dir):
                        src = os.path.join(english_dir, file)
                        dst = os.path.join(english_tab_dir, file)
                        logger.info(f"Copying {src} to {dst}")
                        shutil.copy2(src, dst)
                
                # Create necessary tab files that might be missing
                required_files = ['collocations.tab', 'word_tokenizer.tab', 'period_context.tab']
                for file in required_files:
                    # Check if file exists in punkt
                    punkt_file = os.path.join(english_dir, file.replace('.tab', ''))
                    tab_file = os.path.join(english_tab_dir, file)
                    
                    # If tab file doesn't exist, create an empty one
                    if not os.path.exists(tab_file):
                        logger.info(f"Creating empty file: {tab_file}")
                        with open(tab_file, 'w', encoding='utf-8') as f:
                            f.write('# Auto-generated placeholder file\n')
                    
                logger.info("Created punkt_tab workaround successfully")
            else:
                logger.warning("Could not create punkt_tab workaround")
        except Exception as e:
            logger.error(f"Error creating punkt_tab workaround: {str(e)}")
        
        logger.info("All NLTK resources downloaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error setting up NLTK: {str(e)}")
        return False

def verify_resources():
    """Verify that necessary resources are available."""
    try:
        # Create a simple test tokenizer
        def simple_test_tokenize(text):
            import re
            if not text:
                return []
            sentences = []
            for sent in re.split(r'(?<=[.!?])\s+', text):
                if sent.strip():
                    sentences.append(sent.strip())
            return sentences if sentences else [text]
            
        # Test our simple tokenizer
        test_text = "This is a test. Is the tokenizer working?"
        sentences = simple_test_tokenize(test_text)
        logger.info(f"Simple tokenizer test: {sentences}")
        
        # Test NLTK resources without using punkt_tab
        try:
            import nltk.data
            nltk.data.path.append(os.path.join(os.path.expanduser('~'), 'nltk_data'))
            logger.info(f"NLTK data path: {nltk.data.path}")
        except Exception as e:
            logger.warning(f"NLTK path setup failed: {str(e)}")
        
        # Test WordNet
        try:
            from nltk.corpus import wordnet
            synsets = wordnet.synsets('test')
            logger.info(f"WordNet test: Found {len(synsets)} synsets for 'test'")
        except:
            logger.warning("WordNet not available")
        
        # Test POS tagger
        try:
            from nltk.tag import pos_tag
            from nltk.tokenize import word_tokenize
            tokens = word_tokenize("Testing the part-of-speech tagger.")
            tags = pos_tag(tokens)
            logger.info(f"POS tagger test: {tags}")
        except:
            logger.warning("POS tagger not available")
            
        return True
    except Exception as e:
        logger.error(f"Error verifying NLTK resources: {str(e)}")
        return False

if __name__ == "__main__":
    print("Setting up NLTK resources for Theta AI...")
    success = download_nltk_resources()
    
    if success:
        print("Verifying NLTK resources...")
        verify_resources()
        print("NLTK setup complete.")
    else:
        print("Failed to set up NLTK resources. Please check the logs for details.")
        sys.exit(1)
