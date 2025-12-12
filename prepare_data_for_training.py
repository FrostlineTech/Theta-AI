#prepare_data_for_training.py

"""
Enhanced data preparation script for Theta AI training.
Implements all 5 recommended enhancements:
1. Balance Domain-Specific Data
2. Data Quality Check
3. Technical Context Enhancement
4. Curriculum Learning
5. Augment with Synthetic Examples

Usage:
    python prepare_data_for_training.py

Output:
    - Creates enhanced_training_data.json ready for training
"""

import os
import json
import logging
import random
import re
import torch
import nltk
import numpy as np
import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Any, Tuple, Set
import sys
from tqdm import tqdm

def load_openmath_instruct_dataset(datasets_dir, max_samples=10000):
    """
    Load the NVIDIA OpenMathInstruct-1 Dataset for mathematics instruction training.
    This preserves the problem-solution format while making it compatible with training.
    
    The dataset is massive (8.4GB, 7.3M entries) so we sample a subset.
    """
    import random
    
    math_examples = []
    openmath_paths = [
        Path(datasets_dir) / "diverse_curriculum" / "openmath_instruct_1.json",  # Pre-processed sampled version
        Path(datasets_dir) / "openmath_instruct_1.json",  # Main path (large file)
    ]
    
    loaded = False
    for openmath_path in openmath_paths:
        if openmath_path.exists():
            try:
                file_size = openmath_path.stat().st_size
                file_size_mb = file_size / (1024 * 1024)
                
                # If file is small (pre-processed), load directly
                if file_size_mb < 100:
                    logger.info(f"Loading pre-processed OpenMathInstruct-1 from {openmath_path} ({file_size_mb:.1f} MB)")
                    with open(openmath_path, 'r', encoding='utf-8') as f:
                        openmath_data = json.load(f)
                    
                    if isinstance(openmath_data, list):
                        for entry in openmath_data:
                            if isinstance(entry, dict):
                                problem = entry.get("problem", entry.get("question", ""))
                                solution = entry.get("solution", entry.get("generated_solution", entry.get("answer", "")))
                                expected_answer = entry.get("expected_answer", "")
                                
                                if problem and solution:
                                    math_example = {
                                        "question": problem,
                                        "answer": solution,
                                        "domain": "mathematics",
                                        "source": "nvidia/OpenMathInstruct-1",
                                        "expected_answer": expected_answer,
                                        "original_format": "problem-solution"
                                    }
                                    math_examples.append(math_example)
                    
                    logger.info(f"Loaded {len(math_examples)} examples from pre-processed OpenMathInstruct-1")
                    loaded = True
                    break
                
                # Large file - use streaming/chunked loading
                logger.info(f"Loading sampled subset from large OpenMathInstruct-1 ({file_size_mb:.1f} MB, max {max_samples} samples)...")
                
                try:
                    # Try ijson for streaming if available
                    import ijson
                    entries = []
                    with open(openmath_path, 'rb') as f:
                        parser = ijson.items(f, 'entries.item')
                        for i, entry in enumerate(parser):
                            if i < max_samples * 3:
                                entries.append(entry)
                            else:
                                j = random.randint(0, i)
                                if j < len(entries):
                                    entries[j] = entry
                            if i % 500000 == 0 and i > 0:
                                logger.info(f"  Processed {i:,} entries...")
                    
                    if len(entries) > max_samples:
                        entries = random.sample(entries, max_samples)
                    
                    for entry in entries:
                        problem = entry.get("problem", "")
                        solution = entry.get("solution", entry.get("generated_solution", ""))
                        expected_answer = entry.get("expected_answer", "")
                        
                        if problem and solution:
                            math_example = {
                                "question": problem,
                                "answer": solution,
                                "domain": "mathematics",
                                "source": "nvidia/OpenMathInstruct-1",
                                "expected_answer": expected_answer,
                                "original_format": "problem-solution"
                            }
                            math_examples.append(math_example)
                    
                    logger.info(f"Loaded {len(math_examples)} sampled examples from OpenMathInstruct-1")
                    loaded = True
                    break
                    
                except ImportError:
                    # Fallback: chunked line-by-line parsing
                    logger.info("ijson not available, using chunked loading...")
                    entries = []
                    in_entries = False
                    brace_count = 0
                    current_entry = ""
                    
                    with open(openmath_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if '"entries":' in line:
                                in_entries = True
                                continue
                            
                            if in_entries:
                                for char in line:
                                    if char == '{':
                                        brace_count += 1
                                        current_entry += char
                                    elif char == '}':
                                        brace_count -= 1
                                        current_entry += char
                                        if brace_count == 0 and current_entry.strip():
                                            try:
                                                entry = json.loads(current_entry)
                                                problem = entry.get("problem", "")
                                                solution = entry.get("solution", entry.get("generated_solution", ""))
                                                expected_answer = entry.get("expected_answer", "")
                                                
                                                if problem and solution:
                                                    math_example = {
                                                        "question": problem,
                                                        "answer": solution,
                                                        "domain": "mathematics",
                                                        "source": "nvidia/OpenMathInstruct-1",
                                                        "expected_answer": expected_answer,
                                                        "original_format": "problem-solution"
                                                    }
                                                    entries.append(math_example)
                                                current_entry = ""
                                                
                                                if len(entries) >= max_samples:
                                                    logger.info(f"Loaded {len(entries)} math examples (reached max samples)")
                                                    math_examples = entries
                                                    loaded = True
                                                    break
                                            except json.JSONDecodeError:
                                                current_entry = ""
                                    elif brace_count > 0:
                                        current_entry += char
                            
                            if loaded:
                                break
                    
                    if not loaded and entries:
                        math_examples = entries
                        logger.info(f"Loaded {len(math_examples)} math examples from OpenMathInstruct-1")
                        loaded = True
                    break
                    
            except Exception as e:
                logger.error(f"Error loading OpenMathInstruct-1 Dataset from {openmath_path}: {e}")
                import traceback
                traceback.print_exc()
    
    if not loaded:
        logger.warning("OpenMathInstruct-1 Dataset not found. Run download_openmath_instruct.py to download it.")
    
    return math_examples


def load_theta_opinions_dataset(datasets_dir):
    """
    Load the Theta Opinions Dataset for personality-driven responses.
    This preserves the opinion format without forcing Q&A conversion.
    """
    opinion_examples = []
    opinion_paths = [
        Path(datasets_dir) / "theta_opinions.json",
        Path(datasets_dir) / "diverse_curriculum" / "theta_opinions.json"
    ]
    
    loaded = False
    for opinion_path in opinion_paths:
        if opinion_path.exists():
            try:
                logger.info(f"Loading Theta Opinions Dataset from {opinion_path}")
                with open(opinion_path, 'r', encoding='utf-8') as f:
                    opinions_data = json.load(f)
                
                if isinstance(opinions_data, list):
                    for entry in opinions_data:
                        if isinstance(entry, dict) and "opinion" in entry:
                            # Keep the specialized opinion format but add training-compatible fields
                            opinion_example = {
                                "type": "opinion",
                                "topic": entry.get("topic", "general"),
                                "subtopic": entry.get("subtopic", ""),
                                "content": entry.get("opinion", ""),
                                "strength": entry.get("strength", "moderate"),
                                "context": entry.get("context", ""),
                                "domain": "personality",
                                "source": "theta_opinions"
                            }
                            opinion_examples.append(opinion_example)
                
                logger.info(f"Loaded {len(opinion_examples)} opinion entries from Theta Opinions Dataset")
                loaded = True
                break
            except Exception as e:
                logger.error(f"Error loading Theta Opinions Dataset from {opinion_path}: {e}")
    
    if not loaded:
        logger.info("Theta Opinions Dataset not found")
    
    return opinion_examples


def load_natural_conversation_datasets(datasets_dir):
    """
    Load natural conversation datasets that have personality-rich responses.
    These are kept in conversational format, not forced to Q&A.
    """
    conversation_examples = []
    
    natural_files = [
        ("basic_conversation_natural.json", "conversation"),
        ("small_talk_natural.json", "small_talk"),
        ("basic_conversation.json", "conversation"),  # Fallback to main files
        ("small_talk.json", "small_talk")
    ]
    
    loaded_files = set()
    
    for filename, domain in natural_files:
        # Skip if we already loaded a version of this type
        base_name = filename.replace("_natural", "").replace(".json", "")
        if base_name in loaded_files:
            continue
            
        paths = [
            Path(datasets_dir) / filename,
            Path(datasets_dir) / "diverse_curriculum" / filename
        ]
        
        for conv_path in paths:
            if conv_path.exists():
                try:
                    with open(conv_path, 'r', encoding='utf-8') as f:
                        conv_data = json.load(f)
                    
                    if isinstance(conv_data, list):
                        for entry in conv_data:
                            if isinstance(entry, dict):
                                # Preserve conversational structure
                                conv_example = {
                                    "type": "conversation",
                                    "input": entry.get("question", entry.get("input", "")),
                                    "response": entry.get("answer", entry.get("response", "")),
                                    "domain": entry.get("domain", domain),
                                    "source": filename,
                                    "style": "natural"  # Mark as natural personality style
                                }
                                if conv_example["input"] and conv_example["response"]:
                                    conversation_examples.append(conv_example)
                        
                        logger.info(f"Loaded {len(conv_data)} entries from {filename}")
                        loaded_files.add(base_name)
                        break
                except Exception as e:
                    logger.error(f"Error loading {conv_path}: {e}")
    
    logger.info(f"Total natural conversation examples loaded: {len(conversation_examples)}")
    return conversation_examples


def load_technical_patterns_datasets(datasets_dir):
    """
    Load technical patterns datasets (code review, security, architecture, etc.)
    These are specialized non-Q&A format datasets.
    """
    pattern_examples = []
    
    pattern_files = [
        ("code_review_patterns.json", "code_review"),
        ("security_vulnerabilities.json", "security"),
        ("architecture_patterns.json", "architecture"),
        ("debugging_scenarios.json", "debugging"),
        ("api_design_patterns.json", "api_design"),
        ("error_handling_patterns.json", "error_handling"),
    ]
    
    for filename, pattern_type in pattern_files:
        paths = [
            Path(datasets_dir) / filename,
            Path(datasets_dir) / "diverse_curriculum" / filename
        ]
        
        for pattern_path in paths:
            if pattern_path.exists():
                try:
                    with open(pattern_path, 'r', encoding='utf-8') as f:
                        pattern_data = json.load(f)
                    
                    if isinstance(pattern_data, list):
                        for entry in pattern_data:
                            if isinstance(entry, dict):
                                # Preserve specialized pattern structure
                                pattern_example = {
                                    "type": entry.get("type", pattern_type),
                                    "category": entry.get("category", ""),
                                    "pattern": entry.get("pattern", entry.get("name", "")),
                                    "description": entry.get("description", ""),
                                    "content": entry,  # Keep full entry for training
                                    "domain": "technical_patterns",
                                    "source": filename
                                }
                                pattern_examples.append(pattern_example)
                        
                        logger.info(f"Loaded {len(pattern_data)} entries from {filename}")
                        break
                except Exception as e:
                    logger.error(f"Error loading {pattern_path}: {e}")
    
    logger.info(f"Total technical pattern examples loaded: {len(pattern_examples)}")
    return pattern_examples


def load_human_like_dpo_dataset(datasets_dir):
    """
    Load the Human-Like DPO Dataset for improving human-like responses.
    Apply advanced processing to better integrate with our enhanced curriculum.
    """
    dpo_examples = []
    dpo_paths = [
        Path(datasets_dir) / "human_like_dpo_dataset.json",  # Main path
        Path(datasets_dir) / "human_like_dpo.json",          # Alternate name
        Path(datasets_dir) / "diverse_curriculum" / "human_like_dpo.json"  # In diverse_curriculum dir
    ]
    
    loaded = False
    for dpo_path in dpo_paths:
        if dpo_path.exists():
            try:
                logger.info(f"Loading Human-Like DPO Dataset from {dpo_path}")
                with open(dpo_path, 'r', encoding='utf-8') as f:
                    try:
                        dpo_data = json.load(f)
                    except UnicodeDecodeError:
                        # Fall back to latin1 if utf-8 fails
                        logger.warning(f"UTF-8 decoding failed for {dpo_path}, trying latin1")
                        with open(dpo_path, 'r', encoding='latin1') as f2:
                            dpo_data = json.load(f2)
                
                # Handle different possible formats
                if isinstance(dpo_data, dict) and "entries" in dpo_data:
                    # Standard DPO format
                    count = 0
                    for entry in dpo_data["entries"]:
                        if "prompt" in entry and "chosen" in entry:
                            # Add category tagging for better domain balancing
                            categories = analyze_dpo_content(entry["prompt"], entry["chosen"])
                            
                            dpo_examples.append({
                                "question": entry["prompt"],
                                "answer": entry["chosen"],
                                "domain": "human_like_behavior",  # Changed from human_like_responses for consistency
                                "source": "human_like_dpo",
                                "categories": categories
                            })
                            count += 1
                elif isinstance(dpo_data, list):
                    # Handle list format
                    count = 0
                    for entry in dpo_data:
                        if isinstance(entry, dict):
                            if "prompt" in entry and "chosen" in entry:
                                # Standard DPO format in a list
                                categories = analyze_dpo_content(entry["prompt"], entry["chosen"])
                                dpo_examples.append({
                                    "question": entry["prompt"],
                                    "answer": entry["chosen"],
                                    "domain": "human_like_behavior",
                                    "source": "human_like_dpo",
                                    "categories": categories
                                })
                                count += 1
                            elif "question" in entry and "answer" in entry:
                                # Already in our QA format
                                categories = analyze_dpo_content(entry["question"], entry["answer"])
                                entry["domain"] = "human_like_behavior"  # Ensure consistent domain
                                entry["source"] = "human_like_dpo"
                                entry["categories"] = categories
                                dpo_examples.append(entry)
                                count += 1
                
                logger.info(f"Added {count} examples from Human-Like DPO Dataset")
                loaded = True
                break  # Exit after successfully loading from the first available path
            except Exception as e:
                logger.error(f"Error loading Human-Like DPO Dataset from {dpo_path}: {e}")
    
    if not loaded:
        logger.warning("Human-Like DPO Dataset not found in any of the expected locations")
    
    return dpo_examples


def analyze_dpo_content(question, answer):
    """
    Analyze the content of DPO examples to categorize them for better domain matching.
    This helps integrate DPO data with our diverse curriculum content.
    """
    categories = []
    
    # Define keywords for each category
    category_keywords = {
        "cognitive_reasoning": ["problem", "solution", "reasoning", "logic", "analyze", "think", "deduce", "infer", "conclude"],
        "psychological_frameworks": ["psychology", "behavior", "mental", "cognitive", "framework", "theory", "mind", "personality"],
        "conversational_dynamics": ["conversation", "talk", "discuss", "dialogue", "chat", "communicate", "exchange", "interact"],
        "human_experience": ["feel", "experience", "sensation", "emotion", "perceive", "sense", "subjective"],
        "ethical_reasoning": ["ethics", "moral", "right", "wrong", "should", "ought", "value", "principle", "duty"],
        "interpersonal_intelligence": ["relationship", "social", "interpersonal", "interact", "connect", "empathy", "understand"]
    }
    
    # Combined text for analysis
    text = (question + " " + answer).lower()
    
    # Check each category
    for category, keywords in category_keywords.items():
        for keyword in keywords:
            if keyword in text:
                categories.append(category)
                break  # One match is enough per category
    
    # Default to general human-like behavior if no specific categories matched
    if not categories:
        categories.append("human_like_behavior")
        
    return categories


# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_preparation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Setup NLTK data paths to ensure tokenizers work properly
def setup_nltk_data():
    """Set up NLTK data to ensure it works properly"""
    try:
        # Download required NLTK data
        packages = ['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger']
        
        logger.info("Downloading NLTK data packages...")
        for package in packages:
            nltk.download(package)
            
        # Create appropriate directories with OS-specific path separators
        tokenizers_dir = os.path.join(nltk.data.path[0], 'tokenizers')
        punkt_tab_dir = os.path.join(tokenizers_dir, 'punkt_tab')
        english_dir = os.path.join(punkt_tab_dir, 'english')
        
        # Create necessary directories
        os.makedirs(punkt_tab_dir, exist_ok=True)
        os.makedirs(english_dir, exist_ok=True)
        
        # Set paths for tab files
        collocations_tab_path = os.path.join(english_dir, 'collocations.tab')
        sentence_context_forms_path = os.path.join(english_dir, 'sentence_context_forms.tab')
        
        logger.info(f"NLTK data path: {nltk.data.path}")
        
        # Check for missing tab files and create them if needed
        if not os.path.exists(collocations_tab_path):
            logger.info(f"Creating missing file: {collocations_tab_path}")
            with open(collocations_tab_path, 'w') as f:
                f.write('')
                
        if not os.path.exists(sentence_context_forms_path):
            logger.info(f"Creating missing file: {sentence_context_forms_path}")
            with open(sentence_context_forms_path, 'w') as f:
                f.write('')
                
        logger.info("NLTK data setup complete")
        return True
    except Exception as e:
        logger.error(f"Error setting up NLTK data: {e}")
        return False

# Run NLTK setup
setup_nltk_data()

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

# Import project modules
from src.data_processing.dataset_balancer import DatasetBalancer
from src.data_processing.data_augmentation import DataAugmenter
from src.data_processing.technical_embeddings import TechnicalEmbeddings
from src.data_processing.process_data import load_json_dataset


class EnhancedDataPreparation:
    """Class for enhanced data preparation incorporating all 5 recommendations."""
    
    def _simple_augment_text(self, text: str) -> str:
        """
        Simple text augmentation as fallback when NLTK methods fail.
        Randomly modifies words in the text.
        
        Args:
            text: Input text to augment
            
        Returns:
            Augmented text
        """
        # Split text into words
        words = text.split()
        if not words:
            return text
            
        # Choose a random 15% of words to modify
        num_to_modify = max(1, int(len(words) * 0.15))
        indices_to_modify = random.sample(range(len(words)), min(num_to_modify, len(words)))
        
        # Simple modifications
        for idx in indices_to_modify:
            word = words[idx]
            if len(word) <= 3:
                continue
                
            # Choose a random modification
            mod_type = random.choice(['swap', 'drop', 'repeat'])
            
            if mod_type == 'swap' and len(word) > 3:
                # Swap two adjacent characters
                pos = random.randint(0, len(word) - 2)
                word = word[:pos] + word[pos+1] + word[pos] + word[pos+2:]
            elif mod_type == 'drop' and len(word) > 4:
                # Drop a character
                pos = random.randint(1, len(word) - 2)  # Don't drop first or last char
                word = word[:pos] + word[pos+1:]
            elif mod_type == 'repeat' and len(word) > 2:
                # Repeat a character
                pos = random.randint(0, len(word) - 1)
                word = word[:pos] + word[pos] + word[pos:]
                
            words[idx] = word
            
        return ' '.join(words)
    
    def __init__(self, datasets_dir: Path = None):
        """
        Initialize the enhanced data preparation.
        
        Args:
            datasets_dir: Path to the datasets directory
        """
        if datasets_dir is None:
            self.datasets_dir = project_root / "Datasets"
        else:
            self.datasets_dir = datasets_dir
            
        self.stats = {}
        self.domains = [
            # Technical domains
            "cybersecurity", "programming", "networking", 
            "cloud_computing", "data_science", "general_tech", 
            
            # Human-like behavior domains
            "human_like_behavior", "cognitive_reasoning", "psychological_frameworks",
            "conversational_dynamics", "human_experience", "humor_comprehension",
            "cultural_contexts", "ethical_reasoning", "tactical_knowledge",
            "interpersonal_intelligence", "memory_simulation",
            
            # Additional specialized domains
            "combat_domain", "technical_domain", "personal_domain",
            "emotional_intelligence", "conversational", "mathematics",
            
            # Personality and natural conversation domains
            "personality", "opinions", "small_talk", "conversation",
            
            # Technical patterns domains
            "technical_patterns", "code_review", "security", "architecture",
            "debugging", "api_design", "error_handling",
            
            # Enhanced training domains (10 recommendations)
            "contrastive_personality", "fragment_specific", "multi_turn",
            "trust_progression", "uncertainty", "refusal", "proactive",
            "mood_variations", "long_form_technical", "real_code_reviews"
        ]
        
        # Initialize components
        self.technical_embeddings = TechnicalEmbeddings(self.datasets_dir)
        self.data_augmenter = DataAugmenter()
        
        # Data storage
        self.raw_data = []
        self.processed_data = []
        self.balanced_data = []
        self.enhanced_data = []
        self.curriculum_data = []
        self.final_data = []
        self.human_like_dpo_data = []  # Initialize human-like DPO data storage
        self.curriculum_datasets = {}  # Store native curriculum datasets here
        
        # Domain data
        self.domain_data = {domain: [] for domain in self.domains}
        
        # Ensure Shy&dakota_qa.json exists and is formatted correctly
        self.ensure_personal_data_formatted()
        
    def ensure_personal_data_formatted(self):
        """Ensure personal dataset is properly formatted as QA pairs."""
        source_path = self.datasets_dir / "Shy&dakota.json"
        target_path = self.datasets_dir / "Shy&dakota_qa.json"
        
        if not source_path.exists():
            logger.warning(f"Source file {source_path} does not exist.")
            return
            
        if target_path.exists():
            # Check if it's already correctly formatted
            try:
                with open(target_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list) and len(data) > 0 and 'question' in data[0] and 'answer' in data[0]:
                        logger.info(f"Personal dataset {target_path} already correctly formatted.")
                        return
            except Exception as e:
                logger.warning(f"Error checking {target_path}: {e}")
                
        # Convert to QA format
        try:
            with open(source_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
                
            qa_pairs = []
            
            if isinstance(raw_data, dict):
                # Extract QA pairs from dictionary structure
                for key, value in raw_data.items():
                    if isinstance(value, str):
                        qa_pairs.append({
                            "question": f"What is {key}?",
                            "answer": value,
                            "domain": "general_tech"
                        })
            elif isinstance(raw_data, list):
                # Try to convert list items to QA pairs
                for item in raw_data:
                    if isinstance(item, dict):
                        if "question" in item and "answer" in item:
                            qa_pairs.append(item)
                        elif "title" in item and "content" in item:
                            qa_pairs.append({
                                "question": item["title"],
                                "answer": item["content"],
                                "domain": item.get("domain", "general_tech")
                            })
            
            if qa_pairs:
                with open(target_path, 'w', encoding='utf-8') as f:
                    json.dump(qa_pairs, f, indent=2)
                logger.info(f"Converted {source_path} to QA format with {len(qa_pairs)} pairs.")
            else:
                logger.warning(f"Could not convert {source_path} to QA format.")
                
        except Exception as e:
            logger.error(f"Error converting {source_path} to QA format: {e}")

    def load_curriculum_dataset_native(self, dataset_path, domain):
        """Load a curriculum dataset in its native format (not as QA pairs)"""
        try:
            dataset_path = Path(dataset_path)
            file_size_mb = dataset_path.stat().st_size / (1024 * 1024)
            
            # Skip very large files (>100MB) - they need special handling
            if file_size_mb > 100:
                logger.info(f"Skipping large file {dataset_path.name} ({file_size_mb:.1f} MB) - handled separately")
                return False
            
            with open(dataset_path, 'r', encoding='utf-8') as f:
                try:
                    dataset = json.load(f)
                except UnicodeDecodeError:
                    with open(dataset_path, 'r', encoding='latin1') as f2:
                        dataset = json.load(f2)
            
            # Store the dataset in its original format in the curriculum_datasets dictionary
            dataset_name = os.path.basename(dataset_path)
            self.curriculum_datasets[dataset_name] = {
                "content": dataset,
                "domain": domain,
                "path": str(dataset_path)
            }
            logger.info(f"Loaded native curriculum dataset: {dataset_name} for domain {domain}")
            return True
        except Exception as e:
            logger.error(f"Error loading curriculum dataset {dataset_path}: {e}")
            return False
            
    def validate_json_file(self, file_path):
        """
        Validate a JSON file before loading to prevent silent failures.
        Returns (is_valid, error_message, data) tuple.
        """
        file_path = Path(file_path)
        
        # Check file exists
        if not file_path.exists():
            return False, f"File not found: {file_path}", None
        
        # Check file is not empty
        if file_path.stat().st_size == 0:
            return False, f"File is empty: {file_path}", None
        
        # Try to load and validate JSON
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate data structure
            if data is None:
                return False, f"File contains null data: {file_path}", None
            
            if isinstance(data, list) and len(data) == 0:
                return False, f"File contains empty list: {file_path}", None
            
            if isinstance(data, dict) and len(data) == 0:
                return False, f"File contains empty dict: {file_path}", None
            
            return True, None, data
            
        except json.JSONDecodeError as e:
            return False, f"JSON decode error in {file_path}: {e}", None
        except UnicodeDecodeError:
            # Try latin1 encoding as fallback
            try:
                with open(file_path, 'r', encoding='latin1') as f:
                    data = json.load(f)
                return True, None, data
            except Exception as e2:
                return False, f"Encoding error in {file_path}: {e2}", None
        except Exception as e:
            return False, f"Error loading {file_path}: {e}", None
    
    def load_all_datasets(self):
        """Load and combine all available datasets with strict validation."""
        logger.info("Loading all available datasets...")
        
        # Use process_data.py's functionality
        from src.data_processing.process_data import process_data
        processed_path = process_data(include_new_datasets=True)
        
        # Validate processed_data.json before loading
        is_valid, error_msg, validated_data = self.validate_json_file(processed_path)
        if not is_valid:
            logger.error(f"CRITICAL: {error_msg}")
            logger.error("This may cause entire domains to be missing from training!")
            self.raw_data = []  # Start with empty list instead of crashing
        else:
            # Load the validated processed data
            self.raw_data = validated_data if isinstance(validated_data, list) else load_json_dataset(processed_path)
            logger.info(f"Successfully validated and loaded {len(self.raw_data)} examples from processed_data.json")
        
        # Load diverse curriculum datasets
        logger.info("Loading diverse curriculum datasets...")
        diverse_datasets = [
            # Core diverse curriculum datasets
            "Cognitive_reasoning.json",
            "Psychological_frameworks.json",
            "Conversational_dynamics.json",
            "Human_experience_simulation.json",
            "Humor_comprehension.json",
            "Cultural_contexts.json",
            "Ethical_reasoning.json",
            "Tactical_knowledge.json",
            "Interpersonal_intelligence.json",
            "Memory_simulation.json",
            "Emotional_learning.json",
            "Personal_preferences.json",
            "Ethical_scenarios.json",
            "Technical_concepts.json",
            "Narrative_experiences.json",
            "Emotional_intelligence.json",
            
            # Human-like DPO datasets
            "human_like_dpo_dataset.json",
            "human_like_dpo.json",
            
            # NVIDIA OpenMathInstruct-1 dataset (mathematics)
            "openmath_instruct_1.json",
            
            # OpenAssistant OASST1 dataset (high-quality assistant conversations)
            "openassistant_oasst1.json",
            "openassistant_oasst1_enhanced.json",
            
            # Personality and opinion datasets (specialized - not Q&A)
            "theta_opinions.json",
            "basic_conversation_natural.json",
            "small_talk_natural.json",
            
            # Technical patterns datasets (specialized - non-Q&A)
            "code_review_patterns.json",
            "security_vulnerabilities.json",
            "architecture_patterns.json",
            "debugging_scenarios.json",
            "api_design_patterns.json",
            "error_handling_patterns.json",
            
            # Enhanced training datasets (10 recommendations)
            "enhanced_training/contrastive_personality.json",
            "enhanced_training/fragment_specific_responses.json",
            "enhanced_training/multi_turn_conversations.json",
            "enhanced_training/trust_progression.json",
            "enhanced_training/uncertainty_handling.json",
            "enhanced_training/graceful_refusals.json",
            "enhanced_training/proactive_suggestions.json",
            "enhanced_training/mood_variations.json",
            "enhanced_training/long_form_technical.json",
            "enhanced_training/real_code_reviews.json"
        ]
        
        # Scan for additional files in Datasets directory that might be relevant
        try:
            # Find all JSON files in the Datasets directory
            all_json_files = [f for f in os.listdir(self.datasets_dir) if f.endswith('.json')]
            
            # Filter known dataset types we handle elsewhere
            excluded_patterns = [
                'processed_data.json', 'enhanced_', 'combined_', 'synthetic_',
                'test_set', 'qa.json', 'fragment', 'identity', 'conversation', 
                'cloud_computing', 'programming_', 'cybersecurity_', 'network',
                'data_science', 'general_tech'
            ]
            
            # Check for any JSON files we might have missed
            for json_file in all_json_files:
                # Skip files we know we handle elsewhere
                if any(pattern in json_file.lower() for pattern in excluded_patterns):
                    continue
                    
                # Skip files already in our list
                if json_file in diverse_datasets:
                    continue
                    
                # Add this file to our diverse_datasets list
                if json_file not in diverse_datasets:
                    diverse_datasets.append(json_file)
                    logger.info(f"Added additional dataset file: {json_file}")
        except Exception as e:
            logger.warning(f"Error while scanning for additional datasets: {e}")
        
        # First try to load from diverse_curriculum directory (processed files)
        diverse_curriculum_dir = self.datasets_dir / "diverse_curriculum"
        found_datasets = 0
        
        if diverse_curriculum_dir.exists():
            logger.info("Loading from diverse_curriculum directory...")
            
            for dataset_file in diverse_datasets:
                dataset_path = diverse_curriculum_dir / dataset_file
                if dataset_path.exists():
                    # Load the dataset in its native format
                    dataset_name = dataset_file.split('.')[0]
                    domain = dataset_name.lower()
                    
                    # Use our native loader function that preserves the original structure
                    if self.load_curriculum_dataset_native(dataset_path, domain):
                        found_datasets += 1
                        logger.info(f"Loaded native curriculum dataset from {diverse_curriculum_dir / dataset_file}")
                        continue
                    
                    # Fallback to old QA format conversion if native loading fails
                    dataset = load_json_dataset(dataset_path)
                    qa_pairs = []
                    
                    # Generate targeted questions based on dataset content
                    if dataset_name == "cognitive_reasoning":
                        # For each reasoning pattern, create a specific question
                        if "reasoning_patterns" in dataset:
                            for pattern in dataset.get("reasoning_patterns", []):
                                category = pattern.get("category", "")
                                qa_pairs.append({
                                    "question": f"Explain {category} as a cognitive reasoning pattern",
                                    "answer": f"**{category.title()}** is a cognitive reasoning pattern that involves {', '.join([p.get('process', '') for p in pattern.get('patterns', [])]).lower()}. Here are examples: {json.dumps([p.get('examples', []) for p in pattern.get('patterns', [])][:2], indent=2)}",
                                    "domain": "human_like_behavior",
                                    "source": f"diverse_curriculum/{dataset_file}"
                                })
                    
                    elif dataset_name == "psychological_frameworks":
                        # For each framework, create a specific question
                        if "frameworks" in dataset:
                            for framework in dataset.get("frameworks", []):
                                name = framework.get("name", "")
                                core_concept = framework.get("core_concept", "")
                                qa_pairs.append({
                                    "question": f"Explain the psychological framework of {name}",
                                    "answer": f"**{name}** is a psychological framework based on the core concept that {core_concept}. {json.dumps(framework.get('applications', {}), indent=2)}",
                                    "domain": "human_like_behavior",
                                    "source": f"diverse_curriculum/{dataset_file}"
                                })
                    
                    elif dataset_name == "memory_simulation":
                        # For each memory system, create specific questions
                        if "memory_systems" in dataset:
                            for system in dataset.get("memory_systems", []):
                                name = system.get("system", "")
                                characteristics = system.get("characteristics", [])
                                qa_pairs.append({
                                    "question": f"How does {name} work in human memory?",
                                    "answer": f"**{name}** is a memory system characterized by {', '.join(characteristics).lower()}. {json.dumps(system.get('components', []), indent=2) if 'components' in system else ''}",
                                    "domain": "human_like_behavior",
                                    "source": f"diverse_curriculum/{dataset_file}"
                                })
                                
                                # Add questions about memory distortions if available
                                if "distortion_patterns" in system:
                                    qa_pairs.append({
                                        "question": f"What are common memory distortions in {name}?",
                                        "answer": f"Common memory distortions in **{name}** include {', '.join([d.get('type', '') for d in system.get('distortion_patterns', [])])}. {json.dumps(system.get('distortion_patterns', [])[:2], indent=2)}",
                                        "domain": "human_like_behavior",
                                        "source": f"diverse_curriculum/{dataset_file}"
                                    })
                    
                    # Generic approach for other datasets
                    else:
                        # Create a summary question for the overall content
                        qa_pairs.append({
                            "question": f"What are the key components of {dataset_name.replace('_', ' ')}?",
                            "answer": f"Here's a comprehensive overview of {dataset_name.replace('_', ' ')}: {json.dumps(dataset, indent=2)}",
                            "domain": "human_like_behavior",
                            "source": f"diverse_curriculum/{dataset_file}"
                        })
                        
                        # Extract key components if possible
                        # Check if dataset is a dictionary before trying to use .items()
                        if isinstance(dataset, dict):
                            for key, value in dataset.items():
                                if isinstance(value, list) and key not in ["description"]:
                                    # For list type components
                                    qa_pairs.append({
                                        "question": f"Describe the {key.replace('_', ' ')} in {dataset_name.replace('_', ' ')}",
                                        "answer": f"The {key.replace('_', ' ')} in {dataset_name.replace('_', ' ')} include: {json.dumps(value[:3], indent=2)} (among others)",
                                        "domain": "human_like_behavior",
                                        "source": f"diverse_curriculum/{dataset_file}"
                                    })
                                elif isinstance(value, dict):
                                    # For dictionary type components
                                    qa_pairs.append({
                                        "question": f"What is the structure of {key.replace('_', ' ')} in {dataset_name.replace('_', ' ')}?",
                                        "answer": f"The {key.replace('_', ' ')} in {dataset_name.replace('_', ' ')} has the following structure: {json.dumps(list(value.keys())[:5], indent=2)}",
                                        "domain": "human_like_behavior",
                                        "source": f"diverse_curriculum/{dataset_file}"
                                    })
                        elif isinstance(dataset, list):
                            # Handle list-type datasets
                            qa_pairs.append({
                                "question": f"What items are included in the {dataset_name.replace('_', ' ')} dataset?",
                                "answer": f"The {dataset_name.replace('_', ' ')} dataset includes {len(dataset)} items. Here are the first few examples: {json.dumps(dataset[:3], indent=2)}",
                                "domain": "human_like_behavior",
                                "source": f"diverse_curriculum/{dataset_file}"
                            })
                                
                    
                    # Add these QA pairs to raw data
                    self.raw_data.extend(qa_pairs)
                    found_datasets += 1
                    logger.info(f"Added {len(qa_pairs)} QA pairs from {diverse_curriculum_dir / dataset_file}")
        
        # If we didn't find all datasets in diverse_curriculum, try loading directly from main Datasets directory
        if found_datasets < len(diverse_datasets):
            logger.info(f"Found only {found_datasets}/{len(diverse_datasets)} datasets in diverse_curriculum. Trying main directory...")
            
            for dataset_file in diverse_datasets:
                # Only try to load if we didn't already load it from diverse_curriculum
                if not (diverse_curriculum_dir / dataset_file).exists():
                    # Try loading directly from main Datasets directory
                    direct_path = self.datasets_dir / dataset_file
                    if direct_path.exists():
                        # Load in native format first
                        dataset_name = dataset_file.split('.')[0]
                        domain = dataset_name.lower()
                        
                        # Try loading in native format first
                        if self.load_curriculum_dataset_native(direct_path, domain):
                            found_datasets += 1
                            logger.info(f"Loaded native curriculum dataset from {direct_path}")
                            continue
                            
                        # Fallback to old QA format conversion if native loading fails
                        dataset = load_json_dataset(direct_path)
                        qa_pairs = []
                        
                        # Generate targeted questions based on dataset content (same as above)
                        if dataset_name.lower() == "cognitive_reasoning":
                            if "reasoning_patterns" in dataset:
                                for pattern in dataset.get("reasoning_patterns", []):
                                    category = pattern.get("category", "")
                                    qa_pairs.append({
                                        "question": f"Explain {category} as a cognitive reasoning pattern",
                                        "answer": f"**{category.title()}** is a cognitive reasoning pattern that involves {', '.join([p.get('process', '') for p in pattern.get('patterns', [])]).lower()}. Here are examples: {json.dumps([p.get('examples', []) for p in pattern.get('patterns', [])][:2], indent=2)}",
                                        "domain": "cognitive_reasoning",
                                        "source": f"direct/{dataset_file}"
                                    })
                        
                        elif dataset_name.lower() == "psychological_frameworks":
                            if "frameworks" in dataset:
                                for framework in dataset.get("frameworks", []):
                                    name = framework.get("name", "")
                                    core_concept = framework.get("core_concept", "")
                                    qa_pairs.append({
                                        "question": f"Explain the psychological framework of {name}",
                                        "answer": f"**{name}** is a psychological framework based on the core concept that {core_concept}. {json.dumps(framework.get('applications', {}), indent=2)}",
                                        "domain": "psychological_frameworks",
                                        "source": f"direct/{dataset_file}"
                                    })
                        
                        elif dataset_name.lower() == "memory_simulation":
                            if "memory_systems" in dataset:
                                for system in dataset.get("memory_systems", []):
                                    name = system.get("system", "")
                                    characteristics = system.get("characteristics", [])
                                    qa_pairs.append({
                                        "question": f"How does {name} work in human memory?",
                                        "answer": f"**{name}** is a memory system characterized by {', '.join(characteristics).lower()}. {json.dumps(system.get('components', []), indent=2) if 'components' in system else ''}",
                                        "domain": "memory_simulation",
                                        "source": f"direct/{dataset_file}"
                                    })
                                    
                                    if "distortion_patterns" in system:
                                        qa_pairs.append({
                                            "question": f"What are common memory distortions in {name}?",
                                            "answer": f"Common memory distortions in **{name}** include {', '.join([d.get('type', '') for d in system.get('distortion_patterns', [])])}. {json.dumps(system.get('distortion_patterns', [])[:2], indent=2)}",
                                            "domain": "memory_simulation",
                                            "source": f"direct/{dataset_file}"
                                        })
                        
                        # Generic approach for other datasets
                        else:
                            # Use the dataset_name as the domain for better categorization
                            domain = dataset_name.lower()
                            
                            # Create a summary question for the overall content
                            qa_pairs.append({
                                "question": f"What are the key components of {dataset_name.replace('_', ' ')}?",
                                "answer": f"Here's a comprehensive overview of {dataset_name.replace('_', ' ')}: {json.dumps(dataset, indent=2)}",
                                "domain": domain,
                                "source": f"direct/{dataset_file}"
                            })
                            
                            # Extract key components if possible
                            # Check if dataset is a dictionary before trying to use .items()
                            if isinstance(dataset, dict):
                                for key, value in dataset.items():
                                    if isinstance(value, list) and key not in ["description"]:
                                        # For list type components
                                        qa_pairs.append({
                                            "question": f"Describe the {key.replace('_', ' ')} in {dataset_name.replace('_', ' ')}",
                                            "answer": f"The {key.replace('_', ' ')} in {dataset_name.replace('_', ' ')} include: {json.dumps(value[:3], indent=2)} (among others)",
                                            "domain": domain,
                                            "source": f"direct/{dataset_file}"
                                        })
                                    elif isinstance(value, dict):
                                        # For dictionary type components
                                        qa_pairs.append({
                                            "question": f"What is the structure of {key.replace('_', ' ')} in {dataset_name.replace('_', ' ')}?",
                                            "answer": f"The {key.replace('_', ' ')} in {dataset_name.replace('_', ' ')} has the following structure: {json.dumps(list(value.keys())[:5], indent=2)}",
                                            "domain": domain,
                                            "source": f"direct/{dataset_file}"
                                        })
                            elif isinstance(dataset, list):
                                # Handle list-type datasets
                                qa_pairs.append({
                                    "question": f"What items are included in the {dataset_name.replace('_', ' ')} dataset?",
                                    "answer": f"The {dataset_name.replace('_', ' ')} dataset includes {len(dataset)} items. Here are the first few examples: {json.dumps(dataset[:3], indent=2)}",
                                    "domain": domain,
                                    "source": f"direct/{dataset_file}"
                                })
                        
                        # Add these QA pairs to raw data
                        self.raw_data.extend(qa_pairs)
                        logger.info(f"Added {len(qa_pairs)} QA pairs from direct file {direct_path}")
                        found_datasets += 1
        
        logger.info(f"Loaded {len(self.raw_data)} examples from all datasets.")
        
        # Group data by domain
        for item in self.raw_data:
            domain = item.get("domain", "general_tech")
            if domain not in self.domains:
                domain = "general_tech"
            self.domain_data[domain].append(item)
            
        # Log domain distribution
        for domain, data in self.domain_data.items():
            logger.info(f"Domain '{domain}': {len(data)} examples")
            
        return len(self.raw_data)
    
    def apply_quality_checks(self):
        """
        Apply quality checks to filter out malformed entries (Recommendation 2)
        """
        logger.info("Applying quality checks to data...")
        
        filtered_data = []
        stats = {
            "original_count": len(self.raw_data),
            "too_short_question": 0,
            "too_long_question": 0,
            "too_short_answer": 0,
            "too_long_answer": 0,
            "duplicate_removed": 0,
            "retained_count": 0
        }
        
        # Calculate similarity for deduplication
        text_hashes = set()
        
        for item in tqdm(self.raw_data, desc="Applying quality checks"):
            question = item.get("question", "")
            answer = item.get("answer", "")
            
            # Check lengths
            q_length = len(question.split())
            a_length = len(answer.split())
            
            # Filter by length
            if q_length < 3:
                stats["too_short_question"] += 1
                continue
            if q_length > 100:
                stats["too_long_question"] += 1
                continue
            if a_length < 10:
                stats["too_short_answer"] += 1
                continue
            if a_length > 1024:
                stats["too_long_answer"] += 1
                continue
                
            # Simple deduplication using text hashing
            text_hash = hash(f"{question.lower()}:{answer.lower()[:100]}")
            if text_hash in text_hashes:
                stats["duplicate_removed"] += 1
                continue
            text_hashes.add(text_hash)
            
            # Add to filtered data
            filtered_data.append(item)
        
        stats["retained_count"] = len(filtered_data)
        retention_rate = (stats["retained_count"] / stats["original_count"]) * 100 if stats["original_count"] > 0 else 0
        
        logger.info(f"Quality check results:")
        logger.info(f"- Original count: {stats['original_count']}")
        logger.info(f"- Too short questions: {stats['too_short_question']}")
        logger.info(f"- Too long questions: {stats['too_long_question']}")
        logger.info(f"- Too short answers: {stats['too_short_answer']}")
        logger.info(f"- Too long answers: {stats['too_long_answer']}")
        logger.info(f"- Duplicates removed: {stats['duplicate_removed']}")
        logger.info(f"- Retained count: {stats['retained_count']} ({retention_rate:.1f}%)")
        
        self.processed_data = filtered_data
        self.stats["quality_check"] = stats
        
        return len(filtered_data)
    
    def balance_domains(self):
        """
        Balance data across domains to ensure no domain dominates (Recommendation 1)
        """
        logger.info("Balancing domains in dataset...")
        
        # Count examples per domain
        domain_counts = {domain: len(examples) for domain, examples in self.domain_data.items()}
        logger.info(f"Original domain distribution: {domain_counts}")
        
        # Group related domains for more intelligent balancing
        domain_groups = {
            "technical": ["cybersecurity", "programming", "networking", "cloud_computing", 
                        "data_science", "technical_domain"],
            "human_like": ["human_like_behavior", "cognitive_reasoning", "psychological_frameworks",
                          "conversational_dynamics", "human_experience", "humor_comprehension",
                          "cultural_contexts", "ethical_reasoning", "interpersonal_intelligence", 
                          "memory_simulation", "emotional_intelligence"],
            "specialized": ["combat_domain", "tactical_knowledge", "personal_domain"],
            "general": ["general_tech", "conversational"]
        }
        
        # Calculate total count and group counts
        total_count = sum(domain_counts.values())
        group_counts = {group: 0 for group in domain_groups}
        
        for domain, count in domain_counts.items():
            for group, domains in domain_groups.items():
                if domain in domains:
                    group_counts[group] += count
                    break
        
        logger.info(f"Domain group counts: {group_counts}")
        
        # Calculate target percentages for each group
        # Increased technical and conversational weighting based on domain evaluation scores
        group_percentages = {
            "technical": 0.40,   # 40% technical content (increased from 35%)
            "human_like": 0.30, # 30% human-like behavior content (reduced from 35%)
            "specialized": 0.15, # 15% specialized domains
            "general": 0.15      # 15% general/conversational content
        }
        
        # Calculate minimum examples per domain based on group percentage
        min_count = 300  # Base minimum - increased from 50 to ensure domain diversity
        
        # Calculate maximum allowed count per domain to prevent any single domain from dominating
        max_domain_percentage = 0.20  # Max 20% per domain (reduced from 25%)
        max_allowed = int(total_count * max_domain_percentage)
        
        # Create balanced dataset
        balanced_data = []
        for domain, examples in self.domain_data.items():
            if len(examples) > max_allowed:
                # Downsample domains that are too large
                logger.info(f"Downsampling '{domain}' from {len(examples)} to {max_allowed} examples")
                balanced_data.extend(random.sample(examples, max_allowed))
            elif len(examples) < min_count and len(examples) > 0:
                # Upsample domains that are too small
                needed = min_count - len(examples)
                logger.info(f"Upsampling '{domain}' from {len(examples)} by adding {needed} examples")
                balanced_data.extend(examples)  # Add all original examples
                
                # Add upsampled examples with augmentation
                if needed > 0:
                    augmented = []
                    for _ in range(needed):
                        example = random.choice(examples)
                        augmented_example = dict(example)
                        
                        # Apply light augmentation with fallback to simple approach
                        try:
                            # Try using the data augmenter's method
                            augmented_example["question"] = self.data_augmenter.paraphrase_by_synonym_replacement(
                                example["question"], replacement_prob=0.15
                            )
                            augmented_example["answer"] = self.data_augmenter.paraphrase_by_synonym_replacement(
                                example["answer"], replacement_prob=0.1
                            )
                        except Exception as e:
                            logger.warning(f"Data augmentation error: {e}. Using simple augmentation as fallback.")
                            # Simple fallback augmentation
                            augmented_example["question"] = self._simple_augment_text(example["question"])
                            augmented_example["answer"] = self._simple_augment_text(example["answer"])
                        augmented_example["augmented"] = True
                        
                        augmented.append(augmented_example)
                    balanced_data.extend(augmented)
            else:
                # Keep domains that are within the desired range
                balanced_data.extend(examples)
        
        # Calculate new domain distribution
        domain_distribution = defaultdict(int)
        for item in balanced_data:
            domain = item.get("domain", "general_tech")
            if domain not in self.domains:
                domain = "general_tech"
            domain_distribution[domain] += 1
            
        total = sum(domain_distribution.values())
        domain_percentages = {domain: (count / total) * 100 for domain, count in domain_distribution.items()}
        
        logger.info("Balanced domain distribution:")
        for domain, percentage in domain_percentages.items():
            logger.info(f"- {domain}: {domain_distribution[domain]} examples ({percentage:.1f}%)")
        
        self.balanced_data = balanced_data
        self.stats["domain_balance"] = {
            "original_distribution": domain_counts,
            "balanced_distribution": dict(domain_distribution),
            "balanced_percentages": domain_percentages
        }
        
        return len(balanced_data)
    
    def enhance_with_technical_context(self):
        """
        Enhance data with technical context using TechnicalEmbeddings (Recommendation 3)
        """
        logger.info("Enhancing data with technical context...")
        
        # Initialize technical embeddings
        self.technical_embeddings.collect_technical_terms()
        
        enhanced_data = []
        enhancements = {domain: 0 for domain in self.domains}
        
        for item in tqdm(self.balanced_data, desc="Enhancing with technical context"):
            domain = item.get("domain", "general_tech")
            if domain not in self.domains:
                domain = "general_tech"
                
            question = item["question"]
            answer = item["answer"]
            
            # Enrich question and answer with technical context
            question_enriched = self.technical_embeddings.enrich_text_with_technical_context(question, domain)
            answer_enriched = self.technical_embeddings.enrich_text_with_technical_context(answer, domain)
            
            # Get technical terms found
            question_terms = question_enriched.get("technical_terms", {})
            answer_terms = answer_enriched.get("technical_terms", {})
            
            # Combine terms
            all_terms = {**question_terms, **answer_terms}
            
            # Add technical context if terms were found
            if all_terms:
                item["technical_terms"] = list(all_terms.keys())
                item["technical_context"] = {
                    term: info.get("definition", "") 
                    for term, info in all_terms.items() if info.get("definition")
                }
                enhancements[domain] += 1
                
            enhanced_data.append(item)
            
        # Log enhancement statistics
        logger.info("Technical context enhancement statistics:")
        for domain, count in enhancements.items():
            total = sum(1 for item in enhanced_data if item.get("domain", "general_tech") == domain)
            percentage = (count / total) * 100 if total > 0 else 0
            logger.info(f"- {domain}: {count}/{total} examples enhanced ({percentage:.1f}%)")
            
        self.enhanced_data = enhanced_data
        self.stats["technical_enhancement"] = enhancements
        
        return len(enhanced_data)
    
    def implement_curriculum_learning(self):
        """
        Order training data from simple to complex (Recommendation 4)
        Includes pre-curriculum validation to catch problematic samples.
        """
        logger.info("Implementing curriculum learning...")
        
        # Pre-curriculum validation - filter out problematic samples
        validated_data = []
        validation_stats = {
            "total": len(self.enhanced_data),
            "empty_question": 0,
            "empty_answer": 0,
            "ultra_short": 0,
            "duplicate": 0,
            "passed": 0
        }
        
        seen_hashes = set()
        for item in self.enhanced_data:
            question = item.get("question", "")
            answer = item.get("answer", "")
            
            # Skip empty questions
            if not question or len(question.strip()) < 5:
                validation_stats["empty_question"] += 1
                continue
            
            # Skip empty answers
            if not answer or len(answer.strip()) < 10:
                validation_stats["empty_answer"] += 1
                continue
            
            # Skip ultra-short samples (combined length < 50 chars)
            combined_len = len(question) + len(answer)
            if combined_len < 50:
                validation_stats["ultra_short"] += 1
                continue
            
            # Skip duplicates
            content_hash = hash(f"{question[:100]}:{answer[:100]}")
            if content_hash in seen_hashes:
                validation_stats["duplicate"] += 1
                continue
            seen_hashes.add(content_hash)
            
            validated_data.append(item)
            validation_stats["passed"] += 1
        
        # Log validation results
        logger.info(f"Pre-curriculum validation: {validation_stats['passed']}/{validation_stats['total']} samples passed")
        if validation_stats["empty_question"] > 0:
            logger.warning(f"  - Removed {validation_stats['empty_question']} samples with empty/short questions")
        if validation_stats["empty_answer"] > 0:
            logger.warning(f"  - Removed {validation_stats['empty_answer']} samples with empty/short answers")
        if validation_stats["ultra_short"] > 0:
            logger.warning(f"  - Removed {validation_stats['ultra_short']} ultra-short samples")
        if validation_stats["duplicate"] > 0:
            logger.warning(f"  - Removed {validation_stats['duplicate']} duplicate samples")
        
        # Use validated data for curriculum learning
        self.enhanced_data = validated_data
        
        # Define complexity metrics
        def calculate_complexity(item):
            question = item.get("question", "")
            answer = item.get("answer", "")
            
            # Factors that contribute to complexity
            q_length = len(question.split())
            a_length = len(answer.split())
            tech_terms_count = len(item.get("technical_terms", []))
            
            # Linguistic complexity indicators
            complex_words_pattern = r'\b[a-zA-Z]{10,}\b'  # Words with 10+ characters
            complex_words_q = len(re.findall(complex_words_pattern, question))
            complex_words_a = len(re.findall(complex_words_pattern, answer))
            
            # Calculate complexity score
            score = (
                q_length * 0.1 +                # Question length
                a_length * 0.05 +               # Answer length
                tech_terms_count * 1.0 +        # Technical terms count
                complex_words_q * 0.5 +         # Complex words in question
                complex_words_a * 0.2           # Complex words in answer
            )
            
            return score
        
        # Group data by domain
        domain_groups = defaultdict(list)
        for item in self.enhanced_data:
            domain = item.get("domain", "general_tech")
            if domain not in self.domains:
                domain = "general_tech"
            domain_groups[domain].append(item)
        
        # Sort each domain by complexity
        curriculum_data = []
        for domain, items in domain_groups.items():
            # Calculate complexity for each item
            for item in items:
                item["complexity_score"] = calculate_complexity(item)
                
            # Sort by complexity (ascending)
            sorted_items = sorted(items, key=lambda x: x.get("complexity_score", 0))
            curriculum_data.extend(sorted_items)
            
            # Log curriculum learning statistics
            if sorted_items:
                min_score = min(item.get("complexity_score", 0) for item in sorted_items)
                max_score = max(item.get("complexity_score", 0) for item in sorted_items)
                avg_score = sum(item.get("complexity_score", 0) for item in sorted_items) / len(sorted_items)
                
                logger.info(f"Domain '{domain}' complexity range: {min_score:.1f} - {max_score:.1f} (avg: {avg_score:.1f})")
        
        self.curriculum_data = curriculum_data
        self.stats["curriculum_learning"] = {
            "min_complexity": min(item.get("complexity_score", 0) for item in curriculum_data),
            "max_complexity": max(item.get("complexity_score", 0) for item in curriculum_data),
            "avg_complexity": sum(item.get("complexity_score", 0) for item in curriculum_data) / len(curriculum_data)
        }
        
        return len(curriculum_data)
    
    def augment_with_synthetic_examples(self):
        """
        Augment dataset with synthetic examples (Recommendation 5)
        """
        logger.info("Augmenting with synthetic examples...")
        
        # Start with curriculum data
        final_data = list(self.curriculum_data)
        original_count = len(final_data)
        
        # Calculate target count for each domain
        domain_counts = defaultdict(int)
        for item in final_data:
            domain = item.get("domain", "general_tech")
            if domain not in self.domains:
                domain = "general_tech"
            domain_counts[domain] += 1
            
        # Find underrepresented subdomains
        subdomains = {
            # Technical domains
            "cybersecurity": ["encryption", "network_security", "threat_intelligence", "incident_response", "vulnerability_management"],
            "programming": ["algorithms", "data_structures", "web_development", "databases", "frontend", "backend"],
            "networking": ["routing", "switching", "protocols", "vpn", "wan", "lan"],
            "cloud_computing": ["aws", "azure", "gcp", "kubernetes", "docker", "serverless"],
            "data_science": ["machine_learning", "deep_learning", "statistics", "data_visualization", "big_data"],
            
            # Human-like domains (from diverse curriculum datasets)
            "cognitive_reasoning": ["computational_thinking", "mathematical_intuition", "deductive_reasoning", 
                                  "inductive_reasoning", "abductive_reasoning", "statistical_reasoning", 
                                  "systems_thinking", "creative_reasoning", "ethical_reasoning", "metacognitive_reasoning"],
            "psychological_frameworks": ["cognitive_dissonance", "attachment_theory", "social_identity", 
                                       "self_determination", "transtheoretical_model", "cognitive_behavioral"],
            "memory_simulation": ["working_memory", "autobiographical_memory", "semantic_memory", "episodic_memory",
                               "procedural_memory", "memory_distortion"],
            "interpersonal_intelligence": ["emotional_intelligence", "attachment_patterns", "influence_persuasion",
                                        "conflict_resolution", "communication_styles"]
        }
        
        # Count subdomain examples
        subdomain_counts = defaultdict(int)
        for item in final_data:
            domain = item.get("domain", "general_tech")
            if domain in subdomains:
                question = item.get("question", "").lower()
                answer = item.get("answer", "").lower()
                
                for subdomain in subdomains[domain]:
                    if subdomain.lower() in question or subdomain.lower() in answer:
                        subdomain_counts[f"{domain}_{subdomain}"] += 1
        
        # Different thresholds for different domain types
        thresholds = {
            # Technical domains
            "cybersecurity": 50,
            "programming": 50,
            "networking": 50,
            "cloud_computing": 50,
            "data_science": 50,
            
            # Human-like domains (need more examples for better human-like behavior)
            "cognitive_reasoning": 100, 
            "psychological_frameworks": 100,
            "memory_simulation": 100,
            "interpersonal_intelligence": 100
        }
        
        # Find underrepresented subdomains
        underrepresented = []
        for domain in subdomains:
            threshold = thresholds.get(domain, 50)  # Default to 50 if not specified
            for subdomain in subdomains[domain]:
                key = f"{domain}_{subdomain}"
                if subdomain_counts[key] < threshold:
                    underrepresented.append((domain, subdomain, threshold - subdomain_counts[key]))
        
        logger.info(f"Found {len(underrepresented)} underrepresented subdomains")
        
        # Generate synthetic examples for underrepresented subdomains
        synthetic_examples = []
        
        # Get template examples for each domain
        domain_templates = {}
        for domain in subdomains:
            # Get the best examples from each domain (high quality, with technical terms)
            templates = [
                item for item in final_data 
                if item.get("domain") == domain and item.get("technical_terms")
            ]
            if templates:
                domain_templates[domain] = templates
        
        # Generate synthetic examples
        for domain, subdomain, needed in underrepresented:
            if domain not in domain_templates:
                continue
                
            templates = domain_templates[domain]
            if not templates:
                continue
                
            logger.info(f"Generating {needed} examples for {domain}/{subdomain}")
            
            # Generate up to 20 examples per subdomain
            for _ in range(min(needed, 20)):
                # Get a random template
                template = random.choice(templates)
                
                # Create a synthetic example
                synthetic = dict(template)
                
                # Ensure the subdomain is mentioned
                question = template["question"]
                answer = template["answer"]
                
                # Modify the question to include the subdomain
                if subdomain.lower() not in question.lower():
                    question = f"Can you explain {subdomain} in the context of {domain}? {question}"
                
                # Apply paraphrasing
                synthetic["question"] = self.data_augmenter.paraphrase_by_synonym_replacement(question)
                synthetic["answer"] = self.data_augmenter.paraphrase_by_synonym_replacement(answer)
                synthetic["synthetic"] = True
                synthetic["subdomain"] = subdomain
                
                synthetic_examples.append(synthetic)
        
        logger.info(f"Generated {len(synthetic_examples)} synthetic examples")
        
        # Add synthetic examples to final data
        final_data.extend(synthetic_examples)
        
        self.final_data = final_data
        # Add Human-Like DPO examples to the final dataset
        if hasattr(self, 'human_like_dpo_data') and self.human_like_dpo_data:
            logger.info(f"Adding {len(self.human_like_dpo_data)} Human-Like DPO examples to final dataset")
            final_data.extend(self.human_like_dpo_data)
            
        self.stats["synthetic_augmentation"] = {
            "original_count": original_count,
            "synthetic_count": len(synthetic_examples),
            "final_count": len(final_data),
            "underrepresented_subdomains": len(underrepresented),
            "human_like_dpo_count": len(self.human_like_dpo_data) if hasattr(self, 'human_like_dpo_data') else 0,
            "openmath_instruct_count": len(self.openmath_instruct_data) if hasattr(self, 'openmath_instruct_data') else 0,
            "theta_opinions_count": len(self.theta_opinions_data) if hasattr(self, 'theta_opinions_data') else 0,
            "natural_conversation_count": len(self.natural_conversation_data) if hasattr(self, 'natural_conversation_data') else 0,
            "technical_patterns_count": len(self.technical_patterns_data) if hasattr(self, 'technical_patterns_data') else 0
        }
        
        return len(final_data)
    
    def save_final_dataset(self, output_path=None):
        """Save the final enhanced dataset."""
        if output_path is None:
            output_path = self.datasets_dir / "enhanced_training_data.json"
            
        # Remove complexity scores and clean up metadata not needed for training
        for item in self.final_data:
            # Remove complexity score used for curriculum learning
            if "complexity_score" in item:
                del item["complexity_score"]
            
            # Convert category lists to strings for easier loading in training
            if "categories" in item:
                item["categories_str"] = ",".join(item["categories"])
                
        # Count domains in final dataset
        domain_counts = defaultdict(int)
        for item in self.final_data:
            domain = item.get("domain", "general_tech")
            domain_counts[domain] += 1
            
        # Log domain distribution in final dataset
        logger.info("Final dataset domain distribution:")
        for domain, count in sorted(domain_counts.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"- {domain}: {count} examples ({count/len(self.final_data)*100:.1f}%)")
        
        # Create final data structure with both QA pairs and native curriculum datasets
        final_combined_data = {
            "qa_pairs": self.final_data,
            "curriculum_datasets": self.curriculum_datasets
        }
        
        # Log information about native curriculum datasets
        if hasattr(self, 'curriculum_datasets') and self.curriculum_datasets:
            logger.info(f"Including {len(self.curriculum_datasets)} native curriculum datasets:")
            for name, info in self.curriculum_datasets.items():
                logger.info(f"- {name} (domain: {info['domain']})")
                
        # Save final dataset
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_combined_data, f, indent=2)
            
        logger.info(f"Final enhanced dataset saved to {output_path} ({len(self.final_data)} examples)")
        
        # Save statistics
        stats_path = output_path.parent / "enhanced_data_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2)
            
        logger.info(f"Data enhancement statistics saved to {stats_path}")
        
        return len(self.final_data)
    
    def run_full_preparation(self):
        """
        Run the full data preparation pipeline with all enhancements.
        """
        start_time = logging.Formatter().converter()
        logger.info(f"Starting enhanced data preparation at {start_time}")
        
        # Step 1: Load all datasets
        self.load_all_datasets()
        
        # Step 1.5: Load Human-Like DPO Dataset
        self.human_like_dpo_data = load_human_like_dpo_dataset(self.datasets_dir)
        
        # Process and categorize Human-Like DPO data before integrating it
        dpo_count_by_category = defaultdict(int)
        for item in self.human_like_dpo_data:
            # Check if we already categorized it
            if "categories" in item:
                for category in item["categories"]:
                    dpo_count_by_category[category] += 1
            else:
                # If not categorized yet (legacy data), do it now
                categories = analyze_dpo_content(item.get("question", ""), item.get("answer", ""))
                item["categories"] = categories
                item["domain"] = "human_like_behavior"  # Ensure consistent domain
                for category in categories:
                    dpo_count_by_category[category] += 1
        
        logger.info(f"Loaded {len(self.human_like_dpo_data)} examples from Human-Like DPO Dataset")
        logger.info(f"DPO data categorized into: {dict(dpo_count_by_category)}")
        
        # Add categorized DPO data to raw_data before quality checks
        # This ensures it goes through the same processing pipeline
        self.raw_data.extend(self.human_like_dpo_data)
        self.human_like_dpo_added = True  # Flag that we've added the DPO data

        # Step 1.6: Load OpenMathInstruct-1 Dataset (mathematics instruction)
        self.openmath_instruct_data = load_openmath_instruct_dataset(self.datasets_dir)
        
        if self.openmath_instruct_data:
            logger.info(f"Loaded {len(self.openmath_instruct_data)} examples from OpenMathInstruct-1 Dataset")
            
            # Add to raw_data for processing through the pipeline
            self.raw_data.extend(self.openmath_instruct_data)
            
            # Update domain data for mathematics
            for item in self.openmath_instruct_data:
                domain = item.get("domain", "mathematics")
                if domain in self.domain_data:
                    self.domain_data[domain].append(item)
                else:
                    self.domain_data["general_tech"].append(item)
        else:
            logger.info("OpenMathInstruct-1 Dataset not loaded (file may not exist yet)")
            self.openmath_instruct_data = []

        # Step 1.7: Load Theta Opinions Dataset (specialized personality data)
        self.theta_opinions_data = load_theta_opinions_dataset(self.datasets_dir)
        
        if self.theta_opinions_data:
            logger.info(f"Loaded {len(self.theta_opinions_data)} opinion entries from Theta Opinions Dataset")
            
            # Store opinions separately for specialized handling
            # Don't force into Q&A - keep as specialized format
            if "opinions" not in self.curriculum_datasets:
                self.curriculum_datasets["opinions"] = {
                    "content": self.theta_opinions_data,
                    "domain": "personality",
                    "path": "theta_opinions.json"
                }
        else:
            logger.info("Theta Opinions Dataset not loaded")
            self.theta_opinions_data = []

        # Step 1.8: Load Natural Conversation Datasets (personality-rich responses)
        self.natural_conversation_data = load_natural_conversation_datasets(self.datasets_dir)
        
        if self.natural_conversation_data:
            logger.info(f"Loaded {len(self.natural_conversation_data)} natural conversation entries")
            
            # These can be added to raw_data since they have input/response structure
            # But keep them marked as "natural" style for training
            for item in self.natural_conversation_data:
                # Convert to training format while preserving metadata
                training_item = {
                    "question": item.get("input", ""),
                    "answer": item.get("response", ""),
                    "domain": item.get("domain", "conversation"),
                    "source": item.get("source", "natural_conversation"),
                    "style": "natural"  # Mark as natural personality style
                }
                if training_item["question"] and training_item["answer"]:
                    self.raw_data.append(training_item)
        else:
            logger.info("Natural Conversation Datasets not loaded")
            self.natural_conversation_data = []

        # Step 1.9: Load Technical Patterns Datasets (code review, security, etc.)
        self.technical_patterns_data = load_technical_patterns_datasets(self.datasets_dir)
        
        if self.technical_patterns_data:
            logger.info(f"Loaded {len(self.technical_patterns_data)} technical pattern entries")
            
            # Store patterns in curriculum_datasets for specialized handling
            if "technical_patterns" not in self.curriculum_datasets:
                self.curriculum_datasets["technical_patterns"] = {
                    "content": self.technical_patterns_data,
                    "domain": "technical_patterns",
                    "path": "technical_patterns"
                }
        else:
            logger.info("Technical Patterns Datasets not loaded")
            self.technical_patterns_data = []

        # Step 2: Apply quality checks
        self.apply_quality_checks()
        
        # Step 3: Balance domains - DPO data is now included in the domain balancing
        self.balance_domains()
        
        # Step 4: Enhance with technical context
        self.enhance_with_technical_context()
        
        # Step 5: Implement curriculum learning
        self.implement_curriculum_learning()
        
        # Step 6: Augment with synthetic examples
        self.augment_with_synthetic_examples()
        
        # Step 7: Save final dataset
        count = self.save_final_dataset()
        
        end_time = logging.Formatter().converter()
        logger.info(f"Enhanced data preparation completed at {end_time}")
        logger.info(f"Final dataset contains {count} examples")
        
        # Log Human-Like DPO inclusion
        if hasattr(self, 'human_like_dpo_data') and self.human_like_dpo_data:
            logger.info(f"Included {len(self.human_like_dpo_data)} Human-Like DPO examples in the final dataset")
        else:
            logger.warning("No Human-Like DPO examples were included in the final dataset")
        
        # Log OpenMathInstruct-1 inclusion
        if hasattr(self, 'openmath_instruct_data') and self.openmath_instruct_data:
            logger.info(f"Included {len(self.openmath_instruct_data)} OpenMathInstruct-1 examples in the final dataset")
        else:
            logger.info("No OpenMathInstruct-1 examples were included (dataset may not be downloaded yet)")
        
        # Log Theta Opinions inclusion
        if hasattr(self, 'theta_opinions_data') and self.theta_opinions_data:
            logger.info(f"Included {len(self.theta_opinions_data)} Theta Opinion entries (specialized format)")
        
        # Log Natural Conversation inclusion
        if hasattr(self, 'natural_conversation_data') and self.natural_conversation_data:
            logger.info(f"Included {len(self.natural_conversation_data)} Natural Conversation examples")
        
        # Log Technical Patterns inclusion
        if hasattr(self, 'technical_patterns_data') and self.technical_patterns_data:
            logger.info(f"Included {len(self.technical_patterns_data)} Technical Pattern entries (code review, security, architecture, etc.)")
        
        return {
            "output_path": self.datasets_dir / "enhanced_training_data.json",
            "count": count,
            "stats": self.stats
        }


def main():
    """Run the enhanced data preparation with robust error handling."""
    import argparse
    import traceback
    from shutil import copy
    
    parser = argparse.ArgumentParser(description="Enhanced data preparation for Theta AI")
    parser.add_argument("--datasets-dir", type=Path, default=None, help="Path to datasets directory")
    parser.add_argument("--output", type=Path, default=None, help="Path to save the enhanced dataset")
    
    args = parser.parse_args()
    
    # Get output path
    output_path = args.output if args.output else Path("Datasets/enhanced_training_data.json")
    
    try:
        # Run enhanced data preparation
        enhancer = EnhancedDataPreparation(datasets_dir=args.datasets_dir)
        result = enhancer.run_full_preparation()
        
        print("\n====== Enhanced Data Preparation Complete ======")
        print(f"Final dataset: {result['output_path']}")
        print(f"Example count: {result['count']}")
        
        # Print key statistics
        if 'quality_check' in result['stats']:
            quality = result['stats']['quality_check']
            print(f"\nQuality Check: Retained {quality['retained_count']}/{quality['original_count']} examples ({quality['retained_count']/quality['original_count']*100:.1f}%)")
        
        if 'domain_balance' in result['stats']:
            balance = result['stats']['domain_balance']['balanced_percentages']
            print("\nDomain Balance:")
            for domain, percentage in balance.items():
                print(f"- {domain}: {percentage:.1f}%")
        
        # Print summary of diverse curriculum datasets included
        print("\nDatasets Included in Training:")
        
        # Core dataset files list
        core_datasets = [
            "Cognitive_reasoning.json", "Psychological_frameworks.json", "Conversational_dynamics.json",
            "Human_experience_simulation.json", "Humor_comprehension.json", "Cultural_contexts.json",
            "Ethical_reasoning.json", "Tactical_knowledge.json", "Interpersonal_intelligence.json",
            "Memory_simulation.json", "Emotional_learning.json", "Personal_preferences.json", 
            "Ethical_scenarios.json", "Technical_concepts.json", "Narrative_experiences.json", 
            "Emotional_intelligence.json", "human_like_dpo_dataset.json", "human_like_dpo.json",
            "openmath_instruct_1.json"
        ]
        
        # Find all datasets that were actually loaded
        datasets_dir = Path(args.datasets_dir) if args.datasets_dir else Path("Datasets")
        diverse_curriculum_dir = datasets_dir / "diverse_curriculum"
        
        # Try to find all additional dataset files that might have been included
        found_dataset_files = set()
        
        # Ensure we have a reference to the diverse_datasets list for the main loading function
        diverse_datasets = core_datasets.copy()
        
        # Add files from main directory
        try:
            all_json_files = [f for f in os.listdir(datasets_dir) if f.endswith('.json')]
            for f in all_json_files:
                found_dataset_files.add(f)
        except Exception as e:
            print(f"Error scanning datasets directory: {e}")
            
        # Add files from diverse_curriculum directory
        if diverse_curriculum_dir.exists():
            try:
                dc_json_files = [f for f in os.listdir(diverse_curriculum_dir) if f.endswith('.json')]
                for f in dc_json_files:
                    found_dataset_files.add(f)
            except Exception as e:
                print(f"Error scanning diverse_curriculum directory: {e}")
                
        # Native-format curriculum datasets are tracked in the enhancer object
        
        # Display core datasets status
        print("\n1. Core Diverse Curriculum Datasets:")
        for dataset_file in core_datasets:
            main_path_exists = (datasets_dir / dataset_file).exists()
            dc_path_exists = (diverse_curriculum_dir / dataset_file).exists() if diverse_curriculum_dir.exists() else False
            
            if main_path_exists and dc_path_exists:
                print(f"- {dataset_file} (Found in both locations)")
            elif main_path_exists:
                print(f"- {dataset_file} (Found in main directory)")
            elif dc_path_exists:
                print(f"- {dataset_file} (Found in diverse_curriculum directory)")
            else:
                print(f"- {dataset_file} (NOT FOUND)")
        
        # Show additional datasets that were found
        print("\n2. Additional Dataset Files Found:")
        extra_datasets = [f for f in found_dataset_files if f not in core_datasets 
                        and not any(p in f.lower() for p in ['processed_data.json', 'enhanced_', 'combined_', 'synthetic_'])]
        
        if extra_datasets:
            for dataset_file in sorted(extra_datasets):
                print(f"- {dataset_file}")
        else:
            print("- None")
            
        # Show native curriculum datasets information
        if hasattr(enhancer, 'curriculum_datasets') and enhancer.curriculum_datasets:
            print(f"\n3. Native Curriculum Datasets: {len(enhancer.curriculum_datasets)} datasets")
            for name, info in enhancer.curriculum_datasets.items():
                print(f"- {name} (domain: {info['domain']})")
        
        # Show domain statistics
        if result['stats'].get('synthetic_augmentation', {}).get('human_like_dpo_count', 0) > 0:
            dpo_count = result['stats']['synthetic_augmentation']['human_like_dpo_count']
            print(f"\n4. Human-Like DPO Integration: {dpo_count} examples included")
        
        # Show OpenMathInstruct statistics
        if result['stats'].get('synthetic_augmentation', {}).get('openmath_instruct_count', 0) > 0:
            math_count = result['stats']['synthetic_augmentation']['openmath_instruct_count']
            print(f"\n5. OpenMathInstruct-1 Integration: {math_count} math instruction examples included")
        
        # Show Theta Opinions statistics
        if result['stats'].get('synthetic_augmentation', {}).get('theta_opinions_count', 0) > 0:
            opinions_count = result['stats']['synthetic_augmentation']['theta_opinions_count']
            print(f"\n6. Theta Opinions (Specialized): {opinions_count} personality-driven opinion entries")
        
        # Show Natural Conversation statistics
        if result['stats'].get('synthetic_augmentation', {}).get('natural_conversation_count', 0) > 0:
            conv_count = result['stats']['synthetic_augmentation']['natural_conversation_count']
            print(f"\n7. Natural Conversations: {conv_count} personality-rich conversation examples")
                
        print("\nReady for training!")
        return 0
    except Exception as e:
        # Log error and provide fallback
        logger.error(f"Enhanced data preparation failed: {e}")
        logger.error(traceback.format_exc())
        
        print("\n====== ERROR: Enhanced Data Preparation Failed ======")
        print(f"Error: {e}")
        
        # Fallback to processed_data.json
        processed_path = Path("Datasets/processed_data.json")
        if processed_path.exists():
            print(f"\nFallback: Using processed_data.json as a fallback...")
            try:
                # Load and save processed data
                with open(processed_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Save to output path
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f)
                    
                print(f"Fallback successful! Saved {len(data)} examples to {output_path}")
                print("\nContinuing with training using the fallback data.")
                return 0
            except Exception as fallback_error:
                print(f"Fallback also failed: {fallback_error}")
        else:
            print(f"Fallback failed: processed_data.json not found")
            
        print("\nCannot continue with training.")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
