#!/usr/bin/env python3
"""
OpenAssistant OASST1 Dataset Downloader and Processor

Downloads the OpenAssistant/oasst1 dataset from HuggingFace and converts it
to the Q&A format compatible with Theta AI's data_processor.py and 
prepare_data_for_training.py pipeline.

Dataset: https://huggingface.co/datasets/OpenAssistant/oasst1
- Contains ~161k messages in conversation trees
- Multilingual (we filter for English by default)
- High-quality human-generated assistant conversations
"""

import json
import os
import argparse
import logging
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any, Optional
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Output directory
DEFAULT_OUTPUT_DIR = Path("G:/Theta AI/Datasets")


def download_oasst1_dataset(cache_dir: Optional[Path] = None):
    """
    Download the OpenAssistant oasst1 dataset from HuggingFace.
    
    Args:
        cache_dir: Optional cache directory for HuggingFace datasets
        
    Returns:
        The loaded dataset object
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets library not installed. Installing...")
        import subprocess
        subprocess.check_call(["pip", "install", "datasets"])
        from datasets import load_dataset
    
    logger.info("Downloading OpenAssistant/oasst1 dataset from HuggingFace...")
    
    # Set cache directory if provided
    if cache_dir:
        os.environ["HF_HOME"] = str(cache_dir)
    
    # Load the dataset
    dataset = load_dataset("OpenAssistant/oasst1", trust_remote_code=True)
    
    logger.info(f"Dataset loaded successfully!")
    logger.info(f"Train split: {len(dataset['train'])} messages")
    logger.info(f"Validation split: {len(dataset['validation'])} messages")
    
    return dataset


def build_conversation_trees(messages: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Build conversation trees from flat message list.
    
    The oasst1 dataset stores messages with parent_id references.
    This function reconstructs the conversation trees.
    
    Args:
        messages: List of message dictionaries
        
    Returns:
        Dictionary mapping root message IDs to their conversation trees
    """
    # Index messages by ID
    message_by_id = {msg['message_id']: msg for msg in messages}
    
    # Find children for each message
    children = defaultdict(list)
    root_messages = []
    
    for msg in messages:
        parent_id = msg.get('parent_id')
        if parent_id is None:
            root_messages.append(msg)
        else:
            children[parent_id].append(msg)
    
    # Sort children by rank (quality score)
    for parent_id in children:
        children[parent_id].sort(key=lambda x: x.get('rank', 0) or 0)
    
    return {
        'roots': root_messages,
        'children': children,
        'message_by_id': message_by_id
    }


def extract_qa_pairs_from_tree(
    tree_data: Dict,
    language_filter: str = "en",
    min_score: float = 0.0,
    include_metadata: bool = True
) -> List[Dict[str, Any]]:
    """
    Extract Q&A pairs from conversation trees.
    
    For each prompter message (question), we find the best assistant response.
    
    Args:
        tree_data: Conversation tree data from build_conversation_trees
        language_filter: Language code to filter (default: "en" for English)
        min_score: Minimum quality score threshold
        include_metadata: Whether to include domain/source metadata
        
    Returns:
        List of Q&A pairs in Theta AI format
    """
    qa_pairs = []
    roots = tree_data['roots']
    children = tree_data['children']
    message_by_id = tree_data['message_by_id']
    
    def get_best_response(parent_id: str) -> Optional[Dict]:
        """Get the highest-ranked assistant response to a message."""
        responses = children.get(parent_id, [])
        assistant_responses = [r for r in responses if r.get('role') == 'assistant']
        
        if not assistant_responses:
            return None
        
        # Sort by rank (lower is better) then by labels score
        def score_response(r):
            rank = r.get('rank') or float('inf')
            labels = r.get('labels', {}) or {}
            quality = labels.get('quality', {}).get('value', 0) or 0
            return (rank, -quality)
        
        assistant_responses.sort(key=score_response)
        return assistant_responses[0]
    
    def extract_conversation_chain(msg_id: str, depth: int = 0, max_depth: int = 10) -> List[Dict]:
        """Extract a conversation chain starting from a message."""
        if depth > max_depth:
            return []
        
        msg = message_by_id.get(msg_id)
        if not msg:
            return []
        
        chain = [msg]
        
        # Get best response
        best_response = get_best_response(msg_id)
        if best_response:
            chain.append(best_response)
            
            # Continue the chain if there are follow-up questions
            follow_ups = children.get(best_response['message_id'], [])
            prompter_follow_ups = [f for f in follow_ups if f.get('role') == 'prompter']
            
            if prompter_follow_ups:
                # Take the best follow-up
                prompter_follow_ups.sort(key=lambda x: x.get('rank', 0) or 0)
                chain.extend(extract_conversation_chain(
                    prompter_follow_ups[0]['message_id'], 
                    depth + 1, 
                    max_depth
                ))
        
        return chain
    
    # Process each root message (initial prompts)
    for root in tqdm(roots, desc="Extracting Q&A pairs"):
        # Filter by language
        lang = root.get('lang', 'en')
        if language_filter and lang != language_filter:
            continue
        
        # Filter by role (should be prompter for root)
        if root.get('role') != 'prompter':
            continue
        
        # Get the conversation chain
        chain = extract_conversation_chain(root['message_id'])
        
        # Convert chain to Q&A pairs
        for i in range(0, len(chain) - 1, 2):
            if i + 1 >= len(chain):
                break
            
            question_msg = chain[i]
            answer_msg = chain[i + 1]
            
            # Validate roles
            if question_msg.get('role') != 'prompter' or answer_msg.get('role') != 'assistant':
                continue
            
            question_text = question_msg.get('text', '').strip()
            answer_text = answer_msg.get('text', '').strip()
            
            # Skip empty or too short
            if len(question_text) < 10 or len(answer_text) < 20:
                continue
            
            # Create Q&A pair
            qa_pair = {
                "question": question_text,
                "answer": answer_text
            }
            
            if include_metadata:
                # Determine domain based on content analysis
                domain = categorize_content(question_text, answer_text)
                qa_pair["domain"] = domain
                qa_pair["source"] = "openassistant_oasst1"
                
                # Add quality indicators
                labels = answer_msg.get('labels', {}) or {}
                if labels:
                    quality = labels.get('quality', {}).get('value')
                    if quality:
                        qa_pair["quality_score"] = quality
                
                # Add conversation context indicator
                if i > 0:
                    qa_pair["is_followup"] = True
                    qa_pair["turn_number"] = i // 2 + 1
            
            qa_pairs.append(qa_pair)
    
    return qa_pairs


def categorize_content(question: str, answer: str) -> str:
    """
    Categorize the Q&A pair into a domain based on content analysis.
    
    Args:
        question: The question text
        answer: The answer text
        
    Returns:
        Domain category string
    """
    combined = (question + " " + answer).lower()
    
    # Technical/Programming keywords
    programming_keywords = [
        'code', 'python', 'javascript', 'function', 'class', 'variable',
        'programming', 'algorithm', 'api', 'database', 'sql', 'html', 'css',
        'react', 'node', 'java', 'c++', 'rust', 'golang', 'typescript',
        'debug', 'error', 'exception', 'compile', 'runtime', 'syntax'
    ]
    
    # Cybersecurity keywords
    security_keywords = [
        'security', 'hack', 'vulnerability', 'encryption', 'password',
        'firewall', 'malware', 'virus', 'phishing', 'authentication',
        'cyber', 'breach', 'threat', 'attack', 'penetration'
    ]
    
    # Data science/ML keywords
    data_science_keywords = [
        'machine learning', 'deep learning', 'neural network', 'ai',
        'artificial intelligence', 'model', 'training', 'dataset',
        'tensorflow', 'pytorch', 'sklearn', 'pandas', 'numpy',
        'regression', 'classification', 'clustering', 'nlp'
    ]
    
    # Networking/Cloud keywords
    networking_keywords = [
        'network', 'server', 'cloud', 'aws', 'azure', 'docker',
        'kubernetes', 'devops', 'deployment', 'container', 'vm',
        'ip address', 'dns', 'http', 'tcp', 'udp', 'protocol'
    ]
    
    # Conversational/General keywords
    conversation_keywords = [
        'how are you', 'what do you think', 'tell me about yourself',
        'help me', 'can you explain', 'what is your opinion',
        'recommend', 'suggest', 'advice'
    ]
    
    # Count keyword matches
    scores = {
        'programming': sum(1 for kw in programming_keywords if kw in combined),
        'cybersecurity': sum(1 for kw in security_keywords if kw in combined),
        'data_science': sum(1 for kw in data_science_keywords if kw in combined),
        'networking': sum(1 for kw in networking_keywords if kw in combined),
        'conversational': sum(1 for kw in conversation_keywords if kw in combined)
    }
    
    # Return highest scoring domain, or general_tech as default
    max_score = max(scores.values())
    if max_score == 0:
        return 'general_tech'
    
    for domain, score in scores.items():
        if score == max_score:
            return domain
    
    return 'general_tech'


def save_dataset(
    qa_pairs: List[Dict],
    output_path: Path,
    split_by_domain: bool = False
) -> Dict[str, int]:
    """
    Save the processed dataset to JSON file(s).
    
    Args:
        qa_pairs: List of Q&A pairs
        output_path: Path to save the main output file
        split_by_domain: Whether to also create domain-specific files
        
    Returns:
        Dictionary with save statistics
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    stats = {"total": len(qa_pairs)}
    
    # Save main file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved {len(qa_pairs)} Q&A pairs to {output_path}")
    
    # Optionally split by domain
    if split_by_domain:
        domain_data = defaultdict(list)
        for qa in qa_pairs:
            domain = qa.get('domain', 'general_tech')
            domain_data[domain].append(qa)
        
        domain_dir = output_path.parent / "openassistant_domains"
        domain_dir.mkdir(exist_ok=True)
        
        for domain, data in domain_data.items():
            domain_file = domain_dir / f"oasst1_{domain}.json"
            with open(domain_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            stats[domain] = len(data)
            logger.info(f"  - {domain}: {len(data)} pairs -> {domain_file}")
    
    return stats


def create_enhanced_version(
    qa_pairs: List[Dict],
    output_path: Path
) -> List[Dict]:
    """
    Create an enhanced version with additional processing for prepare_data_for_training.py.
    
    Adds:
    - Complexity scores for curriculum learning
    - Technical term detection
    - Quality filtering
    
    Args:
        qa_pairs: Original Q&A pairs
        output_path: Path to save enhanced version
        
    Returns:
        Enhanced Q&A pairs
    """
    import re
    
    enhanced_pairs = []
    
    for qa in tqdm(qa_pairs, desc="Enhancing dataset"):
        question = qa.get('question', '')
        answer = qa.get('answer', '')
        
        # Skip low quality
        if len(question.split()) < 3 or len(answer.split()) < 10:
            continue
        
        # Calculate complexity score
        q_words = len(question.split())
        a_words = len(answer.split())
        
        # Count technical terms
        tech_pattern = r'\b(function|class|method|variable|algorithm|api|database|server|network|security|encryption|protocol|framework|library|module|package|interface|implementation)\b'
        tech_terms = re.findall(tech_pattern, (question + ' ' + answer).lower())
        
        # Count code blocks
        code_blocks = len(re.findall(r'```[\s\S]*?```', answer))
        
        complexity = (
            q_words * 0.1 +
            a_words * 0.05 +
            len(tech_terms) * 0.5 +
            code_blocks * 2.0
        )
        
        enhanced_qa = {
            **qa,
            "complexity_score": round(complexity, 2),
            "has_code": code_blocks > 0,
            "technical_terms": list(set(tech_terms))[:10]  # Limit to 10 terms
        }
        
        enhanced_pairs.append(enhanced_qa)
    
    # Sort by complexity for curriculum learning
    enhanced_pairs.sort(key=lambda x: x.get('complexity_score', 0))
    
    # Save enhanced version
    enhanced_path = output_path.parent / "openassistant_oasst1_enhanced.json"
    with open(enhanced_path, 'w', encoding='utf-8') as f:
        json.dump(enhanced_pairs, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved {len(enhanced_pairs)} enhanced Q&A pairs to {enhanced_path}")
    
    return enhanced_pairs


def integrate_with_data_processor(output_dir: Path):
    """
    Add the OpenAssistant dataset to data_processor.py's loading list.
    
    This function provides instructions for integration.
    """
    logger.info("\n" + "="*60)
    logger.info("INTEGRATION INSTRUCTIONS")
    logger.info("="*60)
    logger.info("""
To integrate with data_processor.py, add this to the dataset paths section:

    # OpenAssistant OASST1 dataset (high-quality assistant conversations)
    openassistant_oasst1_path = datasets_dir / "openassistant_oasst1.json"

And add this to the loading section:

    openassistant_oasst1_qa = load_json_dataset(openassistant_oasst1_path)

And include it in the combined dataset:

    combined_qa.extend(openassistant_oasst1_qa)

The dataset is already in the correct Q&A format with domain tags.
""")
    logger.info("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Download and process OpenAssistant OASST1 dataset for Theta AI"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("G:/Theta AI/cache"),
        help="HuggingFace cache directory"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language filter (default: en for English, use 'all' for all languages)"
    )
    parser.add_argument(
        "--split-domains",
        action="store_true",
        help="Create separate files for each domain"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of Q&A pairs to extract (default: all)"
    )
    parser.add_argument(
        "--enhanced",
        action="store_true",
        default=True,
        help="Create enhanced version with complexity scores (default: True)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("OpenAssistant OASST1 Dataset Downloader")
    logger.info("="*60)
    
    # Download dataset
    dataset = download_oasst1_dataset(cache_dir=args.cache_dir)
    
    # Combine train and validation splits
    all_messages = list(dataset['train']) + list(dataset['validation'])
    logger.info(f"Total messages to process: {len(all_messages)}")
    
    # Build conversation trees
    logger.info("Building conversation trees...")
    tree_data = build_conversation_trees(all_messages)
    logger.info(f"Found {len(tree_data['roots'])} root conversations")
    
    # Extract Q&A pairs
    language_filter = None if args.language == 'all' else args.language
    qa_pairs = extract_qa_pairs_from_tree(
        tree_data,
        language_filter=language_filter,
        include_metadata=True
    )
    
    logger.info(f"Extracted {len(qa_pairs)} Q&A pairs")
    
    # Limit samples if specified
    if args.max_samples and len(qa_pairs) > args.max_samples:
        import random
        qa_pairs = random.sample(qa_pairs, args.max_samples)
        logger.info(f"Limited to {args.max_samples} samples")
    
    # Save main dataset
    output_path = args.output_dir / "openassistant_oasst1.json"
    stats = save_dataset(
        qa_pairs,
        output_path,
        split_by_domain=args.split_domains
    )
    
    # Create enhanced version
    if args.enhanced:
        enhanced_pairs = create_enhanced_version(qa_pairs, output_path)
        stats["enhanced"] = len(enhanced_pairs)
    
    # Print domain distribution
    logger.info("\n" + "="*60)
    logger.info("DOMAIN DISTRIBUTION")
    logger.info("="*60)
    domain_counts = defaultdict(int)
    for qa in qa_pairs:
        domain_counts[qa.get('domain', 'general_tech')] += 1
    
    for domain, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
        pct = (count / len(qa_pairs)) * 100
        logger.info(f"  {domain}: {count} ({pct:.1f}%)")
    
    # Integration instructions
    integrate_with_data_processor(args.output_dir)
    
    logger.info("="*60)
    logger.info("DOWNLOAD COMPLETE")
    logger.info("="*60)
    logger.info(f"Main dataset: {output_path}")
    if args.enhanced:
        logger.info(f"Enhanced dataset: {args.output_dir / 'openassistant_oasst1_enhanced.json'}")
    logger.info(f"Total Q&A pairs: {len(qa_pairs)}")
    logger.info("="*60)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
