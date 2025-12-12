"""
Main Knowledge Base Enhancement Script for Theta AI

This script coordinates all knowledge base enhancements:
1. Enhanced technical datasets
2. Retrieval augmented generation
3. Factual consistency checking
4. Synthetic data generation
5. Technical knowledge graphs
6. Technical term embeddings
7. Feedback-based improvements
"""

import os
import json
import logging
import argparse
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set, Optional

# Import enhancement modules
from knowledge_enhancer import KnowledgeEnhancer
from retrieval_augmentation import RetrievalAugmentation
from consistency_checker import EnhancedConsistencyChecker
from synthetic_data_generator import SyntheticDataGenerator
from knowledge_graph import KnowledgeGraph
from technical_embeddings import TechnicalEmbeddings
from feedback_loop import FeedbackLoop

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler("knowledge_enhancement.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class KnowledgeBaseEnhancer:
    """Coordinates all knowledge base enhancements."""
    
    def __init__(self, datasets_dir: Path):
        """
        Initialize the knowledge base enhancer.
        
        Args:
            datasets_dir: Path to the datasets directory
        """
        self.datasets_dir = datasets_dir
        
        # Create enhancement directories
        self.enhanced_dir = datasets_dir / "enhanced"
        os.makedirs(self.enhanced_dir, exist_ok=True)
        
        # Initialize enhancement components
        self.knowledge_enhancer = KnowledgeEnhancer(datasets_dir)
        self.retrieval_system = RetrievalAugmentation(datasets_dir)
        self.consistency_checker = EnhancedConsistencyChecker(datasets_dir)
        self.synthetic_generator = SyntheticDataGenerator(datasets_dir)
        self.knowledge_graph = KnowledgeGraph(datasets_dir)
        self.technical_embeddings = TechnicalEmbeddings(datasets_dir)
        self.feedback_loop = FeedbackLoop(datasets_dir)
        
        # Define domains
        self.domains = [
            "cybersecurity", "programming", "networking", 
            "cloud_computing", "data_science", "general_tech"
        ]
    
    def run_full_enhancement(self, steps: List[str] = None):
        """
        Run a full knowledge base enhancement process.
        
        Args:
            steps: Specific steps to run, or None for all steps
        """
        start_time = time.time()
        logger.info("Starting knowledge base enhancement process")
        
        # Define all enhancement steps
        all_steps = [
            "curated_datasets",
            "retrieval_system",
            "consistency_checking",
            "synthetic_data",
            "knowledge_graphs", 
            "technical_embeddings",
            "feedback_processing",
            "combine_datasets"
        ]
        
        # Determine which steps to run
        steps_to_run = steps if steps else all_steps
        logger.info(f"Running enhancement steps: {', '.join(steps_to_run)}")
        
        # Step 1: Create curated datasets
        if "curated_datasets" in steps_to_run:
            logger.info("Step 1: Creating curated technical datasets")
            self.knowledge_enhancer.create_all_resources(self.domains)
        
        # Step 2: Set up retrieval system
        if "retrieval_system" in steps_to_run:
            logger.info("Step 2: Setting up retrieval-augmented generation")
            self.retrieval_system.process_knowledge_base()
        
        # Step 3: Set up consistency checking
        if "consistency_checking" in steps_to_run:
            logger.info("Step 3: Setting up factual consistency checking")
            # Add some verified facts for each domain
            for domain in self.domains:
                self._add_sample_facts(domain)
        
        # Step 4: Generate synthetic data
        if "synthetic_data" in steps_to_run:
            logger.info("Step 4: Generating synthetic training data")
            self.synthetic_generator.generate_all_domains(count_per_domain=200)
        
        # Step 5: Create knowledge graphs
        if "knowledge_graphs" in steps_to_run:
            logger.info("Step 5: Creating technical knowledge graphs")
            for domain in self.domains:
                # Get QA data for domain
                qa_data = self._get_domain_qa_data(domain)
                if qa_data:
                    self.knowledge_graph.create_graph_from_qa(domain, qa_data)
                    self.knowledge_graph.visualize_graph(domain)
        
        # Step 6: Create technical embeddings
        if "technical_embeddings" in steps_to_run:
            logger.info("Step 6: Creating technical term embeddings")
            self.technical_embeddings.collect_technical_terms()
            self.technical_embeddings.create_technical_embeddings()
        
        # Step 7: Process feedback and address knowledge gaps
        if "feedback_processing" in steps_to_run:
            logger.info("Step 7: Processing feedback and addressing knowledge gaps")
            self.feedback_loop.run_improvement_cycle()
        
        # Step 8: Combine all datasets
        if "combine_datasets" in steps_to_run:
            logger.info("Step 8: Combining all enhanced datasets")
            combined_dataset = self._combine_enhanced_datasets()
            
            # Save combined dataset with UTF-8 encoding
            combined_path = self.enhanced_dir / "combined_enhanced.json"
            with open(combined_path, 'w', encoding='utf-8') as f:
                json.dump(combined_dataset, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Saved combined dataset with {len(combined_dataset)} examples to {combined_path}")
        
        # Calculate and log duration
        duration = time.time() - start_time
        logger.info(f"Knowledge base enhancement completed in {duration:.2f} seconds")
        
        return {
            "duration": duration,
            "combined_dataset_size": len(combined_dataset) if "combine_datasets" in steps_to_run else 0
        }
    
    def _add_sample_facts(self, domain: str):
        """
        Add sample verified facts for consistency checking.
        
        Args:
            domain: Domain to add facts for
        """
        facts = []
        
        if domain == "cybersecurity":
            facts = [
                ("AES", "AES is a symmetric encryption algorithm with key sizes of 128, 192, or 256 bits."),
                ("phishing", "Phishing is a type of social engineering attack that uses fraudulent messages to trick users into revealing sensitive information."),
                ("firewall", "A firewall is a network security device that monitors and filters incoming and outgoing network traffic.")
            ]
        elif domain == "programming":
            facts = [
                ("Python", "Python is an interpreted, high-level, general-purpose programming language created by Guido van Rossum."),
                ("JavaScript", "JavaScript is a high-level, interpreted programming language that conforms to the ECMAScript specification."),
                ("binary search", "Binary search is an algorithm with O(log n) time complexity that finds the position of a target value in a sorted array.")
            ]
        elif domain == "networking":
            facts = [
                ("TCP", "TCP is a connection-oriented protocol that provides reliable, ordered, and error-checked delivery of data."),
                ("DNS", "DNS (Domain Name System) translates human-readable domain names to IP addresses."),
                ("DHCP", "DHCP (Dynamic Host Configuration Protocol) automatically assigns IP addresses to devices on a network.")
            ]
        
        # Add facts to consistency checker
        for subject, content in facts:
            self.consistency_checker.add_verified_fact(domain, subject, content)
    
    def _get_domain_qa_data(self, domain: str) -> List[Dict]:
        """
        Get QA data for a specific domain.
        
        Args:
            domain: Domain to get data for
            
        Returns:
            List of QA pairs
        """
        qa_data = []
        
        # Check domain-specific datasets
        domain_files = [
            f"{domain}.json",
            f"{domain}_qa.json",
            f"{domain}_synthetic.json",
            f"{domain}_feedback.json"
        ]
        
        # Get data from different directories
        dirs_to_check = [
            self.datasets_dir,
            self.datasets_dir / "curated_qa",
            self.datasets_dir / "synthetic_data",
            self.datasets_dir / "feedback_based"
        ]
        
        # Process all potential domain files
        for directory in dirs_to_check:
            if not directory.exists():
                continue
                
            for filename in domain_files:
                file_path = directory / filename
                if file_path.exists():
                    try:
                        # First try with UTF-8 encoding and error handling
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                                data = json.load(f)
                        except json.JSONDecodeError:
                            # If that fails, try a more aggressive approach with binary reading
                            logger.info(f"JSON decode error with utf-8 for {file_path}, trying alternative approach")
                            with open(file_path, 'rb') as f:
                                content = f.read()
                            
                            # Replace or remove non-UTF-8 bytes
                            cleaned_content = b''
                            for i in range(0, len(content)):
                                byte = content[i:i+1]
                                try:
                                    byte.decode('utf-8')
                                    cleaned_content += byte
                                except UnicodeDecodeError:
                                    # Replace problematic bytes
                                    cleaned_content += b'\xEF\xBF\xBD'  # UTF-8 replacement character
                            
                            # Try parsing the cleaned content
                            data = json.loads(cleaned_content.decode('utf-8'))
                            
                        if isinstance(data, list):
                            qa_data.extend(data)
                            logger.info(f"Added {len(data)} QA pairs from {file_path}")
                    except Exception as e:
                        logger.error(f"Error loading {file_path}: {e}")
        
        return qa_data
    
    def _combine_enhanced_datasets(self) -> List[Dict]:
        """
        Combine all enhanced datasets into a single dataset.
        
        Returns:
            Combined dataset
        """
        combined_data = []
        
        # Directories to combine data from
        dirs_to_combine = [
            self.datasets_dir / "curated_qa",
            self.datasets_dir / "synthetic_data",
            self.datasets_dir / "feedback_based",
            self.datasets_dir / "case_studies"
        ]
        
        # Process all JSON files in these directories
        for directory in dirs_to_combine:
            if not directory.exists():
                continue
                
            for file_path in directory.glob("*.json"):
                try:
                    # First try with UTF-8 encoding and error handling
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                            data = json.load(f)
                    except json.JSONDecodeError:
                        # If that fails, try a more aggressive approach with binary reading
                        logger.info(f"JSON decode error with utf-8 for {file_path}, trying alternative approach")
                        with open(file_path, 'rb') as f:
                            content = f.read()
                        
                        # Replace or remove non-UTF-8 bytes
                        cleaned_content = b''
                        for i in range(0, len(content)):
                            byte = content[i:i+1]
                            try:
                                byte.decode('utf-8')
                                cleaned_content += byte
                            except UnicodeDecodeError:
                                # Replace problematic bytes
                                cleaned_content += b'\xEF\xBF\xBD'  # UTF-8 replacement character
                        
                        # Try parsing the cleaned content
                        data = json.loads(cleaned_content.decode('utf-8'))
                    
                    if isinstance(data, list):
                        # Add source metadata
                        for item in data:
                            if isinstance(item, dict):
                                if "metadata" not in item:
                                    item["metadata"] = {}
                                item["metadata"]["source_file"] = str(file_path.relative_to(self.datasets_dir))
                        
                        combined_data.extend(data)
                        logger.info(f"Added {len(data)} items from {file_path}")
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
        
        # Also check for feedback-generated case studies
        case_studies_dir = self.datasets_dir / "case_studies"
        if case_studies_dir.exists():
            for file_path in case_studies_dir.glob("*.json"):
                try:
                    # First try with UTF-8 encoding and error handling
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                            data = json.load(f)
                    except json.JSONDecodeError:
                        # If that fails, try a more aggressive approach with binary reading
                        logger.info(f"JSON decode error with utf-8 for {file_path}, trying alternative approach")
                        with open(file_path, 'rb') as f:
                            content = f.read()
                        
                        # Replace or remove non-UTF-8 bytes
                        cleaned_content = b''
                        for i in range(0, len(content)):
                            byte = content[i:i+1]
                            try:
                                byte.decode('utf-8')
                                cleaned_content += byte
                            except UnicodeDecodeError:
                                # Replace problematic bytes
                                cleaned_content += b'\xEF\xBF\xBD'  # UTF-8 replacement character
                        
                        # Try parsing the cleaned content
                        data = json.loads(cleaned_content.decode('utf-8'))
                    
                    # Process case study QA pairs
                    if isinstance(data, dict) and "qa_pairs" in data:
                        for qa_pair in data["qa_pairs"]:
                            qa_pair["domain"] = data.get("domain", "general_tech")
                            qa_pair["metadata"] = {
                                "source": "case_study",
                                "case_study": data.get("title", ""),
                                "source_file": str(file_path.relative_to(self.datasets_dir))
                            }
                            combined_data.append(qa_pair)
                except Exception as e:
                    logger.error(f"Error loading case study {file_path}: {e}")
        
        return combined_data

def main():
    """Main function to run knowledge base enhancements."""
    parser = argparse.ArgumentParser(description="Enhance Theta AI Knowledge Base")
    parser.add_argument("--steps", nargs="+", help="Specific enhancement steps to run")
    parser.add_argument("--datasets_dir", default=None, help="Path to datasets directory")
    args = parser.parse_args()
    
    # Get project root and datasets directory
    project_root = Path(__file__).resolve().parent.parent.parent
    datasets_dir = Path(args.datasets_dir) if args.datasets_dir else project_root / "Datasets"
    
    # Create and run knowledge base enhancer
    enhancer = KnowledgeBaseEnhancer(datasets_dir)
    results = enhancer.run_full_enhancement(steps=args.steps)
    
    # Print summary
    print(f"\nKnowledge Base Enhancement Summary:")
    print(f"- Duration: {results['duration']:.2f} seconds")
    if "combine_datasets" in (args.steps or []):
        print(f"- Combined Dataset Size: {results['combined_dataset_size']} examples")
    print("\nEnhancement complete. The knowledge base is ready for training.")

if __name__ == "__main__":
    main()
