"""
Dataset balancing and filtering utilities for Theta AI.
Provides functions to analyze, balance, and filter training datasets.
"""

import json
import os
import re
import random
import numpy as np
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetBalancer:
    """Class for analyzing and balancing datasets."""
    
    def __init__(self, data_path=None, data=None):
        """
        Initialize the dataset balancer.
        
        Args:
            data_path: Path to the dataset JSON file (optional)
            data: Dataset as a list of dictionaries (optional)
        """
        if data_path:
            with open(data_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            self.data_path = data_path
        elif data:
            self.data = data
            self.data_path = None
        else:
            self.data = []
            self.data_path = None
            
        self.domain_patterns = {
            'CYBERSECURITY': r'\b(cyber|security|attack|vulnerability|exploit|threat|malware|ransomware|phishing)\b',
            'SOFTWARE_DEV': r'\b(code|software|programming|development|algorithm|function|variable|api|database)\b',
            'CLOUD': r'\b(cloud|aws|azure|google cloud|serverless|container|kubernetes|docker|microservice)\b',
            'NETWORKING': r'\b(network|tcp|ip|dns|router|switch|firewall|vpn|subnet|protocol)\b',
            'AI': r'\b(ai|machine learning|deep learning|neural network|model|training|inference|algorithm)\b'
        }
        
        # Categories and topics statistics
        self.categories = {}
        self.topic_distribution = {}
        self.length_stats = {}
        
        # Analysis results
        self.analysis_complete = False
        
    def analyze_dataset(self):
        """Analyze the dataset and compute statistics."""
        if not self.data:
            logger.error("No data available for analysis")
            return
            
        # Initialize counters
        categories = defaultdict(int)
        topic_distribution = defaultdict(int)
        question_lengths = []
        answer_lengths = []
        
        # Extract domains mentioned in questions
        for item in self.data:
            question = item.get('question', '')
            answer = item.get('answer', '')
            
            # Count domains in questions
            domains_found = self._extract_domains(question)
            for domain in domains_found:
                categories[domain] += 1
                
            # Process topics (extract keywords)
            topics = self._extract_topics(question)
            for topic in topics:
                topic_distribution[topic] += 1
                
            # Record lengths
            question_lengths.append(len(question.split()))
            answer_lengths.append(len(answer.split()))
            
        # Store results
        self.categories = dict(categories)
        self.topic_distribution = dict(topic_distribution)
        
        # Compute length statistics
        self.length_stats = {
            'question': {
                'mean': np.mean(question_lengths),
                'median': np.median(question_lengths),
                'min': min(question_lengths),
                'max': max(question_lengths),
                'std': np.std(question_lengths)
            },
            'answer': {
                'mean': np.mean(answer_lengths),
                'median': np.median(answer_lengths),
                'min': min(answer_lengths),
                'max': max(answer_lengths),
                'std': np.std(answer_lengths)
            }
        }
        
        self.analysis_complete = True
        
        # Log basic statistics
        logger.info(f"Dataset analysis complete. {len(self.data)} examples analyzed.")
        logger.info(f"Categories distribution: {self.categories}")
        logger.info(f"Question length (words): mean={self.length_stats['question']['mean']:.1f}, "
                   f"median={self.length_stats['question']['median']}, "
                   f"range={self.length_stats['question']['min']}-{self.length_stats['question']['max']}")
        logger.info(f"Answer length (words): mean={self.length_stats['answer']['mean']:.1f}, "
                   f"median={self.length_stats['answer']['median']}, "
                   f"range={self.length_stats['answer']['min']}-{self.length_stats['answer']['max']}")
        
        return {
            'categories': self.categories,
            'topic_distribution': self.topic_distribution,
            'length_stats': self.length_stats
        }
    
    def visualize_distribution(self, output_path=None):
        """
        Visualize the dataset distribution.
        
        Args:
            output_path: Path to save visualizations
        """
        if not self.analysis_complete:
            self.analyze_dataset()
            
        # Create directory for visualizations if needed
        if output_path:
            os.makedirs(output_path, exist_ok=True)
            
        # Plot domain distribution
        plt.figure(figsize=(10, 6))
        domains = list(self.categories.keys())
        counts = list(self.categories.values())
        plt.bar(domains, counts)
        plt.title('Domain Distribution in Dataset')
        plt.xlabel('Domain')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(os.path.join(output_path, 'domain_distribution.png'))
            plt.close()
        else:
            plt.show()
            
        # Plot topic distribution (top 20)
        top_topics = sorted(self.topic_distribution.items(), key=lambda x: x[1], reverse=True)[:20]
        topics, counts = zip(*top_topics) if top_topics else ([], [])
        
        plt.figure(figsize=(12, 6))
        plt.bar(topics, counts)
        plt.title('Top 20 Topics in Dataset')
        plt.xlabel('Topic')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(os.path.join(output_path, 'topic_distribution.png'))
            plt.close()
        else:
            plt.show()
            
        # Plot length distributions
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        question_lengths = [len(item.get('question', '').split()) for item in self.data]
        plt.hist(question_lengths, bins=20)
        plt.title('Question Length Distribution (words)')
        plt.xlabel('Length')
        plt.ylabel('Count')
        
        plt.subplot(1, 2, 2)
        answer_lengths = [len(item.get('answer', '').split()) for item in self.data]
        plt.hist(answer_lengths, bins=20)
        plt.title('Answer Length Distribution (words)')
        plt.xlabel('Length')
        plt.ylabel('Count')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(os.path.join(output_path, 'length_distribution.png'))
            plt.close()
        else:
            plt.show()
    
    def balance_domains(self, target_counts=None, min_count=None):
        """
        Balance the dataset by domains.
        
        Args:
            target_counts: Dictionary of target counts per domain
            min_count: Minimum count per domain (if target_counts not provided)
            
        Returns:
            Balanced dataset
        """
        if not self.analysis_complete:
            self.analyze_dataset()
            
        # Group examples by domain
        domain_examples = defaultdict(list)
        unclassified_examples = []
        
        for idx, item in enumerate(self.data):
            question = item.get('question', '')
            domains = self._extract_domains(question)
            
            if domains:
                # Assign to first detected domain (could be improved)
                domain_examples[domains[0]].append(idx)
            else:
                unclassified_examples.append(idx)
                
        # Determine target counts for each domain
        if not target_counts:
            if min_count:
                # Use specified minimum count
                target_counts = {domain: max(len(examples), min_count) 
                                for domain, examples in domain_examples.items()}
            else:
                # Use maximum count across domains
                max_count = max(len(examples) for examples in domain_examples.values())
                target_counts = {domain: max_count for domain in domain_examples}
                
        # Balance each domain
        balanced_indices = []
        
        for domain, examples in domain_examples.items():
            target = target_counts.get(domain, len(examples))
            
            if len(examples) >= target:
                # Downsample if needed
                balanced_indices.extend(random.sample(examples, target))
            else:
                # Upsample if needed
                balanced_indices.extend(examples)  # Add all existing examples
                
                # Determine how many more are needed
                needed = target - len(examples)
                
                # Randomly sample with replacement
                if examples:  # Only if we have at least one example
                    additional = random.choices(examples, k=needed)
                    balanced_indices.extend(additional)
        
        # Include unclassified examples
        balanced_indices.extend(unclassified_examples)
        
        # Create balanced dataset
        balanced_data = [self.data[i] for i in balanced_indices]
        
        logger.info(f"Balanced dataset created with {len(balanced_data)} examples")
        logger.info(f"Domain distribution: {target_counts}")
        
        return balanced_data
    
    def filter_by_quality(self, min_question_length=5, max_question_length=200,
                         min_answer_length=10, max_answer_length=1000):
        """
        Filter dataset by quality metrics.
        
        Args:
            min_question_length: Minimum question length in words
            max_question_length: Maximum question length in words
            min_answer_length: Minimum answer length in words
            max_answer_length: Maximum answer length in words
            
        Returns:
            Filtered dataset
        """
        filtered_data = []
        
        for item in self.data:
            question = item.get('question', '')
            answer = item.get('answer', '')
            
            q_length = len(question.split())
            a_length = len(answer.split())
            
            # Apply filtering criteria
            if (min_question_length <= q_length <= max_question_length and
                min_answer_length <= a_length <= max_answer_length):
                filtered_data.append(item)
                
        logger.info(f"Quality filter applied. {len(filtered_data)}/{len(self.data)} examples retained.")
        return filtered_data
    
    def create_balanced_dataset(self, output_path, balance_domains=True, filter_quality=True,
                              domain_targets=None, quality_params=None):
        """
        Create a balanced and filtered dataset.
        
        Args:
            output_path: Path to save the balanced dataset
            balance_domains: Whether to balance domains
            filter_quality: Whether to filter by quality
            domain_targets: Target counts for domains
            quality_params: Parameters for quality filtering
            
        Returns:
            Path to the balanced dataset
        """
        # Make a copy of the original data
        processed_data = self.data.copy()
        
        # Apply quality filtering if requested
        if filter_quality:
            quality_params = quality_params or {}
            processed_data = self.filter_by_quality(**quality_params)
            
        # Balance domains if requested
        if balance_domains:
            # Set the data to the filtered data before balancing
            temp_balancer = DatasetBalancer(data=processed_data)
            temp_balancer.analyze_dataset()
            processed_data = temp_balancer.balance_domains(target_counts=domain_targets)
            
        # Save the processed dataset
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2)
            
        logger.info(f"Balanced and filtered dataset saved to {output_path}")
        logger.info(f"Final dataset size: {len(processed_data)} examples")
        
        return output_path
    
    def create_train_val_test_split(self, output_dir, train_ratio=0.8, val_ratio=0.1, 
                                  test_ratio=0.1, seed=42):
        """
        Create train/validation/test splits.
        
        Args:
            output_dir: Directory to save splits
            train_ratio: Ratio of training data
            val_ratio: Ratio of validation data
            test_ratio: Ratio of test data
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary with paths to split files
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, "Ratios must sum to 1"
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set random seed
        random.seed(seed)
        
        # Shuffle data
        shuffled_data = self.data.copy()
        random.shuffle(shuffled_data)
        
        # Calculate split indices
        n_samples = len(shuffled_data)
        train_end = int(n_samples * train_ratio)
        val_end = train_end + int(n_samples * val_ratio)
        
        # Split data
        train_data = shuffled_data[:train_end]
        val_data = shuffled_data[train_end:val_end]
        test_data = shuffled_data[val_end:]
        
        # Save splits
        train_path = os.path.join(output_dir, 'train.json')
        val_path = os.path.join(output_dir, 'val.json')
        test_path = os.path.join(output_dir, 'test.json')
        
        with open(train_path, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, indent=2)
            
        with open(val_path, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, indent=2)
            
        with open(test_path, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=2)
            
        logger.info(f"Data split created:")
        logger.info(f"  Train: {len(train_data)} examples ({train_ratio*100:.1f}%)")
        logger.info(f"  Validation: {len(val_data)} examples ({val_ratio*100:.1f}%)")
        logger.info(f"  Test: {len(test_data)} examples ({test_ratio*100:.1f}%)")
        
        return {
            'train': train_path,
            'val': val_path,
            'test': test_path
        }
    
    def _extract_domains(self, text):
        """Extract domains from text based on keywords."""
        domains = []
        for domain, pattern in self.domain_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                domains.append(domain)
        return domains
    
    def _extract_topics(self, text, min_word_length=4):
        """Extract potential topics from text."""
        # Simple approach: extract meaningful words
        words = re.findall(r'\b[a-zA-Z]{' + str(min_word_length) + r',}\b', text.lower())
        
        # Filter out common stop words
        stop_words = {'what', 'when', 'where', 'which', 'who', 'whom', 'whose', 'why', 'how', 
                     'that', 'this', 'these', 'those', 'from', 'into', 'your', 'with'}
        return [w for w in words if w not in stop_words]

def main():
    """Run dataset analysis and balancing from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze, balance, and filter datasets for Theta AI")
    parser.add_argument("--input", type=str, required=True, help="Path to input dataset")
    parser.add_argument("--output", type=str, required=True, help="Path to save processed dataset")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    parser.add_argument("--viz-dir", type=str, default="dataset_viz", help="Directory for visualizations")
    parser.add_argument("--balance", action="store_true", help="Balance domain distribution")
    parser.add_argument("--filter", action="store_true", help="Apply quality filters")
    parser.add_argument("--split", action="store_true", help="Create train/val/test splits")
    parser.add_argument("--split-dir", type=str, default="splits", help="Directory for data splits")
    
    args = parser.parse_args()
    
    # Initialize balancer
    balancer = DatasetBalancer(data_path=args.input)
    
    # Analyze dataset
    balancer.analyze_dataset()
    
    # Generate visualizations if requested
    if args.visualize:
        viz_dir = args.viz_dir
        os.makedirs(viz_dir, exist_ok=True)
        balancer.visualize_distribution(output_path=viz_dir)
    
    # Process dataset
    if args.balance or args.filter:
        balancer.create_balanced_dataset(
            args.output,
            balance_domains=args.balance,
            filter_quality=args.filter
        )
    
    # Create splits if requested
    if args.split:
        split_dir = args.split_dir
        os.makedirs(split_dir, exist_ok=True)
        balancer.create_train_val_test_split(output_dir=split_dir)

if __name__ == "__main__":
    main()
