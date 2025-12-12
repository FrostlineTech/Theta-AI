"""
Domain-Specific Fine-Tuning for Theta AI

This module handles specialized fine-tuning for different technical domains.
"""

import json
import logging
import os
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import random
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DomainTuning:
    """Domain-specific fine-tuning for Theta AI."""
    
    def __init__(self, datasets_dir: Path, models_dir: Path):
        """
        Initialize the domain tuning module.
        
        Args:
            datasets_dir: Path to the datasets directory
            models_dir: Path to the models directory
        """
        self.datasets_dir = datasets_dir
        self.models_dir = models_dir
        
        # Create output directories
        self.domain_datasets_dir = datasets_dir / "domain_specific"
        self.domain_models_dir = models_dir / "domain_specific"
        
        os.makedirs(self.domain_datasets_dir, exist_ok=True)
        os.makedirs(self.domain_models_dir, exist_ok=True)
        
        # Define supported domains
        self.domains = ["cybersecurity", "programming", "networking", 
                      "cloud_computing", "data_science"]
        
        # Track domain datasets
        self.domain_datasets = {domain: [] for domain in self.domains}
        self.domain_stats = {domain: {"examples": 0, "tokens": 0} for domain in self.domains}
    
    def prepare_domain_datasets(self):
        """
        Prepare specialized datasets for each domain.
        """
        logger.info("Preparing domain-specific datasets...")
        
        # Find all QA datasets in the main directory
        all_qa_files = list(self.datasets_dir.glob("*.json"))
        
        # Include synthetic data if available
        synthetic_dir = self.datasets_dir / "synthetic_data"
        if synthetic_dir.exists():
            all_qa_files.extend(list(synthetic_dir.glob("*.json")))
        
        # Include curated QA if available
        curated_qa_dir = self.datasets_dir / "curated_qa"
        if curated_qa_dir.exists():
            all_qa_files.extend(list(curated_qa_dir.glob("*.json")))
            
        # Process all files and categorize by domain
        for qa_file in all_qa_files:
            self._process_qa_file(qa_file)
        
        # Save domain-specific datasets
        for domain in self.domains:
            if self.domain_datasets[domain]:
                output_path = self.domain_datasets_dir / f"{domain}_specialized.json"
                
                with open(output_path, 'w') as f:
                    json.dump(self.domain_datasets[domain], f, indent=2)
                    
                logger.info(f"Saved {len(self.domain_datasets[domain])} examples for domain '{domain}' to {output_path}")
                
                # Update stats
                self.domain_stats[domain]["examples"] = len(self.domain_datasets[domain])
                self.domain_stats[domain]["tokens"] = self._estimate_tokens(self.domain_datasets[domain])
    
    def _process_qa_file(self, file_path: Path):
        """
        Process a QA file and categorize examples by domain.
        
        Args:
            file_path: Path to the QA file
        """
        try:
            # Try with UTF-8 encoding first with error handling
            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                # If that fails, try a more aggressive approach with binary reading
                logger.warning(f"JSON decode error with utf-8 for {file_path}, trying alternative approach")
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
            
            if not isinstance(data, list):
                logger.warning(f"File {file_path} does not contain a list of QA pairs")
                return
            
            # Process each QA pair
            for item in data:
                if not isinstance(item, dict) or "question" not in item or "answer" not in item:
                    continue
                
                # Determine domain
                domain = item.get("domain")
                if domain is None:
                    # Try to infer domain from content
                    domain = self._infer_domain(item["question"], item["answer"])
                
                # Skip if domain not supported
                if domain not in self.domains:
                    continue
                
                # Add to domain dataset
                self.domain_datasets[domain].append(item)
                
            logger.debug(f"Processed {file_path}")
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
    
    def _infer_domain(self, question: str, answer: str) -> Optional[str]:
        """
        Infer the domain of a QA pair based on content.
        
        Args:
            question: The question text
            answer: The answer text
            
        Returns:
            Inferred domain or None if uncertain
        """
        # Combine text for analysis
        text = (question + " " + answer).lower()
        
        # Define domain keywords
        domain_keywords = {
            "cybersecurity": ["security", "threat", "vulnerability", "attack", "malware", 
                            "phishing", "ransomware", "encryption", "firewall", "authentication"],
            "programming": ["code", "program", "function", "class", "algorithm", "language", 
                          "variable", "compiler", "library", "api", "framework"],
            "networking": ["network", "protocol", "router", "switch", "tcp", "ip", "dns", 
                         "http", "ethernet", "lan", "wan", "subnet"],
            "cloud_computing": ["cloud", "aws", "azure", "gcp", "container", "kubernetes", 
                              "docker", "serverless", "iaas", "paas", "saas"],
            "data_science": ["data", "machine learning", "ml", "algorithm", "model", "neural", 
                           "tensorflow", "pytorch", "dataset", "training", "regression"]
        }
        
        # Count keywords for each domain
        domain_counts = {domain: 0 for domain in domain_keywords}
        
        for domain, keywords in domain_keywords.items():
            for keyword in keywords:
                domain_counts[domain] += text.count(keyword)
        
        # Select domain with highest keyword count
        max_count = max(domain_counts.values())
        if max_count > 0:
            # Find domains with max count (could be multiple)
            max_domains = [d for d, c in domain_counts.items() if c == max_count]
            return random.choice(max_domains)
        
        return None
    
    def _estimate_tokens(self, qa_pairs: List[Dict]) -> int:
        """
        Estimate the number of tokens in a list of QA pairs.
        
        Args:
            qa_pairs: List of QA pairs
            
        Returns:
            Estimated token count
        """
        # Simple estimation: 1 token ≈ 4 characters
        total_chars = sum(len(qa.get("question", "")) + len(qa.get("answer", "")) for qa in qa_pairs)
        return total_chars // 4
    
    def create_fine_tuning_config(self, domain: str, base_learning_rate: float = 5e-5) -> Dict:
        """
        Create fine-tuning configuration for a domain.
        
        Args:
            domain: Domain to create config for
            base_learning_rate: Base learning rate
            
        Returns:
            Fine-tuning configuration
        """
        # Scale learning rate based on domain dataset size
        dataset_size = self.domain_stats.get(domain, {}).get("examples", 0)
        if dataset_size == 0:
            logger.warning(f"No examples found for domain '{domain}', using default configuration")
            dataset_size = 1000  # Default size
        
        # Adjust learning rate based on dataset size
        # Smaller datasets need higher learning rates to learn quickly
        if dataset_size < 500:
            learning_rate = base_learning_rate * 1.5
            epochs = 20
        elif dataset_size < 2000:
            learning_rate = base_learning_rate * 1.2
            epochs = 15
        elif dataset_size < 5000:
            learning_rate = base_learning_rate
            epochs = 10
        else:
            learning_rate = base_learning_rate * 0.8
            epochs = 5
        
        # Calculate batch size based on expected data size
        # Start with a base value and adjust for memory constraints
        tokens_estimate = self.domain_stats.get(domain, {}).get("tokens", 0)
        
        # Adjust for the 12GB VRAM on RTX 3060
        if tokens_estimate > 2000000:  # Very large dataset
            batch_size = 2
            gradient_accumulation = 8
        elif tokens_estimate > 1000000:  # Large dataset
            batch_size = 3
            gradient_accumulation = 6
        else:  # Medium to small dataset
            batch_size = 4
            gradient_accumulation = 4
        
        # Create configuration
        config = {
            "domain": domain,
            "dataset_path": str(self.domain_datasets_dir / f"{domain}_specialized.json"),
            "output_dir": str(self.domain_models_dir / domain),
            "model_name": "gpt2-medium",  # Default base model
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation,
            "epochs": epochs,
            "warmup_proportion": 0.1,
            "weight_decay": 0.01,
            "scheduler_type": "cosine",
            "gradient_checkpointing": True,
            "fp16": True  # Use fp16 precision for memory efficiency
        }
        
        # Save configuration
        config_path = self.domain_models_dir / f"{domain}_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Created fine-tuning configuration for domain '{domain}' with learning rate {learning_rate} and batch size {batch_size}")
        return config
    
    def generate_fine_tuning_script(self, domain: str, config: Dict) -> str:
        """
        Generate a shell script for domain-specific fine-tuning.
        
        Args:
            domain: Domain to generate script for
            config: Fine-tuning configuration
            
        Returns:
            Path to the generated script
        """
        # Create script
        script = "@echo off\n"
        script += f"echo Theta AI - Domain-Specific Fine-Tuning: {domain.upper()}\n"
        script += "echo =============================================\n"
        script += "echo.\n"
        script += f"echo Started at: %date% %time%\n"
        script += "echo.\n\n"
        
        # Set environment variables
        script += "REM Set environment variables and optimizations\n"
        script += "set CUDA_VISIBLE_DEVICES=0\n"
        script += "set PYTHONPATH=.\n"
        script += "set OMP_NUM_THREADS=6\n"
        script += "set MKL_NUM_THREADS=6\n"
        script += "set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128\n\n"
        
        # Create log directory
        script += "REM Create log directory\n"
        script += "if not exist \"logs\" mkdir logs\n\n"
        
        # Set log file
        script += "REM Set log file\n"
        script += f"set logfile=logs\\{domain}_tuning_%date:~-4,4%%date:~-7,2%%date:~-10,2%_%time:~0,2%%time:~3,2%%time:~6,2%.log\n"
        script += "set logfile=%logfile: =0%\n\n"
        
        # Run fine-tuning
        script += f"echo Starting domain-specific fine-tuning for {domain}...\n"
        script += "python src/training/train_enhanced.py ^\n"
        script += f"  --data_path \"{config['dataset_path']}\" ^\n"
        script += f"  --output_dir \"{config['output_dir']}\" ^\n"
        script += f"  --model_name \"{config['model_name']}\" ^\n"
        script += f"  --batch_size {config['batch_size']} ^\n"
        script += f"  --gradient_accumulation_steps {config['gradient_accumulation_steps']} ^\n"
        script += f"  --learning_rate {config['learning_rate']} ^\n"
        script += f"  --epochs {config['epochs']} ^\n"
        script += f"  --patience 3 ^\n"
        script += f"  --warmup_proportion {config['warmup_proportion']} ^\n"
        script += f"  --scheduler_type \"{config['scheduler_type']}\" ^\n"
        script += f"  --weight_decay {config['weight_decay']} ^\n"
        script += f"  --domain_name \"{domain}\" ^\n"
        script += "  --log_file \"%logfile%\"\n\n"
        
        # Add completion message
        script += "echo.\n"
        script += "echo Training completed at: %date% %time%\n"
        script += "echo.\n"
        script += f"echo Final model saved to: {config['output_dir']}\\{domain}_final\n"
        script += "echo.\n\n"
        
        script += "pause\n"
        
        # Save script
        script_path = self.domain_models_dir / f"tune_{domain}.bat"
        with open(script_path, 'w') as f:
            f.write(script)
        
        logger.info(f"Created fine-tuning script for domain '{domain}' at {script_path}")
        return str(script_path)
    
    def create_all_domain_scripts(self):
        """
        Create fine-tuning scripts for all domains.
        
        Returns:
            List of created script paths
        """
        scripts = []
        
        # First prepare domain datasets
        self.prepare_domain_datasets()
        
        # Create combined script
        combined_script = "@echo off\n"
        combined_script += "echo Theta AI - All Domain Fine-Tuning\n"
        combined_script += "echo =============================================\n"
        combined_script += "echo.\n"
        combined_script += "echo This script will run fine-tuning for all domains sequentially\n"
        combined_script += "echo Started at: %date% %time%\n"
        combined_script += "echo.\n\n"
        
        # Create scripts for each domain
        for domain in self.domains:
            if self.domain_stats[domain]["examples"] > 0:
                config = self.create_fine_tuning_config(domain)
                script_path = self.generate_fine_tuning_script(domain, config)
                scripts.append(script_path)
                
                # Add to combined script
                combined_script += f"echo Running fine-tuning for {domain}...\n"
                combined_script += f"call {os.path.basename(script_path)}\n"
                combined_script += "echo.\n\n"
        
        # Add completion message to combined script
        combined_script += "echo.\n"
        combined_script += "echo All domain fine-tuning completed at: %date% %time%\n"
        combined_script += "echo.\n\n"
        combined_script += "pause\n"
        
        # Save combined script
        combined_script_path = self.domain_models_dir / "tune_all_domains.bat"
        with open(combined_script_path, 'w') as f:
            f.write(combined_script)
        
        logger.info(f"Created combined fine-tuning script at {combined_script_path}")
        scripts.append(str(combined_script_path))
        
        return scripts

def main():
    """Main function to prepare domain-specific fine-tuning."""
    # Get project root and directories
    project_root = Path(__file__).resolve().parent.parent.parent
    datasets_dir = project_root / "Datasets"
    models_dir = project_root / "models"
    
    # Create domain tuning manager
    domain_tuning = DomainTuning(datasets_dir, models_dir)
    
    # Create domain-specific fine-tuning scripts
    scripts = domain_tuning.create_all_domain_scripts()
    
    print(f"Created {len(scripts)} fine-tuning scripts:")
    for script in scripts:
        print(f"- {script}")

if __name__ == "__main__":
    main()
