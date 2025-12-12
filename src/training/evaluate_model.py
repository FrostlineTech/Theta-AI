"""
Model Evaluation Script for Theta AI

This module evaluates a trained model on a test dataset and generates performance metrics.
"""

import os
import json
import argparse
import torch
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import time
from tqdm import tqdm

from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    GPT2Config
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Evaluates trained Theta AI models on test datasets."""
    
    def __init__(self, model_path: str, device: str = None):
        """
        Initialize the model evaluator.
        
        Args:
            model_path: Path to the trained model
            device: Device to run evaluation on (cpu, cuda, or None for auto-detect)
        """
        self.model_path = model_path
        
        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Load model and tokenizer
        self.model, self.tokenizer = self._load_model()
        
        # Set evaluation parameters
        self.max_length = 256
        self.metrics = {}
    
    def _load_model(self):
        """
        Load the model and tokenizer.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        try:
            logger.info(f"Loading model from {self.model_path}")
            
            # Load tokenizer
            tokenizer = GPT2Tokenizer.from_pretrained(self.model_path)
            
            # Add padding token if it doesn't exist
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model
            model = GPT2LMHeadModel.from_pretrained(self.model_path)
            model.to(self.device)
            model.eval()
            
            logger.info(f"Model loaded successfully on {self.device}")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def evaluate(self, test_data_path: str, batch_size: int = 8):
        """
        Evaluate the model on a test dataset.
        
        Args:
            test_data_path: Path to the test dataset
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            # Load test data
            test_data = self._load_test_data(test_data_path)
            if not test_data:
                logger.error(f"No test data found at {test_data_path}")
                return {}
            
            logger.info(f"Evaluating model on {len(test_data)} test examples")
            
            # Initialize metrics
            metrics = {
                "perplexity": 0.0,
                "accuracy": 0.0,
                "bleu": 0.0,
                "domain_specific": {},
                "examples": [],
                "evaluation_time": 0.0
            }
            
            # Start evaluation timer
            start_time = time.time()
            
            # Evaluate perplexity
            perplexity = self._evaluate_perplexity(test_data, batch_size)
            metrics["perplexity"] = perplexity
            
            # Evaluate generated responses
            response_metrics = self._evaluate_responses(test_data)
            metrics.update(response_metrics)
            
            # Calculate domain-specific metrics if available
            domain_metrics = self._calculate_domain_metrics(test_data, response_metrics["examples"])
            metrics["domain_specific"] = domain_metrics
            
            # Record evaluation time
            metrics["evaluation_time"] = time.time() - start_time
            
            # Save metrics
            self.metrics = metrics
            
            logger.info(f"Evaluation completed in {metrics['evaluation_time']:.2f} seconds")
            logger.info(f"Overall perplexity: {metrics['perplexity']:.4f}")
            logger.info(f"Overall accuracy: {metrics['accuracy']:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            return {"error": str(e)}
    
    def _load_test_data(self, test_data_path: str) -> List[Dict]:
        """
        Load test data from a JSON file.
        
        Args:
            test_data_path: Path to the test data file
            
        Returns:
            List of test examples
        """
        try:
            with open(test_data_path, 'r') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                logger.warning(f"Test data at {test_data_path} is not a list")
                return []
            
            # Filter for valid QA pairs
            valid_data = []
            for item in data:
                if isinstance(item, dict) and "question" in item and "answer" in item:
                    valid_data.append(item)
            
            logger.info(f"Loaded {len(valid_data)} valid test examples from {test_data_path}")
            return valid_data
            
        except Exception as e:
            logger.error(f"Error loading test data: {e}")
            return []
    
    def _evaluate_perplexity(self, test_data: List[Dict], batch_size: int) -> float:
        """
        Evaluate model perplexity on test data.
        
        Args:
            test_data: List of test examples
            batch_size: Batch size for evaluation
            
        Returns:
            Perplexity score
        """
        total_loss = 0.0
        total_length = 0
        
        # Process in batches
        for i in tqdm(range(0, len(test_data), batch_size), desc="Evaluating perplexity"):
            batch = test_data[i:i+batch_size]
            batch_inputs = []
            
            # Create evaluation prompts
            for item in batch:
                prompt = f"Question: {item['question']}\nAnswer:"
                response = item['answer']
                evaluation_text = f"{prompt} {response}"
                batch_inputs.append(evaluation_text)
            
            # Tokenize inputs
            with torch.no_grad():
                # Encode batch
                encodings = self.tokenizer(batch_inputs, padding=True, truncation=True, 
                                          return_tensors="pt").to(self.device)
                
                # Get token IDs
                input_ids = encodings.input_ids
                attention_mask = encodings.attention_mask
                
                # Calculate loss
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs.loss.item()
                
                # Update totals
                batch_length = attention_mask.sum().item()
                total_loss += loss * batch_length
                total_length += batch_length
        
        # Calculate perplexity
        avg_loss = total_loss / total_length if total_length > 0 else float('inf')
        perplexity = float(np.exp(avg_loss))
        
        return perplexity
    
    def _evaluate_responses(self, test_data: List[Dict]) -> Dict:
        """
        Evaluate model-generated responses against reference answers.
        
        Args:
            test_data: List of test examples
            
        Returns:
            Dictionary of response metrics
        """
        num_examples = min(50, len(test_data))  # Limit to 50 examples for detailed evaluation
        test_examples = test_data[:num_examples]
        
        examples = []
        total_accuracy = 0.0
        total_bleu = 0.0
        
        # Generate responses for each test example
        for item in tqdm(test_examples, desc="Generating responses"):
            # Create prompt
            prompt = f"Question: {item['question']}\nAnswer:"
            reference = item['answer']
            
            # Generate response
            response = self._generate_response(prompt)
            
            # Calculate accuracy (simple token overlap for now)
            accuracy = self._calculate_accuracy(response, reference)
            total_accuracy += accuracy
            
            # Calculate BLEU score
            bleu = self._calculate_bleu(response, reference)
            total_bleu += bleu
            
            # Save example
            example = {
                "question": item["question"],
                "reference": reference,
                "generated": response,
                "accuracy": accuracy,
                "bleu": bleu,
                "domain": item.get("domain", "general")
            }
            examples.append(example)
        
        # Calculate average metrics
        avg_accuracy = total_accuracy / len(examples) if examples else 0.0
        avg_bleu = total_bleu / len(examples) if examples else 0.0
        
        return {
            "accuracy": avg_accuracy,
            "bleu": avg_bleu,
            "examples": examples
        }
    
    def _generate_response(self, prompt: str) -> str:
        """
        Generate a response for a prompt.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated response
        """
        try:
            with torch.no_grad():
                # Tokenize prompt
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                
                # Generate response
                output = self.model.generate(
                    inputs.input_ids,
                    max_length=self.max_length + inputs.input_ids.shape[1],
                    temperature=0.7,
                    top_p=0.9,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                # Decode response
                generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
                
                # Extract only the answer part
                response = generated_text[len(prompt):].strip()
                
                return response
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return ""
    
    def _calculate_accuracy(self, generated: str, reference: str) -> float:
        """
        Calculate simple token overlap accuracy.
        
        Args:
            generated: Generated text
            reference: Reference text
            
        Returns:
            Accuracy score (0.0-1.0)
        """
        # Simple token overlap for now
        gen_tokens = set(generated.lower().split())
        ref_tokens = set(reference.lower().split())
        
        if not ref_tokens:
            return 0.0
        
        overlap = len(gen_tokens.intersection(ref_tokens))
        return overlap / len(ref_tokens)
    
    def _calculate_bleu(self, generated: str, reference: str) -> float:
        """
        Calculate simplified BLEU score.
        
        Args:
            generated: Generated text
            reference: Reference text
            
        Returns:
            BLEU score (0.0-1.0)
        """
        try:
            # Import nltk only when needed
            from nltk.translate.bleu_score import sentence_bleu
            from nltk.tokenize import word_tokenize
            
            # Tokenize
            gen_tokens = word_tokenize(generated.lower())
            ref_tokens = word_tokenize(reference.lower())
            
            # Calculate BLEU score
            return sentence_bleu([ref_tokens], gen_tokens, weights=(0.25, 0.25, 0.25, 0.25))
        except ImportError:
            logger.warning("NLTK not available, using simple accuracy instead of BLEU")
            return self._calculate_accuracy(generated, reference)
        except Exception as e:
            logger.error(f"Error calculating BLEU: {e}")
            return 0.0
    
    def _calculate_domain_metrics(self, test_data: List[Dict], evaluated_examples: List[Dict]) -> Dict:
        """
        Calculate metrics broken down by domain.
        
        Args:
            test_data: Complete test dataset
            evaluated_examples: Examples with generated responses
            
        Returns:
            Dictionary of domain-specific metrics
        """
        # Group by domain
        domains = {}
        for example in evaluated_examples:
            domain = example.get("domain", "general")
            
            if domain not in domains:
                domains[domain] = {
                    "count": 0,
                    "accuracy": 0.0,
                    "bleu": 0.0,
                    "examples": []
                }
                
            domains[domain]["count"] += 1
            domains[domain]["accuracy"] += example["accuracy"]
            domains[domain]["bleu"] += example["bleu"]
            domains[domain]["examples"].append(example)
        
        # Calculate averages
        for domain, stats in domains.items():
            if stats["count"] > 0:
                stats["accuracy"] /= stats["count"]
                stats["bleu"] /= stats["count"]
        
        return domains
    
    def save_metrics(self, output_path: str):
        """
        Save evaluation metrics to a JSON file.
        
        Args:
            output_path: Path to save metrics to
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.metrics:
                logger.warning("No metrics to save")
                return False
                
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Add timestamp
            self.metrics["timestamp"] = datetime.now().isoformat()
            self.metrics["model_path"] = self.model_path
            
            # Save metrics
            with open(output_path, 'w') as f:
                json.dump(self.metrics, f, indent=2)
                
            logger.info(f"Metrics saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
            return False

def main():
    """Main function for model evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate Theta AI model")
    parser.add_argument("--model_path", required=True, help="Path to the trained model")
    parser.add_argument("--test_data", required=True, help="Path to the test data")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--device", choices=["cpu", "cuda"], help="Device to run evaluation on")
    parser.add_argument("--output_file", help="Path to save metrics to")
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = ModelEvaluator(args.model_path, device=args.device)
    
    # Run evaluation
    metrics = evaluator.evaluate(args.test_data, batch_size=args.batch_size)
    
    # Print summary
    if "perplexity" in metrics:
        print(f"\nEvaluation Results:")
        print(f"- Perplexity: {metrics['perplexity']:.4f}")
        print(f"- Accuracy: {metrics['accuracy']:.4f}")
        print(f"- BLEU Score: {metrics['bleu']:.4f}")
        
        # Print domain-specific metrics
        if "domain_specific" in metrics:
            print("\nDomain-Specific Results:")
            for domain, stats in metrics["domain_specific"].items():
                print(f"- {domain.capitalize()} (n={stats['count']}): Accuracy {stats['accuracy']:.4f}, BLEU {stats['bleu']:.4f}")
    
    # Save metrics if output file specified
    if args.output_file:
        evaluator.save_metrics(args.output_file)
        print(f"\nMetrics saved to {args.output_file}")

if __name__ == "__main__":
    main()
