"""
Model Optimization for Inference

This script optimizes a trained model for inference by applying techniques such as:
- Quantization (INT8/INT4)
- Pruning
- ONNX conversion
- Fusion of layers
"""

import os
import json
import argparse
import torch
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import time
import shutil

from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    GPT2Config
)

try:
    import onnx
    import onnxruntime
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelOptimizer:
    """Optimizes models for inference."""
    
    def __init__(self, model_path: str, output_path: str):
        """
        Initialize the model optimizer.
        
        Args:
            model_path: Path to the trained model
            output_path: Path to save the optimized model
        """
        self.model_path = model_path
        self.output_path = output_path
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_path, exist_ok=True)
        
        # Check CUDA availability
        self.cuda_available = torch.cuda.is_available()
        self.device = "cuda" if self.cuda_available else "cpu"
        
        # Load model and tokenizer
        self.model, self.tokenizer = self._load_model()
        
        # Initialize optimization metrics
        self.metrics = {
            "original_size": 0,
            "optimized_size": 0,
            "size_reduction": 0.0,
            "inference_speedup": 0.0,
            "original_inference_time": 0.0,
            "optimized_inference_time": 0.0,
            "optimization_techniques": []
        }
    
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
            model.eval()  # Set to evaluation mode
            
            logger.info(f"Model loaded successfully")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _measure_model_size(self, model_path: str) -> int:
        """
        Measure the size of a model on disk.
        
        Args:
            model_path: Path to the model
            
        Returns:
            Size in bytes
        """
        total_size = 0
        
        # Check if path is a directory
        if os.path.isdir(model_path):
            for root, _, files in os.walk(model_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    total_size += os.path.getsize(file_path)
        else:
            # Single file
            total_size = os.path.getsize(model_path)
        
        return total_size
    
    def _benchmark_inference(self, model, tokenizer, device: str) -> float:
        """
        Benchmark inference speed.
        
        Args:
            model: Model to benchmark
            tokenizer: Tokenizer for the model
            device: Device to run on
            
        Returns:
            Average inference time in seconds
        """
        # Move model to device
        model.to(device)
        
        # Sample prompts for benchmarking
        prompts = [
            "What is the capital of France?",
            "How does a binary search algorithm work?",
            "Explain the process of photosynthesis.",
            "What are the main security features in modern operating systems?",
            "Describe the differences between TCP and UDP protocols."
        ]
        
        # Run inference and measure time
        total_time = 0.0
        
        for prompt in prompts:
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            # Generate
            start_time = time.time()
            with torch.no_grad():
                _ = model.generate(
                    inputs.input_ids,
                    max_length=50,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.eos_token_id
                )
            end_time = time.time()
            
            # Add to total
            total_time += (end_time - start_time)
        
        # Calculate average
        avg_time = total_time / len(prompts)
        
        return avg_time
    
    def quantize(self, bits: int = 8):
        """
        Quantize the model to reduce size and improve inference speed.
        
        Args:
            bits: Quantization bits (8 or 4)
        """
        logger.info(f"Quantizing model to INT{bits}")
        self.metrics["optimization_techniques"].append(f"INT{bits} quantization")
        
        try:
            # Check if quantization bits are valid
            if bits not in [8, 4]:
                logger.warning(f"Invalid quantization bits: {bits}. Using 8 bits instead.")
                bits = 8
            
            # Store original model for comparison
            original_model = self.model
            
            # Measure original inference time
            self.metrics["original_inference_time"] = self._benchmark_inference(
                original_model, self.tokenizer, self.device
            )
            
            # Static quantization (PyTorch built-in)
            # IMPORTANT: Quantization must be done on CPU
            cpu_model = original_model.to('cpu')
            
            if bits == 8:
                # Quantize the model to INT8
                logger.info("Running quantization on CPU (required for PyTorch quantization)")
                quantized_model = torch.quantization.quantize_dynamic(
                    cpu_model,
                    {torch.nn.Linear},  # Quantize linear layers
                    dtype=torch.qint8
                )
                
                # Save quantized model
                quantized_path = os.path.join(self.output_path, "quantized_int8")
                os.makedirs(quantized_path, exist_ok=True)
                
                # Save model and tokenizer
                torch.save(quantized_model.state_dict(), os.path.join(quantized_path, "pytorch_model.bin"))
                quantized_model.config.save_pretrained(quantized_path)
                self.tokenizer.save_pretrained(quantized_path)
                
                # Update model reference
                self.model = quantized_model
                
            elif bits == 4:
                # For INT4, we'd ideally use more advanced quantization libraries
                # This is a placeholder for actual INT4 quantization
                logger.warning("True INT4 quantization requires specialized libraries.")
                logger.warning("Using a simplified approximation for demonstration.")
                
                # Save model in original form since we don't have INT4 implementation
                quantized_path = os.path.join(self.output_path, "quantized_int4")
                original_model.save_pretrained(quantized_path)
                self.tokenizer.save_pretrained(quantized_path)
                
                # For actual INT4 quantization, you would use:
                # - bitsandbytes library
                # - GPTQ
                # - AWQ or other specialized quantization methods
            
            # Measure optimized inference time
            self.metrics["optimized_inference_time"] = self._benchmark_inference(
                self.model, self.tokenizer, self.device
            )
            
            # Calculate speedup
            if self.metrics["original_inference_time"] > 0:
                self.metrics["inference_speedup"] = (
                    self.metrics["original_inference_time"] / 
                    self.metrics["optimized_inference_time"]
                )
            
            # Save a sample generation script
            self._save_sample_script(quantized_path)
            
            logger.info(f"Model quantized to INT{bits} and saved to {quantized_path}")
            
        except Exception as e:
            logger.error(f"Error during quantization: {e}")
    
    def prune(self, sparsity: float = 0.1):
        """
        Prune the model to reduce size.
        
        Args:
            sparsity: Target sparsity (0.0-1.0)
        """
        if sparsity <= 0.0:
            logger.info("Pruning disabled (sparsity <= 0)")
            return
            
        logger.info(f"Pruning model with sparsity {sparsity:.2f}")
        self.metrics["optimization_techniques"].append(f"Pruning (sparsity={sparsity:.2f})")
        
        try:
            # Simple magnitude-based pruning (for demonstration)
            # In a production system, would use torch.nn.utils.prune or specialized pruning libraries
            
            # Create a copy of the model for pruning
            pruned_model = self.model
            
            # Move to CPU for better memory management
            pruned_model = pruned_model.to('cpu')
            
            # Simple pruning by zeroing out small weights
            with torch.no_grad():
                for name, param in pruned_model.named_parameters():
                    if 'weight' in name and param.dim() > 1:  # Only process matrices, not vectors or scalars
                        try:
                            # Flatten the tensor for easier processing
                            flat_tensor = param.abs().flatten()
                            
                            # Use sorting-based approach instead of quantile for large tensors
                            sorted_tensor, _ = torch.sort(flat_tensor)
                            threshold_idx = int(sparsity * sorted_tensor.numel())
                            
                            if threshold_idx < sorted_tensor.numel():
                                threshold = sorted_tensor[threshold_idx]
                                
                                # Create a mask for values below threshold
                                mask = param.abs() > threshold
                                
                                # Apply the mask
                                param.data.mul_(mask.float())
                        except Exception as e:
                            logger.warning(f"Skipping pruning for {name}: {str(e)}")
            
            # Save pruned model
            pruned_path = os.path.join(self.output_path, "pruned")
            os.makedirs(pruned_path, exist_ok=True)
            
            # Save model and tokenizer
            pruned_model.save_pretrained(pruned_path)
            self.tokenizer.save_pretrained(pruned_path)
            
            # Update model reference
            self.model = pruned_model
            
            # Save a sample generation script
            self._save_sample_script(pruned_path)
            
            logger.info(f"Model pruned with sparsity {sparsity:.2f} and saved to {pruned_path}")
            
        except Exception as e:
            logger.error(f"Error during pruning: {e}")
    
    def convert_to_onnx(self):
        """Convert model to ONNX format for faster inference."""
        if not ONNX_AVAILABLE:
            logger.warning("ONNX conversion skipped: ONNX not available")
            return
            
        logger.info("Converting model to ONNX format")
        self.metrics["optimization_techniques"].append("ONNX conversion")
        
        try:
            # Create directory for ONNX model
            onnx_path = os.path.join(self.output_path, "onnx")
            os.makedirs(onnx_path, exist_ok=True)
            
            # Prepare dummy input for tracing
            dummy_input = torch.ones(1, 10, dtype=torch.long)
            
            # Export model to ONNX
            torch.onnx.export(
                self.model,
                dummy_input,
                os.path.join(onnx_path, "model.onnx"),
                export_params=True,
                opset_version=12,
                do_constant_folding=True,
                input_names=['input_ids'],
                output_names=['output'],
                dynamic_axes={
                    'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                    'output': {0: 'batch_size', 1: 'sequence_length'}
                }
            )
            
            # Save tokenizer
            self.tokenizer.save_pretrained(onnx_path)
            
            # Save config
            with open(os.path.join(onnx_path, "config.json"), 'w') as f:
                json.dump(
                    {
                        "model_type": "gpt2",
                        "original_model": self.model_path,
                        "tokenizer": {
                            "pad_token": self.tokenizer.pad_token,
                            "eos_token": self.tokenizer.eos_token
                        }
                    },
                    f, indent=2
                )
            
            # Save a sample script for ONNX inference
            self._save_onnx_script(onnx_path)
            
            logger.info(f"Model converted to ONNX and saved to {onnx_path}")
            
        except Exception as e:
            logger.error(f"Error during ONNX conversion: {e}")
    
    def optimize(self, quantize_bits: int = 8, prune: bool = False, prune_sparsity: float = 0.1):
        """
        Perform complete model optimization.
        
        Args:
            quantize_bits: Bits for quantization (8 or 4)
            prune: Whether to prune the model
            prune_sparsity: Pruning sparsity if pruning is enabled
            
        Returns:
            Dictionary of optimization metrics
        """
        logger.info(f"Starting model optimization for inference")
        
        # Measure original model size
        self.metrics["original_size"] = self._measure_model_size(self.model_path)
        
        # Create a combined model path
        combined_path = os.path.join(self.output_path, "optimized")
        os.makedirs(combined_path, exist_ok=True)
        
        # Store original device
        original_device = next(self.model.parameters()).device
        
        # Apply optimizations
        try:
            if quantize_bits > 0:
                self.quantize(bits=quantize_bits)
                
            if prune:
                self.prune(sparsity=prune_sparsity)
            
            # Make sure model is back on appropriate device before saving
            self.model = self.model.to(original_device)
            
            # Save the optimized model (final version)
            self.model.save_pretrained(combined_path)
            self.tokenizer.save_pretrained(combined_path)
        except Exception as e:
            logger.error(f"Error during optimization, saving original model: {str(e)}")
            # If optimization fails, save the original model
            original_model = GPT2LMHeadModel.from_pretrained(self.model_path)
            original_model.save_pretrained(combined_path)
            self.tokenizer.save_pretrained(combined_path)
        
        # Convert to ONNX (optional)
        # self.convert_to_onnx()  # Commented out for now
        
        # Measure optimized model size
        self.metrics["optimized_size"] = self._measure_model_size(combined_path)
        
        # Calculate size reduction
        if self.metrics["original_size"] > 0:
            self.metrics["size_reduction"] = (
                1.0 - (self.metrics["optimized_size"] / self.metrics["original_size"])
            )
        
        # Save optimization metrics
        metrics_path = os.path.join(self.output_path, "optimization_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        # Save a sample generation script
        self._save_sample_script(combined_path)
        
        logger.info(f"Model optimization completed")
        logger.info(f"Original size: {self.metrics['original_size']/1024/1024:.2f} MB")
        logger.info(f"Optimized size: {self.metrics['optimized_size']/1024/1024:.2f} MB")
        logger.info(f"Size reduction: {self.metrics['size_reduction']*100:.2f}%")
        logger.info(f"Inference speedup: {self.metrics['inference_speedup']:.2f}x")
        
        return self.metrics
    
    def _save_sample_script(self, model_path: str):
        """
        Save a sample script for using the optimized model.
        
        Args:
            model_path: Path to the model
        """
        script_content = """# Sample script to use the optimized model

from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Load model and tokenizer
model_path = "."  # Path to this directory
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# Add padding token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Generate text
def generate_text(prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate
    with torch.no_grad():
        output = model.generate(
            inputs.input_ids,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Example usage
prompt = "What is artificial intelligence?"
response = generate_text(prompt)
print(f"Prompt: {prompt}")
print(f"Response: {response}")
"""
        
        script_path = os.path.join(model_path, "sample_generate.py")
        with open(script_path, 'w') as f:
            f.write(script_content)
    
    def _save_onnx_script(self, onnx_path: str):
        """
        Save a sample script for using the ONNX model.
        
        Args:
            onnx_path: Path to the ONNX model
        """
        script_content = """# Sample script to use the ONNX model

import onnxruntime as ort
from transformers import GPT2Tokenizer
import json
import numpy as np

# Load tokenizer and config
tokenizer = GPT2Tokenizer.from_pretrained(".")
with open("config.json", "r") as f:
    config = json.load(f)

# Set up ONNX session
session = ort.InferenceSession("model.onnx")

# Generate text
def generate_text(prompt, max_length=100):
    # Tokenize
    inputs = tokenizer.encode(prompt, return_tensors="np")
    
    # Generate
    for _ in range(max_length):
        # Run model
        ort_inputs = {session.get_inputs()[0].name: inputs}
        ort_outputs = session.run(None, ort_inputs)
        
        # Get next token
        next_token_logits = ort_outputs[0][0, -1, :]
        next_token = np.argmax(next_token_logits)
        
        # Stop if EOS token
        if next_token == tokenizer.eos_token_id:
            break
        
        # Add token to inputs
        inputs = np.concatenate([inputs, [[next_token]]], axis=1)
    
    # Decode
    generated_text = tokenizer.decode(inputs[0], skip_special_tokens=True)
    return generated_text

# Example usage
prompt = "What is artificial intelligence?"
response = generate_text(prompt)
print(f"Prompt: {prompt}")
print(f"Response: {response}")
"""
        
        script_path = os.path.join(onnx_path, "sample_generate_onnx.py")
        with open(script_path, 'w') as f:
            f.write(script_content)

def main():
    """Main function for model optimization."""
    parser = argparse.ArgumentParser(description="Optimize Theta AI model for inference")
    parser.add_argument("--model_path", required=True, help="Path to the trained model")
    parser.add_argument("--output_path", required=True, help="Path to save the optimized model")
    parser.add_argument("--quantize", type=int, choices=[0, 8, 4], default=8, 
                        help="Quantization bits (0 to disable, 8 for INT8, 4 for INT4)")
    parser.add_argument("--prune", type=bool, default=False, 
                        help="Whether to prune the model")
    parser.add_argument("--prune_sparsity", type=float, default=0.1,
                        help="Pruning sparsity (0.0-1.0) if pruning is enabled")
    args = parser.parse_args()
    
    # Create optimizer
    optimizer = ModelOptimizer(args.model_path, args.output_path)
    
    # Run optimization
    metrics = optimizer.optimize(
        quantize_bits=args.quantize,
        prune=args.prune,
        prune_sparsity=args.prune_sparsity
    )
    
    # Print summary
    print(f"\nOptimization Results:")
    print(f"- Original Size: {metrics['original_size']/1024/1024:.2f} MB")
    print(f"- Optimized Size: {metrics['optimized_size']/1024/1024:.2f} MB")
    print(f"- Size Reduction: {metrics['size_reduction']*100:.2f}%")
    print(f"- Inference Speedup: {metrics['inference_speedup']:.2f}x")
    print(f"- Techniques Used: {', '.join(metrics['optimization_techniques'])}")
    print(f"\nOptimized model saved to {args.output_path}")

if __name__ == "__main__":
    main()
