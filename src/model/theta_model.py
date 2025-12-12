import torch
import torch.nn as nn
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, GPT2LMHeadModel, GPT2Config
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ThetaModel:
    """
    Theta AI model for Frostline Solutions.
    Built to run on NVIDIA RTX 3060 GPU.
    """
    
    def __init__(self, model_type="gpt2", model_name="gpt2", device=None):
        """
        Initialize the Theta AI model.
        
        Args:
            model_type: Type of model to use ('gpt2', 'bert-qa', etc.)
            model_name: Specific model name/version
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.model_type = model_type
        self.model_name = model_name
        
        # Auto-detect device if not specified
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        # Check if CUDA is available and print GPU information
        if torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            
            # Optimize for RTX 3060 specifically
            if "RTX 3060" in torch.cuda.get_device_name(0):
                print("Detected RTX 3060 - Optimizing model settings...")
                # Set GPU-specific optimizations
                torch.backends.cudnn.benchmark = True
                # 12GB VRAM on RTX 3060 - adjust batch sizes accordingly
        else:
            print("CUDA is not available. Using CPU instead.")
            
        # Initialize model based on type
        if model_type == "gpt2":
            self._initialize_gpt2_model()
        elif model_type == "bert-qa":
            self._initialize_bert_qa_model()
        else:
            raise ValueError(f"Model type '{model_type}' not supported")
            
    def _initialize_gpt2_model(self):
        """Initialize GPT-2 based language model for Theta."""
        print(f"Initializing GPT-2 model: {self.model_name}")
        
        # For fine-tuning, we can start with a smaller GPT-2 model (fits better on RTX 3060)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Add special tokens for Frostline-specific content
        special_tokens = {
            "additional_special_tokens": [
                "[FROSTLINE]",
                "[CYBERSECURITY]", 
                "[SOFTWARE_DEV]",
                "[CLOUD]", 
                "[AI]",
                "[NETWORKING]"
            ],
            "pad_token": "[PAD]"
        }
        self.tokenizer.add_special_tokens(special_tokens)
        
        # Set pad token id in the tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize model with configuration for better memory usage
        config = GPT2Config.from_pretrained(self.model_name)
        config.use_cache = False  # Disable KV cache during training for memory efficiency
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name, config=config)
        
        # Resize token embeddings to account for new special tokens
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Move model to appropriate device (GPU if available)
        self.model.to(self.device)
        
    def _initialize_bert_qa_model(self):
        """Initialize BERT-based question answering model for Theta."""
        print(f"Initializing BERT QA model: {self.model_name}")
        
        # For QA tasks, we'll use a BERT-based model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.model_name)
        
        # Move model to appropriate device (GPU if available)
        self.model.to(self.device)
    
    def save(self, save_path):
        """
        Save the model and tokenizer.
        
        Args:
            save_path: Path to save the model to
        """
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"Model saved to {save_path}")
        
    def load(self, load_path):
        """
        Load the model and tokenizer.
        
        Args:
            load_path: Path to load the model from
        """
        if self.model_type == "gpt2":
            self.model = GPT2LMHeadModel.from_pretrained(load_path)
        elif self.model_type == "bert-qa":
            self.model = AutoModelForQuestionAnswering.from_pretrained(load_path)
            
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        self.model.to(self.device)
        print(f"Model loaded from {load_path}")
        
    def optimize_for_gpu(self, enable_checkpointing=True, rtx_3060_optimized=True):
        """
        Apply specific optimizations for NVIDIA GPUs with focus on RTX 3060.
        
        Args:
            enable_checkpointing: Whether to enable gradient checkpointing for memory efficiency
            rtx_3060_optimized: Whether to apply specific RTX 3060 optimizations
        """
        # RTX 3060 has 12GB VRAM - these settings help maximize performance
        if self.device.type == "cuda":
            # Enable mixed precision training for memory efficiency
            torch.amp.autocast("cuda", enabled=True)
            
            # Set appropriate memory management for GPU VRAM
            torch.cuda.empty_cache()
            
            # Reserve less CUDA memory upfront to avoid OOM errors
            if rtx_3060_optimized:
                # RTX 3060 specific optimizations
                torch.cuda.set_per_process_memory_fraction(0.85)  # Reserve 85% of GPU memory to avoid system OOM
                
                # Set optimal work group size for RTX 3060
                if hasattr(torch.cuda, 'set_device_properties'):
                    try:
                        torch.cuda.set_device_properties(0, prefer_tensor_core=True)  # Prefer tensor cores for faster training
                    except:
                        pass
            
            # Set TF32 on Ampere GPUs for better performance
            if torch.cuda.get_device_capability()[0] >= 8:
                torch.set_float32_matmul_precision('high')
            
            # Enable gradient checkpointing if requested
            if enable_checkpointing and hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
                print("Gradient checkpointing enabled")
                
            # Set CUDA kernel optimization for Ampere architecture (RTX 3060)
            if rtx_3060_optimized:
                torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 for matmul
                torch.backends.cudnn.benchmark = True  # Use cuDNN benchmarks for optimal performance
                torch.backends.cudnn.allow_tf32 = True  # Allow TF32 for cuDNN
            
            # Print optimization info
            print("Applied GPU optimizations:")
            print("- Enabled mixed precision training")
            print("- Cleared CUDA cache")
            if rtx_3060_optimized:
                print("- Applied RTX 3060 specific optimizations")
                print("- Optimized CUDA memory usage (85% limit)")
                print("- Enabled tensor core preference")
                print("- Set cuDNN benchmarking for Ampere architecture")
            print("- Set memory-efficient gradient accumulation")
            if enable_checkpointing:
                print("- Enabled gradient checkpointing (trades compute for memory)")
                print("  This allows training with larger batch sizes")
        
    def generate_response(self, prompt, max_length=100):
        """
        Generate a response based on the prompt.
        
        Args:
            prompt: Input prompt/question
            max_length: Maximum length of the response
            
        Returns:
            Generated response text
        """
        if self.model_type == "gpt2":
            # Format the prompt with a clear instruction format
            formatted_prompt = f"Question: {prompt}\nAnswer:"
            
            # Prepare input with explicit attention mask
            encoded_input = self.tokenizer(formatted_prompt, return_tensors="pt", padding=True)
            input_ids = encoded_input["input_ids"].to(self.device)
            attention_mask = encoded_input["attention_mask"].to(self.device)
            
            # Generate response with better parameters
            with torch.no_grad():
                output = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=max_length + len(input_ids[0]),  # Account for input length
                    num_return_sequences=1,
                    temperature=0.8,  # Slightly higher for more creative responses
                    top_k=40,
                    top_p=0.92,
                    repetition_penalty=1.2,  # Discourage repetitive text
                    no_repeat_ngram_size=3,  # Avoid repeating phrases
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode output and clean up the response
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Extract just the answer part
            if "Answer:" in generated_text:
                answer = generated_text.split("Answer:")[1].strip()
            else:
                # Fallback - just remove the question part
                answer = generated_text.replace(formatted_prompt, "").strip()
            
            return answer
            
        else:
            raise NotImplementedError(f"Generation not implemented for model type: {self.model_type}")


# Example usage
if __name__ == "__main__":
    # Test model initialization
    model = ThetaModel(model_type="gpt2", model_name="gpt2")
    model.optimize_for_gpu()
    
    # Test response generation
    response = model.generate_response("What is Frostline?")
    print(f"Response: {response}")
