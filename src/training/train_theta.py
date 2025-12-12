import os
import json
import torch
import argparse
import numpy as np
import math
from pathlib import Path
from datetime import datetime
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, GPT2TokenizerFast, GPT2LMHeadModel
from tqdm.auto import tqdm
import sys
import time
from colorama import Fore, Back, Style, init

# Initialize colorama for Windows support
init()

# Add project root to path to import model
project_root = str(Path(__file__).resolve().parent.parent.parent)
sys.path.append(project_root)
from src.model.theta_model import ThetaModel

class ThetaDataset(Dataset):
    """Custom dataset for Theta AI training."""
    
    def __init__(self, data_path, model_name, tokenizer, max_length=512):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the processed data JSON file
            model_name: Name of the model to use
            tokenizer: Tokenizer for encoding the data
            max_length: Maximum sequence length
        """
        # Specify model architecture and tokenizer
        try:
            self.model = GPT2LMHeadModel.from_pretrained(model_name)
            self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print(f"Attempting to download {model_name} from Hugging Face...")
            self.model = GPT2LMHeadModel.from_pretrained(model_name)
            self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        
        # Set special tokens
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.sep_token = '<|sep|>'
        if self.tokenizer.sep_token not in self.tokenizer.get_vocab():
            self.model.resize_token_embeddings(len(self.tokenizer))
            
        print(f"Loaded {model_name} with {sum(p.numel() for p in self.model.parameters())/1000000:.2f}M parameters")
        self.max_length = max_length
        
        # Load data
        with open(data_path, 'r') as f:
            self.data = json.load(f)
            
        # Prepare formatted data for GPT-2 style training
        self.formatted_data = []
        for item in self.data:
            question = item['question']
            answer = item['answer']
            
            # Format as instruction-following text
            text = f"Question: {question}\nAnswer: {answer}\n\n"
            self.formatted_data.append(text)
            
        print(f"Loaded {len(self.formatted_data)} training examples")
    
    def __len__(self):
        return len(self.formatted_data)
    
    def __getitem__(self, idx):
        text = self.formatted_data[idx]
        
        # Tokenize text
        encodings = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Create language modeling labels (same as input_ids)
        input_ids = encodings['input_ids'].squeeze()
        attention_mask = encodings['attention_mask'].squeeze()
        
        # For GPT-2 training, labels are the same as input_ids
        labels = input_ids.clone()
        
        # Set labels for padding tokens to -100 so they're ignored in the loss
        labels[attention_mask == 0] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

def train(args):
    """
    Train the Theta AI model.
    
    Args:
        args: Training arguments
    """
    # Set default gradient accumulation steps if not provided
    if not hasattr(args, 'gradient_accumulation_steps'):
        args.gradient_accumulation_steps = 4
        
    print(f"Training Theta AI model with the following settings:")
    print(f"- Model type: {args.model_type}")
    print(f"- Model name: {args.model_name}")
    print(f"- Data path: {args.data_path}")
    print(f"- Output dir: {args.output_dir}")
    print(f"- Batch size: {args.batch_size}")
    print(f"- Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"- Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"- Learning rate: {args.learning_rate}")
    print(f"- Epochs: {args.epochs}")
    
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        
    # Initialize model
    theta = ThetaModel(model_type=args.model_type, model_name=args.model_name, device=device)
    
    # Apply GPU optimizations
    theta.optimize_for_gpu()
    
    # Create dataset and dataloader
    dataset = ThetaDataset(args.data_path, args.model_name, theta.tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Set up optimizer and scheduler
    optimizer = AdamW(theta.model.parameters(), lr=args.learning_rate)
    
    # Learning rate scheduler with stronger decay
    total_steps = len(dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.05 * total_steps),  # Shorter warmup
        num_training_steps=total_steps
    )
    
    # Custom LR decay for later epochs with cosine annealing
    def lr_lambda(current_step):
        # Start with warmup then cosine decay
        progress = current_step / total_steps
        if progress < 0.1:  # First 10% is warmup
            return progress / 0.1  # Linear warmup
        else:
            # Cosine decay with min learning rate of 10% of original
            return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * (progress - 0.1) / 0.9))
    
    # Add additional custom scheduler
    custom_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    # Split dataset into train and validation
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Keep track of best validation loss for early stopping and model saving
    best_val_loss = float('inf')
    loss_history = []
    val_loss_history = []
    
    # Helper functions for colorful output
    def print_header(text):
        print(f"\n{Fore.CYAN}{Back.BLACK}{Style.BRIGHT}== {text} =={Style.RESET_ALL}")
        
    def print_metric(name, value, color=Fore.GREEN, is_good=True):
        trend = "↓" if is_good and len(loss_history) > 1 and value < loss_history[-2] else "↑"
        trend = Fore.GREEN + trend if is_good else Fore.RED + trend
        print(f"{color}{name}: {value:.4f} {trend}{Style.RESET_ALL}")
        
    def progress_bar_colored(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filled_length = int(length * iteration // total)
        bar = fill * filled_length + '-' * (length - filled_length)
        
        # Choose color based on progress
        if filled_length < length * 0.33:
            color = Fore.RED
        elif filled_length < length * 0.66:
            color = Fore.YELLOW
        else:
            color = Fore.GREEN
            
        print(f'\r{prefix} |{color}{bar}{Style.RESET_ALL}| {percent}% {suffix}', end='\r')
        return iteration == total
    
    # Training loop
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        print_header(f"Epoch {epoch+1}/{args.epochs}")
        theta.model.train()
        
        # Use mixed precision training for RTX 3060 optimization
        scaler = torch.cuda.amp.GradScaler()
        
        total_loss = 0
        batch_count = len(train_dataloader)
        
        for i, batch in enumerate(train_dataloader):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                outputs = theta.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                # Normalize loss for gradient accumulation
                loss = outputs.loss / args.gradient_accumulation_steps
                
            # Backward pass with scaling for mixed precision
            scaler.scale(loss).backward()
            
            # Only update weights after accumulating enough gradients
            if (i + 1) % args.gradient_accumulation_steps == 0 or (i + 1) == len(train_dataloader):
                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()  # Zero gradients after optimizer step
                
                # Update learning rate schedulers
                scheduler.step()
                custom_scheduler.step()
            
            # Update progress
            current_loss = loss.item()
            total_loss += current_loss
            
            # Display fancy progress bar and metrics
            is_last = progress_bar_colored(i+1, batch_count, 
                      prefix=f'{Fore.BLUE}Training:{Style.RESET_ALL}', 
                      suffix=f'{Fore.YELLOW}Loss: {current_loss:.4f}{Style.RESET_ALL} | '
                             f'{Fore.CYAN}Batch: {i+1}/{batch_count}{Style.RESET_ALL}')
            
            # Print full metrics on last batch
            if is_last:
                print()  # New line after progress bar completes
        
        # Calculate average training loss
        avg_train_loss = total_loss / batch_count
        loss_history.append(avg_train_loss)
        
        # Validation phase
        theta.model.eval()
        total_val_loss = 0
        batch_count_val = len(val_dataloader)
        
        print_header("Validation")
        
        with torch.no_grad():
            for i, batch in enumerate(val_dataloader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                with torch.cuda.amp.autocast():
                    outputs = theta.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    val_loss = outputs.loss
                    
                total_val_loss += val_loss.item()
                
                # Display fancy progress bar for validation
                is_last = progress_bar_colored(i+1, batch_count_val, 
                          prefix=f'{Fore.MAGENTA}Validating:{Style.RESET_ALL}', 
                          suffix=f'{Fore.YELLOW}Batch: {i+1}/{batch_count_val}{Style.RESET_ALL}')
                          
                if is_last:
                    print()  # New line after progress bar completes
        
        # Calculate average validation loss
        avg_val_loss = total_val_loss / batch_count_val
        val_loss_history.append(avg_val_loss)
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Calculate perplexity
        train_perplexity = math.exp(avg_train_loss)
        val_perplexity = math.exp(avg_val_loss)
        
        # Print colorful metrics summary
        print_header("Metrics Summary")
        print_metric("Training Loss", avg_train_loss, Fore.GREEN, is_good=True)
        print_metric("Validation Loss", avg_val_loss, Fore.YELLOW, is_good=True)
        print_metric("Training Perplexity", train_perplexity, Fore.CYAN, is_good=True)
        print_metric("Validation Perplexity", val_perplexity, Fore.MAGENTA, is_good=True)
        
        # Print improvement percentage if applicable
        if len(val_loss_history) > 1:
            prev_loss = val_loss_history[-2]
            improvement = (prev_loss - avg_val_loss) / prev_loss * 100
            direction = "improved" if improvement > 0 else "worsened"
            color = Fore.GREEN if improvement > 0 else Fore.RED
            print(f"{color}Validation loss {direction} by {abs(improvement):.2f}%{Style.RESET_ALL}")
        
        print(f"{Fore.BLUE}Epoch completed in {epoch_time:.2f} seconds{Style.RESET_ALL}")
        
        # Check if this is the best model so far
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            print(f"{Fore.GREEN}{Style.BRIGHT}★ New best validation loss achieved! ★{Style.RESET_ALL}")
        
        # Save checkpoint after each epoch
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
            
        checkpoint_dir = os.path.join(
            args.output_dir, 
            f"theta_checkpoint_epoch_{epoch+1}"
        )
        
        theta.save(checkpoint_dir)
        print(f"Saved checkpoint to {checkpoint_dir}")
    
    # Save final model
    final_model_dir = os.path.join(args.output_dir, "theta_final")
    theta.save(final_model_dir)
    print(f"{Fore.GREEN}{Style.BRIGHT}Training complete!{Style.RESET_ALL} Final model saved to {final_model_dir}")
    
    # Print final performance summary
    print_header("Final Performance Summary")
    print(f"{Fore.CYAN}Initial training loss: {loss_history[0]:.4f}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Final training loss: {loss_history[-1]:.4f}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Best validation loss: {best_val_loss:.4f}{Style.RESET_ALL}")
    
    # Calculate overall improvement
    if len(loss_history) > 1:
        total_improvement = (loss_history[0] - loss_history[-1]) / loss_history[0] * 100
        print(f"{Fore.GREEN}Overall loss improvement: {total_improvement:.2f}%{Style.RESET_ALL}")

def main(passed_args=None):
    # Only parse arguments if not passed in
    if passed_args is None:
        parser = argparse.ArgumentParser(description="Train the Theta AI model")
        
        parser.add_argument("--model_type", type=str, default="gpt2", help="Model type (gpt2, bert-qa)")
        parser.add_argument("--model_name", type=str, default="gpt2", help="Specific model name/version")
        parser.add_argument("--data_path", type=str, required=True, help="Path to the processed data")
        parser.add_argument("--output_dir", type=str, required=True, help="Directory to save model checkpoints")
        parser.add_argument("--batch_size", type=int, default=2, help="Training batch size")
        parser.add_argument("--gradient_accumulation_steps", type=int, default=4, 
                          help="Number of steps to accumulate gradients before performing a backward/update pass")
        parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
        parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
        parser.add_argument("--no-color", action="store_true", help="Disable colored output")
        
        args = parser.parse_args()
    else:
        args = passed_args
    
    # Disable colors if requested
    if hasattr(args, 'no_color') and args.no_color:
        init(autoreset=True, strip=True)
    else:
        init(autoreset=True)
        
    # Print colorful banner
    print(f"{Fore.CYAN}{Style.BRIGHT}╔════════════════════════════════════════╗{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{Style.BRIGHT}║         THETA AI TRAINING SYSTEM       ║{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{Style.BRIGHT}╚════════════════════════════════════════╝{Style.RESET_ALL}")
    
    train(args)

if __name__ == "__main__":
    main()
