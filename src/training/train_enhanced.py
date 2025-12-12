#train_enhanced.py

"""
Enhanced training script for Theta AI with early stopping, improved learning rate schedule, 
and validation-based model saving.

Optimizations for RTX 3060 12GB:
- CPU offloading for optimizer states
- Label smoothing (0.1)
- R-Drop regularization
- Layer-wise Learning Rate Decay (LLRD)
- Gradient noise injection
- Exponential Moving Average (EMA)
- Curriculum learning by difficulty
- Extended validation metrics
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
import math
import time
import gc
import psutil
import copy
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, random_split, Sampler
from torch.optim import AdamW
from transformers import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    GPT2TokenizerFast, 
    GPT2LMHeadModel
)
from tqdm.auto import tqdm
import sys
from colorama import Fore, Back, Style, init
import logging
import shutil

# Try to import optional evaluation metrics
try:
    from evaluate import load as load_metric
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    logging.warning("evaluate library not available. Install with: pip install evaluate")

# Import cleanup utilities
sys.path.append(str(Path(__file__).resolve().parent))
from cleanup_utils import cleanup_old_checkpoints, cleanup_temp_files, check_disk_space, optimize_memory_usage, keep_only_best_checkpoint

# Import email notification system
from email_integration import get_notifier

# Initialize colorama for Windows support
init()

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Add project root to path to import model
project_root = str(Path(__file__).resolve().parent.parent.parent)
sys.path.append(project_root)
from src.model.theta_model import ThetaModel


# =============================================================================
# RTX 3060 12GB OPTIMIZATIONS - New Utility Classes
# =============================================================================

class ExponentialMovingAverage:
    """
    Exponential Moving Average for model weights.
    Smooths weight updates for better final model quality.
    Optimized for RTX 3060 with CPU offloading option.
    
    Supports warmup epochs where EMA is disabled to allow initial learning.
    """
    def __init__(self, model, decay=0.999, cpu_offload=True, warmup_epochs=0):
        self.decay = decay
        self.cpu_offload = cpu_offload
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        self.device = 'cpu' if cpu_offload else next(model.parameters()).device
        
        # Store shadow copies of parameters
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().to(self.device)
        
        if warmup_epochs > 0:
            logger.info(f"EMA will be inactive for first {warmup_epochs} epochs (warmup)")
    
    def set_epoch(self, epoch):
        """Set current epoch for warmup tracking."""
        self.current_epoch = epoch
    
    def is_active(self):
        """Check if EMA should be active (past warmup period)."""
        return self.current_epoch >= self.warmup_epochs
    
    def update(self, model):
        """Update shadow weights with current model weights."""
        # Fix #5: During warmup, don't update shadow at all (let model learn freely)
        # Shadow weights stay at initialization until warmup ends
        if not self.is_active():
            return
        
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                new_average = (1.0 - self.decay) * param.data.to(self.device) + self.decay * self.shadow[name]
                self.shadow[name] = new_average
    
    def apply_shadow(self, model):
        """Apply shadow weights to model for evaluation."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].to(param.device)
    
    def restore(self, model):
        """Restore original weights after evaluation."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing loss for language modeling.
    Reduces overconfidence and improves generalization.
    """
    def __init__(self, smoothing=0.1, ignore_index=-100):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.confidence = 1.0 - smoothing
    
    def forward(self, logits, labels):
        """
        Args:
            logits: (batch_size, seq_len, vocab_size)
            labels: (batch_size, seq_len)
        """
        vocab_size = logits.size(-1)
        
        # Reshape for computation
        logits_flat = logits.view(-1, vocab_size)
        labels_flat = labels.view(-1)
        
        # Create mask for non-ignored tokens
        mask = labels_flat != self.ignore_index
        
        # Compute log probabilities
        log_probs = F.log_softmax(logits_flat, dim=-1)
        
        # One-hot encode labels
        labels_masked = labels_flat.clone()
        labels_masked[~mask] = 0
        
        # Compute NLL loss component
        nll_loss = -log_probs.gather(dim=-1, index=labels_masked.unsqueeze(-1)).squeeze(-1)
        
        # Compute smooth loss component (uniform distribution)
        smooth_loss = -log_probs.mean(dim=-1)
        
        # Combine losses
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        
        # Apply mask and compute mean
        loss = (loss * mask.float()).sum() / mask.float().sum().clamp(min=1.0)
        
        return loss


class CurriculumSampler(Sampler):
    """
    Curriculum learning sampler - orders samples from simple to complex.
    Simple = shorter sequences, Complex = longer sequences.
    
    Optimized: Uses character length as proxy for token length (much faster).
    """
    def __init__(self, dataset, tokenizer, num_epochs, current_epoch=0, start_fraction=0.7):
        self.dataset = dataset
        self.num_epochs = num_epochs
        self.current_epoch = current_epoch
        self.start_fraction = start_fraction  # Configurable starting fraction
        
        # Fix #6: Enhanced difficulty score beyond just character length
        # Character length alone misleads: long conversational fluff ≠ hard, short math ≠ easy
        logger.info(f"Calculating difficulty scores for {len(dataset.formatted_data)} examples...")
        
        try:
            self.lengths = []
            for text in dataset.formatted_data:
                # Base: character length
                difficulty = len(text)
                # Bonus for code blocks (technical complexity)
                difficulty += 3 * text.count("```")
                # Bonus for equations/assignments (mathematical complexity)
                difficulty += 2 * text.count("=")
                # Bonus for bullet points/lists (structured content)
                difficulty += text.count("\n-") + text.count("\n*")
                # Bonus for technical terms (approximation)
                difficulty += text.count("function") + text.count("class") + text.count("import")
                self.lengths.append(difficulty)
            logger.info(f"Difficulty scores calculated (min: {min(self.lengths)}, max: {max(self.lengths)}, avg: {np.mean(self.lengths):.0f})")
        except Exception as e:
            logger.error(f"Error calculating difficulty: {e}")
            # Fallback: random ordering
            self.lengths = list(range(len(dataset.formatted_data)))
        
        # Sort indices by length
        self.sorted_indices = np.argsort(self.lengths)
        logger.info(f"Curriculum sampler ready with {len(self.sorted_indices)} samples (start fraction: {start_fraction:.0%})")
        
    def set_epoch(self, epoch):
        """Update the current epoch for curriculum progression."""
        self.current_epoch = epoch
        
    def __iter__(self):
        # Calculate curriculum progress (0.0 to 1.0)
        progress = min(1.0, (self.current_epoch + 1) / max(1, self.num_epochs // 2))
        
        # Determine how much of the dataset to use based on progress
        # Start with configurable fraction, gradually include all
        use_fraction = self.start_fraction + (1.0 - self.start_fraction) * progress
        
        num_samples = int(len(self.sorted_indices) * use_fraction)
        indices = self.sorted_indices[:num_samples].copy()
        
        # Shuffle within the selected subset
        np.random.shuffle(indices)
        
        return iter(indices.tolist())
    
    def __len__(self):
        progress = min(1.0, (self.current_epoch + 1) / max(1, self.num_epochs // 2))
        use_fraction = self.start_fraction + (1.0 - self.start_fraction) * progress
        return int(len(self.sorted_indices) * use_fraction)


class DomainStratifiedSampler(Sampler):
    """
    RECOMMENDATION #3: Domain-Stratified Batch Sampling
    
    Ensures each batch contains samples from multiple domains to prevent
    catastrophic forgetting of underrepresented domains during training.
    """
    
    def __init__(self, dataset, batch_size, domain_indices=None, target_distribution=None):
        """
        Args:
            dataset: Dataset (or subset) to sample from
            batch_size: Size of each batch
            domain_indices: Dict of domain -> list of indices in dataset coordinate space
                           If None, will try to read from dataset.domain_indices
            target_distribution: Dict of domain -> target fraction (optional)
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.dataset_size = len(dataset)
        
        # Use provided domain_indices or fall back to dataset attribute
        if domain_indices is not None:
            self.domain_indices = domain_indices
        elif hasattr(dataset, 'domain_indices'):
            self.domain_indices = dataset.domain_indices
        else:
            raise ValueError("domain_indices must be provided or dataset must have domain_indices attribute")
        
        # Default target distribution if not specified
        self.target_distribution = target_distribution or {
            'programming': 0.20,
            'cybersecurity': 0.15,
            'conversational': 0.15,
            'general_tech': 0.15,
            'data_science': 0.10,
            'networking': 0.10,
            'human_like_behavior': 0.10,
            'other': 0.05
        }
        
        # Map domains to target distribution keys
        self.domain_mapping = self._create_domain_mapping()
        
        logger.info(f"DomainStratifiedSampler initialized with {len(self.domain_indices)} domains")
    
    def _create_domain_mapping(self):
        """Map actual domain names to target distribution categories."""
        mapping = {}
        target_keys = list(self.target_distribution.keys())
        
        for domain in self.domain_indices.keys():
            domain_lower = domain.lower()
            matched = False
            
            for target in target_keys:
                if target in domain_lower or domain_lower in target:
                    mapping[domain] = target
                    matched = True
                    break
            
            if not matched:
                mapping[domain] = 'other'
        
        return mapping
    
    def __iter__(self):
        """Generate stratified batches."""
        # Calculate samples per domain per batch
        samples_per_domain = {}
        for domain, fraction in self.target_distribution.items():
            samples_per_domain[domain] = max(1, int(self.batch_size * fraction))
        
        # Fix #1: Ensure total exactly equals batch_size with shave loop
        total_per_batch = sum(samples_per_domain.values())
        if total_per_batch > self.batch_size:
            # Scale down proportionally first
            scale = self.batch_size / total_per_batch
            samples_per_domain = {d: max(1, int(c * scale)) for d, c in samples_per_domain.items()}
            total_per_batch = sum(samples_per_domain.values())
            
            # Shave loop: reduce counts until sum == batch_size
            while total_per_batch > self.batch_size:
                # Find domain with highest count (excluding minimum of 1)
                max_domain = max((d for d, c in samples_per_domain.items() if c > 1), 
                                key=lambda d: samples_per_domain[d], default=None)
                if max_domain is None:
                    break  # All at minimum, can't reduce further
                samples_per_domain[max_domain] -= 1
                total_per_batch -= 1
        
        if total_per_batch < self.batch_size:
            samples_per_domain['other'] = samples_per_domain.get('other', 0) + (self.batch_size - total_per_batch)
        
        # Create shuffled copies of domain indices
        domain_queues = {}
        for domain, indices in self.domain_indices.items():
            shuffled = indices.copy()
            np.random.shuffle(shuffled)
            target_domain = self.domain_mapping.get(domain, 'other')
            if target_domain not in domain_queues:
                domain_queues[target_domain] = []
            domain_queues[target_domain].extend(shuffled)
        
        # Shuffle each queue
        for domain in domain_queues:
            np.random.shuffle(domain_queues[domain])
        
        # Fix #7: Store original queues for refilling to guarantee __len__() indices
        original_queues = {d: list(q) for d, q in domain_queues.items()}
        
        # Generate batches
        num_batches = self.dataset_size // self.batch_size
        target_indices = num_batches * self.batch_size
        
        all_indices = []
        for _ in range(num_batches):
            batch = []
            for domain, count in samples_per_domain.items():
                queue = domain_queues.get(domain, [])
                for _ in range(count):
                    if queue:
                        batch.append(queue.pop())
                    else:
                        # Fix #2: Pick from largest non-empty bucket instead of always 'other'
                        non_empty = [(d, q) for d, q in domain_queues.items() if q]
                        if non_empty:
                            largest_domain = max(non_empty, key=lambda x: len(x[1]))[0]
                            batch.append(domain_queues[largest_domain].pop())
            
            # Fill remaining slots from any non-empty queue
            while len(batch) < self.batch_size:
                non_empty = [(d, q) for d, q in domain_queues.items() if q]
                if non_empty:
                    # Pick from largest to balance
                    largest_domain = max(non_empty, key=lambda x: len(x[1]))[0]
                    batch.append(domain_queues[largest_domain].pop())
                else:
                    # Fix #7: All queues empty - refill from original and reshuffle
                    for d, orig_q in original_queues.items():
                        domain_queues[d] = list(orig_q)
                        np.random.shuffle(domain_queues[d])
                    # Try again
                    non_empty = [(d, q) for d, q in domain_queues.items() if q]
                    if non_empty:
                        largest_domain = max(non_empty, key=lambda x: len(x[1]))[0]
                        batch.append(domain_queues[largest_domain].pop())
                    else:
                        break  # Truly no data
            
            all_indices.extend(batch[:self.batch_size])
        
        # Fix #7: Guarantee exactly __len__() indices
        while len(all_indices) < target_indices:
            # Refill if needed
            non_empty = [(d, q) for d, q in domain_queues.items() if q]
            if not non_empty:
                for d, orig_q in original_queues.items():
                    domain_queues[d] = list(orig_q)
                    np.random.shuffle(domain_queues[d])
                non_empty = [(d, q) for d, q in domain_queues.items() if q]
            if non_empty:
                largest_domain = max(non_empty, key=lambda x: len(x[1]))[0]
                all_indices.append(domain_queues[largest_domain].pop())
            else:
                break
        
        return iter(all_indices[:target_indices])
    
    def __len__(self):
        return (self.dataset_size // self.batch_size) * self.batch_size


class DynamicWeightedSampler(Sampler):
    """
    Sampler that uses dynamic weights from DynamicCurriculumTracker.
    Implements true weighted curriculum: hard samples appear MORE frequently.
    
    Fix #1: Uses subset indices (not original dataset indices)
    Fix #2: Uses replacement=True so hard samples can appear multiple times
    """
    
    def __init__(self, dataset_size, tracker=None, oversample_ratio=1.2):
        """
        Args:
            dataset_size: Size of the training subset (not original dataset)
            tracker: DynamicCurriculumTracker instance
            oversample_ratio: How many samples per epoch (1.2 = 20% more, hard samples repeated)
        """
        self.dataset_size = dataset_size
        self.tracker = tracker
        self.weights = [1.0] * dataset_size
        self.oversample_ratio = oversample_ratio
        # Index mapping: subset_idx -> original_idx (set during training setup)
        self.subset_to_original = None
    
    def set_index_mapping(self, subset_indices):
        """Set mapping from subset indices to original dataset indices."""
        # subset_indices is train_dataset.indices
        self.subset_to_original = {i: orig_idx for i, orig_idx in enumerate(subset_indices)}
        self.original_to_subset = {orig_idx: i for i, orig_idx in enumerate(subset_indices)}
    
    def update_weights(self):
        """Update weights from tracker, converting to subset index space."""
        if self.tracker is not None:
            # Get weights in original index space
            original_weights = self.tracker.get_sample_weights()
            
            # Convert to subset index space
            if self.original_to_subset is not None:
                self.weights = []
                for subset_idx in range(self.dataset_size):
                    orig_idx = self.subset_to_original.get(subset_idx, subset_idx)
                    if orig_idx < len(original_weights):
                        self.weights.append(original_weights[orig_idx])
                    else:
                        self.weights.append(1.0)
            else:
                # No mapping, use directly (assumes tracker uses subset indices)
                self.weights = original_weights[:self.dataset_size]
    
    def __iter__(self):
        # Fix #2: Use replacement=True for true weighted curriculum
        # Hard samples will appear multiple times per epoch
        weights_tensor = torch.tensor(self.weights, dtype=torch.float)
        weights_tensor = weights_tensor / weights_tensor.sum()  # Normalize
        
        # Sample with replacement - hard samples appear more often
        num_samples = int(self.dataset_size * self.oversample_ratio)
        indices = torch.multinomial(weights_tensor, num_samples, replacement=True)
        return iter(indices.tolist())
    
    def __len__(self):
        return int(self.dataset_size * self.oversample_ratio)


class DynamicCurriculumTracker:
    """
    RECOMMENDATION #5: Dynamic Curriculum Based on Loss
    
    Tracks per-sample loss and adjusts sampling to focus on difficult examples.
    """
    
    def __init__(self, dataset_size, warmup_epochs=2):
        """
        Args:
            dataset_size: Total number of samples
            warmup_epochs: Number of epochs before activating dynamic curriculum
        """
        self.dataset_size = dataset_size
        self.warmup_epochs = warmup_epochs
        self.sample_losses = {}  # idx -> running average loss
        self.sample_counts = {}  # idx -> number of times seen
        self.current_epoch = 0
        self.loss_ema_decay = 0.9  # Exponential moving average for loss tracking
        self.sampler = None  # Will be set when sampler is created
        
        logger.info(f"DynamicCurriculumTracker initialized for {dataset_size} samples")
    
    def update_loss(self, indices, losses):
        """
        Update loss tracking for a batch of samples.
        
        Args:
            indices: Tensor or list of sample indices
            losses: Tensor or list of corresponding losses
        """
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
        if isinstance(losses, torch.Tensor):
            losses = losses.tolist()
        
        for idx, loss in zip(indices, losses):
            if idx in self.sample_losses:
                # EMA update
                self.sample_losses[idx] = (
                    self.loss_ema_decay * self.sample_losses[idx] + 
                    (1 - self.loss_ema_decay) * loss
                )
                self.sample_counts[idx] += 1
            else:
                self.sample_losses[idx] = loss
                self.sample_counts[idx] = 1
    
    def set_epoch(self, epoch):
        """Update current epoch."""
        self.current_epoch = epoch
    
    def get_sample_weights(self):
        """
        Get sampling weights based on loss (higher loss = higher weight).
        
        Returns:
            List of weights for each sample index
        """
        if self.current_epoch < self.warmup_epochs or not self.sample_losses:
            # During warmup, use uniform weights
            return [1.0] * self.dataset_size
        
        # Calculate median loss
        losses = list(self.sample_losses.values())
        median_loss = np.median(losses) if losses else 1.0
        
        weights = []
        for idx in range(self.dataset_size):
            if idx in self.sample_losses:
                loss = self.sample_losses[idx]
                # Boost weight for high-loss samples (up to 2x)
                if loss > median_loss:
                    weight = min(2.0, 1.0 + (loss - median_loss) / median_loss)
                else:
                    weight = 1.0
            else:
                # Never-seen samples get slight boost
                weight = 1.2
            weights.append(weight)
        
        return weights
    
    def get_difficult_sample_ratio(self):
        """Get ratio of samples above median loss."""
        if not self.sample_losses:
            return 0.0
        
        losses = list(self.sample_losses.values())
        median_loss = np.median(losses)
        difficult_count = sum(1 for l in losses if l > median_loss)
        return difficult_count / len(losses)
    
    def get_stats(self):
        """Get curriculum tracking statistics."""
        if not self.sample_losses:
            return {"tracked_samples": 0}
        
        losses = list(self.sample_losses.values())
        return {
            "tracked_samples": len(self.sample_losses),
            "mean_loss": np.mean(losses),
            "median_loss": np.median(losses),
            "min_loss": np.min(losses),
            "max_loss": np.max(losses),
            "std_loss": np.std(losses),
            "difficult_ratio": self.get_difficult_sample_ratio()
        }


class CodeContrastiveLoss(nn.Module):
    """
    RECOMMENDATION #4: Contrastive Loss for Code vs Natural Language
    
    Adds auxiliary binary classification loss to help model distinguish
    when code examples are appropriate.
    """
    
    def __init__(self, hidden_size, dropout=0.1):
        """
        Args:
            hidden_size: Size of model hidden states (e.g., 1024 for GPT-2 medium)
            dropout: Dropout probability
        """
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
        self.loss_fn = nn.BCEWithLogitsLoss()
        
        logger.info(f"CodeContrastiveLoss initialized with hidden_size={hidden_size}")
    
    def forward(self, hidden_states, has_code_labels, attention_mask=None):
        """
        Compute contrastive loss for code detection.
        
        Args:
            hidden_states: Model hidden states [batch, seq_len, hidden_size]
            has_code_labels: Binary labels [batch] indicating if response has code
            attention_mask: Optional mask for pooling [batch, seq_len]
        
        Returns:
            loss: Binary cross-entropy loss
            predictions: Sigmoid probabilities [batch]
        """
        if isinstance(hidden_states, (tuple, list)):
            hidden_states = hidden_states[-1]
        # Pool hidden states (mean pooling over sequence)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_hidden = (hidden_states * mask_expanded).sum(dim=1)
            count = mask_expanded.sum(dim=1).clamp(min=1)
            pooled = sum_hidden / count
        else:
            pooled = hidden_states.mean(dim=1)
        
        # Classify
        logits = self.classifier(pooled).squeeze(-1)
        predictions = torch.sigmoid(logits)
        
        # Compute loss
        loss = self.loss_fn(logits, has_code_labels)
        
        return loss, predictions


class CPUOffloadOptimizer:
    """
    Wrapper for optimizer with CPU offloading for RTX 3060 12GB.
    Offloads optimizer states to CPU to save GPU memory.
    
    Optimized for 6-core Ryzen 5-5500:
    - Uses non-blocking transfers to overlap CPU/GPU work
    - Batches state transfers to reduce overhead
    """
    def __init__(self, optimizer, offload_fraction=0.5):
        self.optimizer = optimizer
        self.offload_fraction = offload_fraction
        self.offloaded_states = {}
        self._setup_offloading()
        
    def _setup_offloading(self):
        """Setup CPU offloading for optimizer states."""
        total_params = sum(len(group['params']) for group in self.optimizer.param_groups)
        offload_count = int(total_params * self.offload_fraction)
        
        current_count = 0
        for group_idx, group in enumerate(self.optimizer.param_groups):
            for param_idx, param in enumerate(group['params']):
                if current_count < offload_count and param.requires_grad:
                    key = (group_idx, param_idx)
                    self.offloaded_states[key] = True
                    current_count += 1
                    
        logger.info(f"CPU Offloading: {current_count}/{total_params} parameters marked for offloading to CPU")
    
    def pre_step(self):
        """Move offloaded states to GPU before optimizer step. Call before scaler.step()."""
        for (group_idx, param_idx), _ in self.offloaded_states.items():
            param = self.optimizer.param_groups[group_idx]['params'][param_idx]
            state = self.optimizer.state.get(param)
            if state:
                for key, value in state.items():
                    if isinstance(value, torch.Tensor) and value.device.type == 'cpu':
                        # Use non_blocking for better CPU/GPU overlap
                        state[key] = value.cuda(non_blocking=True)
    
    def post_step(self):
        """Move states back to CPU after optimizer step. Call after scaler.step()."""
        # Synchronize to ensure GPU operations are complete
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        for (group_idx, param_idx), _ in self.offloaded_states.items():
            param = self.optimizer.param_groups[group_idx]['params'][param_idx]
            state = self.optimizer.state.get(param)
            if state:
                for key, value in state.items():
                    if isinstance(value, torch.Tensor) and value.device.type == 'cuda':
                        state[key] = value.cpu()
        
        # Clear CUDA cache periodically (not every step to reduce overhead)
        # This is handled externally now
    
    def step(self):
        """Perform full optimizer step with CPU offloading (for non-scaler use)."""
        self.pre_step()
        self.optimizer.step()
        self.post_step()
    
    def zero_grad(self, set_to_none=False):
        self.optimizer.zero_grad(set_to_none=set_to_none)
    
    @property
    def param_groups(self):
        return self.optimizer.param_groups
    
    @property
    def state(self):
        return self.optimizer.state


class DomainMetrics:
    """
    Domain-specific metrics configuration and evaluation.
    Defines keywords, expected patterns, and quality indicators per domain.
    """
    
    # Domain categories for grouped evaluation
    DOMAIN_CATEGORIES = {
        'conversation': ['small_talk', 'conversation', 'conversational_dynamics', 
                         'conversational', 'basic_conversation'],
        'technical': ['cybersecurity', 'programming', 'networking', 'cloud_computing',
                      'data_science', 'general_tech', 'technical_domain', 'technical_patterns',
                      'code_review', 'security', 'architecture', 'debugging', 
                      'api_design', 'error_handling'],
        'human_behavior': ['human_like_behavior', 'cognitive_reasoning', 'psychological_frameworks',
                           'human_experience', 'emotional_intelligence', 'interpersonal_intelligence',
                           'memory_simulation', 'personality', 'opinions'],
        'reasoning': ['ethical_reasoning', 'tactical_knowledge', 'combat_domain'],
        'creative': ['humor_comprehension', 'cultural_contexts', 'narrative_experiences'],
        'mathematics': ['mathematics', 'math']
    }
    
    # Domain-specific keywords for quality assessment
    DOMAIN_KEYWORDS = {
        'conversation': [
            'hello', 'hi', 'how are you', 'nice to meet', 'thanks', 'please',
            'tell me', 'what do you think', 'i think', 'feel', 'believe',
            'interesting', 'agree', 'understand', 'sure', 'of course'
        ],
        'technical': [
            'function', 'class', 'method', 'algorithm', 'data', 'system',
            'implement', 'code', 'error', 'debug', 'security', 'network',
            'api', 'database', 'server', 'client', 'protocol', 'encryption',
            'authentication', 'framework', 'library', 'module', 'package'
        ],
        'human_behavior': [
            'emotion', 'feeling', 'cognitive', 'psychology', 'behavior',
            'personality', 'memory', 'experience', 'perception', 'awareness',
            'empathy', 'understand', 'perspective', 'motivation', 'reaction'
        ],
        'reasoning': [
            'therefore', 'because', 'consequently', 'thus', 'hence',
            'consider', 'analyze', 'evaluate', 'conclude', 'reason',
            'logic', 'evidence', 'argument', 'premise', 'conclusion'
        ],
        'creative': [
            'imagine', 'story', 'creative', 'humor', 'joke', 'funny',
            'culture', 'tradition', 'art', 'expression', 'metaphor'
        ],
        'mathematics': [
            'calculate', 'equation', 'formula', 'solve', 'proof',
            'theorem', 'number', 'variable', 'function', 'derivative',
            'integral', 'matrix', 'vector', 'probability', 'statistics'
        ]
    }
    
    # Sample prompts for each domain category (for generation testing)
    SAMPLE_PROMPTS = {
        'conversation': [
            "Question: How are you doing today?\nAnswer:",
            "Question: What's your favorite thing to talk about?\nAnswer:",
            "Question: Tell me something interesting about yourself.\nAnswer:",
            "Question: What do you think about the weather?\nAnswer:",
            "Question: Nice to meet you! What should we discuss?\nAnswer:"
        ],
        'technical': [
            "Question: How do I fix a null pointer exception in Python?\nAnswer:",
            "Question: Explain how HTTPS encryption works.\nAnswer:",
            "Question: What is the difference between TCP and UDP?\nAnswer:",
            "Question: How do I implement a binary search algorithm?\nAnswer:",
            "Question: What are best practices for API security?\nAnswer:"
        ],
        'human_behavior': [
            "Question: How do humans process emotional experiences?\nAnswer:",
            "Question: What is cognitive dissonance?\nAnswer:",
            "Question: Explain the concept of empathy.\nAnswer:",
            "Question: How does memory work in the brain?\nAnswer:",
            "Question: What motivates human behavior?\nAnswer:"
        ],
        'reasoning': [
            "Question: If A implies B, and B implies C, what can we conclude?\nAnswer:",
            "Question: What is the ethical consideration in this scenario?\nAnswer:",
            "Question: Analyze the strategic implications of this decision.\nAnswer:",
            "Question: How would you evaluate this argument?\nAnswer:",
            "Question: What evidence supports this conclusion?\nAnswer:"
        ],
        'mathematics': [
            "Question: Solve for x: 2x + 5 = 15\nAnswer:",
            "Question: What is the derivative of x^2 + 3x?\nAnswer:",
            "Question: Calculate the probability of rolling a 6 twice.\nAnswer:",
            "Question: Simplify the expression: (a+b)^2\nAnswer:",
            "Question: What is the area of a circle with radius 5?\nAnswer:"
        ]
    }
    
    # Expected response characteristics per domain
    EXPECTED_CHARACTERISTICS = {
        'conversation': {'min_length': 10, 'max_length': 100, 'formal': False},
        'technical': {'min_length': 30, 'max_length': 300, 'formal': True},
        'human_behavior': {'min_length': 30, 'max_length': 200, 'formal': True},
        'reasoning': {'min_length': 40, 'max_length': 250, 'formal': True},
        'creative': {'min_length': 20, 'max_length': 200, 'formal': False},
        'mathematics': {'min_length': 15, 'max_length': 150, 'formal': True}
    }


class ValidationMetrics:
    """
    Extended validation metrics with domain-specific evaluation.
    Tracks perplexity, generation quality, domain accuracy, and conversation metrics.
    
    Optimized for Theta AI's diverse training domains:
    - Technical (cybersecurity, programming, networking, etc.)
    - Conversational (small talk, basic conversation)
    - Human behavior (emotional intelligence, cognitive reasoning)
    - Reasoning (ethical, tactical, logical)
    - Mathematics
    """
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.metrics_history = defaultdict(list)
        self.domain_metrics = defaultdict(lambda: defaultdict(list))
        self.domain_config = DomainMetrics()
        
        # Try to load evaluation metrics
        if METRICS_AVAILABLE:
            try:
                self.bleu = load_metric("bleu")
                self.rouge = load_metric("rouge")
            except Exception as e:
                logger.warning(f"Could not load metrics: {e}")
                self.bleu = None
                self.rouge = None
        else:
            self.bleu = None
            self.rouge = None
    
    def compute_perplexity(self, loss):
        """Compute perplexity from loss with safety guard."""
        if not np.isfinite(loss):
            return float('inf')
        return math.exp(min(loss, 100))  # Clamp to avoid overflow
    
    def compute_token_accuracy(self, logits, labels):
        """Compute token-level prediction accuracy."""
        predictions = torch.argmax(logits, dim=-1)
        mask = labels != -100
        correct = (predictions == labels) & mask
        accuracy = correct.sum().float() / mask.sum().float()
        return accuracy.item()
    
    def _get_domain_category(self, domain):
        """Map a specific domain to its category."""
        domain_lower = domain.lower() if domain else 'general'
        for category, domains in self.domain_config.DOMAIN_CATEGORIES.items():
            if domain_lower in domains:
                return category
        return 'general'
    
    def compute_keyword_overlap(self, text, domain_category):
        """Compute how many domain keywords appear in the text."""
        if domain_category not in self.domain_config.DOMAIN_KEYWORDS:
            return 0.0
        
        keywords = self.domain_config.DOMAIN_KEYWORDS[domain_category]
        text_lower = text.lower()
        
        matches = sum(1 for kw in keywords if kw in text_lower)
        return matches / len(keywords) if keywords else 0.0
    
    def compute_response_length_score(self, text, domain_category):
        """
        Score response length appropriateness for domain.
        Returns 1.0 for perfect length, decreasing for too short/long.
        """
        if domain_category not in self.domain_config.EXPECTED_CHARACTERISTICS:
            return 1.0
        
        chars = self.domain_config.EXPECTED_CHARACTERISTICS[domain_category]
        word_count = len(text.split())
        
        if word_count < chars['min_length']:
            # Too short - penalize
            return max(0.0, word_count / chars['min_length'])
        elif word_count > chars['max_length']:
            # Too long - slight penalty
            return max(0.5, 1.0 - (word_count - chars['max_length']) / chars['max_length'])
        else:
            return 1.0
    
    def compute_coherence_score(self, text):
        """
        Compute basic coherence score based on:
        - Sentence structure
        - No excessive repetition
        - Proper ending punctuation
        """
        if not text or len(text) < 5:
            return 0.0
        
        score = 1.0
        
        # Check for sentence ending
        if not any(text.rstrip().endswith(p) for p in '.!?'):
            score -= 0.2
        
        # Check for excessive repetition (same word 3+ times in a row)
        words = text.lower().split()
        if len(words) >= 3:
            for i in range(len(words) - 2):
                if words[i] == words[i+1] == words[i+2]:
                    score -= 0.3
                    break
        
        # Check for minimum word variety
        if len(words) > 5:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:  # Less than 30% unique words
                score -= 0.2
        
        return max(0.0, score)
    
    def compute_conversational_markers(self, text):
        """
        Detect conversational quality markers for dialogue domains.
        """
        markers = {
            'greeting': ['hello', 'hi', 'hey', 'greetings'],
            'acknowledgment': ['yes', 'sure', 'of course', 'certainly', 'absolutely'],
            'empathy': ['understand', 'feel', 'appreciate', 'sorry'],
            'engagement': ['tell me', 'what about', 'how about', 'what do you think'],
            'politeness': ['please', 'thank', 'appreciate', 'kind']
        }
        
        text_lower = text.lower()
        scores = {}
        
        for marker_type, keywords in markers.items():
            scores[marker_type] = any(kw in text_lower for kw in keywords)
        
        return sum(scores.values()) / len(scores)
    
    def compute_technical_accuracy_markers(self, text):
        """
        Detect technical accuracy markers for code/tech domains.
        """
        markers = {
            'code_blocks': text.count('```') >= 2 or text.count('`') >= 2,
            'technical_terms': any(term in text.lower() for term in 
                                   ['function', 'class', 'method', 'variable', 'parameter']),
            'structured_explanation': any(phrase in text.lower() for phrase in 
                                          ['first', 'step', 'then', 'finally', 'because']),
            'examples': 'example' in text.lower() or 'e.g.' in text.lower(),
            'specificity': any(char.isdigit() for char in text)  # Contains numbers
        }
        
        return sum(markers.values()) / len(markers)
    
    def compute_reasoning_markers(self, text):
        """
        Detect logical reasoning markers for reasoning domains.
        """
        markers = {
            'causal': any(word in text.lower() for word in 
                         ['because', 'therefore', 'thus', 'hence', 'consequently']),
            'conditional': any(word in text.lower() for word in 
                              ['if', 'when', 'unless', 'provided']),
            'comparative': any(word in text.lower() for word in 
                              ['however', 'although', 'whereas', 'compared']),
            'conclusion': any(word in text.lower() for word in 
                             ['conclude', 'result', 'finally', 'ultimately']),
            'evidence': any(word in text.lower() for word in 
                           ['evidence', 'data', 'shows', 'indicates', 'suggests'])
        }
        
        return sum(markers.values()) / len(markers)
    
    def compute_domain_specific_score(self, text, domain_category):
        """Compute domain-specific quality score."""
        if domain_category == 'conversation':
            return self.compute_conversational_markers(text)
        elif domain_category == 'technical':
            return self.compute_technical_accuracy_markers(text)
        elif domain_category in ['reasoning', 'human_behavior']:
            return self.compute_reasoning_markers(text)
        elif domain_category == 'mathematics':
            # Check for mathematical content
            has_numbers = any(c.isdigit() for c in text)
            has_operators = any(op in text for op in ['+', '-', '*', '/', '=', '^'])
            has_math_words = any(w in text.lower() for w in ['equals', 'solve', 'calculate', 'answer'])
            return (has_numbers + has_operators + has_math_words) / 3
        else:
            return 0.5  # Default score for unknown domains
    
    def evaluate_generation(self, model, device, domain_category='general', num_samples=3):
        """
        Evaluate model generation quality for a specific domain.
        Returns dict of domain-specific metrics.
        """
        model.eval()
        metrics = {
            'coherence': [],
            'length_appropriateness': [],
            'keyword_overlap': [],
            'domain_score': [],
            'avg_length': []
        }
        
        # Get prompts for this domain
        prompts = self.domain_config.SAMPLE_PROMPTS.get(
            domain_category, 
            self.domain_config.SAMPLE_PROMPTS.get('conversation', [])
        )
        
        with torch.no_grad():
            for prompt in prompts[:num_samples]:
                inputs = self.tokenizer(
                    prompt, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=256
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                try:
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=100,
                        do_sample=True,
                        temperature=0.8,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.pad_token_id,
                        no_repeat_ngram_size=3
                    )
                    
                    # Decode response (remove prompt)
                    generated = self.tokenizer.decode(
                        outputs[0][inputs['input_ids'].shape[1]:], 
                        skip_special_tokens=True
                    )
                    
                    # Compute metrics
                    metrics['coherence'].append(self.compute_coherence_score(generated))
                    metrics['length_appropriateness'].append(
                        self.compute_response_length_score(generated, domain_category)
                    )
                    metrics['keyword_overlap'].append(
                        self.compute_keyword_overlap(generated, domain_category)
                    )
                    metrics['domain_score'].append(
                        self.compute_domain_specific_score(generated, domain_category)
                    )
                    metrics['avg_length'].append(len(generated.split()))
                    
                except Exception as e:
                    logger.warning(f"Generation failed for {domain_category}: {e}")
                    continue
        
        # Average the metrics
        return {k: np.mean(v) if v else 0.0 for k, v in metrics.items()}
    
    def update(self, loss, logits=None, labels=None, epoch=None, domain=None):
        """Update metrics with new batch results."""
        perplexity = self.compute_perplexity(loss)
        self.metrics_history['perplexity'].append(perplexity)
        
        if logits is not None and labels is not None:
            accuracy = self.compute_token_accuracy(logits, labels)
            self.metrics_history['token_accuracy'].append(accuracy)
            
            # Track per-domain accuracy if domain provided
            if domain:
                category = self._get_domain_category(domain)
                self.domain_metrics[category]['accuracy'].append(accuracy)
                self.domain_metrics[category]['perplexity'].append(perplexity)
    
    def get_summary(self):
        """Get summary of all metrics including domain-specific."""
        summary = {}
        
        # Global metrics
        for metric, values in self.metrics_history.items():
            if values:
                summary[f'{metric}_mean'] = np.mean(values)
                summary[f'{metric}_std'] = np.std(values)
        
        # Domain-specific metrics
        for domain, metrics in self.domain_metrics.items():
            for metric, values in metrics.items():
                if values:
                    summary[f'{domain}_{metric}_mean'] = np.mean(values)
        
        return summary
    
    def get_domain_summary(self):
        """Get detailed domain-specific summary."""
        summary = {}
        for domain, metrics in self.domain_metrics.items():
            summary[domain] = {}
            for metric, values in metrics.items():
                if values:
                    summary[domain][metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'count': len(values)
                    }
        return summary
    
    def run_full_domain_evaluation(self, model, device):
        """
        Run comprehensive evaluation across all domain categories.
        This is more expensive - run at end of epoch or training.
        """
        logger.info("Running full domain evaluation...")
        results = {}
        
        for domain_category in ['conversation', 'technical', 'reasoning', 'mathematics']:
            try:
                domain_results = self.evaluate_generation(
                    model, device, domain_category, num_samples=2
                )
                results[domain_category] = domain_results
                logger.info(f"  {domain_category}: coherence={domain_results['coherence']:.3f}, "
                           f"domain_score={domain_results['domain_score']:.3f}")
            except Exception as e:
                logger.warning(f"Evaluation failed for {domain_category}: {e}")
                results[domain_category] = {}
        
        return results
    
    def reset(self):
        """Reset metrics for new epoch."""
        self.metrics_history = defaultdict(list)
        self.domain_metrics = defaultdict(lambda: defaultdict(list))


def get_llrd_optimizer_groups(model, base_lr=2e-5, weight_decay=0.03, llrd_factor=0.95):
    """
    Create optimizer parameter groups with Layer-wise Learning Rate Decay.
    Lower layers get smaller learning rates since they contain more general features.
    
    Args:
        model: The GPT-2 model
        base_lr: Base learning rate for top layers
        weight_decay: Weight decay for regularization
        llrd_factor: Decay factor per layer (0.95 = 5% reduction per layer)
    
    Returns:
        List of parameter groups for optimizer
    """
    no_decay = ['bias', 'LayerNorm.weight', 'ln_']
    
    # Get the transformer layers
    if hasattr(model, 'transformer'):
        layers = model.transformer.h
        num_layers = len(layers)
    else:
        # Fallback for different model structures
        logger.warning("Could not detect transformer layers for LLRD, using uniform LR")
        return [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay, 'lr': base_lr},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': base_lr}
        ]
    
    optimizer_groups = []
    
    # Embeddings - lowest learning rate
    embed_lr = base_lr * (llrd_factor ** (num_layers + 1))
    optimizer_groups.append({
        'params': [p for n, p in model.named_parameters() if 'wte' in n or 'wpe' in n],
        'weight_decay': 0.0,
        'lr': embed_lr
    })
    
    # Transformer layers - increasing LR from bottom to top
    for layer_idx in range(num_layers):
        layer_lr = base_lr * (llrd_factor ** (num_layers - layer_idx))
        layer_name = f'transformer.h.{layer_idx}.'
        
        # Parameters with weight decay
        optimizer_groups.append({
            'params': [p for n, p in model.named_parameters() 
                      if layer_name in n and not any(nd in n for nd in no_decay)],
            'weight_decay': weight_decay,
            'lr': layer_lr
        })
        
        # Parameters without weight decay
        optimizer_groups.append({
            'params': [p for n, p in model.named_parameters() 
                      if layer_name in n and any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
            'lr': layer_lr
        })
    
    # LM head - highest learning rate
    optimizer_groups.append({
        'params': [p for n, p in model.named_parameters() if 'lm_head' in n],
        'weight_decay': weight_decay,
        'lr': base_lr
    })
    
    # Final layer norm
    optimizer_groups.append({
        'params': [p for n, p in model.named_parameters() if 'ln_f' in n],
        'weight_decay': 0.0,
        'lr': base_lr
    })
    
    # Filter out empty groups
    optimizer_groups = [g for g in optimizer_groups if len(g['params']) > 0]
    
    logger.info(f"LLRD configured with {len(optimizer_groups)} parameter groups")
    logger.info(f"Learning rates range from {embed_lr:.2e} (embeddings) to {base_lr:.2e} (top layers)")
    
    return optimizer_groups


def add_gradient_noise(model, noise_scale=0.01):
    """
    Add gradient noise to help escape sharp minima.
    
    Fix #8: Removed unused decay_factor parameter. Noise decay is handled
    externally via noise_scale_epoch = gradient_noise_scale / (1 + epoch * 0.1)
    
    Args:
        model: The model with computed gradients
        noise_scale: Scale of noise (pre-decayed by caller)
    """
    for param in model.parameters():
        if param.grad is not None:
            noise = torch.randn_like(param.grad) * noise_scale
            param.grad.add_(noise)


def compute_rdrop_loss(model, input_ids, attention_mask, labels, alpha=0.1, 
                       label_smoothing_fn=None, scaler=None, device_type="cuda",
                       output_hidden_states=False, quality_weights=None,
                       compute_per_sample=False):
    """
    Compute R-Drop regularization loss.
    Runs forward pass twice with different dropout and adds KL divergence.
    
    Args:
        model: The language model
        input_ids: Input token IDs
        attention_mask: Attention mask
        labels: Target labels
        alpha: Weight for KL divergence term
        label_smoothing_fn: Optional label smoothing loss function
        scaler: Optional gradient scaler for mixed precision
        device_type: Device type for autocast ('cuda' or 'cpu')
        output_hidden_states: Whether to return hidden states (for code contrastive)
        quality_weights: Optional per-sample quality weights
    
    Returns:
        dict with:
            total_loss: Combined loss (CE/LS + KL divergence)
            ce_loss: Average CE loss from both passes
            kl_loss: KL divergence between the two passes
            logits: Logits from first pass (for downstream use)
            hidden_states: Hidden states from first pass (if requested)
            per_sample_loss: Per-sample losses (if quality_weights provided or needed)
    """
    model.train()  # Ensure dropout is active
    
    # Use autocast for mixed precision (only effective on CUDA)
    use_amp = device_type == "cuda"
    
    # First forward pass (with hidden states if needed)
    with torch.amp.autocast(device_type, enabled=use_amp):
        outputs1 = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                        output_hidden_states=output_hidden_states)
        logits1 = outputs1.logits
        
        # Use label smoothing if provided, otherwise use model's CE loss
        if label_smoothing_fn is not None:
            loss1 = label_smoothing_fn(logits1, labels)
        else:
            loss1 = outputs1.loss
    
    # Second forward pass (different dropout mask, no hidden states needed)
    with torch.amp.autocast(device_type, enabled=use_amp):
        outputs2 = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits2 = outputs2.logits
        
        # Use label smoothing if provided, otherwise use model's CE loss
        if label_smoothing_fn is not None:
            loss2 = label_smoothing_fn(logits2, labels)
        else:
            loss2 = outputs2.loss
    
    # Average CE/LS loss from both passes
    ce_loss = (loss1 + loss2) / 2
    
    # Compute KL divergence between the two predictions
    # Only for non-padding tokens
    mask = labels != -100
    
    if mask.sum() > 0:
        p = F.log_softmax(logits1[mask], dim=-1)
        q = F.softmax(logits2[mask], dim=-1)
        kl_loss1 = F.kl_div(p, q, reduction='batchmean')
        
        p = F.log_softmax(logits2[mask], dim=-1)
        q = F.softmax(logits1[mask], dim=-1)
        kl_loss2 = F.kl_div(p, q, reduction='batchmean')
        
        kl_loss = (kl_loss1 + kl_loss2) / 2
    else:
        kl_loss = torch.tensor(0.0, device=ce_loss.device)
    
    # Fix #6: Compute per-sample losses when needed for quality weighting OR dynamic curriculum
    per_sample_loss = None
    need_per_sample = (quality_weights is not None) or compute_per_sample
    if need_per_sample:
        with torch.amp.autocast(device_type, enabled=use_amp):
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
            # Fix #4: Compute per-sample loss from BOTH passes and average
            token_loss1 = loss_fct(logits1.view(-1, logits1.size(-1)), labels.view(-1)).view(labels.size())
            token_loss2 = loss_fct(logits2.view(-1, logits2.size(-1)), labels.view(-1)).view(labels.size())
            token_loss = (token_loss1 + token_loss2) / 2  # Average both passes
            mask_float = (labels != -100).float()
            per_sample_loss = (token_loss * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1)
    
    # Apply quality weighting if provided
    if quality_weights is not None and per_sample_loss is not None:
        weighted_ce = (per_sample_loss * quality_weights).mean()
        # Replace ce_loss with weighted version
        total_loss = weighted_ce + alpha * kl_loss
    else:
        total_loss = ce_loss + alpha * kl_loss
    
    # Return dict with all needed values
    return {
        'total_loss': total_loss,
        'ce_loss': ce_loss,
        'kl_loss': kl_loss,
        'logits': logits1,
        'hidden_states': (outputs1.hidden_states[-1] if (output_hidden_states and hasattr(outputs1, 'hidden_states') and outputs1.hidden_states is not None) else None),
        'per_sample_loss': per_sample_loss
    }


# =============================================================================
# End of RTX 3060 Optimizations
# =============================================================================


class ThetaDataset(Dataset):
    """
    Custom dataset for Theta AI training with enhanced features:
    - Quality weighting for sample importance
    - Multi-turn context injection for conversation coherence
    - Domain tracking for stratified sampling
    - Code detection for contrastive learning
    
    IMPORTANT: This dataset only uses the tokenizer, NOT the model.
    Loading a model here would waste VRAM, break CPU offloading,
    and cause multiprocess locking issues with DataLoader workers.
    """
    
    def __init__(self, data_path, tokenizer, max_length=512):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the processed data JSON file
            tokenizer: Pre-initialized tokenizer from the model (DO NOT load model here!)
            max_length: Maximum sequence length
        """
        # Use the passed tokenizer - DO NOT load a model here!
        # Loading a model in the dataset wastes VRAM and breaks CPU offloading
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Ensure tokenizer has pad token set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"ThetaDataset initialized with tokenizer (vocab size: {len(self.tokenizer)})")
        
        # Load data
        with open(data_path, 'r', encoding='utf-8') as f:
            data_container = json.load(f)
            
        # Check if data is in the new format with qa_pairs and curriculum_datasets
        if isinstance(data_container, dict) and 'qa_pairs' in data_container:
            logger.info("Detected enhanced data format with QA pairs and curriculum datasets")
            self.data = data_container['qa_pairs']
            
            # Note the number of curriculum datasets available
            if 'curriculum_datasets' in data_container:
                num_curriculum_datasets = len(data_container['curriculum_datasets'])
                logger.info(f"Dataset includes {num_curriculum_datasets} native curriculum datasets")
        else:
            # Legacy format - data is already a list of QA pairs
            logger.info("Detected legacy data format (flat QA pairs)")
            self.data = data_container
        
        # === RECOMMENDATION #1: Quality Weighting ===
        # Extract quality scores for weighted sampling
        self.quality_weights = []
        
        # === RECOMMENDATION #2: Multi-Turn Context ===
        # Track conversation context for multi-turn samples
        
        # === RECOMMENDATION #3: Domain Tracking ===
        # Track domains for stratified batch sampling
        self.domains = []
        self.domain_indices = {}  # domain -> list of indices
        
        # === RECOMMENDATION #4: Code Detection ===
        # Track which samples contain code for contrastive learning
        self.has_code = []
            
        # Prepare formatted data for GPT-2 style training
        self.formatted_data = []
        for idx, item in enumerate(self.data):
            question = item.get('question', '')
            answer = item.get('answer', '')
            
            # === Multi-Turn Context Injection (Recommendation #2) ===
            # Check if this is a follow-up in a conversation
            if item.get('is_followup') and item.get('turn_number', 1) > 1:
                context_prefix = "[Continuing previous conversation] "
                question = context_prefix + question
            
            # Format as instruction-following text
            text = f"Question: {question}\nAnswer: {answer}\n\n"
            self.formatted_data.append(text)
            
            # === Quality Weighting (Recommendation #1) ===
            # Extract quality score (default 1.0 if not present)
            quality = item.get('quality_score', 1.0)
            if quality is None:
                quality = 1.0
            # Normalize quality to [0.5, 2.0] range to avoid extreme weights
            quality = max(0.5, min(2.0, quality))
            self.quality_weights.append(quality)
            
            # === Domain Tracking (Recommendation #3) ===
            domain = item.get('domain', 'general_tech')
            self.domains.append(domain)
            if domain not in self.domain_indices:
                self.domain_indices[domain] = []
            self.domain_indices[domain].append(idx)
            
            # === Code Detection (Recommendation #4) ===
            # Check if answer contains code blocks
            has_code = '```' in answer or item.get('has_code', False)
            self.has_code.append(1.0 if has_code else 0.0)
        
        # Log domain distribution
        domain_counts = {d: len(indices) for d, indices in self.domain_indices.items()}
        logger.info(f"Domain distribution: {domain_counts}")
        
        # Log code sample stats
        code_count = sum(self.has_code)
        logger.info(f"Samples with code: {int(code_count)}/{len(self.data)} ({100*code_count/len(self.data):.1f}%)")
            
        logger.info(f"Loaded {len(self.formatted_data)} training examples")
    
    def __len__(self):
        return len(self.formatted_data)
    
    def get_sample_weight(self, idx):
        """Get quality-based sample weight for weighted sampling."""
        return self.quality_weights[idx]
    
    def get_domain(self, idx):
        """Get domain for an index."""
        return self.domains[idx]
    
    def get_has_code(self, idx):
        """Get whether sample contains code."""
        return self.has_code[idx]
    
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
        
        # Fix #3: Explicitly return original dataset index for dynamic curriculum tracking
        # IMPORTANT: When using torch.utils.data.Subset, the `idx` passed here is the 
        # ORIGINAL dataset index (Subset calls self.dataset[self.indices[subset_idx]])
        # This is correct behavior - dynamic curriculum tracker uses original indices
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'quality_weight': torch.tensor(self.quality_weights[idx], dtype=torch.float),
            'has_code': torch.tensor(self.has_code[idx], dtype=torch.float),
            'idx': torch.tensor(idx, dtype=torch.long)  # Original dataset index
        }

class EarlyStopping:
    """Early stopping implementation to prevent overfitting."""
    
    def __init__(self, patience=3, min_delta=0.0, path='checkpoint.pt'):
        """
        Args:
            patience: Number of epochs to wait for improvement before stopping
            min_delta: Minimum change to qualify as improvement
            path: Path to save the best model
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.path = path
        
    def __call__(self, val_loss, model, state_dict_override=None):
        """
        Check if training should stop and save best model.
        
        Args:
            val_loss: Current validation loss
            model: Model to potentially save
            state_dict_override: Optional state dict to save instead of model.state_dict()
                                (Fix #5: allows saving EMA weights)
        """
        score = -val_loss  # Higher score is better (negative loss)
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, state_dict_override)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, state_dict_override)
            self.counter = 0
            
    def save_checkpoint(self, val_loss, model, state_dict_override=None):
        """Save model when validation loss decreases."""
        logger.info(f"Validation loss decreased to {val_loss:.6f}. Saving model...")
        if state_dict_override is not None:
            torch.save(state_dict_override, self.path)
        else:
            torch.save(model.state_dict(), self.path)

def train(args):
    """
    Train the Theta AI model with enhanced features.
    
    RTX 3060 12GB Optimizations:
    - LLRD (Layer-wise Learning Rate Decay)
    - CPU offloading for optimizer states
    - Label smoothing (0.1)
    - R-Drop regularization
    - Gradient noise injection
    - EMA (Exponential Moving Average)
    - Curriculum learning by difficulty
    - Extended validation metrics
    
    Args:
        args: Training arguments
    """
    # Set default gradient accumulation steps if not provided
    if not hasattr(args, 'gradient_accumulation_steps'):
        args.gradient_accumulation_steps = 4
    
    # Set new training enhancement defaults
    use_label_smoothing = getattr(args, 'label_smoothing', 0.1) > 0
    label_smoothing_value = getattr(args, 'label_smoothing', 0.1)
    use_rdrop = getattr(args, 'use_rdrop', True)
    rdrop_alpha = getattr(args, 'rdrop_alpha', 0.1)
    use_llrd = getattr(args, 'use_llrd', True)
    llrd_factor = getattr(args, 'llrd_factor', 0.95)
    use_ema = getattr(args, 'use_ema', True)
    ema_decay = getattr(args, 'ema_decay', 0.999)
    use_curriculum = getattr(args, 'use_curriculum', True)
    use_gradient_noise = getattr(args, 'use_gradient_noise', True)
    gradient_noise_scale = getattr(args, 'gradient_noise_scale', 0.01)
    use_cpu_offload = getattr(args, 'use_cpu_offload', True)
    cpu_offload_fraction = getattr(args, 'cpu_offload_fraction', 0.5)
    
    # === NEW TRAINING ENHANCEMENTS (5 Recommendations) ===
    use_quality_weighting = getattr(args, 'use_quality_weighting', True)  # Rec #1
    use_domain_stratified = getattr(args, 'use_domain_stratified', True)  # Rec #3
    use_code_contrastive = getattr(args, 'use_code_contrastive', True)    # Rec #4
    code_contrastive_weight = getattr(args, 'code_contrastive_weight', 0.1)
    use_dynamic_curriculum = getattr(args, 'use_dynamic_curriculum', True)  # Rec #5
    dynamic_curriculum_warmup = getattr(args, 'dynamic_curriculum_warmup', 2)
        
    # Check disk space before starting
    if hasattr(args, 'optimize_disk') and args.optimize_disk:
        min_gb = args.min_disk_space_gb if hasattr(args, 'min_disk_space_gb') else 10.0
        if not check_disk_space(min_gb=min_gb):
            logger.warning(f"Low disk space detected! You may want to free up space before continuing.")
            
    # Optimize memory if enabled
    if hasattr(args, 'optimize_memory') and args.optimize_memory:
        logger.info("Optimizing system memory for training...")
        available_gb = optimize_memory_usage(target_gb=24)  # Target 24GB for 32GB system
        
        # Clean up Python memory
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    # Log all training parameters
    logger.info(f"Training Theta AI model with the following settings:")
    logger.info(f"- Model type: {args.model_type}")
    logger.info(f"- Model name: {args.model_name}")
    logger.info(f"- Data path: {args.data_path}")
    logger.info(f"- Output dir: {args.output_dir}")
    logger.info(f"- Batch size: {args.batch_size}")
    logger.info(f"- Gradient accumulation steps: {args.gradient_accumulation_steps}")
    logger.info(f"- Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    logger.info(f"- Learning rate: {args.learning_rate}")
    logger.info(f"- Epochs: {args.epochs}")
    logger.info(f"- Patience: {args.patience}")
    logger.info(f"- Warmup proportion: {args.warmup_proportion}")
    logger.info(f"- Weight decay: {args.weight_decay}")
    
    # Log RTX 3060 optimizations
    logger.info(f"\n=== RTX 3060 12GB Optimizations ===")
    logger.info(f"- Label Smoothing: {use_label_smoothing} (value: {label_smoothing_value})")
    logger.info(f"- R-Drop Regularization: {use_rdrop} (alpha: {rdrop_alpha})")
    logger.info(f"- LLRD (Layer-wise LR Decay): {use_llrd} (factor: {llrd_factor})")
    logger.info(f"- EMA (Exponential Moving Average): {use_ema} (decay: {ema_decay})")
    logger.info(f"- Curriculum Learning: {use_curriculum}")
    logger.info(f"- Gradient Noise: {use_gradient_noise} (scale: {gradient_noise_scale})")
    logger.info(f"- CPU Offloading: {use_cpu_offload} (fraction: {cpu_offload_fraction})")
    
    # Fix #3: Auto-disable domain stratification when gradient_accumulation > 1
    # Domain stratification ensures per-batch diversity, but gradient accumulation merges batches
    if use_domain_stratified and args.gradient_accumulation_steps > 1:
        logger.warning(f"Disabling domain stratification: incompatible with gradient_accumulation_steps={args.gradient_accumulation_steps}")
        use_domain_stratified = False
    
    # Log new training enhancements (5 Recommendations)
    logger.info(f"\n=== Training Enhancements (5 Recommendations) ===")
    logger.info(f"- Quality Weighting (Rec #1): {use_quality_weighting}")
    logger.info(f"- Domain Stratified Batches (Rec #3): {use_domain_stratified}")
    logger.info(f"- Code Contrastive Loss (Rec #4): {use_code_contrastive} (weight: {code_contrastive_weight})")
    logger.info(f"- Dynamic Curriculum (Rec #5): {use_dynamic_curriculum} (warmup: {dynamic_curriculum_warmup} epochs)")
    logger.info(f"- Multi-Turn Context (Rec #2): Always enabled in dataset")
    
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"Memory: {gpu_memory:.2f} GB")
        
        # Adjust CPU offloading based on VRAM
        if gpu_memory <= 12:
            logger.info("Detected ≤12GB VRAM - enabling aggressive memory optimizations")
            use_cpu_offload = True
            cpu_offload_fraction = 0.6  # Offload 60% of optimizer states
        
    # Initialize model
    theta = ThetaModel(model_type=args.model_type, model_name=args.model_name, device=device)
    
    # Apply GPU optimizations
    theta.optimize_for_gpu()
    
    # Create dataset and dataloader
    # IMPORTANT: Pass only the tokenizer, NOT the model - dataset should not load models
    dataset = ThetaDataset(args.data_path, theta.tokenizer)
    
    # Split dataset into train and validation
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # DataLoader settings optimized for Ryzen 5-5500 (6 cores) with CPU offloading
    # Use 2 workers to leave CPU headroom for optimizer state transfers
    num_workers = int(os.environ.get('PYTORCH_DATALOADER_WORKERS', 2))
    pin_memory = device.type == 'cuda'  # Pin memory for faster GPU transfer
    
    logger.info(f"DataLoader config: num_workers={num_workers}, pin_memory={pin_memory}")
    
    # === Initialize Dynamic Curriculum Tracker (Recommendation #5) ===
    dynamic_curriculum_tracker = None
    if use_dynamic_curriculum:
        dynamic_curriculum_tracker = DynamicCurriculumTracker(
            dataset_size=len(dataset),
            warmup_epochs=dynamic_curriculum_warmup
        )
        logger.info(f"Dynamic curriculum tracker initialized")
    
    # === Initialize Code Contrastive Loss Head (Recommendation #4) ===
    code_contrastive_head = None
    if use_code_contrastive:
        # Get hidden size from model config
        hidden_size = theta.model.config.n_embd  # 1024 for GPT-2 medium
        code_contrastive_head = CodeContrastiveLoss(hidden_size=hidden_size).to(device)
        logger.info(f"Code contrastive loss head initialized (hidden_size={hidden_size})")
    
    # Build domain_indices in subset-index space (Fix #1)
    subset_domain_indices = {}
    for subset_i, orig_idx in enumerate(train_dataset.indices):
        domain = dataset.domains[orig_idx]
        subset_domain_indices.setdefault(domain, []).append(subset_i)
    logger.info(f"Built domain indices for {len(subset_domain_indices)} domains in subset space")
    
    # Setup sampler based on enabled features
    # Fix #3 & #4: Properly combine curriculum, domain stratification, and dynamic curriculum
    train_sampler = None
    curriculum_sampler = None
    
    # Create wrapper for curriculum learning if needed
    class DatasetWrapper:
        def __init__(self, subset, parent_dataset):
            self.subset = subset
            self.formatted_data = [parent_dataset.formatted_data[i] for i in subset.indices]
    
    train_wrapper = DatasetWrapper(train_dataset, dataset)
    
    # Determine which sampler to use (priority: dynamic > domain > curriculum > shuffle)
    if use_dynamic_curriculum and dynamic_curriculum_tracker:
        # Fix #1: Use DynamicWeightedSampler with proper index mapping
        logger.info("Using dynamic curriculum sampling (loss-based weights)...")
        dynamic_sampler = DynamicWeightedSampler(len(train_dataset), tracker=dynamic_curriculum_tracker)
        # Set index mapping so tracker can convert between subset and original indices
        dynamic_sampler.set_index_mapping(train_dataset.indices)
        dynamic_curriculum_tracker.sampler = dynamic_sampler  # Connect sampler to tracker
        train_sampler = dynamic_sampler
        
    elif use_domain_stratified:
        # Domain stratified sampling
        logger.info("Using domain-stratified batch sampling...")
        train_sampler = DomainStratifiedSampler(
            train_dataset,
            batch_size=args.batch_size,
            domain_indices=subset_domain_indices
        )
        
    elif use_curriculum:
        # Curriculum learning (easier examples first)
        logger.info("Setting up curriculum learning sampler (easier examples first)...")
        curriculum_start_fraction = getattr(args, 'curriculum_start_fraction', 0.7)
        curriculum_sampler = CurriculumSampler(
            train_wrapper, 
            theta.tokenizer, 
            num_epochs=args.epochs,
            start_fraction=curriculum_start_fraction
        )
        train_sampler = curriculum_sampler
    
    # Create dataloader with appropriate sampler
    if train_sampler is not None:
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0
        )
    else:
        # Default: simple shuffle
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0
        )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0
    )
    
    # Set up optimizer with LLRD (Layer-wise Learning Rate Decay) if enabled
    if use_llrd:
        logger.info("Setting up optimizer with Layer-wise Learning Rate Decay (LLRD)...")
        optimizer_grouped_parameters = get_llrd_optimizer_groups(
            theta.model,
            base_lr=args.learning_rate,
            weight_decay=args.weight_decay,
            llrd_factor=llrd_factor
        )
    else:
        # Fallback to standard optimizer groups
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in theta.model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': args.weight_decay,
                'lr': args.learning_rate
            },
            {
                'params': [p for n, p in theta.model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': args.learning_rate
            }
        ]
    
    # Create base optimizer
    base_optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    
    # === Add Code Contrastive Head parameters to optimizer (Recommendation #4) ===
    if use_code_contrastive and code_contrastive_head is not None:
        # Add code contrastive head parameters with higher learning rate
        code_head_params = {
            'params': list(code_contrastive_head.parameters()),
            'weight_decay': 0.01,
            'lr': args.learning_rate * 10  # Higher LR for the auxiliary head
        }
        base_optimizer.add_param_group(code_head_params)
        logger.info(f"Added code contrastive head to optimizer ({sum(p.numel() for p in code_contrastive_head.parameters())} params)")
    
    # Wrap with CPU offloading if enabled (critical for RTX 3060 12GB)
    if use_cpu_offload and device.type == "cuda":
        logger.info(f"Enabling CPU offloading for optimizer states ({cpu_offload_fraction*100:.0f}%)...")
        optimizer = CPUOffloadOptimizer(base_optimizer, offload_fraction=cpu_offload_fraction)
    else:
        optimizer = base_optimizer
    
    # Calculate total training steps and warmup steps
    total_steps = len(train_dataloader) * args.epochs // args.gradient_accumulation_steps
    warmup_steps = int(total_steps * args.warmup_proportion)
    
    # Learning rate scheduler - default to cosine with hard restarts for better exploration
    scheduler_type = getattr(args, 'scheduler_type', 'cosine_hard_restarts')
    num_cycles = getattr(args, 'num_cycles', 3)
    
    # Select the appropriate scheduler based on scheduler_type
    opt_for_scheduler = base_optimizer if use_cpu_offload else optimizer
    
    if scheduler_type == 'cosine_hard_restarts':
        # Cosine with hard restarts - helps escape local minima
        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            opt_for_scheduler,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            num_cycles=num_cycles
        )
        logger.info(f"Using cosine schedule with hard restarts and {num_cycles} cycles")
    elif scheduler_type == 'cosine':
        # Regular cosine schedule
        scheduler = get_cosine_schedule_with_warmup(
            opt_for_scheduler,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        logger.info("Using cosine schedule with warmup")
    elif scheduler_type == 'linear':
        # Linear warmup then linear decay
        scheduler = get_linear_schedule_with_warmup(
            opt_for_scheduler,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        logger.info("Using linear schedule with warmup")
    else:
        raise ValueError(f"Unknown scheduler_type: {scheduler_type}. Choose from: linear, cosine, cosine_hard_restarts")
    
    # Initialize label smoothing loss if enabled
    label_smoothing_loss = None  # Initialize to None to avoid NameError
    if use_label_smoothing:
        label_smoothing_loss = LabelSmoothingLoss(smoothing=label_smoothing_value)
        logger.info(f"Label smoothing loss initialized with smoothing={label_smoothing_value}")
    
    # Initialize EMA if enabled (with optional warmup)
    ema_warmup_epochs = getattr(args, 'ema_warmup_epochs', 0)
    # Fix #2: Ensure EMA warmup < total epochs (otherwise EMA never activates)
    if ema_warmup_epochs >= args.epochs:
        logger.warning(f"EMA warmup ({ema_warmup_epochs}) >= epochs ({args.epochs}). Reducing to {max(0, args.epochs - 2)}")
        ema_warmup_epochs = max(0, args.epochs - 2)
    if use_ema:
        ema = ExponentialMovingAverage(theta.model, decay=ema_decay, cpu_offload=True, warmup_epochs=ema_warmup_epochs)
        logger.info(f"EMA initialized with decay={ema_decay}, CPU offload enabled, warmup={ema_warmup_epochs} epochs")
    
    # Initialize validation metrics tracker
    validation_metrics = ValidationMetrics(theta.tokenizer)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    # Clean up temporary files if disk optimization is enabled
    if hasattr(args, 'optimize_disk') and args.optimize_disk:
        cleanup_temp_files()
    
    # Set up early stopping
    early_stopping = EarlyStopping(
        patience=args.patience,
        path=os.path.join(args.output_dir, "best_model.pt")
    )
    
    # Keep track of best validation loss
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    token_accuracies = []
    kl_losses = []
    
    # Helper functions for colorful output
    def print_header(text):
        print(f"\n{Fore.CYAN}{Back.BLACK}{Style.BRIGHT}== {text} =={Style.RESET_ALL}")
        
    def print_metric(name, value, history=None, color=Fore.GREEN, is_good=True):
        """Print metric with trend arrow based on history."""
        if history and len(history) > 1:
            prev = history[-2]
            better = value < prev if is_good else value > prev
            trend = "↓" if better else "↑"
            trend = Fore.GREEN + trend if better else Fore.RED + trend
        else:
            trend = "-"
        print(f"{color}{name}: {value:.4f} {trend}{Style.RESET_ALL}")

    # Save a copy of the script for reproducibility
    script_path = os.path.abspath(__file__)
    script_backup = os.path.join(args.output_dir, "training_script_backup.py")
    shutil.copy2(script_path, script_backup)
    logger.info(f"Saved script backup to {script_backup}")
    
    # Save training configuration
    config_backup = os.path.join(args.output_dir, "training_config.json")
    with open(config_backup, 'w') as f:
        json.dump(vars(args), f, indent=2)
    logger.info(f"Saved training configuration to {config_backup}")
    
    # Start time for the training run
    training_start_time = time.time()
    
    # Initialize email notifier with error handling
    model_name = f"Theta AI ({args.model_name})"
    try:
        notifier = get_notifier(model_name)
        notifier.start_notification(args)
    except Exception as e:
        logger.warning(f"Email notifier disabled due to error: {e}")
        class DummyNotifier:
            def start_notification(self, *a, **k): pass
            def epoch_notification(self, *a, **k): pass
            def completion_notification(self, *a, **k): pass
            def update_metrics(self, *a, **k): pass
        notifier = DummyNotifier()
    
    # Initialize scaler for mixed precision (moved outside loop for efficiency)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    # Device type for autocast (fixes CPU compatibility)
    device_type = device.type  # 'cuda' or 'cpu'
    use_amp = device_type == 'cuda'
    
    # Track additional metrics
    total_kl_loss = 0
    
    # Enhancement B: Warm-start strategy - gradually enable features
    warm_start = getattr(args, 'warm_start', False)
    
    # Training loop
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        print_header(f"Epoch {epoch+1}/{args.epochs}")
        theta.model.train()
        
        # Enhancement B: Warm-start feature enabling
        # Epochs 0-1: vanilla CE + LLRD only
        # Epochs 2-3: add label smoothing
        # Epochs 4+: add R-Drop + EMA
        if warm_start:
            epoch_use_label_smoothing = use_label_smoothing and (epoch >= 2)
            epoch_use_rdrop = use_rdrop and (epoch >= 4)
            epoch_use_ema = use_ema and (epoch >= 4)
            if epoch < 2:
                logger.info("Warm-start: using vanilla CE + LLRD only")
            elif epoch < 4:
                logger.info("Warm-start: added label smoothing")
            else:
                logger.info("Warm-start: full features (R-Drop + EMA active)")
        else:
            epoch_use_label_smoothing = use_label_smoothing
            epoch_use_rdrop = use_rdrop
            epoch_use_ema = use_ema
        
        # Fix #7: Update curriculum sampler for this epoch (check if it's actually set, not just in scope)
        if use_curriculum and curriculum_sampler is not None:
            curriculum_sampler.set_epoch(epoch)
            logger.info(f"Curriculum progress: using {len(curriculum_sampler)}/{len(train_dataset)} samples")
        
        # Fix #2: Dynamic curriculum + quality weighting compete - disable dynamic curriculum during warmup
        # When both are enabled, hard + low-quality samples can dominate training
        epoch_use_dynamic_curriculum = use_dynamic_curriculum and (epoch >= dynamic_curriculum_warmup)
        if use_dynamic_curriculum and use_quality_weighting and epoch < dynamic_curriculum_warmup:
            logger.info(f"Dynamic curriculum disabled for epoch {epoch+1} (quality weighting warmup)")
        
        # Update dynamic weighted sampler with latest loss-based weights (only after warmup)
        if epoch_use_dynamic_curriculum and dynamic_curriculum_tracker:
            if hasattr(dynamic_curriculum_tracker, 'sampler') and dynamic_curriculum_tracker.sampler is not None:
                dynamic_curriculum_tracker.sampler.update_weights()
                logger.info(f"Dynamic curriculum: updated sampling weights based on {len(dynamic_curriculum_tracker.sample_losses)} tracked losses")
        
        # Update EMA epoch for warmup tracking
        if use_ema:
            ema.set_epoch(epoch)
            if not ema.is_active():
                logger.info(f"EMA warmup: epoch {epoch+1}/{ema_warmup_epochs} (EMA inactive)")
        
        # Update dynamic curriculum tracker epoch (Recommendation #5)
        if use_dynamic_curriculum and dynamic_curriculum_tracker:
            dynamic_curriculum_tracker.set_epoch(epoch)
            if epoch >= dynamic_curriculum_warmup:
                stats = dynamic_curriculum_tracker.get_stats()
                logger.info(f"Dynamic curriculum: {stats['tracked_samples']} samples tracked, "
                           f"difficult ratio: {stats.get('difficult_ratio', 0):.2%}")
        
        total_train_loss = 0
        total_kl_loss = 0
        total_code_loss = 0  # Track code contrastive loss
        batch_count = len(train_dataloader)
        
        # Calculate noise decay for this epoch (decays over time)
        noise_scale_epoch = gradient_noise_scale / (1 + epoch * 0.1)
        
        # Training phase
        for i, batch in enumerate(train_dataloader):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Get additional batch info for new enhancements
            quality_weights = batch.get('quality_weight', None)
            has_code_labels = batch.get('has_code', None)
            sample_indices = batch.get('idx', None)
            
            if quality_weights is not None:
                quality_weights = quality_weights.to(device)
            if has_code_labels is not None:
                has_code_labels = has_code_labels.to(device)
            if sample_indices is not None:
                sample_indices = sample_indices.to(device)
            
            # Initialize variables that both paths need
            logits = None
            hidden_states = None
            per_sample_loss = None
            
            # Choose loss computation based on R-Drop setting (uses epoch-specific flag for warm-start)
            if epoch_use_rdrop:
                # R-Drop: Run forward twice with different dropout masks
                # Now returns dict with all needed values for downstream features
                rdrop_result = compute_rdrop_loss(
                    theta.model, input_ids, attention_mask, labels,
                    alpha=rdrop_alpha,
                    label_smoothing_fn=label_smoothing_loss if epoch_use_label_smoothing else None,
                    scaler=scaler,
                    device_type=device_type,
                    output_hidden_states=use_code_contrastive,
                    quality_weights=quality_weights if use_quality_weighting else None,
                    compute_per_sample=use_dynamic_curriculum  # Fix #6: compute per-sample for dynamic curriculum
                )
                loss = rdrop_result['total_loss']
                kl_loss = rdrop_result['kl_loss']
                logits = rdrop_result['logits']
                hidden_states = rdrop_result['hidden_states']
                per_sample_loss = rdrop_result['per_sample_loss']
                total_kl_loss += kl_loss.item()
                
                # Normalize loss for gradient accumulation
                loss = loss / args.gradient_accumulation_steps
                
            else:
                # Standard forward pass with mixed precision
                # Request hidden states if code contrastive is enabled
                with torch.amp.autocast(device_type, enabled=use_amp):
                    outputs = theta.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        output_hidden_states=use_code_contrastive
                    )
                    logits = outputs.logits
                    hidden_states = outputs.hidden_states[-1] if use_code_contrastive and hasattr(outputs, 'hidden_states') else None
                    
                    # === Compute per-sample losses for quality weighting & dynamic curriculum ===
                    if use_quality_weighting or use_dynamic_curriculum:
                        # Compute per-sample loss with reduction="none"
                        loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
                        token_loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1)).view(labels.size())
                        mask = (labels != -100).float()
                        per_sample_loss = (token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                    
                    # === Quality Weighting (Recommendation #1) ===
                    if use_quality_weighting and quality_weights is not None and per_sample_loss is not None:
                        # Proper per-sample weighting
                        weighted_loss = (per_sample_loss * quality_weights).mean()
                        loss = weighted_loss
                    elif epoch_use_label_smoothing:  # Use epoch-specific flag for warm-start
                        loss = label_smoothing_loss(logits, labels)
                    else:
                        loss = outputs.loss
                    
                    # Normalize loss for gradient accumulation
                    loss = loss / args.gradient_accumulation_steps
            
            # === Code Contrastive Loss (Recommendation #4) - Works with both R-Drop and non-R-Drop ===
            # Fix #5: Apply code contrastive loss less frequently (every 4 steps) to reduce competition with LM objective
            code_loss = torch.tensor(0.0, device=device)
            global_step = epoch * len(train_dataloader) + i
            apply_code_loss = (global_step % 4 == 0)  # Only every 4th step
            if use_code_contrastive and code_contrastive_head is not None and has_code_labels is not None and apply_code_loss:
                if hidden_states is not None:
                    with torch.amp.autocast(device_type, enabled=use_amp):
                        # Compute code contrastive loss (gradients flow to base model)
                        code_loss, _ = code_contrastive_head(
                            hidden_states, 
                            has_code_labels,
                            attention_mask
                        )
                        code_loss = code_loss * code_contrastive_weight / args.gradient_accumulation_steps
                        total_code_loss += code_loss.item() * args.gradient_accumulation_steps
                        
                        # Add code loss to main loss
                        loss = loss + code_loss
            
            # === Dynamic Curriculum Tracking (Recommendation #5) - Works with both paths ===
            if use_dynamic_curriculum and dynamic_curriculum_tracker and sample_indices is not None:
                with torch.no_grad():
                    if per_sample_loss is not None:
                        # Use actual per-sample losses
                        dynamic_curriculum_tracker.update_loss(sample_indices, per_sample_loss.detach())
                    elif logits is not None:
                        # Fallback: compute per-sample loss
                        loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
                        token_loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1)).view(labels.size())
                        mask = (labels != -100).float()
                        fallback_per_sample = (token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                        dynamic_curriculum_tracker.update_loss(sample_indices, fallback_per_sample.detach())
            
            # Backward pass with scaling for mixed precision
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Only update weights after accumulating enough gradients
            if (i + 1) % args.gradient_accumulation_steps == 0 or (i + 1) == len(train_dataloader):
                if scaler is not None:
                    # Gradient clipping (unscale first)
                    scaler.unscale_(base_optimizer)
                    
                    # Fix #4: Add gradient noise BEFORE clipping so clipping can bound the noise
                    if use_gradient_noise:
                        add_gradient_noise(theta.model, noise_scale=noise_scale_epoch)
                    
                    # Now clip (this will bound both original gradients AND noise)
                    torch.nn.utils.clip_grad_norm_(theta.model.parameters(), max_norm=1.0)
                    
                    # CPU offloading: move states to GPU before step
                    if use_cpu_offload:
                        optimizer.pre_step()
                    
                    # Optimizer step with gradient scaler
                    scaler.step(base_optimizer)
                    scaler.update()
                    
                    # CPU offloading: move states back to CPU after step
                    if use_cpu_offload:
                        optimizer.post_step()
                        # Fix #8: Clear CUDA cache only at end of epoch (not every 50 steps)
                        # Frequent cache clears cause noisy step timing and slower convergence
                else:
                    # Non-CUDA path - Fix #4: noise before clipping
                    if use_gradient_noise:
                        add_gradient_noise(theta.model, noise_scale=noise_scale_epoch)
                    torch.nn.utils.clip_grad_norm_(theta.model.parameters(), max_norm=1.0)
                    if use_cpu_offload:
                        optimizer.step()  # Full step with pre/post
                    else:
                        optimizer.step()
                
                optimizer.zero_grad(set_to_none=True)  # More memory efficient
                
                # Update learning rate scheduler
                scheduler.step()
                
                # Update EMA if enabled (uses epoch-specific flag for warm-start)
                if epoch_use_ema:
                    ema.update(theta.model)
            
            # Update progress
            current_loss = loss.item() * args.gradient_accumulation_steps  # Denormalize for display
            total_train_loss += current_loss
            
            # Print progress every 10 batches or at the end
            # Enhancement C: Include separate loss component info
            if (i + 1) % 10 == 0 or (i + 1) == batch_count:
                kl_info = f" KL: {total_kl_loss/(i+1):.4f}" if epoch_use_rdrop else ""
                code_info = f" Code: {total_code_loss/(i+1):.4f}" if use_code_contrastive and total_code_loss > 0 else ""
                print(f"\rProgress: {i+1}/{batch_count} batches - Loss: {current_loss:.4f}{kl_info}{code_info}", end="")
        
        # Calculate average training loss for this epoch
        avg_train_loss = total_train_loss / batch_count
        avg_code_loss = total_code_loss / batch_count if batch_count > 0 else 0
        train_losses.append(avg_train_loss)
        
        # Fix #8: Clear CUDA cache once per epoch (not during training loop)
        if use_cpu_offload and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Print epoch summary with new enhancement metrics
        print(f"\nEpoch {epoch+1} - Avg training loss: {avg_train_loss:.4f}")
        if use_code_contrastive and avg_code_loss > 0:
            logger.info(f"  Code contrastive loss: {avg_code_loss:.4f}")
        if use_dynamic_curriculum and dynamic_curriculum_tracker:
            stats = dynamic_curriculum_tracker.get_stats()
            if stats.get('tracked_samples', 0) > 0:
                logger.info(f"  Difficult samples ratio: {stats.get('difficult_ratio', 0):.1%}")
        
        # Validation phase
        # Fix #5: Periodically log non-EMA validation to detect divergence
        non_ema_val_loss = None
        if use_ema and (epoch + 1) % 3 == 0:  # Every 3 epochs, also check non-EMA
            theta.model.eval()
            with torch.no_grad():
                # Quick non-EMA check on first few batches
                temp_loss = 0
                check_batches = min(5, len(val_dataloader))
                for i, batch in enumerate(val_dataloader):
                    if i >= check_batches:
                        break
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    with torch.amp.autocast(device_type, enabled=use_amp):
                        outputs = theta.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                        temp_loss += outputs.loss.item()
                non_ema_val_loss = temp_loss / check_batches
                logger.info(f"Non-EMA validation loss (sampled): {non_ema_val_loss:.4f}")
        
        # Apply EMA weights for validation if enabled
        if use_ema:
            ema.apply_shadow(theta.model)
            logger.info("Using EMA weights for validation")
        
        theta.model.eval()
        total_val_loss = 0
        batch_count_val = len(val_dataloader)
        
        # Reset validation metrics
        validation_metrics.reset()
        
        print_header("Validation")
        
        with torch.no_grad():
            for i, batch in enumerate(val_dataloader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                with torch.amp.autocast(device_type, enabled=use_amp):
                    outputs = theta.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    val_loss = outputs.loss
                    
                total_val_loss += val_loss.item()
                
                # Update validation metrics (token accuracy, etc.)
                validation_metrics.update(
                    val_loss.item(),
                    logits=outputs.logits,
                    labels=labels,
                    epoch=epoch
                )
                
                # Print progress every 10 batches or at the end
                if (i + 1) % 10 == 0 or (i + 1) == batch_count_val:
                    print(f"\rValidating: {i+1}/{batch_count_val} batches", end="")
        
        # Restore original weights after validation if using EMA
        if use_ema:
            ema.restore(theta.model)
        
        # Calculate average validation loss
        avg_val_loss = total_val_loss / batch_count_val
        val_losses.append(avg_val_loss)
        
        # Get extended validation metrics summary
        metrics_summary = validation_metrics.get_summary()
        avg_token_accuracy = metrics_summary.get('token_accuracy_mean', 0.0)
        token_accuracies.append(avg_token_accuracy)
        
        # Track KL loss if using R-Drop
        if use_rdrop:
            avg_kl_loss = total_kl_loss / batch_count
            kl_losses.append(avg_kl_loss)
        
        print(f"\nEpoch {epoch+1} - Validation loss: {avg_val_loss:.4f}")
        
        # Calculate perplexity (with safety guard against NaN/inf)
        # Fix #9: Note that perplexity is DIAGNOSTIC only when using label smoothing + R-Drop
        # With these regularizers, perplexity ≠ quality. Prioritize token accuracy + validation CE.
        def safe_perplexity(loss):
            if not np.isfinite(loss):
                return float('inf')
            return math.exp(min(loss, 100))  # Clamp to avoid overflow
        
        train_perplexity = safe_perplexity(avg_train_loss)
        val_perplexity = safe_perplexity(avg_val_loss)
        
        if use_label_smoothing or use_rdrop:
            logger.debug("Note: Perplexity is diagnostic only (label smoothing/R-Drop affect loss scale)")
        
        # Print metrics summary (with correct history for trend arrows)
        print_header("Metrics Summary")
        print_metric("Training Loss", avg_train_loss, train_losses, Fore.GREEN, is_good=True)
        print_metric("Validation Loss", avg_val_loss, val_losses, Fore.YELLOW, is_good=True)
        print_metric("Training Perplexity", train_perplexity, None, Fore.CYAN, is_good=True)
        print_metric("Validation Perplexity", val_perplexity, None, Fore.MAGENTA, is_good=True)
        
        # Print new metrics from RTX 3060 optimizations
        if avg_token_accuracy > 0:
            print(f"{Fore.WHITE}Token Accuracy: {avg_token_accuracy*100:.2f}%{Style.RESET_ALL}")
        if use_rdrop and len(kl_losses) > 0:
            print(f"{Fore.WHITE}R-Drop KL Loss: {kl_losses[-1]:.4f}{Style.RESET_ALL}")
        if use_ema:
            print(f"{Fore.WHITE}EMA: Active (decay={ema_decay}){Style.RESET_ALL}")
        if use_curriculum:
            curriculum_progress = min(1.0, (epoch + 1) / max(1, args.epochs // 2))
            print(f"{Fore.WHITE}Curriculum Progress: {curriculum_progress*100:.0f}%{Style.RESET_ALL}")
        
        # Run domain-specific evaluation every 5 epochs or on last epoch
        run_domain_eval = getattr(args, 'domain_eval_frequency', 5)
        if (epoch + 1) % run_domain_eval == 0 or (epoch + 1) == args.epochs:
            print_header("Domain-Specific Evaluation")
            try:
                # Use EMA weights for evaluation if available
                if use_ema:
                    ema.apply_shadow(theta.model)
                
                domain_results = validation_metrics.run_full_domain_evaluation(theta.model, device)
                
                # Print domain results
                for domain_cat, metrics in domain_results.items():
                    if metrics:
                        print(f"{Fore.CYAN}{domain_cat.upper()}: "
                              f"coherence={metrics.get('coherence', 0):.2f}, "
                              f"domain_score={metrics.get('domain_score', 0):.2f}, "
                              f"length={metrics.get('avg_length', 0):.0f}w{Style.RESET_ALL}")
                
                # Restore original weights
                if use_ema:
                    ema.restore(theta.model)
                    
            except Exception as e:
                logger.warning(f"Domain evaluation failed: {e}")
        
        # Print improvement percentage if applicable
        if len(val_losses) > 1:
            prev_loss = val_losses[-2]
            improvement = (prev_loss - avg_val_loss) / prev_loss * 100
            direction = "improved" if improvement > 0 else "worsened"
            color = Fore.GREEN if improvement > 0 else Fore.RED
            print(f"{color}Validation loss {direction} by {abs(improvement):.2f}%{Style.RESET_ALL}")
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        print(f"{Fore.BLUE}Epoch completed in {epoch_time:.2f} seconds{Style.RESET_ALL}")
        
        # Update metrics for 10-min status updates
        current_metrics = {
            'current_epoch': epoch + 1,
            'total_epochs': args.epochs,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'perplexity': val_perplexity,
            'token_accuracy': avg_token_accuracy,
            'current_lr': scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else args.learning_rate,
            'ema_active': use_ema and ema.is_active() if use_ema else False,
        }
        
        # Add optional metrics if available
        if use_rdrop and len(kl_losses) > 0:
            current_metrics['kl_loss'] = kl_losses[-1]
        if use_code_contrastive and avg_code_loss > 0:
            current_metrics['code_loss'] = avg_code_loss
        if use_curriculum:
            current_metrics['curriculum_progress'] = min(1.0, (epoch + 1) / max(1, args.epochs // 2))
        if use_dynamic_curriculum and dynamic_curriculum_tracker:
            stats = dynamic_curriculum_tracker.get_stats()
            current_metrics['difficult_ratio'] = stats.get('difficult_ratio', 0)
        
        notifier.update_metrics(current_metrics)
        
        # Send epoch notification email
        notifier.epoch_notification(
            epoch=epoch+1,
            total_epochs=args.epochs,
            train_loss=avg_train_loss,
            val_loss=avg_val_loss,
            perplexity=val_perplexity,
            duration=epoch_time
        )
        
        # Save only the last N epochs and delete older checkpoints to save space
        keep_last_n_epochs = args.keep_last_n_epochs if hasattr(args, 'keep_last_n_epochs') else 3
        
        checkpoint_dir = os.path.join(
            args.output_dir, 
            f"theta_checkpoint_epoch_{epoch+1}"
        )
        
        # Save current checkpoint
        theta.save(checkpoint_dir)
        print(f"Saved checkpoint to {checkpoint_dir}")
        
        # Use the cleanup utility to manage checkpoints
        if hasattr(args, 'optimize_disk') and args.optimize_disk:
            cleanup_old_checkpoints(args.output_dir, keep_last_n=keep_last_n_epochs)
            
            # Periodically clean temp files during long training runs
            if epoch > 0 and epoch % 10 == 0:  # Every 10 epochs
                cleanup_temp_files()
                
            # Release memory after checkpointing
            if hasattr(args, 'optimize_memory') and args.optimize_memory:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Check for early stopping (Fix #5: save EMA weights when EMA is enabled)
        ema_state_dict = None
        if use_ema:
            # Get EMA weights to save instead of current model weights
            ema.apply_shadow(theta.model)
            ema_state_dict = {k: v.clone() for k, v in theta.model.state_dict().items()}
            ema.restore(theta.model)
        early_stopping(avg_val_loss, theta.model, state_dict_override=ema_state_dict)
        if early_stopping.early_stop:
            print(f"{Fore.YELLOW}Early stopping triggered! No improvement for {args.patience} epochs.{Style.RESET_ALL}")
            break
    
    # Load best model
    best_model_path = os.path.join(args.output_dir, "best_model.pt")
    if os.path.exists(best_model_path):
        theta.model.load_state_dict(torch.load(best_model_path))
        logger.info("Loaded best model based on validation loss")
        
    # Final memory cleanup
    if hasattr(args, 'optimize_memory') and args.optimize_memory:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    # Check final disk space
    if hasattr(args, 'optimize_disk') and args.optimize_disk:
        check_disk_space(min_gb=5.0)  # Lower threshold for final check
    
    # Save final model (use EMA weights if enabled for better generalization)
    final_model_dir = os.path.join(args.output_dir, "theta_final")
    
    if use_ema:
        # Fix #9: Only save EMA weights once (theta_final = EMA weights)
        logger.info("Applying EMA weights to final model for better generalization...")
        ema.apply_shadow(theta.model)
        theta.save(final_model_dir)
        # Keep EMA applied for domain evaluation below, restore after
    else:
        theta.save(final_model_dir)
    
    logger.info(f"Training complete! Final model saved to {final_model_dir}")
    
    # Calculate total training time
    total_training_time = time.time() - training_start_time
    hours, remainder = divmod(total_training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info(f"Total training time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    
    # Find best epoch and validation loss
    best_val_loss = min(val_losses)
    best_epoch = val_losses.index(best_val_loss) + 1
    
    # Keep only the best checkpoint if requested
    if hasattr(args, 'keep_best_only') and args.keep_best_only:
        logger.info(f"Keeping only the best checkpoint from epoch {best_epoch}")
        keep_only_best_checkpoint(args.output_dir, best_epoch)
    
    # Send completion notification
    notifier.completion_notification(
        total_epochs=len(train_losses),
        best_val_loss=best_val_loss,
        best_epoch=best_epoch,
        total_duration=total_training_time
    )
    
    # Print final performance summary
    print_header("Final Performance Summary")
    print(f"{Fore.CYAN}Initial training loss: {train_losses[0]:.4f}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Final training loss: {train_losses[-1]:.4f}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Best validation loss: {min(val_losses):.4f}{Style.RESET_ALL}")
    
    # Calculate overall improvement
    if len(train_losses) > 1:
        total_improvement = (train_losses[0] - train_losses[-1]) / train_losses[0] * 100
        print(f"{Fore.GREEN}Overall loss improvement: {total_improvement:.2f}%{Style.RESET_ALL}")
    
    # Run final comprehensive domain evaluation
    # Note: If EMA is enabled, model already has EMA weights applied from save above
    print_header("Final Domain Evaluation")
    try:
        final_domain_results = validation_metrics.run_full_domain_evaluation(theta.model, device)
    except Exception as e:
        logger.warning(f"Final domain evaluation failed: {e}")
        final_domain_results = {}
    
    # Restore original weights after everything is done (if EMA was applied)
    if use_ema:
        ema.restore(theta.model)
    
    # Get domain-specific metrics summary
    domain_summary = validation_metrics.get_domain_summary()
    
    # Save learning curves with extended metrics
    loss_data = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'token_accuracies': token_accuracies,
        'kl_losses': kl_losses if use_rdrop else [],
        'domain_metrics': {
            'final_evaluation': final_domain_results,
            'per_category_summary': domain_summary
        },
        'rtx3060_optimizations': {
            'label_smoothing': label_smoothing_value if use_label_smoothing else None,
            'rdrop_alpha': rdrop_alpha if use_rdrop else None,
            'llrd_factor': llrd_factor if use_llrd else None,
            'ema_decay': ema_decay if use_ema else None,
            'curriculum_learning': use_curriculum,
            'gradient_noise_scale': gradient_noise_scale if use_gradient_noise else None,
            'cpu_offload_fraction': cpu_offload_fraction if use_cpu_offload else None
        }
    }
    with open(os.path.join(args.output_dir, 'loss_history.json'), 'w') as f:
        json.dump(loss_data, f, indent=2)
    
    # Print RTX 3060 optimization summary
    print_header("RTX 3060 Optimization Summary")
    print(f"{Fore.CYAN}Label Smoothing: {label_smoothing_value if use_label_smoothing else 'Disabled'}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}R-Drop: {'Enabled (α=' + str(rdrop_alpha) + ')' if use_rdrop else 'Disabled'}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}LLRD: {'Enabled (factor=' + str(llrd_factor) + ')' if use_llrd else 'Disabled'}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}EMA: {'Enabled (decay=' + str(ema_decay) + ')' if use_ema else 'Disabled'}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Curriculum Learning: {'Enabled' if use_curriculum else 'Disabled'}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Gradient Noise: {'Enabled (scale=' + str(gradient_noise_scale) + ')' if use_gradient_noise else 'Disabled'}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}CPU Offloading: {'Enabled (' + str(int(cpu_offload_fraction*100)) + '%)' if use_cpu_offload else 'Disabled'}{Style.RESET_ALL}")
    
    return final_model_dir

def main(passed_args=None):
    # Only parse arguments if not passed in
    if passed_args is None:
        parser = argparse.ArgumentParser(description="Train the Theta AI model with enhanced features")
        
        parser.add_argument("--model_type", type=str, default="gpt2", help="Model type (gpt2, bert-qa)")
        parser.add_argument("--model_name", type=str, default="gpt2", help="Specific model name/version")
        parser.add_argument("--data_path", type=str, required=True, help="Path to the processed data")
        parser.add_argument("--output_dir", type=str, required=True, help="Directory to save model checkpoints")
        parser.add_argument("--batch_size", type=int, default=2, help="Training batch size")
        parser.add_argument("--gradient_accumulation_steps", type=int, default=4, 
                          help="Number of steps to accumulate gradients before performing a backward/update pass")
        parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
        parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
        parser.add_argument("--patience", type=int, default=3, help="Patience for early stopping")
        parser.add_argument("--warmup_proportion", type=float, default=0.1, 
                          help="Proportion of training steps for learning rate warmup")
        parser.add_argument("--weight_decay", type=float, default=0.01, 
                          help="Weight decay for regularization")
        parser.add_argument("--scheduler_type", type=str, default="cosine", 
                          choices=["linear", "cosine", "cosine_hard_restarts"],
                          help="Type of learning rate scheduler")
        parser.add_argument("--num_cycles", type=int, default=3,
                          help="Number of cycles for cosine schedule with hard restarts")
        parser.add_argument("--keep_last_n_epochs", type=int, default=3,
                          help="Number of most recent epoch checkpoints to keep")
        parser.add_argument("--optimize_disk", action="store_true", default=True,
                          help="Enable disk space optimization")
        parser.add_argument("--optimize_memory", action="store_true", default=True,
                          help="Enable memory optimization for 32GB systems")
        parser.add_argument("--min_disk_space_gb", type=float, default=10.0,
                          help="Minimum required disk space in GB")
        parser.add_argument("--keep_best_only", action="store_true", default=True,
                          help="Keep only the best checkpoint at the end of training")
        parser.add_argument("--no-color", action="store_true", help="Disable colored output")
        parser.add_argument("--log_file", type=str, help="Path to log file")
        
        # Domain evaluation arguments
        parser.add_argument("--domain_eval_frequency", type=int, default=5,
                          help="Run domain-specific evaluation every N epochs (default: 5)")
        
        # RTX 3060 12GB Optimization arguments
        parser.add_argument("--label_smoothing", type=float, default=0.1,
                          help="Label smoothing value (0.0 to disable, default: 0.1)")
        parser.add_argument("--use_rdrop", action="store_true", default=True,
                          help="Enable R-Drop regularization")
        parser.add_argument("--no_rdrop", action="store_true",
                          help="Disable R-Drop regularization")
        parser.add_argument("--rdrop_alpha", type=float, default=0.1,
                          help="R-Drop KL divergence weight (default: 0.1)")
        parser.add_argument("--use_llrd", action="store_true", default=True,
                          help="Enable Layer-wise Learning Rate Decay")
        parser.add_argument("--no_llrd", action="store_true",
                          help="Disable Layer-wise Learning Rate Decay")
        parser.add_argument("--llrd_factor", type=float, default=0.95,
                          help="LLRD decay factor per layer (default: 0.95)")
        parser.add_argument("--use_ema", action="store_true", default=True,
                          help="Enable Exponential Moving Average")
        parser.add_argument("--no_ema", action="store_true",
                          help="Disable Exponential Moving Average")
        parser.add_argument("--ema_decay", type=float, default=0.999,
                          help="EMA decay rate (default: 0.999)")
        parser.add_argument("--use_curriculum", action="store_true", default=True,
                          help="Enable curriculum learning")
        parser.add_argument("--no_curriculum", action="store_true",
                          help="Disable curriculum learning")
        parser.add_argument("--use_gradient_noise", action="store_true", default=True,
                          help="Enable gradient noise injection")
        parser.add_argument("--no_gradient_noise", action="store_true",
                          help="Disable gradient noise injection")
        parser.add_argument("--gradient_noise_scale", type=float, default=0.01,
                          help="Gradient noise scale (default: 0.01)")
        parser.add_argument("--use_cpu_offload", action="store_true", default=True,
                          help="Enable CPU offloading for optimizer states")
        parser.add_argument("--no_cpu_offload", action="store_true",
                          help="Disable CPU offloading")
        parser.add_argument("--cpu_offload_fraction", type=float, default=0.5,
                          help="Fraction of optimizer states to offload to CPU (default: 0.5)")
        
        # New training enhancements (5 Recommendations) - Fix #6: defaults match batch file
        parser.add_argument("--ema_warmup_epochs", type=int, default=3,
                          help="Number of epochs to disable EMA during warmup (default: 3)")
        parser.add_argument("--curriculum_start_fraction", type=float, default=0.7,
                          help="Fraction of data to start with in curriculum learning (default: 0.7)")
        parser.add_argument("--use_quality_weighting", action="store_true", default=True,
                          help="Enable quality-based sample weighting (Rec #1)")
        parser.add_argument("--no_quality_weighting", action="store_true",
                          help="Disable quality-based sample weighting")
        parser.add_argument("--use_domain_stratified", action="store_true", default=True,
                          help="Enable domain-stratified batch sampling (Rec #3)")
        parser.add_argument("--no_domain_stratified", action="store_true",
                          help="Disable domain-stratified batch sampling")
        parser.add_argument("--use_code_contrastive", action="store_true", default=True,
                          help="Enable code contrastive loss (Rec #4)")
        parser.add_argument("--no_code_contrastive", action="store_true",
                          help="Disable code contrastive loss")
        parser.add_argument("--code_contrastive_weight", type=float, default=0.05,
                          help="Weight for code contrastive loss (default: 0.05)")
        parser.add_argument("--use_dynamic_curriculum", action="store_true", default=True,
                          help="Enable dynamic curriculum based on loss (Rec #5)")
        parser.add_argument("--no_dynamic_curriculum", action="store_true",
                          help="Disable dynamic curriculum")
        parser.add_argument("--dynamic_curriculum_warmup", type=int, default=3,
                          help="Warmup epochs before dynamic curriculum activates (default: 3)")
        
        # Enhancement A: Ablation mode for debugging
        parser.add_argument("--ablation_mode", action="store_true",
                          help="Ablation mode: disables R-Drop, gradient noise, dynamic curriculum, "
                               "and code contrastive to test baseline training")
        
        # Enhancement B: Warm-start strategy
        parser.add_argument("--warm_start", action="store_true",
                          help="Enable warm-start: vanilla CE epochs 0-1, add label smoothing epochs 2-3, "
                               "add R-Drop + EMA epochs 4+")
        
        args = parser.parse_args()
        
        # Handle --no_* flags
        if args.no_rdrop:
            args.use_rdrop = False
        if args.no_llrd:
            args.use_llrd = False
        if args.no_ema:
            args.use_ema = False
        if args.no_curriculum:
            args.use_curriculum = False
        if args.no_gradient_noise:
            args.use_gradient_noise = False
        if args.no_cpu_offload:
            args.use_cpu_offload = False
        if args.no_quality_weighting:
            args.use_quality_weighting = False
        if args.no_domain_stratified:
            args.use_domain_stratified = False
        if args.no_code_contrastive:
            args.use_code_contrastive = False
        if args.no_dynamic_curriculum:
            args.use_dynamic_curriculum = False
        
        # Enhancement A: Ablation mode - disable experimental features for baseline testing
        if getattr(args, 'ablation_mode', False):
            logger.info("ABLATION MODE: Disabling R-Drop, gradient noise, dynamic curriculum, code contrastive")
            args.use_rdrop = False
            args.use_gradient_noise = False
            args.use_dynamic_curriculum = False
            args.use_code_contrastive = False
    else:
        args = passed_args
    
    # Set up file logging if specified
    if hasattr(args, 'log_file') and args.log_file:
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
    
    # Disable colors if requested
    if hasattr(args, 'no_color') and args.no_color:
        init(autoreset=True, strip=True)
    else:
        init(autoreset=True)
        
    # Print colorful banner
    print(f"{Fore.CYAN}{Style.BRIGHT}╔══════════════════════════════════════════════════════════╗{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{Style.BRIGHT}║     THETA AI ENHANCED TRAINING SYSTEM v3.0               ║{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{Style.BRIGHT}║     RTX 3060 12GB Optimized Edition                      ║{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{Style.BRIGHT}╠══════════════════════════════════════════════════════════╣{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{Style.BRIGHT}║  Features: LLRD | R-Drop | EMA | Label Smoothing         ║{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{Style.BRIGHT}║            Curriculum Learning | CPU Offloading          ║{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{Style.BRIGHT}╚══════════════════════════════════════════════════════════╝{Style.RESET_ALL}")
    
    try:
        train(args)
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        print(f"\n{Fore.RED}{'='*60}")
        print(f"TRAINING FAILED")
        print(f"{'='*60}")
        print(f"Error: {e}")
        print(f"See log file for full traceback{Style.RESET_ALL}\n")
        raise  # Re-raise so batch file can detect the error

if __name__ == "__main__":
    main()
