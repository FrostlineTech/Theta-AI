"""
Cross-validation utilities for Theta AI training.
Implements k-fold cross-validation for more robust model evaluation.
"""

import os
import json
import torch
import numpy as np
from sklearn.model_selection import KFold
from pathlib import Path
import logging
from typing import List, Dict, Any, Tuple, Optional
import time
import shutil

logger = logging.getLogger(__name__)

class KFoldTrainer:
    """Class for k-fold cross-validation training."""
    
    def __init__(
        self, 
        dataset, 
        train_fn, 
        n_splits=5, 
        random_state=42,
        output_dir=None,
        fold_prefix="fold_"
    ):
        """
        Initialize the k-fold trainer.
        
        Args:
            dataset: Dataset to split into folds
            train_fn: Training function to call for each fold
            n_splits: Number of folds for cross-validation
            random_state: Random seed for reproducibility
            output_dir: Directory to save fold models and results
            fold_prefix: Prefix for fold directories
        """
        self.dataset = dataset
        self.train_fn = train_fn
        self.n_splits = n_splits
        self.random_state = random_state
        self.output_dir = output_dir
        self.fold_prefix = fold_prefix
        self.kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
    def run(self, args, fold_indices=None):
        """
        Run k-fold cross-validation.
        
        Args:
            args: Arguments to pass to the training function
            fold_indices: Optional list of fold indices to run (for partial runs)
            
        Returns:
            Dictionary with fold results and overall statistics
        """
        if not self.output_dir:
            self.output_dir = args.output_dir
            
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Generate fold splits
        indices = list(range(len(self.dataset)))
        fold_splits = list(self.kfold.split(indices))
        
        # Track results across folds
        fold_results = []
        best_val_losses = []
        best_model_paths = []
        
        # Determine which folds to run
        fold_indices = fold_indices or range(self.n_splits)
        
        # Save fold split information for future reference
        fold_info = {
            "n_splits": self.n_splits,
            "random_state": self.random_state,
            "fold_splits": [
                {"train_indices": train.tolist(), "val_indices": val.tolist()} 
                for train, val in fold_splits
            ]
        }
        
        with open(os.path.join(self.output_dir, "fold_info.json"), "w") as f:
            json.dump(fold_info, f, indent=2)
        
        # Run training for each fold
        overall_start_time = time.time()
        
        for fold_idx in fold_indices:
            fold_start_time = time.time()
            
            train_indices, val_indices = fold_splits[fold_idx]
            logger.info(f"Starting fold {fold_idx+1}/{self.n_splits}")
            logger.info(f"Train size: {len(train_indices)}, Validation size: {len(val_indices)}")
            
            # Create fold-specific output directory
            fold_dir = os.path.join(self.output_dir, f"{self.fold_prefix}{fold_idx+1}")
            os.makedirs(fold_dir, exist_ok=True)
            
            # Copy configuration to fold directory
            if hasattr(args, "config_path") and args.config_path:
                shutil.copy2(args.config_path, fold_dir)
            
            # Prepare fold-specific arguments
            fold_args = self._prepare_fold_args(args, fold_dir, fold_idx, train_indices, val_indices)
            
            # Run training for this fold
            fold_result = self.train_fn(fold_args)
            
            # Collect fold results
            fold_results.append(fold_result)
            
            # Extract validation loss if available
            val_loss_file = os.path.join(fold_dir, "loss_history.json")
            if os.path.exists(val_loss_file):
                with open(val_loss_file, "r") as f:
                    loss_data = json.load(f)
                    val_losses = loss_data.get("val_loss", [])
                    best_val_loss = min(val_losses) if val_losses else None
                    best_val_losses.append(best_val_loss)
            
            # Track best model path
            best_model_path = os.path.join(fold_dir, "best_model.pt")
            if os.path.exists(best_model_path):
                best_model_paths.append(best_model_path)
            
            fold_end_time = time.time()
            fold_duration = fold_end_time - fold_start_time
            logger.info(f"Completed fold {fold_idx+1}/{self.n_splits} in {fold_duration:.2f} seconds")
        
        overall_end_time = time.time()
        overall_duration = overall_end_time - overall_start_time
        logger.info(f"Completed {len(fold_indices)} folds in {overall_duration:.2f} seconds")
        
        # Calculate cross-validation statistics
        cv_stats = self._calculate_cv_statistics(best_val_losses)
        
        # Find best fold model
        if best_val_losses:
            best_fold_idx = np.argmin(best_val_losses)
            best_fold = fold_indices[best_fold_idx]
            best_model_path = best_model_paths[best_fold_idx] if best_model_paths else None
            logger.info(f"Best model from fold {best_fold+1} with validation loss: {best_val_losses[best_fold_idx]:.4f}")
        else:
            best_fold = None
            best_model_path = None
        
        # Prepare final results
        final_results = {
            "fold_results": fold_results,
            "best_val_losses": best_val_losses,
            "cv_statistics": cv_stats,
            "best_fold": best_fold,
            "best_model_path": best_model_path,
            "total_duration": overall_duration
        }
        
        # Save final results
        with open(os.path.join(self.output_dir, "cv_results.json"), "w") as f:
            # Convert any non-serializable values to strings
            serializable_results = self._make_serializable(final_results)
            json.dump(serializable_results, f, indent=2)
        
        return final_results
    
    def _prepare_fold_args(self, args, fold_dir, fold_idx, train_indices, val_indices):
        """Prepare arguments for training function with fold-specific settings."""
        # Create a copy of the arguments to modify for this fold
        fold_args = self._copy_args(args)
        
        # Update output directory to fold-specific directory
        fold_args.output_dir = fold_dir
        
        # Add fold-specific information
        fold_args.fold_idx = fold_idx
        fold_args.train_indices = train_indices
        fold_args.val_indices = val_indices
        
        # Add total number of folds
        fold_args.n_splits = self.n_splits
        
        return fold_args
    
    def _copy_args(self, args):
        """Create a copy of the arguments."""
        from copy import deepcopy
        if hasattr(args, "__dict__"):
            # If args is an argparse.Namespace
            new_args = type(args)()
            new_args.__dict__.update(deepcopy(args.__dict__))
            return new_args
        else:
            # For other types, try a direct copy
            return deepcopy(args)
    
    def _calculate_cv_statistics(self, val_losses):
        """Calculate cross-validation statistics."""
        if not val_losses:
            return {
                "mean": None,
                "std": None,
                "min": None,
                "max": None
            }
        
        val_losses = np.array(val_losses)
        return {
            "mean": float(np.mean(val_losses)),
            "std": float(np.std(val_losses)),
            "min": float(np.min(val_losses)),
            "max": float(np.max(val_losses))
        }
    
    def _make_serializable(self, obj):
        """Convert non-serializable objects to serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(x) for x in obj]
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            try:
                json.dumps(obj)
                return obj
            except (TypeError, OverflowError):
                return str(obj)

def run_kfold_training(dataset, train_fn, args, n_splits=5):
    """
    Run k-fold cross-validation training.
    
    Args:
        dataset: Dataset to split into folds
        train_fn: Training function to call for each fold
        args: Arguments to pass to the training function
        n_splits: Number of folds
        
    Returns:
        Cross-validation results
    """
    # Create output directory for cross-validation
    cv_output_dir = os.path.join(args.output_dir, "cv")
    os.makedirs(cv_output_dir, exist_ok=True)
    
    # Initialize k-fold trainer
    kfold_trainer = KFoldTrainer(
        dataset=dataset,
        train_fn=train_fn,
        n_splits=n_splits,
        output_dir=cv_output_dir
    )
    
    # Run cross-validation
    results = kfold_trainer.run(args)
    
    # Copy best model to main output directory
    if results["best_model_path"] and os.path.exists(results["best_model_path"]):
        best_model_dest = os.path.join(args.output_dir, "best_cv_model.pt")
        shutil.copy2(results["best_model_path"], best_model_dest)
        logger.info(f"Best cross-validation model saved to {best_model_dest}")
    
    return results
