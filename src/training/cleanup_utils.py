"""
Disk space and memory management utilities for Theta AI training.

This module provides functions to clean up temporary files, manage disk space,
and optimize memory usage during training.
"""

import os
import shutil
import glob
import logging
import psutil
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def cleanup_old_checkpoints(output_dir, keep_last_n=3):
    """
    Clean up old checkpoints to save disk space.
    
    Args:
        output_dir: Directory containing checkpoints
        keep_last_n: Number of most recent checkpoints to keep
    """
    try:
        output_path = Path(output_dir)
        if not output_path.exists():
            return
            
        # Find all checkpoint directories
        checkpoint_dirs = [d for d in output_path.glob("theta_checkpoint_epoch_*") if d.is_dir()]
        
        # Sort by epoch number
        checkpoint_dirs.sort(key=lambda d: int(str(d).split("_")[-1]))
        
        # Keep only the last n checkpoints
        if len(checkpoint_dirs) > keep_last_n:
            dirs_to_remove = checkpoint_dirs[:-keep_last_n]
            for dir_path in dirs_to_remove:
                logger.info(f"Removing old checkpoint: {dir_path}")
                shutil.rmtree(dir_path)
                
        return len(dirs_to_remove) if len(checkpoint_dirs) > keep_last_n else 0
    except Exception as e:
        logger.error(f"Error cleaning up checkpoints: {e}")
        return 0


def keep_only_best_checkpoint(output_dir, best_epoch):
    """
    Keep only the checkpoint for the best epoch and delete all others.
    
    Args:
        output_dir: Directory containing checkpoints
        best_epoch: The epoch number corresponding to the best validation loss
        
    Returns:
        int: Number of checkpoints removed
    """
    try:
        output_path = Path(output_dir)
        if not output_path.exists():
            return 0
            
        # Find all checkpoint directories
        checkpoint_dirs = [d for d in output_path.glob("theta_checkpoint_epoch_*") if d.is_dir()]
        
        # Find the best checkpoint directory
        best_checkpoint = None
        for d in checkpoint_dirs:
            epoch_num = int(str(d).split("_")[-1])
            if epoch_num == best_epoch:
                best_checkpoint = d
                break
                
        if best_checkpoint is None:
            logger.warning(f"Best checkpoint for epoch {best_epoch} not found. Not cleaning up any checkpoints.")
            return 0
            
        # Remove all checkpoints except the best one
        removed_count = 0
        for dir_path in checkpoint_dirs:
            if dir_path != best_checkpoint:
                logger.info(f"Removing checkpoint: {dir_path}")
                shutil.rmtree(dir_path)
                removed_count += 1
                
        logger.info(f"Kept only the best checkpoint: {best_checkpoint}")
        return removed_count
    except Exception as e:
        logger.error(f"Error cleaning up checkpoints: {e}")
        return 0

def cleanup_temp_files(base_dir=None, extensions=None):
    """
    Clean up temporary files to save disk space.
    
    Args:
        base_dir: Base directory to clean (default: project root)
        extensions: List of file extensions to clean (default: common temp files)
    """
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent.parent.parent
    else:
        base_dir = Path(base_dir)
        
    if extensions is None:
        extensions = [".tmp", ".temp", ".bak", ".pyc", ".pyo", "__pycache__"]
        
    try:
        # Clean temporary files
        count = 0
        for ext in extensions:
            if ext.startswith("__"):
                # Handle directory patterns
                for path in base_dir.glob(f"**/{ext}"):
                    if path.is_dir():
                        shutil.rmtree(path)
                        count += 1
            else:
                # Handle file patterns
                for path in base_dir.glob(f"**/*{ext}"):
                    if path.is_file():
                        os.remove(path)
                        count += 1
                        
        logger.info(f"Cleaned up {count} temporary files/directories")
        return count
    except Exception as e:
        logger.error(f"Error cleaning up temp files: {e}")
        return 0

def optimize_memory_usage(target_gb=24):
    """
    Optimize memory usage for training.
    
    Args:
        target_gb: Target memory usage in GB for training
    """
    try:
        # Calculate system memory stats
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024 ** 3)
        available_gb = mem.available / (1024 ** 3)
        
        logger.info(f"System memory: {total_gb:.1f}GB total, {available_gb:.1f}GB available")
        
        if available_gb < target_gb * 0.8:  # If less than 80% of target is available
            # Clean up memory
            logger.info("Pre-optimizing system memory...")
            
            # Suggest clearing disk cache on Windows
            if os.name == 'nt':
                logger.info("Consider running 'EmptyStandbyList' utility to clear Windows disk cache")
                
            # Clean Python memory
            import gc
            gc.collect()
            
            # Check memory again
            mem = psutil.virtual_memory()
            available_gb = mem.available / (1024 ** 3)
            logger.info(f"Available memory after optimization: {available_gb:.1f}GB")
            
        return available_gb
    except Exception as e:
        logger.error(f"Error optimizing memory usage: {e}")
        return 0

def check_disk_space(min_gb=10):
    """
    Check available disk space.
    
    Args:
        min_gb: Minimum required disk space in GB
    
    Returns:
        bool: True if enough disk space is available
    """
    try:
        # Get disk usage statistics
        disk_usage = shutil.disk_usage(Path(__file__).resolve().parent.parent.parent)
        free_gb = disk_usage.free / (1024 ** 3)
        total_gb = disk_usage.total / (1024 ** 3)
        
        logger.info(f"Disk space: {free_gb:.1f}GB free of {total_gb:.1f}GB total")
        
        if free_gb < min_gb:
            logger.warning(f"Low disk space: {free_gb:.1f}GB available, minimum recommended is {min_gb}GB")
            return False
        return True
    except Exception as e:
        logger.error(f"Error checking disk space: {e}")
        return True  # Assume enough space if check fails

if __name__ == "__main__":
    # Test utilities
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean up disk space and optimize memory")
    parser.add_argument("--clean_temp", action="store_true", help="Clean temporary files")
    parser.add_argument("--check_disk", action="store_true", help="Check disk space")
    parser.add_argument("--optimize_mem", action="store_true", help="Optimize memory usage")
    parser.add_argument("--clean_checkpoints", action="store_true", help="Clean old checkpoints")
    parser.add_argument("--keep_best_only", action="store_true", help="Keep only the best checkpoint")
    parser.add_argument("--output_dir", type=str, help="Model output directory for checkpoint cleaning")
    parser.add_argument("--keep", type=int, default=3, help="Number of checkpoints to keep")
    parser.add_argument("--best_epoch", type=int, help="Best epoch number (for keep_best_only)")
    
    args = parser.parse_args()
    
    if args.clean_temp:
        cleanup_temp_files()
        
    if args.check_disk:
        check_disk_space()
        
    if args.optimize_mem:
        optimize_memory_usage()
        
    if args.clean_checkpoints and args.output_dir:
        cleanup_old_checkpoints(args.output_dir, args.keep)
        
    if args.keep_best_only and args.output_dir and args.best_epoch:
        keep_only_best_checkpoint(args.output_dir, args.best_epoch)
