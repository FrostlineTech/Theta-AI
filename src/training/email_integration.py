"""
Email notification integration for Theta AI training.

This module connects the training process with the email notification system.
"""

import os
import sys
from pathlib import Path

# Add parent directories to path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

# Import email notifier
from src.utils.email_notifier import TrainingEmailNotifier

class TrainingNotifier:
    """
    Integration class for training notifications.
    """
    
    def __init__(self, model_name="Theta AI", sender=None, recipient=None):
        """
        Initialize the training notifier.
        
        Args:
            model_name (str): Name of the model being trained
            sender (str): Sender email address
            recipient (str): Recipient email address
        """
        self.email_notifier = TrainingEmailNotifier(
            model_name=model_name,
            sender=sender,
            recipient=recipient
        )
        
    def start_notification(self, args):
        """
        Send notification that training has started.
        
        Args:
            args: Training arguments
        """
        self.email_notifier.start_training_notification(
            epochs=args.epochs,
            batch_size=args.batch_size * (args.gradient_accumulation_steps if hasattr(args, 'gradient_accumulation_steps') else 1),
            learning_rate=args.learning_rate
        )
        
    def epoch_notification(self, epoch, total_epochs, train_loss, val_loss, perplexity, duration):
        """
        Send notification after an epoch completes.
        
        Args:
            epoch (int): Current epoch number
            total_epochs (int): Total number of epochs
            train_loss (float): Training loss
            val_loss (float): Validation loss
            perplexity (float): Perplexity value
            duration (float): Duration of epoch in seconds
        """
        self.email_notifier.epoch_update(
            epoch=epoch,
            total_epochs=total_epochs,
            train_loss=train_loss,
            val_loss=val_loss,
            perplexity=perplexity,
            duration=duration
        )
        
    def completion_notification(self, total_epochs, best_val_loss, best_epoch, total_duration):
        """
        Send notification that training has completed.
        
        Args:
            total_epochs (int): Total number of epochs trained
            best_val_loss (float): Best validation loss achieved
            best_epoch (int): Epoch with the best validation loss
            total_duration (float): Total training duration in seconds
        """
        self.email_notifier.training_completed(
            total_epochs=total_epochs,
            best_val_loss=best_val_loss,
            best_epoch=best_epoch,
            total_duration=total_duration
        )
    
    def update_metrics(self, metrics: dict):
        """
        Update current training metrics for 10-min status updates.
        
        Args:
            metrics (dict): Dictionary containing current training metrics:
                - train_loss, val_loss, perplexity
                - token_accuracy, kl_loss, code_loss
                - current_epoch, total_epochs
                - curriculum_progress, difficult_ratio
                - domain_scores (dict), ema_active
                - current_lr, gradient_clip_ratio
        """
        self.email_notifier.update_metrics(metrics)

def get_notifier(model_name=None):
    """
    Get a training notifier instance.
    
    Args:
        model_name (str): Name of the model being trained
        
    Returns:
        TrainingNotifier: A training notifier instance
    """
    # Use model_name or default to "Theta AI"
    model_name = model_name or "Theta AI"
    
    return TrainingNotifier(model_name=model_name)
