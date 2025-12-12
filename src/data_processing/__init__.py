"""
Data Processing Package for Theta AI.

This package handles all data loading, processing, and preparation for training.
"""

# Export specialized data loader components
try:
    from src.data_processing.specialized_data_loader import (
        load_specialized_dataset,
        load_personality_datasets,
        extract_training_samples,
        merge_specialized_datasets,
        DatasetType,
        LoadedDataset
    )
    SPECIALIZED_LOADER_AVAILABLE = True
except ImportError:
    SPECIALIZED_LOADER_AVAILABLE = False

__all__ = [
    'SPECIALIZED_LOADER_AVAILABLE',
]

if SPECIALIZED_LOADER_AVAILABLE:
    __all__.extend([
        'load_specialized_dataset',
        'load_personality_datasets', 
        'extract_training_samples',
        'merge_specialized_datasets',
        'DatasetType',
        'LoadedDataset',
    ])
