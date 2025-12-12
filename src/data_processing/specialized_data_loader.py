"""
Specialized Data Loader for Theta AI.

This module provides flexible loading for different data formats without
forcing everything into Q&A pairs. Preserves the native structure of:
- Opinion datasets
- Personality datasets
- Humor frameworks
- Tactical knowledge
- Fragment personality data
- Conversational patterns
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class DatasetType(Enum):
    """Types of datasets we can handle."""
    QA_PAIRS = "qa_pairs"                    # Standard question/answer format
    OPINIONS = "opinions"                     # Topic-based opinions
    PERSONALITY = "personality"               # Personality traits and patterns
    FRAMEWORK = "framework"                   # Structured frameworks (humor, emotional intelligence)
    CONVERSATIONAL = "conversational"         # Conversation patterns
    TACTICAL = "tactical"                     # Tactical knowledge and insights
    FRAGMENT = "fragment"                     # AI fragment personality data
    PROBLEM_SOLUTION = "problem_solution"     # Math/coding problems with solutions
    STRUCTURED = "structured"                 # Generic structured data
    RAW = "raw"                               # Keep as-is


@dataclass
class LoadedDataset:
    """Represents a loaded dataset with metadata."""
    name: str
    path: str
    data_type: DatasetType
    content: Any
    entry_count: int
    domain: str = "general"
    metadata: Dict = field(default_factory=dict)


def detect_dataset_type(data: Any, filename: str) -> DatasetType:
    """
    Detect the type of dataset based on its structure and filename.
    
    Args:
        data: The loaded JSON data
        filename: Name of the file
        
    Returns:
        DatasetType enum value
    """
    filename_lower = filename.lower()
    
    # Check filename hints first
    if "opinion" in filename_lower:
        return DatasetType.OPINIONS
    elif "personality" in filename_lower or "fragment" in filename_lower:
        return DatasetType.FRAGMENT
    elif "humor" in filename_lower:
        return DatasetType.FRAMEWORK
    elif "emotional" in filename_lower or "intelligence" in filename_lower:
        return DatasetType.FRAMEWORK
    elif "tactical" in filename_lower:
        return DatasetType.TACTICAL
    elif "conversation" in filename_lower or "small_talk" in filename_lower:
        return DatasetType.CONVERSATIONAL
    elif "math" in filename_lower or "instruct" in filename_lower:
        return DatasetType.PROBLEM_SOLUTION
    
    # Check data structure
    if isinstance(data, list):
        if len(data) > 0 and isinstance(data[0], dict):
            first_item = data[0]
            # Check for Q&A structure
            if "question" in first_item and "answer" in first_item:
                return DatasetType.QA_PAIRS
            # Check for problem-solution structure
            elif "problem" in first_item and ("solution" in first_item or "generated_solution" in first_item):
                return DatasetType.PROBLEM_SOLUTION
            # Check for opinion structure
            elif "topic" in first_item and "opinion" in first_item:
                return DatasetType.OPINIONS
    
    elif isinstance(data, dict):
        # Check for framework structure
        if any(key in data for key in ["framework", "components", "taxonomy", "humor_types"]):
            return DatasetType.FRAMEWORK
        # Check for entries wrapper
        elif "entries" in data:
            return DatasetType.STRUCTURED
        # Check for metadata wrapper
        elif "metadata" in data and "content" in data:
            return DatasetType.STRUCTURED
    
    return DatasetType.RAW


def count_entries(data: Any, data_type: DatasetType) -> int:
    """Count the number of entries in a dataset based on its type."""
    if isinstance(data, list):
        return len(data)
    elif isinstance(data, dict):
        if "entries" in data:
            return len(data["entries"])
        elif data_type == DatasetType.FRAMEWORK:
            # Count framework components
            total = 0
            for key, value in data.items():
                if isinstance(value, list):
                    total += len(value)
                elif isinstance(value, dict):
                    total += len(value)
            return max(total, 1)
        else:
            return len(data)
    return 1


def load_specialized_dataset(file_path: Union[str, Path], 
                            domain: str = None,
                            preserve_structure: bool = True) -> Optional[LoadedDataset]:
    """
    Load a dataset while preserving its specialized structure.
    
    Args:
        file_path: Path to the JSON file
        domain: Optional domain override
        preserve_structure: If True, don't convert to Q&A format
        
    Returns:
        LoadedDataset object or None if loading fails
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        logger.warning(f"Dataset file not found: {file_path}")
        return None
    
    try:
        # Try UTF-8 first, fall back to latin1
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except UnicodeDecodeError:
            logger.warning(f"UTF-8 decode failed for {file_path}, trying latin1")
            with open(file_path, 'r', encoding='latin1') as f:
                data = json.load(f)
        
        filename = file_path.name
        data_type = detect_dataset_type(data, filename)
        entry_count = count_entries(data, data_type)
        
        # Determine domain from filename if not provided
        if domain is None:
            domain = filename.replace('.json', '').lower()
            # Clean up domain name
            domain = domain.replace('_', ' ').replace('-', ' ').strip()
        
        logger.info(f"Loaded {filename}: type={data_type.value}, entries={entry_count}, domain={domain}")
        
        return LoadedDataset(
            name=filename,
            path=str(file_path),
            data_type=data_type,
            content=data,
            entry_count=entry_count,
            domain=domain,
            metadata={
                "source_file": filename,
                "preserved_structure": preserve_structure
            }
        )
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error for {file_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return None


def extract_training_samples(dataset: LoadedDataset, 
                            max_samples: int = None) -> List[Dict[str, Any]]:
    """
    Extract training samples from a loaded dataset in a format suitable for training.
    Does NOT force Q&A format - returns appropriate format based on dataset type.
    
    Args:
        dataset: LoadedDataset to extract from
        max_samples: Optional limit on samples
        
    Returns:
        List of training samples in appropriate format
    """
    samples = []
    data = dataset.content
    
    if dataset.data_type == DatasetType.QA_PAIRS:
        # Standard Q&A format - return as-is
        if isinstance(data, list):
            samples = data
        
    elif dataset.data_type == DatasetType.CONVERSATIONAL:
        # Conversational data - keep the conversational structure
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    sample = {
                        "type": "conversation",
                        "domain": dataset.domain,
                        **item
                    }
                    samples.append(sample)
    
    elif dataset.data_type == DatasetType.OPINIONS:
        # Opinion data - preserve opinion structure
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    sample = {
                        "type": "opinion",
                        "domain": dataset.domain,
                        "topic": item.get("topic", "general"),
                        "subtopic": item.get("subtopic", ""),
                        "opinion": item.get("opinion", ""),
                        "strength": item.get("strength", "moderate"),
                        "context": item.get("context", "")
                    }
                    samples.append(sample)
    
    elif dataset.data_type == DatasetType.PROBLEM_SOLUTION:
        # Math/coding problems - preserve problem-solution structure
        if isinstance(data, dict) and "entries" in data:
            entries = data["entries"]
        elif isinstance(data, list):
            entries = data
        else:
            entries = []
            
        for item in entries:
            if isinstance(item, dict):
                sample = {
                    "type": "problem_solution",
                    "domain": dataset.domain,
                    "problem": item.get("problem", item.get("question", "")),
                    "solution": item.get("solution", item.get("generated_solution", item.get("answer", ""))),
                    "expected_answer": item.get("expected_answer", ""),
                    "source": item.get("source", dataset.name)
                }
                samples.append(sample)
    
    elif dataset.data_type == DatasetType.FRAGMENT:
        # Fragment personality data - keep personality structure
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    sample = {
                        "type": "fragment_personality",
                        "domain": dataset.domain,
                        **item
                    }
                    samples.append(sample)
    
    elif dataset.data_type == DatasetType.FRAMEWORK:
        # Framework data - preserve the framework structure
        sample = {
            "type": "framework",
            "domain": dataset.domain,
            "name": dataset.name.replace('.json', ''),
            "content": data
        }
        samples.append(sample)
    
    elif dataset.data_type == DatasetType.TACTICAL:
        # Tactical knowledge - preserve structure
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    sample = {
                        "type": "tactical",
                        "domain": dataset.domain,
                        **item
                    }
                else:
                    sample = {
                        "type": "tactical",
                        "domain": dataset.domain,
                        "content": item
                    }
                samples.append(sample)
        elif isinstance(data, dict):
            sample = {
                "type": "tactical",
                "domain": dataset.domain,
                "content": data
            }
            samples.append(sample)
    
    else:  # RAW or STRUCTURED
        # Keep as-is but tag with metadata
        sample = {
            "type": "raw",
            "domain": dataset.domain,
            "content": data
        }
        samples.append(sample)
    
    # Apply sample limit if specified
    if max_samples and len(samples) > max_samples:
        samples = samples[:max_samples]
    
    return samples


def load_personality_datasets(datasets_dir: Union[str, Path]) -> Dict[str, LoadedDataset]:
    """
    Load all personality-related datasets from the Datasets directory.
    
    Args:
        datasets_dir: Path to the Datasets directory
        
    Returns:
        Dictionary mapping dataset names to LoadedDataset objects
    """
    datasets_dir = Path(datasets_dir)
    personality_files = [
        "theta_opinions.json",
        "basic_conversation.json",
        "basic_conversation_natural.json",
        "small_talk.json",
        "small_talk_natural.json",
        "Humor_comprehension.json",
        "Emotional_intelligence.json",
        "Conversational_dynamics.json",
        "delta_fragment.json",
        "sigma_fragment.json",
        "omega_fragment.json",
        "gamma_fragment.json",
        "eta_fragment.json",
        "iota_fragment.json",
        "beta_fragment.json",
        "lambda_fragment.json",
        "kappa_fragment.json",
    ]
    
    loaded = {}
    for filename in personality_files:
        file_path = datasets_dir / filename
        if file_path.exists():
            dataset = load_specialized_dataset(file_path, preserve_structure=True)
            if dataset:
                loaded[filename] = dataset
    
    logger.info(f"Loaded {len(loaded)} personality datasets")
    return loaded


def merge_specialized_datasets(datasets: List[LoadedDataset], 
                              group_by_type: bool = True) -> Dict[str, List[Dict]]:
    """
    Merge multiple specialized datasets, optionally grouping by type.
    
    Args:
        datasets: List of LoadedDataset objects
        group_by_type: If True, group samples by their type
        
    Returns:
        Dictionary mapping type names to lists of samples, or {"all": [...]} if not grouped
    """
    if group_by_type:
        grouped = {}
        for dataset in datasets:
            samples = extract_training_samples(dataset)
            for sample in samples:
                sample_type = sample.get("type", "unknown")
                if sample_type not in grouped:
                    grouped[sample_type] = []
                grouped[sample_type].append(sample)
        return grouped
    else:
        all_samples = []
        for dataset in datasets:
            samples = extract_training_samples(dataset)
            all_samples.extend(samples)
        return {"all": all_samples}
