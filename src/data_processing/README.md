# Theta AI Advanced Datasets

This directory contains modules for generating advanced datasets to enhance Theta AI's training with more comprehensive and diverse data.

## Dataset Types

### Stack Exchange Technical Corpus

- **Source**: Stack Exchange Q&A sites (Stack Overflow, Server Fault, Super User)
- **Content**: High-quality technical Q&A from experts
- **Size**: ~10,000 examples
- **Benefits**: Real-world technical problems with well-structured solutions
- **Module**: `stack_exchange_processor.py`

### GitHub Issue Conversations

- **Source**: GitHub issues and pull requests from popular technical repositories
- **Content**: Multi-turn troubleshooting conversations and technical discussions
- **Size**: ~2,000 conversations
- **Benefits**: Natural back-and-forth technical problem-solving patterns
- **Module**: `github_issue_conversations.py`

### Technical Documentation

- **Source**: Official documentation from major platforms (AWS, Azure, Linux)
- **Content**: Authoritative explanations of technical concepts
- **Size**: ~5,000 Q&A pairs
- **Benefits**: Accurate, well-structured technical knowledge
- **Module**: `technical_documentation.py`

### Tutorial Dialogues

- **Source**: Technical tutorials converted to conversational format
- **Content**: Step-by-step guides in question-answer format
- **Size**: ~1,000 tutorials
- **Benefits**: Teaching Theta to guide users through complex procedures
- **Module**: `tutorial_dialogues.py`

## Usage

### Generate all advanced datasets

```python
python -m src.data_processing.generate_advanced_datasets
```

This will generate all datasets and save them to the `Datasets` directory.

### Generate individual datasets

```python
# Generate Stack Exchange dataset
python -m src.data_processing.stack_exchange_processor

# Generate GitHub conversations dataset
python -m src.data_processing.github_issue_conversations

# Generate technical documentation dataset
python -m src.data_processing.technical_documentation

# Generate tutorial dialogues dataset
python -m src.data_processing.tutorial_dialogues
```

### Process all datasets for training

```python
python -m src.data_processing.process_data
```

This will combine all available datasets (including the advanced ones) into a single training file.

## Dataset Format

All datasets follow a consistent format:

```json
[
  {
    "question": "How do I implement X?",
    "answer": "To implement X, follow these steps...",
    "additional_metadata": "values depend on the dataset"
  }
]
```

Some datasets have additional fields specific to their source, such as:

- Stack Exchange: includes source site, score, tags
- GitHub: includes repository, issue number, domain
- Documentation: includes source, topic, section
- Tutorials: includes domain, topic, step number

## Integration with Training

The advanced datasets are automatically included in the training data by the `process_data.py` script.

When running the training process, all available datasets will be combined, balanced, and used for model training.
