# Theta AI - Enhanced Conversation Edition

Theta is an internal AI assistant for Frostline Solutions employees, with newly enhanced multi-turn conversational capabilities. The system is designed to maintain context across conversations, track topics, detect inconsistencies, and provide more natural dialogue experiences.

## Enhanced Features

### Advanced Conversation Management

- **Multi-turn Context Awareness**: Maintains conversation history and references previous exchanges
- **Topic Detection and Tracking**: Identifies discussion topics and maintains coherence
- **Self-consistency Checking**: Prevents contradictions in responses
- **Follow-up Generation**: Creates natural follow-up questions to continue conversations
- **Enhanced Response Formatting**: Adds natural transitions and references to previous context

### Rich Training Data

- **Stack Exchange Technical Corpus**: Real-world technical Q&A from experts
- **GitHub Issue Conversations**: Multi-turn troubleshooting discussions
- **Technical Documentation**: Structured knowledge from official documentation
- **Tutorial Dialogues**: Step-by-step guides in conversation format

### Core Features

- Trained on Frostline-specific information with enhanced conversational capabilities
- Optimized for RTX 3060 GPU (12GB VRAM)
- Mobile-friendly web interface
- Robust response validation and hallucination prevention

## Documentation

Comprehensive documentation is available in the `docs/` directory:

- [Getting Started Guide](docs/getting_started.md)
- [Architecture Overview](docs/architecture.md)
- [Training Guide](docs/training_guide.md)
- [Dataset Guide](docs/dataset_guide.md)
- [Full Documentation Index](docs/README.md)

For contributors:

- [Contributing Guidelines](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)

## Getting Started

### Requirements

- NVIDIA RTX 3060 GPU (12GB)
- AMD Ryzen 5-5500
- CUDA toolkit
- Python 3.8+

### Installation

1. Clone this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

For web interface dependencies:

```bash
pip install -r web_requirements.txt
```

### Usage

#### Training

For overnight training with optimal parameters for RTX 3060:

```bash
train_overnight.bat
```

For enhanced training with additional optimizations:

```bash
train_overnight_enhanced.bat
```

See the [Training Guide](docs/training_guide.md) for detailed information on training options and parameters.

#### Using Theta AI

To use the command-line interface:

```bash
interface.bat
```

To start the web interface:

```bash
web_interface.bat
```

Both interfaces automatically use the best model checkpoint (epoch 26) with the lowest validation loss for optimal responses. If this checkpoint isn't available, they will fall back to the latest available checkpoint or the final model.

## Project Structure

- `src/`: Source code
  - `interface/`: Enhanced conversation management components
    - `conversation_manager.py`: Core conversation tracking and context handling
    - `topic_detection.py`: Topic identification and tracking
    - `consistency_checker.py`: Self-consistency verification
    - `followup_generator.py`: Natural follow-up question generation
  - `data_processing/`: Advanced dataset generators
    - `stack_exchange_processor.py`: Technical Q&A processing
    - `github_issue_conversations.py`: Multi-turn technical discussions
    - `technical_documentation.py`: Documentation corpus processor
    - `tutorial_dialogues.py`: Step-by-step guide converter
    - `generate_advanced_datasets.py`: Combined dataset generation
  - `model/`: Model architecture and interfaces
  - `database/`: Conversation persistence and retrieval
- `datasets/`: Contains training data in JSON format
- `models/`: Saved model checkpoints and configurations
- `templates/`: Web interface templates
- `static/`: Static assets for web interface
- `docs/`: Comprehensive documentation

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) before submitting pull requests.

## Version History

See the [Changelog](CHANGELOG.md) for version history and release notes.
