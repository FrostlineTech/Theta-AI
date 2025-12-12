# Contributing to Theta AI

Thank you for your interest in contributing to Theta AI! This project is designed as a foundation for you to build your own unique AI assistant.

## Branching Strategy

**This is important**: Theta AI is meant to be forked and customized. Each project should be unique and tailored to your specific needs.

### Create Your Own Branch

1. **Fork the repository** to your own GitHub account

2. **Create your own branch** for your unique project:

   ```bash
   git checkout -b your-project-name
   ```

3. **Build your own version** - customize the model, training data, and features to make it yours
4. **Keep your branch separate** - your project is your own, not intended to merge back to main

### Pull Requests to Main

**Only submit pull requests to `main` for bug fixes** that benefit everyone:

- Bug fixes in core training pipeline
- Security patches
- Documentation corrections
- Critical dependency updates

**Do NOT submit PRs to main for:**

- Custom features specific to your project
- New training data or datasets
- Model architecture changes for your use case
- Interface customizations

## Getting Started

1. **Fork and clone**:

   ```bash
   git clone https://github.com/yourusername/theta-ai.git
   cd theta-ai
   git checkout -b my-theta-project
   ```

2. **Set up environment**:

   ```bash
   pip install -r requirements.txt
   cp .env.example .env
   # Edit .env with your settings
   ```

3. **Start customizing** - make Theta your own!

## Development Guidelines

### Code Style

- Follow PEP 8 style guide for Python code
- Use descriptive variable and function names
- Include docstrings for functions and classes
- Use type hints where appropriate

### Commit Messages

- Use clear, descriptive commit messages
- Start with a verb (e.g., "Add", "Fix", "Update")
- Reference issue numbers where applicable

## Project Structure

```text
src/
├── model/           # Model architecture
├── training/        # Training pipeline
├── inference/       # Inference utilities
├── data_processing/ # Dataset processing
└── utils/           # Utilities
Datasets/            # Training data (JSON)
models/              # Saved checkpoints
documentation/       # Guides and references
```

## Working with Datasets

- Follow the JSON format in existing dataset files
- Create your own domain-specific training data
- Use the provided download scripts for base datasets
- Quality over quantity - curate your data carefully

## Bug Fix Pull Requests

If you find a bug in the core training pipeline:

1. Create a branch from `main`:

   ```bash
   git checkout main
   git pull origin main
   git checkout -b fix/description-of-bug
   ```

2. Fix the bug with minimal changes

3. Submit PR to `main` with:
   - Clear description of the bug
   - How your fix resolves it
   - Any testing performed

## Questions and Support

- Open an issue for bugs in the core pipeline
- Share your projects - we'd love to see what you build!

Thank you for using Theta AI as your foundation!
