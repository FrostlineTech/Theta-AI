# Contributing to Theta AI

Thank you for your interest in contributing to Theta AI! This guide will help you get started with the development process and outline the best practices for contributing to this project.

## Getting Started

1. **Fork the repository** and clone your fork
2. **Set up the development environment**:
   ```bash
   pip install -r requirements.txt
   pip install -r web_requirements.txt  # If you'll be working on the web interface
   ```
3. **Create a new branch** for your feature or bug fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Guidelines

### Code Style

- Follow PEP 8 style guide for Python code
- Use descriptive variable and function names
- Include docstrings for all functions and classes
- Use type hints where appropriate

### Commit Messages

- Use clear, descriptive commit messages
- Start with a verb in the imperative mood (e.g., "Add", "Fix", "Update")
- Reference issue numbers where applicable

### Pull Request Process

1. Update the README.md or documentation with details of changes if appropriate
2. Update the CHANGELOG.md following the format described in that file
3. Make sure your code passes all tests
4. The PR should be reviewed by at least one maintainer
5. Once approved, a maintainer will merge your PR

## Project Structure

- `src/`: Source code
  - `data_processing/`: Scripts for data preparation
  - `model/`: Model architecture definition
  - `training/`: Training pipelines
  - `interface/`: User interaction interfaces
- `datasets/`: Contains training data
- `docs/`: Documentation
- `models/`: Model checkpoints and configurations
- `static/`: Static assets for web interface
- `templates/`: HTML templates for web interface

## Working with Datasets

When adding new training data:
1. Follow the JSON format in existing dataset files
2. Ensure high quality of information (accuracy, relevance, completeness)
3. Run validation scripts to check for proper formatting

## Working with Models

- Model checkpoints (`.pt`, `.bin`, `.pb` files) are excluded from version control
- Model configuration files (`.json`, `.yaml`, `.py`) should be included
- Document hyperparameters and training settings in code or in the docs

## Testing

- Add unit tests for new functionality
- Ensure existing tests pass with your changes
- For interface changes, perform manual testing and document the process

## Documentation

- Update documentation for any new features or changes
- Place comprehensive documentation in the `docs/` folder
- Include code examples where appropriate

## Questions and Support

If you have questions or need help with your contribution, please:

- Open an issue in the repository
- Reach out to the maintainers through appropriate channels

Thank you for contributing to Theta AI!
