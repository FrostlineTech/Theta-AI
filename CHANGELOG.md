# Changelog

All notable changes to the Theta AI project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- Enhanced documentation structure
- Improved repository sharing setup
- Added contributing guidelines

## [1.2.0] - 2025-09-10

### Added

- Enhanced training pipeline with mixed precision support
- New hardware_knowledge.json dataset for improved hardware-related responses
- Web interface improvements with mobile responsiveness
- Conversation history export functionality

### Changed

- Optimized model architecture for better performance on RTX 3060
- Updated retrieval mechanism with TF-IDF similarity for more relevant responses
- Improved error handling in training scripts

### Fixed

- Memory leak in long conversation sessions
- Inconsistent response formatting in certain scenarios
- Training stability issues with large batch sizes

## [1.1.0] - 2025-08-15

### Added

- Web interface with Flask backend
- Response validation to prevent hallucinations
- New datasets for cybersecurity and cloud computing
- Batch scripts for easier training and deployment

### Changed

- Refactored code structure for better maintainability
- Enhanced tokenization process for improved performance
- Updated training parameters for faster convergence

### Fixed

- GPU memory optimization issues
- Incorrect handling of special tokens
- Data preprocessing inconsistencies

## [1.0.0] - 2025-07-01

### Added

- Initial release of Theta AI
- Command-line interface for interaction
- Training pipeline optimized for RTX 3060
- Basic datasets covering company information, programming, and cybersecurity
- Model checkpoint saving and loading functionality
- Logging system for training and interface interactions
