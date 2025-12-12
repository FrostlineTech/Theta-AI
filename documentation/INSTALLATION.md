# Installation Guide

## System Requirements

### Hardware (Recommended)
- **GPU**: NVIDIA RTX 3060 12GB (or equivalent with 12GB+ VRAM)
- **CPU**: AMD Ryzen 5-5500 or Intel equivalent (6+ cores)
- **RAM**: 32GB DDR4
- **Storage**: 100GB+ free space (SSD recommended)

### Software
- Windows 10/11 or Linux
- Python 3.8 - 3.11
- CUDA Toolkit 11.7+
- cuDNN 8.x

## Step-by-Step Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/theta-ai.git
cd theta-ai
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install CUDA (if not already installed)

Download and install from: https://developer.nvidia.com/cuda-downloads

Verify installation:
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

### 5. Configure Environment

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your settings
# Required: Email settings for notifications
# Optional: Database settings, API keys
```

### 6. Setup NLTK Data

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('stopwords')"
```

### 7. Verify Installation

```bash
python -c "from src.model.theta_model import ThetaModel; print('Installation successful!')"
```

## Directory Setup

The training script will create these directories automatically:
- `models/` - Saved checkpoints
- `logs/` - Training logs
- `cache/` - HuggingFace cache

## Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` in training arguments
- Enable CPU offloading (default in train_overnight_enhanced.bat)
- Close other GPU applications

### ImportError: No module named 'xxx'
```bash
pip install -r requirements.txt --force-reinstall
```

### NLTK Data Not Found
```bash
python setup_nltk_data.py
```

## Next Steps

- [Quick Start Training](QUICKSTART.md)
- [Dataset Preparation](DATASETS.md)
