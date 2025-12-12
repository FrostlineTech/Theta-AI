# Model Configuration

Configuration guide for Theta AI model settings.

## Base Models

| Model | Parameters | VRAM Required | Speed |
|-------|------------|---------------|-------|
| `gpt2` | 124M | ~4GB | Fast |
| `gpt2-medium` | 355M | ~8GB | Medium |
| `gpt2-large` | 774M | ~16GB | Slow |
| `gpt2-xl` | 1.5B | ~24GB | Very Slow |

**Recommended**: `gpt2-medium` for RTX 3060 12GB

## Tokenizer Configuration

### Special Tokens

The tokenizer uses special tokens for conversation formatting.

### Max Sequence Length

Default: 512 tokens (adjustable via tokenizer settings)

## Generation Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `max_length` | 256 | 1-1024 | Maximum tokens to generate |
| `temperature` | 0.8 | 0.1-2.0 | Randomness (higher = more random) |
| `top_p` | 0.9 | 0.0-1.0 | Nucleus sampling threshold |
| `top_k` | 50 | 1-100 | Top-k sampling |
| `repetition_penalty` | 1.2 | 1.0-2.0 | Penalty for repetition |
| `no_repeat_ngram_size` | 3 | 0-5 | Prevent n-gram repetition |

## Model Architecture

GPT-2 Medium architecture:

- Layers: 24
- Hidden size: 1024
- Attention heads: 16
- Vocabulary size: 50,257

## Checkpoint Format

Saved checkpoints include:

```text
theta_final/
├── config.json          # Model configuration
├── pytorch_model.bin    # Model weights
├── tokenizer.json       # Tokenizer config
├── special_tokens_map.json
├── vocab.json
└── merges.txt
```

## Loading Models

```python
from src.model.theta_model import ThetaModel

# Load from checkpoint
model = ThetaModel.load("models/theta_enhanced_YYYYMMDD/theta_final")

# Generate response
response = model.generate(
    prompt="What is machine learning?",
    max_length=200,
    temperature=0.8
)
```

## Fine-tuning from Checkpoint

To continue training from a saved model:

```bash
python src/training/train_enhanced.py \
  --model_name "models/theta_enhanced_YYYYMMDD/theta_final" \
  --data_path "Datasets/new_data.json" \
  --output_dir "models/theta_continued" \
  --epochs 5 \
  --learning_rate 1e-5
```

Note: Use lower learning rate when fine-tuning to preserve learned knowledge.
