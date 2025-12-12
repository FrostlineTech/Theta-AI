# Dataset Guide

Complete guide to Theta AI training datasets.

## Dataset Format

All datasets use JSON format with this structure:

```json
[
  {
    "input": "User question or prompt",
    "output": "Expected AI response",
    "domain": "category_name",
    "quality_score": 0.95
  }
]
```

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `input` | string | User prompt or question |
| `output` | string | Expected response |

### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `domain` | string | Category (cybersecurity, programming, etc.) |
| `quality_score` | float | 0.0-1.0, used for sample weighting |
| `has_code` | bool | Whether response contains code |
| `difficulty` | int | 1-5 difficulty rating |

## Included Datasets

### Core Datasets (Small, Included in Repo)

| Dataset | Size | Description |
|---------|------|-------------|
| `basic_conversation.json` | 3.5KB | Simple greetings and small talk |
| `small_talk.json` | 2.5KB | Casual conversation examples |
| `theta_identity.json` | 4.7KB | Theta AI personality and identity |
| `code_block_usage.json` | 4.8KB | Code formatting examples |
| `technical_qa.json` | 10KB | Technical Q&A pairs |

### Domain Datasets (Medium, Included)

| Dataset | Domain | Description |
|---------|--------|-------------|
| `cybersecurity_part1/2.json` | Security | Cybersecurity concepts |
| `programming_part1/2/3.json` | Coding | Programming tutorials |
| `data_science_part1/2.json` | Data | ML/AI concepts |
| `cloud_computing.json` | Cloud | AWS/Azure/GCP |
| `networking.json` | Networks | Network protocols |

### Large Datasets (Not in Repo - Generate Locally)

| Dataset | Size | How to Generate |
|---------|------|-----------------|
| `enhanced_training_data.json` | 270MB | `prepare_data_for_training.py` |
| `openassistant_oasst1.json` | 6GB | `download_openassistant.bat` |
| `human_like_dpo_dataset.json` | 28MB | `download_human_like_dpo.bat` |
| `openwebtext_processed.json` | 5MB | Auto-generated during training |

## Generating Training Data

### Full Enhanced Dataset

```bash
python prepare_data_for_training.py --output "Datasets/enhanced_training_data.json"
```

This script:
1. Loads all JSON datasets from `Datasets/`
2. Balances domain distribution
3. Applies quality scoring
4. Generates synthetic examples
5. Applies curriculum ordering

### Download External Datasets

```bash
# OpenAssistant (6GB)
download_openassistant.bat

# Human-like DPO dataset
download_human_like_dpo.bat

# OpenMath (large, optional)
download_openmath_instruct.bat
```

## Creating Custom Datasets

### Step 1: Create JSON File

```json
[
  {
    "input": "What is your specialty?",
    "output": "I specialize in cybersecurity and technical support.",
    "domain": "identity"
  },
  {
    "input": "How do I secure my API?",
    "output": "Here are key API security practices:\n1. Use HTTPS...",
    "domain": "cybersecurity"
  }
]
```

### Step 2: Validate Format

```bash
python validate_json_files.py
```

### Step 3: Add to Training

Place in `Datasets/` folder. The `prepare_data_for_training.py` script automatically includes all JSON files.

## Domain Categories

| Category | Keywords | Description |
|----------|----------|-------------|
| `conversation` | small_talk, greetings | Basic dialogue |
| `technical` | programming, cybersecurity | Technical content |
| `human_behavior` | emotion, cognition | Human-like responses |
| `reasoning` | logic, ethics | Reasoning tasks |
| `mathematics` | math, calculate | Math problems |

## Data Quality Guidelines

### Good Examples

```json
{
  "input": "How do I handle null pointer exceptions?",
  "output": "Null pointer exceptions occur when you try to use an object reference that points to null. Here's how to handle them:\n\n1. **Check for null before use**:\n```python\nif obj is not None:\n    obj.method()\n```\n\n2. **Use optional chaining** (where available)\n3. **Initialize variables** with default values",
  "domain": "programming",
  "quality_score": 0.95
}
```

### Avoid

- Single-word responses
- Responses without context
- Contradictory information
- Excessive repetition
- Hallucinated facts

## Dataset Statistics

View dataset statistics:

```bash
python check_dataset.py
```

Output includes:
- Total examples per domain
- Average input/output length
- Quality score distribution
- Domain balance ratios
