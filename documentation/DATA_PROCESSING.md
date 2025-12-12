# Data Processing Guide

Comprehensive guide to data preparation for Theta AI training.

## Overview

The data processing pipeline converts raw datasets into training-ready format with:

- Domain balancing
- Quality scoring
- Curriculum ordering
- Synthetic augmentation
- Consistency checking

## Pipeline Architecture

```text
Raw Datasets (JSON)
        │
        ▼
┌─────────────────────┐
│   Load & Merge      │  load_all_datasets()
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│   Validate Format   │  validate_examples()
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│   Balance Domains   │  balance_domains()
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│   Quality Scoring   │  compute_quality_scores()
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│   Curriculum Order  │  apply_curriculum_ordering()
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│   Synthetic Augment │  generate_synthetic()
└─────────────────────┘
        │
        ▼
enhanced_training_data.json
```

## Quick Start

```bash
python prepare_data_for_training.py --output "Datasets/enhanced_training_data.json"
```

## Data Format

### Input Format

```json
[
  {
    "input": "User question or prompt",
    "output": "Expected AI response"
  }
]
```

### Enhanced Output Format

```json
[
  {
    "input": "User question or prompt",
    "output": "Expected AI response",
    "domain": "cybersecurity",
    "quality_score": 0.92,
    "difficulty": 3,
    "has_code": false,
    "length_bucket": "medium"
  }
]
```

## Processing Steps

### 1. Domain Balancing

Ensures representation across all domains.

**Target Distribution**:

| Domain | Target % |
|--------|----------|
| programming | 20% |
| cybersecurity | 15% |
| conversational | 15% |
| general_tech | 15% |
| data_science | 10% |
| networking | 10% |
| human_behavior | 10% |
| other | 5% |

**Methods**:
- Undersampling: Cap overrepresented domains
- Oversampling: Duplicate underrepresented examples
- Synthetic: Generate new examples for gaps

### 2. Quality Scoring

Assigns 0.0-1.0 quality score based on:

| Factor | Weight | Description |
|--------|--------|-------------|
| Length ratio | 0.2 | Output length vs input |
| Coherence | 0.25 | Sentence structure |
| Specificity | 0.2 | Technical terms, examples |
| Completeness | 0.2 | Proper endings, structure |
| Formatting | 0.15 | Code blocks, lists |

**Example**:

```python
quality = compute_quality_score({
    "input": "How do I sort a list in Python?",
    "output": "Use the sorted() function..."
})
# Returns: 0.85
```

### 3. Curriculum Ordering

Orders examples from easy to hard based on:

- Character length (primary)
- Code block count
- Technical term density
- List/structure complexity

**Difficulty Buckets**:

| Bucket | Characteristics |
|--------|-----------------|
| Easy | Short, conversational |
| Medium | Moderate length, some technical |
| Hard | Long, code-heavy, complex |

### 4. Code Detection

Marks examples containing code:

```python
has_code = (
    "```" in output or
    "def " in output or
    "class " in output or
    "import " in output
)
```

### 5. Synthetic Augmentation

Generates additional examples for underrepresented domains:

**Techniques**:
- Paraphrasing (input variation)
- Response expansion
- Question generation
- Error scenario generation

## Processing Scripts

### Main Scripts

| Script | Purpose |
|--------|---------|
| `prepare_data_for_training.py` | Full pipeline |
| `validate_json_files.py` | Validate format |
| `check_dataset.py` | Statistics |

### Data Processing (`src/data_processing/`)

| Script | Purpose |
|--------|---------|
| `dataset_balancer.py` | Domain balancing |
| `data_augmentation.py` | Synthetic generation |
| `consistency_checker.py` | Quality validation |
| `knowledge_enhancer.py` | RAG enhancement |
| `fix_encodings.py` | Encoding fixes |

## Dataset Sources

### Included (Small)

| Source | Description |
|--------|-------------|
| Custom JSON | Hand-crafted examples |
| Case studies | Domain-specific scenarios |
| Curated Q&A | Validated question-answers |

### External (Download Required)

| Source | Script | Size |
|--------|--------|------|
| OpenAssistant | `download_openassistant.bat` | 6GB |
| Human-like DPO | `download_human_like_dpo.bat` | 28MB |
| OpenMath | `download_openmath_instruct.bat` | 9GB |

### Generated

| Source | Script | Description |
|--------|--------|-------------|
| OpenWebText | Auto during training | General text |
| Tutorials | `generate_tutorials.py` | Step-by-step guides |
| Technical Q&A | `generate_technical_qa.py` | Tech questions |

## Quality Assurance

### Validation Checks

```bash
python validate_json_files.py
```

Checks:
- JSON syntax
- Required fields (input, output)
- Encoding (UTF-8)
- Minimum lengths

### Statistics

```bash
python check_dataset.py
```

Output:
- Total examples
- Domain distribution
- Length statistics
- Quality score distribution

## Advanced Configuration

### Custom Domain Weights

```python
# In prepare_data_for_training.py
DOMAIN_WEIGHTS = {
    'cybersecurity': 2.0,  # 2x importance
    'programming': 1.5,
    'conversation': 1.0,
}
```

### Quality Thresholds

```python
MIN_QUALITY_SCORE = 0.5  # Exclude low-quality
MIN_INPUT_LENGTH = 10    # Minimum input chars
MIN_OUTPUT_LENGTH = 20   # Minimum output chars
```

### Augmentation Settings

```python
AUGMENTATION_RATIO = 0.2  # 20% synthetic
MAX_SYNTHETIC_PER_DOMAIN = 1000
```

## Troubleshooting

### Memory Issues

For large datasets:

```bash
# Process in chunks
python prepare_data_for_training.py --chunk_size 10000
```

### Encoding Errors

```bash
python src/data_processing/fix_encodings.py
```

### Duplicate Detection

```python
# Remove duplicates by input hash
seen = set()
unique = []
for item in data:
    h = hash(item['input'])
    if h not in seen:
        seen.add(h)
        unique.append(item)
```
