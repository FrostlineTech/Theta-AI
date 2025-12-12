# Troubleshooting Guide

Common issues and solutions for Theta AI training.

## Installation Issues

### CUDA Not Detected

**Symptom**: `torch.cuda.is_available()` returns `False`

**Solutions**:

1. Verify NVIDIA driver:
   ```bash
   nvidia-smi
   ```

2. Check CUDA version compatibility:
   ```bash
   nvcc --version
   python -c "import torch; print(torch.version.cuda)"
   ```

3. Reinstall PyTorch with correct CUDA:
   ```bash
   pip uninstall torch
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```

### Import Errors

**Symptom**: `ModuleNotFoundError: No module named 'xxx'`

**Solution**:
```bash
pip install -r requirements.txt --force-reinstall
```

### NLTK Data Missing

**Symptom**: `Resource punkt not found`

**Solution**:
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('stopwords')"
```

Or run:
```bash
python setup_nltk_data.py
```

## Training Issues

### CUDA Out of Memory

**Symptom**: `RuntimeError: CUDA out of memory`

**Solutions** (in order of effectiveness):

1. **Reduce batch size**:
   ```bash
   --batch_size 2
   ```

2. **Increase gradient accumulation**:
   ```bash
   --gradient_accumulation_steps 6
   ```

3. **Enable CPU offloading**:
   ```bash
   --use_cpu_offload --cpu_offload_fraction 0.6
   ```

4. **Disable R-Drop** (removes dual forward pass):
   ```bash
   --no_rdrop
   ```

5. **Use smaller model**:
   ```bash
   --model_name gpt2  # Instead of gpt2-medium
   ```

### Training Divergence (Loss → NaN/Inf)

**Symptom**: Loss becomes NaN or infinity

**Solutions**:

1. **Lower learning rate**:
   ```bash
   --learning_rate 1e-5
   ```

2. **Increase warmup**:
   ```bash
   --warmup_proportion 0.2
   ```

3. **Check data for issues**:
   ```bash
   python validate_json_files.py
   ```

4. **Use ablation mode** (baseline training):
   ```bash
   --ablation_mode
   ```

### Slow Training

**Symptom**: Training takes longer than expected

**Diagnosis**:
```bash
nvidia-smi -l 1  # Check GPU utilization
```

**Solutions**:

1. **Low GPU utilization (< 50%)**:
   - Reduce `num_workers` (CPU bottleneck)
   - Check disk speed (HDD vs SSD)

2. **High GPU utilization but slow**:
   - Verify TF32 is enabled
   - Check for thermal throttling
   - Reduce batch size if memory fragmented

3. **Verify environment variables**:
   ```batch
   set NVIDIA_TF32_OVERRIDE=1
   set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
   ```

### Validation Loss Not Improving

**Symptom**: Val loss plateaus or increases

**Solutions**:

1. **Overfitting** (train loss drops, val loss increases):
   - Increase `--weight_decay`
   - Enable `--use_rdrop`
   - Increase `--label_smoothing`
   - Check data quality

2. **Underfitting** (both losses high):
   - Increase `--learning_rate`
   - Train more epochs
   - Reduce regularization

3. **Data quality issues**:
   - Check for duplicates
   - Verify input/output pairing
   - Balance domain distribution

### Early Stopping Too Soon

**Symptom**: Training stops after few epochs

**Solution**:
```bash
--patience 7  # Increase from default 5
```

## Email Notification Issues

### Emails Not Sending

**Symptom**: No email notifications received

**Diagnosis**:
```bash
python test_email_notification.py
```

**Solutions**:

1. **Verify .env settings**:
   ```bash
   EMAIL_PASSWORD=your_16_char_app_password
   EMAIL_SENDER=your_email@gmail.com
   EMAIL_SMTP_SERVER=smtp.gmail.com
   EMAIL_SMTP_PORT=587
   ```

2. **Gmail requires App Password**:
   - Enable 2FA on Google account
   - Generate App Password at Security → App passwords

3. **Check firewall**:
   - Allow outbound port 587

### Missing Metrics in Emails

**Symptom**: 10-min updates don't show training metrics

**Cause**: Metrics update after first epoch completes

**Solution**: Wait for first epoch to finish

## Data Issues

### JSON Parse Error

**Symptom**: `JSONDecodeError`

**Solution**:
```bash
python validate_json_files.py
```

Common fixes:
- Remove trailing commas
- Fix unescaped quotes in strings
- Check encoding (should be UTF-8)

### Dataset Too Large

**Symptom**: Memory error loading dataset

**Solutions**:

1. Use streaming:
   ```python
   # In custom data loader
   for line in open(file):
       yield json.loads(line)
   ```

2. Split large files:
   ```bash
   python -c "import json; d=json.load(open('large.json')); [json.dump(d[i:i+10000], open(f'part{i//10000}.json','w')) for i in range(0,len(d),10000)]"
   ```

### Encoding Issues

**Symptom**: Unicode errors or garbled text

**Solution**:
```bash
python src/data_processing/fix_encodings.py
```

## Model Issues

### Checkpoint Not Loading

**Symptom**: Error loading saved model

**Solutions**:

1. **Check path**:
   ```python
   import os
   print(os.path.exists("models/theta_enhanced_YYYYMMDD/theta_final"))
   ```

2. **Version mismatch**:
   - Ensure same transformers version as training
   - Try `from_pretrained` instead of `load_state_dict`

### Poor Generation Quality

**Symptom**: Model outputs are repetitive or nonsensical

**Solutions**:

1. **Adjust generation parameters**:
   ```python
   model.generate(
       ...,
       temperature=0.8,
       top_p=0.9,
       no_repeat_ngram_size=3,
       repetition_penalty=1.2
   )
   ```

2. **Check training**:
   - Verify val_loss decreased during training
   - Check token accuracy > 50%

3. **Use EMA weights**:
   - EMA model usually generates better than base model

## System Issues

### Disk Space Full

**Symptom**: Training fails with disk error

**Solutions**:

1. **Enable disk optimization**:
   ```bash
   --optimize_disk --keep_best_only
   ```

2. **Clean checkpoints**:
   ```bash
   python -c "from src.training.cleanup_utils import cleanup_old_checkpoints; cleanup_old_checkpoints('models/', keep_last_n=1)"
   ```

3. **Move cache to larger drive**:
   ```batch
   set HF_HOME=D:\cache
   ```

### Process Killed

**Symptom**: Training process terminates unexpectedly

**Causes**:
- OOM Killer (Linux) - reduce memory usage
- Windows memory limits - increase virtual memory
- Thermal shutdown - improve cooling

## Getting Help

If issues persist:

1. Check log file: `logs/training_enhanced_*.log`
2. Review training config: `models/*/training_config.json`
3. Check system resources: `nvidia-smi`, `htop`/Task Manager
