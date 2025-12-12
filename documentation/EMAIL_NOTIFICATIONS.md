# Email Notifications System

Complete documentation for the training email notification system.

## Overview

The email notification system sends real-time updates during model training, including:

- Training start/completion
- Epoch progress updates
- 10-minute status updates with metric alerts
- GPU health monitoring

## Configuration

### Environment Variables

Add to your `.env` file:

```bash
EMAIL_PASSWORD=your_app_password
EMAIL_SENDER=your_email@gmail.com
EMAIL_RECIPIENT=your_email@gmail.com
EMAIL_SMTP_SERVER=smtp.gmail.com
EMAIL_SMTP_PORT=587
```

### Gmail Setup

1. Enable 2-Factor Authentication on your Google account
2. Generate an App Password:
   - Go to Google Account → Security → App passwords
   - Generate password for "Mail"
   - Use this 16-character password in `EMAIL_PASSWORD`

### Testing

```bash
python test_email_notification.py
```

## Notification Types

### 1. Training Start

**Trigger**: When training begins

**Content**:
- Training parameters (epochs, batch size, learning rate)
- GPU information (name, temperature, memory)
- System information (CPU, RAM)

### 2. Epoch Updates

**Trigger**: After each epoch completes

**Content**:
- Progress bar
- Training/validation loss
- Perplexity
- Token accuracy
- Time elapsed/remaining
- GPU status

### 3. 10-Minute Status Updates

**Trigger**: Every 10 minutes during training

**Content**:
- Active alerts (if any)
- Current training metrics
- GPU status
- System status

### 4. Training Completion

**Trigger**: When training finishes

**Content**:
- Training summary
- Best validation loss and epoch
- Total training time
- Final model location

## Metric Alerts

The 10-minute status updates include color-coded alerts:

| Metric | Alert Threshold | Color |
|--------|-----------------|-------|
| Perplexity | > 50 | 🔴 Critical |
| Token Accuracy | < 50% | 🟡 Warning |
| KL Loss | > 1.0 | 🟡 Warning |
| Difficult Ratio | > 70% | 🟡 Warning |
| Gradient Clip | > 80% | 🟡 Warning |
| Learning Rate | < 1e-8 | 🟡 Warning |
| Domain Coherence | < 0.5 | 🟡 Warning |

When alerts are present, the email subject changes to:
```
⚠️ Theta AI - Status Update (N alerts)
```

## Metrics Included

### Core Metrics

| Metric | Description |
|--------|-------------|
| `train_loss` | Current epoch training loss |
| `val_loss` | Current epoch validation loss |
| `perplexity` | Model perplexity (exp of loss) |
| `token_accuracy` | Token prediction accuracy |
| `current_lr` | Current learning rate |

### Advanced Metrics

| Metric | Description |
|--------|-------------|
| `kl_loss` | R-Drop KL divergence loss |
| `code_loss` | Code contrastive auxiliary loss |
| `curriculum_progress` | Curriculum learning progress |
| `difficult_ratio` | Ratio of difficult samples |
| `ema_active` | Whether EMA is active |
| `domain_scores` | Per-domain coherence scores |

## Architecture

```text
src/utils/email_notifier.py
├── TrainingEmailNotifier
│   ├── start_training_notification()
│   ├── epoch_update()
│   ├── gpu_status_update()      # 10-min updates
│   ├── training_completed()
│   ├── update_metrics()         # Called by training loop
│   └── _build_metrics_alert_section()

src/training/email_integration.py
├── TrainingNotifier            # Wrapper class
│   ├── start_notification()
│   ├── epoch_notification()
│   ├── completion_notification()
│   └── update_metrics()
└── get_notifier()              # Factory function
```

## Integration with Training

The training script automatically:

1. Initializes notifier at training start
2. Sends start notification with configuration
3. Updates metrics after each epoch
4. Background thread sends 10-min status updates
5. Sends completion notification

```python
# In train_enhanced.py
notifier = get_notifier(model_name)
notifier.start_notification(args)

# After each epoch
current_metrics = {
    'train_loss': avg_train_loss,
    'val_loss': avg_val_loss,
    # ... other metrics
}
notifier.update_metrics(current_metrics)
notifier.epoch_notification(...)
```

## Disabling Notifications

If email is not configured, training continues with a dummy notifier:

```python
class DummyNotifier:
    def start_notification(self, *a, **k): pass
    def epoch_notification(self, *a, **k): pass
    def completion_notification(self, *a, **k): pass
    def update_metrics(self, *a, **k): pass
```

## Troubleshooting

### Emails Not Sending

1. Verify `.env` credentials
2. Check Gmail App Password (not regular password)
3. Ensure less secure apps or App Password is enabled
4. Check firewall for SMTP port 587

### Delayed Emails

- Gmail may rate limit (max ~100 emails/day)
- Reduce notification frequency if needed

### Missing Metrics

- Metrics update after first epoch completes
- 10-min updates show last known values
