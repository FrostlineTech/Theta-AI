"""
Email notification system for Theta AI training.

Sends branded email updates about training progress.
"""

import os
import time
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from datetime import datetime
import threading
from pathlib import Path
import sys

# Try to import optional dependencies with fallbacks
try:
    from dotenv import load_dotenv
except ImportError:
    print("Warning: python-dotenv not found. Environment variables may not be loaded correctly.")
    # Simple fallback for load_dotenv
    def load_dotenv():
        print("Using fallback environment loader (limited functionality)")
        return

# Import our custom GPU info module
from src.utils.gpu_info import get_best_gpu_info

# System utilities with fallback
try:
    import psutil
    has_psutil = True
except ImportError:
    print("Warning: psutil not found. System information will be limited.")
    has_psutil = False

# Load environment variables
load_dotenv()

# Email configuration
DEFAULT_SENDER = os.getenv('email_sender', 'frostlinesolutionsllc@gmail.com')
DEFAULT_RECIPIENT = os.getenv('email_recipient', 'frostlinesolutionsllc@gmail.com')
EMAIL_PASSWORD = os.getenv('email_password')
SMTP_SERVER = os.getenv('email_smtp_server', 'smtp.gmail.com')
SMTP_PORT = int(os.getenv('email_smtp_port', '587'))

# Frostline brand colors
FROSTLINE_BLUE = "#0088cc"  # Primary blue color
FROSTLINE_DARK = "#2c3e50"  # Dark navy/text color
FROSTLINE_LIGHT = "#ecf0f1"  # Light background color
FROSTLINE_ACCENT = "#e74c3c"  # Accent color for warnings/alerts

class TrainingEmailNotifier:
    """
    Manages email notifications for Theta AI training.
    """
    
    def __init__(self, sender=None, recipient=None, model_name="Theta AI"):
        """
        Initialize the email notifier.
        
        Args:
            sender (str): Sender email address
            recipient (str): Recipient email address
            model_name (str): Name of the model being trained
        """
        self.sender = sender or DEFAULT_SENDER
        self.recipient = recipient or DEFAULT_RECIPIENT
        self.model_name = model_name
        self.password = EMAIL_PASSWORD
        self.training_start_time = None
        self.last_gpu_update = None
        self.gpu_monitor_thread = None
        self.stop_monitoring = False
        
        # Current training metrics for 10-min status updates
        self.current_metrics = {}
        
        # Find logo path
        self.logo_path = self._find_logo()
        
    def _find_logo(self):
        """Find the Frostline logo in the static directory."""
        possible_paths = [
            Path("static/theta-symbol.png"),
            Path("static/frostline-logo.png"),
            Path("static/logo.png"),
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        return None
    
    def start_training_notification(self, epochs, batch_size, learning_rate):
        """
        Send notification that training has started.
        
        Args:
            epochs (int): Total number of epochs
            batch_size (int): Batch size
            learning_rate (float): Learning rate
        """
        self.training_start_time = time.time()
        
        subject = f"🚀 {self.model_name} Training Started"
        
        # Get GPU info
        gpu_info = self._get_gpu_info()
        
        # Create message content
        html = self._get_html_template()
        html = html.replace("{TITLE}", "Training Process Initiated")
        
        # Training parameters section
        parameters_content = f"""
        <h3>Training Parameters</h3>
        <table style="width:100%; border-collapse: collapse;">
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;"><strong>Total Epochs</strong></td>
                <td style="padding: 8px; border: 1px solid #ddd;">{epochs}</td>
            </tr>
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;"><strong>Batch Size</strong></td>
                <td style="padding: 8px; border: 1px solid #ddd;">{batch_size}</td>
            </tr>
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;"><strong>Learning Rate</strong></td>
                <td style="padding: 8px; border: 1px solid #ddd;">{learning_rate}</td>
            </tr>
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;"><strong>Start Time</strong></td>
                <td style="padding: 8px; border: 1px solid #ddd;">{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</td>
            </tr>
        </table>
        """
        
        # GPU information section
        gpu_content = f"""
        <h3>GPU Information</h3>
        <table style="width:100%; border-collapse: collapse;">
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;"><strong>GPU Name</strong></td>
                <td style="padding: 8px; border: 1px solid #ddd;">{gpu_info['name']}</td>
            </tr>
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;"><strong>Temperature</strong></td>
                <td style="padding: 8px; border: 1px solid #ddd;">{gpu_info['temperature']}°C</td>
            </tr>
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;"><strong>Memory Used</strong></td>
                <td style="padding: 8px; border: 1px solid #ddd;">{gpu_info['memory_used']} MB / {gpu_info['memory_total']} MB</td>
            </tr>
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;"><strong>GPU Utilization</strong></td>
                <td style="padding: 8px; border: 1px solid #ddd;">{gpu_info['utilization']}%</td>
            </tr>
        </table>
        """
        
        # System information
        system_content = f"""
        <h3>System Information</h3>
        <p>CPU Usage: {psutil.cpu_percent()}%<br>
        Memory Usage: {round(psutil.virtual_memory().percent, 2)}%<br>
        Available RAM: {round(psutil.virtual_memory().available / (1024**3), 2)} GB</p>
        """
        
        content = f"""
        <p>The training process for {self.model_name} has been initiated. You will receive updates after each epoch and GPU status updates every 10 minutes.</p>
        
        {parameters_content}
        
        {gpu_content}
        
        {system_content}
        
        <p>The system will automatically send you updates throughout the training process.</p>
        """
        
        html = html.replace("{CONTENT}", content)
        
        # Send the email
        self._send_email(subject, html)
        
        # Start GPU monitoring
        self._start_gpu_monitoring()
        
    def epoch_update(self, epoch, total_epochs, train_loss, val_loss, perplexity, duration):
        """
        Send notification after an epoch completes.
        
        Args:
            epoch (int): Current epoch number
            total_epochs (int): Total number of epochs
            train_loss (float): Training loss
            val_loss (float): Validation loss
            perplexity (float): Perplexity value
            duration (float): Duration of epoch in seconds
        """
        subject = f"⏱️ {self.model_name} - Epoch {epoch}/{total_epochs} Completed"
        
        # Get GPU info
        gpu_info = self._get_gpu_info()
        
        # Create message content
        html = self._get_html_template()
        html = html.replace("{TITLE}", f"Training Update: Epoch {epoch}/{total_epochs}")
        
        # Progress bar
        progress_percent = int((epoch / total_epochs) * 100)
        progress_bar = self._generate_progress_bar(progress_percent)
        
        # Calculate time elapsed and estimated time remaining
        time_elapsed = time.time() - self.training_start_time
        time_per_epoch = time_elapsed / epoch if epoch > 0 else 0
        time_remaining = time_per_epoch * (total_epochs - epoch)
        
        # Format time strings
        elapsed_str = self._format_time(time_elapsed)
        remaining_str = self._format_time(time_remaining)
        
        # Training metrics section
        metrics_content = f"""
        <h3>Training Metrics</h3>
        <table style="width:100%; border-collapse: collapse;">
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;"><strong>Train Loss</strong></td>
                <td style="padding: 8px; border: 1px solid #ddd;">{train_loss:.4f}</td>
            </tr>
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;"><strong>Validation Loss</strong></td>
                <td style="padding: 8px; border: 1px solid #ddd;">{val_loss:.4f}</td>
            </tr>
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;"><strong>Perplexity</strong></td>
                <td style="padding: 8px; border: 1px solid #ddd;">{perplexity:.2f}</td>
            </tr>
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;"><strong>Epoch Duration</strong></td>
                <td style="padding: 8px; border: 1px solid #ddd;">{self._format_time(duration)}</td>
            </tr>
        </table>
        """
        
        # GPU information section
        gpu_content = f"""
        <h3>GPU Status</h3>
        <table style="width:100%; border-collapse: collapse;">
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;"><strong>Temperature</strong></td>
                <td style="padding: 8px; border: 1px solid #ddd;">{gpu_info['temperature']}°C</td>
            </tr>
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;"><strong>Memory Used</strong></td>
                <td style="padding: 8px; border: 1px solid #ddd;">{gpu_info['memory_used']} MB / {gpu_info['memory_total']} MB</td>
            </tr>
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;"><strong>GPU Utilization</strong></td>
                <td style="padding: 8px; border: 1px solid #ddd;">{gpu_info['utilization']}%</td>
            </tr>
        </table>
        """
        
        content = f"""
        <div style="margin-bottom: 20px;">
            <p><strong>Progress: {progress_percent}%</strong> ({epoch}/{total_epochs} epochs)</p>
            {progress_bar}
        </div>
        
        {metrics_content}
        
        {gpu_content}
        
        <div style="margin-top: 20px;">
            <p><strong>Time Elapsed:</strong> {elapsed_str}<br>
            <strong>Estimated Time Remaining:</strong> {remaining_str}</p>
        </div>
        """
        
        html = html.replace("{CONTENT}", content)
        
        # Send the email
        self._send_email(subject, html)
        
    def update_metrics(self, metrics: dict):
        """
        Update current training metrics for inclusion in 10-min status updates.
        
        Args:
            metrics (dict): Dictionary containing current training metrics:
                - train_loss, val_loss, perplexity
                - token_accuracy, kl_loss, code_loss
                - current_epoch, total_epochs
                - curriculum_progress, difficult_ratio
                - domain_scores (dict), ema_active
                - current_lr, gradient_clip_ratio
        """
        self.current_metrics = metrics
    
    def _get_metric_color(self, metric_name, value):
        """Get color based on metric thresholds for alerts."""
        thresholds = {
            'val_loss': (1.0, 2.0),  # green < 1.0, yellow < 2.0, red >= 2.0
            'perplexity': (10, 50),
            'token_accuracy': (0.7, 0.5),  # reversed: green > 0.7, yellow > 0.5, red <= 0.5
            'kl_loss': (0.5, 1.0),
            'code_loss': (0.1, 0.3),
            'difficult_ratio': (0.5, 0.7),
            'gradient_clip_ratio': (0.5, 0.8),
        }
        
        if metric_name not in thresholds:
            return "#6c757d"  # Gray for unknown
        
        low, high = thresholds[metric_name]
        
        # Reversed metrics (higher is better)
        if metric_name == 'token_accuracy':
            if value >= low:
                return "#28a745"  # Green
            elif value >= high:
                return "#ffc107"  # Yellow
            else:
                return "#dc3545"  # Red
        else:
            # Normal metrics (lower is better)
            if value < low:
                return "#28a745"  # Green
            elif value < high:
                return "#ffc107"  # Yellow
            else:
                return "#dc3545"  # Red
    
    def _build_metrics_alert_section(self):
        """Build HTML section for training metrics with color-coded alerts."""
        m = self.current_metrics
        if not m:
            return ""
        
        # Build alerts list
        alerts = []
        if m.get('perplexity', 0) > 50:
            alerts.append(f"🚨 Perplexity Critical: {m['perplexity']:.2f}")
        if m.get('token_accuracy', 1) < 0.5:
            alerts.append(f"⚠️ Token Accuracy Low: {m['token_accuracy']*100:.1f}%")
        if m.get('kl_loss', 0) > 1.0:
            alerts.append(f"⚠️ KL Loss High: {m['kl_loss']:.4f}")
        if m.get('difficult_ratio', 0) > 0.7:
            alerts.append(f"⚠️ Difficult Sample Ratio High: {m['difficult_ratio']*100:.0f}%")
        if m.get('gradient_clip_ratio', 0) > 0.8:
            alerts.append(f"⚠️ Excessive Gradient Clipping: {m['gradient_clip_ratio']*100:.0f}%")
        if m.get('current_lr', 1) < 1e-8:
            alerts.append(f"⚠️ Learning Rate Near Zero: {m['current_lr']:.2e}")
        
        # Domain alerts
        domain_scores = m.get('domain_scores', {})
        for domain, score in domain_scores.items():
            if score < 0.5:
                alerts.append(f"⚠️ {domain} Coherence Low: {score:.2f}")
        
        # Build alerts HTML
        alerts_html = ""
        if alerts:
            alerts_html = f"""
            <h3 style="color: {FROSTLINE_ACCENT};">⚠️ Active Alerts</h3>
            <ul style="color: {FROSTLINE_ACCENT};">
            """ + "".join(f"<li>{a}</li>" for a in alerts) + "</ul>"
        
        # Build metrics table
        metrics_rows = ""
        
        # Progress
        if 'current_epoch' in m and 'total_epochs' in m:
            progress = int((m['current_epoch'] / m['total_epochs']) * 100)
            metrics_rows += f"""
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;"><strong>Progress</strong></td>
                <td style="padding: 8px; border: 1px solid #ddd;">{m['current_epoch']}/{m['total_epochs']} epochs ({progress}%)</td>
            </tr>"""
        
        # Core metrics
        if 'train_loss' in m:
            metrics_rows += f"""
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;"><strong>Train Loss</strong></td>
                <td style="padding: 8px; border: 1px solid #ddd;">{m['train_loss']:.4f}</td>
            </tr>"""
        
        if 'val_loss' in m:
            color = self._get_metric_color('val_loss', m['val_loss'])
            metrics_rows += f"""
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;"><strong>Val Loss</strong></td>
                <td style="padding: 8px; border: 1px solid #ddd;"><span style="color: {color}">{m['val_loss']:.4f}</span></td>
            </tr>"""
        
        if 'perplexity' in m:
            color = self._get_metric_color('perplexity', m['perplexity'])
            metrics_rows += f"""
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;"><strong>Perplexity</strong></td>
                <td style="padding: 8px; border: 1px solid #ddd;"><span style="color: {color}">{m['perplexity']:.2f}</span></td>
            </tr>"""
        
        if 'token_accuracy' in m:
            color = self._get_metric_color('token_accuracy', m['token_accuracy'])
            metrics_rows += f"""
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;"><strong>Token Accuracy</strong></td>
                <td style="padding: 8px; border: 1px solid #ddd;"><span style="color: {color}">{m['token_accuracy']*100:.2f}%</span></td>
            </tr>"""
        
        if 'kl_loss' in m:
            color = self._get_metric_color('kl_loss', m['kl_loss'])
            metrics_rows += f"""
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;"><strong>KL Loss (R-Drop)</strong></td>
                <td style="padding: 8px; border: 1px solid #ddd;"><span style="color: {color}">{m['kl_loss']:.4f}</span></td>
            </tr>"""
        
        if 'code_loss' in m and m['code_loss'] > 0:
            color = self._get_metric_color('code_loss', m['code_loss'])
            metrics_rows += f"""
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;"><strong>Code Contrastive Loss</strong></td>
                <td style="padding: 8px; border: 1px solid #ddd;"><span style="color: {color}">{m['code_loss']:.4f}</span></td>
            </tr>"""
        
        if 'current_lr' in m:
            metrics_rows += f"""
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;"><strong>Learning Rate</strong></td>
                <td style="padding: 8px; border: 1px solid #ddd;">{m['current_lr']:.2e}</td>
            </tr>"""
        
        if 'curriculum_progress' in m:
            metrics_rows += f"""
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;"><strong>Curriculum Progress</strong></td>
                <td style="padding: 8px; border: 1px solid #ddd;">{m['curriculum_progress']*100:.0f}%</td>
            </tr>"""
        
        if 'difficult_ratio' in m:
            color = self._get_metric_color('difficult_ratio', m['difficult_ratio'])
            metrics_rows += f"""
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;"><strong>Difficult Sample Ratio</strong></td>
                <td style="padding: 8px; border: 1px solid #ddd;"><span style="color: {color}">{m['difficult_ratio']*100:.1f}%</span></td>
            </tr>"""
        
        if 'gradient_clip_ratio' in m:
            color = self._get_metric_color('gradient_clip_ratio', m['gradient_clip_ratio'])
            metrics_rows += f"""
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;"><strong>Gradient Clip Ratio</strong></td>
                <td style="padding: 8px; border: 1px solid #ddd;"><span style="color: {color}">{m['gradient_clip_ratio']*100:.1f}%</span></td>
            </tr>"""
        
        if 'ema_active' in m:
            ema_status = "Active" if m['ema_active'] else "Warmup"
            metrics_rows += f"""
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;"><strong>EMA Status</strong></td>
                <td style="padding: 8px; border: 1px solid #ddd;">{ema_status}</td>
            </tr>"""
        
        # Domain scores section
        domain_html = ""
        if domain_scores:
            domain_rows = ""
            for domain, score in domain_scores.items():
                color = "#28a745" if score >= 0.7 else ("#ffc107" if score >= 0.5 else "#dc3545")
                domain_rows += f"""
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;"><strong>{domain.title()}</strong></td>
                    <td style="padding: 8px; border: 1px solid #ddd;"><span style="color: {color}">{score:.2f}</span></td>
                </tr>"""
            domain_html = f"""
            <h3>Domain Performance</h3>
            <table style="width:100%; border-collapse: collapse;">
                {domain_rows}
            </table>
            """
        
        if not metrics_rows:
            return ""
        
        return f"""
        {alerts_html}
        
        <h3>Training Metrics</h3>
        <table style="width:100%; border-collapse: collapse;">
            {metrics_rows}
        </table>
        
        {domain_html}
        """
    
    def gpu_status_update(self):
        """Send a GPU status update with training metrics and alerts."""
        # Determine if there are any alerts
        alerts = []
        m = self.current_metrics
        if m:
            if m.get('perplexity', 0) > 50:
                alerts.append("Perplexity Critical")
            if m.get('token_accuracy', 1) < 0.5:
                alerts.append("Token Accuracy Low")
            if m.get('kl_loss', 0) > 1.0:
                alerts.append("KL Loss High")
            if m.get('difficult_ratio', 0) > 0.7:
                alerts.append("Difficulty High")
        
        # Modify subject if alerts present
        if alerts:
            subject = f"⚠️ {self.model_name} - Status Update ({len(alerts)} alerts)"
        else:
            subject = f"🔄 {self.model_name} - Status Update"
        
        # Get GPU info
        gpu_info = self._get_gpu_info()
        
        # Create message content
        html = self._get_html_template()
        html = html.replace("{TITLE}", "Training Status Update")
        
        # Training metrics section (with alerts)
        metrics_content = self._build_metrics_alert_section()
        
        # GPU information section
        gpu_content = f"""
        <h3>GPU Status</h3>
        <table style="width:100%; border-collapse: collapse;">
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;"><strong>Temperature</strong></td>
                <td style="padding: 8px; border: 1px solid #ddd;"><span style="color: {self._get_temperature_color(gpu_info['temperature'])}">{gpu_info['temperature']}°C</span></td>
            </tr>
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;"><strong>Memory Used</strong></td>
                <td style="padding: 8px; border: 1px solid #ddd;">{gpu_info['memory_used']} MB / {gpu_info['memory_total']} MB ({int(gpu_info['memory_used']/gpu_info['memory_total']*100)}%)</td>
            </tr>
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;"><strong>GPU Utilization</strong></td>
                <td style="padding: 8px; border: 1px solid #ddd;">{gpu_info['utilization']}%</td>
            </tr>
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;"><strong>Power Draw</strong></td>
                <td style="padding: 8px; border: 1px solid #ddd;">{gpu_info['power_draw']} W</td>
            </tr>
        </table>
        
        <h3>System Status</h3>
        <table style="width:100%; border-collapse: collapse;">
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;"><strong>CPU Usage</strong></td>
                <td style="padding: 8px; border: 1px solid #ddd;">{psutil.cpu_percent()}%</td>
            </tr>
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;"><strong>Memory Usage</strong></td>
                <td style="padding: 8px; border: 1px solid #ddd;">{round(psutil.virtual_memory().percent, 2)}%</td>
            </tr>
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;"><strong>Available RAM</strong></td>
                <td style="padding: 8px; border: 1px solid #ddd;">{round(psutil.virtual_memory().available / (1024**3), 2)} GB</td>
            </tr>
        </table>
        """
        
        content = metrics_content + gpu_content
        
        html = html.replace("{CONTENT}", content)
        
        # Send the email
        self._send_email(subject, html)
        
    def training_completed(self, total_epochs, best_val_loss, best_epoch, total_duration):
        """
        Send notification that training has completed.
        
        Args:
            total_epochs (int): Total number of epochs trained
            best_val_loss (float): Best validation loss achieved
            best_epoch (int): Epoch with the best validation loss
            total_duration (float): Total training duration in seconds
        """
        # Stop GPU monitoring
        self.stop_monitoring = True
        if self.gpu_monitor_thread and self.gpu_monitor_thread.is_alive():
            self.gpu_monitor_thread.join()
        
        subject = f"✅ {self.model_name} Training Completed!"
        
        # Get GPU info
        gpu_info = self._get_gpu_info()
        
        # Create message content
        html = self._get_html_template()
        html = html.replace("{TITLE}", "Training Process Complete")
        
        # Summary section
        summary_content = f"""
        <h3>Training Summary</h3>
        <table style="width:100%; border-collapse: collapse;">
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;"><strong>Total Epochs</strong></td>
                <td style="padding: 8px; border: 1px solid #ddd;">{total_epochs}</td>
            </tr>
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;"><strong>Best Validation Loss</strong></td>
                <td style="padding: 8px; border: 1px solid #ddd;">{best_val_loss:.4f}</td>
            </tr>
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;"><strong>Best Epoch</strong></td>
                <td style="padding: 8px; border: 1px solid #ddd;">{best_epoch}</td>
            </tr>
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;"><strong>Total Training Time</strong></td>
                <td style="padding: 8px; border: 1px solid #ddd;">{self._format_time(total_duration)}</td>
            </tr>
        </table>
        """
        
        # GPU final status
        gpu_content = f"""
        <h3>Final GPU Status</h3>
        <table style="width:100%; border-collapse: collapse;">
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;"><strong>Temperature</strong></td>
                <td style="padding: 8px; border: 1px solid #ddd;">{gpu_info['temperature']}°C</td>
            </tr>
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;"><strong>Memory Used</strong></td>
                <td style="padding: 8px; border: 1px solid #ddd;">{gpu_info['memory_used']} MB / {gpu_info['memory_total']} MB</td>
            </tr>
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;"><strong>GPU Utilization</strong></td>
                <td style="padding: 8px; border: 1px solid #ddd;">{gpu_info['utilization']}%</td>
            </tr>
        </table>
        """
        
        content = f"""
        <p>Training for {self.model_name} has been successfully completed! 🎉</p>
        
        {summary_content}
        
        {gpu_content}
        
        <div style="margin-top: 20px;">
            <p>The best model checkpoint has been saved and is ready for use.</p>
            <p>Start time: {datetime.fromtimestamp(self.training_start_time).strftime('%Y-%m-%d %H:%M:%S')}<br>
            End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        """
        
        html = html.replace("{CONTENT}", content)
        
        # Send the email
        self._send_email(subject, html)
    
    def _get_gpu_info(self):
        """Get current GPU information using our improved gpu_info module."""
        try:
            # Use our custom GPU info utility that tries multiple methods
            return get_best_gpu_info()
        except Exception as e:
            print(f"Error getting GPU info: {e}")
            return self._get_default_gpu_info()
    
    def _get_default_gpu_info(self):
        """Return default GPU information when actual info cannot be obtained."""
        return {
            'name': 'No GPU information available',
            'temperature': 0,
            'memory_used': 0,
            'memory_total': 1,
            'utilization': 0,
            'power_draw': 'N/A'
        }
    
    def _get_temperature_color(self, temp):
        """Get color based on temperature."""
        # Handle non-numeric temperature values
        if temp == 'N/A' or not isinstance(temp, (int, float)):
            return "#6c757d"  # Gray for N/A
            
        # Handle numeric temperatures
        if temp < 50:
            return "#28a745"  # Green
        elif temp < 70:
            return "#ffc107"  # Yellow
        else:
            return "#dc3545"  # Red
    
    def _generate_progress_bar(self, percent):
        """Generate HTML progress bar."""
        return f"""
        <div style="width: 100%; background-color: #e9ecef; border-radius: 4px; overflow: hidden;">
            <div style="width: {percent}%; background-color: {FROSTLINE_BLUE}; height: 20px;"></div>
        </div>
        """
    
    def _format_time(self, seconds):
        """Format time in seconds to a readable string."""
        hours, remainder = divmod(int(seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    
    def _get_html_template(self):
        """Get HTML template for emails with Frostline branding."""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{{TITLE}}</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    color: {FROSTLINE_DARK};
                    margin: 0;
                    padding: 0;
                    background-color: #f4f4f4;
                }}
                .container {{
                    max-width: 600px;
                    margin: 0 auto;
                    background-color: #ffffff;
                    padding: 20px;
                }}
                .header {{
                    background-color: {FROSTLINE_DARK};
                    color: white;
                    padding: 20px;
                    text-align: center;
                }}
                .logo {{
                    max-width: 150px;
                    margin-bottom: 10px;
                }}
                .content {{
                    padding: 20px;
                }}
                .footer {{
                    text-align: center;
                    padding: 20px;
                    font-size: 0.8em;
                    color: #777;
                }}
                h1 {{
                    color: {FROSTLINE_BLUE};
                    margin-top: 0;
                }}
                h3 {{
                    color: {FROSTLINE_BLUE};
                    border-bottom: 1px solid #eee;
                    padding-bottom: 10px;
                }}
                table {{
                    width: 100%;
                }}
                td {{
                    padding: 8px;
                }}
                .button {{
                    display: inline-block;
                    background-color: {FROSTLINE_BLUE};
                    color: white;
                    padding: 10px 20px;
                    text-decoration: none;
                    border-radius: 4px;
                    margin-top: 20px;
                }}
                .highlight {{
                    background-color: {FROSTLINE_LIGHT};
                    padding: 15px;
                    border-radius: 4px;
                    margin: 20px 0;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h2>Frostline Solutions</h2>
                    <p>Theta AI Training Monitor</p>
                </div>
                <div class="content">
                    <h1>{{TITLE}}</h1>
                    {{CONTENT}}
                </div>
                <div class="footer">
                    <p>This is an automated notification from the Theta AI training system.<br>
                    &copy; {datetime.now().year} Frostline Solutions</p>
                </div>
            </div>
        </body>
        </html>
        """
    
    def _send_email(self, subject, html_content):
        """Send email with HTML content."""
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.sender
            msg['To'] = self.recipient
            
            # Attach HTML content
            msg.attach(MIMEText(html_content, 'html'))
            
            # Try to attach logo if available
            if self.logo_path and self.logo_path.exists():
                with open(self.logo_path, 'rb') as img_file:
                    img = MIMEImage(img_file.read())
                    img.add_header('Content-ID', '<logo>')
                    msg.attach(img)
            
            # Create secure connection and send email
            context = ssl.create_default_context()
            
            # Connect to server
            server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
            server.ehlo()
            server.starttls(context=context)
            server.ehlo()
            server.login(self.sender, self.password)
            
            # Send email
            server.sendmail(self.sender, self.recipient, msg.as_string())
            
            # Quit server
            server.quit()
            
            print(f"Email notification sent: {subject}")
            return True
            
        except Exception as e:
            pass  # Using fallback methods instead
            return False
    
    def _start_gpu_monitoring(self):
        """Start GPU monitoring thread that sends updates every 10 minutes."""
        self.stop_monitoring = False
        # Initialize the thread if it doesn't exist
        self.gpu_monitor_thread = threading.Thread(target=self._gpu_monitor_task)
        self.gpu_monitor_thread.daemon = True
        self.gpu_monitor_thread.start()
    
    def _gpu_monitor_task(self):
        """GPU monitoring task."""
        interval_seconds = 10 * 60  # 10 minutes
        
        while not self.stop_monitoring:
            time.sleep(interval_seconds)
            
            if self.stop_monitoring:
                break
                
            self.gpu_status_update()
