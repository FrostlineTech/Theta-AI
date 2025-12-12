"""
Script to apply inference improvements after training.

This script is automatically called by the training pipeline to apply
the inference improvements to the trained model.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def main():
    """Apply inference improvements to the model"""
    from src.inference import install_inference_improvements
    from src.interface.theta_interface import ThetaInterface
    
    print("Applying inference improvements to the trained model...")
    
    # Get the path to the latest trained model
    model_dir = os.environ.get("THETA_TRAINED_MODEL_PATH", "")
    if not model_dir:
        # Try to find the latest model directory
        models_dir = project_root / "models"
        if models_dir.exists():
            # Find the most recent model directory
            model_dirs = [d for d in models_dir.iterdir() if d.is_dir() and d.name.startswith("theta_enhanced_")]
            if model_dirs:
                model_dir = str(max(model_dirs, key=lambda d: d.stat().st_mtime))
                print(f"Found latest model directory: {model_dir}")
            else:
                print("No model directories found")
        else:
            print(f"Models directory does not exist: {models_dir}")
    
    if not model_dir:
        print("Error: Could not determine model directory")
        return 1
    
    print(f"Using model directory: {model_dir}")
    
    # Initialize ThetaInterface with the trained model
    try:
        theta = ThetaInterface(model_path=model_dir)
        
        # Install inference improvements
        install_inference_improvements(theta)
        
        # Save config to the model directory
        inference_config_src = project_root / "src" / "inference" / "inference_config.json"
        inference_config_dst = Path(model_dir) / "inference_config.json"
        
        if inference_config_src.exists():
            import shutil
            shutil.copy2(inference_config_src, inference_config_dst)
            print(f"Copied inference configuration to {inference_config_dst}")
        
        print("Successfully applied inference improvements to the model!")
        
        # Create a flag file to indicate inference improvements have been applied
        with open(Path(model_dir) / "inference_improvements_applied.flag", "w") as f:
            import datetime
            f.write(f"Inference improvements applied at: {datetime.datetime.now()}")
        
        return 0
    except Exception as e:
        print(f"Error applying inference improvements: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
