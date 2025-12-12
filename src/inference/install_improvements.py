#!/usr/bin/env python3
"""
Script to install inference improvements into the Theta AI training pipeline.
This updates the necessary files to integrate the inference enhancements.
"""

import os
import sys
import shutil
from pathlib import Path

def main():
    """Main function to install improvements"""
    print("Installing Theta AI inference improvements...")
    
    # Get script directory and project root
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent.parent
    
    # Check if we're in the right directory structure
    if not (project_root / "train_overnight_enhanced.bat").exists():
        print("Error: Could not find train_overnight_enhanced.bat in the expected location.")
        print(f"Current directory structure: {project_root}")
        return 1
    
    # Backup original training script
    train_script = project_root / "train_overnight_enhanced.bat"
    backup_script = project_root / "train_overnight_enhanced.bat.bak"
    
    if not backup_script.exists():
        print(f"Backing up original training script to {backup_script}")
        shutil.copy2(train_script, backup_script)
    
    # Modify the training script to include our improvements
    try:
        with open(train_script, 'r') as f:
            content = f.read()
        
        # Only modify if not already modified
        if "inference_improvements" not in content:
            # Find the line where we run the enhanced training
            run_training_line = "python src/training/train_enhanced.py "
            
            # Add environment variable for inference improvements
            inference_env_line = "set THETA_USE_INFERENCE_IMPROVEMENTS=true\n"
            
            # Insert it before training command
            if "set PYTORCH_CUDA_ALLOC_CONF" in content:
                modified_content = content.replace(
                    "set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128",
                    "set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128\n" + inference_env_line
                )
            else:
                # Fallback if we can't find the exact spot
                modified_content = content.replace(
                    "REM PyTorch optimizations",
                    "REM PyTorch optimizations\n" + inference_env_line
                )
            
            # Add our post-training step
            inference_install_step = "\nREM Setup inference improvements\necho Setting up inference improvements...\npython src/inference/apply_inference_improvements.py\n"
            
            # Add before the final summary
            if "Training completed at:" in modified_content:
                modified_content = modified_content.replace(
                    "echo.\necho Training completed at:", 
                    f"{inference_install_step}\necho.\necho Training completed at:"
                )
            else:
                # Append at the end if we can't find the right spot
                modified_content += inference_install_step
            
            # Write the modified content back
            with open(train_script, 'w') as f:
                f.write(modified_content)
                
            print("Successfully updated training script to include inference improvements")
        else:
            print("Training script already includes inference improvements")
        
        # Create the apply_inference_improvements.py script if it doesn't exist
        apply_script_path = project_root / "src" / "inference" / "apply_inference_improvements.py"
        if not apply_script_path.exists():
            apply_script_content = '''"""
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
        date_str = Path(__file__).resolve().parent.parent.parent / "train_overnight_enhanced.bat"
        try:
            with open(date_str, "r") as f:
                content = f.read()
                import re
                match = re.search(r'models/theta_enhanced_(\d+)', content)
                if match:
                    model_dir = f"models/theta_enhanced_{match.group(1)}"
        except Exception as e:
            print(f"Error finding model directory: {e}")
    
    print(f"Using model directory: {model_dir}")
    
    # Initialize ThetaInterface with the trained model
    try:
        theta = ThetaInterface(model_path=model_dir)
        
        # Install inference improvements
        install_inference_improvements(theta)
        
        print("Successfully applied inference improvements to the model!")
        return 0
    except Exception as e:
        print(f"Error applying inference improvements: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
'''
            # Create the directory if it doesn't exist
            apply_script_path.parent.mkdir(exist_ok=True)
            
            # Write the script
            with open(apply_script_path, 'w') as f:
                f.write(apply_script_content)
                
            print(f"Created application script at {apply_script_path}")
    
    except Exception as e:
        print(f"Error updating training script: {e}")
        return 1
    
    print("\nInference improvements installation complete!")
    print("Next steps:")
    print("1. Run the training script as usual: train_overnight_enhanced.bat")
    print("2. The script will automatically apply inference improvements after training")
    print("3. To manually apply improvements to an existing model, run:")
    print("   python src/inference/apply_inference_improvements.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
