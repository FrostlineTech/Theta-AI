"""
GPU information utilities for Theta AI.

This module provides alternative methods to get GPU information
when gputil doesn't work properly.
"""

import os
import subprocess
import platform
import re
import json

def get_gpu_info_from_nvidia_smi():
    """
    Get GPU information using nvidia-smi command directly.
    
    Returns:
        dict: GPU information including temperature, memory usage, etc.
    """
    try:
        # First check if nvidia-smi is available
        if platform.system() == "Windows":
            # Check common paths for nvidia-smi on Windows
            nvidia_smi_paths = [
                "C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe",
                "C:\\Windows\\System32\\nvidia-smi.exe",
                "nvidia-smi.exe"  # If in PATH
            ]
            
            nvidia_smi_path = None
            for path in nvidia_smi_paths:
                if os.path.exists(path) or os.system(f"where {path} >nul 2>&1") == 0:
                    nvidia_smi_path = path
                    break
                    
            if not nvidia_smi_path:
                return None
                
            # Get GPU info in JSON format
            result = subprocess.run(
                [nvidia_smi_path, "--query-gpu=name,temperature.gpu,memory.used,memory.total,utilization.gpu,power.draw", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=False
            )
        else:
            # Linux/Mac
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,temperature.gpu,memory.used,memory.total,utilization.gpu,power.draw", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=False
            )
            
        if result.returncode != 0:
            return None
            
        # Parse the CSV output
        output = result.stdout.strip()
        if not output:
            return None
            
        # Format is: name, temp, memory_used, memory_total, utilization, power_draw
        parts = output.split(", ")
        if len(parts) >= 5:  # At least name, temp, memory_used, memory_total, utilization
            return {
                'name': parts[0],
                'temperature': float(parts[1]) if parts[1] and parts[1] != 'N/A' else 'N/A',
                'memory_used': float(parts[2]) if parts[2] and parts[2] != 'N/A' else 0,
                'memory_total': float(parts[3]) if parts[3] and parts[3] != 'N/A' else 1,
                'utilization': float(parts[4]) if parts[4] and parts[4] != 'N/A' else 0,
                'power_draw': float(parts[5]) if len(parts) > 5 and parts[5] and parts[5] != 'N/A' else 'N/A'
            }
    except Exception as e:
        print(f"Error getting GPU info from nvidia-smi: {e}")
        return None

def get_windows_gpu_info_wmi():
    """
    Get GPU information using Windows Management Instrumentation (WMI).
    This is a Windows-specific approach when nvidia-smi is not available.
    
    Returns:
        dict: GPU information including name, memory, etc.
    """
    if platform.system() != "Windows":
        return None
        
    try:
        import wmi
        computer = wmi.WMI()
        gpu_info = computer.Win32_VideoController()[0]
        
        # WMI doesn't provide temperature, but we can get other info
        return {
            'name': gpu_info.Name,
            'temperature': 'N/A',  # WMI doesn't provide temperature
            'memory_used': 0,  # WMI doesn't provide memory usage
            'memory_total': int(gpu_info.AdapterRAM / 1024 / 1024) if hasattr(gpu_info, 'AdapterRAM') else 1,
            'utilization': 0,  # WMI doesn't provide utilization
            'power_draw': 'N/A'  # WMI doesn't provide power draw
        }
    except Exception as e:
        print(f"Error getting GPU info from WMI: {e}")
        return None

def get_best_gpu_info():
    """
    Try different methods to get GPU information.
    
    Returns:
        dict: GPU information or default values if no method works
    """
    
    # Try nvidia-smi
    nvidia_info = get_gpu_info_from_nvidia_smi()
    if nvidia_info:
        return nvidia_info
        
    # Try WMI on Windows
    wmi_info = get_windows_gpu_info_wmi()
    if wmi_info:
        return wmi_info
        
    # Default fallback values
    return {
        'name': 'GPU information unavailable',
        'temperature': 'N/A',
        'memory_used': 0,
        'memory_total': 1,
        'utilization': 0,
        'power_draw': 'N/A'
    }

if __name__ == "__main__":
    # Test the module
    gpu_info = get_best_gpu_info()
    print(json.dumps(gpu_info, indent=2))
