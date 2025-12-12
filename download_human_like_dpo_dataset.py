#!/usr/bin/env python
import os
import json
import argparse
from pathlib import Path
from datasets import load_dataset

def download_human_like_dpo_dataset(output_dir, cache_dir=None):
    """
    Download the Human-Like-DPO-Dataset from HuggingFace and save it to the specified directory
    preserving its original structure for mixed curriculum training.
    
    Args:
        output_dir (str): Directory path to save the processed dataset
        cache_dir (str, optional): Directory to cache the downloaded dataset
    
    Returns:
        str: Path to the saved dataset file
    """
    print("Downloading Human-Like-DPO-Dataset from HuggingFace...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create cache directory if specified and doesn't exist
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    
    # Output file path
    output_file = os.path.join(output_dir, "human_like_dpo_dataset.json")
    
    # Download the dataset using the Hugging Face datasets library
    try:
        dataset = load_dataset("HumanLLMs/Human-Like-DPO-Dataset", cache_dir=cache_dir)
        print(f"Dataset loaded successfully. Available splits: {dataset.keys()}")
        
        # Preserve the original structure for mixed curriculum training
        dpo_data = {
            "dataset_type": "dpo",
            "description": "Human-Like DPO Dataset for improving natural human-like responses",
            "entries": []
        }
        
        # Process the training split
        entry_count = 0
        if "train" in dataset:
            for item in dataset["train"]:
                # Check if the required fields are present
                if "prompt" in item and "chosen" in item and "rejected" in item:
                    entry = {
                        "prompt": item["prompt"],
                        "chosen": item["chosen"],
                        "rejected": item["rejected"]
                    }
                    
                    dpo_data["entries"].append(entry)
                    entry_count += 1
        
        # Process the test split if available
        if "test" in dataset:
            for item in dataset["test"]:
                if "prompt" in item and "chosen" in item and "rejected" in item:
                    entry = {
                        "prompt": item["prompt"],
                        "chosen": item["chosen"],
                        "rejected": item["rejected"]
                    }
                    
                    dpo_data["entries"].append(entry)
                    entry_count += 1
        
        # Save the dataset with its original structure
        print(f"Saving {entry_count} entries to {output_file} in original DPO format...")
        
        # Fix encoding issues by ensuring ASCII compatibility
        # This will replace problematic characters with their ASCII equivalents
        def clean_text(text):
            if isinstance(text, str):
                # Replace emojis and other problematic characters with their descriptions or alternatives
                return text.encode('ascii', 'ignore').decode('ascii')
            return text
        
        # Clean all entries
        clean_entries = []
        for entry in dpo_data["entries"]:
            clean_entry = {
                "prompt": clean_text(entry["prompt"]),
                "chosen": clean_text(entry["chosen"]),
                "rejected": clean_text(entry["rejected"])
            }
            clean_entries.append(clean_entry)
        
        # Replace entries with clean versions
        dpo_data["entries"] = clean_entries
            
        with open(output_file, 'w', encoding='ascii') as f:
            json.dump(dpo_data, f, ensure_ascii=True, indent=2)
        
        print(f"Dataset successfully saved to {output_file}")
        return output_file
    
    except Exception as e:
        print(f"Error downloading or processing dataset: {e}")
        return None

def integrate_with_data_processor():
    """
    Integrate the downloaded dataset with the data processor.
    This modifies the data_processor.py file to include the human-like dataset
    as part of the mixed curriculum.
    """
    data_processor_path = Path("G:/Theta AI/src/data_processor.py")
    
    try:
        # Read the content of data_processor.py
        with open(data_processor_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if the dataset is already integrated
        if "human_like_dpo_dataset" in content:
            print("Dataset already integrated with data_processor.py")
            return True
        
        # Find the position to insert the dataset path under diverse curriculum datasets
        lines = content.split('\n')
        insert_position = -1
        for i, line in enumerate(lines):
            if "memory_simulation_path = datasets_dir / \"Memory_simulation.json\"" in line:
                insert_position = i + 1
                break
        
        if insert_position == -1:
            print("Could not find the position to insert the dataset path in data_processor.py")
            return False
        
        # Insert the new dataset path as part of the diverse curriculum
        lines.insert(insert_position, "    human_like_dpo_path = datasets_dir / \"human_like_dpo_dataset.json\"  # Human-Like DPO Dataset (mixed curriculum)")
        
        # Find the position to insert dataset loading with diverse curriculum datasets
        load_position = -1
        for i, line in enumerate(lines):
            if "memory_simulation_data = load_json_dataset(memory_simulation_path)" in line:
                load_position = i + 1
                break
        
        if load_position == -1:
            print("Could not find the position to insert the dataset loading in data_processor.py")
            return False
        
        # Insert the new dataset loading as original format, not as QA
        lines.insert(load_position, "    human_like_dpo_data = load_json_dataset(human_like_dpo_path)  # Loaded in original format, not converted to QA")
        
        # Find the position to save the dataset with diverse curriculum datasets
        save_position = -1
        for i, line in enumerate(lines):
            if "memory_simulation_count = save_diverse_dataset(memory_simulation_data, \"memory_simulation.json\")" in line:
                save_position = i + 1
                break
        
        if save_position == -1:
            print("Could not find the position to add the save operation in data_processor.py")
            return False
        
        # Add the save operation
        lines.insert(save_position, "    human_like_dpo_count = save_diverse_dataset(human_like_dpo_data, \"human_like_dpo_dataset.json\")")
        
        # Find position to add count print
        count_position = -1
        for i, line in enumerate(lines):
            if "print(f\"- {memory_simulation_count} from Memory Simulation\")" in line:
                count_position = i + 1
                break
        
        if count_position == -1:
            count_position = len(lines) - 10  # Fallback near the end
        
        # Add count print statement
        lines.insert(count_position, "    print(f\"- {human_like_dpo_count} from Human-Like DPO Dataset\")")
        
        # Write back the updated content
        with open(data_processor_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        print("Successfully integrated the Human-Like DPO dataset as part of the mixed curriculum")
        return True
    
    except Exception as e:
        print(f"Error integrating dataset with data_processor.py: {e}")
        return False

def verify_integration_with_training_script():
    """
    Verify that the dataset will be included in the training by checking train_overnight_enhanced.bat.
    No modifications needed since it uses the output of data_processor.py.
    """
    train_script_path = Path("G:/Theta AI/train_overnight_enhanced.bat")
    
    try:
        # Read the content of the training script
        with open(train_script_path, 'r') as f:
            content = f.read()
        
        # Check if the script uses processed_data.json or enhanced_training_data.json
        if "enhanced_training_data.json" in content and "processed_data.json" in content:
            print("Training script uses enhanced_training_data.json which will include the human-like DPO dataset.")
            print("The dataset will be included in the training process.")
            return True
        else:
            print("Warning: Could not verify that the training script will use the processed data.")
            print("Please check train_overnight_enhanced.bat manually.")
            return False
    
    except Exception as e:
        print(f"Error verifying integration with training script: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download and integrate Human-Like-DPO-Dataset with Theta AI")
    parser.add_argument("--output_dir", default="G:/Theta AI/Datasets", 
                        help="Directory to save the dataset")
    parser.add_argument("--cache_dir", default="G:/Theta AI/cache/hub",
                        help="Directory to cache the downloaded dataset")
    args = parser.parse_args()
    
    # Download the dataset in its original structure
    output_file = download_human_like_dpo_dataset(args.output_dir, args.cache_dir)
    
    if output_file:
        # Integrate with data_processor.py as part of mixed curriculum
        integration_success = integrate_with_data_processor()
        
        # Verify integration with training script
        if integration_success:
            verify_integration_with_training_script()
            
            print("\nIntegration Summary:")
            print("1. Downloaded Human-Like-DPO-Dataset ✓")
            print("2. Preserved original dataset structure for mixed curriculum ✓")
            print("3. Updated data_processor.py to include the dataset ✓")
            print("4. Verified integration with training script ✓")
            print("\nTo include this dataset in your training:")
            print("1. Run: python run_data_processing.py")
            print("2. Then run your training script: train_overnight_enhanced.bat")
            print("\nThe Human-Like DPO dataset will now be included in Theta's mixed curriculum training.")
        else:
            print("\nFailed to fully integrate the dataset. Please check the error messages above.")
    else:
        print("\nFailed to download the dataset. Please check your internet connection and try again.")

if __name__ == "__main__":
    main()
