"""
Structured Data Adapter for Theta AI

This module allows conversion between structured hierarchical data formats 
and the standard Q&A format required by the data processors.
"""

import json
from pathlib import Path

def nested_to_qa(data):
    """
    Convert nested JSON structure to a list of question-answer pairs.
    
    Args:
        data: A nested JSON structure containing personal information
        
    Returns:
        list: A list of dictionaries with 'question' and 'answer' keys
    """
    qa_pairs = []
    
    # Process Dakota's information
    if 'Dakota' in data:
        dakota = data['Dakota']
        
        # Basic information
        if 'birthday' in dakota:
            qa_pairs.append({
                "question": "When is Dakota's birthday?",
                "answer": f"Dakota's birthday is {dakota['birthday']}."
            })
        
        if 'zodiac' in dakota:
            qa_pairs.append({
                "question": "What is Dakota's zodiac sign?",
                "answer": f"Dakota is a {dakota['zodiac']}."
            })
            
        if 'birthplace' in dakota:
            qa_pairs.append({
                "question": "Where was Dakota born?",
                "answer": f"Dakota was born in {dakota['birthplace']}."
            })
        
        # Favorite numbers
        if 'favorite_numbers' in dakota:
            numbers = dakota['favorite_numbers'].get('numbers', [])
            significance = dakota['favorite_numbers'].get('significance', '')
            
            if numbers:
                qa_pairs.append({
                    "question": "What are Dakota's favorite numbers?",
                    "answer": f"Dakota's favorite numbers are {', '.join(map(str, numbers))}."
                })
                
            if significance:
                qa_pairs.append({
                    "question": "What is the significance of Dakota's favorite numbers?",
                    "answer": significance
                })
        
        # Favorite colors
        if 'favorite_colors' in dakota:
            colors = dakota['favorite_colors']
            if colors:
                qa_pairs.append({
                    "question": "What are Dakota's favorite colors?",
                    "answer": f"Dakota's favorite colors are {' and '.join(colors)}."
                })
        
        # Personality traits
        if 'personality_traits' in dakota:
            traits = dakota['personality_traits']
            if traits:
                qa_pairs.append({
                    "question": "What are Dakota's personality traits?",
                    "answer": f"Dakota's personality traits include: {', '.join(traits)}."
                })
        
        # Notes
        if 'notes' in dakota and dakota['notes']:
            qa_pairs.append({
                "question": "What additional information is known about Dakota?",
                "answer": dakota['notes']
            })
    
    # Process Shyanne's information
    if 'Shyanne' in data:
        shyanne = data['Shyanne']
        
        # Basic information
        if 'birthday' in shyanne:
            qa_pairs.append({
                "question": "When is Shyanne's birthday?",
                "answer": f"Shyanne's birthday is {shyanne['birthday']}."
            })
        
        if 'zodiac' in shyanne:
            qa_pairs.append({
                "question": "What is Shyanne's zodiac sign?",
                "answer": f"Shyanne is a {shyanne['zodiac']}."
            })
            
        if 'birthplace' in shyanne:
            qa_pairs.append({
                "question": "Where is Shyanne from?",
                "answer": f"Shyanne is from {shyanne['birthplace']}."
            })
        
        # Favorite numbers
        if 'favorite_numbers' in shyanne:
            numbers = shyanne['favorite_numbers'].get('numbers', [])
            significance = shyanne['favorite_numbers'].get('significance', '')
            
            if numbers:
                qa_pairs.append({
                    "question": "What are Shyanne's favorite numbers?",
                    "answer": f"Shyanne's favorite numbers are {', '.join(map(str, numbers))}."
                })
                
            if significance:
                qa_pairs.append({
                    "question": "What is the significance of Shyanne's favorite numbers?",
                    "answer": significance
                })
        
        # Favorite colors
        if 'favorite_colors' in shyanne:
            colors = shyanne['favorite_colors']
            if colors:
                qa_pairs.append({
                    "question": "What are Shyanne's favorite colors?",
                    "answer": f"Shyanne's favorite colors are {' and '.join(colors)}."
                })
        
        # Personality traits
        if 'personality_traits' in shyanne:
            traits = shyanne['personality_traits']
            if traits:
                qa_pairs.append({
                    "question": "What are Shyanne's personality traits?",
                    "answer": f"Shyanne's personality traits include: {', '.join(traits)}."
                })
        
        # Notes
        if 'notes' in shyanne and shyanne['notes']:
            qa_pairs.append({
                "question": "What additional information is known about Shyanne?",
                "answer": shyanne['notes']
            })
    
    # Process relationship information
    if 'relationship' in data:
        relationship = data['relationship']
        
        # First kiss
        if 'first_kiss' in relationship:
            first_kiss = relationship['first_kiss']
            date = first_kiss.get('date', '')
            notes = first_kiss.get('notes', '')
            
            if date:
                qa_pairs.append({
                    "question": "When was Dakota and Shyanne's first kiss?",
                    "answer": f"Dakota and Shyanne's first kiss was on {date}."
                })
            
            if notes:
                qa_pairs.append({
                    "question": "What happened during Dakota and Shyanne's first kiss?",
                    "answer": notes
                })
        
        # Shared preferences
        if 'shared_preferences' in relationship:
            shared = relationship['shared_preferences']
            
            if 'colors' in shared and shared['colors']:
                qa_pairs.append({
                    "question": "What colors do both Dakota and Shyanne like?",
                    "answer": f"They both share a preference for {', '.join(shared['colors'])}."
                })
            
            if 'notes' in shared and shared['notes']:
                qa_pairs.append({
                    "question": "What is significant about Dakota and Shyanne's shared preferences?",
                    "answer": shared['notes']
                })
        
        # Significant moments
        if 'significant_moments' in relationship:
            moments = relationship['significant_moments']
            for i, moment in enumerate(moments):
                event = moment.get('event', '')
                details = moment.get('details', '')
                date = moment.get('date', '')
                
                if event and details:
                    date_prefix = f"On {date}, " if date else ""
                    qa_pairs.append({
                        "question": f"What is significant about the {event} in Dakota and Shyanne's relationship?",
                        "answer": f"{date_prefix}{details}"
                    })
        
        # Relationship dynamics
        if 'relationship_dynamics' in relationship:
            dynamics = relationship['relationship_dynamics']
            
            if 'Dakota_role' in dynamics:
                qa_pairs.append({
                    "question": "What role does Dakota play in the relationship?",
                    "answer": dynamics['Dakota_role']
                })
                
            if 'Shyanne_role' in dynamics:
                qa_pairs.append({
                    "question": "What role does Shyanne play in the relationship?",
                    "answer": dynamics['Shyanne_role']
                })
        
        # Connection notes
        if 'connection_notes' in relationship:
            qa_pairs.append({
                "question": "How would you describe Dakota and Shyanne's connection?",
                "answer": relationship['connection_notes']
            })
    
    # Combined summary
    if 'combined_summary' in data:
        qa_pairs.append({
            "question": "Summarize Dakota and Shyanne's relationship",
            "answer": data['combined_summary']
        })
        
        qa_pairs.append({
            "question": "Who are Dakota and Shyanne?",
            "answer": data['combined_summary']
        })
    
    return qa_pairs

def update_json_file(file_path, keep_original=True):
    """
    Updates a JSON file by converting nested structure to QA format
    
    Args:
        file_path: Path to the JSON file
        keep_original: If True, saves original data with .original extension
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load the original data
        with open(file_path, 'r') as f:
            original_data = json.load(f)
        
        # Keep a backup of the original data
        if keep_original:
            backup_path = f"{file_path}.original"
            with open(backup_path, 'w') as f:
                json.dump(original_data, f, indent=2)
                print(f"Original data saved to {backup_path}")
        
        # Convert the nested structure to QA format
        qa_data = nested_to_qa(original_data)
        
        # Save the QA format data
        qa_path = file_path
        with open(qa_path, 'w') as f:
            json.dump(qa_data, f, indent=2)
            print(f"Converted QA pairs saved to {qa_path}")
        
        return True
    except Exception as e:
        print(f"Error updating JSON file: {e}")
        return False

def convert_file_from_cli():
    """Command line interface for converting a single file"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert nested JSON to QA format')
    parser.add_argument('file_path', help='Path to the JSON file to convert')
    parser.add_argument('--no-backup', action='store_true', help='Do not keep original data')
    
    args = parser.parse_args()
    
    success = update_json_file(args.file_path, not args.no_backup)
    return 0 if success else 1

if __name__ == "__main__":
    import sys
    sys.exit(convert_file_from_cli())
