"""
Data augmentation techniques for Theta AI training.
Provides functions to expand datasets using paraphrasing, back-translation,
and noise injection techniques.
"""

import json
import random
import re
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
from transformers import MarianMTModel, MarianTokenizer
import torch

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class DataAugmenter:
    """Class for augmenting question-answer datasets."""
    
    def __init__(self, device=None):
        """
        Initialize the data augmenter.
        
        Args:
            device: Device to run models on ('cuda' or 'cpu')
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.back_translation_models = {}
        
    def load_back_translation_models(self, target_languages=['de', 'fr', 'es']):
        """
        Load models for back translation.
        
        Args:
            target_languages: List of language codes to use for back translation
        """
        print("Loading back-translation models. This may take a few minutes...")
        
        # Load models for translation to target languages and back to English
        for lang in target_languages:
            # Load English -> Target language model
            en_to_lang_name = f"Helsinki-NLP/opus-mt-en-{lang}"
            try:
                en_to_lang_model = MarianMTModel.from_pretrained(en_to_lang_name).to(self.device)
                en_to_lang_tokenizer = MarianTokenizer.from_pretrained(en_to_lang_name)
                
                # Load Target language -> English model
                lang_to_en_name = f"Helsinki-NLP/opus-mt-{lang}-en"
                lang_to_en_model = MarianMTModel.from_pretrained(lang_to_en_name).to(self.device)
                lang_to_en_tokenizer = MarianTokenizer.from_pretrained(lang_to_en_name)
                
                # Store models and tokenizers
                self.back_translation_models[lang] = {
                    'en_to_lang': {
                        'model': en_to_lang_model,
                        'tokenizer': en_to_lang_tokenizer
                    },
                    'lang_to_en': {
                        'model': lang_to_en_model,
                        'tokenizer': lang_to_en_tokenizer
                    }
                }
                print(f"Successfully loaded models for {lang}")
            except Exception as e:
                print(f"Failed to load models for {lang}: {e}")
                
    def paraphrase_by_synonym_replacement(self, text, replacement_prob=0.15):
        """
        Paraphrase text by replacing words with synonyms.
        
        Args:
            text: Input text to paraphrase
            replacement_prob: Probability of replacing each word
            
        Returns:
            Paraphrased text
        """
        try:
            from nltk.corpus import wordnet
        except LookupError:
            nltk.download('wordnet')
            from nltk.corpus import wordnet
            
        words = word_tokenize(text)
        new_words = []
        
        for word in words:
            # Only replace with some probability and skip short words
            if random.random() < replacement_prob and len(word) > 3:
                synonyms = []
                
                # Find synonyms
                for syn in wordnet.synsets(word):
                    for lemma in syn.lemmas():
                        synonym = lemma.name().replace("_", " ")
                        if synonym != word and len(synonym) > 2:
                            synonyms.append(synonym)
                            
                # Replace with a synonym if found
                if synonyms:
                    new_words.append(random.choice(synonyms))
                else:
                    new_words.append(word)
            else:
                new_words.append(word)
                
        return " ".join(new_words)
    
    def back_translate(self, text, language='de'):
        """
        Paraphrase text through back translation.
        
        Args:
            text: Input text
            language: Target language for back translation
            
        Returns:
            Back-translated text
        """
        # Check if models are loaded
        if not self.back_translation_models:
            print("Back translation models not loaded. Loading models...")
            self.load_back_translation_models([language])
            
        if language not in self.back_translation_models:
            print(f"No model loaded for language {language}")
            return text
            
        # Get models and tokenizers
        en_to_lang = self.back_translation_models[language]['en_to_lang']
        lang_to_en = self.back_translation_models[language]['lang_to_en']
        
        # Translate to target language
        en_to_lang_inputs = en_to_lang['tokenizer'](text, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            en_to_lang_outputs = en_to_lang['model'].generate(**en_to_lang_inputs)
        intermediate_text = en_to_lang['tokenizer'].decode(en_to_lang_outputs[0], skip_special_tokens=True)
        
        # Translate back to English
        lang_to_en_inputs = lang_to_en['tokenizer'](intermediate_text, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            lang_to_en_outputs = lang_to_en['model'].generate(**lang_to_en_inputs)
        back_translated_text = lang_to_en['tokenizer'].decode(lang_to_en_outputs[0], skip_special_tokens=True)
        
        return back_translated_text
        
    def add_noise(self, text, typo_prob=0.05, swap_prob=0.05):
        """
        Add noise to text by introducing typos and swapping words.
        
        Args:
            text: Input text
            typo_prob: Probability of introducing a typo in each word
            swap_prob: Probability of swapping adjacent words
            
        Returns:
            Text with added noise
        """
        words = word_tokenize(text)
        new_words = []
        
        # Introduce typos
        for word in words:
            if len(word) > 3 and random.random() < typo_prob:
                # Choose a random typo operation
                operation = random.choice(['insert', 'delete', 'replace'])
                
                if operation == 'insert' and len(word) > 0:
                    pos = random.randint(0, len(word))
                    char = random.choice('abcdefghijklmnopqrstuvwxyz')
                    word = word[:pos] + char + word[pos:]
                elif operation == 'delete' and len(word) > 3:
                    pos = random.randint(0, len(word) - 1)
                    word = word[:pos] + word[pos+1:]
                elif operation == 'replace' and len(word) > 0:
                    pos = random.randint(0, len(word) - 1)
                    char = random.choice('abcdefghijklmnopqrstuvwxyz')
                    word = word[:pos] + char + word[pos+1:]
                    
            new_words.append(word)
            
        # Swap adjacent words
        i = 0
        while i < len(new_words) - 1:
            if random.random() < swap_prob:
                new_words[i], new_words[i+1] = new_words[i+1], new_words[i]
                i += 2
            else:
                i += 1
                
        return " ".join(new_words)
        
    def augment_dataset(self, data_path, output_path, augmentation_factor=2, 
                        techniques=['synonym', 'noise', 'backtranslation']):
        """
        Augment a dataset with multiple techniques.
        
        Args:
            data_path: Path to the input dataset JSON file
            output_path: Path to save the augmented dataset
            augmentation_factor: How many times to augment each example
            techniques: List of augmentation techniques to use
            
        Returns:
            Number of examples in the augmented dataset
        """
        # Load dataset
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        augmented_data = data.copy()
        original_count = len(data)
        
        print(f"Original dataset: {original_count} examples")
        print(f"Applying augmentation with factor {augmentation_factor} using techniques: {techniques}")
        
        # Apply augmentation techniques
        for i, item in enumerate(data):
            question = item['question']
            answer = item['answer']
            
            # Create augmented examples
            for _ in range(augmentation_factor):
                # Choose a random technique for each field
                q_technique = random.choice(techniques)
                a_technique = random.choice(['synonym', 'noise'])  # Be more conservative with answers
                
                # Apply techniques to question
                if q_technique == 'synonym':
                    new_question = self.paraphrase_by_synonym_replacement(question)
                elif q_technique == 'backtranslation':
                    new_question = self.back_translate(question, language=random.choice(['de', 'fr', 'es']))
                elif q_technique == 'noise':
                    new_question = self.add_noise(question)
                else:
                    new_question = question
                    
                # Apply techniques to answer
                if a_technique == 'synonym':
                    new_answer = self.paraphrase_by_synonym_replacement(answer, replacement_prob=0.1)
                elif a_technique == 'noise':
                    new_answer = self.add_noise(answer, typo_prob=0.03, swap_prob=0.03)
                else:
                    new_answer = answer
                    
                # Add augmented example to dataset
                augmented_data.append({
                    'question': new_question,
                    'answer': new_answer,
                    'augmented': True,
                    'original_idx': i
                })
                
            # Print progress
            if (i + 1) % 20 == 0:
                print(f"Processed {i + 1}/{original_count} examples")
                
        # Save augmented dataset
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(augmented_data, f, indent=2)
            
        print(f"Augmented dataset saved with {len(augmented_data)} examples")
        return len(augmented_data)

def main():
    """Test data augmentation functionality."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Augment a dataset for Theta AI training")
    parser.add_argument("--input", type=str, required=True, help="Path to input dataset")
    parser.add_argument("--output", type=str, required=True, help="Path to save augmented dataset")
    parser.add_argument("--factor", type=int, default=2, help="Augmentation factor")
    parser.add_argument("--techniques", nargs='+', 
                       default=['synonym', 'noise', 'backtranslation'],
                       help="Augmentation techniques to use")
    
    args = parser.parse_args()
    
    augmenter = DataAugmenter()
    augmenter.augment_dataset(args.input, args.output, args.factor, args.techniques)

if __name__ == "__main__":
    main()
