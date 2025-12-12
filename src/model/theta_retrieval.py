import json
import os
from pathlib import Path
import numpy as np
from collections import defaultdict
import re

class ThetaRetrieval:
    """
    Retrieval-based question answering for Theta AI.
    This class handles direct retrieval of answers from the dataset.
    """
    
    def __init__(self, dataset_path=None):
        """
        Initialize the retrieval system.
        
        Args:
            dataset_path: Path to the processed dataset JSON file
        """
        if dataset_path is None:
            # Default path
            project_dir = Path(os.path.abspath(__file__)).parent.parent.parent
            dataset_path = project_dir / "Datasets" / "processed_data.json"
        
        self.qa_pairs = self._load_dataset(dataset_path)
        self.categories = self._categorize_qa_pairs()
        
    def _load_dataset(self, dataset_path):
        """Load the QA dataset from a JSON file."""
        try:
            with open(dataset_path, 'r') as file:
                return json.load(file)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return []
            
    def _categorize_qa_pairs(self):
        """Categorize QA pairs by topic for better retrieval."""
        categories = defaultdict(list)
        
        # Define keywords for each category
        category_keywords = {
            "frostline": ["frostline", "ceo", "headquarter", "texas", "dakota", "devin", "fox", "fryberger", "copperas"],
            "cybersecurity": ["security", "cyber", "defense", "depth", "firewall", "penetration", "zero", "trust", 
                            "vulnerability", "ransomware", "phishing", "authentication", "ids", "ips"],
            "software": ["software", "programming", "api", "rest", "code", "development", "object", "version", 
                        "continuous", "integration", "docker", "test", "agile", "full-stack"],
            "hardware": ["hardware", "gpu", "cpu", "ram", "nvidia", "amd", "rtx", "power", "supply", "graphics"],
            "it_support": ["troubleshoot", "network", "computer", "slow", "recover", "files", "clean", "boot", "replace"]
        }
        
        # Categorize each QA pair
        for qa_pair in self.qa_pairs:
            question = qa_pair["question"].lower()
            answer = qa_pair["answer"].lower()
            text = question + " " + answer
            
            assigned = False
            for category, keywords in category_keywords.items():
                for keyword in keywords:
                    if keyword.lower() in text:
                        categories[category].append(qa_pair)
                        assigned = True
                        break
                if assigned:
                    break
                    
            # If not assigned to any category, put in general
            if not assigned:
                categories["general"].append(qa_pair)
                
        return categories
    
    def _calculate_similarity(self, query, text):
        """
        Calculate a simple keyword-based similarity score between query and text.
        
        Args:
            query: The user's question
            text: Text to compare against (question or answer)
            
        Returns:
            Similarity score (higher is better match)
        """
        query = query.lower()
        text = text.lower()
        
        # Clean and tokenize
        query_words = set(re.findall(r'\w+', query))
        text_words = set(re.findall(r'\w+', text))
        
        # Calculate overlap
        common_words = query_words.intersection(text_words)
        
        if not query_words:
            return 0
            
        # Calculate similarity score based on word overlap
        direct_match_score = len(common_words) / len(query_words)
        
        # Boost exact question matches
        if query in text or text in query:
            direct_match_score += 0.5
            
        return direct_match_score
    
    def retrieve_answer(self, query, top_k=3):
        """
        Retrieve the best answer for the query from the dataset.
        
        Args:
            query: The user's question
            top_k: Number of top matches to consider
            
        Returns:
            Best matching answer, or None if no good match found
        """
        # First, try to identify the category
        query_lower = query.lower()
        
        # Initialize variables to track best matches
        best_matches = []
        
        # Check all categories for potential answers
        for category, qa_pairs in self.categories.items():
            for qa_pair in qa_pairs:
                # Calculate similarity to question
                question_similarity = self._calculate_similarity(query, qa_pair["question"])
                
                # Also check similarity to answer (sometimes the question might be phrased differently)
                answer_similarity = self._calculate_similarity(query, qa_pair["answer"]) * 0.5  # Less weight for answer similarity
                
                # Combined similarity score
                similarity = question_similarity + answer_similarity
                
                best_matches.append((similarity, qa_pair))
        
        # Sort by similarity score (descending)
        best_matches.sort(key=lambda x: x[0], reverse=True)
        
        # Take top-k matches
        top_matches = best_matches[:top_k]
        
        # Return the answer if we have a good match
        if top_matches and top_matches[0][0] > 0.3:  # Threshold for a good match
            return top_matches[0][1]["answer"]
            
        # If we have multiple decent matches, combine them
        if len(top_matches) > 1 and top_matches[0][0] > 0.2:
            combined_answer = "Based on available information:\n\n"
            for i, (score, qa_pair) in enumerate(top_matches[:3], 1):
                if score > 0.2:  # Only include reasonably good matches
                    combined_answer += f"{qa_pair['answer']}\n\n"
            return combined_answer
            
        # No good match found
        return None
        
    def get_all_questions(self):
        """Get all available questions in the dataset."""
        return [qa_pair["question"] for qa_pair in self.qa_pairs]


# For testing
if __name__ == "__main__":
    retriever = ThetaRetrieval()
    test_questions = [
        "What is Frostline?",
        "Where is Frostline headquartered?",
        "What is defense in depth?",
        "How do I troubleshoot a slow computer?"
    ]
    
    for question in test_questions:
        print(f"Q: {question}")
        answer = retriever.retrieve_answer(question)
        print(f"A: {answer}\n")
