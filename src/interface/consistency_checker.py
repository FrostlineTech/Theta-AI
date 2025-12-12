"""
Consistency Checker for Theta AI.
Checks responses for consistency with previous statements in a conversation.
"""

import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define our own simple sentence tokenizer that doesn't use NLTK at all
def custom_sent_tokenize(text):
    """Custom sentence tokenizer that doesn't depend on NLTK"""
    if not text:
        return []
    
    # Simple tokenizer using regex
    sentences = []
    # Split on ., !, ? followed by whitespace or end of string
    for sent in re.split(r'(?<=[.!?])\s+', text):
        if sent.strip():
            sentences.append(sent.strip())
    
    # If no sentences were found, return the entire text as one sentence
    if not sentences and text.strip():
        return [text.strip()]
    return sentences

class ConsistencyChecker:
    """
    Checks responses for consistency with previous statements.
    """
    
    def __init__(self):
        """Initialize consistency checker."""
        # Store key facts from conversation
        self.facts = {}
        # Track contradictions
        self.contradictions = []
    
    def extract_facts(self, text):
        """
        Extract simple facts from text.
        Focuses on "X is Y" and "X has Y" statements.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Extracted facts
        """
        if not text:
            return {}
        
        facts = {}
        
        # Split into sentences using our custom tokenizer
        sentences = custom_sent_tokenize(text)
        
        for sentence in sentences:
            # Extract "X is Y" facts
            is_matches = re.findall(r'(\w+(?:\s+\w+){0,3})\s+(?:is|are|was|were)\s+(\w+(?:\s+\w+){0,5})', sentence)
            for subject, attribute in is_matches:
                subject = subject.lower().strip()
                if subject not in facts:
                    facts[subject] = []
                facts[subject].append(('is', attribute.lower().strip()))
            
            # Extract "X has Y" facts
            has_matches = re.findall(r'(\w+(?:\s+\w+){0,3})\s+(?:has|have|had)\s+(\w+(?:\s+\w+){0,5})', sentence)
            for subject, attribute in has_matches:
                subject = subject.lower().strip()
                if subject not in facts:
                    facts[subject] = []
                facts[subject].append(('has', attribute.lower().strip()))
        
        return facts
    
    def update_facts(self, text):
        """
        Update tracked facts with new text.
        
        Args:
            text (str): New text to analyze
        """
        new_facts = self.extract_facts(text)
        
        # Merge new facts with existing ones
        for subject, attributes in new_facts.items():
            if subject not in self.facts:
                self.facts[subject] = []
            
            for attr_type, attribute in attributes:
                self.facts[subject].append((attr_type, attribute))
    
    def check_consistency(self, new_response):
        """
        Check if new response is consistent with previous facts.
        
        Args:
            new_response (str): New response to check
            
        Returns:
            bool: True if consistent, False if contradictions found
            list: List of contradictions if any
        """
        # Reset contradictions
        self.contradictions = []
        
        # Extract facts from new response
        new_facts = self.extract_facts(new_response)
        
        # Check for contradictions
        for subject, attributes in new_facts.items():
            if subject in self.facts:
                # Check new attributes against existing ones
                for attr_type, attribute in attributes:
                    for existing_type, existing_attr in self.facts[subject]:
                        # Only compare same types of attributes
                        if attr_type == existing_type:
                            # Check for opposite statements
                            if self._are_contradictory(attribute, existing_attr):
                                contradiction = f"'{subject} {attr_type} {attribute}' contradicts '{subject} {existing_type} {existing_attr}'"
                                self.contradictions.append(contradiction)
        
        # Update facts after checking
        self.update_facts(new_response)
        
        return len(self.contradictions) == 0, self.contradictions
    
    def _are_contradictory(self, attr1, attr2):
        """
        Check if two attributes are contradictory.
        Simple implementation focusing on direct opposites.
        
        Args:
            attr1 (str): First attribute
            attr2 (str): Second attribute
            
        Returns:
            bool: True if contradictory
        """
        # Direct opposites
        opposites = {
            'true': 'false',
            'yes': 'no',
            'correct': 'incorrect',
            'right': 'wrong',
            'good': 'bad',
            'high': 'low',
            'big': 'small',
            'fast': 'slow',
            'hot': 'cold',
            'new': 'old',
            'open': 'closed',
            'positive': 'negative'
        }
        
        # Check both directions
        attr1 = attr1.lower().strip()
        attr2 = attr2.lower().strip()
        
        # Direct opposition
        if attr1 in opposites and opposites[attr1] == attr2:
            return True
        if attr2 in opposites and opposites[attr2] == attr1:
            return True
            
        # Check for negation
        if attr1 == f"not {attr2}" or attr2 == f"not {attr1}":
            return True
        
        # Simple opposite check (not comprehensive)
        if attr1.startswith("not ") and attr1[4:] == attr2:
            return True
        if attr2.startswith("not ") and attr2[4:] == attr1:
            return True
        
        return False
    
    def get_recent_facts(self, subject=None, limit=5):
        """
        Get recent facts, optionally filtered by subject.
        
        Args:
            subject (str, optional): Subject to filter by
            limit (int): Maximum number of facts to return
            
        Returns:
            list: Recent facts
        """
        facts_list = []
        
        if subject:
            # Get facts for specific subject
            if subject in self.facts:
                for attr_type, attribute in self.facts[subject]:
                    facts_list.append(f"{subject} {attr_type} {attribute}")
        else:
            # Get all facts
            for subject, attributes in self.facts.items():
                for attr_type, attribute in attributes:
                    facts_list.append(f"{subject} {attr_type} {attribute}")
        
        # Return most recent facts
        return facts_list[-limit:]
    
    def reset(self):
        """Reset tracked facts and contradictions."""
        self.facts = {}
        self.contradictions = []
