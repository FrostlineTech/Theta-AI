"""
Enhanced Factual Consistency Checker for Theta AI

This module provides improved factual consistency checking for Theta AI responses.
It verifies technical information, tracks claims, and prevents contradictions.
"""

import re
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set, Optional
import hashlib
import itertools

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedConsistencyChecker:
    """
    Enhanced consistency checker that verifies factual accuracy and prevents contradictions.
    Tracks claims across conversations and checks against trusted knowledge sources.
    """
    
    def __init__(self, datasets_dir: Path):
        """
        Initialize the enhanced consistency checker.
        
        Args:
            datasets_dir: Path to the datasets directory
        """
        self.datasets_dir = datasets_dir
        self.facts_store = {}
        self.claim_history = {}
        self.contradictions = {}
        
        # Create fact checking directory if it doesn't exist
        self.facts_dir = datasets_dir / "verified_facts"
        os.makedirs(self.facts_dir, exist_ok=True)
        
        # Load verified facts
        self.verified_facts = self._load_verified_facts()
        
        # Define knowledge domains
        self.domains = ["cybersecurity", "programming", "networking", 
                       "cloud_computing", "data_science", "hardware", "general_tech"]
        
        # Track entities by domain
        self.domain_entities = {domain: set() for domain in self.domains}
        
        # Define consistency rules
        self.consistency_rules = self._define_consistency_rules()
    
    def _load_verified_facts(self) -> Dict[str, List[Dict]]:
        """
        Load verified facts from the facts directory.
        
        Returns:
            Dictionary of verified facts by domain
        """
        verified_facts = {}
        
        try:
            # Check if directory exists
            if not self.facts_dir.exists():
                return verified_facts
                
            # Load all JSON files in the directory
            for file_path in self.facts_dir.glob("*.json"):
                try:
                    # Load with UTF-8 encoding and error handling
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                            facts = json.load(f)
                    except json.JSONDecodeError:
                        # If that fails, try a more aggressive approach with binary reading
                        logger.info(f"JSON decode error with utf-8 for {file_path}, trying alternative approach")
                        with open(file_path, 'rb') as f:
                            content = f.read()
                        
                        # Replace or remove non-UTF-8 bytes
                        cleaned_content = b''
                        for i in range(0, len(content)):
                            byte = content[i:i+1]
                            try:
                                byte.decode('utf-8')
                                cleaned_content += byte
                            except UnicodeDecodeError:
                                # Replace problematic bytes
                                cleaned_content += b'\xEF\xBF\xBD'  # UTF-8 replacement character
                        
                        # Try parsing the cleaned content
                        facts = json.loads(cleaned_content.decode('utf-8'))
                    
                    domain = file_path.stem
                    verified_facts[domain] = facts
                    logger.info(f"Loaded {len(facts)} verified facts for domain '{domain}'")
                    
                except Exception as e:
                    logger.error(f"Error loading verified facts from {file_path}: {e}")
        
        except Exception as e:
            logger.error(f"Error loading verified facts: {e}")
        
        return verified_facts
    
    def _define_consistency_rules(self) -> Dict:
        """
        Define consistency rules for different domains.
        
        Returns:
            Dictionary of consistency rules
        """
        return {
            "general": [
                {
                    "pattern": r"(.*) (is|are) both (.*) and (?:not|neither) (.*)",
                    "message": "Contradiction: something cannot be X and not X at the same time"
                },
                {
                    "pattern": r"(.*) (is|are|was|were) (.*) and.* (is|are|was|were) not (.*)",
                    "check": lambda match: match.group(3).lower() in match.group(5).lower() 
                                          or match.group(5).lower() in match.group(3).lower(),
                    "message": "Contradiction: claiming something both is and isn't a certain property"
                },
                {
                    "pattern": r"(?:first|earlier|previously|before).*?(?:claim|state|mention).*?(.*?),.*(?:now|then|later|but).*?(?:actually|however|instead|rather).*?(?:claim|state|mention).*?(.*)",
                    "check": lambda match: self._check_potential_contradiction(match.group(1), match.group(2)),
                    "message": "Potential contradiction between earlier and later statements"
                }
            ],
            "technical": [
                {
                    "pattern": r"([A-Za-z0-9_-]+) (algorithm|protocol|framework|language) has (O\([^)]+\)) time complexity.*([A-Za-z0-9_-]+) \1 has (O\([^)]+\)) time complexity",
                    "check": lambda match: match.group(3) != match.group(5),
                    "message": "Contradiction: claiming different time complexities for the same algorithm"
                },
                {
                    "pattern": r"([A-Za-z0-9_-]+) (operates|works) at (layer \d+|the [a-z]+ layer).*([A-Za-z0-9_-]+) \1 (operates|works) at (layer \d+|the [a-z]+ layer)",
                    "check": lambda match: match.group(3) != match.group(6),
                    "message": "Contradiction: claiming a protocol operates at different OSI layers"
                }
            ],
            "cybersecurity": [
                {
                    "pattern": r"(.*) (encryption|protocol) is (secure|not secure).*\1 \2 is (secure|not secure)",
                    "check": lambda match: match.group(3) != match.group(4),
                    "message": "Contradiction: claiming a security technology is both secure and not secure"
                }
            ],
            "versioning": [
                {
                    "pattern": r"version ([0-9.]+) of ([A-Za-z0-9_-]+) was released in (\d{4}).*version \1 of \2 was released in (\d{4})",
                    "check": lambda match: match.group(3) != match.group(4),
                    "message": "Contradiction: claiming different release years for the same version"
                }
            ]
        }
    
    def _check_potential_contradiction(self, statement1: str, statement2: str) -> bool:
        """
        Check if two statements potentially contradict each other.
        
        Args:
            statement1: First statement
            statement2: Second statement
            
        Returns:
            True if potential contradiction, False otherwise
        """
        # Very basic contradiction check - can be significantly improved with NLP
        # Just checking if statement2 contains negation of something in statement1
        words1 = set(re.findall(r'\b\w+\b', statement1.lower()))
        
        # Check for direct negation
        if "not" in statement2.lower():
            for word in words1:
                if word != "not" and f"not {word}" in statement2.lower():
                    return True
                    
        return False
    
    def extract_claims(self, text: str) -> List[Dict]:
        """
        Extract factual claims from text.
        
        Args:
            text: The text to extract claims from
            
        Returns:
            List of extracted claims
        """
        claims = []
        
        # Define claim patterns
        claim_patterns = [
            # Definition claims
            r"([A-Za-z0-9_\-]+) (?:is|are) (?:defined as|described as) ([^.]+)",
            r"([A-Za-z0-9_\-]+) refers to ([^.]+)",
            
            # Technical properties
            r"([A-Za-z0-9_\-]+) has (?:a|an) ([^.]+) (?:of|rate|level|complexity) ([^.]+)",
            r"([A-Za-z0-9_\-]+) (?:uses|supports|implements|requires) ([^.]+)",
            
            # Comparative claims
            r"([A-Za-z0-9_\-]+) is (?:faster|slower|better|worse|more secure|less secure) than ([A-Za-z0-9_\-]+)",
            
            # Factual claims
            r"([A-Za-z0-9_\-]+) was (?:created|developed|invented|designed) (?:in|by) ([^.]+)",
            r"([A-Za-z0-9_\-]+) (?:version|release) ([0-9.]+) was released in ([^.]+)"
        ]
        
        # Extract claims using patterns
        for pattern in claim_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                # Create claim structure
                claim = {
                    "text": match.group(0),
                    "subject": match.group(1),
                    "content": match.group(0),
                    "pattern": pattern,
                    "claim_id": self._generate_claim_id(match.group(0))
                }
                claims.append(claim)
        
        return claims
    
    def _generate_claim_id(self, claim_text: str) -> str:
        """Generate a unique ID for a claim based on its content."""
        return hashlib.md5(claim_text.encode()).hexdigest()
    
    def check_consistency(self, text: str, conversation_id: str = None) -> Tuple[bool, List[Dict]]:
        """
        Check the consistency of a text against facts and previous claims.
        
        Args:
            text: The text to check
            conversation_id: ID of the conversation (to track claims across exchanges)
            
        Returns:
            Tuple of (is_consistent, list of inconsistencies)
        """
        inconsistencies = []
        
        # If no conversation ID, generate one
        if conversation_id is None:
            conversation_id = f"conv_{hashlib.md5(text[:20].encode()).hexdigest()}"
        
        # Initialize conversation history if needed
        if conversation_id not in self.claim_history:
            self.claim_history[conversation_id] = []
        
        # Extract claims from text
        current_claims = self.extract_claims(text)
        
        # Add claims to history
        self.claim_history[conversation_id].extend(current_claims)
        
        # Check for internal contradictions using rules
        for rule_set in self.consistency_rules.values():
            for rule in rule_set:
                pattern = rule["pattern"]
                matches = re.finditer(pattern, text)
                
                for match in matches:
                    # If there's a custom check function, use it
                    if "check" in rule and callable(rule["check"]):
                        if rule["check"](match):
                            inconsistencies.append({
                                "type": "contradiction",
                                "text": match.group(0),
                                "message": rule["message"],
                                "severity": "high"
                            })
                    else:
                        # Otherwise, the pattern itself indicates inconsistency
                        inconsistencies.append({
                            "type": "contradiction",
                            "text": match.group(0),
                            "message": rule["message"],
                            "severity": "high"
                        })
        
        # Check claims against verified facts
        for claim in current_claims:
            # Find relevant domain for the claim
            relevant_domain = self._determine_claim_domain(claim)
            
            if relevant_domain and relevant_domain in self.verified_facts:
                # Check against verified facts
                for fact in self.verified_facts[relevant_domain]:
                    if claim["subject"].lower() == fact.get("subject", "").lower():
                        # Check if the claim contradicts the fact
                        if self._contradicts(claim["content"], fact.get("content", "")):
                            inconsistencies.append({
                                "type": "fact_contradiction",
                                "text": claim["text"],
                                "fact": fact.get("content", ""),
                                "message": f"Contradicts known fact about {claim['subject']}",
                                "severity": "high"
                            })
        
        # Check for contradictions in conversation history
        if len(self.claim_history[conversation_id]) > 1:
            for claim1, claim2 in itertools.combinations(self.claim_history[conversation_id], 2):
                if claim1["subject"].lower() == claim2["subject"].lower():
                    if self._contradicts(claim1["content"], claim2["content"]):
                        inconsistencies.append({
                            "type": "historical_contradiction",
                            "text1": claim1["text"],
                            "text2": claim2["text"],
                            "message": f"Contradicts previous statement about {claim1['subject']}",
                            "severity": "medium"
                        })
        
        # Return consistency result
        is_consistent = len(inconsistencies) == 0
        return is_consistent, inconsistencies
    
    def _determine_claim_domain(self, claim: Dict) -> Optional[str]:
        """
        Determine the domain for a claim based on its subject and content.
        
        Args:
            claim: The claim to determine domain for
            
        Returns:
            Domain name or None if no domain detected
        """
        content = claim["content"].lower()
        subject = claim["subject"].lower()
        
        # Domain-specific keywords
        domain_keywords = {
            "cybersecurity": ["security", "attack", "vulnerability", "threat", "malware", 
                             "encryption", "firewall", "authentication", "hacker", "phishing"],
            "programming": ["code", "language", "function", "variable", "class", "algorithm", 
                           "compiler", "framework", "library", "api"],
            "networking": ["network", "protocol", "router", "switch", "packet", "ip", 
                          "tcp", "dns", "http", "ethernet", "wifi"],
            "cloud_computing": ["cloud", "aws", "azure", "google cloud", "saas", "paas", 
                               "iaas", "virtual machine", "container", "kubernetes"],
            "data_science": ["data", "machine learning", "algorithm", "model", "neural network", 
                            "statistics", "classification", "regression", "clustering", "prediction"],
            "hardware": ["cpu", "gpu", "memory", "ram", "storage", "disk", "processor", 
                        "motherboard", "device", "hardware"]
        }
        
        # Check each domain
        for domain, keywords in domain_keywords.items():
            # Check if subject or content contains domain keywords
            if subject in keywords or any(keyword in content for keyword in keywords):
                return domain
        
        # Default to general tech if no specific domain detected
        return "general_tech"
    
    def _contradicts(self, statement1: str, statement2: str) -> bool:
        """
        Check if two statements contradict each other.
        
        Args:
            statement1: First statement
            statement2: Second statement
            
        Returns:
            True if contradiction detected, False otherwise
        """
        # This is a simplified implementation - would need NLP for better contradiction detection
        # Currently looks for direct negation patterns
        
        # Clean and normalize statements
        s1 = statement1.lower().strip()
        s2 = statement2.lower().strip()
        
        # Direct negation patterns
        negation_pairs = [
            ("is", "is not"), ("can", "cannot"), ("will", "will not"),
            ("has", "does not have"), ("supports", "does not support"),
            ("enables", "does not enable"), ("allows", "does not allow")
        ]
        
        # Check for direct negations
        for pos, neg in negation_pairs:
            if pos in s1 and neg in s2 and self._similar_context(s1, s2, pos, neg):
                return True
            if neg in s1 and pos in s2 and self._similar_context(s1, s2, neg, pos):
                return True
        
        # Check for opposite adjectives
        opposite_pairs = [
            ("faster", "slower"), ("better", "worse"), 
            ("higher", "lower"), ("more", "less"),
            ("secure", "insecure"), ("safe", "unsafe")
        ]
        
        for pos, neg in opposite_pairs:
            if pos in s1 and neg in s2 and self._similar_context(s1, s2, pos, neg):
                return True
            if neg in s1 and pos in s2 and self._similar_context(s1, s2, neg, pos):
                return True
        
        return False
    
    def _similar_context(self, s1: str, s2: str, word1: str, word2: str) -> bool:
        """
        Check if two statements have similar context around contradictory words.
        
        Args:
            s1: First statement
            s2: Second statement
            word1: Word in first statement
            word2: Word in second statement
            
        Returns:
            True if the contexts are similar, False otherwise
        """
        # Get words around the target words
        try:
            words1 = s1.split()
            words2 = s2.split()
            
            pos1 = words1.index(word1) if word1 in words1 else -1
            pos2 = words2.index(word2) if word2 in words2 else -1
            
            if pos1 == -1 or pos2 == -1:
                return False
                
            # Get context (up to 3 words before and after)
            context1 = set(words1[max(0, pos1-3):min(len(words1), pos1+4)])
            context2 = set(words2[max(0, pos2-3):min(len(words2), pos2+4)])
            
            # Remove the target words from context
            if word1 in context1:
                context1.remove(word1)
            if word2 in context2:
                context2.remove(word2)
                
            # Check if contexts have sufficient overlap
            overlap = context1.intersection(context2)
            return len(overlap) >= 2
            
        except Exception:
            return False
    
    def add_verified_fact(self, domain: str, subject: str, content: str) -> bool:
        """
        Add a verified fact to the facts store.
        
        Args:
            domain: Knowledge domain
            subject: Subject of the fact
            content: Content of the fact
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create fact object
            fact = {
                "subject": subject,
                "content": content,
                "fact_id": f"{domain}_{self._normalize_string(subject)}_{hashlib.md5(content.encode()).hexdigest()[:8]}"
            }
            
            # Initialize domain if needed
            if domain not in self.verified_facts:
                self.verified_facts[domain] = []
            
            # Add fact to store
            self.verified_facts[domain].append(fact)
            
            # Save to disk
            self._save_facts(domain)
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding verified fact: {e}")
            return False
    
    def _save_facts(self, domain: str) -> bool:
        """
        Save facts for a domain to disk.
        
        Args:
            domain: Knowledge domain
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if domain not in self.verified_facts:
                return False
                
            # Create file path
            file_path = self.facts_dir / f"{domain}.json"
            
            # Save to file with UTF-8 encoding
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.verified_facts[domain], f, indent=2, ensure_ascii=False)
                
            logger.info(f"Saved {len(self.verified_facts[domain])} facts for domain '{domain}'")
            return True
            
        except Exception as e:
            logger.error(f"Error saving facts for domain '{domain}': {e}")
            return False
    
    def _normalize_string(self, s: str) -> str:
        """Normalize a string for use in IDs."""
        return re.sub(r'[^a-zA-Z0-9]', '_', s.lower())
    
    def get_recent_facts(self, domain: str = None, limit: int = 10) -> List[Dict]:
        """
        Get recent verified facts, optionally filtered by domain.
        
        Args:
            domain: Knowledge domain to filter by, or None for all domains
            limit: Maximum number of facts to return
            
        Returns:
            List of facts
        """
        facts = []
        
        if domain and domain in self.verified_facts:
            # Return facts from specific domain
            facts = self.verified_facts[domain][-limit:]
        else:
            # Collect facts from all domains
            for d, domain_facts in self.verified_facts.items():
                facts.extend(domain_facts)
            
            # Sort by fact_id (which may contain timestamp) and take most recent
            facts.sort(key=lambda x: x.get("fact_id", ""))
            facts = facts[-limit:]
        
        return facts

def main():
    """Main function to test the enhanced consistency checker."""
    # Get project root and datasets directory
    project_root = Path(__file__).resolve().parent.parent.parent
    datasets_dir = project_root / "Datasets"
    
    # Create consistency checker
    checker = EnhancedConsistencyChecker(datasets_dir)
    
    # Add some test facts
    checker.add_verified_fact("cybersecurity", "AES", "AES is a symmetric encryption algorithm with key sizes of 128, 192, or 256 bits.")
    checker.add_verified_fact("programming", "Python", "Python is an interpreted, high-level, general-purpose programming language.")
    checker.add_verified_fact("networking", "TCP", "TCP is a connection-oriented protocol that provides reliable, ordered, and error-checked delivery of data.")
    
    # Test consistency checking
    test_texts = [
        "AES is a symmetric encryption algorithm with key sizes of 128, 192, or 256 bits. It is widely used for secure communications.",
        "AES is an asymmetric encryption algorithm with key sizes of 64 or 128 bits. It is not suitable for secure communications.",
        "TCP is both connection-oriented and connectionless. It provides reliable and unreliable delivery of data."
    ]
    
    for i, text in enumerate(test_texts):
        print(f"\nChecking text {i+1}:")
        is_consistent, inconsistencies = checker.check_consistency(text)
        print(f"Is consistent: {is_consistent}")
        if not is_consistent:
            print("Inconsistencies:")
            for issue in inconsistencies:
                print(f"- {issue['message']}: {issue['text']}")

if __name__ == "__main__":
    main()
3