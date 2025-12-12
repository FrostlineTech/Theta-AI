"""
Technical Embeddings for Theta AI

This module implements specialized embeddings for technical vocabulary,
improving understanding of domain-specific terms and concepts.
"""

import json
import logging
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import re
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TechnicalEmbeddings:
    """Manages specialized embeddings for technical vocabulary."""
    
    def __init__(self, datasets_dir: Path, vector_dim: int = 768):
        """
        Initialize the technical embeddings manager.
        
        Args:
            datasets_dir: Path to the datasets directory
            vector_dim: Dimension of embedding vectors
        """
        self.datasets_dir = datasets_dir
        self.vector_dim = vector_dim
        
        # Create embeddings directory
        self.embeddings_dir = datasets_dir / "embeddings"
        os.makedirs(self.embeddings_dir, exist_ok=True)
        
        # Initialize dictionaries for technical terms
        self.technical_terms = {}
        self.domain_vocabularies = {}
        self.term_embeddings = {}
        
        # Define domains
        self.domains = [
            "cybersecurity", "programming", "networking", 
            "cloud_computing", "data_science", "general_tech"
        ]
        
        # Initialize domain vocabularies
        for domain in self.domains:
            self.domain_vocabularies[domain] = set()
    
    def collect_technical_terms(self, scan_datasets: bool = True):
        """
        Collect technical terms from various sources.
        
        Args:
            scan_datasets: Whether to scan datasets for terms
        """
        logger.info("Collecting technical terms")
        
        # Load predefined technical terms for each domain
        self._load_predefined_terms()
        
        # Scan datasets if requested
        if scan_datasets:
            self._scan_datasets_for_terms()
        
        # Log collection results
        for domain in self.domains:
            term_count = len(self.domain_vocabularies.get(domain, []))
            logger.info(f"Collected {term_count} technical terms for domain '{domain}'")
    
    def _load_predefined_terms(self):
        """Load predefined technical terms from vocabulary files."""
        for domain in self.domains:
            vocab_file = self.embeddings_dir / f"{domain}_vocabulary.json"
            
            # Create vocabulary file if it doesn't exist
            if not vocab_file.exists():
                self._create_initial_vocabulary(domain)
            
            # Load vocabulary file
            try:
                # Load with UTF-8 encoding and error handling
                try:
                    with open(vocab_file, 'r', encoding='utf-8', errors='replace') as f:
                        vocab_data = json.load(f)
                except json.JSONDecodeError:
                    # If that fails, try a more aggressive approach with binary reading
                    logger.info(f"JSON decode error with utf-8 for {vocab_file}, trying alternative approach")
                    with open(vocab_file, 'rb') as f:
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
                    vocab_data = json.loads(cleaned_content.decode('utf-8'))
                
                if isinstance(vocab_data, dict) and "terms" in vocab_data:
                    # Add terms to domain vocabulary
                    self.domain_vocabularies[domain].update(vocab_data["terms"])
                    
                    # Add term definitions if available
                    if "definitions" in vocab_data:
                        for term, definition in vocab_data["definitions"].items():
                            if term not in self.technical_terms:
                                self.technical_terms[term] = {
                                    "domains": [domain],
                                    "definition": definition
                                }
                            else:
                                if domain not in self.technical_terms[term]["domains"]:
                                    self.technical_terms[term]["domains"].append(domain)
                                
                                # Only update definition if none exists
                                if "definition" not in self.technical_terms[term] or not self.technical_terms[term]["definition"]:
                                    self.technical_terms[term]["definition"] = definition
            
            except Exception as e:
                logger.error(f"Error loading vocabulary for domain '{domain}': {e}")
    
    def _create_initial_vocabulary(self, domain: str):
        """
        Create initial vocabulary file for a domain.
        
        Args:
            domain: The domain to create vocabulary for
        """
        vocab_data = {
            "domain": domain,
            "terms": [],
            "definitions": {}
        }
        
        # Add domain-specific terms
        if domain == "cybersecurity":
            terms_and_defs = {
                "zero trust": "A security concept centered on the belief that organizations should not automatically trust anything inside or outside their perimeters.",
                "malware": "Software designed to disrupt, damage, or gain unauthorized access to computer systems.",
                "phishing": "The practice of sending fraudulent communications to obtain sensitive information.",
                "ransomware": "A type of malicious software designed to block access to a computer system until a sum of money is paid.",
                "firewall": "A security system that monitors and controls incoming and outgoing network traffic.",
                "encryption": "The process of converting information into code to prevent unauthorized access.",
                "vulnerability": "A weakness which can be exploited by a threat actor to perform unauthorized actions.",
                "exploit": "A piece of software or sequence of commands that takes advantage of a vulnerability.",
                "penetration testing": "An authorized simulated attack on a computer system to evaluate its security.",
                "SIEM": "Security Information and Event Management, software that provides real-time analysis of security alerts."
            }
        elif domain == "programming":
            terms_and_defs = {
                "algorithm": "A step-by-step procedure for solving a problem or accomplishing a task.",
                "API": "Application Programming Interface, a set of rules allowing programs to communicate with each other.",
                "compiler": "A program that translates code written in a high-level language to a lower-level language.",
                "framework": "A platform for developing software applications that provides a foundation on which software developers can build programs.",
                "function": "A block of code that performs a specific task and can be reused throughout a program.",
                "variable": "A storage location paired with an associated symbolic name which contains a value.",
                "class": "A blueprint for creating objects in object-oriented programming.",
                "inheritance": "A mechanism where a new class inherits properties and behaviors from an existing class.",
                "polymorphism": "The ability to present the same interface for different underlying data types.",
                "encapsulation": "The bundling of data with the methods that operate on that data."
            }
        elif domain == "networking":
            terms_and_defs = {
                "TCP/IP": "Transmission Control Protocol/Internet Protocol, the basic communication language of the Internet.",
                "router": "A device that forwards data packets between computer networks.",
                "switch": "A device that connects devices on a computer network by using packet switching.",
                "DNS": "Domain Name System, a naming system for computers and services connected to the Internet.",
                "DHCP": "Dynamic Host Configuration Protocol, a protocol that automatically assigns IP addresses.",
                "subnet": "A logical subdivision of an IP network.",
                "VPN": "Virtual Private Network, a technology that creates a safe and encrypted connection over a less secure network.",
                "firewall": "A network security system that monitors and controls incoming and outgoing network traffic.",
                "packet": "A formatted unit of data carried by a packet-switched network.",
                "bandwidth": "The maximum rate of data transfer across a given path."
            }
        elif domain == "cloud_computing":
            terms_and_defs = {
                "IaaS": "Infrastructure as a Service, providing virtualized computing resources over the Internet.",
                "PaaS": "Platform as a Service, providing a platform allowing customers to develop, run, and manage applications.",
                "SaaS": "Software as a Service, a software licensing and delivery model in which software is centrally hosted.",
                "container": "A standard unit of software that packages code and all its dependencies.",
                "microservices": "An architectural style that structures an application as a collection of loosely coupled services.",
                "serverless": "A cloud computing execution model where the cloud provider dynamically manages the allocation of machine resources.",
                "virtualization": "The creation of a virtual version of something, such as a server or storage device.",
                "orchestration": "Automated configuration, coordination, and management of computer systems and software.",
                "autoscaling": "A cloud computing feature that automatically adjusts resources to match demand.",
                "multi-tenancy": "A software architecture where a single instance of software serves multiple customers."
            }
        elif domain == "data_science":
            terms_and_defs = {
                "machine learning": "A field of artificial intelligence that uses statistical techniques to give computers the ability to learn from data.",
                "neural network": "A series of algorithms that mimic the human brain to recognize patterns in data.",
                "deep learning": "A subset of machine learning where artificial neural networks learn from large amounts of data.",
                "regression": "A statistical method that estimates the relationships between variables.",
                "classification": "The problem of identifying to which category a new observation belongs.",
                "clustering": "The task of grouping a set of objects such that similar objects are in the same group.",
                "feature extraction": "The process of reducing the dimensionality of initial data set.",
                "data mining": "The process of discovering patterns in large data sets.",
                "overfitting": "A modeling error where a function is too closely fit to a limited set of data points.",
                "bias-variance tradeoff": "The conflict in minimizing two sources of error that prevent supervised learning algorithms from generalizing."
            }
        else:
            # General tech
            terms_and_defs = {
                "algorithm": "A process or set of rules to be followed in calculations or other problem-solving operations.",
                "API": "Application Programming Interface, a set of rules and protocols for building software applications.",
                "bandwidth": "The maximum rate of data transfer across a given path.",
                "cache": "A hardware or software component that stores data to serve future requests faster.",
                "database": "An organized collection of data stored and accessed electronically.",
                "encryption": "The process of encoding information in such a way that only authorized parties can access it.",
                "interface": "A point where two systems, subjects, organizations, etc., meet and interact.",
                "latency": "The delay before a transfer of data begins following an instruction for its transfer.",
                "protocol": "A set of rules governing the exchange or transmission of data between devices.",
                "scalability": "The capability of a system to handle a growing amount of work."
            }
        
        # Add terms and definitions to vocabulary
        vocab_data["terms"] = list(terms_and_defs.keys())
        vocab_data["definitions"] = terms_and_defs
        
        # Save vocabulary file with UTF-8 encoding
        vocab_file = self.embeddings_dir / f"{domain}_vocabulary.json"
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Created initial vocabulary for domain '{domain}' with {len(vocab_data['terms'])} terms")
    
    def _scan_datasets_for_terms(self):
        """Scan datasets to extract additional technical terms."""
        # Find all JSON files in the datasets directory
        json_files = []
        for ext in ["*.json"]:
            json_files.extend(list(self.datasets_dir.glob(ext)))
        
        # Process each file
        for file_path in json_files:
            try:
                # Load with UTF-8 encoding and error handling
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                        data = json.load(f)
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
                    data = json.loads(cleaned_content.decode('utf-8'))
                
                # Process list of QA pairs
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "question" in item and "answer" in item:
                            # Get domain
                            domain = item.get("domain", "general_tech")
                            if domain not in self.domains:
                                domain = "general_tech"
                            
                            # Extract terms from question and answer
                            text = f"{item['question']} {item['answer']}"
                            self._extract_terms_from_text(text, domain)
            
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
        
        # Update vocabulary files with new terms
        self._update_vocabulary_files()
    
    def _extract_terms_from_text(self, text: str, domain: str):
        """
        Extract technical terms from text.
        
        Args:
            text: Text to extract terms from
            domain: Domain of the text
        """
        # Simple term extraction (in a real system, use NLP)
        
        # Look for capitalized terms (potential proper nouns)
        capitalized_terms = re.findall(r'\b([A-Z][a-z]{2,}(?:/[A-Z][a-z]{2,})*)\b', text)
        
        # Look for acronyms
        acronyms = re.findall(r'\b([A-Z]{2,})\b', text)
        
        # Look for technical terms with numbers or special characters
        special_terms = re.findall(r'\b([a-zA-Z]+[0-9-_]+[a-zA-Z]*)\b', text)
        
        # Look for known technical patterns
        tech_patterns = [
            r'\b([a-z]+\.[a-z]+\.[a-z]+)\b',  # version numbers, package names
            r'\b([a-z]+-[a-z]+-[a-z]+)\b',    # hyphenated terms
            r'\b(v[0-9]+\.[0-9]+\.[0-9]+)\b'  # version strings
        ]
        
        pattern_matches = []
        for pattern in tech_patterns:
            pattern_matches.extend(re.findall(pattern, text))
        
        # Combine all potential terms
        potential_terms = capitalized_terms + acronyms + special_terms + pattern_matches
        
        # Filter duplicates and add to domain vocabulary
        unique_terms = set(term.lower() for term in potential_terms)
        self.domain_vocabularies[domain].update(unique_terms)
    
    def _update_vocabulary_files(self):
        """Update vocabulary files with newly discovered terms."""
        for domain in self.domains:
            vocab_file = self.embeddings_dir / f"{domain}_vocabulary.json"
            
            try:
                # Load existing vocabulary with UTF-8 encoding and error handling
                try:
                    with open(vocab_file, 'r', encoding='utf-8', errors='replace') as f:
                        vocab_data = json.load(f)
                except json.JSONDecodeError:
                    # If that fails, try a more aggressive approach with binary reading
                    logger.info(f"JSON decode error with utf-8 for {vocab_file}, trying alternative approach")
                    with open(vocab_file, 'rb') as f:
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
                    vocab_data = json.loads(cleaned_content.decode('utf-8'))
                
                # Update terms
                existing_terms = set(vocab_data.get("terms", []))
                new_terms = self.domain_vocabularies[domain] - existing_terms
                
                if new_terms:
                    vocab_data["terms"] = sorted(list(existing_terms.union(new_terms)))
                    
                    # Save updated vocabulary with UTF-8 encoding
                    with open(vocab_file, 'w', encoding='utf-8') as f:
                        json.dump(vocab_data, f, indent=2, ensure_ascii=False)
                    
                    logger.info(f"Added {len(new_terms)} new terms to {domain} vocabulary")
            
            except Exception as e:
                logger.error(f"Error updating vocabulary file for domain '{domain}': {e}")
    
    def create_technical_embeddings(self):
        """
        Create embeddings for technical terms.
        In a real implementation, this would use proper embeddings from a language model.
        """
        logger.info("Creating technical embeddings")
        
        # Collect terms if not done already
        if not self.technical_terms:
            self.collect_technical_terms()
        
        # Get all terms from all domains
        all_terms = set()
        for domain, terms in self.domain_vocabularies.items():
            all_terms.update(terms)
        
        # Create random embeddings for each term (placeholder for real embeddings)
        for term in all_terms:
            # In a real system, this would use a language model to create embeddings
            # Here we just create random vectors for demonstration
            self.term_embeddings[term] = self._create_random_embedding()
        
        # Save embeddings
        self.save_embeddings()
        
        logger.info(f"Created embeddings for {len(self.term_embeddings)} technical terms")
    
    def _create_random_embedding(self) -> np.ndarray:
        """
        Create a random embedding vector.
        This is a placeholder for actual embedding creation.
        
        Returns:
            Random embedding vector
        """
        # Create random vector
        vector = np.random.randn(self.vector_dim)
        # Normalize to unit length
        return vector / np.linalg.norm(vector)
    
    def save_embeddings(self):
        """Save embeddings to disk."""
        # Save embeddings for each domain
        for domain in self.domains:
            # Get terms for this domain
            domain_terms = self.domain_vocabularies.get(domain, set())
            
            if not domain_terms:
                continue
            
            # Create embeddings dictionary for this domain
            domain_embeddings = {}
            for term in domain_terms:
                if term in self.term_embeddings:
                    # Convert numpy array to list for JSON serialization
                    domain_embeddings[term] = self.term_embeddings[term].tolist()
            
            # Save embeddings with UTF-8 encoding
            embeddings_file = self.embeddings_dir / f"{domain}_embeddings.json"
            with open(embeddings_file, 'w', encoding='utf-8') as f:
                json.dump(domain_embeddings, f, ensure_ascii=False)
            
            logger.info(f"Saved {len(domain_embeddings)} embeddings for domain '{domain}'")
    
    def load_embeddings(self, domain: str = None):
        """
        Load embeddings from disk.
        
        Args:
            domain: Domain to load embeddings for, or None for all domains
        """
        # Determine domains to load
        domains_to_load = [domain] if domain else self.domains
        
        # Load embeddings for each domain
        for d in domains_to_load:
            embeddings_file = self.embeddings_dir / f"{d}_embeddings.json"
            
            if not embeddings_file.exists():
                logger.warning(f"No embeddings file found for domain '{d}'")
                continue
            
            try:
                # Load with UTF-8 encoding and error handling
                try:
                    with open(embeddings_file, 'r', encoding='utf-8', errors='replace') as f:
                        domain_embeddings = json.load(f)
                except json.JSONDecodeError:
                    # If that fails, try a more aggressive approach with binary reading
                    logger.info(f"JSON decode error with utf-8 for {embeddings_file}, trying alternative approach")
                    with open(embeddings_file, 'rb') as f:
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
                    domain_embeddings = json.loads(cleaned_content.decode('utf-8'))
                
                # Convert lists back to numpy arrays
                for term, embedding in domain_embeddings.items():
                    self.term_embeddings[term] = np.array(embedding)
                
                logger.info(f"Loaded {len(domain_embeddings)} embeddings for domain '{d}'")
            
            except Exception as e:
                logger.error(f"Error loading embeddings for domain '{d}': {e}")
    
    def get_embedding(self, term: str) -> Optional[np.ndarray]:
        """
        Get embedding for a term.
        
        Args:
            term: Term to get embedding for
            
        Returns:
            Embedding vector or None if not found
        """
        term_lower = term.lower()
        return self.term_embeddings.get(term_lower)
    
    def get_similar_terms(self, term: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Get terms similar to the given term.
        
        Args:
            term: Term to find similar terms for
            top_k: Number of similar terms to return
            
        Returns:
            List of (term, similarity) tuples
        """
        term_lower = term.lower()
        
        # Get embedding for the term
        query_embedding = self.get_embedding(term_lower)
        if query_embedding is None:
            logger.warning(f"No embedding found for term '{term}'")
            return []
        
        # Calculate similarities
        similarities = []
        for other_term, embedding in self.term_embeddings.items():
            if other_term == term_lower:
                continue
            
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, embedding)
            similarities.append((other_term, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k results
        return similarities[:top_k]
    
    def enrich_text_with_technical_context(self, text: str, domain: str = None) -> Dict:
        """
        Enrich text with technical context.
        
        Args:
            text: Text to enrich
            domain: Optional domain for context
            
        Returns:
            Dictionary with enriched text and identified terms
        """
        # Make lowercase for matching
        text_lower = text.lower()
        
        # Find technical terms in text
        found_terms = {}
        
        # If domain is specified, only look for terms in that domain
        domains_to_check = [domain] if domain else self.domains
        
        # Check each domain
        for d in domains_to_check:
            # Skip if no vocabulary for this domain
            if d not in self.domain_vocabularies:
                continue
                
            # Check each term in the domain vocabulary
            for term in self.domain_vocabularies[d]:
                # Simple substring check (in a real system, use NLP for better matching)
                if term in text_lower:
                    # Add term to found terms
                    if term not in found_terms:
                        found_terms[term] = {
                            "domains": [d],
                            "definition": self._get_term_definition(term)
                        }
                    elif d not in found_terms[term]["domains"]:
                        found_terms[term]["domains"].append(d)
        
        # Create enriched text
        enriched_text = text
        
        # In a real system, you would modify the text to include tooltips or links
        # to technical term definitions, or inject additional context
        
        return {
            "enriched_text": enriched_text,
            "technical_terms": found_terms
        }
    
    def _get_term_definition(self, term: str) -> str:
        """
        Get definition for a term.
        
        Args:
            term: Term to get definition for
            
        Returns:
            Definition string or empty string if not found
        """
        if term in self.technical_terms and "definition" in self.technical_terms[term]:
            return self.technical_terms[term]["definition"]
        
        # Check domain vocabularies for definition
        for domain in self.domains:
            vocab_file = self.embeddings_dir / f"{domain}_vocabulary.json"
            
            if vocab_file.exists():
                try:
                    # Load with UTF-8 encoding and error handling
                    try:
                        with open(vocab_file, 'r', encoding='utf-8', errors='replace') as f:
                            vocab_data = json.load(f)
                    except json.JSONDecodeError:
                        # If that fails, try a more aggressive approach with binary reading
                        logger.debug(f"JSON decode error with utf-8 for {vocab_file}, trying alternative approach")
                        with open(vocab_file, 'rb') as f:
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
                        vocab_data = json.loads(cleaned_content.decode('utf-8'))
                    
                    if "definitions" in vocab_data and term in vocab_data["definitions"]:
                        return vocab_data["definitions"][term]
                
                except Exception:
                    pass
        
        return ""

def main():
    """Main function to test technical embeddings."""
    # Get project root and datasets directory
    project_root = Path(__file__).resolve().parent.parent.parent
    datasets_dir = project_root / "Datasets"
    
    # Create technical embeddings
    embeddings = TechnicalEmbeddings(datasets_dir)
    
    # Collect technical terms
    embeddings.collect_technical_terms()
    
    # Create embeddings
    embeddings.create_technical_embeddings()
    
    # Test enriching text
    test_text = "Implementing proper encryption and firewall configurations are essential parts of a zero trust security model."
    
    enriched = embeddings.enrich_text_with_technical_context(test_text, domain="cybersecurity")
    
    print("\nEnriched Text Information:")
    print(f"Original text: {test_text}")
    print(f"Found technical terms: {len(enriched['technical_terms'])}")
    
    for term, info in enriched['technical_terms'].items():
        print(f"- {term} ({', '.join(info['domains'])}): {info['definition']}")

if __name__ == "__main__":
    main()
