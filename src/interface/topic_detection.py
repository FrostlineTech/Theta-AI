"""
Topic Detection Module for Theta AI.
Detects and tracks conversation topics.
"""

import re
import logging
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TopicDetector:
    """
    Detects and tracks conversation topics.
    """
    
    # Technical domains and their keywords
    TOPIC_KEYWORDS = {
        "programming": [
            "code", "programming", "function", "variable", "class", "object", 
            "method", "algorithm", "loop", "conditional", "python", "javascript",
            "java", "c++", "ruby", "go", "php", "typescript", "html", "css"
        ],
        "networking": [
            "network", "tcp", "ip", "protocol", "router", "switch", "firewall",
            "dns", "dhcp", "subnet", "vpn", "lan", "wan", "ethernet", "wifi",
            "packet", "latency", "bandwidth", "routing", "gateway"
        ],
        "cybersecurity": [
            "security", "encryption", "vulnerability", "exploit", "threat",
            "malware", "virus", "phishing", "authentication", "authorization",
            "firewall", "penetration", "testing", "incident", "response", "hack"
        ],
        "cloud_computing": [
            "cloud", "aws", "azure", "gcp", "kubernetes", "docker", "container",
            "serverless", "iaas", "paas", "saas", "virtualization", "scalability",
            "microservice", "orchestration", "instance", "cluster"
        ],
        "database": [
            "database", "sql", "nosql", "query", "table", "schema", "index",
            "mysql", "postgresql", "mongodb", "oracle", "join", "transaction",
            "crud", "backup", "restore", "replication", "shard"
        ],
        "machine_learning": [
            "machine learning", "ml", "ai", "artificial intelligence", "model", 
            "neural network", "deep learning", "training", "dataset", "feature", 
            "prediction", "classification", "regression", "clustering"
        ],
        "operating_systems": [
            "operating system", "os", "linux", "windows", "macos", "unix",
            "kernel", "process", "thread", "file system", "memory", "cpu",
            "permission", "registry", "bash", "powershell", "command"
        ]
    }
    
    def __init__(self):
        """Initialize topic detector."""
        # Previous topics detected to maintain context
        self.topic_history = []
        # Counter for overall conversation topics
        self.topic_counter = Counter()
    
    def detect_topics(self, question, answer=None):
        """
        Detect topics in text from question and optionally answer.
        
        Args:
            question (str): User's question text to analyze
            answer (str, optional): AI's answer text to analyze
            
        Returns:
            list: Detected topics sorted by relevance
        """
        if not question:
            return []
        
        # Combine question and answer if answer is provided
        if answer:
            text = question.lower() + " " + answer.lower()
        else:
            text = question.lower()
        
        detected_topics = Counter()
        
        for topic, keywords in self.TOPIC_KEYWORDS.items():
            for keyword in keywords:
                # Find whole word matches (not substrings)
                count = len(re.findall(r'\b{}\b'.format(re.escape(keyword)), text))
                if count > 0:
                    detected_topics[topic] += count
        
        # Return topics sorted by count
        return [topic for topic, _ in detected_topics.most_common()]
    
    def update_topic_history(self, question, answer=None):
        """
        Update topic history based on new exchange.
        
        Args:
            question (str): User's question text to analyze
            answer (str, optional): AI's answer text to analyze
            
        Returns:
            list: Current active topics
        """
        new_topics = self.detect_topics(question, answer)
        
        # Update the counter with new topics
        self.topic_counter.update(new_topics)
        
        # Add new topics to history
        if new_topics:
            self.topic_history.append(new_topics[0] if new_topics else None)
            
            # Keep only last 5 exchanges in history
            if len(self.topic_history) > 5:
                self.topic_history = self.topic_history[-5:]
        
        return self.get_active_topics()
    
    def get_active_topics(self):
        """
        Get currently active topics based on recent history.
        
        Returns:
            list: Active topics
        """
        recent_topics = Counter(self.topic_history)
        # Topic is active if it appears in at least 2 of the last 5 exchanges
        active_topics = [topic for topic, count in recent_topics.items() 
                         if count >= 2 and topic is not None]
        
        return active_topics
    
    def get_primary_topic(self):
        """
        Get the primary topic of the conversation.
        
        Returns:
            str: Primary topic or None
        """
        if not self.topic_counter:
            return None
        
        return self.topic_counter.most_common(1)[0][0]
    
    def reset(self):
        """Reset topic history and counter."""
        self.topic_history = []
        self.topic_counter = Counter()
