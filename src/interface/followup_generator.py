"""
Follow-up Question Generator for Theta AI.
Generates contextual follow-up questions based on conversation history.
"""

import random
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FollowupGenerator:
    """
    Generates follow-up questions based on conversation context.
    """
    
    # Templates for follow-up questions by domain
    FOLLOWUP_TEMPLATES = {
        "programming": [
            "Would you like to see an example of {topic}?",
            "Do you want to learn more about {topic} best practices?",
            "Are you trying to implement {topic} in a specific programming language?",
            "What specific aspect of {topic} are you struggling with?",
            "Would you like me to explain how {topic} relates to {related_topic}?"
        ],
        "networking": [
            "Would you like to know more about {topic} configurations?",
            "Are you troubleshooting a specific issue with {topic}?",
            "Would you like me to explain how {topic} works with {related_topic}?",
            "Do you need help implementing {topic} in your environment?",
            "What specific aspect of {topic} would you like to explore further?"
        ],
        "cybersecurity": [
            "Would you like to learn about common {topic} vulnerabilities?",
            "Do you need information on how to protect against {topic} threats?",
            "Are you interested in best practices for {topic}?",
            "Would you like to understand how {topic} relates to compliance requirements?",
            "Should we discuss mitigation strategies for {topic} risks?"
        ],
        "cloud_computing": [
            "Would you like to compare different providers' approaches to {topic}?",
            "Do you need help implementing {topic} in a specific cloud platform?",
            "Are you concerned about scaling issues with {topic}?",
            "Would you like to discuss cost optimization for {topic}?",
            "Are there specific {topic} features you want to explore?"
        ],
        "database": [
            "Would you like to see an example query for {topic}?",
            "Do you want to know more about {topic} performance optimization?",
            "Are you considering migrating from one {topic} system to another?",
            "Would you like to discuss {topic} backup strategies?",
            "Should we talk about scaling your {topic} solution?"
        ],
        "machine_learning": [
            "Would you like to explore different algorithms for {topic}?",
            "Do you need help with preparing data for your {topic} model?",
            "Are you interested in evaluating the performance of your {topic} solution?",
            "Would you like to discuss deployment strategies for {topic} models?",
            "Should we talk about ethical considerations in {topic}?"
        ],
        "operating_systems": [
            "Do you need help with specific {topic} commands?",
            "Would you like to compare different {topic} distributions/versions?",
            "Are you having trouble with {topic} permissions or security?",
            "Would you like to learn about automating tasks in {topic}?",
            "Should we discuss {topic} troubleshooting techniques?"
        ],
        "general": [
            "Would you like to know more about this topic?",
            "Do you have any other questions about what we discussed?",
            "Would you like me to clarify anything from my explanation?",
            "Is there a specific aspect of this you'd like to explore further?",
            "Would you like to see a practical example of what I explained?"
        ]
    }
    
    # Generic follow-up questions when no specific topic is detected
    GENERIC_FOLLOWUPS = [
        "Do you have any other questions I can help with?",
        "Would you like me to explain any part of that in more detail?",
        "Is there anything else you'd like to know about this topic?",
        "Does that answer your question, or would you like more information?",
        "Can I help you with anything else related to this?"
    ]
    
    def __init__(self):
        """Initialize follow-up generator."""
        # Track recently used templates to avoid repetition
        self.recent_templates = []
    
    def extract_entities(self, text):
        """
        Extract important entities/terms from text.
        Simple implementation using capitalized terms and tech keywords.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            list: Extracted entities
        """
        if not text:
            return []
        
        # Technical terms that might appear in lowercase
        tech_terms = [
            "api", "rest", "json", "xml", "http", "https", "sql", "nosql", 
            "crud", "dns", "dhcp", "tcp", "ip", "udp", "vpn", "ssh",
            "html", "css", "js", "python", "java", "javascript", "c++", "ruby"
        ]
        
        # Find capitalized terms (potential entities)
        capitalized = re.findall(r'\b[A-Z][a-zA-Z]+\b', text)
        
        # Find tech terms
        found_tech_terms = []
        for term in tech_terms:
            if re.search(r'\b{}\b'.format(re.escape(term)), text.lower()):
                found_tech_terms.append(term)
        
        # Combine and remove duplicates
        all_entities = list(set(capitalized + found_tech_terms))
        
        return all_entities
    
    def generate_followup(self, topic, previous_response=None):
        """
        Generate a follow-up question based on topic and previous response.
        
        Args:
            topic (str): Detected topic
            previous_response (str, optional): Previous AI response
            
        Returns:
            str: Follow-up question
        """
        # Use appropriate templates based on detected topic
        templates = self.FOLLOWUP_TEMPLATES.get(topic, self.FOLLOWUP_TEMPLATES["general"])
        
        # Filter out recently used templates to avoid repetition
        available_templates = [t for t in templates if t not in self.recent_templates]
        
        # If all templates have been recently used, reset
        if not available_templates:
            available_templates = templates
            self.recent_templates = []
        
        # Select a template
        template = random.choice(available_templates)
        self.recent_templates.append(template)
        
        # Keep track of last 3 templates only
        if len(self.recent_templates) > 3:
            self.recent_templates = self.recent_templates[-3:]
            
        # Extract entities from previous response for more specific follow-ups
        related_topics = []
        if previous_response:
            related_topics = self.extract_entities(previous_response)
        
        # Format the template
        if "{related_topic}" in template and related_topics:
            related_topic = random.choice(related_topics)
            return template.format(topic=topic, related_topic=related_topic)
        elif "{topic}" in template:
            return template.format(topic=topic)
        else:
            return template
    
    def generate_generic_followup(self):
        """
        Generate a generic follow-up question when no specific topic is detected.
        
        Returns:
            str: Follow-up question
        """
        return random.choice(self.GENERIC_FOLLOWUPS)
