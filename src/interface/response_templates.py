"""
Response Templates for Theta AI.
Provides templates for generating consistent responses.
"""

import random
import re
from datetime import datetime

class ResponseTemplateEngine:
    """
    Template-based response generator that provides consistent responses
    for common conversation patterns.
    """
    
    def __init__(self):
        """Initialize the template engine with predefined templates."""
        # Basic conversation templates
        self.templates = {
            # Greeting templates
            "greeting": [
                "Hello{name_suffix}! How can I help you today?",
                "Hi there{name_suffix}! What can I assist you with?",
                "Hey{name_suffix}! I'm Theta. What would you like to know about?",
                "Hello{name_suffix}! I'm here to assist with cybersecurity, software development, or Frostline information."
            ],
            
            # Farewell templates
            "farewell": [
                "Goodbye{name_suffix}! Feel free to come back if you have more questions.",
                "See you later{name_suffix}! Have a great day!",
                "Farewell! Don't hesitate to return if you need assistance.",
                "Bye for now! I'm here whenever you need help."
            ],
            
            # Gratitude acknowledgement templates
            "gratitude": [
                "You're welcome{name_suffix}! I'm glad I could help.",
                "Happy to assist{name_suffix}! Let me know if you need anything else.",
                "No problem at all! Feel free to ask more questions.",
                "My pleasure! Is there anything else you'd like to know?"
            ],
            
            # Clarification request templates
            "clarification": [
                "I'm not sure I understand. Could you please rephrase that?",
                "I didn't quite catch that. Can you explain it differently?",
                "Could you provide more details about what you're asking?",
                "I'm having trouble understanding your request. Could you elaborate?"
            ],
            
            # Not understood templates
            "not_understood": [
                "I'm not sure I understand. Could you rephrase that?",
                "I didn't quite catch that. Can you explain it differently?",
                "I'm still learning and didn't understand your request. Could you try asking in another way?"
            ],
            
            # Information not available templates
            "info_not_available": [
                "I don't have specific information about '{topic}' in my knowledge base. This topic might be specialized or outside my training data.",
                "I'm afraid I don't have detailed information about '{topic}'. My knowledge is focused on cybersecurity, software development, and related technical areas.",
                "I don't have enough information about '{topic}' to provide a reliable answer."
            ],
            
            # Facts templates
            "fact": [
                "{fact}",
                "Here's what I know about {topic}: {fact}",
                "According to my knowledge base, {fact}"
            ],
            
            # Definition templates
            "definition": [
                "{term} refers to {definition}",
                "{term} is {definition}",
                "The term '{term}' means {definition}"
            ],
            
            # Question answer templates
            "question_what": [
                "{answer}",
                "To answer your question about {topic}: {answer}",
                "Regarding {topic}: {answer}"
            ],
            
            # How-to question templates
            "question_how": [
                "Here's how to {topic}:\n\n{answer}",
                "To {topic}, you would:\n\n{answer}",
                "The process for {topic} is as follows:\n\n{answer}"
            ],
            
            # Why question templates
            "question_why": [
                "The reason for {topic} is:\n\n{answer}",
                "{answer}",
                "To understand why {topic}: {answer}"
            ],
            
            # Encouragement templates
            "encouragement": [
                "Is there anything else you'd like to know?",
                "Do you have any other questions I can help with?",
                "What else would you like to learn about?",
                "Feel free to ask more questions!"
            ],
            
            # Error templates
            "error": [
                "I apologize, but I encountered an error while processing your request. Could you try asking in a different way?",
                "Something went wrong while generating a response. Could we try a different approach?",
                "I'm having technical difficulties answering that. Could you rephrase your question?"
            ],
            
            # Topic transitions
            "topic_transition": [
                "Shifting to {topic}, {answer}",
                "On the topic of {topic}, {answer}",
                "Regarding {topic}: {answer}"
            ]
        }
    
    def generate(self, template_type, context=None):
        """
        Generate a response from a template.
        
        Args:
            template_type (str): Type of template to use
            context (dict, optional): Context variables for the template
            
        Returns:
            str: Generated response from template
        """
        if template_type not in self.templates:
            template_type = "not_understood"
            
        # Get a random template of the specified type
        template = random.choice(self.templates[template_type])
        
        # Default context
        if context is None:
            context = {}
            
        # Format user name if available
        name_suffix = ""
        if 'user_name' in context and context['user_name']:
            name_suffix = f" {context['user_name']}"
            
        # Replace name suffix placeholder
        template = template.replace("{name_suffix}", name_suffix)
        
        # Fill in other template placeholders
        if context:
            for key, value in context.items():
                placeholder = "{" + key + "}"
                if placeholder in template:
                    template = template.replace(placeholder, str(value))
                    
        return template
        
    def add_template(self, template_type, template_text):
        """
        Add a new template or add to existing template type.
        
        Args:
            template_type (str): Type of template
            template_text (str): Template text
        """
        if template_type not in self.templates:
            self.templates[template_type] = []
            
        self.templates[template_type].append(template_text)
        
    def add_greeting_time(self, greeting):
        """
        Add time-appropriate greeting.
        
        Args:
            greeting (str): Base greeting
            
        Returns:
            str: Time-appropriate greeting
        """
        hour = datetime.now().hour
        
        if 5 <= hour < 12:
            time_greeting = "Good morning"
        elif 12 <= hour < 18:
            time_greeting = "Good afternoon"
        else:
            time_greeting = "Good evening"
            
        return f"{time_greeting}! {greeting}"
        
    def format_list(self, items, with_bullets=True):
        """
        Format a list of items with bullets or numbers.
        
        Args:
            items (list): List of items to format
            with_bullets (bool): Whether to use bullets or numbers
            
        Returns:
            str: Formatted list
        """
        if not items:
            return ""
            
        if with_bullets:
            return "\n".join([f"• {item}" for item in items])
        else:
            return "\n".join([f"{i+1}. {item}" for i, item in enumerate(items)])
            
    def get_confidence_phrase(self, confidence):
        """
        Get a phrase indicating confidence level.
        
        Args:
            confidence (float): Confidence value between 0 and 1
            
        Returns:
            str: Confidence phrase
        """
        if confidence >= 0.9:
            return random.choice(["I'm confident that", "I know that", "I'm certain that"])
        elif confidence >= 0.7:
            return random.choice(["I believe that", "From what I understand,", "I think that"])
        elif confidence >= 0.5:
            return random.choice(["I think", "From my knowledge,", "As I understand it,"])
        else:
            return random.choice(["I'm not entirely sure, but", "I'm not confident, but", "I believe, though I'm not certain, that"])
            
    def get_response_with_named_entity(self, template_type, entity_type, entity_name, answer):
        """
        Get a response that references a named entity.
        
        Args:
            template_type (str): Template type
            entity_type (str): Type of entity (person, company, etc.)
            entity_name (str): Name of the entity
            answer (str): Answer to format
            
        Returns:
            str: Response with entity reference
        """
        context = {
            "entity_name": entity_name,
            "entity_type": entity_type,
            "answer": answer
        }
        
        # Entity-specific templates
        entity_templates = {
            "person": [
                "Regarding {entity_name}, {answer}",
                "About {entity_name}: {answer}",
                "Speaking of {entity_name}, {answer}"
            ],
            "organization": [
                "About {entity_name}, {answer}",
                "Regarding {entity_name}: {answer}",
                "In the case of {entity_name}, {answer}"
            ],
            "location": [
                "Regarding {entity_name}, {answer}",
                "About {entity_name}: {answer}",
                "For {entity_name}, {answer}"
            ],
            "product": [
                "About {entity_name}: {answer}",
                "Regarding {entity_name}, {answer}",
                "When it comes to {entity_name}, {answer}"
            ]
        }
        
        if entity_type in entity_templates:
            template = random.choice(entity_templates[entity_type])
            return self._format_template(template, context)
        else:
            return self.generate(template_type, {"answer": answer})
            
    def _format_template(self, template, context):
        """
        Format a template with context variables.
        
        Args:
            template (str): Template string
            context (dict): Context variables
            
        Returns:
            str: Formatted template
        """
        # Replace placeholders
        for key, value in context.items():
            placeholder = "{" + key + "}"
            if placeholder in template:
                template = template.replace(placeholder, str(value))
                
        return template
