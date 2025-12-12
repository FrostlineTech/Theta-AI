"""
Conversation Flow Manager for Theta AI.
Manages the flow and state of conversations.
"""

import re
import random
from datetime import datetime, timedelta

class ConversationFlowManager:
    """
    Manages conversation state and flow to create more natural dialogues.
    Keeps track of conversation context, user preferences, and dialogue state.
    """
    
    # Conversation states
    GREETING = "greeting"        # Initial greeting
    OPEN = "open"                # Open conversation, waiting for topic
    TOPIC_FOCUSED = "focused"    # Discussing a specific topic
    FOLLOWUP = "followup"        # In a follow-up exchange
    CLOSING = "closing"          # Ending the conversation
    
    def __init__(self):
        """Initialize the conversation flow manager."""
        # Core conversation state
        self.state = self.GREETING
        self.previous_states = []
        self.conversation_start_time = datetime.now()
        self.last_interaction_time = datetime.now()
        self.exchange_count = 0
        self.consecutive_short_responses = 0
        
        # User information
        self.user_name = None
        self.user_preferences = {}
        
        # Topic tracking
        self.current_topic = None
        self.previous_topics = []
        self.topic_change_count = 0
        
        # Context tracking
        self.context = {}
        self.entities_mentioned = {}
    
    def process_input(self, user_input):
        """
        Process user input and update conversation state.
        
        Args:
            user_input (str): User's message
            
        Returns:
            dict: Updated conversation state and context
        """
        # Update timing information
        self.last_interaction_time = datetime.now()
        conversation_duration = (self.last_interaction_time - self.conversation_start_time).total_seconds()
        
        # Increment exchange counter
        self.exchange_count += 1
        
        # Save previous state for context
        self.previous_states.append(self.state)
        if len(self.previous_states) > 5:  # Keep only last 5 states
            self.previous_states = self.previous_states[-5:]
            
        # Check for user name if not already known
        if not self.user_name:
            extracted_name = self._extract_name(user_input)
            if extracted_name:
                self.user_name = extracted_name
                
        # Try to detect topic
        detected_topic = self._detect_topic(user_input)
        if detected_topic and detected_topic != self.current_topic:
            self.previous_topics.append(self.current_topic)
            self.current_topic = detected_topic
            self.topic_change_count += 1
            
        # Extract entities
        new_entities = self._extract_entities(user_input)
        for entity_type, entities in new_entities.items():
            if entity_type not in self.entities_mentioned:
                self.entities_mentioned[entity_type] = []
            self.entities_mentioned[entity_type].extend(entities)
            
        # Update state based on input and context
        self._update_state(user_input)
        
        # Detect if user's message was a short response
        is_short_response = len(user_input.split()) <= 3
        if is_short_response:
            self.consecutive_short_responses += 1
        else:
            self.consecutive_short_responses = 0
            
        # Return state information
        return {
            'state': self.state,
            'exchange_count': self.exchange_count,
            'conversation_duration': conversation_duration,
            'current_topic': self.current_topic,
            'user_name': self.user_name,
            'consecutive_short_responses': self.consecutive_short_responses,
            'entities': self.entities_mentioned
        }
        
    def _update_state(self, user_input):
        """
        Update the conversation state based on user input and current state.
        
        Args:
            user_input (str): User's message
        """
        input_lower = user_input.lower().strip()
        
        # Check for conversation closers first
        if any(closer in input_lower for closer in ['goodbye', 'bye', 'see you', 'talk to you later', 'gotta go']):
            self.state = self.CLOSING
            return
            
        # State transitions
        if self.state == self.GREETING:
            # Move to open state after initial greeting
            self.state = self.OPEN
            
        elif self.state == self.OPEN:
            # Move to topic focused if we detected a topic
            if self.current_topic:
                self.state = self.TOPIC_FOCUSED
                
        elif self.state == self.TOPIC_FOCUSED:
            # Stay in topic focused or move to followup
            if any(question_word in input_lower for question_word in ['what', 'how', 'why', 'when', 'where', 'who']):
                self.state = self.FOLLOWUP
                
        elif self.state == self.FOLLOWUP:
            # Move back to topic focused after followup
            if self.consecutive_short_responses > 2:
                self.state = self.OPEN  # After several short responses, reset to open state
            else:
                self.state = self.TOPIC_FOCUSED
        
        elif self.state == self.CLOSING:
            # If user continues after closing, go back to open state
            if not any(closer in input_lower for closer in ['goodbye', 'bye', 'see you', 'exit']):
                self.state = self.OPEN
    
    def _extract_name(self, text):
        """
        Try to extract a user's name from their message.
        This is a simplified implementation.
        
        Args:
            text (str): User's message
            
        Returns:
            str or None: Extracted name or None
        """
        # Common name introduction patterns
        name_patterns = [
            r"(?:my name is|I'm|I am) ([A-Z][a-z]+)",
            r"(?:call me) ([A-Z][a-z]+)",
            r"(?:this is) ([A-Z][a-z]+)",
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
                
        return None
        
    def _detect_topic(self, text):
        """
        Detect the main topic of the message.
        This is a simplified implementation.
        
        Args:
            text (str): User's message
            
        Returns:
            str or None: Detected topic or None
        """
        # Very simple topic detection based on keywords
        topics = {
            "cybersecurity": ["security", "hack", "breach", "cyber", "virus", "malware", "phishing", "password"],
            "programming": ["code", "program", "development", "software", "bug", "function", "class", "variable"],
            "networking": ["network", "router", "server", "ip", "dns", "firewall", "tcp", "http"],
            "databases": ["database", "sql", "query", "table", "record", "field", "join", "schema"],
            "hardware": ["computer", "hardware", "cpu", "gpu", "ram", "storage", "server", "device"],
            "career": ["job", "career", "interview", "resume", "salary", "skills", "position", "hire"]
        }
        
        # Count topic keywords
        text_lower = text.lower()
        topic_counts = {}
        
        for topic, keywords in topics.items():
            count = sum(1 for keyword in keywords if keyword in text_lower)
            if count > 0:
                topic_counts[topic] = count
                
        # Return the topic with the most keywords, if any
        if topic_counts:
            return max(topic_counts, key=topic_counts.get)
            
        return None
        
    def _extract_entities(self, text):
        """
        Extract basic entities from text.
        This is a simplified implementation.
        
        Args:
            text (str): User's message
            
        Returns:
            dict: Dictionary of entity types and values
        """
        entities = {
            "dates": [],
            "numbers": [],
            "urls": [],
            "emails": []
        }
        
        # Extract dates
        date_patterns = [
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',  # MM/DD/YYYY
            r'\b\d{1,2}-\d{1,2}-\d{4}\b'      # MM-DD-YYYY
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text)
            entities["dates"].extend(matches)
            
        # Extract numbers
        number_pattern = r'\b\d+(?:\.\d+)?\b'
        entities["numbers"] = re.findall(number_pattern, text)
        
        # Extract URLs
        url_pattern = r'https?://\S+'
        entities["urls"] = re.findall(url_pattern, text)
        
        # Extract emails
        email_pattern = r'\S+@\S+\.\S+'
        entities["emails"] = re.findall(email_pattern, text)
        
        return entities
        
    def get_state_based_prompt(self):
        """
        Get a prompt based on the current conversation state.
        
        Returns:
            str: State-appropriate prompt
        """
        if self.state == self.GREETING:
            return "Hello! How can I help you today?"
            
        elif self.state == self.OPEN:
            if self.exchange_count < 3:
                return "What would you like to discuss today?"
            else:
                return "What else would you like to know?"
                
        elif self.state == self.TOPIC_FOCUSED:
            if self.current_topic:
                prompts = [
                    f"What specific aspects of {self.current_topic} are you interested in?",
                    f"Is there something specific about {self.current_topic} you'd like to explore?",
                    f"What questions do you have about {self.current_topic}?"
                ]
                return random.choice(prompts)
            return "Can you tell me more about what you're interested in?"
            
        elif self.state == self.FOLLOWUP:
            return "Is there anything else you'd like to know about this?"
            
        elif self.state == self.CLOSING:
            return "Goodbye! Feel free to ask if you need anything else."
            
        return "How can I assist you?"
        
    def should_add_personal_touch(self):
        """
        Determine if we should add a personal touch to the response.
        
        Returns:
            bool: True if we should personalize the response
        """
        # Add personal touches after several exchanges
        if self.exchange_count >= 3 and self.user_name:
            # Don't overdo it - roughly 1 in 4 messages
            return random.random() < 0.25
            
        return False
        
    def generate_personalized_prefix(self):
        """
        Generate a personalized prefix for a response.
        
        Returns:
            str: Personalized prefix
        """
        if not self.user_name:
            return ""
            
        prefixes = [
            f"Sure thing, {self.user_name}! ",
            f"{self.user_name}, ",
            f"Absolutely, {self.user_name}. ",
            f"Great question, {self.user_name}. "
        ]
        
        return random.choice(prefixes)
        
    def should_encourage_continuation(self):
        """
        Determine if we should encourage the user to continue the conversation.
        
        Returns:
            bool: True if we should add a continuation prompt
        """
        # Add continuation prompts for short responses
        if self.consecutive_short_responses >= 2:
            return True
            
        # Also occasionally prompt when in open state
        if self.state == self.OPEN and random.random() < 0.3:
            return True
            
        return False
        
    def generate_continuation_prompt(self):
        """
        Generate a prompt to encourage conversation continuation.
        
        Returns:
            str: Continuation prompt
        """
        prompts = [
            "Was there something specific you were curious about?",
            "Is there a particular topic you'd like to explore?",
            "Do you have any other questions I can help with?",
            "What else would you like to know?"
        ]
        
        if self.current_topic:
            topic_prompts = [
                f"Would you like to know more about {self.current_topic}?",
                f"Is there anything else about {self.current_topic} that interests you?",
                f"What aspects of {self.current_topic} would you like me to explain?"
            ]
            prompts.extend(topic_prompts)
            
        return random.choice(prompts)
