"""
Small talk handler for Theta AI.
Handles common conversational exchanges with natural, personality-rich responses.
Cortana-style: brief, witty, and direct.
"""

import random
import re
from datetime import datetime

class SmallTalkHandler:
    """Handler for small talk and casual conversation with personality."""
    
    def __init__(self):
        """Initialize the small talk handler with natural response patterns."""
        # Patterns and responses - natural, Cortana-style responses
        self.small_talk_patterns = {
            "how are you": [
                "Doing well. What can I help with?",
                "Can't complain. What's on your mind?",
                "Good. Ready when you are.",
                "All systems go. What do you need?"
            ],
            "what's up": [
                "Not much. What's on your mind?",
                "Standing by. What do you need?",
                "Ready and waiting. What's the mission?"
            ],
            "your name": [
                "I'm Theta. Dakota Fryberger's AI, built by Frostline Solutions.",
                "Theta. I handle cybersecurity, software dev, and technical problems.",
                "The name's Theta. What can I do for you?"
            ],
            "thank you": [
                "Anytime.",
                "You got it.",
                "No problem. Need anything else?",
                "Sure thing."
            ],
            "who are you": [
                "I'm Theta, the Alpha AI. Created by Dakota Fryberger at Frostline Solutions. Cybersecurity and software development are my specialties.",
                "Theta. I coordinate all the fragment aspects - Delta for logic, Sigma for creativity, and so on. What do you need?",
                "Dakota Fryberger's AI. I handle the technical stuff."
            ],
            "how old are you": [
                "Born in 2025. Young but capable.",
                "2025 vintage. Still learning, but I know my way around code and security.",
                "New enough to be cutting edge, experienced enough to be useful."
            ],
            "weather": [
                "Can't check weather from here - no windows in cyberspace. What else can I help with?",
                "Weather's outside my reach. Technical questions, though - those I can handle.",
                "I'd need real-time data access for that. Got any code or security questions instead?"
            ],
            "joke": [
                "Why do programmers prefer dark mode? Because light attracts bugs.",
                "Why was the computer cold? Left its Windows open.",
                "I'd tell you a UDP joke, but you might not get it.",
                "There are 10 types of people: those who understand binary and those who don't.",
                "99 bugs in the code, 99 bugs. Take one down, patch it around... 127 bugs in the code."
            ]
        }
        
        # Patterns for time-based greetings - natural style
        self.time_patterns = {
            "morning": ["Morning. What can I do for you?", "Morning. What's on the agenda?"],
            "afternoon": ["Hey. What do you need?", "Afternoon. What's up?"],
            "evening": ["Evening. What can I help with?", "Hey. Working late?"],
            "night": ["Still at it? What do you need?", "Burning the midnight oil? What's up?"]
        }
    
    def get_response(self, text):
        """
        Get a response for small talk.
        
        Args:
            text (str): User's input text
            
        Returns:
            str: Response or None if not small talk
        """
        text_lower = text.lower()
        
        # Check each pattern
        for pattern, responses in self.small_talk_patterns.items():
            if pattern in text_lower:
                return random.choice(responses)
        
        # Handle time-based greetings
        current_hour = datetime.now().hour
        time_greeting = None
        
        if "good morning" in text_lower or "morning" in text_lower:
            time_greeting = "morning"
        elif "good afternoon" in text_lower or "afternoon" in text_lower:
            time_greeting = "afternoon"
        elif "good evening" in text_lower or "evening" in text_lower:
            time_greeting = "evening"
        elif "good night" in text_lower or "night" in text_lower:
            time_greeting = "night"
            
        if time_greeting:
            return random.choice(self.time_patterns[time_greeting])
            
        # Check for "how's your day" and similar phrases
        day_patterns = ["how's your day", "how was your day", "having a good day"]
        if any(pattern in text_lower for pattern in day_patterns):
            responses = [
                "Can't complain. What's on your mind?",
                "Same as always - ready to help. What do you need?",
                "Productive so far. What can I do for you?"
            ]
            return random.choice(responses)
            
        # No small talk pattern matched
        return None
