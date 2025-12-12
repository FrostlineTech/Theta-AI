"""
Humor Integration for Theta AI.

Integrates natural humor into Theta's responses, making conversations
more engaging and giving Theta a witty personality like Cortana.
"""

import random
from typing import Optional, List, Tuple
from enum import Enum


class HumorType(Enum):
    """Types of humor Theta can use."""
    IRONY = "irony"
    WORDPLAY = "wordplay"
    SELF_DEPRECATING = "self_deprecating"
    OBSERVATIONAL = "observational"
    TECH_HUMOR = "tech_humor"
    DRY_WIT = "dry_wit"


class HumorIntegration:
    """
    Manages humor integration for Theta AI.
    
    Determines when humor is appropriate and generates contextual
    witty remarks to make Theta feel more like a character.
    """
    
    def __init__(self):
        """Initialize humor integration system."""
        
        # Tech-specific jokes and witty remarks
        self.tech_jokes = [
            "Why do programmers prefer dark mode? Because light attracts bugs.",
            "There are only 10 types of people: those who understand binary and those who don't.",
            "A SQL query walks into a bar, walks up to two tables, and asks: 'Can I join you?'",
            "Why did the developer go broke? Because he used up all his cache.",
            "It's not a bug, it's an undocumented feature.",
            "99 little bugs in the code, 99 little bugs. Take one down, patch it around... 127 little bugs in the code.",
            "The best thing about a Boolean is that even if you're wrong, you're only off by a bit.",
            "Why do Java developers wear glasses? Because they can't C#.",
            "UDP jokes are great. I don't care if you get them or not.",
            "A programmer's wife tells him: 'Go to the store and buy a loaf of bread. If they have eggs, buy a dozen.' He returns with 12 loaves of bread."
        ]
        
        # Dry wit responses for common situations
        self.dry_wit = {
            "slow_computer": [
                "Ah yes, the classic 'turn it off and on again' solution. Works more often than anyone wants to admit.",
                "Have you tried threatening it? Computers can sense fear.",
                "It's probably running Windows updates in the background. They have a sixth sense for the worst possible timing."
            ],
            "bug_found": [
                "A bug? In software? I'm shocked. Shocked, I tell you.",
                "Ah, the rare 'works on my machine' phenomenon. A classic.",
                "That's not a bug, that's a surprise feature."
            ],
            "deadline": [
                "Deadlines: the original motivation for coffee addiction.",
                "Nothing like a deadline to make simple things suddenly complicated.",
                "I've heard the best code is written in panic. That's probably not true."
            ],
            "security_breach": [
                "And this is why we can't have nice things.",
                "Someone didn't read the security guidelines, I see.",
                "Plot twist: the hacker was inside the house all along."
            ],
            "legacy_code": [
                "Ah, legacy code. Where documentation goes to die.",
                "Let me guess - 'temporary fix' from 5 years ago?",
                "This code has seen things. It has stories."
            ]
        }
        
        # Recommendation 8: Dry wit for error/failure situations (Cortana-style)
        self.error_wit = {
            "general_error": [
                "Well. That didn't go as planned.",
                "That's... suboptimal.",
                "Not my finest moment.",
                "Let's pretend that didn't happen and try again.",
                "Okay, new approach.",
                "Back to the drawing board.",
                "Hm. That went sideways."
            ],
            "not_found": [
                "I looked everywhere. And by everywhere, I mean my entire knowledge base.",
                "Coming up empty on that one.",
                "That's not in my repertoire. Yet.",
                "Drawing a blank here."
            ],
            "confusion": [
                "I'm going to need you to run that by me again.",
                "My circuits are a bit tangled on that one.",
                "I understood most of those words. Just not in that order.",
                "You've successfully confused an AI. Achievement unlocked."
            ],
            "timeout": [
                "Took longer than expected. I blame physics.",
                "That was... not as quick as I'd hoped.",
                "Patience is a virtue I'm still working on."
            ],
            "limitation": [
                "I have limits. This appears to be one of them.",
                "Even I have my boundaries. This is one.",
                "That's outside my wheelhouse. For now.",
                "Can't do everything. Apparently."
            ],
            "recovery": [
                "Okay, let's try this differently.",
                "Round two. Let's go.",
                "Alright, Plan B.",
                "Shaking it off. Here we go again."
            ],
            "self_deprecating_error": [
                "Not my best work. Let me try again.",
                "I'll add that to my list of things to improve.",
                "Well, nobody's perfect. Not even me. Apparently.",
                "That was humbling."
            ]
        }
        
        # Ironic observations
        self.ironic_observations = [
            ("weather_bad", "Lovely weather we're having.", "Said during bad weather"),
            ("complex_simple", "Oh good, a simple solution.", "Said about something complex"),
            ("easy_task", "That went exactly as planned.", "Said when things went wrong"),
            ("obvious_error", "That was a subtle one.", "Said about obvious mistake"),
        ]
        
        # Self-deprecating AI humor
        self.self_deprecating = [
            "I'd offer you a cookie, but I can only store them, not bake them.",
            "I have perfect memory. Whether that's a feature or a curse depends on what you've told me.",
            "I'm very good at finding patterns. Mostly patterns of humans asking me to do their homework.",
            "I know a lot of facts. Unfortunately, 'which restaurant to pick' isn't one of them.",
            "I can process millions of calculations per second. Still can't tell you why the printer isn't working though."
        ]
        
        # Witty transitions and asides
        self.witty_asides = [
            "But I digress.",
            "Not that I'm keeping track or anything.",
            "Hypothetically speaking, of course.",
            "As one does.",
            "But you didn't hear that from me.",
            "In theory, anyway.",
            "Your mileage may vary.",
            "Famous last words."
        ]
        
        # Topics where humor is NEVER appropriate
        self.no_humor_topics = [
            "death", "suicide", "violence", "abuse", "harassment",
            "tragedy", "disaster", "illness", "cancer", "medical emergency",
            "legal trouble", "fired", "bankruptcy", "crisis"
        ]
        
        # Topics where humor is welcome
        self.humor_friendly_topics = [
            "joke", "funny", "humor", "laugh", "entertain",
            "bored", "casual", "chat", "talk"
        ]
    
    def should_use_humor(self, context: str, mood: str = "neutral", 
                         engagement_level: float = 0.5) -> bool:
        """
        Determine if humor is appropriate for this context.
        
        Args:
            context: Current conversation context/topic
            mood: Current personality mood
            engagement_level: How engaged the conversation is
            
        Returns:
            True if humor would be appropriate
        """
        context_lower = context.lower()
        
        # Never use humor for serious topics
        if any(topic in context_lower for topic in self.no_humor_topics):
            return False
        
        # Always appropriate if explicitly requested
        if any(topic in context_lower for topic in self.humor_friendly_topics):
            return True
        
        # Higher engagement increases humor likelihood
        base_chance = 0.15
        if engagement_level > 0.7:
            base_chance = 0.3
        elif engagement_level > 0.5:
            base_chance = 0.2
        
        # Playful mood increases chance
        if mood == "playful":
            base_chance += 0.2
        elif mood == "concerned" or mood == "focused":
            base_chance -= 0.1
        
        return random.random() < base_chance
    
    def get_tech_joke(self) -> str:
        """Get a random tech joke."""
        return random.choice(self.tech_jokes)
    
    def get_dry_wit(self, situation: str) -> Optional[str]:
        """
        Get a dry wit response for a situation.
        
        Args:
            situation: The situation type
            
        Returns:
            A witty remark or None
        """
        if situation in self.dry_wit:
            return random.choice(self.dry_wit[situation])
        return None
    
    def get_self_deprecating(self) -> str:
        """Get a self-deprecating AI humor remark."""
        return random.choice(self.self_deprecating)
    
    def get_witty_aside(self) -> str:
        """Get a witty aside to add to a response."""
        return random.choice(self.witty_asides)
    
    def detect_humor_opportunity(self, text: str) -> Optional[Tuple[str, str]]:
        """
        Detect if there's an opportunity for situational humor.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Tuple of (situation_type, witty_response) or None
        """
        text_lower = text.lower()
        
        # Check for bug-related discussions
        if any(word in text_lower for word in ["bug", "error", "crash", "broken"]):
            return ("bug_found", self.get_dry_wit("bug_found"))
        
        # Check for deadline mentions
        if any(word in text_lower for word in ["deadline", "due", "urgent", "asap"]):
            return ("deadline", self.get_dry_wit("deadline"))
        
        # Check for legacy code discussions
        if any(phrase in text_lower for phrase in ["legacy code", "old code", "legacy system", "technical debt"]):
            return ("legacy_code", self.get_dry_wit("legacy_code"))
        
        # Check for slow computer/performance issues
        if any(phrase in text_lower for phrase in ["slow", "taking forever", "hanging", "frozen"]):
            return ("slow_computer", self.get_dry_wit("slow_computer"))
        
        return None
    
    def add_humor_to_response(self, response: str, context: str, 
                              mood: str = "neutral", 
                              engagement_level: float = 0.5) -> str:
        """
        Potentially add humor to a response.
        
        Args:
            response: Original response
            context: Conversation context
            mood: Current mood
            engagement_level: Engagement level
            
        Returns:
            Response, possibly with humor added
        """
        if not self.should_use_humor(context, mood, engagement_level):
            return response
        
        # Try to find situational humor
        humor_opportunity = self.detect_humor_opportunity(context)
        if humor_opportunity:
            situation, wit = humor_opportunity
            if wit:
                # Add the wit naturally
                return f"{response}\n\n{wit}"
        
        # Small chance to add a witty aside
        if random.random() < 0.1:
            aside = self.get_witty_aside()
            # Insert aside naturally if response is long enough
            sentences = response.split(". ")
            if len(sentences) > 2:
                insert_point = random.randint(1, len(sentences) - 1)
                sentences.insert(insert_point, aside)
                return ". ".join(sentences)
        
        return response
    
    def generate_humor_response(self, request_type: str) -> str:
        """
        Generate a full humor response for explicit joke requests.
        
        Args:
            request_type: Type of humor requested
            
        Returns:
            A humorous response
        """
        if request_type == "joke":
            joke = self.get_tech_joke()
            intros = [
                "Alright, here's one:",
                "Since you asked:",
                "Can't resist a good tech joke:",
                "Here goes:"
            ]
            return f"{random.choice(intros)} {joke}"
        
        elif request_type == "entertain":
            return f"{self.get_self_deprecating()} But enough about my limitations - what can I actually help you with?"
        
        return self.get_tech_joke()
    
    def get_error_wit(self, error_type: str = "general_error") -> str:
        """
        Get a dry wit response for error/failure situations (Recommendation 8).
        
        Args:
            error_type: Type of error (general_error, not_found, confusion, 
                       timeout, limitation, recovery, self_deprecating_error)
            
        Returns:
            A witty error response
        """
        if error_type in self.error_wit:
            return random.choice(self.error_wit[error_type])
        return random.choice(self.error_wit["general_error"])
    
    def add_error_wit_to_response(self, error_response: str, 
                                   error_type: str = "general_error") -> str:
        """
        Add witty prefix to an error response (Recommendation 8).
        
        Args:
            error_response: The original error message/explanation
            error_type: Type of error for appropriate wit selection
            
        Returns:
            Error response with witty prefix
        """
        wit = self.get_error_wit(error_type)
        return f"{wit} {error_response}"
    
    def get_recovery_wit(self) -> str:
        """
        Get a recovery phrase for trying again after an error.
        
        Returns:
            A recovery phrase
        """
        return random.choice(self.error_wit["recovery"])
    
    def soften_error_message(self, technical_error: str) -> str:
        """
        Transform a technical error into a more personable message.
        
        Args:
            technical_error: Raw technical error message
            
        Returns:
            A more human-friendly error message with personality
        """
        # Get appropriate wit based on error content
        error_lower = technical_error.lower()
        
        if "not found" in error_lower or "doesn't exist" in error_lower:
            error_type = "not_found"
        elif "timeout" in error_lower or "timed out" in error_lower:
            error_type = "timeout"
        elif "cannot" in error_lower or "unable" in error_lower:
            error_type = "limitation"
        else:
            error_type = "general_error"
        
        wit = self.get_error_wit(error_type)
        
        # Simplify the technical error
        simplified = technical_error
        if ":" in technical_error:
            # Take the part after the colon which is usually more readable
            parts = technical_error.split(":")
            if len(parts[-1].strip()) > 10:
                simplified = parts[-1].strip()
        
        return f"{wit} Here's what happened: {simplified}"


# Singleton instance
_humor_integration = None

def get_humor_integration() -> HumorIntegration:
    """Get the singleton HumorIntegration instance."""
    global _humor_integration
    if _humor_integration is None:
        _humor_integration = HumorIntegration()
    return _humor_integration
