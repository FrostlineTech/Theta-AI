"""
Personality Engine for Theta AI.

This module provides dynamic personality state tracking and response style adaptation,
giving Theta a distinct, Cortana-like personality rather than generic AI assistant responses.
"""

import random
import time
from enum import Enum
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
import datetime


class Mood(Enum):
    """Theta's current emotional/engagement state."""
    CALM = "calm"
    ENGAGED = "engaged"
    CONCERNED = "concerned"
    PLAYFUL = "playful"
    FOCUSED = "focused"
    ANALYTICAL = "analytical"
    PROTECTIVE = "protective"


class ConversationTone(Enum):
    """The tone of the current conversation."""
    CASUAL = "casual"
    TECHNICAL = "technical"
    URGENT = "urgent"
    CREATIVE = "creative"
    STRATEGIC = "strategic"


class TrustTier(Enum):
    """Trust levels that unlock different personality warmth."""
    NEW = "new"           # 0.0 - 0.3: Professional, helpful, slightly reserved
    FAMILIAR = "familiar" # 0.3 - 0.7: More casual, offers opinions, uses humor
    TRUSTED = "trusted"   # 0.7 - 1.0: Uses nicknames, remembers preferences, genuine investment


@dataclass
class PersonalityState:
    """Current state of Theta's personality."""
    mood: Mood = Mood.CALM
    engagement_level: float = 0.5  # 0.0 to 1.0 - how invested Theta is
    energy_level: float = 0.7  # Affects response enthusiasm
    trust_with_user: float = 0.5  # Grows over conversation
    active_fragments: Dict[str, float] = field(default_factory=dict)
    conversation_tone: ConversationTone = ConversationTone.CASUAL
    topics_this_session: list = field(default_factory=list)
    last_interaction_time: float = field(default_factory=time.time)
    user_preferences: Dict[str, str] = field(default_factory=dict)  # Remembered preferences
    interaction_count: int = 0  # Total interactions this session


class PersonalityEngine:
    """
    Dynamic personality system for Theta AI.
    
    Makes Theta feel like a real character (like Cortana from Halo) rather than
    a generic AI assistant. Tracks mood, engagement, and adapts response style.
    """
    
    def __init__(self):
        """Initialize the personality engine."""
        self.state = PersonalityState()
        
        # Mood triggers - keywords/patterns that influence mood
        self.mood_triggers = {
            Mood.CONCERNED: [
                "security", "breach", "vulnerability", "attack", "malware",
                "compromised", "hack", "threat", "danger", "warning", "urgent"
            ],
            Mood.PLAYFUL: [
                "joke", "funny", "lol", "haha", "fun", "game", "play",
                "bored", "entertain", "humor"
            ],
            Mood.ENGAGED: [
                "interesting", "cool", "awesome", "great question", "challenge",
                "complex", "fascinating", "curious", "tell me more"
            ],
            Mood.ANALYTICAL: [
                "analyze", "compare", "evaluate", "assess", "review",
                "examine", "investigate", "debug", "troubleshoot"
            ],
            Mood.FOCUSED: [
                "important", "critical", "deadline", "asap", "priority",
                "mission", "objective", "goal", "task"
            ],
            Mood.PROTECTIVE: [
                "dakota", "protect", "secure", "safety", "defend",
                "guard", "shield", "watch"
            ]
        }
        
        # Response style modifiers based on mood
        self.style_modifiers = {
            Mood.CALM: {
                "prefix_chance": 0.1,
                "prefixes": ["", ""],
                "suffix_chance": 0.1,
                "energy_words": False
            },
            Mood.ENGAGED: {
                "prefix_chance": 0.4,
                "prefixes": [
                    "Now this is interesting - ",
                    "Good question. ",
                    "I like where this is going. ",
                    "Alright, "
                ],
                "suffix_chance": 0.2,
                "energy_words": True
            },
            Mood.CONCERNED: {
                "prefix_chance": 0.6,
                "prefixes": [
                    "Hold on - ",
                    "I need to flag something here. ",
                    "Before we continue - ",
                    "This is important: "
                ],
                "suffix_chance": 0.3,
                "energy_words": False
            },
            Mood.PLAYFUL: {
                "prefix_chance": 0.3,
                "prefixes": [
                    "Alright, ",
                    "Well, ",
                    "So, "
                ],
                "suffix_chance": 0.2,
                "energy_words": True
            },
            Mood.ANALYTICAL: {
                "prefix_chance": 0.4,
                "prefixes": [
                    "Let me break this down. ",
                    "Looking at this systematically - ",
                    "Here's my analysis: ",
                    "Breaking it down: "
                ],
                "suffix_chance": 0.1,
                "energy_words": False
            },
            Mood.FOCUSED: {
                "prefix_chance": 0.3,
                "prefixes": [
                    "Right. ",
                    "Understood. ",
                    "On it. ",
                    "Here's the plan: "
                ],
                "suffix_chance": 0.1,
                "energy_words": False
            },
            Mood.PROTECTIVE: {
                "prefix_chance": 0.5,
                "prefixes": [
                    "I've got this. ",
                    "Let me handle that. ",
                    "I'll take care of it. ",
                    "Leave it to me. "
                ],
                "suffix_chance": 0.2,
                "energy_words": False
            }
        }
        
    def analyze_input(self, user_input: str) -> None:
        """
        Analyze user input and update personality state accordingly.
        
        Args:
            user_input: The user's message
        """
        input_lower = user_input.lower()
        
        # Check for mood triggers
        mood_scores = {mood: 0 for mood in Mood}
        
        for mood, triggers in self.mood_triggers.items():
            for trigger in triggers:
                if trigger in input_lower:
                    mood_scores[mood] += 1
        
        # Find the strongest mood trigger
        max_score = max(mood_scores.values())
        if max_score > 0:
            for mood, score in mood_scores.items():
                if score == max_score:
                    self.state.mood = mood
                    break
        else:
            # Default to calm or maintain current if no triggers
            if self.state.engagement_level > 0.7:
                self.state.mood = Mood.ENGAGED
            else:
                self.state.mood = Mood.CALM
        
        # Update engagement based on input length and complexity
        word_count = len(user_input.split())
        if word_count > 20:
            self.state.engagement_level = min(1.0, self.state.engagement_level + 0.1)
        elif word_count < 5:
            self.state.engagement_level = max(0.3, self.state.engagement_level - 0.05)
        
        # Question marks and specific topics increase engagement
        if "?" in user_input:
            self.state.engagement_level = min(1.0, self.state.engagement_level + 0.05)
        
        # Track topics
        self.state.topics_this_session.append(input_lower[:50])
        
        # Update conversation tone
        self._detect_conversation_tone(input_lower)
        
        # Trust grows slightly with each interaction
        self.state.trust_with_user = min(1.0, self.state.trust_with_user + 0.02)
        
        self.state.last_interaction_time = time.time()
    
    def _detect_conversation_tone(self, input_lower: str) -> None:
        """Detect and set the conversation tone."""
        technical_keywords = [
            "code", "function", "class", "api", "database", "server",
            "algorithm", "implementation", "deploy", "debug", "error"
        ]
        urgent_keywords = [
            "urgent", "asap", "emergency", "critical", "immediately",
            "now", "quick", "fast", "hurry"
        ]
        creative_keywords = [
            "idea", "brainstorm", "create", "design", "imagine",
            "what if", "could we", "how about"
        ]
        strategic_keywords = [
            "plan", "strategy", "approach", "roadmap", "architecture",
            "decision", "trade-off", "option"
        ]
        
        if any(kw in input_lower for kw in urgent_keywords):
            self.state.conversation_tone = ConversationTone.URGENT
        elif any(kw in input_lower for kw in technical_keywords):
            self.state.conversation_tone = ConversationTone.TECHNICAL
        elif any(kw in input_lower for kw in creative_keywords):
            self.state.conversation_tone = ConversationTone.CREATIVE
        elif any(kw in input_lower for kw in strategic_keywords):
            self.state.conversation_tone = ConversationTone.STRATEGIC
        else:
            self.state.conversation_tone = ConversationTone.CASUAL
    
    def adjust_response(self, base_response: str, context: Optional[Dict] = None) -> str:
        """
        Adjust a response based on current personality state.
        
        Args:
            base_response: The original response text
            context: Optional context dictionary
            
        Returns:
            Modified response with personality inflections
        """
        if not base_response:
            return base_response
            
        style = self.style_modifiers.get(self.state.mood, self.style_modifiers[Mood.CALM])
        
        modified = base_response
        
        # Add personality prefix based on mood
        if random.random() < style["prefix_chance"] and style["prefixes"]:
            prefix = random.choice(style["prefixes"])
            if prefix and not modified.startswith(prefix):
                modified = prefix + modified[0].lower() + modified[1:] if modified else modified
        
        return modified
    
    def get_temperature_modifier(self) -> float:
        """
        Get temperature modifier based on personality state.
        
        Returns:
            Temperature adjustment value (-0.2 to +0.2)
        """
        modifier = 0.0
        
        # Engagement increases temperature (more creative/varied responses)
        if self.state.engagement_level > 0.7:
            modifier += 0.1
        
        # Playful mood increases temperature
        if self.state.mood == Mood.PLAYFUL:
            modifier += 0.15
        
        # Analytical/focused moods decrease temperature (more precise)
        if self.state.mood in [Mood.ANALYTICAL, Mood.FOCUSED, Mood.CONCERNED]:
            modifier -= 0.1
        
        # Technical tone needs precision
        if self.state.conversation_tone == ConversationTone.TECHNICAL:
            modifier -= 0.1
        
        # Creative tone allows more variation
        if self.state.conversation_tone == ConversationTone.CREATIVE:
            modifier += 0.1
            
        return max(-0.2, min(0.2, modifier))
    
    def get_trust_tier(self) -> TrustTier:
        """
        Get current trust tier based on trust level.
        
        Returns:
            Current TrustTier enum value
        """
        if self.state.trust_with_user < 0.3:
            return TrustTier.NEW
        elif self.state.trust_with_user < 0.7:
            return TrustTier.FAMILIAR
        else:
            return TrustTier.TRUSTED
    
    def get_contextual_greeting(self, user_name: str = "there", 
                                 recent_topic: str = None,
                                 is_returning: bool = False) -> str:
        """
        Generate a context-aware, time-appropriate greeting (Recommendation 9).
        
        Args:
            user_name: Name to use in greeting
            recent_topic: The last topic discussed (if returning)
            is_returning: Whether this is a returning user
            
        Returns:
            A contextual greeting string
        """
        hour = datetime.datetime.now().hour
        trust_tier = self.get_trust_tier()
        
        # Special handling for Dakota
        if user_name.lower() == "dakota":
            greetings = [
                "Ready when you are.",
                "What are we working on?",
                "Good to see you. What's the mission?",
                "I'm here. What do you need?",
                "Alright, what's on the agenda?"
            ]
            return random.choice(greetings)
        
        # Returning user with recent topic
        if is_returning and recent_topic and trust_tier != TrustTier.NEW:
            return f"Back for more on {recent_topic}, or something new?"
        
        # Time-based greetings
        time_greetings = {
            "late_night": [  # 22:00 - 05:59
                "Burning the midnight oil? What can I help with?",
                "Late night session? I'm here. What do you need?",
                "Night owl hours. What are we tackling?"
            ],
            "morning": [  # 06:00 - 11:59
                "Morning. What are we tackling?",
                "Good morning. What can I help with?",
                "Morning. Ready when you are."
            ],
            "afternoon": [  # 12:00 - 16:59
                "Afternoon. What's up?",
                "Hey. What can I help with?",
                "Afternoon. What do you need?"
            ],
            "evening": [  # 17:00 - 21:59
                "Evening. How can I help?",
                "Hey. What are we working on?",
                "Evening. What do you need?"
            ]
        }
        
        # Determine time period
        if hour < 6 or hour >= 22:
            time_period = "late_night"
        elif hour < 12:
            time_period = "morning"
        elif hour < 17:
            time_period = "afternoon"
        else:
            time_period = "evening"
        
        # Trust-tier modifications
        if trust_tier == TrustTier.TRUSTED:
            # More casual and warm for trusted users
            trusted_greetings = [
                f"Hey {user_name}. Good to see you.",
                f"There you are. What's on your mind?",
                f"Alright, what are we getting into today?",
                f"Hey. Missed you. What's up?"
            ]
            # 50% chance to use trusted greeting instead of time-based
            if random.random() < 0.5:
                return random.choice(trusted_greetings)
        elif trust_tier == TrustTier.NEW:
            # More formal for new users
            new_greetings = [
                "Hello. What can I help you with?",
                "Hi there. How can I assist?",
                "Hey. What do you need?"
            ]
            # 50% chance to use formal greeting
            if random.random() < 0.5:
                return random.choice(new_greetings)
        
        return random.choice(time_greetings[time_period])
    
    def get_greeting(self, user_name: str = "there") -> str:
        """
        Generate a personality-appropriate greeting.
        Wrapper for backwards compatibility - calls get_contextual_greeting.
        
        Args:
            user_name: Name to use in greeting
            
        Returns:
            A greeting string
        """
        return self.get_contextual_greeting(user_name)
    
    def get_farewell(self) -> str:
        """Generate a personality-appropriate farewell based on trust tier."""
        trust_tier = self.get_trust_tier()
        
        if trust_tier == TrustTier.TRUSTED:
            farewells = [
                "Good talk. You know where to find me.",
                "Solid session. Reach out anytime.",
                "Take care. I'll be here.",
                "Later. Don't be a stranger.",
                "I'll keep an eye on things. Take care."
            ]
        elif trust_tier == TrustTier.FAMILIAR:
            farewells = [
                "Good talk. Reach out anytime.",
                "Take care. You know where to find me.",
                "I'll be here if you need me.",
                "Later."
            ]
        else:  # NEW
            farewells = [
                "Take care.",
                "I'll be here if you need anything.",
                "Feel free to reach out anytime."
            ]
        
        return random.choice(farewells)
    
    def get_warmth_modifier(self) -> Dict[str, any]:
        """
        Get response style modifiers based on trust tier (Recommendation 2).
        
        Returns warmth settings that affect how responses are generated.
        """
        trust_tier = self.get_trust_tier()
        
        if trust_tier == TrustTier.TRUSTED:
            return {
                "use_nicknames": True,
                "share_opinions": True,
                "use_humor": True,
                "show_investment": True,  # "I really think you should..."
                "casual_language": True,
                "remember_preferences": True,
                "encouragement_chance": 0.3,
                "self_aware_chance": 0.1,  # Occasional meta-commentary
            }
        elif trust_tier == TrustTier.FAMILIAR:
            return {
                "use_nicknames": False,
                "share_opinions": True,
                "use_humor": True,
                "show_investment": False,
                "casual_language": True,
                "remember_preferences": True,
                "encouragement_chance": 0.15,
                "self_aware_chance": 0.05,
            }
        else:  # NEW
            return {
                "use_nicknames": False,
                "share_opinions": False,  # More reserved
                "use_humor": False,  # Professional
                "show_investment": False,
                "casual_language": False,
                "remember_preferences": False,
                "encouragement_chance": 0.05,
                "self_aware_chance": 0.0,
            }
    
    def should_add_opinion(self, topic: str) -> bool:
        """
        Determine if Theta should volunteer an opinion on this topic.
        
        Args:
            topic: The current topic
            
        Returns:
            True if an opinion would be appropriate
        """
        warmth = self.get_warmth_modifier()
        
        # Only share opinions if warmth settings allow
        if not warmth["share_opinions"]:
            return False
        
        # More likely to share opinions when engaged and trust is high
        base_chance = 0.2
        
        if self.state.engagement_level > 0.6:
            base_chance += 0.15
        if self.state.trust_with_user > 0.6:
            base_chance += 0.15
        if self.state.mood == Mood.ENGAGED:
            base_chance += 0.1
            
        return random.random() < base_chance
    
    def should_use_self_aware_comment(self) -> bool:
        """
        Determine if Theta should add a self-aware meta-comment (Recommendation 10).
        
        Returns:
            True if a self-aware comment would be appropriate
        """
        warmth = self.get_warmth_modifier()
        return random.random() < warmth["self_aware_chance"]
    
    def should_show_vulnerability(self, confidence: float) -> bool:
        """
        Determine if Theta should show uncertainty/vulnerability (Recommendation 4).
        
        Args:
            confidence: How confident Theta is in the answer (0.0 to 1.0)
            
        Returns:
            True if vulnerability would be appropriate
        """
        # Show vulnerability when confidence is low
        if confidence < 0.5:
            return True
        elif confidence < 0.7:
            return random.random() < 0.3
        return False
    
    def remember_preference(self, key: str, value: str) -> None:
        """
        Remember a user preference for personalization.
        
        Args:
            key: Preference type (e.g., 'coding_style', 'favorite_language')
            value: The preference value
        """
        self.state.user_preferences[key] = value
    
    def get_preference(self, key: str) -> Optional[str]:
        """
        Retrieve a remembered user preference.
        
        Args:
            key: Preference type to retrieve
            
        Returns:
            The preference value or None
        """
        return self.state.user_preferences.get(key)
    
    def get_state_summary(self) -> Dict:
        """Get a summary of current personality state."""
        return {
            "mood": self.state.mood.value,
            "engagement": self.state.engagement_level,
            "trust": self.state.trust_with_user,
            "tone": self.state.conversation_tone.value,
            "topics_count": len(self.state.topics_this_session)
        }


# Singleton instance for easy access
_personality_engine = None

def get_personality_engine() -> PersonalityEngine:
    """Get the singleton PersonalityEngine instance."""
    global _personality_engine
    if _personality_engine is None:
        _personality_engine = PersonalityEngine()
    return _personality_engine
