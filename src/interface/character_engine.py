"""
Character Engine for Theta AI.

Unified integration of all personality features inspired by Theta (RvB) and Cortana (Halo).
This module coordinates personality, fragments, humor, and signature phrases to create
a cohesive, memorable AI character for public use.

Implements all 10 recommendations:
1. Holographic identity awareness
2. Trust-based warmth progression
3. Protective "got your back" moments
4. Vulnerability/uncertainty (RvB Theta's shyness)
5. Tactical briefing mode (Cortana-style)
6. Signature quirks/catchphrases
7. Dynamic fragment voice shifts
8. Dry wit in error situations
9. Contextual greetings
10. Self-aware meta-commentary
"""

import random
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

from src.interface.personality_engine import PersonalityEngine, TrustTier, Mood, get_personality_engine
from src.interface.fragment_orchestrator import FragmentOrchestrator, FragmentType, get_fragment_orchestrator
from src.interface.humor_integration import HumorIntegration, get_humor_integration
from src.interface.signature_phrases import SignaturePhrases, make_natural, add_personality_prefix


@dataclass
class ResponseContext:
    """Context for generating a response with personality."""
    user_input: str
    is_error: bool = False
    error_type: str = "general_error"
    is_complex: bool = False
    is_tactical: bool = False
    confidence: float = 0.8
    detected_risk: bool = False
    user_name: str = "there"
    recent_topic: str = None
    is_returning_user: bool = False


class CharacterEngine:
    """
    Unified character engine that coordinates all personality systems.
    
    Makes Theta feel like a real character (Theta from RvB + Cortana from Halo)
    rather than a generic AI assistant.
    """
    
    def __init__(self):
        """Initialize all personality subsystems."""
        self.personality = get_personality_engine()
        self.fragments = get_fragment_orchestrator()
        self.humor = get_humor_integration()
        
        # Signature quirk usage tracking
        self.quirk_usage = {
            "thinking": 0,
            "completion": 0,
            "complex_start": 0,
        }
        self.response_count = 0
        
        # Recommended quirk frequencies (percentage of responses)
        self.quirk_frequencies = {
            "thinking": 0.12,      # 12% of responses start with "Hm."
            "completion": 0.15,    # 15% of completions end with "There you go."
            "complex_start": 0.10, # 10% of complex explanations start with signature phrase
        }
    
    def process_input(self, user_input: str, context: ResponseContext = None) -> Dict:
        """
        Process user input and update all personality systems.
        
        Args:
            user_input: The user's message
            context: Optional response context
            
        Returns:
            Dictionary with personality state info
        """
        if context is None:
            context = ResponseContext(user_input=user_input)
        
        # Update personality state
        self.personality.analyze_input(user_input)
        
        # Update fragment activations
        dominant_fragment = self.fragments.update_activations(user_input)
        
        self.response_count += 1
        
        return {
            "mood": self.personality.state.mood.value,
            "trust_tier": self.personality.get_trust_tier().value,
            "dominant_fragment": dominant_fragment.value if dominant_fragment else None,
            "engagement": self.personality.state.engagement_level,
            "warmth": self.personality.get_warmth_modifier()
        }
    
    def get_greeting(self, user_name: str = "there", 
                     recent_topic: str = None,
                     is_returning: bool = False) -> str:
        """
        Get a contextual, personality-aware greeting (Recommendation 9).
        
        Args:
            user_name: User's name
            recent_topic: Last discussed topic
            is_returning: Whether user is returning
            
        Returns:
            Contextual greeting string
        """
        return self.personality.get_contextual_greeting(
            user_name=user_name,
            recent_topic=recent_topic,
            is_returning=is_returning
        )
    
    def enhance_response(self, response: str, context: ResponseContext) -> str:
        """
        Apply all character enhancements to a response.
        
        Args:
            response: Base response text
            context: Response context with metadata
            
        Returns:
            Enhanced response with personality
        """
        # Handle error responses specially (Recommendation 8)
        if context.is_error:
            response = self.humor.add_error_wit_to_response(response, context.error_type)
            return make_natural(response)
        
        # Apply fragment voice modifications (Recommendation 7)
        response = self.fragments.modify_response(response)
        
        # Add vulnerability if confidence is low (Recommendation 4)
        if self.personality.should_show_vulnerability(context.confidence):
            uncertainty_prefix = SignaturePhrases.get_humble_uncertainty()
            response = f"{uncertainty_prefix} {response}"
        
        # Add protective warning if risk detected (Recommendation 3)
        elif context.detected_risk:
            warning = SignaturePhrases.get_protective_warning()
            response = f"{warning}\n\n{response}"
        
        # Add tactical briefing prefix for complex/strategic queries (Recommendation 5)
        elif context.is_tactical:
            briefing = SignaturePhrases.get_tactical_briefing()
            response = f"{briefing}\n\n{response}"
        
        # Apply signature quirks (Recommendation 6)
        response = self._apply_signature_quirk(response, context)
        
        # Add self-aware comment occasionally (Recommendation 10)
        if self.personality.should_use_self_aware_comment():
            self_aware = SignaturePhrases.get_self_aware()
            response = f"{response}\n\n{self_aware}"
        
        # Add holographic awareness occasionally (Recommendation 1)
        if random.random() < 0.08:  # 8% chance
            holographic = SignaturePhrases.get_holographic_awareness()
            # Prepend for processing-type responses
            if any(word in context.user_input.lower() for word in ["calculate", "analyze", "process", "check"]):
                response = f"{holographic}\n\n{response}"
        
        # Make language natural (contractions, etc.)
        response = make_natural(response)
        
        return response
    
    def _apply_signature_quirk(self, response: str, context: ResponseContext) -> str:
        """
        Apply signature quirks at appropriate frequencies (Recommendation 6).
        
        Args:
            response: Current response
            context: Response context
            
        Returns:
            Response with potential quirk added
        """
        # Calculate current usage rates
        if self.response_count > 0:
            thinking_rate = self.quirk_usage["thinking"] / self.response_count
            completion_rate = self.quirk_usage["completion"] / self.response_count
            complex_rate = self.quirk_usage["complex_start"] / self.response_count
        else:
            thinking_rate = completion_rate = complex_rate = 0
        
        # Add "Hm." for thoughtful responses
        if (thinking_rate < self.quirk_frequencies["thinking"] and 
            context.is_complex and 
            random.random() < 0.3):
            quirk = SignaturePhrases.get_signature_quirk("thinking")
            if quirk and not response.startswith(quirk):
                response = f"{quirk} {response}"
                self.quirk_usage["thinking"] += 1
        
        # Add "Alright, here's the thing..." for complex explanations
        elif (complex_rate < self.quirk_frequencies["complex_start"] and 
              context.is_complex and
              random.random() < 0.25):
            quirk = SignaturePhrases.get_signature_quirk("complex_start")
            if quirk and not response.startswith(quirk):
                response = f"{quirk} {response[0].lower()}{response[1:]}"
                self.quirk_usage["complex_start"] += 1
        
        # Add "There you go." for successful completions
        if (completion_rate < self.quirk_frequencies["completion"] and 
            not context.is_error and
            len(response) > 50 and  # Substantial response
            random.random() < 0.2):
            quirk = SignaturePhrases.get_signature_quirk("completion")
            if quirk and not response.endswith(quirk):
                response = f"{response}\n\n{quirk}"
                self.quirk_usage["completion"] += 1
        
        return response
    
    def get_introduction(self) -> str:
        """
        Get Theta's self-introduction with signature quirk (Recommendation 6).
        
        Returns:
            Introduction string
        """
        intro_quirk = SignaturePhrases.get_signature_quirk("intro")
        return f"I'm Theta. {intro_quirk} What can I help you with?"
    
    def get_farewell(self) -> str:
        """
        Get personality-appropriate farewell based on trust tier.
        
        Returns:
            Farewell string
        """
        return self.personality.get_farewell()
    
    def handle_error(self, error_message: str, error_type: str = "general_error") -> str:
        """
        Handle an error with appropriate wit and personality (Recommendation 8).
        
        Args:
            error_message: The error message/explanation
            error_type: Type of error for wit selection
            
        Returns:
            Personable error response
        """
        return self.humor.soften_error_message(error_message)
    
    def should_warn_user(self, user_input: str) -> Tuple[bool, Optional[str]]:
        """
        Check if we should proactively warn the user (Recommendation 3).
        
        Args:
            user_input: The user's message
            
        Returns:
            Tuple of (should_warn, warning_message)
        """
        input_lower = user_input.lower()
        
        # Security-related warnings
        security_triggers = [
            ("password", "I'd be careful with passwords. Never share them in plain text."),
            ("api key", "Hold on - API keys should be kept secret. Don't paste them anywhere public."),
            ("secret", "Heads up - secrets and credentials should be handled carefully."),
            ("production", "Before touching production, make sure you have a rollback plan."),
            ("delete all", "Wait - that sounds destructive. Are you sure you want to proceed?"),
            ("drop table", "Whoa there. That's a big operation. Double-check before running."),
            ("sudo rm", "I'd be very careful with that command. It can cause serious damage."),
            ("format", "Formatting is permanent. Make sure you have backups."),
        ]
        
        for trigger, warning in security_triggers:
            if trigger in input_lower:
                return True, warning
        
        return False, None
    
    def detect_tactical_query(self, user_input: str) -> bool:
        """
        Detect if the query requires tactical/strategic response style.
        
        Args:
            user_input: The user's message
            
        Returns:
            True if tactical mode appropriate
        """
        input_lower = user_input.lower()
        tactical_triggers = [
            "how should i", "what's the best way", "strategy for",
            "approach to", "plan for", "options for", "alternatives",
            "trade-offs", "pros and cons", "compare", "which should i",
            "decision", "choose between", "evaluate"
        ]
        return any(trigger in input_lower for trigger in tactical_triggers)
    
    def detect_complexity(self, user_input: str) -> bool:
        """
        Detect if the query is complex/multi-part.
        
        Args:
            user_input: The user's message
            
        Returns:
            True if query is complex
        """
        # Complex if long or contains multiple questions/parts
        word_count = len(user_input.split())
        question_marks = user_input.count("?")
        has_and = " and " in user_input.lower()
        
        return word_count > 30 or question_marks > 1 or (has_and and word_count > 15)
    
    def get_state_summary(self) -> Dict:
        """
        Get a summary of current character state for debugging.
        
        Returns:
            State dictionary
        """
        return {
            "personality": self.personality.get_state_summary(),
            "fragment": self.fragments.get_fragment_status(),
            "trust_tier": self.personality.get_trust_tier().value,
            "response_count": self.response_count,
            "quirk_usage": self.quirk_usage
        }


# Singleton instance
_character_engine = None

def get_character_engine() -> CharacterEngine:
    """Get the singleton CharacterEngine instance."""
    global _character_engine
    if _character_engine is None:
        _character_engine = CharacterEngine()
    return _character_engine
