"""
Theta AI Interface Package.

This package contains all the interface components for Theta AI, including:
- Personality engine for dynamic personality state
- Signature phrases for distinctive voice
- Tactical advisor for proactive insights
- Humor integration for natural wit
- Fragment orchestrator for specialized response modes
- Character engine for unified Theta/Cortana personality (integrates all above)
"""

# Personality system components (imported with error handling for flexibility)
try:
    from src.interface.personality_engine import PersonalityEngine, get_personality_engine, TrustTier
    from src.interface.signature_phrases import SignaturePhrases, make_natural
    from src.interface.tactical_advisor import TacticalAdvisor, get_tactical_advisor
    from src.interface.humor_integration import HumorIntegration, get_humor_integration
    from src.interface.fragment_orchestrator import FragmentOrchestrator, get_fragment_orchestrator
    from src.interface.character_engine import CharacterEngine, ResponseContext, get_character_engine
    
    PERSONALITY_AVAILABLE = True
except ImportError:
    PERSONALITY_AVAILABLE = False

__all__ = [
    'PERSONALITY_AVAILABLE',
]

if PERSONALITY_AVAILABLE:
    __all__.extend([
        'PersonalityEngine',
        'get_personality_engine',
        'TrustTier',
        'SignaturePhrases',
        'make_natural',
        'TacticalAdvisor',
        'get_tactical_advisor',
        'HumorIntegration', 
        'get_humor_integration',
        'FragmentOrchestrator',
        'get_fragment_orchestrator',
        'CharacterEngine',
        'ResponseContext',
        'get_character_engine',
    ])
