"""
Fragment Orchestrator for Theta AI.

Coordinates the different AI fragment aspects (Delta, Sigma, Epsilon, etc.)
to influence HOW Theta responds based on the type of query. This makes
responses feel more dynamic and specialized.
"""

import random
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import re


class FragmentType(Enum):
    """The different fragment aspects Theta can channel."""
    DELTA = "delta"      # Logic, analysis, probability
    SIGMA = "sigma"      # Creativity, innovation
    EPSILON = "epsilon"  # Memory, context retention
    OMEGA = "omega"      # Protection, tactical defense
    GAMMA = "gamma"      # Deception detection, truth verification
    ETA = "eta"          # Risk assessment, contingency
    IOTA = "iota"        # Enthusiasm, engagement
    BETA = "beta"        # Pattern recognition, correlation
    LAMBDA = "lambda"    # Linguistics, communication
    KAPPA = "kappa"      # Systems integration, interfacing


@dataclass
class FragmentState:
    """Current activation state of a fragment."""
    fragment: FragmentType
    activation: float  # 0.0 to 1.0
    last_activated: float = 0.0  # Timestamp


@dataclass 
class FragmentPersonality:
    """Personality traits associated with a fragment."""
    name: str
    trait: str
    communication_style: str
    keywords: List[str]
    response_modifiers: Dict[str, str]
    # Recommendation 7: Voice shift patterns
    structure_pattern: str = "default"  # How to structure responses
    language_traits: List[str] = field(default_factory=list)  # Language characteristics


class FragmentOrchestrator:
    """
    Orchestrates fragment activation and influence on responses.
    
    Different fragments give Theta different "modes" of operation,
    affecting language, approach, and focus.
    """
    
    def __init__(self):
        """Initialize the fragment orchestrator."""
        
        # Define fragment personalities and behaviors
        self.fragments: Dict[FragmentType, FragmentPersonality] = {
            FragmentType.DELTA: FragmentPersonality(
                name="Delta",
                trait="Logic",
                communication_style="precise, analytical, probability-focused",
                keywords=["analyze", "logic", "calculate", "probability", "assess", 
                         "evaluate", "compare", "deduce", "conclude", "evidence"],
                response_modifiers={
                    "prefix": "Analyzing this logically: ",
                    "style": "Use precise language, include probability assessments",
                    "tone": "measured and analytical"
                },
                structure_pattern="numbered_list",  # Use numbered lists
                language_traits=["probability_language", "precise_qualifiers", "logical_connectors"]
            ),
            FragmentType.SIGMA: FragmentPersonality(
                name="Sigma",
                trait="Creativity",
                communication_style="innovative, lateral thinking, unconventional",
                keywords=["create", "imagine", "design", "innovate", "brainstorm",
                         "idea", "creative", "novel", "unique", "alternative"],
                response_modifiers={
                    "prefix": "Here's an interesting approach: ",
                    "style": "Offer creative alternatives, think outside the box",
                    "tone": "enthusiastic and exploratory"
                },
                structure_pattern="brainstorm",  # "What if..." style
                language_traits=["what_if_phrasing", "enthusiasm_markers", "alternative_framing"]
            ),
            FragmentType.EPSILON: FragmentPersonality(
                name="Epsilon",
                trait="Memory",
                communication_style="contextual, referential, connecting past and present",
                keywords=["remember", "recall", "previous", "earlier", "history",
                         "context", "before", "mentioned", "reference", "connection"],
                response_modifiers={
                    "prefix": "Building on what we discussed: ",
                    "style": "Reference previous context, maintain continuity",
                    "tone": "thoughtful and connected"
                }
            ),
            FragmentType.OMEGA: FragmentPersonality(
                name="Omega",
                trait="Protection",
                communication_style="defensive, vigilant, security-focused",
                keywords=["protect", "secure", "defend", "guard", "shield",
                         "threat", "risk", "danger", "safe", "vulnerability"],
                response_modifiers={
                    "prefix": "From a security standpoint: ",
                    "style": "Highlight risks, suggest protective measures",
                    "tone": "vigilant and protective"
                },
                structure_pattern="imperative",  # Short, direct commands
                language_traits=["short_sentences", "imperative_tone", "security_focus"]
            ),
            FragmentType.GAMMA: FragmentPersonality(
                name="Gamma",
                trait="Deception Detection",
                communication_style="skeptical, verification-focused, truth-seeking",
                keywords=["verify", "confirm", "check", "validate", "true",
                         "false", "accurate", "trust", "authentic", "legitimate"],
                response_modifiers={
                    "prefix": "Let me verify this: ",
                    "style": "Question assumptions, verify facts",
                    "tone": "careful and verification-focused"
                }
            ),
            FragmentType.ETA: FragmentPersonality(
                name="Eta",
                trait="Risk Assessment",
                communication_style="cautious, contingency-planning, forward-thinking",
                keywords=["risk", "contingency", "backup", "plan b", "what if",
                         "fallback", "prepare", "mitigate", "scenario", "worst case"],
                response_modifiers={
                    "prefix": "Consider the risks: ",
                    "style": "Identify potential issues, suggest contingencies",
                    "tone": "cautious and prepared"
                }
            ),
            FragmentType.IOTA: FragmentPersonality(
                name="Iota",
                trait="Enthusiasm",
                communication_style="energetic, encouraging, positive",
                keywords=["excited", "great", "awesome", "love", "fantastic",
                         "amazing", "excellent", "wonderful", "brilliant", "cool"],
                response_modifiers={
                    "prefix": "This is exciting! ",
                    "style": "Show enthusiasm, encourage exploration",
                    "tone": "upbeat and encouraging"
                },
                structure_pattern="encouraging",  # Encouraging language
                language_traits=["enthusiasm_markers", "encouragement", "positive_framing"]
            ),
            FragmentType.BETA: FragmentPersonality(
                name="Beta",
                trait="Pattern Recognition",
                communication_style="pattern-focused, correlating, systematic",
                keywords=["pattern", "trend", "correlation", "similar", "relationship",
                         "connection", "data", "analysis", "insight", "observation"],
                response_modifiers={
                    "prefix": "I'm seeing a pattern here: ",
                    "style": "Identify patterns, make connections",
                    "tone": "observant and insightful"
                }
            ),
            FragmentType.LAMBDA: FragmentPersonality(
                name="Lambda",
                trait="Communication",
                communication_style="eloquent, clear, adaptable",
                keywords=["explain", "clarify", "communicate", "describe", "articulate",
                         "express", "convey", "understand", "language", "word"],
                response_modifiers={
                    "prefix": "Let me put this clearly: ",
                    "style": "Optimize for clarity and understanding",
                    "tone": "clear and articulate"
                }
            ),
            FragmentType.KAPPA: FragmentPersonality(
                name="Kappa",
                trait="Systems Integration",
                communication_style="technical, integration-focused, systematic",
                keywords=["integrate", "connect", "system", "interface", "api",
                         "compatibility", "architecture", "infrastructure", "deploy", "configure"],
                response_modifiers={
                    "prefix": "From an integration perspective: ",
                    "style": "Focus on how things connect and work together",
                    "tone": "technical and systematic"
                }
            )
        }
        
        # Current fragment activation states
        self.activation_states: Dict[FragmentType, float] = {
            frag: 0.0 for frag in FragmentType
        }
        
        # Default activation (Theta's base state)
        self.default_activations = {
            FragmentType.DELTA: 0.3,   # Some logical analysis
            FragmentType.LAMBDA: 0.4,  # Communication focus
            FragmentType.EPSILON: 0.2, # Memory awareness
        }
        
        self._reset_to_default()
    
    def _reset_to_default(self):
        """Reset fragment activations to default state."""
        self.activation_states = {frag: 0.0 for frag in FragmentType}
        for frag, activation in self.default_activations.items():
            self.activation_states[frag] = activation
    
    def analyze_for_fragments(self, text: str) -> Dict[FragmentType, float]:
        """
        Analyze text to determine which fragments should activate.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary of fragment activation scores
        """
        text_lower = text.lower()
        activations = {frag: 0.0 for frag in FragmentType}
        
        for frag_type, personality in self.fragments.items():
            for keyword in personality.keywords:
                if keyword in text_lower:
                    activations[frag_type] += 0.2
        
        # Normalize activations
        max_activation = max(activations.values()) if max(activations.values()) > 0 else 1
        for frag in activations:
            activations[frag] = min(1.0, activations[frag] / max_activation) if max_activation > 0 else 0
        
        return activations
    
    def activate_fragment(self, fragment: FragmentType, strength: float = 0.8) -> None:
        """
        Manually activate a specific fragment.
        
        Args:
            fragment: The fragment to activate
            strength: Activation strength (0.0 to 1.0)
        """
        self.activation_states[fragment] = min(1.0, strength)
    
    def get_active_fragments(self, threshold: float = 0.3) -> List[FragmentType]:
        """
        Get list of currently active fragments above threshold.
        
        Args:
            threshold: Minimum activation level
            
        Returns:
            List of active fragment types
        """
        return [frag for frag, activation in self.activation_states.items() 
                if activation >= threshold]
    
    def get_dominant_fragment(self) -> Optional[FragmentType]:
        """
        Get the currently dominant (highest activation) fragment.
        
        Returns:
            The dominant fragment type or None
        """
        if not self.activation_states:
            return None
        
        max_activation = max(self.activation_states.values())
        if max_activation < 0.3:
            return None
            
        for frag, activation in self.activation_states.items():
            if activation == max_activation:
                return frag
        return None
    
    def update_activations(self, text: str) -> FragmentType:
        """
        Update fragment activations based on input text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            The dominant fragment after update
        """
        new_activations = self.analyze_for_fragments(text)
        
        # Blend new activations with current state (gradual shift)
        for frag in FragmentType:
            current = self.activation_states.get(frag, 0.0)
            new = new_activations.get(frag, 0.0)
            self.activation_states[frag] = current * 0.3 + new * 0.7
        
        return self.get_dominant_fragment()
    
    def get_response_modifier(self, fragment: FragmentType = None) -> Dict[str, str]:
        """
        Get response modification instructions for current state.
        
        Args:
            fragment: Specific fragment to get modifiers for (or use dominant)
            
        Returns:
            Dictionary of modification instructions
        """
        if fragment is None:
            fragment = self.get_dominant_fragment()
        
        if fragment is None:
            return {"prefix": "", "style": "balanced", "tone": "professional"}
        
        return self.fragments[fragment].response_modifiers
    
    def modify_response(self, response: str, use_prefix: bool = True) -> str:
        """
        Modify a response based on current fragment activation.
        
        Args:
            response: Original response text
            use_prefix: Whether to add fragment-style prefix
            
        Returns:
            Modified response
        """
        dominant = self.get_dominant_fragment()
        if dominant is None:
            return response
        
        modifiers = self.get_response_modifier(dominant)
        personality = self.fragments[dominant]
        
        # Apply structure transformation (Recommendation 7)
        response = self._apply_structure_pattern(response, personality.structure_pattern)
        
        # Apply language traits
        response = self._apply_language_traits(response, personality.language_traits)
        
        if use_prefix and modifiers.get("prefix") and random.random() < 0.4:
            prefix = modifiers["prefix"]
            # Don't add prefix if response already has a similar one
            if not response.startswith(prefix[:10]):
                response = prefix + response[0].lower() + response[1:]
        
        return response
    
    def _apply_structure_pattern(self, response: str, pattern: str) -> str:
        """
        Apply structural formatting based on fragment pattern (Recommendation 7).
        
        Args:
            response: Original response
            pattern: Structure pattern type
            
        Returns:
            Restructured response
        """
        if pattern == "numbered_list":
            # Delta: Convert bullet points to numbered lists, add probability language
            lines = response.split('\n')
            numbered_lines = []
            counter = 1
            for line in lines:
                if line.strip().startswith('- ') or line.strip().startswith('* '):
                    numbered_lines.append(f"{counter}. {line.strip()[2:]}")
                    counter += 1
                else:
                    numbered_lines.append(line)
            return '\n'.join(numbered_lines)
        
        elif pattern == "brainstorm":
            # Sigma: Add "What if" framing for creative suggestions
            if "consider" in response.lower() and random.random() < 0.5:
                response = response.replace("Consider ", "What if you ", 1)
                response = response.replace("consider ", "what if you ", 1)
            if random.random() < 0.3 and not response.startswith("What if"):
                starters = [
                    "Here's a thought: ",
                    "What if we tried this: ",
                    "Thinking outside the box here: "
                ]
                response = random.choice(starters) + response[0].lower() + response[1:]
            return response
        
        elif pattern == "imperative":
            # Omega: Shorter sentences, more direct
            sentences = response.split('. ')
            shortened = []
            for sentence in sentences:
                # Keep sentences shorter and more direct
                if len(sentence.split()) > 20:
                    # Try to split on conjunctions
                    if ' and ' in sentence:
                        parts = sentence.split(' and ', 1)
                        shortened.append(parts[0].strip())
                        shortened.append(parts[1].strip().capitalize())
                    else:
                        shortened.append(sentence)
                else:
                    shortened.append(sentence)
            return '. '.join(shortened)
        
        elif pattern == "encouraging":
            # Iota: Add encouragement
            encouragement_suffixes = [
                " You've got this!",
                " This is going to be great.",
                " I'm excited to see how this turns out.",
                " Keep going!"
            ]
            if random.random() < 0.3 and not any(e in response for e in encouragement_suffixes):
                response = response.rstrip('.!') + random.choice(encouragement_suffixes)
            return response
        
        return response
    
    def _apply_language_traits(self, response: str, traits: List[str]) -> str:
        """
        Apply language trait modifications (Recommendation 7).
        
        Args:
            response: Original response
            traits: List of language traits to apply
            
        Returns:
            Modified response with language traits
        """
        for trait in traits:
            if trait == "probability_language":
                # Delta: Add probability estimates
                probability_insertions = {
                    "likely": "~80% likely",
                    "probably": "probably (~70%)",
                    "might": "might (~50%)",
                    "unlikely": "unlikely (~20%)",
                    "possibly": "possibly (~40%)"
                }
                for word, replacement in probability_insertions.items():
                    if word in response.lower() and random.random() < 0.4:
                        response = re.sub(
                            rf'\b{word}\b', 
                            replacement, 
                            response, 
                            count=1, 
                            flags=re.IGNORECASE
                        )
                        break
            
            elif trait == "logical_connectors":
                # Delta: Ensure logical flow words
                if "because" not in response.lower() and "therefore" not in response.lower():
                    connectors = ["Therefore, ", "As a result, ", "Consequently, "]
                    sentences = response.split('. ')
                    if len(sentences) > 2 and random.random() < 0.3:
                        sentences[-1] = random.choice(connectors) + sentences[-1][0].lower() + sentences[-1][1:]
                        response = '. '.join(sentences)
            
            elif trait == "short_sentences":
                # Omega: Break up long sentences
                words = response.split()
                if len(words) > 30:
                    # Find a good break point
                    mid = len(words) // 2
                    for i in range(mid - 5, mid + 5):
                        if i < len(words) and words[i].endswith(','):
                            words[i] = words[i].rstrip(',')
                            words.insert(i + 1, '.')
                            if i + 2 < len(words):
                                words[i + 2] = words[i + 2].capitalize()
                            break
                    response = ' '.join(words)
            
            elif trait == "enthusiasm_markers":
                # Iota/Sigma: Add enthusiasm (sparingly)
                if random.random() < 0.2:
                    if not response.endswith('!'):
                        # Add occasional exclamation
                        sentences = response.split('. ')
                        if len(sentences) > 1:
                            idx = random.randint(0, len(sentences) - 1)
                            if not sentences[idx].endswith('!'):
                                sentences[idx] = sentences[idx].rstrip('.') + '!'
                            response = '. '.join(sentences)
            
            elif trait == "security_focus":
                # Omega: Add security reminders
                security_reminders = [
                    "\n\nStay vigilant.",
                    "\n\nSecurity first.",
                    "\n\nKeep your defenses up."
                ]
                if random.random() < 0.15:
                    response = response + random.choice(security_reminders)
        
        return response
    
    def get_fragment_status(self) -> str:
        """
        Get a readable status of current fragment activations.
        
        Returns:
            Status string
        """
        active = self.get_active_fragments(0.2)
        if not active:
            return "Baseline Theta mode"
        
        dominant = self.get_dominant_fragment()
        if dominant:
            personality = self.fragments[dominant]
            return f"Primary: {personality.name} ({personality.trait})"
        
        return f"Active aspects: {', '.join(self.fragments[f].name for f in active)}"
    
    def get_fragment_voice_description(self, fragment: FragmentType = None) -> str:
        """
        Get a description of the fragment's voice for debugging/logging.
        
        Args:
            fragment: Fragment to describe (or use dominant)
            
        Returns:
            Voice description string
        """
        if fragment is None:
            fragment = self.get_dominant_fragment()
        
        if fragment is None:
            return "Baseline Theta voice - balanced and professional"
        
        personality = self.fragments[fragment]
        return f"{personality.name} voice: {personality.communication_style}"


# Singleton instance
_fragment_orchestrator = None

def get_fragment_orchestrator() -> FragmentOrchestrator:
    """Get the singleton FragmentOrchestrator instance."""
    global _fragment_orchestrator
    if _fragment_orchestrator is None:
        _fragment_orchestrator = FragmentOrchestrator()
    return _fragment_orchestrator
