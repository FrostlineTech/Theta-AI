"""
Enhanced response generation for Theta AI.

This module provides improved response generation capabilities with dynamic temperature,
response verification, quality filtering, and personality integration.

Cortana-style personality is integrated through:
- Dynamic temperature based on personality state
- Fragment-influenced response modification
- Natural language transformations
"""

import os
import logging
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import personality modules (with fallback if not available)
try:
    from src.interface.personality_engine import get_personality_engine
    from src.interface.signature_phrases import make_natural, SignaturePhrases
    from src.interface.fragment_orchestrator import get_fragment_orchestrator
    from src.interface.humor_integration import get_humor_integration
    PERSONALITY_MODULES_AVAILABLE = True
except ImportError:
    PERSONALITY_MODULES_AVAILABLE = False
    logger.warning("Personality modules not available - using basic response generation")

class EnhancedResponseGeneration:
    """
    Enhanced response generation capabilities for Theta AI
    """
    
    def __init__(self, theta_interface=None):
        """
        Initialize the enhanced response generation with reference to ThetaInterface
        
        Args:
            theta_interface: Reference to ThetaInterface instance
        """
        self.theta_interface = theta_interface
        
        # Debug mode for detailed logging
        self.debug = os.environ.get('DEBUG_WEB_SEARCH', 'false').lower() == 'true'
        
        # Default configuration
        self.config = {
            "temperature": {
                "factual": 0.3,  # Lower temperature for factual responses
                "creative": 0.8,  # Higher temperature for creative content
                "default": 0.6    # Default temperature
            },
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.2,
            "verify_responses": True
        }
        
        # Tokens that indicate uncertainty or hedging in responses
        self.uncertainty_tokens = [
            "I think", "perhaps", "maybe", "possibly", "might", "could be",
            "I believe", "probably", "I'm not sure", "it seems", "likely"
        ]
        
    def generate_response(self, prompt: str, query_type: str = "general", 
                         uses_web_search: bool = False, **kwargs) -> str:
        """
        Generate a response with dynamic parameters based on query type
        
        Args:
            prompt: The prompt to generate from
            query_type: Type of query (factual, creative, general)
            uses_web_search: Whether web search was used
            **kwargs: Additional parameters for generation
            
        Returns:
            str: Generated response
        """
        if self.theta_interface is None:
            logger.error("No ThetaInterface instance available for response generation")
            return "Error: Response generation system not properly initialized."
        
        # 1. Dynamic Temperature Control
        temperature = self._get_dynamic_temperature(prompt, query_type, uses_web_search)
        kwargs['temperature'] = temperature
        
        # Set appropriate generation parameters based on query type
        top_p = kwargs.get('top_p', self.config['top_p'])
        if query_type == "factual" or uses_web_search:
            # More focused sampling for factual queries
            top_p = min(top_p, 0.8)
        
        top_k = kwargs.get('top_k', self.config['top_k'])
        repetition_penalty = kwargs.get('repetition_penalty', self.config['repetition_penalty'])
        
        # If factual query, increase repetition penalty to discourage fabrication
        if query_type == "factual" or uses_web_search:
            repetition_penalty = max(repetition_penalty, 1.3)
        
        # Override for creative queries - allow more diversity
        if query_type == "creative":
            top_p = max(top_p, 0.92)
            repetition_penalty = min(repetition_penalty, 1.1)  # Lower penalty for creative content
        
        # Log generation parameters if debug mode is enabled
        if self.debug:
            logger.info(f"Generation parameters: temp={temperature}, top_p={top_p}, "
                       f"top_k={top_k}, repetition_penalty={repetition_penalty}")
        
        # 2. Generate initial response using the original method
        # The original method is already optimized for general text generation
        try:
            # Call the original generate_response method from theta_interface
            # We're accessing the original method that was saved during monkey-patching
            if hasattr(self.theta_interface, "_original_generate_response"):
                original_method = self.theta_interface._original_generate_response
                response = original_method(prompt, temperature=temperature, top_p=top_p,
                                       top_k=top_k, repetition_penalty=repetition_penalty, **kwargs)
            else:
                # Fall back to standard method if original not saved
                response = self.theta_interface.generate_response(prompt, temperature=temperature, 
                                                              top_p=top_p, top_k=top_k, 
                                                              repetition_penalty=repetition_penalty, 
                                                              **kwargs)
        except Exception as e:
            logger.error(f"Error in response generation: {str(e)}")
            response = f"I'm sorry, I encountered an error while generating a response. Please try again."
            return response
            
        # 3. Verify response if using web search and verification is enabled
        if uses_web_search and self.config["verify_responses"] and hasattr(self.theta_interface, 'inference_improvements'):
            retrieved_content = kwargs.get('retrieved_content', [])
            if retrieved_content and len(retrieved_content) > 0:
                confidence, verified_response = self.theta_interface.inference_improvements.verify_response_accuracy(
                    response, retrieved_content)
                
                # If confidence is low, adjust the response
                if confidence < 0.6 and verified_response != response:
                    logger.info(f"Response verification adjusted output (confidence: {confidence:.2f})")
                    response = verified_response
                
        # 4. Response Quality Filtering
        response = self._filter_response_quality(response, query_type, uses_web_search)
        
        # 5. Apply personality transformations (natural language, fragment influence)
        response = self._apply_personality(response, prompt)
        
        return response
    
    def _get_dynamic_temperature(self, prompt: str, query_type: str, uses_web_search: bool) -> float:
        """
        Determine appropriate temperature based on query type, content, and personality state.
        
        Args:
            prompt: The prompt being processed
            query_type: Type of query (factual, creative, general)
            uses_web_search: Whether web search was used
        
        Returns:
            float: Appropriate temperature value
        """
        # Default temperature based on query type
        if query_type == "factual" or uses_web_search:
            temperature = self.config["temperature"]["factual"]
        elif query_type == "creative":
            temperature = self.config["temperature"]["creative"]
        else:
            temperature = self.config["temperature"]["default"]
            
        # Further adjust based on prompt content
        prompt_lower = prompt.lower()
        
        # Lower temperature for specific factual query indicators
        if any(term in prompt_lower for term in ["what is", "how to", "when did", "where is", 
                                              "define", "explain", "calculate"]):
            temperature = min(temperature, self.config["temperature"]["factual"] + 0.1)
            
        # Higher temperature for specific creative query indicators
        elif any(term in prompt_lower for term in ["imagine", "creative", "story", "generate", 
                                                "write", "create"]):
            temperature = max(temperature, self.config["temperature"]["creative"] - 0.1)
            
        # Additional adjustments for specific content types
        if "code" in prompt_lower or "programming" in prompt_lower:
            # Lower temperature for code generation to improve accuracy
            temperature = min(temperature, 0.4)
            
        if "weather" in prompt_lower or "temperature" in prompt_lower:
            # Very low temperature for weather queries to ensure factual responses
            temperature = 0.2
        
        # Apply personality-based temperature modifier
        if PERSONALITY_MODULES_AVAILABLE:
            try:
                personality = get_personality_engine()
                personality.analyze_input(prompt)
                temp_modifier = personality.get_temperature_modifier()
                temperature += temp_modifier
                
                if self.debug:
                    logger.info(f"Personality temperature modifier: {temp_modifier}")
            except Exception as e:
                logger.debug(f"Could not apply personality temperature: {e}")
        
        # Clamp temperature to valid range
        return max(0.1, min(1.0, temperature))
    
    def _apply_personality(self, response: str, prompt: str) -> str:
        """
        Apply personality transformations to make responses more natural.
        
        Args:
            response: The original response
            prompt: The original prompt
            
        Returns:
            Response with personality applied
        """
        if not PERSONALITY_MODULES_AVAILABLE:
            return response
        
        try:
            # Make response sound more natural (contractions, remove robotic phrases)
            response = make_natural(response)
            
            # Apply personality state adjustments
            personality = get_personality_engine()
            response = personality.adjust_response(response)
            
            # Update and apply fragment influence
            orchestrator = get_fragment_orchestrator()
            orchestrator.update_activations(prompt)
            response = orchestrator.modify_response(response, use_prefix=True)
            
            # Potentially add humor if appropriate
            humor = get_humor_integration()
            mood = personality.state.mood.value if hasattr(personality.state, 'mood') else 'neutral'
            engagement = personality.state.engagement_level if hasattr(personality.state, 'engagement_level') else 0.5
            response = humor.add_humor_to_response(response, prompt, mood, engagement)
            
            return response
            
        except Exception as e:
            logger.debug(f"Error applying personality: {e}")
            return response
        
    def _filter_response_quality(self, response: str, query_type: str, uses_web_search: bool) -> str:
        """
        Filter response for quality issues like uncertainty or hallucination markers
        
        Args:
            response: The generated response
            query_type: Type of query
            uses_web_search: Whether web search was used
            
        Returns:
            str: Filtered response
        """
        # For factual queries using web search, check for uncertainty markers
        if uses_web_search and query_type == "factual":
            response_lower = response.lower()
            
            # Count uncertainty markers
            uncertainty_count = sum(token.lower() in response_lower for token in self.uncertainty_tokens)
            
            # If too many uncertainty markers in a factual response, add a disclaimer
            if uncertainty_count >= 3:
                disclaimer = ("\n\nNote: This response contains some uncertainty. For the most accurate "
                             "information, you may want to verify with additional sources.")
                response += disclaimer
        
        # Future: Implement more sophisticated response quality filtering here
        # Such as checking for internal contradictions, removing unnecessary hedging, etc.
        
        return response
    
def install_enhanced_response_generation(theta_instance):
    """
    Install enhanced response generation into a ThetaInterface instance
    
    Args:
        theta_instance: The ThetaInterface instance to enhance
    """
    if not hasattr(theta_instance, "enhanced_response_generation"):
        theta_instance.enhanced_response_generation = EnhancedResponseGeneration(theta_instance)
        
        # Save original generate_response method
        if not hasattr(theta_instance, "_original_generate_response"):
            theta_instance._original_generate_response = theta_instance.generate_response
            
        # Replace with enhanced version
        def enhanced_generate_response(self, prompt, **kwargs):
            """Enhanced response generation with dynamic parameters"""
            # Determine if this query uses web search
            uses_web_search = hasattr(self, 'last_used_web_search') and self.last_used_web_search
            
            # Determine query type based on content
            query_type = kwargs.get('query_type', 'general')
            if not query_type or query_type == 'general':
                prompt_lower = prompt.lower()
                if any(term in prompt_lower for term in ["fact", "what is", "define", "explain", "when", "where"]):
                    query_type = "factual"
                elif any(term in prompt_lower for term in ["imagine", "creative", "story", "write", "poem"]):
                    query_type = "creative"
            
            # Use enhanced generation
            return self.enhanced_response_generation.generate_response(
                prompt, query_type=query_type, uses_web_search=uses_web_search, **kwargs)
                
        # Monkey patch the method
        theta_instance.generate_response = enhanced_generate_response.__get__(theta_instance)
        
        logger.info("Installed enhanced response generation into ThetaInterface")
        
    return theta_instance
