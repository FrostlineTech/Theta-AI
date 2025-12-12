"""
Theta AI Inference Improvements Package

This package contains enhancements for Theta AI's inference process,
focusing on web search integration, response quality, and performance optimization.

This package implements the following improvements:

1. Dynamic Temperature Control - Automatically adjusts temperature based on query type
2. Two-Stage RAG Pipeline - Separates retrieval and generation with verification
3. Hybrid Search Implementation - Combines semantic and keyword search methods
4. Query Preprocessing - Expands queries for better search relevance
5. Response Quality Filtering - Identifies and corrects potential hallucinations
6. Contextual Memory Management - Optimizes context window usage
7. Domain-Specific Embeddings - Uses specialized embedding models for different domains
8. Adaptive Retrieval Count - Dynamically adjusts document retrieval based on query complexity
9. Intelligent Caching System - Caches search results with appropriate TTL values
10. Continuous Learning Loop - Tracks search effectiveness to improve strategies
"""

from .inference_improvements import InferenceImprovements, install_inference_improvements
from .response_generation import EnhancedResponseGeneration, install_enhanced_response_generation
from .module_integration import integrate_all_improvements, verify_integration

# Primary integration function for all improvements
def apply_all_improvements(theta_instance):
    """Apply all inference improvements to the given ThetaInterface instance"""
    return integrate_all_improvements(theta_instance)

__all__ = [
    'InferenceImprovements', 
    'install_inference_improvements',
    'EnhancedResponseGeneration',
    'install_enhanced_response_generation',
    'integrate_all_improvements',
    'verify_integration',
    'apply_all_improvements'
]
