"""
Inference Improvements for Theta AI

This module implements recommended improvements for Theta AI's inference process, 
particularly focusing on web search integration, response quality, and performance.
"""

import os
import json
import torch
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InferenceImprovements:
    """
    Implements enhanced inference capabilities for Theta AI
    """
    
    def __init__(self, theta_interface=None):
        """
        Initialize the inference improvements with optional reference to ThetaInterface
        
        Args:
            theta_interface: Reference to ThetaInterface instance (optional)
        """
        self.theta_interface = theta_interface
        
        # Cache for frequently searched queries with TTL (time to live)
        self.search_cache = {}
        self.cache_ttl = {
            "weather": 60*30,  # 30 minutes for weather
            "news": 60*60,     # 1 hour for news
            "stocks": 60*5,    # 5 minutes for stocks
            "default": 60*60*24  # 24 hours default
        }
        
        # Load configuration
        self.config_path = Path(__file__).parent / "inference_config.json"
        self.config = self._load_config()
        
    def _load_config(self) -> Dict:
        """Load configuration or create default if not exists"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.error(f"Error loading config from {self.config_path}. Using defaults.")
        
        # Default configuration
        config = {
            "temperature": {
                "factual": 0.3,
                "creative": 0.8,
                "default": 0.6
            },
            "retrieval": {
                "max_docs": 5,
                "semantic_weight": 0.7,
                "keyword_weight": 0.3
            },
            "cache": {
                "enabled": True,
                "max_size": 1000
            }
        }
        
        # Create directory and save default config
        self.config_path.parent.mkdir(exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        return config
        
    def dynamic_temperature(self, query: str, uses_web_search: bool = False) -> float:
        """
        Dynamically adjust temperature based on query type and web search usage
        
        Args:
            query: The user's query
            uses_web_search: Whether web search is being used
            
        Returns:
            float: Appropriate temperature value
        """
        query_lower = query.lower()
        
        # Default to the mid-range temperature
        temperature = self.config["temperature"]["default"]
        
        # Lower temperature for factual queries
        if uses_web_search or any(term in query_lower for term in [
            "what is", "how to", "when did", "where is", "who is", "why does",
            "explain", "define", "calculate", "weather", "time", "date"
        ]):
            temperature = self.config["temperature"]["factual"]
            
        # Higher temperature for creative queries
        elif any(term in query_lower for term in [
            "imagine", "creative", "story", "generate", "write", "create",
            "fiction", "fantasy", "poem", "song", "design"
        ]):
            temperature = self.config["temperature"]["creative"]
            
        logger.info(f"Dynamic temperature set to {temperature} for query type")
        return temperature
        
    def hybrid_search(self, query: str, collection_name: str) -> List[Dict]:
        """
        Implement hybrid search combining semantic and keyword-based search
        
        Args:
            query: The user's query
            collection_name: Name of vector collection to search
            
        Returns:
            List[Dict]: Search results
        """
        if not self.theta_interface or not hasattr(self.theta_interface, "web_search"):
            logger.warning("No web search interface available for hybrid search")
            return []
        
        try:
            # This is a simplified implementation - in practice, you would:
            # 1. Perform vector search using embeddings
            # 2. Perform keyword search using BM25 or similar
            # 3. Combine results with weighted scoring
            
            semantic_weight = self.config["retrieval"]["semantic_weight"]
            keyword_weight = self.config["retrieval"]["keyword_weight"]
            
            # For now, delegate to the web search implementation
            search_results = self.theta_interface.web_search.search(query)
            
            # Future: Implement actual hybrid search logic here
            logger.info(f"Performed hybrid search with weights: semantic={semantic_weight}, keyword={keyword_weight}")
            
            return search_results
        except Exception as e:
            logger.error(f"Error in hybrid search: {str(e)}")
            return []
            
    def query_preprocessing(self, query: str) -> str:
        """
        Preprocess query to improve search relevance
        
        Args:
            query: Original user query
            
        Returns:
            str: Enhanced query for search
        """
        # Remove filler words that don't add semantic meaning
        filler_words = ["the", "a", "an", "is", "are", "was", "were", "be", "to", "of", "and", "that", "have", "with"]
        
        # Only remove filler words if query is longer than 5 words to preserve short queries
        tokens = query.lower().split()
        if len(tokens) > 5:
            important_tokens = [token for token in tokens if token not in filler_words]
            # Don't lose more than 30% of the original words
            if len(important_tokens) >= len(tokens) * 0.7:
                tokens = important_tokens
                
        # Future: Implement more sophisticated query expansion here
        
        enhanced_query = " ".join(tokens)
        if enhanced_query != query:
            logger.info(f"Preprocessed query: '{query}' -> '{enhanced_query}'")
            
        return enhanced_query
        
    def verify_response_accuracy(self, response: str, retrieved_content: List[str]) -> Tuple[float, str]:
        """
        Verify response against retrieved content for accuracy
        
        Args:
            response: Generated response
            retrieved_content: Content used for generation
            
        Returns:
            Tuple[float, str]: Confidence score and potentially corrected response
        """
        # This is a placeholder for a more sophisticated verification system
        # In a real implementation, you would:
        # 1. Extract claims from the generated response
        # 2. Check each claim against the retrieved content
        # 3. Calculate a confidence score
        # 4. Correct any inaccuracies
        
        confidence = 0.8  # Default high confidence
        corrected_response = response
        
        # Simple check: see if key phrases from the response appear in retrieved content
        response_sentences = [s.strip() for s in response.split('.') if s.strip()]
        
        for sentence in response_sentences:
            sentence_found = any(sentence.lower() in content.lower() for content in retrieved_content)
            if not sentence_found and len(sentence.split()) > 5:  # Only check substantive sentences
                confidence -= 0.1  # Reduce confidence for unverified claims
                if confidence < 0.4:  # Don't let it go too low
                    break
                    
        logger.info(f"Response verification confidence: {confidence:.2f}")
        
        # In a real implementation, you would correct inaccuracies here
        
        return confidence, corrected_response
        
    def adaptive_retrieval_count(self, query: str) -> int:
        """
        Dynamically determine how many documents to retrieve based on query complexity
        
        Args:
            query: User query
            
        Returns:
            int: Number of documents to retrieve
        """
        base_count = self.config["retrieval"]["max_docs"]
        query_tokens = query.split()
        
        # Simple heuristics - adjust count based on query length and complexity
        if len(query_tokens) <= 3:
            # Short queries might be ambiguous, get more context
            count = min(base_count + 2, 10)
        elif len(query_tokens) >= 10:
            # Longer queries tend to be more specific
            count = base_count + 1
        elif any(term in query.lower() for term in ["compare", "difference", "versus", "pros and cons"]):
            # Comparative queries need more documents to cover multiple viewpoints
            count = base_count + 3
        else:
            count = base_count
            
        logger.info(f"Adaptive retrieval count: {count} for query length {len(query_tokens)}")
        return count
        
    def manage_search_cache(self, query: str, result: Dict, query_type: str = "default") -> None:
        """
        Manage the search cache with TTL based on query type
        
        Args:
            query: The search query
            result: Search result to cache
            query_type: Type of query (weather, news, etc.) for TTL selection
        """
        if not self.config["cache"]["enabled"]:
            return
            
        # Simple cache management - in production, use a proper cache system
        import time
        current_time = time.time()
        
        # Add to cache with timestamp
        cache_entry = {
            "result": result,
            "timestamp": current_time,
            "type": query_type
        }
        
        # Normalize query for caching (lowercase, remove extra spaces)
        cache_key = " ".join(query.lower().split())
        self.search_cache[cache_key] = cache_entry
        
        # Prune cache if it exceeds max size
        if len(self.search_cache) > self.config["cache"]["max_size"]:
            # Remove oldest entries
            oldest_keys = sorted(self.search_cache.keys(), 
                               key=lambda k: self.search_cache[k]["timestamp"])[:100]
            for key in oldest_keys:
                del self.search_cache[key]
                
        logger.info(f"Added query to cache with TTL for type: {query_type}")
        
    def get_from_cache(self, query: str) -> Optional[Dict]:
        """
        Try to get result from cache if not expired
        
        Args:
            query: Search query
            
        Returns:
            Optional[Dict]: Cached result or None if not found/expired
        """
        if not self.config["cache"]["enabled"] or not self.search_cache:
            return None
            
        import time
        current_time = time.time()
        cache_key = " ".join(query.lower().split())
        
        if cache_key in self.search_cache:
            entry = self.search_cache[cache_key]
            ttl = self.cache_ttl.get(entry["type"], self.cache_ttl["default"])
            
            # Check if entry is still valid
            if current_time - entry["timestamp"] < ttl:
                logger.info(f"Cache hit for query: '{query[:30]}...'")
                return entry["result"]
            else:
                # Expired
                logger.info(f"Cache expired for query: '{query[:30]}...'")
                del self.search_cache[cache_key]
                
        return None
        
    def track_search_effectiveness(self, query: str, search_results: List[Dict], used_results: List[int]) -> None:
        """
        Track which search results were most useful for future optimization
        
        Args:
            query: The original query
            search_results: All search results returned
            used_results: Indices of results actually used in response
        """
        # In a production system, this would store data for later analysis
        # Here we just log it
        if not search_results:
            return
            
        total_results = len(search_results)
        used_count = len(used_results) if used_results else 0
        
        logger.info(f"Search effectiveness: Used {used_count}/{total_results} results for query '{query[:30]}...'")
        
        # Future: Save this data for analyzing and improving search relevance

# Function to install into ThetaInterface
def install_inference_improvements(theta_instance):
    """
    Install inference improvements into an existing ThetaInterface instance
    
    Args:
        theta_instance: The ThetaInterface instance to enhance
    """
    if not hasattr(theta_instance, "inference_improvements"):
        theta_instance.inference_improvements = InferenceImprovements(theta_instance)
        
        # Store original generate_response method
        original_generate = theta_instance.generate_response
        
        # Replace with enhanced version
        def enhanced_generate_response(self, prompt, max_length=150, temperature=0.7, 
                                     uses_web_search=False, **kwargs):
            """Enhanced response generation with dynamic temperature and quality filtering"""
            
            # Apply dynamic temperature based on query type
            adj_temperature = self.inference_improvements.dynamic_temperature(prompt, uses_web_search)
            
            # Call original method with adjusted temperature
            response = original_generate(prompt, max_length, adj_temperature, **kwargs)
            
            # In a full implementation, we would add post-generation verification here
            
            return response
            
        # Monkey patch the method
        theta_instance.generate_response = enhanced_generate_response.__get__(theta_instance)
        
        logger.info("Installed inference improvements into ThetaInterface")
        
    return theta_instance
