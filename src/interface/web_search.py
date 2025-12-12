"""
Web Search Module for Theta AI.
Integrates with Brave Search API to provide web search capabilities.
"""

import os
import requests
import json
import logging
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WebSearchManager:
    """
    Manages web search functionality for Theta AI using the Brave Search API.
    
    Attributes:
        api_key (str): Brave Search API key.
        api_name (str): Name of the API for logging purposes.
        base_url (str): Base URL for the Brave Search API.
        headers (dict): HTTP headers for API requests.
    """
    
    def __init__(self):
        """Initialize the web search manager with Brave Search API settings."""
        # Load API key from environment variables
        self.api_key = os.environ.get('WEBSEARCH_API_KEY', '')
        self.api_name = os.environ.get('WEBSEARCH_API_NAME', 'Theta_API')
        
        if not self.api_key:
            logger.warning("No WEBSEARCH_API_KEY found in environment variables.")
        else:
            logger.info(f"WebSearchManager initialized with API key (first 4 chars): {self.api_key[:4]}...")
        
        # Log the environment variables for debugging
        logger.info(f"EXTERNAL_DATA_SOURCES environment variable: {os.environ.get('EXTERNAL_DATA_SOURCES', 'not set')}")
        
        self.base_url = "https://api.search.brave.com/res/v1/web/search"
        self.headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.api_key
        }
    
    def search(self, query: str, count: int = 5, country: str = "US", search_lang: str = "en") -> Dict[str, Any]:
        """
        Perform a web search using the Brave Search API.
        
        Args:
            query (str): The search query.
            count (int): Number of results to return (default: 5).
            country (str): Country code for search results (default: "US").
            search_lang (str): Language code for search results (default: "en").
        
        Returns:
            Dict[str, Any]: Dictionary containing search results or error information.
        """
        if not self.api_key:
            logger.error(f"Cannot perform web search: No API key configured. Query was: '{query}'")
            return {"error": "No API key configured for web search"}
        
        params = {
            "q": query,
            "count": count,
            "country": country,
            "search_lang": search_lang
        }
        
        try:
            logger.info(f"Performing Brave web search for query: '{query}' with API key: {self.api_key[:4]}...")
            logger.info(f"Request URL: {self.base_url} with params: {params}")
            
            response = requests.get(
                self.base_url,
                headers=self.headers,
                params=params,
                timeout=10  # Add timeout to avoid hanging
            )
            
            logger.info(f"Received response with status code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Search successful! Response contains {len(result)} top-level keys")
                if "web" in result and "results" in result["web"]:
                    logger.info(f"Found {len(result['web']['results'])} web results")
                return result
            else:
                logger.error(f"Web search failed with status {response.status_code}: {response.text}")
                return {
                    "error": f"Search API returned status {response.status_code}",
                    "details": response.text
                }
                
        except Exception as e:
            logger.error(f"Error during web search: {str(e)}")
            return {"error": f"Web search failed: {str(e)}"}
    
    def extract_relevant_content(self, search_results: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Extract the most relevant content from search results.
        
        Args:
            search_results (Dict[str, Any]): The raw search results from the API.
            
        Returns:
            List[Dict[str, str]]: List of dictionaries containing title, description, and URL.
        """
        extracted_results = []
        
        if "error" in search_results:
            return [{"error": search_results["error"]}]
        
        try:
            if "web" in search_results and "results" in search_results["web"]:
                for result in search_results["web"]["results"]:
                    extracted_results.append({
                        "title": result.get("title", ""),
                        "description": result.get("description", ""),
                        "url": result.get("url", "")
                    })
            
            # Add news results if available
            if "news" in search_results and "results" in search_results["news"]:
                for result in search_results["news"]["results"][:2]:  # Limit to top 2 news results
                    extracted_results.append({
                        "title": f"[NEWS] {result.get('title', '')}",
                        "description": result.get("description", ""),
                        "url": result.get("url", ""),
                        "published": result.get("age", "")
                    })
            
            # Add discussions if available (e.g. forum posts)
            if "discussions" in search_results and "results" in search_results["discussions"]:
                for result in search_results["discussions"]["results"][:2]:
                    extracted_results.append({
                        "title": f"[DISCUSSION] {result.get('title', '')}",
                        "description": result.get("description", ""),
                        "url": result.get("url", "")
                    })
                    
        except Exception as e:
            logger.error(f"Error extracting content from search results: {str(e)}")
            extracted_results.append({"error": f"Failed to parse search results: {str(e)}"})
        
        return extracted_results
    
    def format_for_context(self, extracted_results: List[Dict[str, str]]) -> str:
        """
        Format extracted search results into a coherent context string.
        
        Args:
            extracted_results: List of dictionaries containing extracted content
            
        Returns:
            str: Formatted context string
        """
        if not extracted_results:
            return ""
        
        # Check if this is a weather or time query
        weather_patterns = ["weather", "temperature", "forecast", "climate"]
        time_patterns = ["time", "timezone", "local time", "current time"]
        
        is_weather_query = False
        is_time_query = False
        
        # Check the titles and snippets for patterns
        for result in extracted_results:
            title = result.get('title', '').lower()
            snippet = result.get('snippet', '').lower()
            
            if any(pattern in title or pattern in snippet for pattern in weather_patterns):
                is_weather_query = True
            if any(pattern in title or pattern in snippet for pattern in time_patterns):
                is_time_query = True
        
        # Special handling for weather and time queries
        if is_weather_query:
            return self._format_weather_results(extracted_results)
        elif is_time_query:
            return self._format_time_results(extracted_results)
            
        # Standard formatting for other queries
        formatted_text = "Here's what I found from the web:\n\n"
        
        for i, result in enumerate(extracted_results, 1):
            title = result.get('title', 'Untitled')
            snippet = result.get('snippet', 'No description available')
            url = result.get('url', '')
            
            formatted_text += f"{i}. {title}\n   {snippet[:150]}...\n\n"
            
        return formatted_text.strip()
        
    def _format_weather_results(self, results: List[Dict[str, str]]) -> str:
        """
        Special formatting for weather-related search results.
        
        Args:
            results: List of dictionaries containing extracted content
            
        Returns:
            str: Weather-formatted context string
        """
        import re
        # Extract temperature patterns: look for numbers followed by degrees or °F/°C
        temperature_patterns = [
            r'(\d+)\s*(?:degree|degrees|°)\s*(?:F|C|fahrenheit|celsius)?',
            r'(\d+)\s*(?:F|C)\b',
            r'(-?\d+)[\.,]?\d*\s*(?:°|deg)\s*(?:F|C|fahrenheit|celsius)'
        ]
        
        # Extract weather condition patterns
        condition_patterns = [
            r'(?:conditions?|weather)\s*(?:is|are|:)\s*(\w+(?:\s+\w+)?)',
            r'(sunny|cloudy|rainy|snowy|clear|overcast|partly cloudy|thunderstorms?|showers?)'
        ]
        
        # Extract location patterns
        location_patterns = [
            r'(?:in|for|at)\s+([A-Z][a-z]+(?:[\s,]+[A-Z][a-z]+){1,3})'
        ]
        
        temps = []
        conditions = []
        locations = []
        
        # Extract data from results
        for result in results:
            text = result.get('title', '') + ' ' + result.get('snippet', '')
            
            # Extract temperatures
            for pattern in temperature_patterns:
                matches = re.findall(pattern, text)
                if matches:
                    temps.extend(matches)
            
            # Extract conditions
            for pattern in condition_patterns:
                matches = re.findall(pattern, text)
                if matches:
                    conditions.extend(matches)
            
            # Extract locations
            for pattern in location_patterns:
                matches = re.findall(pattern, text)
                if matches:
                    locations.extend(matches)
        
        # Format the weather information
        formatted_text = ""
        
        # If we have temperature data
        if temps:
            formatted_text += f"Current temperature: {temps[0]}°"  
            if len(temps) > 1:
                formatted_text += f" (range: {min(temps)} - {max(temps)}°)"
            formatted_text += "\n"
        
        # If we have weather conditions
        if conditions:
            formatted_text += f"Weather conditions: {conditions[0].capitalize()}\n"
        
        # Add source attribution
        formatted_text += "\nBased on recent web search results. Conditions may change."
        
        return formatted_text
        
    def _format_time_results(self, results: List[Dict[str, str]]) -> str:
        """
        Special formatting for time-related search results.
        
        Args:
            results: List of dictionaries containing extracted content
            
        Returns:
            str: Time-formatted context string
        """
        import re
        # Extract time patterns
        time_patterns = [
            r'(\d{1,2}:\d{2}\s*(?:am|pm|AM|PM))',
            r'(\d{1,2}\s*(?:am|pm|AM|PM))',
            r'(\d{1,2}:\d{2})'
        ]
        
        # Extract timezone patterns
        timezone_patterns = [
            r'([A-Z]{3,4})\s*(?:time|timezone)',
            r'(?:timezone|time zone)\s*(?:is)?\s*([A-Z]{3,4})'
        ]
        
        times = []
        timezones = []
        
        # Extract data from results
        for result in results:
            text = result.get('title', '') + ' ' + result.get('snippet', '')
            
            # Extract times
            for pattern in time_patterns:
                matches = re.findall(pattern, text)
                if matches:
                    times.extend(matches)
            
            # Extract timezones
            for pattern in timezone_patterns:
                matches = re.findall(pattern, text)
                if matches:
                    timezones.extend(matches)
        
        # Format the time information
        formatted_text = ""
        
        # If we have time data
        if times:
            formatted_text += f"Current time: {times[0]}"
            if timezones and len(timezones) > 0:
                formatted_text += f" {timezones[0]}"
            formatted_text += "\n"
        
        # Add source attribution
        formatted_text += "\nBased on recent web search results."
        
        return formatted_text
