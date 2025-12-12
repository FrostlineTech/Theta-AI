"""
Short-Term Memory for Theta AI.
Manages conversation context memory for better continuity.
"""

from datetime import datetime, timedelta
import re
import json

class ShortTermMemory:
    """
    Short-term memory module for maintaining conversation context.
    Provides contextual awareness across multiple turns of conversation.
    """
    
    def __init__(self, capacity=10):
        """
        Initialize the short-term memory.
        
        Args:
            capacity (int): Maximum number of exchanges to remember
        """
        self.capacity = capacity
        self.memory = []
        self.creation_time = datetime.now()
        self.last_access_time = datetime.now()
        self.memory_id = f"mem_{self.creation_time.strftime('%Y%m%d%H%M%S')}"
        self.entity_memory = {}  # Store entities mentioned in conversation
        self.topic_memory = {}   # Store topic frequency
    
    def add_exchange(self, user_input, ai_response, metadata=None):
        """
        Add a conversation exchange to memory.
        
        Args:
            user_input (str): User's message
            ai_response (str): AI's response
            metadata (dict, optional): Additional information about the exchange
        """
        # Update access time
        self.last_access_time = datetime.now()
        
        # Create exchange record
        exchange = {
            "user_input": user_input,
            "ai_response": ai_response,
            "timestamp": self.last_access_time.isoformat(),
            "exchange_id": len(self.memory),
            "metadata": metadata or {}
        }
        
        # Extract and store entities from this exchange
        self._extract_and_store_entities(user_input, ai_response)
        
        # Extract and store topics
        self._extract_and_store_topics(user_input, ai_response)
        
        # Add to memory
        self.memory.append(exchange)
        
        # Maintain capacity constraint
        if len(self.memory) > self.capacity:
            self.memory = self.memory[-self.capacity:]
            
    def get_recent_exchanges(self, count=None):
        """
        Get the most recent exchanges from memory.
        
        Args:
            count (int, optional): Number of exchanges to return. If None, returns all.
            
        Returns:
            list: Recent conversation exchanges
        """
        self.last_access_time = datetime.now()
        
        if count is None or count >= len(self.memory):
            return self.memory
        
        return self.memory[-count:]
    
    def get_formatted_context(self, max_exchanges=None):
        """
        Get formatted conversation context for the model.
        
        Args:
            max_exchanges (int, optional): Maximum number of exchanges to include
            
        Returns:
            str: Formatted conversation context
        """
        self.last_access_time = datetime.now()
        
        # Determine how many exchanges to include
        exchanges = self.get_recent_exchanges(max_exchanges)
        
        if not exchanges:
            return ""
            
        # Format context
        context = "Previous conversation:\n\n"
        for exchange in exchanges:
            context += f"User: {exchange['user_input']}\n"
            context += f"Assistant: {exchange['ai_response']}\n\n"
            
        return context
    
    def search_memory(self, query):
        """
        Search memory for relevant content.
        
        Args:
            query (str): Search query
            
        Returns:
            list: Matching exchanges
        """
        self.last_access_time = datetime.now()
        
        query_lower = query.lower()
        results = []
        
        for exchange in self.memory:
            user_input = exchange['user_input'].lower()
            ai_response = exchange['ai_response'].lower()
            
            if query_lower in user_input or query_lower in ai_response:
                results.append(exchange)
                
        return results
    
    def get_memory_age(self):
        """
        Get the age of this memory in seconds.
        
        Returns:
            float: Age in seconds
        """
        return (datetime.now() - self.creation_time).total_seconds()
    
    def get_entity(self, entity_type, name=None):
        """
        Get entities of a specific type from memory.
        
        Args:
            entity_type (str): Type of entity (name, location, date, etc.)
            name (str, optional): Specific entity name
            
        Returns:
            dict or list: Entity information or list of entities
        """
        if entity_type not in self.entity_memory:
            return None
            
        if name:
            # Return specific entity
            for entity in self.entity_memory[entity_type]:
                if entity['name'].lower() == name.lower():
                    return entity
            return None
        else:
            # Return all entities of this type
            return self.entity_memory[entity_type]
    
    def get_most_frequent_topics(self, limit=3):
        """
        Get the most frequently discussed topics.
        
        Args:
            limit (int): Maximum number of topics to return
            
        Returns:
            list: Most frequent topics
        """
        sorted_topics = sorted(self.topic_memory.items(), key=lambda x: x[1], reverse=True)
        return [topic for topic, count in sorted_topics[:limit]]
    
    def _extract_and_store_entities(self, user_input, ai_response):
        """
        Extract entities from conversation and store in memory.
        
        Args:
            user_input (str): User's message
            ai_response (str): AI's response
        """
        # Simple entity extraction (in a real system, use NER)
        # Extract names (capitalized words)
        names = re.findall(r'(?<![.?!]\s)(?<![A-Z]\.)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', user_input + " " + ai_response)
        
        # Extract dates
        dates = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', user_input + " " + ai_response)
        
        # Extract emails
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', user_input + " " + ai_response)
        
        # Store extracted entities
        self._store_entities('name', names)
        self._store_entities('date', dates)
        self._store_entities('email', emails)
    
    def _store_entities(self, entity_type, entities):
        """
        Store entities in memory.
        
        Args:
            entity_type (str): Type of entity
            entities (list): Extracted entities
        """
        if not entities:
            return
            
        if entity_type not in self.entity_memory:
            self.entity_memory[entity_type] = []
            
        for entity in entities:
            # Check if entity already exists
            exists = False
            for existing in self.entity_memory[entity_type]:
                if existing['name'] == entity:
                    existing['count'] += 1
                    existing['last_seen'] = datetime.now().isoformat()
                    exists = True
                    break
                    
            if not exists:
                self.entity_memory[entity_type].append({
                    'name': entity,
                    'count': 1,
                    'first_seen': datetime.now().isoformat(),
                    'last_seen': datetime.now().isoformat()
                })
    
    def _extract_and_store_topics(self, user_input, ai_response):
        """
        Extract topics from conversation and store in memory.
        
        Args:
            user_input (str): User's message
            ai_response (str): AI's response
        """
        # Simple keyword-based topic detection
        topics = {
            'cybersecurity': ['security', 'hack', 'breach', 'password', 'encryption', 'malware', 'virus', 'phishing'],
            'programming': ['code', 'program', 'software', 'function', 'algorithm', 'variable', 'class', 'object'],
            'networking': ['network', 'router', 'server', 'protocol', 'ip', 'dns', 'tcp', 'http'],
            'database': ['database', 'sql', 'query', 'table', 'record', 'schema', 'field', 'join'],
            'ai': ['ai', 'artificial intelligence', 'machine learning', 'neural network', 'ml', 'data science'],
            'personal': ['you', 'your', 'yourself', 'think', 'feel', 'opinion', 'preference']
        }
        
        combined_text = (user_input + " " + ai_response).lower()
        
        for topic, keywords in topics.items():
            if any(keyword in combined_text for keyword in keywords):
                if topic in self.topic_memory:
                    self.topic_memory[topic] += 1
                else:
                    self.topic_memory[topic] = 1
    
    def get_last_n_exchanges(self, n=1):
        """
        Get the last N exchanges.
        
        Args:
            n (int): Number of exchanges to return
            
        Returns:
            list: Last N exchanges
        """
        if not self.memory:
            return []
            
        return self.memory[-n:]
        
    def get_user_preferences(self):
        """
        Extract user preferences from conversation history.
        
        Returns:
            dict: User preferences
        """
        preferences = {}
        
        # Look for explicit preferences in conversation
        for exchange in self.memory:
            user_input = exchange['user_input'].lower()
            
            # Check for preference statements
            preference_match = re.search(r'i (?:like|prefer|want|love|enjoy) (.*?)(?:\.|\?|!|$)', user_input)
            if preference_match:
                preference = preference_match.group(1).strip()
                preferences[preference] = True
                
            # Check for dislike statements
            dislike_match = re.search(r'i (?:dislike|don\'t like|hate|don\'t want) (.*?)(?:\.|\?|!|$)', user_input)
            if dislike_match:
                dislike = dislike_match.group(1).strip()
                preferences[dislike] = False
                
        return preferences
        
    def to_dict(self):
        """
        Convert memory to dictionary for serialization.
        
        Returns:
            dict: Memory as dictionary
        """
        return {
            'memory_id': self.memory_id,
            'creation_time': self.creation_time.isoformat(),
            'last_access_time': self.last_access_time.isoformat(),
            'capacity': self.capacity,
            'exchanges': self.memory,
            'entity_memory': self.entity_memory,
            'topic_memory': self.topic_memory
        }
        
    def from_dict(self, data):
        """
        Load memory from dictionary.
        
        Args:
            data (dict): Memory data
        """
        self.memory_id = data.get('memory_id', self.memory_id)
        self.creation_time = datetime.fromisoformat(data.get('creation_time', self.creation_time.isoformat()))
        self.last_access_time = datetime.fromisoformat(data.get('last_access_time', self.last_access_time.isoformat()))
        self.capacity = data.get('capacity', self.capacity)
        self.memory = data.get('exchanges', self.memory)
        self.entity_memory = data.get('entity_memory', self.entity_memory)
        self.topic_memory = data.get('topic_memory', self.topic_memory)
        
    def save_to_json(self, filepath):
        """
        Save memory to JSON file.
        
        Args:
            filepath (str): Path to save file
        """
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
            
    def load_from_json(self, filepath):
        """
        Load memory from JSON file.
        
        Args:
            filepath (str): Path to load file
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                self.from_dict(data)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading memory from file: {e}")
            return False
        return True
