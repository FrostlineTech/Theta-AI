"""
Conversation Manager for Theta AI.
Handles conversation context and history.
"""

import json
import uuid
import logging
import re
from datetime import datetime
import random
from src.database.database_manager import DatabaseManager
from src.interface.topic_detection import TopicDetector
from src.interface.followup_generator import FollowupGenerator
from src.interface.consistency_checker import ConsistencyChecker

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConversationManager:
    """
    Enhanced manager for handling conversation context and history.
    Includes topic detection, follow-up generation, and consistency checking.
    """
    
    # Conversation transitional phrases for more natural responses
    CONTEXT_PHRASES = [
        "As we discussed earlier,",
        "Building on what we talked about,",
        "To continue our discussion on {topic},",
        "Going back to your question about {topic},",
        "Following up on our earlier point,",
        "Continuing from where we left off,",
        "Adding to what I mentioned earlier,",
        "To elaborate on my previous response,"
    ]
    
    # Phrases to reference previous user information
    REFERENCE_PHRASES = [
        "As you mentioned earlier about {topic},",
        "Given what you said about {topic},",
        "Based on your interest in {topic},",
        "Since you asked about {topic} previously,",
        "Considering your earlier question about {topic},"
    ]
    
    def __init__(self, session_id=None, context_window=10):
        """
        Initialize enhanced conversation manager.
        
        Args:
            session_id (str, optional): Session identifier. If not provided, a new one is created.
            context_window (int): Number of recent exchanges to keep in context
        """
        self.db_manager = DatabaseManager()
        self.session_id = session_id if session_id else str(uuid.uuid4())
        self.context_window = context_window
        self.conversation_history = []
        
        # Topic tracking
        self.topic_detector = TopicDetector()
        self.current_topics = []
        self.primary_topic = None
        
        # Follow-up generator
        self.followup_generator = FollowupGenerator()
        
        # Consistency checker
        self.consistency_checker = ConsistencyChecker()
        
        # Conversation state
        self.conversation_state = {
            'questions_asked': 0,
            'topics_covered': set(),
            'session_start': datetime.now(),
            'last_activity': datetime.now(),
            'user_preferences': {},
            'conversation_depth': 0,  # 0: new, 1: ongoing, 2: deep
            'emotional_state': 'neutral',  # neutral, positive, negative, stressed, curious
            'response_style': 'balanced',  # balanced, technical, empathetic, motivational, concise, narrative
            'recent_intents': [],  # Tracks last 3 intent classifications
            'personal_topic_count': 0,  # Count of personal topics discussed
            'technical_topic_count': 0,  # Count of technical topics discussed
            'narrative_mode': False  # Whether to use storytelling mode
        }
        
        # Load context
        self.load_context()
        
    def load_context(self):
        """
        Load existing conversation context from database or initialize new one.
        """
        try:
            # Get context from database
            self.conversation_history = self.db_manager.get_conversation_context(self.session_id)
            
            if not self.conversation_history:
                self.conversation_history = []
                logger.info(f"Created new conversation context for session {self.session_id}")
            else:
                logger.info(f"Loaded existing conversation context for session {self.session_id}")
                
        except Exception as e:
            logger.error(f"Error loading conversation context: {str(e)}")
            self.conversation_history = []
    
    def add_exchange(self, question, answer, conversation_id=None, web_search_used=False):
        """
        Add a conversation exchange and update conversation state.
        
        Args:
            question (str): User's question
            answer (str): AI's answer
            conversation_id (str, optional): ID of the conversation if available
            web_search_used (bool): Whether web search was used to generate the answer
            
        Returns:
            str: The conversation ID (either existing or newly created)
        """
        try:
            # Add to memory
            timestamp = datetime.now().isoformat()
            
            # Detect topics in the exchange
            exchange_topics = self.topic_detector.detect_topics(question, answer)
            self.update_topics(exchange_topics)
            
            # Record if web search was used
            if web_search_used:
                self.record_web_search_usage(question, answer)
            
            # Create enhanced exchange with metadata
            exchange = {
                "user": question,
                "assistant": answer,
                "timestamp": timestamp,
                "topics": exchange_topics,
                "web_search_used": web_search_used
            }
            
            self.conversation_history.append(exchange)
            
            # Trim history to context window
            if len(self.conversation_history) > self.context_window:
                self.conversation_history = self.conversation_history[-self.context_window:]
            
            # Save to database
            if conversation_id:
                conversation_id = self.db_manager.save_conversation(self.session_id, question, answer, conversation_id)
            else:
                conversation_id = self.db_manager.save_conversation(self.session_id, question, answer)
            
            # Save detected topics to the database
            if conversation_id and exchange_topics:
                self.db_manager.save_detected_topics(conversation_id, exchange_topics)
            
            # Save context with enhanced metadata
            self.db_manager.save_conversation_context(self.session_id, self.conversation_history)
            
            logger.info(f"Added conversation exchange for session {self.session_id}")
            return conversation_id
            
        except Exception as e:
            logger.error(f"Error adding conversation exchange: {str(e)}")
            return None
            
    def update_conversation_state(self, user_input, ai_response, intent=None):
        """
        Update conversation state based on current exchange.
        Detects emotional content and adjusts response style accordingly.
        
        Args:
            user_input (str): User's message
            ai_response (str): AI's response
            intent (str, optional): Detected intent from intent classifier
        """
        # Update basic metrics
        self.conversation_state['questions_asked'] += 1
        self.conversation_state['last_activity'] = datetime.now()
        
        # Track topics covered
        if self.current_topics:
            self.conversation_state['topics_covered'].update(self.current_topics)
        
        # Update conversation depth
        if self.conversation_state['questions_asked'] >= 5:
            self.conversation_state['conversation_depth'] = 2  # deep conversation
        elif self.conversation_state['questions_asked'] >= 2:
            self.conversation_state['conversation_depth'] = 1  # ongoing conversation
            
        # Update intent history
        if intent:
            self.conversation_state['recent_intents'].append(intent)
            # Keep only the most recent 3 intents
            if len(self.conversation_state['recent_intents']) > 3:
                self.conversation_state['recent_intents'] = self.conversation_state['recent_intents'][-3:]
        
        # Detect emotional content in user input
        emotional_state = self._detect_emotional_content(user_input)
        if emotional_state:
            self.conversation_state['emotional_state'] = emotional_state
        
        # Update topic type counts
        if intent == 'personal' or self._is_personal_topic(user_input):
            self.conversation_state['personal_topic_count'] += 1
        elif self._is_technical_topic(user_input):
            self.conversation_state['technical_topic_count'] += 1
            
        # Adjust response style based on conversation flow
        self._update_response_style()
        
        # Check if narrative mode should be activated
        if self._should_use_narrative_mode(user_input):
            self.conversation_state['narrative_mode'] = True
        else:
            self.conversation_state['narrative_mode'] = False
    
    def _detect_emotional_content(self, text):
        """
        Detect emotional content in user input.
        
        Args:
            text (str): User's message
            
        Returns:
            str or None: Detected emotional state or None
        """
        # Convert to lowercase for pattern matching
        text = text.lower()
        
        # Check for emotional indicators
        if re.search(r'\b(happy|excited|great|wonderful|love|enjoy|pleased|delighted|thrilled)\b', text):
            return 'positive'
        elif re.search(r'\b(sad|upset|unhappy|depressed|miserable|disappointed|regret|sorry)\b', text):
            return 'negative'
        elif re.search(r'\b(anxious|worried|stressed|overwhelmed|confused|unsure|concerned|nervous)\b', text):
            return 'stressed'
        elif re.search(r'\b(interested|curious|wonder|how does|explain|tell me about|what is|how to)\b', text):
            return 'curious'
        
        return None
    
    def _is_personal_topic(self, text):
        """
        Check if the topic is personal rather than technical.
        
        Args:
            text (str): User's message
            
        Returns:
            bool: True if personal topic detected
        """
        text = text.lower()
        return bool(re.search(r'\b(i feel|i am|i\'m|my|me|myself|i think|i believe|i want|i need|i\'ve|i have)\b', text))
    
    def _is_technical_topic(self, text):
        """
        Check if the topic is technical in nature.
        
        Args:
            text (str): User's message
            
        Returns:
            bool: True if technical topic detected
        """
        # Check for technical topic indicators using existing topic patterns
        technical_patterns = [
            r'\b(code|program|develop|software|function|class|api|algorithm)\b',
            r'\b(secur|hack|malware|virus|threat|vulnerab|breach|encrypt)\b',
            r'\b(network|server|database|cloud|hardware|system|protocol)\b',
            r'\b(bug|error|exception|debug|test|compile|runtime|syntax)\b'
        ]
        
        text = text.lower()
        for pattern in technical_patterns:
            if re.search(pattern, text):
                return True
        
        return False
    
    def _update_response_style(self):
        """
        Update response style based on conversation flow and emotional state.
        """
        recent_intents = self.conversation_state['recent_intents']
        
        # Count intent types
        personal_intent_count = recent_intents.count('personal')
        technical_intent_count = recent_intents.count('technical')
        question_intent_count = recent_intents.count('question')
        
        # Determine response style based on intent patterns and emotional state
        if personal_intent_count > 1 or self.conversation_state['personal_topic_count'] > 2:
            if self.conversation_state['emotional_state'] in ['negative', 'stressed']:
                self.conversation_state['response_style'] = 'empathetic'
            else:
                self.conversation_state['response_style'] = 'reflective'
        elif technical_intent_count > 1 or self.conversation_state['technical_topic_count'] > 2:
            self.conversation_state['response_style'] = 'technical'
        elif question_intent_count > 1 and self.conversation_state['emotional_state'] == 'curious':
            self.conversation_state['response_style'] = 'explanatory'
        elif self.conversation_state['emotional_state'] == 'stressed':
            self.conversation_state['response_style'] = 'calming'
        elif self.conversation_state['emotional_state'] == 'positive':
            self.conversation_state['response_style'] = 'enthusiastic'
        else:
            self.conversation_state['response_style'] = 'balanced'
    
    def update_topics(self, topics):
        """
        Update the current conversation topics.
        
        Args:
            topics (list): List of detected topics
        """
        if not topics:
            return
            
        # Update current topics list
        self.current_topics = topics
        
        # Set primary topic if available
        if topics:
            self.primary_topic = topics[0]
            
        # Add topics to conversation state
        if hasattr(self, 'conversation_state'):
            for topic in topics:
                self.conversation_state['topics_covered'].add(topic)
    
    def record_web_search_usage(self, question, answer):
        """
        Record when web search is used to answer a question for learning purposes.
        
        Args:
            question (str): User's question
            answer (str): Generated answer
        """
        try:
            # Log web search usage
            logger.info(f"Web search used for question: {question[:50]}...")
            
            # Add this to learning data for future training if available
            if hasattr(self, 'learning_data'):
                self.learning_data.append({
                    'question': question,
                    'answer': answer,
                    'web_search_used': True,
                    'timestamp': datetime.now().isoformat()
                })
            
        except Exception as e:
            logger.error(f"Error recording web search usage: {e}")

    def _should_use_narrative_mode(self, text):
        """
        Determine if narrative mode should be used based on user input.
        
        Args:
            text (str): User's message
            
        Returns:
            bool: True if narrative mode should be activated
        """
        text = text.lower()
        
        # Check if the user is asking about Theta's identity, history, or story
        identity_patterns = [
            r'\b(your (story|history|background|creation|origin))\b',
            r'\b(how (were|was) you (created|built|developed|made))\b',
            r'\b(tell me about (yourself|your development|your creator))\b',
            r'\b(what (is|was) (your|theta\'s) (purpose|origin|background))\b',
            r'\b(how (did|does) (theta|you) (work|function|operate))\b',
            r'\b((about|tell me about) (the|your) fragments)\b'
        ]
        
        for pattern in identity_patterns:
            if re.search(pattern, text):
                return True
        
        # Also check if the recent topics have been about Theta's identity
        if self.current_topics and any(topic in ['theta_identity', 'fragments', 'frostline'] for topic in self.current_topics):
            return True
            
        return False
    
    def get_formatted_context(self, include_current=False):
        """
        Get enhanced conversation context formatted for the model.
        Includes topic awareness and conversation depth.
        
        Args:
            include_current (bool): Whether to include the latest exchange
            
        Returns:
            str: Formatted conversation context
        """
        if not self.conversation_history:
            return ""
            
        history = self.conversation_history
        if not include_current and history:
            # Exclude the most recent exchange
            history = history[:-1]
            
        if not history:
            return ""
        
        # Build context with appropriate depth based on conversation state
        depth = self.conversation_state['conversation_depth']
        
        if depth == 0:  # New conversation
            # Keep it simple for new conversations
            context = "Previous exchange:\n"
            # Just include the last exchange
            if len(history) > 0:
                last = history[-1]
                context += f"User: {last['user']}\nAssistant: {last['assistant']}\n\n"
        
        elif depth == 1:  # Ongoing conversation
            # Include last few exchanges with topic awareness
            context = "Recent conversation:\n"
            # Include up to 3 recent exchanges
            recent = history[-3:] if len(history) > 3 else history
            for exchange in recent:
                context += f"User: {exchange['user']}\nAssistant: {exchange['assistant']}\n\n"
            
            # Add topic awareness if we have detected topics
            if self.primary_topic:
                context += f"Current topic: {self.primary_topic}\n\n"
        
        else:  # Deep conversation
            # Full context with topic awareness and consistency facts
            context = "Conversation history:\n"
            for exchange in history:
                context += f"User: {exchange['user']}\nAssistant: {exchange['assistant']}\n\n"
            
            # Add topic awareness
            if self.primary_topic:
                context += f"Primary conversation topic: {self.primary_topic}\n"
            
            if self.current_topics:
                context += f"Active topics: {', '.join(self.current_topics)}\n"
            
            # Add key facts for consistency
            facts = self.consistency_checker.get_recent_facts(limit=3)
            if facts:
                context += "\nKey facts mentioned:\n"
                for fact in facts:
                    context += f"- {fact}\n"
        
        return context
        
    def get_context_reference(self):
        """
        Get a natural language reference to the conversation context.
        Used to make AI responses more conversational by referring to context.
        
        Returns:
            str: A natural context reference phrase or empty string
        """
        # Only add references if we're in an ongoing conversation
        if self.conversation_state['questions_asked'] < 2:
            return ""
            
        # 50% chance to add a context reference
        if random.random() < 0.5:
            # Choose reference type based on conversation depth
            if self.primary_topic:
                if self.conversation_state['conversation_depth'] >= 2:
                    # Use more specific reference in deeper conversations
                    phrase = random.choice(self.REFERENCE_PHRASES)
                    return phrase.format(topic=self.primary_topic)
                else:
                    # Use general context phrase
                    phrase = random.choice(self.CONTEXT_PHRASES)
                    if "{topic}" in phrase:
                        return phrase.format(topic=self.primary_topic)
                    return phrase
        
        return ""
    
    def get_current_input(self):
        """
        Get the most recent user input.
        
        Returns:
            str: Most recent user input or empty string
        """
        if self.conversation_history and len(self.conversation_history) > 0:
            return self.conversation_history[-1]['user']
        return ""
        
    def detect_topics(self, question, answer=None):
        """Detect topics in the conversation for better context tracking"""
        # Combine the texts if both are provided
        if answer:
            text = question + " " + answer
        else:
            text = question
        topic_patterns = {
            'cybersecurity': [
                r'\b(secur|hack|malware|virus|threat|vulnerab|breach|encrypt|auth|firewall)\b',
                r'\b(password|phishing|ransomware|ddos|intrusion|zero-day|cve|exploit)\b'
            ],
            'programming': [
                r'\b(code|program|develop|software|function|class|object|method|api|interface)\b',
                r'\b(algorithm|variable|compiler|interpreter|framework|library|dependency|bug)\b',
                r'\b(python|javascript|java|c\+\+|ruby|go|rust|typescript|php|swift)\b'
            ],
            'networking': [
                r'\b(network|server|router|firewall|tcp|ip|dns|http|protocol|packet)\b',
                r'\b(subnet|gateway|vpn|vlan|nat|proxy|load balancer|bandwidth|latency)\b'
            ],
            'database': [
                r'\b(database|sql|query|table|schema|row|column|index|key|join|select)\b',
                r'\b(nosql|mongodb|postgres|mysql|oracle|sqlite|redis|elasticsearch)\b'
            ],
            'cloud': [
                r'\b(cloud|aws|azure|google cloud|docker|kubernetes|container|vm|iaas|saas|paas)\b',
                r'\b(serverless|lambda|ec2|s3|microservice|scaling|devops|ci/cd)\b'
            ],
            'cryptography': [
                r'\b(encrypt|decrypt|cipher|hash|md5|sha|rsa|aes|key|certificate|tls|ssl)\b'
            ],
            'operating_systems': [
                r'\b(linux|unix|windows|macos|android|ios|kernel|shell|bash|powershell)\b'
            ],
            'machine_learning': [
                r'\b(ai|ml|machine learning|neural network|deep learning|data science|algorithm)\b',
                r'\b(model|training|dataset|feature|classification|regression|clustering)\b'
            ],
            'web_development': [
                r'\b(html|css|javascript|dom|api|frontend|backend|fullstack|responsive)\b',
                r'\b(react|angular|vue|node|express|django|flask|rails|laravel)\b'
            ]
        }
        
        detected_topics = []
        text = text.lower()
        
        for topic, patterns in topic_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    detected_topics.append(topic)
                    break
        
        # Save topics to the conversation state
        current_topics = self.conversation_state.get('current_topics', [])
        all_topics = list(set(current_topics + detected_topics))[:5]  # Keep max 5 topics
        self.conversation_state['current_topics'] = all_topics
        self.current_topics = all_topics
        
        return all_topics
            
    def generate_followup_question(self):
        """
        Generate a follow-up question based on conversation context.
        
        Returns:
            str: Follow-up question or empty string
        """
        # Don't generate follow-ups for brand new conversations
        if self.conversation_state['questions_asked'] < 1:
            return ""
        
        # Get the previous AI response and conversation ID for context
        prev_response = ""
        conversation_id = None
        if self.conversation_history and len(self.conversation_history) > 0:
            prev_response = self.conversation_history[-1]['assistant']
            
            # Try to get the conversation ID from database based on last exchange
            try:
                connection = self.db_manager.get_connection()
                cursor = connection.cursor()
                
                cursor.execute(
                    """
                    SELECT id FROM conversations 
                    WHERE session_id = %s
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (self.session_id,)
                )
                
                result = cursor.fetchone()
                if result:
                    conversation_id = result[0]
                    
                cursor.close()
                connection.close()
                
            except Exception as e:
                logger.error(f"Error getting conversation ID for followup: {str(e)}")
        
        # Generate follow-up based on topic if available
        followup_question = ""
        if self.primary_topic:
            followup_question = self.followup_generator.generate_followup(
                self.primary_topic, prev_response
            )
        else:
            followup_question = self.followup_generator.generate_generic_followup()
        
        # Save the followup to database if we have a conversation ID
        if followup_question and conversation_id:
            self.db_manager.save_followup_suggestion(
                conversation_id, 
                followup_question, 
                self.primary_topic
            )
            
        return followup_question
            
    def get_response_style(self):
        """
        Get the current response style based on conversation state.
        
        Returns:
            dict: Response style parameters including style name, temperature, and attributes
        """
        style = self.conversation_state['response_style']
        narrative_mode = self.conversation_state['narrative_mode']
        
        # Default parameters
        style_params = {
            'style': style,
            'temperature': 0.7,
            'attributes': []
        }
        
        # Adjust parameters based on style
        if style == 'technical':
            style_params['temperature'] = 0.5
            style_params['attributes'] = ['precise', 'detailed', 'structured']
        elif style == 'empathetic':
            style_params['temperature'] = 0.8
            style_params['attributes'] = ['supportive', 'understanding', 'compassionate']
        elif style == 'reflective':
            style_params['temperature'] = 0.7
            style_params['attributes'] = ['thoughtful', 'insightful', 'balanced']
        elif style == 'calming':
            style_params['temperature'] = 0.6
            style_params['attributes'] = ['reassuring', 'clear', 'steady']
        elif style == 'enthusiastic':
            style_params['temperature'] = 0.8
            style_params['attributes'] = ['energetic', 'positive', 'engaging']
        elif style == 'explanatory':
            style_params['temperature'] = 0.6
            style_params['attributes'] = ['clear', 'educational', 'thorough']
        
        # Override for narrative mode
        if narrative_mode:
            style_params['style'] = 'narrative'
            style_params['temperature'] = 0.8
            style_params['attributes'] = ['storytelling', 'personal', 'reflective']
        
        return style_params
    
    def enhance_response(self, response):
        """
        Enhance AI response with conversation context references and style adjustments.
        
        Args:
            response (str): Original AI response
            
        Returns:
            str: Enhanced response with context references and appropriate style
        """
        # Only enhance if we have sufficient context
        if self.conversation_state['questions_asked'] < 2:
            return response
        
        enhanced = response
        
        # Get context reference
        context_ref = self.get_context_reference()
        
        # Add context reference if appropriate
        if context_ref and not enhanced.startswith(context_ref):
            enhanced = f"{context_ref} {enhanced}"
        
        # If in narrative mode, check if the response needs to be formatted as a story
        if self.conversation_state['narrative_mode'] and not self._has_narrative_elements(enhanced):
            # Add narrative framing if it's about Theta or fragments
            if any(term in enhanced.lower() for term in ['theta', 'fragment', 'dakota', 'frostline']):
                enhanced = self._add_narrative_framing(enhanced)
        
        return enhanced
    
    def _has_narrative_elements(self, text):
        """
        Check if text already has narrative/first-person elements.
        
        Args:
            text (str): Response text
            
        Returns:
            bool: True if narrative elements detected
        """
        narrative_indicators = [
            r'\b(I was|My|When I|During my|As I|I remember|I function|My development|Dakota created me)\b',
            r'\b(my (creation|development|design|function|purpose|role))\b',
            r'\b(My fragment|My capability|I coordinate)\b'
        ]
        
        for pattern in narrative_indicators:
            if re.search(pattern, text):
                return True
                
        return False
    
    def _add_narrative_framing(self, text):
        """
        Add narrative framing to a response.
        
        Args:
            text (str): Original response text
            
        Returns:
            str: Response with narrative framing
        """
        narrative_frames = [
            "From my perspective as Theta, {text}",
            "As Dakota's Alpha AI, I can share that {text}",
            "My experience at Frostline Solutions has taught me that {text}",
            "In my development as Theta AI, I've learned that {text}",
            "When I consider my role as the governing intelligence, {text}"
        ]
        
        # Select a random frame that fits well with the content
        frame = random.choice(narrative_frames)
        framed_text = frame.format(text=text)
        
        return framed_text
