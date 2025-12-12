"""
Intent Classifier for Theta AI.
Classifies user input into different intent categories.
"""

import re

class IntentClassifier:
    """
    Classifies user intents based on input patterns.
    Helps direct the conversation flow based on identified intent.
    """
    
    # Intent types
    GREETING = "greeting"
    QUESTION = "question"
    COMMAND = "command"
    FAREWELL = "farewell"
    GRATITUDE = "gratitude"
    OPINION = "opinion"
    CLARIFICATION = "clarification"
    AFFIRMATION = "affirmation"
    NEGATION = "negation"
    SMALL_TALK = "small_talk"
    UNKNOWN = "unknown"
    
    def __init__(self):
        """Initialize the intent classifier with patterns."""
        # Define patterns for each intent
        self.intent_patterns = {
            self.GREETING: [
                r'\b(?:hello|hi|hey|greetings|good morning|good afternoon|good evening|sup|yo|howdy)\b',
                r'^(hi|hello|hey)[\s\W]*$'
            ],
            self.QUESTION: [
                r'\b(?:what|how|why|when|where|which|who|whose|whom|is|are|was|were|will|do|does|did|has|have|had|can|could|would|should|may|might)\b.+\?',
                r'\b(?:explain|describe|tell me|show me)\b',
                r'\?$'
            ],
            self.COMMAND: [
                r'\b(?:find|search|look up|get|calculate|compute|show|display|list|create|make|open|close|delete|remove|add|update|change|modify|set)\b',
                r'^(?:please|can you|could you|would you).+\b(?:find|search|get|show|tell)\b'
            ],
            self.FAREWELL: [
                r'\b(?:bye|goodbye|see you|talk to you later|farewell|take care|until next time|have a good day)\b',
                r'^(?:bye|goodbye|exit|quit)[\s\W]*$'
            ],
            self.GRATITUDE: [
                r'\b(?:thanks|thank you|appreciate it|grateful|appreciate your help|thx)\b',
                r'^(?:thanks|thank you|thx)[\s\W]*$'
            ],
            self.OPINION: [
                r'\b(?:think|believe|feel|opinion|view|thought|thoughts|perspective|consider|considered)\b',
                r'\b(?:do you think|what do you think|how do you feel|your opinion|your thoughts)\b'
            ],
            self.CLARIFICATION: [
                r'\b(?:mean|clarify|explain further|elaborate|confused|don\'t understand|what do you mean|not clear)\b',
                r'\b(?:repeat that|say again|what\?|huh\?)\b'
            ],
            self.AFFIRMATION: [
                r'\b(?:yes|yeah|yep|yup|sure|absolutely|definitely|correct|right|agreed|indeed|exactly)\b',
                r'^(?:yes|yeah|yep|yup|sure|ok)[\s\W]*$'
            ],
            self.NEGATION: [
                r'\b(?:no|nope|nah|not|don\'t|cannot|can\'t|won\'t|incorrect|wrong)\b',
                r'^(?:no|nope|nah)[\s\W]*$'
            ],
            self.SMALL_TALK: [
                r'\b(?:how are you|what\'s up|how\'s it going|what are you doing|who are you|tell me about yourself)\b',
                r'\b(?:weather|sports|news|movie|music|book|hobby|weekend|family)\b'
            ]
        }
        
        # Compile patterns for efficiency
        self.compiled_patterns = {}
        for intent, patterns in self.intent_patterns.items():
            self.compiled_patterns[intent] = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    
    def classify(self, text):
        """
        Classify the intent of the given text.
        
        Args:
            text (str): User input text to classify
            
        Returns:
            str: Classified intent type
        """
        # Normalize text
        text = text.strip()
        
        # Score each intent
        intent_scores = {intent: 0 for intent in self.intent_patterns.keys()}
        
        for intent, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                matches = pattern.findall(text)
                if matches:
                    # Higher weight for full matches vs partial matches
                    if any(match == text for match in matches):
                        intent_scores[intent] += 2
                    else:
                        intent_scores[intent] += 1
        
        # Apply intent-specific rules and boosting
        
        # Boost QUESTION if text ends with ?
        if text.endswith('?'):
            intent_scores[self.QUESTION] += 2
            
        # Boost COMMAND if starts with imperative verb
        imperative_verbs = ["find", "search", "get", "show", "tell", "list", "explain", "describe", "calculate"]
        first_word = text.lower().split()[0] if text else ""
        if first_word in imperative_verbs:
            intent_scores[self.COMMAND] += 1
            
        # Boost GREETING/FAREWELL if that's all the text contains
        if len(text.split()) <= 2:
            for greeting in ["hi", "hello", "hey", "greetings"]:
                if greeting in text.lower():
                    intent_scores[self.GREETING] += 1
                    
            for farewell in ["bye", "goodbye", "cya", "see you"]:
                if farewell in text.lower():
                    intent_scores[self.FAREWELL] += 1
        
        # Find the intent with the highest score
        max_score = max(intent_scores.values())
        if max_score == 0:
            return self.UNKNOWN
            
        # Get all intents with the maximum score
        max_intents = [intent for intent, score in intent_scores.items() if score == max_score]
        
        # Tiebreaker logic
        if len(max_intents) > 1:
            # Priority order for tie-breaking
            priority_order = [
                self.GREETING, 
                self.FAREWELL, 
                self.QUESTION, 
                self.COMMAND, 
                self.GRATITUDE,
                self.AFFIRMATION, 
                self.NEGATION, 
                self.CLARIFICATION, 
                self.OPINION, 
                self.SMALL_TALK
            ]
            
            for intent in priority_order:
                if intent in max_intents:
                    return intent
        
        # Return the first max intent if we get here
        return max_intents[0]
        
    def get_question_type(self, text):
        """
        Identify the type of question being asked.
        Only call this if the intent is QUESTION.
        
        Args:
            text (str): Question text
            
        Returns:
            str: Question type (what, how, why, etc.)
        """
        text_lower = text.lower().strip()
        
        # Check question types
        if re.search(r'\bwhat\b', text_lower):
            return "what"
        elif re.search(r'\bhow\b', text_lower):
            return "how"
        elif re.search(r'\bwhy\b', text_lower):
            return "why"
        elif re.search(r'\bwhen\b', text_lower):
            return "when"
        elif re.search(r'\bwhere\b', text_lower):
            return "where"
        elif re.search(r'\bwho\b', text_lower):
            return "who"
        elif re.search(r'\bwhich\b', text_lower):
            return "which"
        elif re.search(r'\bwhose\b', text_lower):
            return "whose"
        elif re.search(r'\b(is|are|was|were|do|does|did|can|could|would|should|will)\b', text_lower):
            return "yes_no"
            
        return "other"
        
    def extract_command_verb(self, text):
        """
        Extract the main command verb from a command intent.
        Only call this if the intent is COMMAND.
        
        Args:
            text (str): Command text
            
        Returns:
            str: The main verb of the command or None
        """
        text_lower = text.lower().strip()
        
        # Common command verbs
        command_verbs = [
            "find", "search", "look", "get", "retrieve", "fetch",
            "show", "display", "list", "present", "calculate", "compute",
            "create", "make", "build", "generate", "add", "insert",
            "update", "modify", "change", "alter", "set", "adjust",
            "delete", "remove", "erase", "clear"
        ]
        
        # Check for explicit command verbs
        for verb in command_verbs:
            if re.search(r'\b' + verb + r'\b', text_lower):
                return verb
                
        # Check for implicit commands with "please" or "can you"
        if "please" in text_lower or "can you" in text_lower or "could you" in text_lower:
            for verb in command_verbs:
                if re.search(r'\b' + verb + r'\b', text_lower):
                    return verb
                    
        return None
        
    def is_multi_intent(self, text):
        """
        Check if the text contains multiple intents.
        
        Args:
            text (str): Input text
            
        Returns:
            bool: True if multiple intents detected
        """
        # Check for sentence boundaries
        sentences = re.split(r'[.!?]\s+', text.strip())
        if len(sentences) <= 1:
            return False
            
        # Classify each sentence
        intents = [self.classify(sentence) for sentence in sentences if sentence.strip()]
        
        # Check if we have different intents
        return len(set(intents)) > 1
        
    def split_intents(self, text):
        """
        Split text into separate intents.
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of (sentence, intent) tuples
        """
        # Split into sentences
        sentences = re.split(r'([.!?])\s+', text.strip())
        
        # Recombine sentence with punctuation
        proper_sentences = []
        i = 0
        while i < len(sentences) - 1:
            if i + 1 < len(sentences) and sentences[i+1] in ".!?":
                proper_sentences.append(sentences[i] + sentences[i+1])
                i += 2
            else:
                proper_sentences.append(sentences[i])
                i += 1
                
        # Handle last element if it exists
        if i < len(sentences):
            proper_sentences.append(sentences[i])
            
        # Classify each sentence
        results = []
        for sentence in proper_sentences:
            if sentence.strip():
                intent = self.classify(sentence)
                results.append((sentence.strip(), intent))
                
        return results
