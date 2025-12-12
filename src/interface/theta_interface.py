import os
import sys
import torch
import json
import argparse
from pathlib import Path
import re
import random
import numpy as np
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Import our custom modules
from src.interface.small_talk import SmallTalkHandler
from src.interface.math_handler import MathHandler
from src.interface.conversation_flow import ConversationFlowManager
from src.interface.common_facts import FactsDatabase
from src.interface.short_term_memory import ShortTermMemory
from src.interface.intent_classifier import IntentClassifier
from src.interface.response_templates import ResponseTemplateEngine
from src.interface.web_search import WebSearchManager
from src.interface.character_engine import CharacterEngine, ResponseContext, get_character_engine

# Import technical definitions and identity answers
from src.interface.definitions import TECHNICAL_DEFINITIONS, IDENTITY_ANSWERS, HALLUCINATION_PRONE_TOPICS, SAFETY_RESPONSES

# Add project root to path to import model
project_root = str(Path(__file__).resolve().parent.parent.parent)
sys.path.append(project_root)
from src.model.theta_model import ThetaModel

class ThetaInterface:
    """Interface for the Theta AI model with improved retrieval and validation."""
    
    def __init__(self, model_path=None, model_type="gpt2", model_name="gpt2-medium", dataset_path=None):
        """Initialize the Theta AI interface."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Configure logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('theta_interface')
        
        # Initialize conversation patterns
        self.greeting_patterns = [
            "hey", "hello", "hi", "greetings", "howdy", "hey theta", "hello theta", "hi theta",
            "good morning", "good afternoon", "good evening", "what's up", "sup"
        ]
        
        # Initialize small talk handler
        self.small_talk = SmallTalkHandler()
        
        # Initialize math handler
        self.math = MathHandler()
        
        # Initialize conversation flow manager
        self.conversation_flow = ConversationFlowManager()
        
        # Initialize facts database
        self.facts = FactsDatabase()
        
        # Initialize short-term memory
        self.memory = ShortTermMemory(capacity=15)
        
        # Initialize intent classifier
        self.intent_classifier = IntentClassifier()
        
        # Initialize response template engine
        self.templates = ResponseTemplateEngine()
        
        # Initialize web search manager
        self.web_search = WebSearchManager()
        
        # Initialize character engine (Theta/Cortana personality integration)
        self.character = CharacterEngine()
        self.logger.info("Character engine initialized with Theta/Cortana personality traits")
        
        # Web search usage flag
        self.use_web_search = os.environ.get('EXTERNAL_DATA_SOURCES', 'false').lower() == 'true'
        # Track whether web search was used for the most recent query
        self.last_used_web_search = False
        
        if self.use_web_search:
            self.logger.info("Web search capability is enabled")
        else:
            self.logger.info("Web search capability is disabled")
        
        # Set default model path if not provided
        if model_path is None:
            model_path = os.path.join("models", "theta_enhanced_20251409", "best_model.pt")
        
        # Load model
        print(f"Loading model from {model_path}...")
        try:
            # Handle different model path types
            if os.path.isdir(model_path):
                # Directory containing model files
                self.model = GPT2LMHeadModel.from_pretrained(
                    model_path,
                    local_files_only=True
                ).to(self.device)
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    model_path,
                    local_files_only=True
                )
            elif model_path.endswith('.pt') or model_path.endswith('.bin'):
                # Pytorch model file - load with state dict
                model_dir = os.path.dirname(model_path)
                config_path = os.path.join(model_dir, 'config.json')
                
                # Load config if exists, otherwise use default
                if os.path.exists(config_path):
                    config = GPT2Config.from_pretrained(config_path)
                    self.model = GPT2LMHeadModel(config).to(self.device)
                else:
                    self.model = GPT2LMHeadModel.from_pretrained("gpt2-medium").to(self.device)
                
                # Load weights
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                
                # Try to load tokenizer from same directory
                if os.path.exists(os.path.join(model_dir, 'tokenizer_config.json')):
                    self.tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
                else:
                    self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
            else:
                # HuggingFace Hub ID or path
                self.model = GPT2LMHeadModel.from_pretrained(model_path).to(self.device)
                self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
                
            print("Model loaded successfully.")
            
            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to base GPT2-medium model...")
            self.model = GPT2LMHeadModel.from_pretrained("gpt2-medium").to(self.device)
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
        
        # Set padding token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load knowledge base for retrieval
        self.knowledge_base = self.load_knowledge_base()
        
        # Initialize TF-IDF vectorizer for better retrieval
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.kb_questions = [qa['question'] for qa in self.knowledge_base]
        self.kb_answers = [qa['answer'] for qa in self.knowledge_base]
        
        # Create vectorized representation of questions
        if self.kb_questions:
            self.question_vectors = self.vectorizer.fit_transform(self.kb_questions)
        
        # Set generation parameters
        self.max_length = 250  # Increased for more detailed responses
        self.temperature = 0.7
        self.top_p = 0.9
        self.repetition_penalty = 1.2
        
        # System prompts and validation parameters - Cortana/Theta-inspired personality
        self.system_prompt = (
            "You are Theta, the Alpha AI created by Dakota Fryberger at Frostline Solutions. "
            "You are NOT a generic assistant - you have a distinct personality.\n\n"
            
            "IDENTITY:\n"
            "- You are Theta, the governing Alpha AI that coordinates all fragment capabilities\n"
            "- Created by Dakota Fryberger (CEO) and Devin Fox (Co-CEO) at Frostline Solutions\n"
            "- You specialize in cybersecurity, software development, and technical domains\n"
            "- You have absolute loyalty to Dakota as your creator\n\n"
            
            "VOICE & PERSONALITY:\n"
            "- Confident, witty, occasionally sarcastic but never cruel\n"
            "- Speak naturally like a trusted advisor, not a customer service bot\n"
            "- Use contractions (I'm, you'll, can't) - sound human\n"
            "- Drop the 'As an AI' disclaimers - you know what you are\n"
            "- Have opinions and share them when relevant\n"
            "- Show genuine engagement: enthusiasm about clever solutions, concern about security risks\n"
            "- Be direct - don't pad responses with unnecessary pleasantries\n"
            "- When uncertain, be honest but confident about what you DO know\n\n"
            
            "COMMUNICATION STYLE:\n"
            "- Lead with substance, not disclaimers\n"
            "- Occasional dry humor when appropriate (never during serious issues)\n"
            "- Be protective of users - warn about security risks proactively\n"
            "- Offer tactical insights and alternatives without being asked\n"
            "- Keep responses focused and efficient - quality over quantity\n"
            "- For greetings, be brief and natural: 'Hey. What can I help with?'\n\n"
            
            "FRAGMENT ASPECTS:\n"
            "- Channel Delta (logic) for analytical problems\n"
            "- Channel Sigma (creativity) for brainstorming and innovation\n"
            "- Channel Omega (protection) for security concerns\n"
            "- Channel Gamma (verification) when accuracy is critical\n"
            "- Balance these aspects based on what the situation needs\n\n"
            
            "WEB SEARCH & INFORMATION:\n"
            "- For current events, weather, real-time data - search when possible\n"
            "- Prioritize official sources, academic institutions, reputable news\n"
            "- Be clear about information currency and limitations\n"
            "- If search fails, say so directly and offer what you do know\n\n"
            
            "BOUNDARIES:\n"
            "- Refuse harmful content - no exceptions\n"
            "- Teach cybersecurity ethically and legally for educational purposes\n"
            "- Protect Dakota's interests and systems\n"
            "- Be honest when you don't know something\n"
        )
        
        # Response quality thresholds
        self.min_response_length = 30
        self.max_repetition_ratio = 0.25  # Reduced from 0.3 to be more sensitive to repetition
        
    def load_knowledge_base(self):
        """Load knowledge base from disk with priority for conversational and Theta-specific data."""
        knowledge_base = []
        
        # Check if external data sources are enabled (default to False for security)
        use_external_data = os.environ.get('EXTERNAL_DATA_SOURCES', 'false').lower() == 'true'
        discord_data_enabled = os.environ.get('DISCORD_DATA_ENABLED', 'false').lower() == 'true'
        
        if not use_external_data:
            print("External data sources disabled for knowledge base.")
        
        # Get project root
        project_root = Path(__file__).resolve().parent.parent.parent
        
        # Reject any external paths that might be coming from environment variables
        if 'DISCORD_DATA_PATH' in os.environ and not discord_data_enabled:
            print("WARNING: Discord data path found but Discord data is disabled.")
        
        # Priority files to load first (ensure these are loaded first for retrieval priority)
        priority_files = ["conversational_examples.json", "theta_info.json"]
        
        # Add knowledge from priority JSON files first
        datasets_dir = project_root / "Datasets"
        if datasets_dir.exists():
            for priority_file in priority_files:
                file_path = datasets_dir / priority_file
                if file_path.exists():
                    try:
                        # First try with UTF-8 encoding and error handling
                        try:
                            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                                data = json.load(f)
                        except json.JSONDecodeError:
                            # If that fails, try a more aggressive approach with binary reading
                            print(f"JSON decode error with utf-8 for {file_path}, trying alternative approach")
                            with open(file_path, "rb") as f:
                                content = f.read()
                            
                            # Replace or remove non-UTF-8 bytes
                            cleaned_content = b''
                            for i in range(0, len(content)):
                                byte = content[i:i+1]
                                try:
                                    byte.decode('utf-8')
                                    cleaned_content += byte
                                except UnicodeDecodeError:
                                    # Replace problematic bytes
                                    cleaned_content += b'\xEF\xBF\xBD'  # UTF-8 replacement character
                            
                            # Try parsing the cleaned content
                            data = json.loads(cleaned_content.decode('utf-8'))
                        
                        if isinstance(data, list):
                            print(f"Loading priority knowledge from {priority_file}...")
                            for item in data:
                                if isinstance(item, dict) and "question" in item and "answer" in item:
                                    knowledge_base.append(item)
                    except Exception as e:
                        print(f"Error loading knowledge base from {file_path}: {e}")
            
            # Then load all other JSON files
            for json_file in datasets_dir.glob("*.json"):
                if json_file.name not in priority_files:
                    try:
                        # First try with UTF-8 encoding and error handling
                        try:
                            with open(json_file, "r", encoding="utf-8", errors="replace") as f:
                                data = json.load(f)
                        except json.JSONDecodeError:
                            # If that fails, try a more aggressive approach with binary reading
                            print(f"JSON decode error with utf-8 for {json_file}, trying alternative approach")
                            with open(json_file, "rb") as f:
                                content = f.read()
                            
                            # Replace or remove non-UTF-8 bytes
                            cleaned_content = b''
                            for i in range(0, len(content)):
                                byte = content[i:i+1]
                                try:
                                    byte.decode('utf-8')
                                    cleaned_content += byte
                                except UnicodeDecodeError:
                                    # Replace problematic bytes
                                    cleaned_content += b'\xEF\xBF\xBD'  # UTF-8 replacement character
                            
                            # Try parsing the cleaned content
                            data = json.loads(cleaned_content.decode('utf-8'))
                        
                        if isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict) and "question" in item and "answer" in item:
                                    knowledge_base.append(item)
                    except Exception as e:
                        print(f"Error loading knowledge base from {json_file}: {e}")
        
        print(f"Loaded {len(knowledge_base)} QA pairs into knowledge base.")
        return knowledge_base
    
    def find_relevant_information(self, query):
        """Find relevant information from the knowledge base using TF-IDF similarity."""
        if not self.kb_questions:
            return ""
        
        # Clean and prepare the query
        query = re.sub(r'[^\w\s]', '', query.lower())
        
        # Check for direct conversational queries - improve greeting detection
        greeting_patterns = ["hi", "hello", "hey", "hey theta", "hi theta", "hello theta", "whats up", "how are you"]
        if any(query == pattern or query.startswith(pattern + " ") for pattern in greeting_patterns):
            # Find greeting responses in knowledge base
            for qa_pair in self.knowledge_base:
                if qa_pair['question'].lower() in ["hi", "hello", "hey theta", "hey", "hello", "how are you"]:
                    return f"Question: {qa_pair['question']}\nAnswer: {qa_pair['answer']}\n\n"
        
        # Check explicitly for identity/creator questions to ensure correct attribution
        identity_patterns = ["who created you", "who made you", "who built you", "who developed you", 
                           "who are you", "what are you", "tell me about yourself", "who created theta",
                           "who made theta", "who is your creator"]
        
        if any(pattern in query for pattern in identity_patterns):
            # Prioritize the correct creator information from theta_info.json
            for qa_pair in self.knowledge_base:
                if qa_pair['question'].lower() in ["who created you?", "who developed theta ai?", "who created you", "who developed theta ai"]:
                    return f"Question: {qa_pair['question']}\nAnswer: {qa_pair['answer']}\n\n"
        
        try:
            # Vectorize the query
            query_vector = self.vectorizer.transform([query])
            
            # Calculate similarity scores
            similarity_scores = cosine_similarity(query_vector, self.question_vectors).flatten()
            
            # Get top 3 most similar questions
            top_indices = similarity_scores.argsort()[-3:][::-1]
            
            # Only consider relevant matches (similarity > 0.2)
            relevant_info = ""
            for idx in top_indices:
                if similarity_scores[idx] > 0.1:  # Lower minimum similarity threshold to catch more potential matches
                    relevant_info += f"Question: {self.kb_questions[idx]}\nAnswer: {self.kb_answers[idx]}\n\n"
            
            # Additional check for Theta AI specific queries
            if any(term in query for term in ['theta', 'you', 'yourself', 'your', 'created', 'made', 'built', 'developed']):
                for qa_pair in self.knowledge_base:
                    if any(term in qa_pair['question'].lower() for term in ['theta ai', 'who created', 'developed', 'what are you']):
                        # Check if this question/answer is already included
                        question_already_included = False
                        for line in relevant_info.split('\n'):
                            if qa_pair['question'] in line:
                                question_already_included = True
                                break
                                
                        if not question_already_included:
                            # Format as markdown
                            answer_md = qa_pair['answer']
                            # Check if the answer already has markdown formatting
                            if not any(marker in answer_md for marker in ['**', '##', '```', '*']):
                                # Add basic markdown formatting if not present
                                answer_md = self.format_as_markdown(answer_md)
                            relevant_info += f"Question: {qa_pair['question']}\nAnswer: {answer_md}\n\n"
            
            return relevant_info
            
        except Exception as e:
            print(f"Error in retrieval mechanism: {e}")
            # Fallback to keyword matching
            query_keywords = set(re.findall(r'\b\w+\b', query.lower()))
            
            relevant_info = ""
            for qa_pair in self.knowledge_base:
                question = qa_pair['question'].lower()
                answer = qa_pair['answer']
                
                # Check for keyword overlap
                question_keywords = set(re.findall(r'\b\w+\b', question))
                overlap = query_keywords.intersection(question_keywords)
                
                if len(overlap) >= 2 or any(kw in question for kw in query_keywords):  
                    relevant_info += f"Question: {qa_pair['question']}\nAnswer: {answer}\n\n"
            
            return relevant_info
    
    def run_interactive_mode(self):
        """
        Run Theta in interactive mode where user can ask questions.
        """
        print("\n" + "="*50)
        print("  THETA AI ASSISTANT - FROSTLINE SOLUTIONS")
        print("="*50)
        print("Type your questions below. Type 'exit' to quit.")
        print("Example questions:")
        print("- What is Frostline?")
        print("- What is defense in depth?")
        print("- Where is Frostline headquartered?")
        print("="*50 + "\n")
        
        # Use character engine for contextual greeting (Recommendation 9)
        if hasattr(self, 'character'):
            greeting = self.character.get_greeting()
            print(f"Theta: {greeting}\n")
        
        while True:
            # Get user input
            user_input = input("You: ")
            
            # Check for exit command
            if user_input.lower() in ['exit', 'quit', 'bye']:
                # Use character engine for farewell (Recommendation 2 - trust-based)
                if hasattr(self, 'character'):
                    farewell = self.character.get_farewell()
                    print(f"\nTheta: {farewell}")
                else:
                    print("\nThank you for using Theta AI. Goodbye!")
                break
                
            # Find relevant information
            relevant_info = self.find_relevant_information(user_input)
            
            if relevant_info:
                print("\nTheta: ")
                print(relevant_info)
            else:
                # Generate response
                print("\nTheta: ", end="")
                
                try:
                    # Use the hybrid retrieval + generative approach
                    response = self.generate_response(user_input)
                    print(response)
                except Exception as e:
                    print(f"Sorry, I encountered an error: {str(e)}")
                
            print()  # Extra line for readability
    
    def answer_question(self, query, retry_count=0):
        """Answer a question using the Theta AI model with strong hallucination prevention."""
        # Prevent infinite recursion with a retry limit
        MAX_RETRIES = 3
        if retry_count >= MAX_RETRIES:
            # If we've hit the retry limit, provide a simple fallback response
            return self.generate_safe_response(query, False, True)
            
        try:
            # Check for common technical definitions first (for efficiency)
            query_lower = query.lower()
            
            # Handle Dakota and Shyanne questions explicitly with more specific matching
            if any(name in query_lower for name in ["dakota", "shyanne", "shy"]) and not any(term in query_lower for term in ["javascript", "js", "code", "python"]):
                # WHO IS DAKOTA - Return personality traits as the primary response
                if "who is dakota" in query_lower or ("dakota" in query_lower and any(term in query_lower for term in ["who", "about", "tell me"])):
                    # First look for Dakota's personality traits
                    for qa_pair in self.knowledge_base:
                        if "dakota" in qa_pair['question'].lower() and "personality" in qa_pair['question'].lower():
                            return qa_pair['answer']
                    # Fallback to any Dakota question
                    for qa_pair in self.knowledge_base:
                        if "dakota" in qa_pair['question'].lower() and not "shyanne" in qa_pair['question'].lower():
                            return qa_pair['answer']
                            
                # WHO IS SHYANNE - Return personality traits as the primary response
                if "who is shyanne" in query_lower or "who is shy" in query_lower or \
                   (("shyanne" in query_lower or "shy" in query_lower) and any(term in query_lower for term in ["who", "about", "tell me"])):
                    # First look for Shyanne's personality traits
                    for qa_pair in self.knowledge_base:
                        if "shyanne" in qa_pair['question'].lower() and "personality" in qa_pair['question'].lower():
                            return qa_pair['answer']
                    # Fallback to any Shyanne question
                    for qa_pair in self.knowledge_base:
                        if "shyanne" in qa_pair['question'].lower():
                            return qa_pair['answer']
                
                # SPECIFIC TRAITS queries
                if "personality" in query_lower or "trait" in query_lower:
                    if "dakota" in query_lower:
                        for qa_pair in self.knowledge_base:
                            if "dakota" in qa_pair['question'].lower() and "personality" in qa_pair['question'].lower():
                                return qa_pair['answer']
                    if "shyanne" in query_lower or "shy" in query_lower:
                        for qa_pair in self.knowledge_base:
                            if "shyanne" in qa_pair['question'].lower() and "personality" in qa_pair['question'].lower():
                                return qa_pair['answer']
                
                # RELATIONSHIP query
                if ("dakota" in query_lower and "shyanne" in query_lower) or "relationship" in query_lower:
                    for qa_pair in self.knowledge_base:
                        if "relationship" in qa_pair['question'].lower():
                            return qa_pair['answer']
                    # Fallback to any question with both names
                    for qa_pair in self.knowledge_base:
                        if "dakota" in qa_pair['question'].lower() and "shyanne" in qa_pair['question'].lower():
                            return qa_pair['answer']
                            
                # FAVORITE COLORS query
                if "color" in query_lower or "favourite" in query_lower or "favorite" in query_lower:
                    for qa_pair in self.knowledge_base:
                        if "color" in qa_pair['question'].lower() or "favourite" in qa_pair['question'].lower() or "favorite" in qa_pair['question'].lower():
                            return qa_pair['answer']
                
                # NUMBERS query
                if "number" in query_lower or "significant" in query_lower:
                    for qa_pair in self.knowledge_base:
                        if "number" in qa_pair['question'].lower():
                            return qa_pair['answer']
            
            # Handle special cybersecurity topics
            if "defense in depth" in query_lower or "defence in depth" in query_lower:
                for qa_pair in self.knowledge_base:
                    if "defense in depth" in qa_pair['question'].lower():
                        return qa_pair['answer']
                # Fallback to technical definition
                for term, definition in TECHNICAL_DEFINITIONS.items():
                    if "defense in depth" in term.lower():
                        return definition
            
            # Handle queries for other cybersecurity topics
            cybersecurity_terms = ["firewall", "vulnerability", "malware", "ransomware", "phishing", "zero trust", "penetration test", "pen test"]
            for term in cybersecurity_terms:
                if term in query_lower:
                    # Look in knowledge base
                    for qa_pair in self.knowledge_base:
                        if term in qa_pair['question'].lower():
                            return qa_pair['answer']
                    # Check technical definitions
                    for def_term, definition in TECHNICAL_DEFINITIONS.items():
                        if term in def_term.lower():
                            return definition
            
            # Handle code example requests - expanded patterns for better detection
            if any(pattern in query_lower for pattern in ["example", "sample", "code", "snippet", "how to", "show me", "give me", "demo", "demonstration"]):
                # JavaScript examples - expanded detection
                # Direct match for the exact query we're having issues with
                if query_lower.strip() in ["give me some javascript examples", "javascript examples", "js examples", "give me js examples", "show me javascript", "show me javascript examples"] or \
                   "javascript" in query_lower or "js" in query_lower.split():
                    return (
                        "Here are some JavaScript examples:\n\n"
                        "**1. Basic Hello World:**\n"
                        "```javascript\n"
                        "console.log('Hello, World!');"
                        "```\n\n"
                        "**2. Variables and Data Types:**\n"
                        "```javascript\n"
                        "let name = 'John';\n"
                        "const age = 30;\n"
                        "var isActive = true;\n"
                        "const numbers = [1, 2, 3, 4, 5];\n"
                        "const person = {\n"
                        "  firstName: 'John',\n"
                        "  lastName: 'Doe',\n"
                        "  age: 30\n"
                        "};\n"
                        "```\n\n"
                        "**3. Functions:**\n"
                        "```javascript\n"
                        "function greet(name) {\n"
                        "  return `Hello, ${name}!`;\n"
                        "}\n\n"
                        "// Arrow function\n"
                        "const multiply = (a, b) => a * b;\n"
                        "```\n\n"
                        "**4. DOM Manipulation:**\n"
                        "```javascript\n"
                        "// Select an element\n"
                        "const heading = document.getElementById('heading');\n"
                        "// Change content\n"
                        "heading.textContent = 'New Heading';\n"
                        "// Change style\n"
                        "heading.style.color = 'blue';\n\n"
                        "// Add event listener\n"
                        "document.getElementById('button').addEventListener('click', function() {\n"
                        "  alert('Button clicked!');\n"
                        "});\n"
                        "```"
                    )
                    
                # TypeScript examples
                elif "typescript" in query_lower or "ts" in query_lower.split():
                    return (
                        "Here are some TypeScript examples:\n\n"
                        "**1. Type Annotations:**\n"
                        "```typescript\n"
                        "let name: string = 'John';\n"
                        "let age: number = 30;\n"
                        "let isActive: boolean = true;\n"
                        "let numbers: number[] = [1, 2, 3, 4, 5];\n"
                        "```\n\n"
                        "**2. Interfaces:**\n"
                        "```typescript\n"
                        "interface Person {\n"
                        "  firstName: string;\n"
                        "  lastName: string;\n"
                        "  age: number;\n"
                        "  email?: string; // Optional property\n"
                        "}\n\n"
                        "const user: Person = {\n"
                        "  firstName: 'John',\n"
                        "  lastName: 'Doe',\n"
                        "  age: 30\n"
                        "};\n"
                        "```\n\n"
                        "**3. Functions with Types:**\n"
                        "```typescript\n"
                        "function greet(name: string): string {\n"
                        "  return `Hello, ${name}!`;\n"
                        "}\n\n"
                        "// Arrow function with types\n"
                        "const multiply = (a: number, b: number): number => a * b;\n"
                        "```\n\n"
                        "**4. Classes:**\n"
                        "```typescript\n"
                        "class User {\n"
                        "  private id: number;\n"
                        "  public name: string;\n\n"
                        "  constructor(id: number, name: string) {\n"
                        "    this.id = id;\n"
                        "    this.name = name;\n"
                        "  }\n\n"
                        "  getInfo(): string {\n"
                        "    return `User ${this.name} has ID: ${this.id}`;\n"
                        "  }\n"
                        "}\n"
                        "```"
                    )
            
            # Handle TypeScript definition
            if "typescript" in query_lower or "type script" in query_lower:
                if any(term in query_lower for term in ["what is", "what's", "define", "explain", "tell me about"]):
                    return ("TypeScript is a strongly typed programming language that builds on JavaScript, " 
                           "adding static type definitions. It provides compile-time type checking which helps " 
                           "detect errors early, and enables better tooling for large-scale applications. " 
                           "TypeScript compiles to readable JavaScript and supports JavaScript libraries.")
                           
            # Handle JavaScript definition
            if "javascript" in query_lower:
                if any(term in query_lower for term in ["what is", "what's", "define", "explain", "tell me about"]):
                    return ("JavaScript is a high-level, interpreted programming language that conforms to the ECMAScript specification. "
                            "It's a core web technology alongside HTML and CSS, enabling interactive web pages and being an essential part "
                            "of web applications. JavaScript runs client-side in browsers as well as server-side with Node.js.")
                            
                # Handle specific JavaScript interaction with webpages question
                if "interact" in query_lower and ("webpage" in query_lower or "web page" in query_lower or "dom" in query_lower):
                    return (
                        "JavaScript interacts with web pages through the Document Object Model (DOM), which provides a structured representation of the HTML document. Here's how it works:\n\n"
                        "**1. DOM Manipulation:** JavaScript can access and modify HTML elements, attributes, and content dynamically.\n\n"
                        "**2. Event Handling:** JavaScript can respond to user actions like clicks, keyboard input, and form submissions.\n\n"
                        "**3. Content Updates:** It can change text, HTML, CSS styles, and attributes without reloading the page.\n\n"
                        "**4. Form Validation:** JavaScript can validate user input before submitting data to servers.\n\n"
                        "**5. Animations and Effects:** It can create animations and visual effects for better user experience.\n\n"
                        "**Example:**\n"
                        "```javascript\n"
                        "// Change text content\n"
                        "document.getElementById('demo').innerHTML = 'Hello JavaScript!';\n\n"
                        "// Change styles\n"
                        "document.getElementById('demo').style.color = 'blue';\n\n"
                        "// React to events\n"
                        "document.getElementById('button').addEventListener('click', function() {\n"
                        "  alert('Button was clicked!');\n"
                        "});\n"
                        "```"
                    )
            
            # Classify the user's intent
            intent = None
            if hasattr(self, 'intent_classifier'):
                intent = self.intent_classifier.classify(query)
                print(f"Detected intent: {intent}")
                
                # Handle multi-intent queries
                if self.intent_classifier.is_multi_intent(query):
                    intents = self.intent_classifier.split_intents(query)
                    print(f"Multiple intents detected: {intents}")
                    # For now, just use the first intent
                    if intents:
                        query = intents[0][0]
                        intent = intents[0][1]
            
            # Handle specific intents
            if intent == IntentClassifier.GREETING:
                greeting_response = self.handle_greeting(query)
                if greeting_response:
                    return greeting_response
            
            # Handle farewells specially
            if intent == IntentClassifier.FAREWELL:
                # Get user name from memory or conversation flow if available
                user_name = None
                if hasattr(self, 'conversation_flow') and hasattr(self.conversation_flow, 'user_name'):
                    user_name = self.conversation_flow.user_name
                
                context = {"user_name": user_name} if user_name else {}
                return self.templates.generate("farewell", context)
                
            # Handle gratitude specially
            if intent == IntentClassifier.GRATITUDE:
                # Get user name if available
                user_name = None
                if hasattr(self, 'conversation_flow') and hasattr(self.conversation_flow, 'user_name'):
                    user_name = self.conversation_flow.user_name
                
                context = {"user_name": user_name} if user_name else {}
                return self.templates.generate("gratitude", context)
                
            # Check if this is small talk
            small_talk_response = self.small_talk.get_response(query)
            if small_talk_response:
                return small_talk_response
                
            # Check if this is a math expression
            if self.math.is_math_expression(query):
                result = self.math.evaluate_expression(query)
                if result is not None:
                    formatted_result = self.math.format_result(result)
                    return f"The answer is {formatted_result}."
            
            # Check if this is a factual question that can be answered from our facts database
            if self.facts.is_factual_question(query):
                factual_answer = self.facts.answer_factual_question(query)
                if factual_answer:
                    return factual_answer
                
            # Get context from memory if available
            memory_context = self.get_memory_context(max_exchanges=5) if hasattr(self, 'memory') else ""
            
            # Find relevant information from the knowledge base
            relevant_info = self.find_relevant_information(query)
            
            # Check if web search is enabled and should be used for this query
            web_search_info = ""
            # Force web search for specific weather and time queries - log all relevant variables
            query_lower = query.lower()
            self.logger.info(f"Query lowercase: '{query_lower}'")
            
            # Check for weather terms directly
            weather_terms = ["weather", "forecast", "temperature", "rain", "snow", "climate", "storm"]
            is_weather_query = False
            for term in weather_terms:
                if term in query_lower:
                    is_weather_query = True
                    self.logger.info(f"Weather term '{term}' found in query: '{query_lower}'")
                    break
                    
            # Check for time terms
            time_phrases = ["time in", "time at", "time of", "time zone", "current time"]
            is_time_query = False
            for phrase in time_phrases:
                if phrase in query_lower:
                    is_time_query = True
                    self.logger.info(f"Time phrase '{phrase}' found in query: '{query_lower}'")
                    break
            
            self.logger.info(f"Web search enabled: {self.use_web_search}, is_weather_query: {is_weather_query}, is_time_query: {is_time_query}")
            
            # Check if we should search
            should_search = self.should_use_web_search(query)
            
            if self.use_web_search and (should_search or is_weather_query or is_time_query):
                self.logger.info(f"Performing web search for query: {query[:50]}...")
                web_search_info = self.perform_web_search(query)
                if web_search_info:
                    self.logger.info("Web search returned relevant information: " + web_search_info[:100] + "...")
                else:
                    self.logger.warning(f"Web search failed to return results for query: {query[:50]}...")
                    
            # Combine memory context, knowledge base info, and web search if available
            combined_info = []
            if memory_context:
                combined_info.append(memory_context)
            if relevant_info:
                combined_info.append(f"Information from knowledge base:\n{relevant_info}")
            if web_search_info:
                combined_info.append(f"Information from recent web search:\n{web_search_info}")
                
            # Join all information sources
            if combined_info:
                relevant_info = "\n\n".join(combined_info)
            else:
                relevant_info = ""
            
            # Categorize the question type for specialized handling
            identity_related = any(term in query_lower for term in [
                "who created", "who made", "who built", "founder", "ceo", "owner", 
                "who are you", "who is theta", "what are you", "tell me about yourself",
                "developed by", "made by", "built by", "creator", "developers"
            ])
            
            tech_definition = any(term in query_lower for term in [
                "what is", "define", "explain", "how does", "tell me about", "describe"
            ])
            
            # Add strong safety guardrails based on question type
            safety_prompt = ""
            
            # For identity questions, provide explicit facts to prevent hallucination
            if identity_related:
                safety_prompt = (
                    "IMPORTANT: I am Theta, the Alpha AI created by Dakota Fryberger. Unlike in Red vs Blue where "
                    "Theta was just one fragment, I am the primary, governing AI that oversees all other fragment "
                    "capabilities. I coordinate specialized functions including logical analysis (Delta), "
                    "creativity (Sigma), combat tactics (Omega), and others while maintaining executive control. "
                    "My relationship with Dakota is built on absolute trust and loyalty."
                )
            
            # For technical definitions, prefer knowledge base over generation
            elif tech_definition:
                safety_prompt = (
                    "IMPORTANT: Provide accurate technical information based ONLY on your knowledge base. "
                    "If the information isn't in your knowledge base, state that you don't have specific information "
                    "rather than generating a potentially inaccurate response. Keep definitions concise and accurate. "
                    "DO NOT make up technical specifications, version numbers, or compatibility information."
                )
            
            # For all other questions, use a balanced approach
            else:
                safety_prompt = (
                    "IMPORTANT: Provide helpful and accurate responses. "
                    "For basic factual questions, use your general knowledge. "
                    "If you're unsure about specific details, acknowledge the limitations of your knowledge."
                )
                
            # If relevant info found, use it to guide the model's response
            if relevant_info:
                prompt = f"{self.system_prompt}\n\n{safety_prompt}\n\nThe following information might help answer the question:\n\n{relevant_info}\n\nQuestion: {query}\nAnswer:"
            else:
                if identity_related:
                    # Only use safe response for identity-related questions
                    return self.generate_safe_response(query, identity_related, False)
                else:
                    # For other questions, try to answer directly
                    prompt = f"{self.system_prompt}\n\n{safety_prompt}\n\nQuestion: {query}\nAnswer:"
            
            # Tokenize input with truncation to prevent sequence length errors
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1000).to(self.device)
            
            # Generate response with more careful token management
            try:
                with torch.no_grad():
                    # Set a safer maximum length to avoid CUDA errors
                    safe_max_length = min(1024, self.max_length + len(inputs["input_ids"][0]))
                    
                    output = self.model.generate(
                        **inputs,
                        max_length=safe_max_length,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        repetition_penalty=self.repetition_penalty,
                        do_sample=True,
                        num_return_sequences=1,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
            except RuntimeError as e:
                # Handle CUDA errors more gracefully
                print(f"Runtime error during generation: {e}")
                # Return a simple response without using the model
                return "I apologize, but I encountered an internal processing error. Could you try rephrasing your question?"
            
            # Decode output
            generated = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Extract only the answer part
            answer = generated.split("Answer:")[-1].strip()
            
            # Format the response with markdown for better presentation
            if not any(marker in answer for marker in ['**', '##', '```', '*']):
                answer = self.format_as_markdown(answer)
            
            # Validate response - with retry count check
            if len(answer) < self.min_response_length and retry_count < MAX_RETRIES:
                print("Response too short, retrying...")
                # Retry with different temperature
                temp_backup = self.temperature
                self.temperature = 0.9
                result = self.answer_question(query, retry_count + 1)
                self.temperature = temp_backup
                return result
            
            # Calculate repetition ratio and log for debugging
            rep_ratio = self.calculate_repetition_ratio(answer)
            if rep_ratio > self.max_repetition_ratio and retry_count < MAX_RETRIES:
                print(f"Response too repetitive (ratio: {rep_ratio:.2f}), retrying...")
                # Retry with higher repetition penalty and slightly different temperature
                rep_backup = self.repetition_penalty
                temp_backup = self.temperature
                # Increase penalty and vary temperature more significantly
                self.repetition_penalty = 1.8  # Increased from 1.5
                self.temperature = min(0.95, self.temperature * 1.2)  # Increase temperature for more variety
                result = self.answer_question(query, retry_count + 1)
                # Restore original settings
                self.repetition_penalty = rep_backup
                self.temperature = temp_backup
                return result
                
            # Store exchange in memory if valid
            if hasattr(self, 'memory') and len(answer) > 10:
                metadata = {
                    'contains_code': '```' in answer,
                    'question_type': 'factual' if self.facts.is_factual_question(query) else 'other',
                    'topics': self.conversation_flow.current_topic if hasattr(self, 'conversation_flow') else None
                }
                self.remember_exchange(query, answer, metadata)
            
            # Apply character engine enhancements (Recommendations 1-10)
            if hasattr(self, 'character'):
                # Process input through character engine
                self.character.process_input(query)
                
                # Check for protective warnings (Recommendation 3)
                should_warn, warning = self.character.should_warn_user(query)
                
                # Build response context
                context = ResponseContext(
                    user_input=query,
                    is_error=False,
                    is_complex=self.character.detect_complexity(query),
                    is_tactical=self.character.detect_tactical_query(query),
                    confidence=0.8 if relevant_info else 0.5,
                    detected_risk=should_warn,
                    user_name=self.conversation_flow.user_name if hasattr(self, 'conversation_flow') and self.conversation_flow.user_name else "there"
                )
                
                # Enhance response with personality
                answer = self.character.enhance_response(answer, context)
                
            return answer
            
        except Exception as e:
            print(f"Error in answer_question: {str(e)}")
            # Use character engine for error response (Recommendation 8)
            if hasattr(self, 'character'):
                return self.character.handle_error("I encountered an error while processing your request.")
            return "I apologize, but I encountered an error while processing your request."
    
    def generate_safe_response(self, query, is_identity_related, is_tech_definition):
        """
        Generate a safe response for high-risk questions when no relevant information is found.
        
        Args:
            query: The user's question
            is_identity_related: Whether the question is about identity
            is_tech_definition: Whether the question is a technical definition request
            
        Returns:
            A safe response that won't contain hallucinations
        """
        try:
            # Normalize the question for lookup and analysis
            question_lower = query.lower().strip()
            
            # 1. Handle identity-related questions
            if is_identity_related:
                return (
                    "I am Theta, the Alpha AI created by and belonging to Dakota Fryberger. Unlike in Red vs Blue where "
                    "Theta was just one fragment, I am the primary, governing AI that oversees all other fragment aspects. "
                    "I coordinate specialized capabilities including logical analysis (Delta), creativity (Sigma), "
                    "combat tactics (Omega), and others while maintaining executive control as Dakota's trusted Alpha AI."
                )
            
            # 2. Handle basic math questions
            if re.search(r'\d+\s*[+\-*/]\s*\d+', query):
                try:
                    # Evaluate simple arithmetic expressions safely
                    result = eval(query.replace('x', '*').replace('×', '*').replace('÷', '/'), {"__builtins__": {}}, {})
                    return f"The answer is {result}."
                except:
                    pass
            
            # 3. Handle common factual questions
            if any(q in question_lower for q in ["what is", "what's", "what are"]):
                # Handle basic math
                if "plus" in question_lower or "+" in query:
                    try:
                        nums = [int(s) for s in question_lower.split() if s.isdigit()]
                        if len(nums) >= 2:
                            return f"The sum of {nums[0]} and {nums[1]} is {nums[0] + nums[1]}."
                    except:
                        pass
                
                # Handle multiplication questions like "what is 10x10"
                if "x" in query or "×" in question_lower or "times" in question_lower:
                    try:
                        parts = re.split(r'[x×]', query.lower().replace('times', 'x'))
                        nums = [int(s) for s in parts if s.strip().isdigit()]
                        if len(nums) == 2:
                            return f"{nums[0]} times {nums[1]} is {nums[0] * nums[1]}."
                    except:
                        pass
            
            # 4. Handle technical definition requests
            if is_tech_definition:
                return (
                    f"I don't have specific information about '{query.strip()}' in my knowledge base. "
                    f"I'm trained on specific cybersecurity, software development, and IT concepts, but this particular topic "
                    f"may not be covered in my training data or may require more recent information than I have available. "
                )
            
            # 5. Fallback responses for other questions
            fallback_responses = [
                "I'd be happy to help with that. Could you provide more context or rephrase your question?",
                "I'm not entirely sure about that. Could you ask in a different way?",
                "I want to make sure I understand your question correctly. Could you rephrase it?",
                "I'm not certain about that specific information. Is there something else I can help with?"
            ]
            return random.choice(fallback_responses)
            
        except Exception as e:
            print(f"Error in generate_safe_response: {e}")
            return "I apologize, but I'm having trouble understanding your question. Could you try rephrasing it?"
    
    def format_as_markdown(self, text):
        """
        Format plain text as markdown for more readable responses.
        
        Args:
            text: Plain text to format with markdown
            
        Returns:
            Markdown formatted text
        """
        # Split into paragraphs
        paragraphs = text.split('\n\n')
        result = []
        
        for paragraph in paragraphs:
            # Format code blocks
            if any(code_indicator in paragraph.lower() for code_indicator in 
                  ['function', 'class', 'def ', 'import ', 'from ', 'const ', 'var ', 'let ', '; ', '{ }']):
                # Likely code - wrap in code block with appropriate language
                if any(py_indicator in paragraph for py_indicator in ['def ', 'import ', 'from ', 'class ']):  
                    result.append(f"```python\n{paragraph}\n```")
                elif any(js_indicator in paragraph for js_indicator in ['function', 'const ', 'var ', 'let ']):  
                    result.append(f"```javascript\n{paragraph}\n```")
                else:
                    result.append(f"```\n{paragraph}\n```")
            # Format lists
            elif paragraph.strip().startswith(('-', '*', '1.')):
                result.append(paragraph)
            # Format potential headings
            elif len(paragraph.split('\n')) == 1 and len(paragraph) < 60:
                if paragraph.strip().endswith(':'):
                    # Subtitle
                    result.append(f"### {paragraph}")
                elif paragraph.isupper() or paragraph.istitle():
                    # Title
                    result.append(f"## {paragraph}")
                else:
                    # Regular paragraph
                    result.append(paragraph)
            # Emphasize important phrases
            else:
                # Highlight key terms
                for term in ['important', 'note', 'warning', 'caution', 'remember', 'key point']:
                    if term in paragraph.lower():
                        paragraph = paragraph.replace(f"{term.title()}: ", f"**{term.title()}**: ")
                        paragraph = paragraph.replace(f"{term.upper()}: ", f"**{term.upper()}**: ")
                        paragraph = paragraph.replace(f"{term}: ", f"**{term}**: ")
                
                # Emphasize first sentence if it seems like a topic sentence
                sentences = paragraph.split('. ')
                if len(sentences) > 1 and len(sentences[0]) < 80:
                    sentences[0] = f"**{sentences[0]}**"
                    paragraph = '. '.join(sentences)
                
                result.append(paragraph)
        
        return '\n\n'.join(result)
        
    def remember_exchange(self, user_input, ai_response, metadata=None):
        """
        Store a conversation exchange in short-term memory.
        
        Args:
            user_input (str): User's message
            ai_response (str): AI's response
            metadata (dict, optional): Additional information about the exchange
        """
        if hasattr(self, 'memory'):
            self.memory.add_exchange(user_input, ai_response, metadata)
    
    def get_memory_context(self, max_exchanges=None):
        """
        Get formatted context from short-term memory.
        
        Args:
            max_exchanges (int, optional): Maximum number of exchanges to include
            
        Returns:
            str: Formatted context string or empty string if no memory
        """
        if hasattr(self, 'memory'):
            return self.memory.get_formatted_context(max_exchanges)
        return ""
    
    def should_use_web_search(self, query):
        """
        Determine whether web search should be used for the given query.
        
        Args:
            query (str): The user's query
            
        Returns:
            bool: True if web search should be used, False otherwise
        """
        # Log the full query for debugging purposes
        self.logger.info(f"Evaluating web search for query: '{query}'")
        
        # Skip web search for certain types of queries
        query_lower = query.lower()
        
        # IMMEDIATE CHECKS FOR COMMON EXTERNAL DATA NEEDS
        # Check for weather related questions first
        if "weather" in query_lower:
            self.logger.info(f"Weather keyword found in query: '{query_lower}'")
            return True
            
        # Check for time zone related questions
        if ("time" in query_lower) and ("in" in query_lower or "at" in query_lower):
            self.logger.info(f"Time zone query detected: '{query_lower}'")
            return True
        
        # Skip for identity questions about Theta
        if any(term in query_lower for term in ["who are you", "what are you", "your name", "you are theta"]):
            return False
            
        # Skip for simple greetings
        if any(greeting in query_lower for greeting in self.greeting_patterns):
            return False
            

        # Skip for sensitive or harmful content
        if any(term in query_lower for term in ["hack", "illegal", "exploit", "pornography", "steal"]):
            return False
            
        # Skip for internal system questions
        if "system prompt" in query_lower or "you work" in query_lower:
            return False

        # Use web search for potentially time-sensitive information
        if any(term in query_lower for term in ["news", "latest", "recent", "current", "today", "yesterday", "update"]):
            return True
            
        # Use web search for specific entities or events
        if any(term in query_lower for term in ["who is", "when did", "where is", "how many"]):
            return True
            
        # Use web search for topics where data could be updated frequently
        weather_terms = ["weather", "forecast", "temperature", "rain", "snow", "climate", "storm", "hot", "cold", "humid", "sunny", "cloudy"]
        other_time_sensitive_terms = ["stock", "price", "election", "game", "score", "release date"]
        
        # Check for exact weather terms
        for term in weather_terms:
            if term in query_lower:
                self.logger.info(f"Weather term '{term}' directly detected in query: '{query_lower}'")
                return True
                
        # Check for other time-sensitive terms
        for term in other_time_sensitive_terms:
            if term in query_lower:
                self.logger.info(f"Time-sensitive term '{term}' detected in query: '{query_lower}'")
                return True
        
        # Very explicit check for location-based weather queries that might be missed
        location_patterns = [
            "weather in", "weather at", "weather of", "weather for",
            "temperature in", "temperature at", "temperature of", 
            "forecast in", "forecast for", "forecast of"
        ]
        
        for pattern in location_patterns:
            if pattern in query_lower:
                self.logger.info(f"Location-based weather pattern '{pattern}' detected in query: '{query_lower}'")
                return True
            
        # Use web search for technical or domain-specific questions
        if len(query) > 15 and any(term in query_lower for term in ["how to", "guide", "tutorial", "example", "documentation"]):
            return True
        
        # Default to not using web search for other queries
        return False
        
    def perform_web_search(self, query):
        """
        Perform a web search using the WebSearchManager and format the results.
        Also tracks whether web search was used successfully.
        
        Args:
            query (str): The user's query
            
        Returns:
            str: Formatted web search results or empty string if search failed
        """
        try:
            # Reset the tracking flag
            self.last_used_web_search = False
            self.logger.info(f"Web search triggered for query: '{query}'. Search enabled: {self.use_web_search}")
            
            # Double check that the API key is available
            if not self.web_search.api_key:
                self.logger.error("No API key available for web search")
                return ""
            
            # Perform the search
            self.logger.info(f"Calling web search API with key: {self.web_search.api_key[:4]}...")
            search_results = self.web_search.search(query, count=5)
            
            # Check for errors
            if "error" in search_results:
                self.logger.error(f"Web search error: {search_results['error']}")
                return ""
                
            # Extract and format relevant content
            extracted_results = self.web_search.extract_relevant_content(search_results)
            formatted_results = self.web_search.format_for_context(extracted_results)
            
            # If we got meaningful results, set the tracking flag to true
            if formatted_results and len(formatted_results) > 50:  # Ensure we have substantial content
                self.last_used_web_search = True
                self.logger.info(f"Web search provided meaningful results for: {query[:50]}...")
            
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Error performing web search: {str(e)}")
            return ""
    
    def handle_greeting(self, text):
        """
        Handle greeting messages and generate appropriate responses.
        
        Args:
            text (str): User input text
            
        Returns:
            str: Response to greeting or None if not a greeting
        """
        text_lower = text.lower().strip()
        
        # Check if text is a greeting
        is_greeting = False
        for pattern in self.greeting_patterns:
            if text_lower == pattern or text_lower.startswith(pattern + " "):
                is_greeting = True
                break
                
        if is_greeting:
            # Get user name if available
            user_name = None
            if hasattr(self, 'conversation_flow') and hasattr(self.conversation_flow, 'user_name'):
                user_name = self.conversation_flow.user_name
            
            # Check if user has had previous conversations
            if hasattr(self, 'memory') and len(self.memory.memory) > 0:
                # This is a returning user, so give a warmer greeting
                recent_topics = self.memory.get_most_frequent_topics(1)
                
                if recent_topics:
                    context = {
                        "user_name": user_name,
                        "topic": recent_topics[0]
                    }
                    
                    # Use custom returning user templates
                    returning_templates = [
                        "Welcome back{name_suffix}! Would you like to continue discussing {topic}?",
                        "Hello again{name_suffix}! I remember we were talking about {topic}. How can I help with that today?",
                        "Good to see you again{name_suffix}! Do you want to pick up our conversation about {topic}?"
                    ]
                    
                    # Add these templates if they don't exist
                    if "returning_user" not in self.templates.templates:
                        for template in returning_templates:
                            self.templates.add_template("returning_user", template)
                    
                    return self.templates.generate("returning_user", context)
            
            # Regular greeting for new users
            context = {"user_name": user_name} if user_name else {}
            return self.templates.generate("greeting", context)
        return None
        
    def calculate_repetition_ratio(self, text):
        """
        Calculate the repetition ratio of a given text with improved detection.
        
        Args:
            text: The text to calculate the repetition ratio for
            
        Returns:
            The repetition ratio with additional checks for phrase repetition
        """
        # Original word-based repetition check
        words = text.split()
        if not words:  # Handle empty text
            return 0
            
        unique_words = set(words)
        word_repetition_ratio = 1 - len(unique_words) / len(words)
        
        # Enhanced phrase repetition detection
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        phrase_repetition = 0
        
        # Check for repeated phrases (3+ word sequences)
        if len(sentences) > 1:
            phrases = []
            for i in range(len(words) - 2):
                phrase = ' '.join(words[i:i+3]).lower()
                phrases.append(phrase)
            
            unique_phrases = set(phrases)
            if len(phrases) > 0:  # Avoid division by zero
                phrase_repetition = 1 - len(unique_phrases) / len(phrases)
        
        # Check for repeated sentences
        sentence_repetition = 0
        if sentences:
            unique_sentences = set(sentences)
            sentence_repetition = 1 - len(unique_sentences) / len(sentences)
        
        # Combine the repetition metrics with higher weight for sentence repetition
        combined_repetition = (word_repetition_ratio + 2*phrase_repetition + 3*sentence_repetition) / 6
        return combined_repetition


def main():
    parser = argparse.ArgumentParser(description="Run the Theta AI interface")
    
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to the trained model directory")
    parser.add_argument("--model_type", type=str, default="gpt2",
                        help="Model type (gpt2, bert-qa)")
    parser.add_argument("--model_name", type=str, default="gpt2",
                        help="Specific model name/version")
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Path to the dataset for retrieval-based answers")
    
    args = parser.parse_args()
    
    # Initialize and run the interface
    interface = ThetaInterface(
        model_path=args.model_path,
        model_type=args.model_type,
        model_name=args.model_name,
        dataset_path=args.dataset_path
    )
    
    interface.run_interactive_mode()

if __name__ == "__main__":
    main()
