from flask import Flask, request, jsonify, render_template, url_for, session, Response, make_response
from flask_wtf.csrf import CSRFProtect, generate_csrf, validate_csrf
import sys
import os
import secrets
import re
import time
import logging
import html
import datetime
import json
import random
import uuid
from pathlib import Path
from dotenv import load_dotenv

# Import definition templates and other improvements
from src.interface.definitions import TECHNICAL_DEFINITIONS, IDENTITY_ANSWERS, HALLUCINATION_PRONE_TOPICS, SAFETY_RESPONSES

# Import database modules
from src.database.db_setup import create_tables, check_database_connection
from src.database.database_manager import DatabaseManager
from src.interface.conversation_manager import ConversationManager

# Add project root to path to import modules
project_root = str(Path(__file__).resolve().parent.parent)
sys.path.append(project_root)
from src.interface.theta_interface import ThetaInterface

class CustomFlask(Flask):
    def get_send_file_max_age(self, name):
        # Override to set longer cache timeout for static files
        return 31536000  # 1 year in seconds

app = Flask(__name__, 
            template_folder=os.path.join(project_root, 'templates'),
            static_folder=os.path.join(project_root, 'static'),
            static_url_path='/static')
            
# Security settings
app.config['SECRET_KEY'] = secrets.token_hex(16)  # Generate a random secret key
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching in development
app.config['WTF_CSRF_TIME_LIMIT'] = 3600  # CSRF token valid for 1 hour

# Initialize CSRF protection
csrf = CSRFProtect(app)

# Set up logging
log_path = os.path.join(project_root, 'logs')
os.makedirs(log_path, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_path, f"theta_ui_{datetime.datetime.now().strftime('%Y%m%d')}.log"),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('theta_ui')

# Add console handler for important messages
console = logging.StreamHandler()
console.setLevel(logging.WARNING)
logger.addHandler(console)

# Request count for monitoring
request_counts = {}

# Define patterns to detect potential XSS or command injection
MALICIOUS_PATTERNS = [
    r'<script[^>]*>',
    r'javascript:',
    r'onerror=',
    r'onclick=',
    r'onload=',
    r'eval\(',
    r'\bexec\b',
    r'\bsystem\b',
    r'\bos\.\w+\b',
    r'\bsubprocess\.\w+\b',
]

# Compile patterns for efficiency
MALICIOUS_REGEX = re.compile('|'.join(MALICIOUS_PATTERNS), re.IGNORECASE)

# Load environment variables
load_dotenv()

# Initialize database connection
print("Checking database connection...")
db_connection_ok = check_database_connection()
if not db_connection_ok:
    print("Warning: Database connection failed. Some features may not work properly.")
else:
    print("Database connection successful!")
    # Create tables idempotently
    print("Setting up database tables...")
    create_tables()
    print("Database setup complete.")

# Initialize database manager
db_manager = DatabaseManager()

# Initialize Theta interface
print("Initializing Theta AI...")

# Check if external data sources are enabled (default to False for security)
use_external_data = os.environ.get('EXTERNAL_DATA_SOURCES', 'false').lower() == 'true'
discord_data_enabled = os.environ.get('DISCORD_DATA_ENABLED', 'false').lower() == 'true'

# Prevent loading Discord data
if not use_external_data or not discord_data_enabled:
    print("External data sources disabled. Using only local datasets.")
    # Ensure no external paths are used
    if 'EXTERNAL_DATA_PATH' in os.environ:
        print("WARNING: External data path found but external data is disabled.")
        del os.environ['EXTERNAL_DATA_PATH']

# Set up model paths
models_dir = os.path.join(project_root, "models")

# Check if USE_LATEST_MODEL environment variable is set
use_latest_model = os.environ.get('USE_LATEST_MODEL', 'false').lower() == 'true'

# Find the latest model directory
if use_latest_model:
    print("Looking for the latest model directory...")
    latest_date = None
    latest_model_dir = None
    
    # List all directories in the models folder
    for item in os.listdir(models_dir):
        if item.startswith('theta_enhanced_'):
            try:
                # Extract date from directory name
                date_str = item.replace('theta_enhanced_', '')
                # Compare as strings since the format is YYYYMMDD
                if latest_date is None or date_str > latest_date:
                    latest_date = date_str
                    latest_model_dir = item
            except Exception as e:
                print(f"Error processing directory {item}: {e}")
    
    if latest_model_dir:
        trained_model_dir = os.path.join(models_dir, latest_model_dir)
        print(f"Found latest model directory: {trained_model_dir}")
    else:
        # Fallback to a default if no model directories found
        trained_model_dir = os.path.join(models_dir, "theta_enhanced_20250310")
        print(f"No model directories found, using default: {trained_model_dir}")
else:
    # Use the default path
    trained_model_dir = os.path.join(models_dir, "theta_enhanced_20250310")
    print(f"Using configured model directory: {trained_model_dir}")

final_model_path = os.path.join(trained_model_dir, "final")

# Check for model files in order of preference
model_path = None

# Helper function to find the most recent checkpoint
def find_latest_checkpoint(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        return None, 0
        
    latest_checkpoint = None
    latest_epoch = 0
    
    for item in os.listdir(checkpoint_dir):
        if item.startswith("theta_checkpoint_epoch_"):
            try:
                # Extract epoch number from checkpoint folder name (e.g., theta_checkpoint_epoch_10 -> 10)
                epoch = int(item.split("_")[-1])
                if epoch > latest_epoch:
                    latest_epoch = epoch
                    latest_checkpoint = os.path.join(checkpoint_dir, item)
            except (IndexError, ValueError) as e:
                print(f"Error processing checkpoint {item}: {e}")
                continue
                
    return latest_checkpoint, latest_epoch

# 1. Check for final model
if os.path.exists(final_model_path):
    model_path = final_model_path
    print(f"Using final model: {model_path}")
else:
    # 2. Check for checkpoints in the trained model directory
    print("No final model found, checking for most recent checkpoint...")
    latest_checkpoint, epoch = find_latest_checkpoint(trained_model_dir)
    
    if latest_checkpoint:
        model_path = latest_checkpoint
        print(f"Using most recent checkpoint: {model_path} (Epoch {epoch})")
    else:
        # 3. Fall back to base model if nothing else is found
        print("No trained models found. Falling back to base GPT2-medium model...")
        model_path = "gpt2-medium"

# Initialize with the selected model
theta = ThetaInterface(model_path=model_path)
print("Theta AI initialized successfully!")

# Security headers middleware
@app.after_request
def add_security_headers(response):
    # Content Security Policy - Allow inline scripts for now
    response.headers['Content-Security-Policy'] = "default-src 'self'; "\
        "script-src 'self' 'unsafe-inline'; "\
        "style-src 'self' 'unsafe-inline'; "\
        "img-src 'self'; "\
        "font-src 'self'; "\
        "connect-src 'self'; "\
        "frame-src 'none'; "\
        "object-src 'none'; "\
        "base-uri 'self'"
    
    # Additional security headers
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    response.headers['Cache-Control'] = 'no-store'
    response.headers['Permissions-Policy'] = 'camera=(), microphone=(), geolocation=()'
    
    return response

@app.route('/')
def home():
    # Log the request - support CloudFlare headers
    client_ip = request.headers.get('CF-Connecting-IP') or \
               request.headers.get('X-Forwarded-For', '').split(',')[0].strip() or \
               request.remote_addr
    logger.info(f"Home page accessed from {client_ip}")
    
    # Track request counts for rate monitoring
    if client_ip not in request_counts:
        request_counts[client_ip] = {'count': 1, 'first_request': time.time()}
    else:
        request_counts[client_ip]['count'] += 1
        # Check for unusual activity (more than 60 requests per minute)
        elapsed = time.time() - request_counts[client_ip]['first_request']
        if elapsed < 60 and request_counts[client_ip]['count'] > 60:
            logger.warning(f"Possible abuse detected from {client_ip}: {request_counts[client_ip]['count']} requests in {elapsed:.1f} seconds")
    
    # Pass static URLs to template to avoid Jinja2 template issues with JavaScript
    static_url_css = url_for('static', filename='styles.css')
    static_url_logo = url_for('static', filename='theta-symbol.png')
    
    # Generate CSRF token for the form
    csrf_token = generate_csrf()
    
    return render_template('index.html', 
                           static_url_css=static_url_css,
                           static_url_logo=static_url_logo,
                           csrf_token=csrf_token)

def sanitize_input(text):
    """Sanitize user input to prevent XSS and command injection"""
    # Log if we find potential malicious patterns but still allow the text
    if MALICIOUS_REGEX.search(text):
        logger.warning(f"Potential suspicious pattern in input: {text[:100]}")
    
    # Escape HTML entities to prevent XSS in responses
    return html.escape(text)

@app.route('/ask', methods=['POST'])
@csrf.exempt  # Exempt the API from CSRF for now, we'll validate manually
def ask():
    try:
        # Get client IP for logging - support CloudFlare headers
        client_ip = request.headers.get('CF-Connecting-IP') or \
                   request.headers.get('X-Forwarded-For', '').split(',')[0].strip() or \
                   request.remote_addr
        
        # Get CSRF token from request
        csrf_token = request.json.get('csrf_token')
        # For now, log but don't block if token is missing - helps with debugging
        if not csrf_token:
            logger.warning(f"Missing CSRF token from {client_ip}")
        # Only validate if a token was provided
        elif csrf_token:
            try:
                validate_csrf(csrf_token)
            except Exception as e:
                logger.warning(f"Invalid CSRF token from {client_ip}: {str(e)}")
                return jsonify({'error': 'Invalid CSRF token'}), 403
        
        # Get session ID or create a new one
        session_id = request.json.get('session_id')
        if not session_id:
            session_id = str(uuid.uuid4())
            logger.info(f"Created new session {session_id} for {client_ip}")
        else:
            logger.info(f"Using existing session {session_id} for {client_ip}")
            
        # Initialize conversation manager for this session
        conv_manager = ConversationManager(session_id=session_id)
            
        # Get and validate question
        raw_question = request.json.get('question')
        if not raw_question:
            logger.info(f"Empty question from {client_ip}")
            return jsonify({'error': 'No question provided'}), 400
        
        # Sanitize input
        question = sanitize_input(raw_question)
        
        # Log the question (without PII)
        truncated_question = question[:100] + '...' if len(question) > 100 else question
        logger.info(f"Question from {client_ip}: {truncated_question}")
        
        # Clean and normalize the question
        question_lower = question.lower().strip()
        term = None  # Define term outside conditional blocks for later use
        
        # Check for identity-related questions
        identity_related = any(term in question_lower for term in [
            "who created", "who made", "who built", "founder", "ceo", "owner", 
            "who are you", "who is theta", "what are you", "tell me about yourself",
            "developed by", "made by", "built by", "creator", "developers"
        ])
        
        # Check for training data questions which can trigger CUDA errors
        training_data_question = any(term in question_lower for term in [
            "what data are you trained on", "what's your training data", "what were you trained on",
            "what were you trained with", "what data were you trained on", "what dataset", 
            "what corpus", "what training data", "trained on what", "training corpus", 
            "data did you train", "training set", "what are you trained with", 
            "what information were you trained on", "what sources were you trained on",
            "what knowledge do you have"
        ])
        
        # Check for definition/explanation questions
        is_definition_question = any(pattern in question_lower for pattern in [
            "what is", "what's", "what are", "define", "explain", "tell me about", 
            "what does", "meaning of", "definition of", "stands for"
        ])
        
        # Check for exact identity questions and training data questions in our templates
        for key, response in IDENTITY_ANSWERS.items():
            # Look for exact match or question with question mark
            if question_lower == key or question_lower.startswith(key + "?"):
                logger.info(f"Using identity template for {client_ip}")
                answer = response
                # Log the response length and return
                logger.info(f"Generated answer of {len(answer)} characters for {client_ip}")
                return jsonify({'answer': answer})
                
        # Special handling for training data questions that aren't exact matches
        if training_data_question:
            logger.info(f"Handling training data question from {client_ip}")
            # Use one of the training data responses from IDENTITY_ANSWERS
            training_keys = ["what data are you trained on", "what's your training data", "what were you trained on"]
            # Randomly select one of the training data responses
            selected_key = random.choice(training_keys)
            answer = IDENTITY_ANSWERS[selected_key]
            logger.info(f"Generated training data answer of {len(answer)} characters for {client_ip}")
            return jsonify({'answer': answer})
        
        # Check for technical definitions
        is_definition_question = any(pattern in question_lower for pattern in [
            "what is", "what's", "what are", "define", "explain", "tell me about", 
            "what does", "meaning of", "definition of", "stands for"
        ])
        
        # Extract the term being defined
        if is_definition_question:
            # Try to extract the term after common patterns
            for pattern in ["what is", "what's", "what does", "define", "explain", "tell me about"]:
                if pattern in question_lower:
                    # Get everything after the pattern
                    term_part = question_lower.split(pattern, 1)[1].strip().rstrip('?')
                    # If it contains "mean" or "stand for", handle specially
                    if "mean" in term_part:
                        term = term_part.split("mean")[0].strip()
                    elif "stand for" in term_part:
                        term = term_part.split("stand for")[0].strip()
                    else:
                        term = term_part
                    break
            
            # If we extracted a term, check if we have a definition
            if term:
                # Clean up term (remove articles like "a", "an", "the")
                term = re.sub(r'^(a|an|the)\s+', '', term).strip()
                # Check in our technical definitions
                if term in TECHNICAL_DEFINITIONS:
                    logger.info(f"Using definition template for '{term}' from {client_ip}")
                    answer = TECHNICAL_DEFINITIONS[term]
                    # Log the response length and return
                    logger.info(f"Generated answer of {len(answer)} characters for {client_ip}")
                    return jsonify({'answer': answer})
        
        # Check for hallucination-prone topics
        if any(topic in question_lower for topic in HALLUCINATION_PRONE_TOPICS):
            # For these topics, sometimes directly return a safety response
            if random.random() < 0.7:  # 70% chance to use safety response
                safety_keys = list(SAFETY_RESPONSES.keys())
                logger.info(f"Using safety response for hallucination-prone topic from {client_ip}")
                answer = SAFETY_RESPONSES[random.choice(safety_keys)]
                # Log the response length and return
                logger.info(f"Generated answer of {len(answer)} characters for {client_ip}")
                return jsonify({'answer': answer})
        
        # Check if input is too long to prevent CUDA errors - limit to 250 tokens for safety
        if len(question.split()) > 250:
            logger.warning(f"Input too long from {client_ip}, using truncated version")
            # Truncate the question to prevent errors
            question_words = question.split()[:200]  # Take first 200 words
            truncated_question = " ".join(question_words) + "..."  # Add ellipsis to indicate truncation
            logger.info(f"Truncated question from {len(question.split())} to 200 words")
            question = truncated_question
        
        try:
            # Use short-term memory for context if available
            model_input = question
            if hasattr(theta, 'memory'):
                # Memory handling is now done inside the ThetaInterface
                logger.info(f"Using memory-based context for session {session_id}")
            else:
                # Fallback to old conversation manager if memory not available
                context = conv_manager.get_formatted_context()
                if context:
                    logger.info(f"Using enhanced conversation context for session {session_id}")
                    model_input = f"{context}\nUser: {question}"
                
            # Update conversation flow state with the user's input
            if hasattr(theta, 'conversation_flow'):
                flow_state = theta.conversation_flow.process_input(question)
                logger.info(f"Conversation state: {flow_state['state']} for session {session_id}")
            
            # Generate answer using the enhanced Theta model
            raw_answer = theta.answer_question(model_input)
            
            # Enhance the response with conversation context references
            answer = conv_manager.enhance_response(raw_answer)
            
            # Add personalized touches from conversation flow if available
            if hasattr(theta, 'conversation_flow'):
                # Add user name if appropriate
                if theta.conversation_flow.should_add_personal_touch():
                    personalized_prefix = theta.conversation_flow.generate_personalized_prefix()
                    if personalized_prefix:
                        answer = personalized_prefix + answer
                
                # Add continuation prompt if needed
                if theta.conversation_flow.should_encourage_continuation():
                    continuation = theta.conversation_flow.generate_continuation_prompt()
                    if continuation:
                        answer = f"{answer}\n\n{continuation}"
            
            # Generate a follow-up question if appropriate
            followup = conv_manager.generate_followup_question()
            if followup:
                answer = f"{answer}\n\n{followup}"
            
            # Save the conversation exchange to the database
            conversation_id = conv_manager.add_exchange(question, answer)
            logger.info(f"Saved enhanced conversation exchange with ID: {conversation_id}")
            
            # Log any topics detected
            if conv_manager.current_topics:
                logger.info(f"Detected topics: {', '.join(conv_manager.current_topics)}")
            
            # Ensure no debug output makes it into the response
            # Look for debug artifacts like 'Duplicate of #9726' and remove them
            debug_patterns = [
                r'Duplicate of #\d+',
                r'Debug: .*',
                r'\[DEBUG\].*',
                r'Error #\d+',
                r'\[TRACE\].*'
            ]
            
            for pattern in debug_patterns:
                answer = re.sub(pattern, '', answer).strip()
            
            # Post-process validation
            # Removed '@' and other markers that may cause false positives
            hallucination_markers = [
                "[Source]", "[source]", "https://", "www.", "(link)", "[link]",
                "[citation", "Source:", ".com/", ".org/",
                "<![CDATA[", "<!DOCTYPE", "<html", "<script"
            ]
            
            if any(marker in answer for marker in hallucination_markers):
                logger.warning(f"Detected potential hallucination markers in response to {client_ip}")
                # Fall back to a safer response
                if is_definition_question and term:
                    answer = f"I don't have a specific definition for '{term}' in my knowledge base. This term might be specialized or outside my training data."
                else:
                    answer = f"I don't have specific information about '{question.strip()}' in my knowledge base. I'm trained on cybersecurity, software development, and IT concepts by Frostline Solutions."
                    
                # Update the saved response with the corrected answer
                if conversation_id:
                    db_manager.update_conversation_response(conversation_id, answer)
        
        except RuntimeError as e:
            # Catch CUDA errors and other runtime errors
            logger.error(f"RuntimeError during model inference: {str(e)}")
            answer = "I apologize, but I encountered a technical issue while processing your request. Please try asking a simpler or shorter question."
            
        except Exception as e:
            # Catch any other unexpected errors
            logger.error(f"Unexpected error during answer generation: {str(e)}")
            answer = "I apologize, but something went wrong while generating your answer. Please try again with a different question."
        
        # Use MathHandler for math expressions (including more advanced ones)
        if hasattr(theta, 'math') and theta.math.is_math_expression(question):
            try:
                result = theta.math.evaluate_expression(question)
                if result is not None:
                    formatted_result = theta.math.format_result(result)
                    answer = f"The answer is {formatted_result}."
                    logger.info(f"Calculated math result for {client_ip}: {question} = {formatted_result}")
                    return jsonify({'answer': answer, 'session_id': session_id})
            except Exception as e:
                logger.warning(f"Failed to evaluate math expression: {str(e)}")
        
        # Check if the response is unusually long for a simple question
        # Improved implementation with better heuristics and more appropriate responses
        word_ratio_threshold = 25  # Maximum ratio of answer words to question words
        question_words = len(question.split())
        answer_words = len(answer.split())
        
        # Don't apply this check to math expressions
        is_math_expression = hasattr(theta, 'math') and theta.math.is_math_expression(question)
        
        if not is_math_expression and question_words <= 5 and answer_words > 100 and (answer_words / max(1, question_words)) > word_ratio_threshold:
            logger.warning(f"Unusually long answer to short question from {client_ip}: {question_words} question words, {answer_words} answer words")
            
            # Determine the type of question to provide a better fallback response
            question_lower = question.lower().strip()
            
            # For "what is X" type questions
            if any(pattern in question_lower for pattern in ["what is", "what's", "whats", "define", "explain", "tell me about"]):
                term = question_lower.replace("what is", "").replace("what's", "").replace("whats", "").replace("define", "").replace("explain", "").replace("tell me about", "").strip('?').strip()
                answer = f"'{term}' has multiple aspects to explain. Could you be more specific about which aspects of {term} you're interested in learning about?"
            
            # For calculation questions
            elif any(char in question_lower for char in "+-*/×÷=") or any(word in question_lower for word in ["plus", "minus", "times", "divided", "equals", "calculate"]):
                answer = f"The answer to your calculation is {eval(re.sub(r'[^0-9+\-*/().]', '', question.replace('x', '*').replace('×', '*').replace('÷', '/')))}."
            
            # For general short questions
            else:
                answer = "I can provide a more helpful response if you ask a more specific question. What particular aspect are you interested in?"
        
        # Log the response length
        logger.info(f"Generated answer of {len(answer)} characters for {client_ip}")
        
        # Return answer with session ID and enhanced metadata for conversation continuity
        response_data = {
            'answer': answer,
            'session_id': session_id,
            'conversation_id': conversation_id,
            'detected_topics': conv_manager.current_topics if hasattr(conv_manager, 'current_topics') else [],
            'conversation_depth': conv_manager.conversation_state['conversation_depth'] if hasattr(conv_manager, 'conversation_state') else 0
        }
        
        # Add conversation flow information if available
        if hasattr(theta, 'conversation_flow'):
            response_data.update({
                'conversation_state': theta.conversation_flow.state,
                'current_topic': theta.conversation_flow.current_topic,
                'exchange_count': theta.conversation_flow.exchange_count
            })
            
            # Add user name if detected
            if theta.conversation_flow.user_name:
                response_data['user_name'] = theta.conversation_flow.user_name
        
        return jsonify(response_data)
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({'error': 'An internal error occurred'}), 500  # Don't expose specific error details

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'loaded'})

@app.route('/v1/chat/completions', methods=['POST'])
@csrf.exempt
def chat_completions():
    """OpenAI-compatible API endpoint for chat completions"""
    try:
        # Get client IP for logging
        client_ip = request.headers.get('CF-Connecting-IP') or \
                   request.headers.get('X-Forwarded-For', '').split(',')[0].strip() or \
                   request.remote_addr
        
        # Parse the request data
        data = request.json
        
        # Extract messages from the request
        messages = data.get('messages', [])
        if not messages:
            return jsonify({'error': 'No messages provided'}), 400
        
        # Get or create session ID from request
        session_id = data.get('session_id')
        if not session_id:
            session_id = str(uuid.uuid4())
            logger.info(f"Created new session {session_id} for API request from {client_ip}")
        
        # Extract the user's message (typically the last one)
        user_message = ""
        for message in reversed(messages):
            if message.get('role') == 'user':
                user_message = message.get('content', '')
                break
        
        if not user_message:
            return jsonify({'error': 'No user message found'}), 400
            
        # Initialize conversation manager
        conv_manager = ConversationManager(session_id=session_id)
        
        # Generate response using Theta
        try:
            # Use existing answer_question function
            response = theta.answer_question(user_message)
            
            # Save the conversation exchange
            conversation_id = conv_manager.add_exchange(user_message, response)
            
            # Format response in OpenAI-compatible format
            result = {
                'id': f'chatcmpl-{uuid.uuid4()}',
                'object': 'chat.completion',
                'created': int(datetime.datetime.now().timestamp()),
                'model': 'theta-ai',
                'choices': [{
                    'index': 0,
                    'message': {
                        'role': 'assistant',
                        'content': response
                    },
                    'finish_reason': 'stop'
                }],
                'usage': {
                    'prompt_tokens': len(user_message.split()),
                    'completion_tokens': len(response.split()),
                    'total_tokens': len(user_message.split()) + len(response.split())
                }
            }
            
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Error generating API response: {str(e)}")
            return jsonify({
                'error': {
                    'message': 'An error occurred during processing',
                    'type': 'internal_error',
                    'code': 500
                }
            }), 500
    
    except Exception as e:
        logger.error(f"Error in chat completions endpoint: {str(e)}")
        return jsonify({
            'error': {
                'message': 'An error occurred during processing',
                'type': 'internal_error',
                'code': 500
            }
        }), 500

@app.route('/feedback', methods=['POST'])
@csrf.exempt
def feedback():
    """Endpoint to collect user feedback on conversations."""
    try:
        # Get client IP for logging - support CloudFlare headers
        client_ip = request.headers.get('CF-Connecting-IP') or \
                   request.headers.get('X-Forwarded-For', '').split(',')[0].strip() or \
                   request.remote_addr
        
        # Get feedback data
        data = request.json
        conversation_id = data.get('conversation_id')
        rating = data.get('rating')  # 1-5 rating
        comment = data.get('comment', '')
        
        # Validate input
        if not conversation_id:
            return jsonify({'error': 'Missing conversation ID'}), 400
        if not rating or not isinstance(rating, int) or rating < 1 or rating > 5:
            return jsonify({'error': 'Invalid rating. Must be an integer from 1 to 5'}), 400
        
        logger.info(f"Received feedback for conversation {conversation_id} from {client_ip}: rating={rating}")
        
        # Save feedback to database
        db_manager = DatabaseManager()
        success = db_manager.save_user_feedback(conversation_id, rating, comment)
        
        if not success:
            logger.warning(f"Failed to save feedback for conversation {conversation_id}")
            return jsonify({'error': 'Failed to save feedback'}), 500
        
        # If good rating (4-5), consider adding to training data
        if rating >= 4:
            try:
                # Get the conversation
                connection = db_manager.get_connection()
                cursor = connection.cursor()
                
                cursor.execute(
                    """
                    SELECT question, answer FROM conversations 
                    WHERE id = %s AND processed_for_training = FALSE
                    """,
                    (conversation_id,)
                )
                
                result = cursor.fetchone()
                
                if result:
                    user_input, ai_response = result
                    
                    # Add to training data
                    db_manager.save_training_data(
                        question=user_input,
                        answer=ai_response,
                        category='conversation',
                        quality_score=float(rating)/5.0,
                        source='user_feedback'
                    )
                    
                    # Mark as processed
                    cursor.execute(
                        "UPDATE conversations SET processed_for_training = TRUE WHERE id = %s",
                        (conversation_id,)
                    )
                    connection.commit()
                    logger.info(f"Added conversation {conversation_id} to training data with quality score {float(rating)/5.0}")
                
                cursor.close()
                connection.close()
                
            except Exception as e:
                logger.error(f"Error adding conversation to training data: {str(e)}")
        
        return jsonify({'status': 'success', 'message': 'Feedback received'})
        
    except Exception as e:
        logger.error(f"Error processing feedback: {str(e)}")
        return jsonify({'error': 'An error occurred'}), 500

if __name__ == '__main__':
    print("Starting Theta AI Web Interface...")
    print("Access the interface at http://localhost:5000")
    # Set to False for production
    app.run(host='0.0.0.0', port=5000, debug=False)  # Accessible on local network
