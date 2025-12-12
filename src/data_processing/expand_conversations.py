"""
Generate expanded conversation examples for Theta AI training.
Creates a large number of varied conversational exchanges.
"""

import json
import random
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_expanded_conversations(count=500):
    """
    Generate expanded conversation examples with varied topics.
    
    Args:
        count: Number of conversation examples to generate
        
    Returns:
        list: Generated conversation examples
    """
    topics = [
        "Python", "JavaScript", "Docker", "Kubernetes", "Linux", "Windows", "Mac", 
        "Database", "SQL", "NoSQL", "Cloud", "AWS", "Azure", "GCP", "AI", "Machine Learning",
        "Neural Networks", "Cybersecurity", "Firewalls", "Encryption", "Networking",
        "Web Development", "Mobile Development", "DevOps", "Git", "Version Control",
        "Data Science", "Big Data", "APIs", "REST", "GraphQL", "Authentication",
        "Authorization", "Virtualization", "Containers", "Microservices", "Serverless",
        "CICD", "Testing", "Test Automation", "Front-end", "Back-end", "Full-stack"
    ]
    
    questions = [
        "How do I start with {}?",
        "What are the best practices for {}?",
        "Can you explain {} in simple terms?",
        "What's the difference between {} and other alternatives?",
        "How do I troubleshoot {} issues?",
        "What are common mistakes when using {}?",
        "What are the security concerns with {}?",
        "How does {} work under the hood?",
        "Can you recommend resources to learn {}?",
        "Is {} worth learning in 2025?",
        "How has {} evolved over the years?",
        "What are the main features of {}?",
        "Why would someone choose {} over alternatives?",
        "What are the limitations of {}?",
        "How do I optimize {} performance?",
        "Can you compare {} with its main competitors?"
    ]
    
    answers = {
        "start": "To get started with {}, I recommend beginning with the fundamentals. First, understand the core concepts of {} which include [SPECIFIC CONCEPTS]. For hands-on practice, try setting up a simple project that demonstrates basic {} functionality. There are excellent resources like the official {} documentation and tutorials on platforms like freeCodeCamp and Codecademy. Would you like me to suggest a specific first project idea for {}?",
        
        "best_practices": "When working with {}, following best practices is crucial for maintainable and efficient code. Some key best practices include: 1) [PRACTICE 1] which helps with [BENEFIT], 2) [PRACTICE 2] to ensure [BENEFIT], 3) Using established patterns for [SPECIFIC ASPECT], 4) Implementing proper error handling, and 5) Writing comprehensive tests. The community generally agrees that [SPECIFIC RECOMMENDATION] is particularly important. Are there any specific aspects of {} best practices you'd like me to elaborate on?",
        
        "simple_terms": "{} can be explained simply as [SIMPLE EXPLANATION]. Think of it like [ANALOGY] where [ANALOGY EXPLANATION]. The main purpose of {} is to [CORE PURPOSE], and it's commonly used when you need to [USE CASE]. Unlike [ALTERNATIVE], {} focuses on [DISTINGUISHING FEATURE]. Does this explanation help clarify what {} is?",
        
        "difference": "The main differences between {} and alternatives include: 1) {} emphasizes [FEATURE] while others focus on [ALTERNATIVE APPROACH], 2) {} has [CHARACTERISTIC] which makes it better for [USE CASE], 3) The learning curve for {} is typically [STEEPER/GENTLER] compared to alternatives, 4) {} offers [UNIQUE BENEFIT] that you won't find elsewhere, and 5) The community and ecosystem around {} is [DESCRIPTION]. Based on your specific needs, I can help determine if {} would be the best choice for your situation.",
        
        "troubleshoot": "When troubleshooting {} issues, follow these steps: 1) Check for [COMMON ERROR SOURCES], 2) Verify your [CONFIGURATION/SETUP], 3) Look at [SPECIFIC LOGS OR ERROR MESSAGES], 4) Use debugging tools like [TOOL NAMES], 5) Consider common pitfalls such as [PITFALL LIST]. One particularly effective strategy is to [SPECIFIC STRATEGY]. If you're facing a specific {} issue, please share the error message or symptoms and I can provide more targeted troubleshooting advice.",
    }
    
    answer_types = list(answers.keys())
    
    conversations = []
    
    for _ in range(count):
        topic = random.choice(topics)
        question_template = random.choice(questions)
        question = question_template.format(topic)
        
        # Choose a relevant answer type
        q_lower = question.lower()
        if "start" in q_lower or "begin" in q_lower:
            answer_type = "start"
        elif "best practice" in q_lower:
            answer_type = "best_practices"
        elif "explain" in q_lower or "what is" in q_lower or "what's" in q_lower:
            answer_type = "simple_terms"
        elif "difference" in q_lower or "compare" in q_lower:
            answer_type = "difference"
        elif "troubleshoot" in q_lower or "issue" in q_lower or "problem" in q_lower or "error" in q_lower:
            answer_type = "troubleshoot"
        else:
            answer_type = random.choice(answer_types)
            
        # Format answer with the topic repeated as needed
        if answer_type == "start":
            answer = answers[answer_type].format(topic, topic, topic, topic, topic)
        elif answer_type == "best_practices":
            answer = answers[answer_type].format(topic, topic)
        elif answer_type == "simple_terms":
            answer = answers[answer_type].format(topic, topic, topic, topic, topic)
        elif answer_type == "difference":
            answer = answers[answer_type].format(topic, topic, topic, topic, topic, topic, topic)
        elif answer_type == "troubleshoot":
            answer = answers[answer_type].format(topic, topic)
        
        conversations.append({
            "question": question,
            "answer": answer
        })
    
    return conversations

def save_expanded_conversations(count=1000):
    """
    Generate and save expanded conversation examples.
    
    Args:
        count: Number of conversation examples to generate
        
    Returns:
        str: Path to the saved dataset
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    datasets_dir = project_root / "Datasets"
    datasets_dir.mkdir(exist_ok=True)
    
    # Generate expanded conversations
    logger.info(f"Generating {count} expanded conversation examples...")
    conversations = generate_expanded_conversations(count)
    
    # Save expanded conversations
    expanded_path = datasets_dir / "expanded_conversations.json"
    with open(expanded_path, 'w', encoding='utf-8') as f:
        json.dump(conversations, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Created {len(conversations)} expanded conversation examples at {expanded_path}")
    return str(expanded_path)

def main():
    """Main function to generate and save expanded conversations."""
    return save_expanded_conversations(1000)

if __name__ == "__main__":
    main()
