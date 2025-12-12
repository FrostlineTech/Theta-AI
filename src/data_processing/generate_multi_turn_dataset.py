"""
Generate multi-turn conversation datasets for Theta AI training.
"""

import json
import random
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Technical domains for conversations
DOMAINS = {
    'programming': [
        'Python', 'JavaScript', 'TypeScript', 'Java', 'C#', 'Go', 'Rust',
        'functional programming', 'object-oriented programming', 'API design',
        'database access', 'microservices', 'web development', 'algorithm optimization',
        'frontend frameworks', 'unit testing', 'CI/CD pipelines'
    ],
    'networking': [
        'TCP/IP', 'network segmentation', 'BGP', 'firewalls', 'VPN', 'routing protocols',
        'DHCP', 'DNS', 'IPv6', 'subnetting', 'OSI model', 'network troubleshooting',
        'network performance', 'SDN', 'zero trust networks'
    ],
    'cybersecurity': [
        'penetration testing', 'vulnerability assessment', 'SIEM', 'encryption',
        'identity management', 'threat detection', 'incident response', 'malware analysis',
        'secure coding practices', 'security operations center', 'ransomware protection',
        'phishing prevention', 'security frameworks', 'compliance'
    ],
    'cloud': [
        'AWS', 'Azure', 'GCP', 'Kubernetes', 'Docker', 'serverless', 'microservices',
        'cloud cost optimization', 'auto-scaling', 'Infrastructure as Code',
        'cloud security', 'cloud migration', 'hybrid cloud', 'multi-cloud strategy',
        'cloud storage', 'cloud databases', 'container orchestration'
    ],
    'data_science': [
        'machine learning', 'data analysis', 'data visualization', 'natural language processing',
        'neural networks', 'computer vision', 'predictive modeling', 'statistical analysis',
        'feature engineering', 'data preprocessing', 'model evaluation', 'deep learning',
        'decision trees', 'clustering algorithms'
    ]
}

# Templates for multi-turn conversations
MULTI_TURN_TEMPLATES = [
    # Progressive technical learning conversation
    {
        "type": "learning",
        "pattern": [
            {"role": "user", "template": "Can you explain {topic} in simple terms?"},
            {"role": "assistant", "template": "Sure! {topic} is {explanation}. Does that make sense?"},
            {"role": "user", "template": "Yes, but how does {topic} relate to {related_topic}?"},
            {"role": "assistant", "template": "Great question! {topic} and {related_topic} are connected because {connection}. This is important because {importance}."},
            {"role": "user", "template": "Can you give me an example of {topic} in a real-world scenario?"},
            {"role": "assistant", "template": "Absolutely! Here's a real-world example: {example}. This demonstrates how {topic} is applied in practice."}
        ]
    },
    # Troubleshooting conversation
    {
        "type": "troubleshooting",
        "pattern": [
            {"role": "user", "template": "I'm having an issue with {topic}. {problem}"},
            {"role": "assistant", "template": "I understand you're having trouble with {topic}. Let's troubleshoot this. First, could you tell me {question}?"},
            {"role": "user", "template": "Yes, {answer_to_question}."},
            {"role": "assistant", "template": "Thanks for that information. Based on what you've described, the issue might be {potential_cause}. Let's try {solution_step}."},
            {"role": "user", "template": "I tried that but I'm still seeing {follow_up_problem}."},
            {"role": "assistant", "template": "I see. Since {solution_step} didn't resolve it, let's try a different approach. {alternative_solution}. This should address the {follow_up_problem}."}
        ]
    },
    # Project planning conversation
    {
        "type": "project_planning",
        "pattern": [
            {"role": "user", "template": "I need to build a {project_type} that {project_requirement}. Where should I start?"},
            {"role": "assistant", "template": "For a {project_type} that {project_requirement}, I recommend starting with {first_step}. This provides {benefit}."},
            {"role": "user", "template": "That makes sense. What technology stack would you recommend for this?"},
            {"role": "assistant", "template": "For your {project_type}, I'd recommend using {technology_stack}. This combination offers {stack_benefits}."},
            {"role": "user", "template": "How long do you think this project would take to implement?"},
            {"role": "assistant", "template": "For a {project_type} with these requirements, I'd estimate {timeframe}. The most time-intensive parts will be {complex_aspects}."}
        ]
    },
    # Comparative discussion
    {
        "type": "comparison",
        "pattern": [
            {"role": "user", "template": "What's better, {option1} or {option2}?"},
            {"role": "assistant", "template": "Both {option1} and {option2} have their strengths. {option1} excels at {strength1}, while {option2} is better for {strength2}."},
            {"role": "user", "template": "Which would you recommend for {specific_use_case}?"},
            {"role": "assistant", "template": "For {specific_use_case}, I'd recommend {recommendation} because {reasoning}. It provides {specific_benefit} which is important in your case."},
            {"role": "user", "template": "What about {alternative_option}? Would that work too?"},
            {"role": "assistant", "template": "{alternative_option} could also work, but there are some considerations: {considerations}. Compared to {recommendation}, it offers {trade_offs}."}
        ]
    },
    # Career advice conversation
    {
        "type": "career_advice",
        "pattern": [
            {"role": "user", "template": "I want to become a {role}. What skills should I learn?"},
            {"role": "assistant", "template": "To become a {role}, you should focus on these key skills: {skills}. Additionally, it's helpful to have knowledge of {additional_knowledge}."},
            {"role": "user", "template": "How long would it take to learn enough to get an entry-level position?"},
            {"role": "assistant", "template": "For an entry-level {role} position, you could be ready in {timeframe} if you study consistently. Focus first on {priority_skills}, as these are most often required for entry-level positions."},
            {"role": "user", "template": "What's the job market like for {role} roles?"},
            {"role": "assistant", "template": "The job market for {role} positions is currently {market_condition}. {market_details}. Industries like {industries} are particularly looking for these skills."}
        ]
    }
]

# Content fillers for templates
CONTENT = {
    "explanations": {
        "Python": "a high-level programming language known for its readability and simplicity. It uses indentation to define code blocks and has a vast ecosystem of libraries",
        "JavaScript": "a programming language primarily used for web development. It allows you to add interactive elements to websites and runs directly in the browser",
        "machine learning": "a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed",
        "TCP/IP": "the fundamental communication protocol of the internet. It defines how data should be formatted, addressed, transmitted, and received by devices on a network",
        "cloud computing": "the delivery of computing services over the internet rather than having local servers or personal devices handling applications",
        "penetration testing": "a simulated cyber attack against your computer system to check for exploitable vulnerabilities"
    },
    "connections": {
        "Python-JavaScript": "both are high-level programming languages, but Python is often used for backend processing while JavaScript is primarily for frontend web development",
        "machine learning-data analysis": "data analysis is a prerequisite for effective machine learning, as clean, well-understood data is essential for training accurate models",
        "TCP/IP-network segmentation": "network segmentation relies on TCP/IP addressing to create separate network zones, improving both security and performance",
        "cloud computing-Kubernetes": "Kubernetes is an orchestration platform that automates the deployment and management of containerized applications in cloud environments"
    },
    "problems": {
        "Python": "I keep getting IndentationError when running my code",
        "JavaScript": "my event listeners aren't firing when I click on elements",
        "network": "users can't access the company intranet from the guest WiFi",
        "cybersecurity": "we're seeing unusual login attempts from foreign IP addresses",
        "cloud": "our AWS Lambda functions are timing out randomly"
    },
    "questions": {
        "Python": "which version of Python you're using and if you're mixing tabs and spaces?",
        "JavaScript": "how you're attaching the event listeners and if there are any console errors?",
        "network": "if your guest WiFi is properly segregated from the internal network?",
        "cybersecurity": "if you have multi-factor authentication enabled on these accounts?",
        "cloud": "what the execution timeout is set to and the typical processing time?"
    }
}

# Technical roles for career conversations
ROLES = [
    "frontend developer", "backend developer", "full stack developer", 
    "DevOps engineer", "cloud architect", "machine learning engineer",
    "data scientist", "network administrator", "cybersecurity analyst",
    "penetration tester", "database administrator", "site reliability engineer",
    "mobile app developer"
]

class MultiTurnGenerator:
    """Generates multi-turn conversation datasets."""
    
    def __init__(self):
        """Initialize generator."""
        random.seed(42)  # For reproducibility
    
    def _get_random_topic(self, domain):
        """Get a random topic from the specified domain."""
        return random.choice(DOMAINS.get(domain, DOMAINS['programming']))
    
    def _get_related_topic(self, topic):
        """Get a topic related to the given topic."""
        # For simplicity, just choose another random topic
        domain = None
        for d, topics in DOMAINS.items():
            if topic in topics:
                domain = d
                break
        
        domain = domain or random.choice(list(DOMAINS.keys()))
        topics = [t for t in DOMAINS[domain] if t != topic]
        return random.choice(topics) if topics else topic
    
    def _fill_template_slot(self, template, slot, options):
        """Fill a slot in a template with one of the provided options."""
        if slot in options:
            return template.replace(f"{{{slot}}}", options[slot])
        
        # Default fillers for common slots
        if slot == "topic":
            domain = random.choice(list(DOMAINS.keys()))
            return template.replace(f"{{{slot}}}", self._get_random_topic(domain))
        
        if slot == "related_topic":
            if "topic" in options:
                return template.replace(f"{{{slot}}}", self._get_related_topic(options["topic"]))
            return template.replace(f"{{{slot}}}", self._get_random_topic(random.choice(list(DOMAINS.keys()))))
        
        if slot == "explanation" and "topic" in options:
            topic_key = options["topic"].lower()
            explanation = CONTENT["explanations"].get(topic_key, f"a technology used in {random.choice(['software development', 'network architecture', 'cybersecurity', 'data science', 'cloud computing'])}")
            return template.replace(f"{{{slot}}}", explanation)
        
        if slot == "connection" and "topic" in options and "related_topic" in options:
            key = f"{options['topic']}-{options['related_topic']}"
            alt_key = f"{options['related_topic']}-{options['topic']}"
            connection = CONTENT["connections"].get(key, CONTENT["connections"].get(alt_key, f"they both play important roles in {random.choice(['modern technology stacks', 'enterprise solutions', 'development workflows', 'security protocols'])}"))
            return template.replace(f"{{{slot}}}", connection)
        
        if slot == "problem" and "topic" in options:
            domain = None
            topic = options["topic"].lower()
            for d, topics in DOMAINS.items():
                if any(t.lower() == topic for t in topics):
                    domain = d
                    break
            
            domain = domain or "Python"
            problem = CONTENT["problems"].get(domain, f"we're experiencing unexpected behavior")
            return template.replace(f"{{{slot}}}", problem)
        
        if slot == "question" and "topic" in options:
            domain = None
            topic = options["topic"].lower()
            for d, topics in DOMAINS.items():
                if any(t.lower() == topic for t in topics):
                    domain = d
                    break
            
            domain = domain or "Python"
            question = CONTENT["questions"].get(domain, f"can you provide more details about the issue?")
            return template.replace(f"{{{slot}}}", question)
        
        # Generic fillers for other slots
        fillers = {
            "importance": f"it enables more efficient and reliable solutions in {random.choice(['modern software', 'enterprise environments', 'cloud architectures', 'security systems'])}",
            "example": f"consider a {random.choice(['financial system', 'e-commerce platform', 'healthcare application', 'social media site'])} that needs to {random.choice(['process transactions', 'authenticate users', 'analyze large datasets', 'ensure high availability'])}",
            "answer_to_question": f"I checked and {random.choice(['yes, that\'s exactly what\'s happening', 'no, I don\'t think that\'s the issue', 'I\'m not sure, but I can investigate further'])}",
            "potential_cause": f"{random.choice(['a configuration mismatch', 'outdated dependencies', 'insufficient permissions', 'resource limitations'])}",
            "solution_step": f"{random.choice(['updating the configuration', 'clearing the cache', 'reinstalling the dependencies', 'checking the logs for more details'])}",
            "follow_up_problem": f"{random.choice(['the same error', 'a different error message', 'poor performance', 'intermittent failures'])}",
            "alternative_solution": f"let's {random.choice(['check the system requirements', 'try an older version', 'apply this specific patch', 'modify the initialization parameters'])}",
            "project_type": f"{random.choice(['web application', 'mobile app', 'data pipeline', 'API service', 'monitoring system'])}",
            "project_requirement": f"{random.choice(['handles high traffic', 'processes sensitive data securely', 'integrates with legacy systems', 'operates in real-time'])}",
            "first_step": f"{random.choice(['defining clear requirements', 'creating a system architecture diagram', 'selecting your technology stack', 'setting up the development environment'])}",
            "benefit": f"{random.choice(['a solid foundation', 'clearer direction for the project', 'easier collaboration', 'fewer complications later'])}",
            "technology_stack": f"{random.choice(['Python with Django', 'Node.js with Express', 'React with a GraphQL API', 'Go with PostgreSQL'])}",
            "stack_benefits": f"{random.choice(['scalability and maintainability', 'rapid development and flexibility', 'performance and security', 'strong community support'])}",
            "timeframe": f"{random.choice(['2-3 months', '4-6 weeks', 'about a quarter', 'at least 6 months'])}",
            "complex_aspects": f"{random.choice(['the data migration', 'user authentication system', 'third-party integrations', 'scalability testing'])}",
            "option1": f"{random.choice(['AWS', 'React', 'MongoDB', 'Docker', 'Python'])}",
            "option2": f"{random.choice(['Azure', 'Angular', 'PostgreSQL', 'Kubernetes', 'Java'])}",
            "strength1": f"{random.choice(['ease of use', 'performance', 'flexibility', 'community support', 'integration capabilities'])}",
            "strength2": f"{random.choice(['enterprise features', 'stability', 'scalability', 'security', 'standardization'])}",
            "specific_use_case": f"{random.choice(['a small startup', 'an enterprise solution', 'a high-traffic application', 'a data-intensive service'])}",
            "recommendation": "{option1}",  # Default to first option
            "reasoning": f"it better aligns with {random.choice(['your specific requirements', 'your team\'s expertise', 'your technology stack', 'your budget constraints'])}",
            "specific_benefit": f"{random.choice(['faster development cycles', 'lower maintenance costs', 'better performance', 'more flexibility'])}",
            "alternative_option": f"{random.choice(['Google Cloud', 'Vue.js', 'Redis', 'Terraform', 'Go'])}",
            "considerations": f"{random.choice(['it has a steeper learning curve', 'it costs more initially', 'it requires specialized knowledge', 'it has fewer integrations'])}",
            "trade_offs": f"{random.choice(['better performance but more complexity', 'lower costs but fewer features', 'easier maintenance but less flexibility', 'better security but higher overhead'])}",
            "role": f"{random.choice(ROLES)}",
            "skills": f"{random.choice(['programming languages like Python and JavaScript', 'cloud services like AWS and Azure', 'data structures and algorithms', 'networking fundamentals'])}",
            "additional_knowledge": f"{random.choice(['version control systems', 'agile methodologies', 'testing frameworks', 'CI/CD pipelines'])}",
            "timeframe": f"{random.choice(['6-12 months', 'about a year', '3-6 months with intensive study', '1-2 years part-time'])}",
            "priority_skills": f"{random.choice(['core programming concepts', 'fundamental design patterns', 'basic security principles', 'common data structures'])}",
            "market_condition": f"{random.choice(['growing rapidly', 'steady and stable', 'competitive but promising', 'in high demand'])}",
            "market_details": f"{random.choice(['Many companies are struggling to fill positions', 'Remote opportunities have expanded the job market', 'Entry-level positions are abundant', 'Specialized roles command high salaries'])}",
            "industries": f"{random.choice(['finance and healthcare', 'technology and e-commerce', 'government and education', 'manufacturing and logistics'])}"
        }
        
        if slot in fillers:
            return template.replace(f"{{{slot}}}", fillers[slot])
        
        # Default case
        return template.replace(f"{{{slot}}}", f"[{slot}]")
    
    def _fill_template(self, template, initial_values=None):
        """
        Fill a template with values.
        
        Args:
            template (dict): Template to fill
            initial_values (dict): Initial values to use
            
        Returns:
            dict: Filled template
        """
        values = initial_values or {}
        result = template["template"]
        
        # Find all slots in the template
        import re
        slots = re.findall(r'\{([^}]+)\}', template["template"])
        
        # Fill each slot
        for slot in slots:
            result = self._fill_template_slot(result, slot, values)
            
            # If the slot was filled with a value, add it to values
            # for use in subsequent templates
            if f"{{{slot}}}" not in result:
                filled_value = result.split(f"{slot}:")[1].split(",")[0] if f"{slot}:" in result else None
                if filled_value:
                    values[slot] = filled_value
        
        return {
            "role": template["role"],
            "content": result
        }
    
    def generate_conversation(self, template_type=None):
        """
        Generate a complete conversation from a template.
        
        Args:
            template_type (str, optional): Type of template to use
            
        Returns:
            list: List of conversation exchanges
        """
        # Select a template
        if template_type:
            templates = [t for t in MULTI_TURN_TEMPLATES if t["type"] == template_type]
            template = random.choice(templates) if templates else random.choice(MULTI_TURN_TEMPLATES)
        else:
            template = random.choice(MULTI_TURN_TEMPLATES)
        
        # Initialize values
        values = {}
        
        # Fill each part of the template
        conversation = []
        for part in template["pattern"]:
            filled = self._fill_template(part, values)
            conversation.append(filled)
            
            # Update values based on filled content
            if part["role"] == "user":
                # Extract topic from user messages if not already set
                if "topic" not in values:
                    for domain, topics in DOMAINS.items():
                        for topic in topics:
                            if topic.lower() in filled["content"].lower():
                                values["topic"] = topic
                                break
                
                # Extract other potential values
                for key in ["option1", "option2", "project_type", "role"]:
                    if key not in values:
                        for option in CONTENT.get(key, []):
                            if option.lower() in filled["content"].lower():
                                values[key] = option
                                break
        
        return conversation
    
    def generate_dataset(self, count=100):
        """
        Generate a dataset of multi-turn conversations.
        
        Args:
            count (int): Number of conversations to generate
            
        Returns:
            list: Generated dataset
        """
        dataset = []
        
        for _ in range(count):
            # Generate a conversation
            conversation = self.generate_conversation()
            
            # Format as QA pairs for Theta
            for i in range(0, len(conversation), 2):
                if i + 1 < len(conversation):
                    user_msg = conversation[i]["content"]
                    ai_msg = conversation[i+1]["content"]
                    
                    dataset.append({
                        "question": user_msg,
                        "answer": ai_msg,
                        "context": conversation[:i] if i > 0 else []
                    })
        
        return dataset
    
    def save_dataset(self, output_file, count=100):
        """
        Generate and save a multi-turn conversation dataset.
        
        Args:
            output_file (str): Path to output file
            count (int): Number of conversations to generate
            
        Returns:
            str: Path to saved file
        """
        # Generate dataset
        dataset = self.generate_dataset(count)
        
        # Ensure output directory exists
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save dataset
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Created {len(dataset)} multi-turn conversation examples at {output_file}")
        return output_file

def generate_multi_turn_dataset(count=1000):
    """
    Generate a multi-turn conversation dataset.
    
    Args:
        count (int): Number of conversation turns to generate
        
    Returns:
        str: Path to saved file
    """
    generator = MultiTurnGenerator()
    output_file = Path("Datasets") / "enhanced_conversations.json"
    return generator.save_dataset(output_file, count)

if __name__ == "__main__":
    logger.info("Generating multi-turn conversation dataset...")
    output_file = generate_multi_turn_dataset(1000)
    logger.info(f"Generated multi-turn conversation dataset at {output_file}")
