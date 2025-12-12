"""
Synthetic Data Generation Pipeline for Theta AI

This module generates synthetic training data to enhance Theta's knowledge base.
"""

import json
import logging
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SyntheticDataGenerator:
    """Generates synthetic training data for Theta AI."""
    
    def __init__(self, datasets_dir: Path):
        """
        Initialize the synthetic data generator.
        
        Args:
            datasets_dir: Path to the datasets directory
        """
        self.datasets_dir = datasets_dir
        
        # Create output directory for synthetic data
        self.synthetic_dir = datasets_dir / "synthetic_data"
        os.makedirs(self.synthetic_dir, exist_ok=True)
        
        # Define templates for different domains
        self.templates = self._define_templates()
        
        # Load existing data for enhancing templates
        self.existing_data = self._load_existing_data()
    
    def _define_templates(self) -> Dict:
        """
        Define templates for synthetic data generation.
        
        Returns:
            Dictionary of templates by domain
        """
        return {
            "cybersecurity": {
                "factual": [
                    {"question": "What is {concept} in cybersecurity?", 
                     "answer_template": "{concept} refers to {definition}. {details}"},
                    {"question": "How does {technique} work?", 
                     "answer_template": "{technique} works by {method}. {process}"},
                    {"question": "What are common {attack_type} attacks?", 
                     "answer_template": "Common {attack_type} attacks include {examples}. {details}"}
                ],
                "procedural": [
                    {"question": "How do I implement {security_measure}?", 
                     "answer_template": "To implement {security_measure}, follow these steps:\n\n{steps}"},
                    {"question": "How can I protect against {threat}?", 
                     "answer_template": "To protect against {threat}, consider these measures:\n\n{measures}"},
                    {"question": "What's the procedure for handling a {incident_type} incident?", 
                     "answer_template": "When handling a {incident_type} incident:\n\n{procedure}"}
                ],
                "scenario": [
                    {"question": "What should I do if {breach_scenario}?", 
                     "answer_template": "If {breach_scenario}, you should:\n\n{response}"},
                    {"question": "How would you respond to {attack_scenario}?", 
                     "answer_template": "When facing {attack_scenario}, the recommended response is:\n\n{strategy}"}
                ]
            },
            "programming": {
                "factual": [
                    {"question": "What is {concept} in {language}?", 
                     "answer_template": "In {language}, {concept} is {definition}. {details}"},
                    {"question": "How does {data_structure} work?", 
                     "answer_template": "A {data_structure} works by {mechanism}. {details}"},
                    {"question": "What's the difference between {concept1} and {concept2}?", 
                     "answer_template": "The key differences between {concept1} and {concept2} are:\n\n{differences}"}
                ],
                "procedural": [
                    {"question": "How do I implement {algorithm} in {language}?", 
                     "answer_template": "To implement {algorithm} in {language}:\n\n```{language}\n{code}\n```\n\n{explanation}"},
                    {"question": "How can I optimize {performance_aspect} in my {language} code?", 
                     "answer_template": "To optimize {performance_aspect} in {language}:\n\n{techniques}\n\n```{language}\n{example_code}\n```"},
                    {"question": "What's the correct way to handle {error_type} exceptions in {language}?", 
                     "answer_template": "To handle {error_type} exceptions in {language}:\n\n```{language}\n{code}\n```\n\n{best_practices}"}
                ],
                "code_examples": [
                    {"question": "Can you give me an example of {feature} in {language}?", 
                     "answer_template": "Here's an example of {feature} in {language}:\n\n```{language}\n{code}\n```\n\n{explanation}"},
                    {"question": "How do I use {library} to {task} in {language}?", 
                     "answer_template": "To use {library} for {task} in {language}:\n\n```{language}\n{code}\n```\n\n{usage_notes}"}
                ]
            },
            "networking": {
                "factual": [
                    {"question": "What is {protocol}?", 
                     "answer_template": "{protocol} is a {type} protocol that {function}. {details}"},
                    {"question": "How does {network_technology} work?", 
                     "answer_template": "{network_technology} works by {mechanism}. {details}"},
                    {"question": "What are the advantages of {technology} over {alternative}?", 
                     "answer_template": "Advantages of {technology} compared to {alternative} include:\n\n{advantages}"}
                ],
                "procedural": [
                    {"question": "How do I configure {network_device} for {purpose}?", 
                     "answer_template": "To configure {network_device} for {purpose}:\n\n{steps}"},
                    {"question": "What's the process for troubleshooting {network_issue}?", 
                     "answer_template": "When troubleshooting {network_issue}, follow these steps:\n\n{troubleshooting_steps}"}
                ]
            }
        }
    
    def _load_existing_data(self) -> Dict:
        """
        Load existing datasets to use for template enhancement.
        
        Returns:
            Dictionary of existing data by domain
        """
        existing_data = {"cybersecurity": [], "programming": [], "networking": []}
        
        try:
            # Find all JSON files in datasets directory
            for file_path in self.datasets_dir.glob("*.json"):
                try:
                    # Load with UTF-8 encoding and error handling
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                            data = json.load(f)
                    except json.JSONDecodeError:
                        # If that fails, try a more aggressive approach with binary reading
                        logger.info(f"JSON decode error with utf-8 for {file_path}, trying alternative approach")
                        with open(file_path, 'rb') as f:
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
                    
                    # Determine domain based on filename
                    filename = file_path.name.lower()
                    domain = None
                    if any(term in filename for term in ["security", "cyber", "threat", "attack"]):
                        domain = "cybersecurity"
                    elif any(term in filename for term in ["program", "code", "dev", "language"]):
                        domain = "programming"
                    elif any(term in filename for term in ["network", "protocol", "routing"]):
                        domain = "networking"
                    
                    # Add to appropriate domain if it's in QA format
                    if domain and isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict) and "question" in item and "answer" in item:
                                existing_data[domain].append(item)
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
        
        except Exception as e:
            logger.error(f"Error loading existing data: {e}")
        
        # Log results
        for domain, items in existing_data.items():
            logger.info(f"Loaded {len(items)} existing QA pairs for {domain}")
        
        return existing_data
    
    def generate_synthetic_data(self, domain: str, count: int = 100) -> List[Dict]:
        """
        Generate synthetic QA pairs for a specific domain.
        
        Args:
            domain: Domain to generate data for
            count: Number of QA pairs to generate
            
        Returns:
            List of generated QA pairs
        """
        if domain not in self.templates:
            logger.warning(f"No templates available for domain '{domain}'")
            return []
        
        # Get domain-specific content
        domain_content = self._get_domain_content(domain)
        
        # Generate QA pairs
        qa_pairs = []
        for _ in range(count):
            # Select random template type
            template_type = random.choice(list(self.templates[domain].keys()))
            template_list = self.templates[domain][template_type]
            
            # Select random template
            template = random.choice(template_list)
            
            # Fill template with domain content
            qa_pair = self._fill_template(domain, template, domain_content)
            
            if qa_pair:
                qa_pairs.append(qa_pair)
        
        # Save generated data
        output_path = self.synthetic_dir / f"{domain}_synthetic.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Generated {len(qa_pairs)} synthetic QA pairs for {domain}")
        
        return qa_pairs
    
    def _get_domain_content(self, domain: str) -> Dict:
        """
        Get content for templates based on domain.
        
        Args:
            domain: The domain to get content for
            
        Returns:
            Dictionary of content for template filling
        """
        # Domain-specific content
        if domain == "cybersecurity":
            return {
                "concepts": ["zero trust", "defense in depth", "principle of least privilege", 
                           "vulnerability management", "threat modeling", "risk assessment", 
                           "security by design", "security control", "security posture"],
                "definitions": {
                    "zero trust": "a security model that requires strict identity verification for every person and device trying to access resources on a private network, regardless of location",
                    "defense in depth": "a cybersecurity strategy that employs multiple layers of security controls throughout an IT system",
                    "principle of least privilege": "a computer security concept that limits user account and process permissions to only those absolutely required"
                },
                "techniques": ["port scanning", "packet sniffing", "penetration testing", "social engineering", 
                              "threat hunting", "security monitoring", "vulnerability scanning"],
                "attack_types": ["phishing", "ransomware", "DDoS", "man-in-the-middle", "SQL injection", 
                               "cross-site scripting", "credential stuffing", "brute force"],
                "security_measures": ["multi-factor authentication", "encryption", "network segmentation", 
                                    "endpoint protection", "security awareness training"],
                "incidents": ["data breach", "ransomware attack", "insider threat", "credential compromise", 
                             "malware infection"]
            }
        elif domain == "programming":
            return {
                "languages": ["Python", "JavaScript", "Java", "C++", "Go", "Ruby", "TypeScript", "PHP", "Rust"],
                "concepts": ["inheritance", "encapsulation", "polymorphism", "recursion", "concurrency", 
                           "functional programming", "object-oriented programming", "asynchronous programming"],
                "data_structures": ["array", "linked list", "stack", "queue", "hash table", "tree", "graph", 
                                  "heap", "trie"],
                "algorithms": ["binary search", "quicksort", "merge sort", "breadth-first search", 
                              "depth-first search", "dynamic programming", "greedy algorithm"],
                "libraries": {
                    "Python": ["pandas", "numpy", "tensorflow", "scikit-learn", "requests", "flask", "django"],
                    "JavaScript": ["react", "vue", "angular", "express", "lodash", "axios", "d3"]
                },
                "code_examples": {
                    "Python": {
                        "binary_search": "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    \n    while left <= right:\n        mid = (left + right) // 2\n        \n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    \n    return -1",
                        "class_definition": "class User:\n    def __init__(self, name, email):\n        self.name = name\n        self.email = email\n        self.is_active = True\n    \n    def deactivate(self):\n        self.is_active = False\n        print(f\"{self.name} has been deactivated\")\n    \n    def __str__(self):\n        status = \"active\" if self.is_active else \"inactive\"\n        return f\"{self.name} ({self.email}) - {status}\""
                    },
                    "JavaScript": {
                        "async_function": "async function fetchUserData(userId) {\n  try {\n    const response = await fetch(`https://api.example.com/users/${userId}`);\n    \n    if (!response.ok) {\n      throw new Error(`HTTP error! Status: ${response.status}`);\n    }\n    \n    const userData = await response.json();\n    return userData;\n  } catch (error) {\n    console.error('Error fetching user data:', error);\n    throw error;\n  }\n}",
                        "react_component": "function UserProfile({ user, onUpdate }) {\n  const [isEditing, setIsEditing] = useState(false);\n  const [name, setName] = useState(user.name);\n  \n  const handleSubmit = (e) => {\n    e.preventDefault();\n    onUpdate({ ...user, name });\n    setIsEditing(false);\n  };\n  \n  return (\n    <div className=\"user-profile\">\n      {isEditing ? (\n        <form onSubmit={handleSubmit}>\n          <input\n            value={name}\n            onChange={(e) => setName(e.target.value)}\n          />\n          <button type=\"submit\">Save</button>\n        </form>\n      ) : (\n        <>\n          <h2>{user.name}</h2>\n          <button onClick={() => setIsEditing(true)}>Edit</button>\n        </>\n      )}\n    </div>\n  );\n}"
                    }
                }
            }
        elif domain == "networking":
            return {
                "protocols": ["TCP", "UDP", "HTTP", "HTTPS", "DNS", "DHCP", "SMTP", "FTP", 
                            "SSH", "TLS/SSL", "ICMP", "BGP", "OSPF"],
                "protocol_info": {
                    "TCP": "a connection-oriented protocol that provides reliable, ordered, and error-checked delivery of data",
                    "UDP": "a connectionless protocol that provides a simple, unreliable datagram service",
                    "HTTP": "an application layer protocol for transmitting hypermedia documents on the World Wide Web",
                    "DNS": "a hierarchical and decentralized naming system for computers, services, or other resources connected to the Internet or a private network"
                },
                "network_technologies": ["VPN", "SDN", "MPLS", "VoIP", "SD-WAN", "CDN", "Load Balancing", "NAT"],
                "network_devices": ["router", "switch", "firewall", "load balancer", "access point", 
                                   "gateway", "proxy server", "IDS/IPS"],
                "network_issues": ["packet loss", "high latency", "DNS resolution failure", 
                                 "routing loop", "bandwidth congestion", "IP conflict"]
            }
        else:
            # Default empty content
            return {}
    
    def _fill_template(self, domain: str, template: Dict, content: Dict) -> Optional[Dict]:
        """
        Fill a template with domain-specific content.
        
        Args:
            domain: Domain for the template
            template: Template dict with question and answer templates
            content: Domain-specific content
            
        Returns:
            Filled QA pair or None if failed
        """
        try:
            question_template = template["question"]
            answer_template = template["answer_template"]
            
            # Identify placeholders in the template
            placeholders = re.findall(r'\{([^}]+)\}', question_template)
            
            # Fill placeholders with appropriate content
            replacements = {}
            for placeholder in placeholders:
                if placeholder in content:
                    # Direct content list
                    replacements[placeholder] = random.choice(content[placeholder])
                elif placeholder.endswith("_type") and placeholder[:-5] in content:
                    # Type of something, like "attack_type" -> select from "attacks"
                    base = placeholder[:-5]
                    replacements[placeholder] = random.choice(content[base + "s"]) if base + "s" in content else "unknown"
                elif placeholder == "language" and "languages" in content:
                    # Programming language
                    replacements[placeholder] = random.choice(content["languages"])
                elif "concepts" in content and placeholder in ["concept", "concept1", "concept2"]:
                    # Select different concepts if we need multiple
                    if "concept1" in placeholders and "concept2" in placeholders and placeholder == "concept1":
                        # For concept1, just pick one
                        replacements["concept1"] = random.choice(content["concepts"])
                    elif "concept1" in placeholders and "concept2" in placeholders and placeholder == "concept2":
                        # For concept2, ensure it's different from concept1
                        available = [c for c in content["concepts"] if c != replacements.get("concept1")]
                        replacements["concept2"] = random.choice(available) if available else "alternative approach"
                    else:
                        # Single concept
                        replacements[placeholder] = random.choice(content["concepts"])
                else:
                    # Default placeholder if no direct match
                    replacements[placeholder] = f"[{placeholder}]"
            
            # Fill question template
            question = question_template
            for placeholder, value in replacements.items():
                question = question.replace(f"{{{placeholder}}}", value)
            
            # Generate answer content
            answer = answer_template
            
            # Specific handling for answer placeholders
            for placeholder, value in replacements.items():
                if placeholder in answer:
                    answer = answer.replace(f"{{{placeholder}}}", value)
            
            # Handle special placeholder types in the answer
            if "{definition}" in answer and "definitions" in content and replacements.get("concept") in content["definitions"]:
                definition = content["definitions"][replacements["concept"]]
                answer = answer.replace("{definition}", definition)
            
            if "{code}" in answer and "code_examples" in content:
                lang = replacements.get("language", "Python")
                if lang in content["code_examples"]:
                    code_keys = list(content["code_examples"][lang].keys())
                    if code_keys:
                        algorithm = replacements.get("algorithm", random.choice(code_keys))
                        if algorithm in content["code_examples"][lang]:
                            code = content["code_examples"][lang][algorithm]
                        else:
                            code = random.choice(list(content["code_examples"][lang].values()))
                        answer = answer.replace("{code}", code)
            
            # Generic placeholders with reasonable defaults
            generic_replacements = {
                "{details}": "This is an important concept to understand in this context.",
                "{process}": "The process involves several steps that work together to achieve the intended outcome.",
                "{examples}": "several techniques that exploit vulnerabilities in different ways",
                "{steps}": "1. Assess your current security posture\n2. Identify vulnerabilities\n3. Implement security controls\n4. Test and validate\n5. Monitor continuously",
                "{measures}": "1. Use strong authentication\n2. Encrypt sensitive data\n3. Regular security updates\n4. Network segmentation\n5. Security awareness training",
                "{procedure}": "1. Containment: Isolate affected systems\n2. Analysis: Identify the scope and impact\n3. Eradication: Remove the threat\n4. Recovery: Restore systems securely\n5. Lessons Learned: Document improvements",
                "{response}": "1. Do not panic\n2. Document everything\n3. Notify appropriate stakeholders\n4. Follow your incident response plan\n5. Engage necessary resources",
                "{strategy}": "1. Identify the type and scope of attack\n2. Contain the incident to prevent further damage\n3. Analyze the attack vector\n4. Implement countermeasures\n5. Document lessons learned",
                "{differences}": "1. Purpose: Different use cases\n2. Implementation: Different approaches\n3. Performance: Different efficiency characteristics\n4. Security implications: Different risk profiles",
                "{explanation}": "This code implements the functionality efficiently while maintaining readability.",
                "{techniques}": "1. Use appropriate data structures\n2. Minimize unnecessary operations\n3. Utilize language-specific optimizations\n4. Consider time-space tradeoffs\n5. Profile code to identify bottlenecks",
                "{example_code}": "// Optimized version of the algorithm",
                "{best_practices}": "Follow these best practices for robust error handling:\n1. Be specific about which exceptions to catch\n2. Handle exceptions at the appropriate level\n3. Log helpful error information\n4. Clean up resources properly\n5. Provide meaningful user feedback",
                "{usage_notes}": "Remember to install the library and handle potential errors appropriately in production code.",
                "{method}": "utilizing specialized techniques that address specific security concerns",
                "{mechanism}": "employing specific algorithms designed for optimal performance",
                "{function}": "enables communication between networked systems",
                "{type}": "standardized",
                "{advantages}": "1. Improved performance\n2. Better security\n3. Lower operational costs\n4. Simplified management\n5. Greater scalability",
                "{troubleshooting_steps}": "1. Verify physical connectivity\n2. Check interface status\n3. Test basic connectivity\n4. Analyze error messages\n5. Review logs and configurations\n6. Isolate the problem domain"
            }
            
            for placeholder, value in generic_replacements.items():
                if placeholder in answer:
                    answer = answer.replace(placeholder, value)
            
            return {
                "question": question,
                "answer": answer,
                "domain": domain,
                "synthetic": True
            }
            
        except Exception as e:
            logger.error(f"Error filling template: {e}")
            return None
    
    def generate_all_domains(self, count_per_domain: int = 100):
        """
        Generate synthetic data for all available domains.
        
        Args:
            count_per_domain: Number of QA pairs to generate per domain
        """
        # Generate data for each domain
        domains = list(self.templates.keys())
        all_synthetic_data = []
        
        for domain in domains:
            logger.info(f"Generating synthetic data for domain: {domain}")
            domain_data = self.generate_synthetic_data(domain, count_per_domain)
            all_synthetic_data.extend(domain_data)
        
        # Save combined data
        combined_path = self.synthetic_dir / "combined_synthetic.json"
        with open(combined_path, 'w', encoding='utf-8') as f:
            json.dump(all_synthetic_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Generated {len(all_synthetic_data)} total synthetic QA pairs across {len(domains)} domains")
        logger.info(f"Saved combined synthetic data to {combined_path}")
        
        return all_synthetic_data

def main():
    """Main function to generate synthetic data."""
    # Get project root and datasets directory
    project_root = Path(__file__).resolve().parent.parent.parent
    datasets_dir = project_root / "Datasets"
    
    # Create synthetic data generator
    generator = SyntheticDataGenerator(datasets_dir)
    
    # Generate synthetic data for all domains
    generator.generate_all_domains(count_per_domain=200)

if __name__ == "__main__":
    main()
