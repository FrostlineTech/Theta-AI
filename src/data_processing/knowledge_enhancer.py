"""
Knowledge Base Enhancer for Theta AI

This module enhances the knowledge base with curated domain-specific datasets,
knowledge graphs, and improved retrieval capabilities.
"""

import os
import json
import logging
import re
from pathlib import Path
import random
from typing import Dict, List, Any, Optional, Set, Tuple
import string

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KnowledgeEnhancer:
    """Enhances Theta AI's knowledge base with domain-specific content."""
    
    def __init__(self, datasets_dir: Path):
        """Initialize the knowledge enhancer.
        
        Args:
            datasets_dir: Path to the datasets directory
        """
        self.datasets_dir = datasets_dir
        self.knowledge_graph = {}
        self.entity_map = {}
        self.relation_map = {}
        
        # Create knowledge directories if they don't exist
        self.knowledge_graphs_dir = datasets_dir / "knowledge_graphs"
        self.curated_qa_dir = datasets_dir / "curated_qa"
        self.domain_corpus_dir = datasets_dir / "domain_corpus"
        self.case_studies_dir = datasets_dir / "case_studies"
        
        # Create all required directories
        for directory in [self.knowledge_graphs_dir, self.curated_qa_dir, 
                          self.domain_corpus_dir, self.case_studies_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def generate_technical_qa_pairs(self, domain: str, num_pairs: int = 100) -> List[Dict]:
        """Generate curated QA pairs for specific technical domains.
        
        Args:
            domain: Technical domain (cybersecurity, programming, etc.)
            num_pairs: Number of pairs to generate
            
        Returns:
            List of QA pairs
        """
        qa_pairs = []
        
        # Domain-specific templates
        templates = {
            "cybersecurity": [
                {"question": "What is {concept} in cybersecurity?", 
                 "concepts": ["zero trust", "defense in depth", "threat modeling", "SIEM", 
                             "penetration testing", "vulnerability assessment", "red teaming",
                             "blue teaming", "purple teaming", "security orchestration"]},
                {"question": "How does {technique} work?", 
                 "concepts": ["port scanning", "packet sniffing", "SQL injection", "cross-site scripting", 
                             "buffer overflow", "credential stuffing", "brute force attack", 
                             "pass-the-hash", "lateral movement", "privilege escalation"]},
                {"question": "What are best practices for {security_area}?", 
                 "concepts": ["network security", "application security", "cloud security", 
                             "container security", "endpoint protection", "data loss prevention", 
                             "incident response", "security awareness training", "secure coding"]}
            ],
            "programming": [
                {"question": "How do I implement {algorithm} in {language}?", 
                 "algorithms": ["binary search", "quicksort", "merge sort", "depth-first search", 
                               "breadth-first search", "dynamic programming", "backtracking"],
                 "languages": ["Python", "JavaScript", "Java", "C++", "Go", "Rust"]},
                {"question": "What is {pattern} design pattern and when should I use it?", 
                 "patterns": ["Singleton", "Factory", "Observer", "Strategy", "Decorator", 
                             "Adapter", "Facade", "Command", "Iterator", "State"]},
                {"question": "How can I optimize {performance_aspect} in my code?", 
                 "aspects": ["memory usage", "execution speed", "database queries", 
                            "algorithm complexity", "API response times", "front-end rendering"]}
            ],
            "networking": [
                {"question": "What is the purpose of {protocol}?", 
                 "protocols": ["TCP", "UDP", "HTTPS", "DNS", "DHCP", "BGP", "OSPF", 
                              "ICMP", "FTP", "SSH", "SMTP", "SNMP", "NTP"]},
                {"question": "How does {network_concept} work?", 
                 "concepts": ["subnetting", "routing", "load balancing", "NAT", 
                             "VPN", "firewall", "proxy server", "CDN", "DNS resolution", "VLAN"]}
            ]
        }
        
        # Generate QA pairs for the specified domain
        if domain in templates:
            for _ in range(num_pairs):
                # Select a random template
                template_data = random.choice(templates[domain])
                template_question = template_data["question"]
                
                # Fill in the template with a random concept
                if domain == "programming" and "{algorithm}" in template_question:
                    algorithm = random.choice(template_data["algorithms"])
                    language = random.choice(template_data["languages"])
                    question = template_question.format(algorithm=algorithm, language=language)
                    answer = self._generate_answer(domain, question, algorithm, language)
                elif "{concept}" in template_question:
                    concept = random.choice(template_data["concepts"])
                    question = template_question.format(concept=concept)
                    answer = self._generate_answer(domain, question, concept)
                elif "{technique}" in template_question:
                    technique = random.choice(template_data["concepts"])
                    question = template_question.format(technique=technique)
                    answer = self._generate_answer(domain, question, technique)
                elif "{security_area}" in template_question:
                    area = random.choice(template_data["concepts"])
                    question = template_question.format(security_area=area)
                    answer = self._generate_answer(domain, question, area)
                elif "{pattern}" in template_question:
                    pattern = random.choice(template_data["patterns"])
                    question = template_question.format(pattern=pattern)
                    answer = self._generate_answer(domain, question, pattern)
                elif "{performance_aspect}" in template_question:
                    aspect = random.choice(template_data["aspects"])
                    question = template_question.format(performance_aspect=aspect)
                    answer = self._generate_answer(domain, question, aspect)
                elif "{protocol}" in template_question:
                    protocol = random.choice(template_data["protocols"])
                    question = template_question.format(protocol=protocol)
                    answer = self._generate_answer(domain, question, protocol)
                elif "{network_concept}" in template_question:
                    concept = random.choice(template_data["concepts"])
                    question = template_question.format(network_concept=concept)
                    answer = self._generate_answer(domain, question, concept)
                else:
                    continue
                
                # Add the QA pair
                qa_pairs.append({
                    "question": question,
                    "answer": answer,
                    "domain": domain
                })
        
        return qa_pairs
    
    def _generate_answer(self, domain: str, question: str, *args) -> str:
        """Generate an answer for a question (placeholder for better generation).
        In a real implementation, this would use a proper language model or templates.
        
        Args:
            domain: Technical domain
            question: The question to answer
            args: Additional arguments for answer generation
            
        Returns:
            Generated answer
        """
        # This is just a placeholder for demonstration
        # In production, you would use a better answer generation method
        # such as using a language model or retrieving from curated datasets
        
        answers = {
            "cybersecurity": {
                "zero trust": "Zero Trust is a security model based on the principle of 'never trust, always verify.' Unlike traditional security models that focus on perimeter defense, Zero Trust assumes breach and verifies each request as though it originates from an untrusted network. This approach requires strict identity verification for every person and device trying to access resources, regardless of whether they are within or outside the network perimeter.",
                "defense in depth": "Defense in depth is a cybersecurity strategy that employs multiple layers of security controls throughout an IT system. Rather than relying on a single security measure, it uses various mechanisms at different layers to protect assets. If one security control fails, others still provide protection. Components typically include firewalls, IDS/IPS, antivirus, access controls, encryption, and security awareness training.",
                "threat modeling": "Threat modeling is a structured process to identify potential security threats and vulnerabilities in a system. It involves identifying assets, threat actors, possible attacks, vulnerabilities, and countermeasures. Common methodologies include STRIDE, PASTA, and OCTAVE. Effective threat modeling helps organizations prioritize security efforts by focusing on realistic threats with the highest potential impact."
            },
            "programming": {
                "binary search": "Binary search is an efficient algorithm for finding a target value within a sorted array. It works by repeatedly dividing the search interval in half until the target is found or determined to be absent. The time complexity is O(log n), making it much faster than linear search for large datasets. To implement binary search in {1}, you would typically use iterative or recursive approaches that compare the middle element with the target and adjust the search space accordingly.",
                "quicksort": "Quicksort is a highly efficient sorting algorithm that uses a divide-and-conquer strategy. It works by selecting a 'pivot' element and partitioning the array around it, then recursively sorting the sub-arrays. While its average time complexity is O(n log n), the worst-case can be O(n²) with poor pivot choices. In {1}, quicksort implementation typically involves selecting a pivot (often the rightmost element), partitioning the array, and recursive calls for each partition."
            },
            "networking": {
                "TCP": "TCP (Transmission Control Protocol) is a connection-oriented protocol that provides reliable, ordered, and error-checked delivery of data between applications. It establishes a connection through a three-way handshake, manages data sequencing, handles retransmission of lost packets, and ensures flow control. TCP is used when data integrity is critical, such as for web browsing (HTTP/HTTPS), email (SMTP), file transfers (FTP), and remote access (SSH).",
                "UDP": "UDP (User Datagram Protocol) is a connectionless protocol that provides a simple, unreliable datagram service. Unlike TCP, it doesn't establish a connection before sending data, doesn't guarantee packet delivery or order, and provides no congestion or flow control mechanisms. UDP is used when speed is more important than reliability, such as in real-time applications like video streaming, VoIP, online gaming, and DNS lookups."
            }
        }
        
        # Try to find a specific answer for the concept
        for arg in args:
            if domain in answers and arg in answers[domain]:
                formatted_answer = answers[domain][arg]
                # Format any placeholders in the answer
                for i, format_arg in enumerate(args[1:], 1):
                    formatted_answer = formatted_answer.replace(f"{{{i}}}", format_arg)
                return formatted_answer
        
        # Generic response if no specific answer found
        return f"[This would contain detailed information about {', '.join(args)} in the {domain} domain, including key concepts, practical applications, and best practices.]"
    
    def create_knowledge_graph(self, domain: str) -> Dict:
        """Create a knowledge graph for a specific domain.
        
        Args:
            domain: Technical domain to create graph for
            
        Returns:
            Knowledge graph as a dictionary
        """
        graph = {
            "entities": {},
            "relations": [],
            "domain": domain
        }
        
        # Domain-specific entity types and relationships
        if domain == "cybersecurity":
            # Define entity types
            entity_types = {
                "threat": ["malware", "ransomware", "phishing", "SQL injection", "XSS", 
                          "CSRF", "DDoS", "brute force", "man-in-the-middle"],
                "defense": ["firewall", "IDS", "IPS", "SIEM", "antivirus", "encryption", 
                           "authentication", "authorization", "access control"],
                "framework": ["NIST CSF", "ISO 27001", "CIS Controls", "MITRE ATT&CK", 
                             "OWASP Top 10", "PCI DSS", "HIPAA", "GDPR"],
                "concept": ["zero trust", "defense in depth", "principle of least privilege", 
                           "security by design", "CIA triad", "threat modeling"]
            }
            
            # Define relationship types
            relation_types = {
                "mitigates": [("defense", "threat")],
                "implements": [("framework", "defense"), ("framework", "concept")],
                "related_to": [("concept", "concept"), ("defense", "defense"), 
                              ("threat", "threat"), ("framework", "framework")],
                "protects_against": [("concept", "threat"), ("defense", "threat")]
            }
        elif domain == "programming":
            # Define entity types
            entity_types = {
                "language": ["Python", "JavaScript", "Java", "C++", "Go", "Rust", 
                            "TypeScript", "PHP", "Ruby", "Swift"],
                "paradigm": ["object-oriented", "functional", "procedural", 
                            "event-driven", "reactive", "declarative"],
                "pattern": ["Singleton", "Factory", "Observer", "Strategy", 
                           "Decorator", "MVC", "MVVM", "Repository"],
                "concept": ["inheritance", "encapsulation", "polymorphism", 
                           "abstraction", "composition", "dependency injection"]
            }
            
            # Define relationship types
            relation_types = {
                "supports": [("language", "paradigm"), ("language", "pattern")],
                "implements": [("pattern", "concept")],
                "related_to": [("concept", "concept"), ("pattern", "pattern"), 
                              ("paradigm", "paradigm"), ("language", "language")]
            }
        elif domain == "networking":
            # Define entity types
            entity_types = {
                "protocol": ["TCP", "UDP", "HTTP", "HTTPS", "DNS", "DHCP", 
                            "SMTP", "FTP", "SSH", "TLS/SSL"],
                "device": ["router", "switch", "firewall", "load balancer", 
                          "proxy", "gateway", "modem", "access point"],
                "concept": ["subnet", "VLAN", "NAT", "routing", "switching", 
                           "VPN", "QoS", "tunneling", "packet filtering"]
            }
            
            # Define relationship types
            relation_types = {
                "operates_at": [("protocol", "layer")],
                "uses": [("device", "protocol"), ("concept", "protocol")],
                "implemented_by": [("concept", "device")],
                "related_to": [("protocol", "protocol"), ("device", "device"), 
                              ("concept", "concept")]
            }
            # Add OSI layers
            layers = ["physical", "data link", "network", "transport", 
                     "session", "presentation", "application"]
            entity_types["layer"] = layers
        else:
            # Default generic entities and relations
            entity_types = {
                "concept": ["concept1", "concept2", "concept3"],
                "tool": ["tool1", "tool2", "tool3"],
                "technique": ["technique1", "technique2", "technique3"]
            }
            relation_types = {
                "related_to": [("concept", "concept"), ("tool", "tool"), 
                              ("technique", "technique")],
                "implements": [("tool", "concept"), ("technique", "concept")]
            }
        
        # Create entities
        for entity_type, entities in entity_types.items():
            for entity in entities:
                entity_id = f"{entity_type}_{self._normalize_string(entity)}"
                graph["entities"][entity_id] = {
                    "id": entity_id,
                    "name": entity,
                    "type": entity_type
                }
        
        # Create relations
        entity_ids = list(graph["entities"].keys())
        for relation_type, relation_pairs in relation_types.items():
            for source_type, target_type in relation_pairs:
                # Find all entities of source and target types
                source_entities = [e for e in entity_ids 
                                  if graph["entities"][e]["type"] == source_type]
                target_entities = [e for e in entity_ids 
                                  if graph["entities"][e]["type"] == target_type]
                
                # Create some relations between them
                for _ in range(min(20, len(source_entities) * len(target_entities))):
                    source = random.choice(source_entities)
                    target = random.choice(target_entities)
                    
                    # Skip self-relations
                    if source == target:
                        continue
                    
                    # Create relation
                    relation = {
                        "source": source,
                        "target": target,
                        "type": relation_type,
                        "weight": random.uniform(0.1, 1.0)
                    }
                    graph["relations"].append(relation)
        
        # Save the knowledge graph
        filename = f"{domain}_knowledge_graph.json"
        filepath = self.knowledge_graphs_dir / filename
        with open(filepath, 'w') as f:
            json.dump(graph, f, indent=2)
        
        logger.info(f"Created knowledge graph for {domain} with "
                   f"{len(graph['entities'])} entities and {len(graph['relations'])} relations")
        
        return graph
    
    def create_case_study(self, domain: str, complexity: str = "medium") -> Dict:
        """Create a case study for a specific domain.
        
        Args:
            domain: Technical domain for the case study
            complexity: Complexity level (simple, medium, complex)
            
        Returns:
            Case study as a dictionary
        """
        case_study = {
            "domain": domain,
            "complexity": complexity,
            "steps": [],
            "qa_pairs": []
        }
        
        # Domain-specific case study templates
        templates = {
            "cybersecurity": {
                "titles": [
                    "Investigating a Security Breach at {company}",
                    "Implementing Zero Trust Architecture for {company}",
                    "Responding to a Ransomware Attack at {company}",
                    "Building a SOC for {company}"
                ],
                "companies": [
                    "Acme Financial", "TechSolutions Inc.", "GlobalHealth Systems",
                    "RetailGiant", "IndustrialTech Manufacturing"
                ],
                "scenarios": [
                    "detected suspicious network traffic from multiple endpoints",
                    "found evidence of data exfiltration in their cloud storage logs",
                    "received a ransomware demand after critical systems were encrypted",
                    "discovered unauthorized access to their customer database"
                ]
            },
            "programming": {
                "titles": [
                    "Refactoring the Legacy Codebase at {company}",
                    "Implementing Microservices Architecture for {company}",
                    "Optimizing Database Performance at {company}",
                    "Building a CI/CD Pipeline for {company}"
                ],
                "companies": [
                    "WebScale Tech", "DataFlow Systems", "AppNexus Solutions",
                    "CodeCraft Software", "CloudNative Applications"
                ],
                "scenarios": [
                    "experienced performance issues with their monolithic application",
                    "needed to scale their system to handle increasing user load",
                    "struggled with long deployment cycles and frequent integration issues",
                    "faced challenges with database queries timing out during peak hours"
                ]
            },
            "networking": {
                "titles": [
                    "Network Redesign for {company}",
                    "Implementing SD-WAN at {company}",
                    "Cloud Migration Strategy for {company}",
                    "Zero Trust Network Implementation at {company}"
                ],
                "companies": [
                    "GlobalConnect", "NetworkSolutions", "DistributedSystems Inc.",
                    "MultiRegional Enterprises", "TechInfra Solutions"
                ],
                "scenarios": [
                    "experienced frequent network outages affecting business operations",
                    "needed to connect multiple branch offices securely and efficiently",
                    "required a hybrid cloud networking solution for their distributed workforce",
                    "faced security challenges with their traditional perimeter-based security model"
                ]
            },
            "cloud_computing": {
                "titles": [
                    "Cloud Migration Strategy for {company}",
                    "Implementing Multi-Cloud Architecture at {company}",
                    "Optimizing Cloud Costs for {company}",
                    "Cloud Security Implementation for {company}"
                ],
                "companies": [
                    "CloudTech Solutions", "DataCenter Innovations", "ServerLess Inc.",
                    "VirtualScale Systems", "ContainerOps Technologies"
                ],
                "scenarios": [
                    "needed to migrate legacy applications to the cloud",
                    "wanted to optimize their cloud spending across multiple providers",
                    "required improved security and compliance in their cloud environment",
                    "faced scalability challenges with their growing cloud infrastructure"
                ]
            },
            "data_science": {
                "titles": [
                    "Building a Predictive Analytics System for {company}",
                    "Implementing Machine Learning Models at {company}",
                    "Data Pipeline Optimization for {company}",
                    "Big Data Architecture for {company}"
                ],
                "companies": [
                    "DataInsights Inc.", "PredictiveTech", "AnalyticsSmart",
                    "MLEngineering", "BigDataSolutions"
                ],
                "scenarios": [
                    "needed to predict customer churn accurately",
                    "wanted to analyze large datasets for business insights",
                    "required automated data processing pipelines",
                    "faced challenges with data quality and integration"
                ]
            },
            "general_tech": {
                "titles": [
                    "Digital Transformation at {company}",
                    "Technology Modernization for {company}",
                    "IT Infrastructure Upgrade at {company}",
                    "Technology Stack Optimization at {company}"
                ],
                "companies": [
                    "Global Enterprises", "Tech Innovators LLC", "Digital Solutions Co.",
                    "Modern Systems Inc.", "Enterprise Technology Group"
                ],
                "scenarios": [
                    "needed to modernize their legacy systems",
                    "wanted to improve operational efficiency through technology",
                    "required better integration between different systems",
                    "faced challenges with outdated technology stack"
                ]
            }
        }
        
        # Select template based on domain
        if domain in templates:
            domain_templates = templates[domain]
            title_template = random.choice(domain_templates["titles"])
            company = random.choice(domain_templates["companies"])
            scenario = random.choice(domain_templates["scenarios"])
            
            # Set case study title and background
            case_study["title"] = title_template.format(company=company)
            case_study["background"] = f"{company} {scenario}. This case study explores how they addressed this challenge."
            
            # Generate steps for the case study
            num_steps = {"simple": 3, "medium": 5, "complex": 8}[complexity]
            
            # Different steps for different domains
            steps = []
            if "Security Breach" in case_study["title"] or "Ransomware" in case_study["title"]:
                steps = [
                    "Initial Detection and Containment",
                    "Forensic Investigation",
                    "Threat Identification",
                    "Impact Assessment",
                    "Remediation Plan Development",
                    "System Restoration",
                    "Post-Incident Analysis",
                    "Security Posture Improvement"
                ]
            elif "Zero Trust" in case_study["title"]:
                steps = [
                    "Current Architecture Assessment",
                    "Identity and Access Management Review",
                    "Network Segmentation Planning",
                    "MFA Implementation",
                    "Least Privilege Enforcement",
                    "Continuous Monitoring Setup",
                    "Policy Enforcement Points Deployment",
                    "Security Validation Testing"
                ]
            elif "Refactoring" in case_study["title"] or "Microservices" in case_study["title"]:
                steps = [
                    "Code Analysis and Dependency Mapping",
                    "Architecture Design",
                    "Service Boundary Definition",
                    "Data Model Redesign",
                    "API Gateway Implementation",
                    "Service Implementation",
                    "Testing Strategy Development",
                    "Gradual Migration and Deployment"
                ]
            elif "Network Redesign" in case_study["title"] or "SD-WAN" in case_study["title"]:
                steps = [
                    "Current Network Assessment",
                    "Requirements Gathering",
                    "Architecture Design",
                    "Vendor Selection",
                    "Implementation Planning",
                    "Pilot Deployment",
                    "Full-Scale Rollout",
                    "Performance Monitoring and Optimization"
                ]
            else:
                # Generic steps if no specific template matched
                steps = [
                    "Problem Analysis",
                    "Solution Design",
                    "Implementation Planning",
                    "Development Phase",
                    "Testing and Validation",
                    "Deployment Strategy",
                    "Post-Implementation Review",
                    "Continuous Improvement"
                ]
            
            # Select and populate steps based on complexity
            selected_steps = steps[:num_steps]
            for i, step in enumerate(selected_steps, 1):
                step_content = f"Step {i}: {step}\n\n"
                step_content += f"[This section would contain detailed information about how {company} implemented this step, "
                step_content += "including specific technologies used, challenges encountered, and outcomes achieved.]"
                
                case_study["steps"].append({
                    "title": step,
                    "content": step_content,
                    "order": i
                })
            
            # Generate QA pairs related to the case study
            qa_templates = [
                {"question": f"What were the main challenges {company} faced in their {domain} implementation?"},
                {"question": f"How did {company} approach the {selected_steps[0].lower()} phase?"},
                {"question": f"What technologies did {company} use for their {domain} solution?"},
                {"question": f"What were the key outcomes of {company}'s {domain} project?"},
                {"question": f"What lessons can be learned from {company}'s approach to {domain}?"}
            ]
            
            # Select some QA pairs based on complexity
            num_qa_pairs = {"simple": 2, "medium": 3, "complex": 5}[complexity]
            selected_qa_templates = random.sample(qa_templates, min(num_qa_pairs, len(qa_templates)))
            
            for qa_template in selected_qa_templates:
                question = qa_template["question"]
                answer = f"[This would be a detailed answer about {company}'s experience, referencing specific steps in the case study "
                answer += "and providing insights relevant to the question asked.]"
                
                case_study["qa_pairs"].append({
                    "question": question,
                    "answer": answer
                })
        
        # Save the case study
        filename = f"{domain}_{self._normalize_string(case_study['title'])}.json"
        filepath = self.case_studies_dir / filename
        with open(filepath, 'w') as f:
            json.dump(case_study, f, indent=2)
        
        logger.info(f"Created case study: {case_study['title']}")
        
        return case_study
    
    def _normalize_string(self, s: str) -> str:
        """Normalize a string for use in filenames and IDs."""
        # Remove punctuation, convert to lowercase, replace spaces with underscores
        translator = str.maketrans('', '', string.punctuation)
        return s.translate(translator).lower().replace(' ', '_')
    
    def create_all_resources(self, domains: List[str] = None):
        """Create all knowledge resources for the specified domains.
        
        Args:
            domains: List of domains to create resources for. If None, uses default domains.
        """
        if domains is None:
            domains = ["cybersecurity", "programming", "networking", "cloud_computing", "data_science"]
        
        for domain in domains:
            logger.info(f"Creating resources for domain: {domain}")
            
            # Create knowledge graph
            self.create_knowledge_graph(domain)
            
            # Create QA pairs
            qa_pairs = self.generate_technical_qa_pairs(domain, num_pairs=100)
            qa_filename = f"{domain}_qa.json"
            qa_filepath = self.curated_qa_dir / qa_filename
            with open(qa_filepath, 'w') as f:
                json.dump(qa_pairs, f, indent=2)
            
            # Create case studies
            complexities = ["simple", "medium", "complex"]
            for complexity in complexities:
                self.create_case_study(domain, complexity)
        
        logger.info("Finished creating all knowledge resources")

def main():
    """Main function to create knowledge resources."""
    # Get project root and datasets directory
    project_root = Path(__file__).resolve().parent.parent.parent
    datasets_dir = project_root / "Datasets"
    
    # Create knowledge enhancer
    enhancer = KnowledgeEnhancer(datasets_dir)
    
    # Create all resources
    domains = ["cybersecurity", "programming", "networking", "cloud_computing", "data_science"]
    enhancer.create_all_resources(domains)

if __name__ == "__main__":
    main()
