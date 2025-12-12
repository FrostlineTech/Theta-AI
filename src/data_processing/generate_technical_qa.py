"""
Generate technical Q&A pairs for Theta AI training.
Creates detailed question and answer pairs across various technical domains.
"""

import json
import random
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_programming_qa(count=200):
    """Generate programming-related Q&A pairs"""
    languages = ["Python", "JavaScript", "Java", "C#", "C++", "Go", "Rust", "Ruby", "PHP", "TypeScript"]
    concepts = ["variables", "functions", "classes", "inheritance", "error handling", "memory management", 
               "concurrency", "file I/O", "data structures", "algorithms", "design patterns", "debugging",
               "performance optimization", "testing", "package management", "frameworks", "libraries"]
    
    qa_pairs = []
    
    for _ in range(count):
        lang = random.choice(languages)
        concept = random.choice(concepts)
        
        # Vary question formats
        question_templates = [
            f"How does {concept} work in {lang}?",
            f"What's the best way to implement {concept} in {lang}?",
            f"Can you explain {lang}'s approach to {concept}?",
            f"What are common mistakes when working with {concept} in {lang}?",
            f"How do {concept}s in {lang} compare to other languages?"
        ]
        
        question = random.choice(question_templates)
        
        # Create detailed answer that looks authentic
        answer = f"In {lang}, {concept} "
        
        if concept == "variables":
            answer += f"are {'statically typed' if lang in ['Java', 'C#', 'C++', 'TypeScript', 'Rust'] else 'dynamically typed'}. "
            answer += f"You declare them using {'specific type annotations' if lang in ['Java', 'C#', 'C++', 'TypeScript', 'Rust'] else 'keywords like var, let, or const' if lang == 'JavaScript' else 'just the variable name'}. "
            answer += f"Variable scoping in {lang} follows {'block scope' if lang in ['JavaScript', 'Java', 'C#', 'C++', 'TypeScript', 'Rust'] else 'function scope for var in older code' if lang == 'JavaScript' else 'indentation-based scope' if lang == 'Python' else 'typical programming language rules'}. "
            answer += f"A unique aspect of variables in {lang} is {'type inference' if lang in ['TypeScript', 'C#', 'Rust'] else 'duck typing' if lang in ['Python', 'JavaScript'] else 'strong type checking' if lang in ['Java', 'C++'] else 'their flexibility'}."
        
        elif concept == "functions":
            answer += f"are {'first-class citizens' if lang in ['JavaScript', 'Python', 'Ruby', 'Go'] else 'defined with specific return types' if lang in ['Java', 'C#', 'C++', 'TypeScript', 'Rust'] else 'core building blocks'}. "
            answer += f"You define them using {'the function keyword' if lang in ['JavaScript', 'PHP'] else 'def' if lang == 'Python' else 'func' if lang == 'Go' else 'fn' if lang == 'Rust' else 'specific syntax including return types'} and can include parameters and return values. "
            answer += f"{lang} supports {'anonymous functions or lambdas' if lang not in ['PHP'] else 'limited anonymous functions'} which are useful for callbacks and functional programming approaches. "
            answer += f"Function overloading {'is supported' if lang in ['Java', 'C#', 'C++', 'TypeScript'] else 'is not directly supported, but can be simulated with default parameters or variable arguments' if lang in ['Python', 'JavaScript'] else 'has limited support'}."
        
        # Add a helpful closing
        answer += f"\n\nWhen working with {concept} in {lang}, I recommend following the language's conventions and best practices. Would you like me to provide a specific example of {concept} in {lang}?"
        
        qa_pairs.append({
            "question": question,
            "answer": answer
        })
    
    return qa_pairs

def generate_sysadmin_qa(count=200):
    """Generate system administration related Q&A pairs"""
    os_list = ["Linux", "Windows Server", "macOS", "Ubuntu", "CentOS", "Debian", "Windows 10", "Windows 11"]
    topics = ["file permissions", "user management", "process management", "system monitoring", 
              "networking", "firewalls", "package management", "system updates", "backups", 
              "log management", "shell scripting", "automation", "security hardening", "virtualization"]
    
    qa_pairs = []
    
    for _ in range(count):
        os = random.choice(os_list)
        topic = random.choice(topics)
        
        question_templates = [
            f"How do I manage {topic} in {os}?",
            f"What are best practices for {topic} on {os} systems?",
            f"What commands are used for {topic} in {os}?",
            f"How can I troubleshoot {topic} issues on {os}?",
            f"Can you explain {os} {topic} configuration options?"
        ]
        
        question = random.choice(question_templates)
        
        # Create detailed answer
        answer = f"Managing {topic} in {os} involves "
        
        if topic == "file permissions":
            if "Linux" in os or "Ubuntu" in os or "CentOS" in os or "Debian" in os:
                answer += "using the chmod, chown, and chgrp commands. The Linux permission model uses read (r), write (w), and execute (x) permissions for user, group, and others. "
                answer += "For example, to give the owner full permissions and others read access, you would use: chmod 744 filename. "
                answer += "For more complex permission management, ACLs (Access Control Lists) can be implemented using setfacl and getfacl commands."
            else:
                answer += "using the Security tab in file properties or icacls command in the command line. Windows uses a more complex permission system with Allow and Deny rules. "
                answer += "You can set permissions for specific users or groups, and Windows offers fine-grained controls like Read, Write, Modify, Full Control, etc. "
                answer += "PowerShell offers advanced permission management with commands like Get-Acl and Set-Acl."
        
        elif topic == "user management":
            if "Linux" in os or "Ubuntu" in os or "CentOS" in os or "Debian" in os:
                answer += "commands like useradd, usermod, and userdel. Creating a new user is as simple as 'sudo useradd username', and you can set a password with 'sudo passwd username'. "
                answer += "Groups are managed with groupadd, groupmod, and groupdel commands. User information is stored in /etc/passwd, while passwords are in /etc/shadow. "
                answer += "To give administrative privileges, you typically add a user to the sudo group with 'sudo usermod -aG sudo username'."
            else:
                answer += "the User Management Console or PowerShell. You can create users with the 'New-LocalUser' cmdlet or through the GUI. "
                answer += "User groups help organize permissions efficiently, and you can add users to groups with 'Add-LocalGroupMember'. "
                answer += "For domain environments, Active Directory provides centralized user management with more advanced features."
        
        # Add a helpful closing
        answer += f"\n\nFor {os} systems, it's important to follow security best practices when dealing with {topic}. Would you like specific examples of {topic} configurations for {os}?"
        
        qa_pairs.append({
            "question": question,
            "answer": answer
        })
    
    return qa_pairs

def generate_networking_qa(count=200):
    """Generate networking related Q&A pairs"""
    topics = ["TCP/IP", "DNS", "DHCP", "routing", "switching", "VLANs", "VPNs", "firewalls", 
              "load balancing", "NAT", "IPv6", "subnetting", "network troubleshooting", 
              "Wi-Fi", "Bluetooth", "network security", "SDN", "cloud networking"]
    
    qa_pairs = []
    
    for _ in range(count):
        topic = random.choice(topics)
        
        question_templates = [
            f"How does {topic} work?",
            f"What are the key components of {topic}?",
            f"Can you explain {topic} configuration best practices?",
            f"What tools are used for {topic} management?",
            f"How do I troubleshoot {topic} issues?",
            f"What are common {topic} vulnerabilities and how do I mitigate them?"
        ]
        
        question = random.choice(question_templates)
        
        # Create detailed answer
        answer = f"{topic} is a critical networking concept that "
        
        if topic == "TCP/IP":
            answer += "forms the foundation of internet communications. It's a suite of protocols organized in four layers: Link Layer, Internet Layer, Transport Layer, and Application Layer. "
            answer += "The Internet Layer includes IP (Internet Protocol) which handles addressing and routing packets to their destination across networks. "
            answer += "The Transport Layer includes TCP (Transmission Control Protocol) which ensures reliable, ordered delivery of data, and UDP (User Datagram Protocol) which provides faster but unreliable transmission. "
            answer += "TCP establishes connections using a three-way handshake (SYN, SYN-ACK, ACK) and manages congestion control, flow control, and retransmission of lost packets."
        
        elif topic == "subnetting":
            answer += "involves dividing a larger network into smaller, more manageable segments. This improves security, reduces network traffic, and makes address management easier. "
            answer += "Subnetting works by borrowing bits from the host portion of an IP address to create a subnet mask. For example, a Class C network (255.255.255.0 or /24) can be subnetted into smaller networks like /25, /26, etc. "
            answer += "To calculate subnets, you need to understand CIDR notation and binary math. For instance, a /24 network has 256 addresses (2^8), while a /25 network has 128 addresses (2^7). "
            answer += "Common subnet masks include 255.255.255.0 (/24) with 254 usable hosts, 255.255.255.128 (/25) with 126 usable hosts, and 255.255.255.192 (/26) with 62 usable hosts."
        
        # Add a helpful closing
        answer += f"\n\nUnderstanding {topic} is essential for effective network design and troubleshooting. Would you like me to explain any specific aspect of {topic} in more detail?"
        
        qa_pairs.append({
            "question": question,
            "answer": answer
        })
    
    return qa_pairs

def generate_cybersecurity_qa(count=200):
    """Generate cybersecurity related Q&A pairs"""
    topics = ["encryption", "authentication", "authorization", "network security", "application security", 
              "penetration testing", "vulnerability assessment", "incident response", "security frameworks", 
              "compliance", "risk management", "threat modeling", "social engineering", "phishing", 
              "malware analysis", "forensics", "SIEM", "zero trust", "PKI", "MFA"]
    
    qa_pairs = []
    
    for _ in range(count):
        topic = random.choice(topics)
        
        question_templates = [
            f"How does {topic} work?",
            f"What are best practices for implementing {topic}?",
            f"How can I improve {topic} in my organization?",
            f"What tools are used for {topic}?",
            f"What are common {topic} vulnerabilities?",
            f"How has {topic} evolved in recent years?"
        ]
        
        question = random.choice(question_templates)
        
        # Create detailed answer
        answer = f"{topic} is a critical cybersecurity concept that "
        
        if topic == "encryption":
            answer += "protects data confidentiality by converting readable data (plaintext) into an encoded format (ciphertext) that can only be read or processed after it's been decrypted with a key. "
            answer += "There are two main types: symmetric encryption uses the same key for encryption and decryption (e.g., AES, 3DES), while asymmetric encryption uses a public-private key pair (e.g., RSA, ECC). "
            answer += "Modern systems often use hybrid approaches, where asymmetric encryption securely exchanges a symmetric key, which then encrypts the actual data for better performance. "
            answer += "Encryption strength depends on key length, algorithm security, and proper implementation. Current recommendations include using AES-256 for symmetric encryption and RSA with at least 2048-bit keys for asymmetric encryption."
        
        elif topic == "MFA":
            answer += "stands for Multi-Factor Authentication and significantly enhances security by requiring multiple forms of verification. "
            answer += "It combines at least two of the following factors: something you know (password), something you have (phone/token), and something you are (biometrics). "
            answer += "Implementation options include SMS codes (less secure), authenticator apps like Google Authenticator or Microsoft Authenticator, hardware tokens like YubiKeys, and biometric verification. "
            answer += "MFA can block over 99.9% of automated attacks, making it one of the most effective security controls available. However, proper implementation is crucial to avoid user experience issues."
        
        # Add a helpful closing
        answer += f"\n\nImplementing strong {topic} measures requires ongoing attention to evolving threats and best practices. Would you like more specific guidance on implementing {topic} in your environment?"
        
        qa_pairs.append({
            "question": question,
            "answer": answer
        })
    
    return qa_pairs

def generate_cloud_qa(count=200):
    """Generate cloud computing related Q&A pairs"""
    providers = ["AWS", "Azure", "Google Cloud", "Oracle Cloud", "IBM Cloud"]
    topics = ["virtual machines", "containers", "serverless", "storage", "networking", "databases", 
             "identity management", "security", "cost optimization", "migration", "hybrid cloud", 
             "multi-cloud", "monitoring", "automation", "DevOps", "IaC", "microservices"]
    
    qa_pairs = []
    
    for _ in range(count):
        provider = random.choice(providers)
        topic = random.choice(topics)
        
        question_templates = [
            f"How do I set up {topic} in {provider}?",
            f"What are best practices for {topic} on {provider}?",
            f"How does {provider} implement {topic}?",
            f"What are the costs associated with {topic} on {provider}?",
            f"How do I troubleshoot {topic} issues in {provider}?",
            f"How does {provider}'s approach to {topic} compare to other cloud providers?"
        ]
        
        question = random.choice(question_templates)
        
        # Create detailed answer
        answer = f"{topic} in {provider} "
        
        if topic == "serverless" and provider == "AWS":
            answer += "primarily refers to AWS Lambda, which lets you run code without provisioning or managing servers. "
            answer += "To set up a Lambda function, you upload your code, configure triggers (like API Gateway, S3 events, or CloudWatch schedules), and specify runtime parameters. "
            answer += "Lambda functions are priced based on the number of requests, execution duration, and allocated memory. The free tier includes 1M requests and 400,000 GB-seconds per month. "
            answer += "Best practices include keeping functions small and focused, optimizing cold start times, implementing proper error handling, and using environment variables for configuration."
        
        elif topic == "containers" and provider == "Azure":
            answer += "offers several options, with Azure Kubernetes Service (AKS) and Azure Container Instances (ACI) being the most popular. "
            answer += "AKS provides a managed Kubernetes service that simplifies deployment and operations of containerized applications, while ACI offers a serverless container runtime for quick deployments without cluster management. "
            answer += "To get started, you can use the Azure CLI or Portal to create a container registry (ACR) to store your images, then deploy to AKS or ACI. "
            answer += "For production workloads, consider implementing proper monitoring with Azure Monitor, setting up CI/CD pipelines with Azure DevOps, and using Helm charts for complex deployments."
        
        # Add a helpful closing
        answer += f"\n\nWhen working with {topic} in {provider}, it's important to follow cloud best practices like implementing proper IAM, monitoring costs, and designing for resilience. Would you like more specific guidance on any aspect of {topic} in {provider}?"
        
        qa_pairs.append({
            "question": question,
            "answer": answer
        })
    
    return qa_pairs

def generate_database_qa(count=200):
    """Generate database related Q&A pairs"""
    db_types = ["SQL", "NoSQL", "MySQL", "PostgreSQL", "SQL Server", "Oracle", "MongoDB", 
               "Cassandra", "Redis", "DynamoDB", "SQLite", "Firebase", "Neo4j", "Elasticsearch"]
    topics = ["schema design", "query optimization", "indexing", "transactions", "replication", 
             "sharding", "backup and recovery", "security", "migration", "administration", 
             "performance tuning", "scaling", "normalization", "data modeling"]
    
    qa_pairs = []
    
    for _ in range(count):
        db = random.choice(db_types)
        topic = random.choice(topics)
        
        question_templates = [
            f"How does {topic} work in {db}?",
            f"What are best practices for {topic} in {db}?",
            f"How can I optimize {db} for better {topic}?",
            f"What are common mistakes when implementing {topic} in {db}?",
            f"How do I implement {topic} in {db}?"
        ]
        
        question = random.choice(question_templates)
        
        # Create detailed answer
        answer = f"{topic} in {db} "
        
        if topic == "indexing" and db == "MySQL":
            answer += "is a critical technique for improving query performance. MySQL supports several index types including B-tree (default), Hash, Full-text, and Spatial indexes. "
            answer += "To create an index, use 'CREATE INDEX idx_name ON table_name (column1, column2);' for a regular index or 'CREATE UNIQUE INDEX' for unique constraints. "
            answer += "Best practices include indexing columns used in WHERE, JOIN, and ORDER BY clauses, putting more selective columns first in composite indexes, and avoiding over-indexing as it slows down writes. "
            answer += "You can analyze index usage with EXPLAIN to see if your queries are using indexes effectively. Common mistakes include indexing low-cardinality columns and not considering the impact on write operations."
        
        elif topic == "sharding" and db == "MongoDB":
            answer += "is a method for horizontally partitioning data across multiple servers or clusters. MongoDB implements sharding through its sharded cluster architecture, which consists of shard servers, config servers, and mongos routers. "
            answer += "To implement sharding, you need to choose a shard key carefully based on your access patterns. Good shard keys distribute writes evenly and allow targeted reads to minimize scatter-gather operations. "
            answer += "Setting up sharding involves enabling sharding on a database ('sh.enableSharding(\"dbName\")'), then on collections ('sh.shardCollection(\"db.collection\", {shardKey: 1})'). "
            answer += "Common challenges include dealing with jumbo chunks, choosing suboptimal shard keys, and managing migrations between shards."
        
        # Add a helpful closing
        answer += f"\n\nWhen working with {topic} in {db}, it's important to test thoroughly and monitor performance metrics. Would you like more specific information about implementing {topic} in your {db} environment?"
        
        qa_pairs.append({
            "question": question,
            "answer": answer
        })
    
    return qa_pairs

def generate_technical_qa(count=2000):
    """
    Generate technical Q&A pairs with detailed answers.
    
    Args:
        count: Total number of Q&A pairs to generate
        
    Returns:
        list: Generated Q&A pairs
    """
    # Distribute count across different categories
    category_count = count // 5  # 5 categories
    
    # Generate QA pairs for each category
    programming_qa = generate_programming_qa(category_count)
    sysadmin_qa = generate_sysadmin_qa(category_count)
    networking_qa = generate_networking_qa(category_count)
    cybersecurity_qa = generate_cybersecurity_qa(category_count)
    cloud_qa = generate_cloud_qa(category_count)
    database_qa = generate_database_qa(count - (category_count * 5))  # Use remaining count
    
    # Combine all QA pairs
    all_qa_pairs = (
        programming_qa + sysadmin_qa + networking_qa + 
        cybersecurity_qa + cloud_qa + database_qa
    )
    
    # Shuffle the combined list
    random.shuffle(all_qa_pairs)
    
    return all_qa_pairs

def save_technical_qa(count=2000):
    """
    Generate and save technical Q&A pairs.
    
    Args:
        count: Number of Q&A pairs to generate
        
    Returns:
        str: Path to the saved dataset
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    datasets_dir = project_root / "Datasets"
    datasets_dir.mkdir(exist_ok=True)
    
    # Generate QA pairs
    logger.info(f"Generating {count} technical Q&A pairs...")
    qa_pairs = generate_technical_qa(count)
    
    # Save QA pairs
    qa_path = datasets_dir / "expanded_technical_qa.json"
    with open(qa_path, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Created {len(qa_pairs)} technical Q&A pairs at {qa_path}")
    return str(qa_path)

def main():
    """Main function to generate and save technical Q&A pairs."""
    return save_technical_qa(2000)

if __name__ == "__main__":
    main()
