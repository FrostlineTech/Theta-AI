"""
Technical Documentation Corpus Processor for Theta AI.

This module processes technical documentation from various sources 
and converts it to a Q&A format for training Theta AI.
"""

import os
import re
import json
import logging
import random
import requests
from pathlib import Path
from bs4 import BeautifulSoup
from tqdm import tqdm
import markdown

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Documentation sources
DOCS_SOURCES = {
    "aws": {
        "url": "https://docs.aws.amazon.com/",
        "services": ["ec2", "s3", "lambda", "rds", "cloudformation"]
    },
    "azure": {
        "url": "https://docs.microsoft.com/en-us/azure/",
        "services": ["virtual-machines", "storage", "functions", "sql-database", "app-service"]
    },
    "linux": {
        "url": "https://man7.org/linux/man-pages/",
        "commands": ["ls", "grep", "awk", "sed", "find", "systemctl", "docker", "git"]
    }
}

class TechnicalDocumentationProcessor:
    """
    Processes technical documentation from various sources.
    """
    
    def __init__(self, output_dir, cache_dir=None, sample_size=5000):
        """
        Initialize the processor.
        
        Args:
            output_dir: Directory to save processed data
            cache_dir: Directory to cache downloaded files
            sample_size: Number of Q&A pairs to generate
        """
        self.output_dir = Path(output_dir) if output_dir else Path(".")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = self.output_dir / "cache" / "docs"
            
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.sample_size = sample_size
        
    def fetch_documentation(self, source, item):
        """
        Fetch documentation for a specific source and item.
        
        Args:
            source (str): Source name
            item (str): Item to fetch (service or command)
            
        Returns:
            str: Raw documentation content
        """
        # This is a simplified implementation
        # In a real-world scenario, you would implement proper web scraping
        # with rate limiting, HTML parsing, etc.
        
        try:
            # Construct URL
            if source == "aws":
                url = f"{DOCS_SOURCES['aws']['url']}{item}/latest/userguide/what-is.html"
            elif source == "azure":
                url = f"{DOCS_SOURCES['azure']['url']}{item}/overview"
            else:  # linux
                url = f"{DOCS_SOURCES['linux']['url']}man1/{item}.1.html"
                
            # Check cache first
            cache_file = self.cache_dir / f"{source}_{item}.html"
            if cache_file.exists():
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return f.read()
            
            # Fetch content
            logger.info(f"Fetching documentation for {source}/{item}")
            response = requests.get(url)
            response.raise_for_status()
            content = response.text
            
            # Save to cache
            with open(cache_file, 'w', encoding='utf-8') as f:
                f.write(content)
                
            return content
            
        except Exception as e:
            logger.error(f"Error fetching documentation for {source}/{item}: {str(e)}")
            return None
    
    def parse_documentation(self, content, source, item):
        """
        Parse documentation content.
        
        Args:
            content (str): Raw documentation content
            source (str): Source name
            item (str): Item (service or command)
            
        Returns:
            dict: Parsed documentation with sections
        """
        # This is a simplified implementation
        # In a real-world scenario, you would use proper HTML parsing
        
        try:
            # Parse HTML
            soup = BeautifulSoup(content, 'html.parser')
            
            # Extract title and main content
            title = soup.title.string if soup.title else f"{source.upper()} {item} Documentation"
            
            # Extract sections (simplified)
            sections = []
            
            for heading in soup.find_all(['h1', 'h2', 'h3']):
                section_title = heading.get_text().strip()
                section_content = ""
                
                # Get content until next heading
                sibling = heading.next_sibling
                while sibling and sibling.name not in ['h1', 'h2', 'h3']:
                    if hasattr(sibling, 'get_text'):
                        section_content += sibling.get_text() + "\n"
                    sibling = sibling.next_sibling
                
                sections.append({
                    "title": section_title,
                    "content": section_content.strip()
                })
            
            return {
                "title": title,
                "source": source,
                "item": item,
                "sections": sections
            }
            
        except Exception as e:
            logger.error(f"Error parsing documentation for {source}/{item}: {str(e)}")
            return None
    
    def convert_to_qa(self, parsed_docs):
        """
        Convert parsed documentation to Q&A format.
        
        Args:
            parsed_docs (list): List of parsed documentation
            
        Returns:
            list: Q&A pairs
        """
        qa_pairs = []
        
        # Process each documentation
        for doc in parsed_docs:
            source = doc["source"]
            item = doc["item"]
            title = doc["title"]
            
            # Create "What is" question from title
            qa_pairs.append({
                "question": f"What is {item}?",
                "answer": self._generate_what_is_answer(doc),
                "source": source,
                "item": item
            })
            
            # Create questions from sections
            for section in doc["sections"]:
                section_title = section["title"]
                section_content = section["content"]
                
                if len(section_content) < 50:  # Skip short sections
                    continue
                
                # Generate question from section title
                question = self._generate_question_from_section(section_title, item)
                
                qa_pairs.append({
                    "question": question,
                    "answer": section_content,
                    "source": source,
                    "item": item,
                    "section": section_title
                })
        
        return qa_pairs
    
    def _generate_what_is_answer(self, doc):
        """Generate a comprehensive 'What is' answer from the documentation"""
        # Try to find an introduction section
        intro_content = ""
        for section in doc["sections"]:
            title = section["title"].lower()
            if "introduction" in title or "overview" in title or "what is" in title:
                intro_content = section["content"]
                break
        
        # If no introduction found, combine first few sections
        if not intro_content:
            combined_content = ""
            for section in doc["sections"][:3]:  # First 3 sections
                combined_content += section["content"] + "\n\n"
            intro_content = combined_content
        
        # If still empty, use a generic template
        if not intro_content:
            intro_content = f"{doc['item']} is a service/feature provided by {doc['source'].upper()}. It allows users to perform various operations related to {doc['item'].replace('-', ' ')}."
        
        return intro_content
    
    def _generate_question_from_section(self, section_title, item):
        """Generate a question from a section title"""
        section_lower = section_title.lower()
        
        # Common patterns
        if "getting started" in section_lower:
            return f"How do I get started with {item}?"
        elif "best practices" in section_lower:
            return f"What are the best practices for {item}?"
        elif "limitations" in section_lower or "quotas" in section_lower:
            return f"What are the limitations or quotas of {item}?"
        elif "pricing" in section_lower or "cost" in section_lower:
            return f"How much does {item} cost?"
        elif "security" in section_lower:
            return f"How secure is {item} and what security features does it offer?"
        elif "example" in section_lower:
            return f"Can you provide examples of using {item}?"
        elif "tutorial" in section_lower:
            return f"Do you have a tutorial for {item}?"
        elif "architecture" in section_lower:
            return f"What is the architecture of {item}?"
        elif "common" in section_lower and ("issues" in section_lower or "problems" in section_lower):
            return f"What are common issues or problems with {item}?"
        else:
            # Generic format
            return f"Can you explain {section_title} for {item}?"
    
    def generate_fallback_qa(self):
        """
        Generate fallback Q&A pairs when scraping fails.
        
        Returns:
            list: Generated Q&A pairs
        """
        logger.warning("Generating fallback documentation Q&A pairs")
        qa_pairs = []
        
        # Define domains and services/concepts
        domains = {
            "cloud": {
                "services": ["EC2", "S3", "Lambda", "Azure VM", "Google Cloud Functions", "DynamoDB", "Kubernetes", "Docker", "Terraform"],
                "concepts": ["IaaS", "PaaS", "SaaS", "Serverless", "Containers", "Microservices", "Infrastructure as Code", "Auto-scaling"]
            },
            "programming": {
                "languages": ["Python", "JavaScript", "Java", "C#", "Go", "Ruby", "TypeScript", "PHP"],
                "concepts": ["Object-Oriented Programming", "Functional Programming", "Asynchronous Programming", "Testing", "Design Patterns", "Data Structures", "Algorithms"]
            },
            "databases": {
                "types": ["Relational Databases", "NoSQL Databases", "Document Stores", "Key-Value Stores", "Graph Databases", "Time-Series Databases"],
                "products": ["MySQL", "PostgreSQL", "MongoDB", "Redis", "Elasticsearch", "Cassandra", "Neo4j"]
            },
            "devops": {
                "tools": ["Git", "Jenkins", "GitHub Actions", "Travis CI", "Ansible", "Puppet", "Chef", "Prometheus", "Grafana"],
                "concepts": ["CI/CD", "GitOps", "Infrastructure as Code", "Monitoring", "Observability", "SRE", "DevSecOps"]
            },
            "security": {
                "concepts": ["Authentication", "Authorization", "Encryption", "PKI", "Zero Trust", "Vulnerability Management", "SIEM", "Penetration Testing"],
                "tools": ["OWASP ZAP", "Metasploit", "Nessus", "Burp Suite", "Wireshark", "Snort", "Splunk", "ELK Stack"]
            }
        }
        
        # Question templates
        templates = {
            "what_is": "What is {concept}?",
            "how_to": "How do I {action} with {concept}?",
            "best_practices": "What are the best practices for {concept}?",
            "difference": "What's the difference between {concept1} and {concept2}?",
            "common_issues": "What are common issues with {concept} and how to resolve them?",
            "architecture": "Can you explain the architecture of {concept}?",
            "security": "What are the security considerations for {concept}?",
            "scalability": "How does {concept} handle scaling?",
            "example": "Can you provide an example of using {concept}?"
        }
        
        # Generate QA pairs
        for domain, categories in domains.items():
            for category, items in categories.items():
                for item in items:
                    # Generate 2-5 questions per item
                    for _ in range(random.randint(2, 5)):
                        template_key = random.choice(list(templates.keys()))
                        template = templates[template_key]
                        
                        if template_key == "difference":
                            # Get a different item from the same category for comparison
                            other_items = [i for i in items if i != item]
                            if other_items:
                                concept2 = random.choice(other_items)
                                question = template.format(concept1=item, concept2=concept2)
                                answer = self._generate_comparison_answer(item, concept2, domain)
                            else:
                                continue
                        elif template_key == "how_to":
                            action = self._generate_action_for_concept(item, domain)
                            question = template.format(action=action, concept=item)
                            answer = self._generate_how_to_answer(action, item, domain)
                        else:
                            question = template.format(concept=item)
                            answer = self._generate_answer(item, template_key, domain)
                        
                        qa_pairs.append({
                            "question": question,
                            "answer": answer,
                            "source": "generated",
                            "domain": domain,
                            "category": category,
                            "item": item
                        })
        
        # Limit to sample size
        random.shuffle(qa_pairs)
        return qa_pairs[:self.sample_size]
    
    def _generate_action_for_concept(self, concept, domain):
        """Generate a relevant action for a concept based on domain"""
        actions = {
            "cloud": ["deploy", "configure", "optimize", "secure", "scale", "monitor", "backup", "restore"],
            "programming": ["implement", "debug", "test", "optimize", "refactor", "document", "integrate"],
            "databases": ["design", "query", "optimize", "back up", "replicate", "migrate", "index", "shard"],
            "devops": ["implement", "automate", "configure", "deploy", "monitor", "maintain", "integrate"],
            "security": ["implement", "configure", "audit", "test", "monitor", "respond to incidents with", "secure"]
        }
        
        domain_actions = actions.get(domain, actions["cloud"])
        return random.choice(domain_actions)
    
    def _generate_answer(self, concept, template_key, domain):
        """Generate an answer based on template type"""
        if template_key == "what_is":
            definitions = {
                "cloud": f"{concept} is a cloud computing service/feature that provides a way to {self._generate_action_for_concept(concept, 'cloud')} resources in the cloud. It offers scalability, reliability, and cost-effectiveness for businesses of all sizes.",
                "programming": f"{concept} is a programming language/paradigm/concept that allows developers to {self._generate_action_for_concept(concept, 'programming')} software applications. It's widely used in various domains including web development, data science, and enterprise applications.",
                "databases": f"{concept} is a database system/concept that provides a way to {self._generate_action_for_concept(concept, 'databases')} data. It's designed for [specific use case] and offers features like [key features].",
                "devops": f"{concept} is a DevOps tool/practice that helps teams {self._generate_action_for_concept(concept, 'devops')} their development and operations workflows. It's an essential part of modern software delivery pipelines.",
                "security": f"{concept} is a security tool/practice/concept that helps organizations {self._generate_action_for_concept(concept, 'security')} their systems and data. It's a critical component of a comprehensive security strategy."
            }
            return definitions.get(domain, f"{concept} is a technical tool or concept related to {domain}.")
            
        elif template_key == "best_practices":
            return f"When working with {concept}, follow these best practices:\n\n" + \
                   f"1. **Plan properly**: Before implementing {concept}, ensure you understand your requirements fully.\n" + \
                   f"2. **Follow standards**: Adhere to industry standards for {concept} implementation.\n" + \
                   f"3. **Document thoroughly**: Maintain comprehensive documentation for your {concept} setup.\n" + \
                   f"4. **Test extensively**: Regularly test your {concept} implementation to ensure it works as expected.\n" + \
                   f"5. **Monitor and optimize**: Continuously monitor and optimize your {concept} for better performance.\n" + \
                   f"6. **Stay updated**: Keep up with the latest developments and updates for {concept}.\n" + \
                   f"7. **Security first**: Always prioritize security in your {concept} implementation.\n"
                   
        elif template_key == "common_issues":
            return f"When working with {concept}, you might encounter these common issues:\n\n" + \
                   f"1. **Performance bottlenecks**: {concept} may experience performance issues under heavy load.\n" + \
                   "   *Solution*: Implement proper caching and optimization strategies.\n\n" + \
                   f"2. **Configuration errors**: Misconfiguration is a common source of problems with {concept}.\n" + \
                   "   *Solution*: Follow the official documentation and use validation tools.\n\n" + \
                   f"3. **Compatibility issues**: {concept} might not work well with certain systems or versions.\n" + \
                   "   *Solution*: Check compatibility matrices and test thoroughly.\n\n" + \
                   f"4. **Resource constraints**: {concept} might require more resources than initially allocated.\n" + \
                   "   *Solution*: Monitor resource usage and scale accordingly.\n\n" + \
                   f"5. **Integration challenges**: Integrating {concept} with existing systems can be challenging.\n" + \
                   "   *Solution*: Use appropriate adapters and follow integration patterns.\n"
                   
        elif template_key == "architecture":
            return f"The architecture of {concept} consists of the following key components:\n\n" + \
                   f"1. **Core Engine**: The central component that handles the main functionality of {concept}.\n\n" + \
                   f"2. **API Layer**: Provides interfaces for interacting with {concept} programmatically.\n\n" + \
                   f"3. **Storage Subsystem**: Manages data persistence and retrieval within {concept}.\n\n" + \
                   f"4. **Processing Units**: Handle computation and business logic for {concept} operations.\n\n" + \
                   f"5. **Security Layer**: Ensures authentication, authorization, and overall security of {concept}.\n\n" + \
                   f"These components work together through well-defined interfaces, allowing {concept} to provide its functionality in a scalable, maintainable way."
                   
        elif template_key == "security":
            return f"When implementing {concept}, consider these security considerations:\n\n" + \
                   f"1. **Authentication and Authorization**: Ensure proper access controls for {concept}.\n\n" + \
                   f"2. **Data Protection**: Implement encryption for data at rest and in transit for {concept}.\n\n" + \
                   f"3. **Vulnerability Management**: Regularly update {concept} to protect against known vulnerabilities.\n\n" + \
                   f"4. **Audit and Logging**: Maintain comprehensive logs for all {concept} activities.\n\n" + \
                   f"5. **Compliance**: Ensure {concept} implementation meets relevant compliance requirements.\n\n" + \
                   f"6. **Secure Configuration**: Follow security best practices when configuring {concept}.\n\n" + \
                   f"7. **Third-party Risk**: Assess security risks associated with any {concept} dependencies.\n"
                   
        elif template_key == "scalability":
            return f"{concept} handles scaling through these mechanisms:\n\n" + \
                   f"1. **Horizontal Scaling**: {concept} can scale horizontally by adding more instances or nodes.\n\n" + \
                   f"2. **Vertical Scaling**: {concept} supports vertical scaling by adding more resources to existing instances.\n\n" + \
                   f"3. **Auto-scaling**: {concept} offers automatic scaling based on defined metrics and thresholds.\n\n" + \
                   f"4. **Load Balancing**: {concept} distributes load across multiple instances for better performance.\n\n" + \
                   f"5. **Partitioning**: {concept} supports data or workload partitioning for improved scalability.\n\n" + \
                   f"6. **Caching**: {concept} implements caching strategies to reduce load on primary resources.\n\n" + \
                   f"These scaling capabilities make {concept} suitable for both small-scale and enterprise-level deployments."
                   
        elif template_key == "example":
            return f"Here's an example of using {concept}:\n\n" + \
                   f"```\n# Example code or configuration for {concept}\n# This demonstrates a basic implementation\n\n" + \
                   f"# Step 1: Initialize or configure {concept}\ninitialize_{concept.lower().replace(' ', '_')}()\n\n" + \
                   f"# Step 2: Implement core functionality\nresult = perform_operation_with_{concept.lower().replace(' ', '_')}(parameters)\n\n" + \
                   f"# Step 3: Handle results\nprocess_results(result)\n```\n\n" + \
                   f"This example shows the basic workflow when working with {concept}. You would typically start by initializing or configuring it, then use its core functionality, and finally process the results or handle any cleanup operations."
        
        # Default
        return f"Information about {concept} would be provided here, including its purpose, features, and common use cases."
    
    def _generate_how_to_answer(self, action, concept, domain):
        """Generate a 'how to' answer"""
        return f"To {action} with {concept}, follow these steps:\n\n" + \
               f"1. **Preparation**: Ensure you have all prerequisites for {concept}.\n\n" + \
               f"2. **Initial Setup**: Configure the basic settings for {concept}.\n\n" + \
               f"```\n# Example configuration\n{concept.lower().replace(' ', '_')}_config = {{\n  'setting1': 'value1',\n  'setting2': 'value2'\n}}\n```\n\n" + \
               f"3. **Implementation**: Implement the core functionality for {action}.\n\n" + \
               f"```\n# Example implementation\nresult = {concept.lower().replace(' ', '_')}.{action.replace(' ', '_')}(parameters)\n```\n\n" + \
               f"4. **Testing**: Verify that your implementation works as expected.\n\n" + \
               f"5. **Monitoring**: Set up monitoring to track the performance and health of your {concept} implementation.\n\n" + \
               f"Additional tips for {action} with {concept}:\n" + \
               f"- Follow the official documentation for the most up-to-date information\n" + \
               f"- Consider performance implications, especially in production environments\n" + \
               f"- Implement proper error handling and logging"
    
    def _generate_comparison_answer(self, concept1, concept2, domain):
        """Generate a comparison between two concepts"""
        return f"# Comparison: {concept1} vs {concept2}\n\n" + \
               f"## Key Differences\n\n" + \
               f"### 1. **Purpose**\n" + \
               f"- **{concept1}** is primarily designed for [specific use case].\n" + \
               f"- **{concept2}** is optimized for [different use case].\n\n" + \
               f"### 2. **Features**\n" + \
               f"- **{concept1}** offers features like [key features].\n" + \
               f"- **{concept2}** provides capabilities such as [key features].\n\n" + \
               f"### 3. **Performance**\n" + \
               f"- **{concept1}** generally excels in [performance characteristic].\n" + \
               f"- **{concept2}** typically performs better for [different performance characteristic].\n\n" + \
               f"### 4. **Use Cases**\n" + \
               f"- **{concept1}** is ideal for [use case scenarios].\n" + \
               f"- **{concept2}** is better suited for [different use case scenarios].\n\n" + \
               f"### 5. **Learning Curve and Community**\n" + \
               f"- **{concept1}** has [learning curve characteristic] and [community characteristic].\n" + \
               f"- **{concept2}** offers [different learning curve] and [community characteristic].\n\n" + \
               f"## When to Choose Each\n\n" + \
               f"Choose **{concept1}** when:\n" + \
               f"- You need [specific capability]\n" + \
               f"- Your project requires [specific characteristic]\n" + \
               f"- You're working with [related technology/environment]\n\n" + \
               f"Choose **{concept2}** when:\n" + \
               f"- You prioritize [different capability]\n" + \
               f"- Your project has [different requirement]\n" + \
               f"- You're working within [different context]"
    
    def process_documentation(self):
        """
        Process technical documentation from all sources.
        
        Returns:
            list: Processed Q&A pairs
        """
        all_docs = []
        qa_pairs = []
        
        # Process each source
        for source, data in DOCS_SOURCES.items():
            # Process each item
            items = data.get("services", data.get("commands", []))
            for item in items:
                try:
                    # Fetch and parse documentation
                    content = self.fetch_documentation(source, item)
                    if content:
                        parsed = self.parse_documentation(content, source, item)
                        if parsed:
                            all_docs.append(parsed)
                            
                except Exception as e:
                    logger.error(f"Error processing {source}/{item}: {str(e)}")
        
        # If we don't have enough docs, generate fallback
        if len(all_docs) < 10:  # Arbitrary threshold
            logger.warning("Insufficient documentation fetched, using fallback data")
            qa_pairs = self.generate_fallback_qa()
        else:
            # Convert to Q&A format
            qa_pairs = self.convert_to_qa(all_docs)
            
            # Limit to sample size
            if len(qa_pairs) > self.sample_size:
                qa_pairs = random.sample(qa_pairs, self.sample_size)
        
        return qa_pairs
    
    def save_data(self, qa_pairs, output_file=None):
        """
        Save processed data to file.
        
        Args:
            qa_pairs (list): QA pairs to save
            output_file (str, optional): Output file path
            
        Returns:
            str: Path to saved file
        """
        if not output_file:
            output_file = self.output_dir / "technical_documentation.json"
            
        # Create parent directory if it doesn't exist
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save to file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Saved {len(qa_pairs)} QA pairs to {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            return None

def main(sample_size=5000, output_dir="./Datasets", cache_dir=None):
    """
    Process technical documentation and save to file.
    
    Args:
        sample_size (int): Number of QA pairs to generate
        output_dir (str): Output directory
        cache_dir (str, optional): Cache directory
        
    Returns:
        str: Path to saved file
    """
    processor = TechnicalDocumentationProcessor(
        output_dir=output_dir,
        cache_dir=cache_dir,
        sample_size=sample_size
    )
    
    qa_pairs = processor.process_documentation()
    output_file = processor.save_data(qa_pairs, os.path.join(output_dir, "technical_documentation.json"))
    
    return output_file

if __name__ == "__main__":
    main()
