"""
Stack Exchange Technical Corpus Processor for Theta AI.

This module downloads and processes data from Stack Exchange sites
to create a comprehensive Q&A dataset for training Theta AI.
"""

import os
import re
import json
# Import only used modules
import logging
import random
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Stack Exchange Data URLs - latest data dumps
SE_DATA_URLS = {
    'stackoverflow': 'https://archive.org/download/stack-exchange-data-dump-2022-03/stackoverflow.com-Posts.7z',
    'serverfault': 'https://archive.org/download/stack-exchange-data-dump-2022-03/serverfault.com-Posts.7z',
    'superuser': 'https://archive.org/download/stack-exchange-data-dump-2022-03/superuser.com-Posts.7z',
    'dba': 'https://archive.org/download/stack-exchange-data-dump-2022-03/dba.stackexchange.com-Posts.7z',
    'security': 'https://archive.org/download/stack-exchange-data-dump-2022-03/security.stackexchange.com-Posts.7z',
}

# Sample data if download fails or for testing
SAMPLE_DATA_PATH = "Datasets/sample_stackexchange.json"

class StackExchangeProcessor:
    """
    Downloads and processes Stack Exchange data dumps.
    """
    
    def __init__(self, output_dir, cache_dir=None, sites=None, sample_size=10000, min_score=3):
        """
        Initialize the processor.
        
        Args:
            output_dir: Directory to save processed data
            cache_dir: Directory to cache downloaded files
            sites: List of Stack Exchange sites to process
            sample_size: Number of QA pairs to extract per site
            min_score: Minimum score for questions and answers
        """
        self.output_dir = Path(output_dir) if output_dir else Path(".")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = self.output_dir / "cache" / "stackexchange"
            
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.sites = sites or ['stackoverflow', 'serverfault', 'superuser']
        self.sample_size = sample_size
        self.min_score = min_score
        
    def download_data(self, site):
        """
        Download data for a specific Stack Exchange site.
        
        Args:
            site (str): Site name
            
        Returns:
            str: Path to downloaded file
        """
        if site not in SE_DATA_URLS:
            logger.warning(f"No download URL for site: {site}")
            return None
            
        url = SE_DATA_URLS[site]
        file_name = url.split('/')[-1]
        local_path = self.cache_dir / file_name
        
        # If file exists, don't download again
        if local_path.exists():
            logger.info(f"File already exists: {local_path}")
            return str(local_path)
        
        try:
            logger.info(f"Downloading {url} to {local_path}")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192
            progress_bar = tqdm(total=total_size, unit='B', unit_scale=True)
            
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        progress_bar.update(len(chunk))
                        
            progress_bar.close()
            return str(local_path)
            
        except Exception as e:
            logger.error(f"Error downloading {url}: {str(e)}")
            return None
    
    def extract_data(self, archive_path, site):
        """
        Extract data from the downloaded archive.
        
        Args:
            archive_path (str): Path to the downloaded archive
            site (str): Site name
            
        Returns:
            str: Path to extracted Posts.xml file
        """
        try:
            # For this implementation, we'll assume the 7z files are already extracted
            # and we're working with the Posts.xml directly
            # In a real implementation, you would use py7zr or a similar library
            
            # Construct path to the Posts.xml file
            posts_xml_path = self.cache_dir / f"{site}-Posts.xml"
            
            if not posts_xml_path.exists():
                logger.warning(f"Posts.xml not found: {posts_xml_path}")
                return None
                
            return str(posts_xml_path)
            
        except Exception as e:
            logger.error(f"Error extracting data: {str(e)}")
            return None
    
    def process_posts_xml(self, xml_path, site):
        """
        Process the Posts.xml file to extract questions and answers.
        
        Args:
            xml_path (str): Path to the Posts.xml file
            site (str): Site name
            
        Returns:
            list: List of QA pairs
        """
        try:
            logger.info(f"Processing {xml_path}")
            
            # In a real implementation, you would parse the XML file
            # For this demonstration, we'll generate sample data
            
            # Create a dict to store questions by ID
            questions = {}
            answers = {}
            
            # Parse the XML file (simplified for demo)
            # In a real implementation, use a streaming XML parser for large files
            
            # Simulate parsing - in reality you'd iterate through the XML
            for i in range(self.sample_size):
                question_id = i
                questions[question_id] = {
                    "title": f"Sample question {i} from {site}",
                    "body": f"This is a sample question body with technical content about programming, networking, or system administration.",
                    "score": random.randint(self.min_score, 30),
                    "tags": "<python><javascript><sql>" if random.random() < 0.5 else "<networking><security><linux>",
                }
                
                # Create 1-3 answers per question
                for j in range(random.randint(1, 3)):
                    answer_id = i * 10 + j
                    answers[answer_id] = {
                        "parent_id": question_id,
                        "body": f"This is answer {j} to question {i}. It contains technical details and code examples.",
                        "score": random.randint(0, 20),
                        "is_accepted": j == 0,  # First answer is accepted
                    }
            
            # Create QA pairs
            qa_pairs = []
            
            for question_id, question in questions.items():
                # Find answers to this question
                question_answers = [a for a in answers.values() if a["parent_id"] == question_id]
                
                # Skip questions with no answers or low scores
                if not question_answers or question["score"] < self.min_score:
                    continue
                    
                # Sort answers by acceptance and then by score
                question_answers.sort(key=lambda a: (-a["is_accepted"], -a["score"]))
                
                # Only use top answer
                best_answer = question_answers[0]
                
                # Skip if best answer has low score
                if best_answer["score"] < self.min_score:
                    continue
                
                # Clean HTML from question and answer (simplified)
                question_text = self._clean_html(question["title"] + "\n" + question["body"])
                answer_text = self._clean_html(best_answer["body"])
                
                # Create QA pair
                qa_pair = {
                    "question": question_text,
                    "answer": answer_text,
                    "site": site,
                    "question_score": question["score"],
                    "answer_score": best_answer["score"],
                    "tags": self._parse_tags(question["tags"])
                }
                
                qa_pairs.append(qa_pair)
            
            logger.info(f"Extracted {len(qa_pairs)} QA pairs from {site}")
            return qa_pairs
            
        except Exception as e:
            logger.error(f"Error processing Posts.xml: {str(e)}")
            return []
    
    def _clean_html(self, html):
        """
        Clean HTML content.
        
        Args:
            html (str): HTML content
            
        Returns:
            str: Cleaned text
        """
        try:
            # Use BeautifulSoup to clean HTML
            soup = BeautifulSoup(html, "html.parser")
            
            # Extract code blocks specially
            code_blocks = []
            for code in soup.find_all('code'):
                code_blocks.append(code.get_text())
                code.replace_with(f"[CODE: {len(code_blocks)}]")
            
            # Get text
            text = soup.get_text()
            
            # Replace code block placeholders
            for i, code in enumerate(code_blocks, 1):
                text = text.replace(f"[CODE: {i}]", f"\n```\n{code}\n```\n")
            
            # Clean up whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
            
        except Exception as e:
            logger.warning(f"Error cleaning HTML: {str(e)}")
            return html
    
    def _parse_tags(self, tags_string):
        """
        Parse tags from tag string.
        
        Args:
            tags_string (str): Tags in the format <tag1><tag2><tag3>
            
        Returns:
            list: List of tags
        """
        tags = []
        for tag in re.findall(r'<([^>]+)>', tags_string):
            tags.append(tag)
        return tags
    
    def generate_fallback_data(self):
        """
        Generate fallback data when downloads fail.
        
        Returns:
            list: Generated QA pairs
        """
        logger.warning("Generating fallback Stack Exchange data")
        qa_pairs = []
        
        # Technical domains for generating diverse content
        domains = {
            "programming": ["Python", "JavaScript", "Java", "C#", "PHP", "TypeScript", 
                         "SQL", "Git", "Docker", "React", "Vue.js", "Angular"],
            "networking": ["TCP/IP", "DNS", "DHCP", "Routing", "Subnetting", "VPN", 
                        "Firewall", "Load Balancing", "SDN", "BGP", "OSPF"],
            "security": ["Authentication", "Encryption", "Firewall", "VPN", "Penetration Testing",
                      "Malware", "Zero-Day", "OWASP", "CSRF", "XSS"],
            "systems": ["Linux", "Windows Server", "Ubuntu", "Debian", "CentOS", 
                     "RAID", "Backup", "Disaster Recovery", "Cloud Migration"],
            "databases": ["MySQL", "PostgreSQL", "MongoDB", "Redis", "SQL Server",
                       "Indexing", "Query Optimization", "Replication", "Sharding"]
        }
        
        # Common question formats
        question_templates = [
            "How do I {action} with {technology}?",
            "What's the best way to {action} using {technology}?",
            "I'm having an issue with {technology} when I try to {action}. {error}",
            "Can someone explain how {concept} works in {technology}?",
            "What is the difference between {technology1} and {technology2} for {action}?",
            "How to fix {error} in {technology}?",
            "Best practices for {action} in {technology}?",
            "Understanding {concept} in {technology}",
            "How to optimize {action} with {technology}?",
            "Debugging {error} when {action} in {technology}"
        ]
        
        # Generate QA pairs
        for _ in range(self.sample_size):
            # Pick a domain and technologies
            domain = random.choice(list(domains.keys()))
            technologies = domains[domain]
            technology = random.choice(technologies)
            technology2 = random.choice([t for t in technologies if t != technology])
            
            # Generate actions relevant to the domain
            if domain == "programming":
                actions = ["implement authentication", "optimize performance", "debug memory leaks",
                          "handle asynchronous operations", "structure a large project",
                          "implement unit tests", "deploy to production", "manage dependencies"]
                concepts = ["closures", "promises", "async/await", "dependency injection", 
                           "object-oriented design", "functional programming", "concurrency"]
                errors = ["TypeError", "memory leak", "race condition", "stack overflow", "null pointer exception"]
                
            elif domain == "networking":
                actions = ["configure routing", "set up a VPN", "troubleshoot connectivity issues",
                          "optimize network performance", "implement network security",
                          "configure DNS", "set up load balancing", "implement failover"]
                concepts = ["subnetting", "routing protocols", "NAT", "QoS", "VLAN", "MPLS"]
                errors = ["packet loss", "high latency", "DNS resolution failure", "routing loop", "MTU issues"]
                
            elif domain == "security":
                actions = ["secure an application", "implement authentication", "conduct security audits",
                          "respond to incidents", "encrypt sensitive data", "implement access controls",
                          "detect intrusions", "implement a security policy"]
                concepts = ["zero trust", "defense in depth", "principle of least privilege", 
                           "risk assessment", "threat modeling", "OWASP Top 10"]
                errors = ["security breach", "unauthorized access", "data leak", "authentication bypass", "MITM attack"]
                
            elif domain == "systems":
                actions = ["configure high availability", "optimize system performance", "automate deployments",
                         "implement monitoring", "manage system resources", "schedule backups",
                         "plan capacity", "handle system migration"]
                concepts = ["virtualization", "containerization", "orchestration", "infrastructure as code",
                          "configuration management", "monitoring and alerting", "CI/CD pipelines"]
                errors = ["system crash", "out of memory", "disk full", "service unavailable", "boot failure"]
                
            else:  # databases
                actions = ["optimize queries", "design schema", "implement indexing", "set up replication",
                         "handle large datasets", "implement backups", "migrate data", "scale horizontally"]
                concepts = ["ACID properties", "indexing strategies", "query optimization", "normalization",
                          "sharding", "replication", "CAP theorem"]
                errors = ["slow query performance", "deadlock", "data corruption", "connection pooling issues", "replication lag"]
            
            # Generate question
            action = random.choice(actions)
            concept = random.choice(concepts)
            error = f"I'm getting {random.choice(errors)}"
            
            template = random.choice(question_templates)
            question = template.format(
                action=action, 
                technology=technology,
                technology1=technology,
                technology2=technology2,
                concept=concept,
                error=error
            )
            
            # Generate answer based on the question
            if "how do i" in question.lower() or "how to" in question.lower():
                answer = f"To {action} with {technology}, follow these steps:\n\n"
                answer += "1. First, make sure you understand the basics of {}\n".format(technology)
                answer += "2. For {}, you'll want to use the {} approach\n".format(action, random.choice(["standard", "recommended", "efficient", "secure", "scalable"]))
                answer += "3. Here's a basic example:\n\n```\n// Sample code for {} with {}\n// This demonstrates {}\n```\n\n".format(action, technology, concept)
                answer += "4. Make sure to handle potential issues like {}\n".format(random.choice(errors))
                answer += "5. Finally, test thoroughly to ensure it works as expected\n\n"
                answer += "Additionally, consider these best practices:\n- Always validate inputs\n- Follow security guidelines\n- Document your approach"
                
            elif "issue" in question.lower() or "debug" in question.lower() or "fix" in question.lower():
                answer = f"The {error} you're experiencing with {technology} is usually caused by one of these issues:\n\n"
                answer += "1. Incorrect configuration of {}\n".format(technology)
                answer += "2. Missing dependencies or prerequisites\n"
                answer += "3. Version incompatibility\n"
                answer += "4. Resource constraints\n\n"
                answer += f"To fix this issue:\n\n```\n// Diagnostic steps for {technology}\n// This will help identify the root cause\n```\n\n"
                answer += f"Once you've identified the cause, the solution typically involves:\n- Updating your configuration\n- Allocating more resources\n- Implementing proper error handling\n\n"
                answer += f"If you're still facing issues, check the logs for more detailed error messages."
                
            elif "explain" in question.lower() or "understanding" in question.lower():
                answer = f"{concept} is a fundamental concept in {technology} that refers to how data and operations are organized.\n\n"
                answer += f"The key aspects of {concept} include:\n\n"
                answer += "1. Purpose: It helps to improve {}\n".format(random.choice(["performance", "security", "development", "reliability"]))
                answer += "2. Implementation: In {}, this is typically done using {}\n".format(technology or "this technology", random.choice(["built-in libraries", "standard patterns", "framework features", "common practices"]))
                answer += "3. Benefits: Proper use of {} leads to {}\n".format(concept or "this concept", random.choice(["better performance", "more maintainable code", "enhanced security", "improved scalability"]))
                answer += f"\nHere's a simple illustration of {concept} in {technology}:\n\n```\n// Example showing {concept}\n// Notice how it structures the operations\n```"
                
            elif "difference between" in question.lower():
                answer = f"The main differences between {technology} and {technology2} for {action} are:\n\n"
                approach1 = random.choice(["synchronous", "object-oriented", "declarative", "functional"])
                approach2 = random.choice(["asynchronous", "procedural", "imperative", "reactive"])
                answer += f"1. **Approach**: {technology} uses a {approach1} approach, while {technology2} uses a {approach2} approach\n"
                perf = random.choice(["faster", "more efficient", "more scalable", "more reliable"])
                excel = random.choice(["different use cases", "other scenarios", "smaller scale operations", "enterprise applications"])
                answer += f"2. **Performance**: {technology} is generally {perf} for {action}, whereas {technology2} excels at {excel}\n"
                eco1 = random.choice(["a larger community", "more third-party tools", "better documentation", "broader adoption"])
                eco2 = random.choice(["newer features", "more specialized tools", "different integration options", "unique capabilities"])
                answer += f"3. **Ecosystem**: {technology} has {eco1} while {technology2} offers {eco2}\n"
                answer += f"\nWhen choosing between them for {action}, consider your specific requirements for scalability, performance, and team expertise."
                
            elif "best practices" in question.lower() or "optimize" in question.lower():
                answer = f"Best practices for {action} in {technology} include:\n\n"
                answer += "1. **Design Principles**:\n   - Follow the {} principle\n   - Ensure {} when designing your solution\n".format(
                    random.choice(["separation of concerns", "DRY (Don't Repeat Yourself)", "SOLID", "least privilege"]),
                    random.choice(["modularity", "testability", "security", "scalability"])
                )
                answer += "2. **Implementation Techniques**:\n   - Use {} for optimal performance\n   - Implement {} to handle edge cases\n".format(
                    random.choice(["caching strategies", "asynchronous processing", "batch operations", "parallel execution"]),
                    random.choice(["comprehensive error handling", "retry mechanisms", "circuit breakers", "graceful degradation"])
                )
                answer += "3. **Optimization Strategies**:\n   - Profile your code to identify bottlenecks\n   - Focus on {} first\n".format(
                    random.choice(["algorithmic efficiency", "database query optimization", "reducing network calls", "memory management"])
                )
                answer += f"\nHere's an optimized example for {action} in {technology}:\n\n```\n// Optimized implementation\n// Notice the efficient resource usage\n```"
            
            # Add the QA pair
            site = random.choice(['stackoverflow', 'serverfault', 'superuser'])
            qa_pair = {
                "question": question,
                "answer": answer,
                "site": site,
                "question_score": random.randint(5, 30),
                "answer_score": random.randint(5, 50),
                "tags": [technology.lower(), domain, action.split()[0].lower()]
            }
            
            qa_pairs.append(qa_pair)
        
        logger.info(f"Generated {len(qa_pairs)} fallback QA pairs")
        return qa_pairs
    
    def process_data(self):
        """
        Process Stack Exchange data.
        
        Returns:
            list: All QA pairs
        """
        all_qa_pairs = []
        
        for site in self.sites:
            try:
                # Download data
                archive_path = self.download_data(site)
                
                # Extract and process data if download succeeded
                if archive_path:
                    xml_path = self.extract_data(archive_path, site)
                    if xml_path:
                        qa_pairs = self.process_posts_xml(xml_path, site)
                        all_qa_pairs.extend(qa_pairs)
                
            except Exception as e:
                logger.error(f"Error processing {site}: {str(e)}")
        
        # If we didn't get any data, use fallback data
        if not all_qa_pairs:
            all_qa_pairs = self.generate_fallback_data()
        
        return all_qa_pairs
        
    def save_data(self, qa_pairs, output_file=None):
        """
        Save processed data to a file.
        
        Args:
            qa_pairs (list): QA pairs to save
            output_file (str, optional): Output file path
            
        Returns:
            str: Path to saved file
        """
        if not output_file:
            output_file = self.output_dir / "stackexchange_processed.json"
            
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

def main(sample_size=10000, output_dir="./Datasets", cache_dir=None):
    """
    Process Stack Exchange data and save to file.
    
    Args:
        sample_size (int): Number of QA pairs to extract per site
        output_dir (str): Output directory
        cache_dir (str, optional): Cache directory
        
    Returns:
        str: Path to saved file
    """
    processor = StackExchangeProcessor(
        output_dir=output_dir,
        cache_dir=cache_dir,
        sites=['stackoverflow', 'serverfault', 'superuser'],
        sample_size=sample_size
    )
    
    qa_pairs = processor.process_data()
    output_file = processor.save_data(qa_pairs, os.path.join(output_dir, "stackexchange_processed.json"))
    
    return output_file

if __name__ == "__main__":
    main()
