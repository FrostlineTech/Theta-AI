"""
GitHub Issue Conversations Dataset Generator for Theta AI.

This module downloads and processes GitHub issues and pull requests
to create conversation datasets with realistic multi-turn technical discussions.
"""

import os
import re
import json
import logging
import random
import requests
import time
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Popular GitHub repositories across different domains
REPOS = {
    "programming": [
        "facebook/react", "tensorflow/tensorflow", "microsoft/TypeScript", 
        "pandas-dev/pandas", "django/django", "spring-projects/spring-boot"
    ],
    "devops": [
        "kubernetes/kubernetes", "docker/docker-ce", "prometheus/prometheus",
        "hashicorp/terraform", "ansible/ansible", "grafana/grafana"
    ],
    "security": [
        "OWASP/CheatSheetSeries", "metasploit/metasploit-framework", 
        "fail2ban/fail2ban", "nmap/nmap", "sqlmapproject/sqlmap"
    ]
}

class GitHubIssueProcessor:
    """
    Downloads and processes GitHub issues and pull requests.
    """
    
    def __init__(self, output_dir, cache_dir=None, github_token=None, sample_size=2000, min_comments=3):
        """
        Initialize the processor.
        
        Args:
            output_dir: Directory to save processed data
            cache_dir: Directory to cache downloaded files
            github_token: GitHub API token for higher rate limits
            sample_size: Number of conversations to extract
            min_comments: Minimum number of comments for a conversation
        """
        self.output_dir = Path(output_dir) if output_dir else Path(".")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = self.output_dir / "cache" / "github"
            
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.github_token = github_token
        self.sample_size = sample_size
        self.min_comments = min_comments
        
        # Headers for GitHub API
        self.headers = {
            'Accept': 'application/vnd.github.v3+json'
        }
        
        if github_token:
            self.headers['Authorization'] = f'token {github_token}'
    
    def get_issues(self, repo, state='all', per_page=100):
        """
        Get issues and pull requests for a repository.
        
        Args:
            repo (str): Repository in format owner/repo
            state (str): Issue state (open, closed, all)
            per_page (int): Number of results per page
            
        Returns:
            list: Issues and pull requests
        """
        issues = []
        page = 1
        max_pages = 10  # Limit to avoid hitting API rate limits
        
        try:
            while len(issues) < 200 and page <= max_pages:  # Get up to 200 issues per repo
                url = f"https://api.github.com/repos/{repo}/issues"
                params = {
                    'state': state,
                    'per_page': per_page,
                    'page': page
                }
                
                logger.info(f"Fetching issues for {repo}, page {page}")
                response = requests.get(url, headers=self.headers, params=params)
                
                if response.status_code == 403 and 'rate limit exceeded' in response.text:
                    logger.warning("GitHub API rate limit exceeded")
                    break
                    
                response.raise_for_status()
                page_issues = response.json()
                
                if not page_issues:
                    break
                    
                issues.extend(page_issues)
                page += 1
                
                # Respect GitHub's rate limits
                time.sleep(1)
                
            logger.info(f"Fetched {len(issues)} issues for {repo}")
            return issues
            
        except Exception as e:
            logger.error(f"Error fetching issues for {repo}: {str(e)}")
            return []
    
    def get_comments(self, comments_url):
        """
        Get comments for an issue or pull request.
        
        Args:
            comments_url (str): URL to fetch comments
            
        Returns:
            list: Comments
        """
        try:
            logger.info(f"Fetching comments from {comments_url}")
            response = requests.get(comments_url, headers=self.headers)
            
            if response.status_code == 403 and 'rate limit exceeded' in response.text:
                logger.warning("GitHub API rate limit exceeded")
                return []
                
            response.raise_for_status()
            comments = response.json()
            
            # Respect GitHub's rate limits
            time.sleep(1)
            
            return comments
            
        except Exception as e:
            logger.error(f"Error fetching comments: {str(e)}")
            return []
    
    def process_issue(self, issue):
        """
        Process an issue or pull request.
        
        Args:
            issue (dict): Issue data
            
        Returns:
            dict: Processed conversation
        """
        try:
            # Get comments
            comments = self.get_comments(issue['comments_url'])
            
            # Skip if not enough comments
            if len(comments) < self.min_comments:
                return None
            
            # Prepare conversation
            conversation = {
                'title': issue['title'],
                'url': issue['html_url'],
                'state': issue['state'],
                'created_at': issue['created_at'],
                'updated_at': issue['updated_at'],
                'number': issue['number'],
                'repo': issue['repository_url'].split('/')[-2] + '/' + issue['repository_url'].split('/')[-1],
                'is_pr': 'pull_request' in issue,
                'exchanges': []
            }
            
            # Add initial post as first exchange
            body = self._clean_markdown(issue['body'] or '')
            
            conversation['exchanges'].append({
                'role': 'user',
                'content': f"**{issue['title']}**\n\n{body}",
                'author': issue['user']['login'],
                'created_at': issue['created_at']
            })
            
            # Add comments as exchanges, alternating between roles
            for i, comment in enumerate(comments):
                body = self._clean_markdown(comment['body'] or '')
                
                # Determine role - alternate between assistant and user
                # This is a simplification - in a real scenario, you'd track unique users
                role = 'assistant' if i % 2 == 0 else 'user'
                
                conversation['exchanges'].append({
                    'role': role,
                    'content': body,
                    'author': comment['user']['login'],
                    'created_at': comment['created_at']
                })
            
            return conversation
            
        except Exception as e:
            logger.error(f"Error processing issue: {str(e)}")
            return None
    
    def _clean_markdown(self, markdown):
        """
        Clean markdown content.
        
        Args:
            markdown (str): Markdown content
            
        Returns:
            str: Cleaned text
        """
        try:
            # Remove HTML comments
            markdown = re.sub(r'<!--.*?-->', '', markdown, flags=re.DOTALL)
            
            # Preserve code blocks
            code_blocks = []
            
            def replace_code_block(match):
                code_blocks.append(match.group(1))
                return f"[CODE_BLOCK_{len(code_blocks)}]"
                
            markdown = re.sub(r'```(?:.+?\n)?(.*?)```', replace_code_block, markdown, flags=re.DOTALL)
            
            # Convert to plain text (simplified)
            text = markdown
            
            # Replace code block placeholders
            for i, code in enumerate(code_blocks, 1):
                text = text.replace(f"[CODE_BLOCK_{i}]", f"\n```\n{code}\n```\n")
            
            # Clean up whitespace
            text = re.sub(r'\n\n+', '\n\n', text).strip()
            
            return text
            
        except Exception as e:
            logger.warning(f"Error cleaning markdown: {str(e)}")
            return markdown
    
    def process_repos(self):
        """
        Process repositories to extract conversations.
        
        Returns:
            list: All conversations
        """
        all_conversations = []
        
        # Flatten repos list
        all_repos = []
        for domain, repos in REPOS.items():
            all_repos.extend([(domain, repo) for repo in repos])
        
        # Shuffle repos to get diverse data
        random.shuffle(all_repos)
        
        # Process repos
        for domain, repo in all_repos:
            try:
                issues = self.get_issues(repo)
                
                # Filter issues with enough comments
                for issue in issues:
                    if issue.get('comments', 0) >= self.min_comments:
                        conversation = self.process_issue(issue)
                        
                        if conversation:
                            conversation['domain'] = domain
                            all_conversations.append(conversation)
                            
                            # Check if we have enough conversations
                            if len(all_conversations) >= self.sample_size:
                                break
                
                # Check if we have enough conversations
                if len(all_conversations) >= self.sample_size:
                    break
                    
            except Exception as e:
                logger.error(f"Error processing repo {repo}: {str(e)}")
        
        logger.info(f"Processed {len(all_conversations)} conversations")
        
        # If we don't have enough real data, generate fallback data
        if len(all_conversations) < self.sample_size:
            fallback_count = self.sample_size - len(all_conversations)
            fallback_conversations = self.generate_fallback_data(fallback_count)
            all_conversations.extend(fallback_conversations)
        
        return all_conversations
    
    def generate_fallback_data(self, count):
        """
        Generate fallback data when API fails or for testing.
        
        Args:
            count (int): Number of conversations to generate
            
        Returns:
            list: Generated conversations
        """
        logger.warning(f"Generating {count} fallback GitHub conversations")
        conversations = []
        
        # Technical domains and issues for fallback data
        domains = {
            "programming": {
                "issues": [
                    "Memory leak when processing large datasets",
                    "Performance degradation with parallel requests",
                    "TypeScript compiler error with generic types",
                    "React component not re-rendering after state update",
                    "Database connection pooling issue in high load",
                    "API rate limiting strategy for public endpoints",
                    "Webpack build optimization for large projects",
                    "GraphQL query complexity and performance",
                    "CSS layout breaking on specific browsers",
                    "Authentication token refresh mechanism"
                ],
                "repos": ["react/react", "python/cpython", "microsoft/TypeScript", "angular/angular", "django/django"]
            },
            "devops": {
                "issues": [
                    "Kubernetes pod scheduling failures",
                    "Docker container networking between services",
                    "Terraform state locking during parallel apply",
                    "CI pipeline failing with timeout on large builds",
                    "Prometheus alerting rules optimization",
                    "Ansible playbook idempotency issues",
                    "Load balancer health check configuration",
                    "Auto-scaling policy for spiky workloads",
                    "Zero-downtime deployment strategy",
                    "Logs aggregation for distributed systems"
                ],
                "repos": ["kubernetes/kubernetes", "docker/docker-ce", "hashicorp/terraform", "ansible/ansible"]
            },
            "security": {
                "issues": [
                    "CSRF protection bypass in form submission",
                    "JWT token validation vulnerability",
                    "SQL injection in dynamic query builder",
                    "Insecure direct object reference in API",
                    "TLS certificate validation issue",
                    "Authentication bypass in password reset",
                    "Cross-site scripting in markdown renderer",
                    "Directory traversal in file upload",
                    "Rate limiting bypass using distributed requests",
                    "Secure storage for API keys and secrets"
                ],
                "repos": ["OWASP/CheatSheetSeries", "metasploit/metasploit-framework", "fail2ban/fail2ban"]
            }
        }
        
        # Generate conversations
        for i in range(count):
            # Select domain and issue
            domain = random.choice(list(domains.keys()))
            issue_title = random.choice(domains[domain]["issues"])
            repo = random.choice(domains[domain]["repos"])
            
            # Generate conversation
            conversation = {
                'title': issue_title,
                'url': f"https://github.com/{repo}/issues/{i+1000}",
                'state': random.choice(['open', 'closed']),
                'created_at': "2025-01-15T10:30:45Z",
                'updated_at': "2025-01-20T14:25:30Z",
                'number': i+1000,
                'repo': repo,
                'is_pr': random.random() > 0.7,  # 30% are PRs
                'domain': domain,
                'exchanges': []
            }
            
            # Initial post
            conversation['exchanges'].append({
                'role': 'user',
                'content': self._generate_issue_body(issue_title, domain),
                'author': f"user{random.randint(1000, 9999)}",
                'created_at': conversation['created_at']
            })
            
            # Generate 3-7 comment exchanges
            num_exchanges = random.randint(3, 7)
            for j in range(num_exchanges):
                role = 'assistant' if j % 2 == 0 else 'user'
                conversation['exchanges'].append({
                    'role': role,
                    'content': self._generate_comment(issue_title, domain, j, role),
                    'author': f"user{random.randint(1000, 9999)}",
                    'created_at': f"2025-01-{15+j}T{10+j}:30:45Z"
                })
            
            conversations.append(conversation)
        
        return conversations
    
    def _generate_issue_body(self, title, domain):
        """Generate a realistic issue body based on title and domain"""
        body = f"## Problem\nI'm experiencing an issue with {title.lower()}.\n\n"
        
        if domain == "programming":
            body += f"When I try to implement this feature, I'm running into the following error:\n\n"
            body += "```\nError: Something went wrong with the implementation\nStackTrace: Line 42, function process()\n```\n\n"
            body += "## Expected Behavior\nI expected the code to process the data correctly and return the result.\n\n"
            body += "## Actual Behavior\nInstead, it's failing with the error above and not completing the operation.\n\n"
            body += "## Environment\n- Language version: 3.9.2\n- Framework version: 2.1.0\n- OS: Ubuntu 24.04\n\n"
            body += "Has anyone else encountered this? Any suggestions would be appreciated."
            
        elif domain == "devops":
            body += f"Our deployment pipeline is failing with the following error when attempting to {title.lower()}:\n\n"
            body += "```\nError: Configuration validation failed\nDetails: Invalid resource specification\n```\n\n"
            body += "## Steps to Reproduce\n1. Run the deployment script\n2. Wait for the provisioning step\n3. Observe the error in the logs\n\n"
            body += "## Infrastructure Details\n- Cloud Provider: AWS\n- Kubernetes version: 1.24\n- Terraform version: 1.3.2\n\n"
            body += "Has anyone found a workaround for this issue?"
            
        else:  # security
            body += f"I've identified a potential security vulnerability related to {title.lower()}.\n\n"
            body += "## Vulnerability Details\nWhen a user performs the following actions:\n"
            body += "1. Log in to the application\n2. Navigate to the profile page\n3. Submit a specially crafted request\n\n"
            body += "The system allows unauthorized access to protected resources.\n\n"
            body += "## Impact\nThis could potentially allow an attacker to access sensitive information or perform unauthorized actions.\n\n"
            body += "I've tested this in a controlled environment and can provide more details if needed."
            
        return body
    
    def _generate_comment(self, title, domain, position, role):
        """Generate a realistic comment based on context"""
        if position == 0 and role == 'assistant':  # First response
            responses = [
                f"Thanks for reporting this issue with {title.lower()}. Could you provide more information about your setup? Specifically, I'm interested in:\n\n1. Are you using the latest version?\n2. Can you share a minimal reproducible example?\n3. Have you checked the logs for additional error messages?",
                f"I've seen similar issues with {title.lower()} before. It's usually related to configuration or environment setup. Can you confirm whether you've tried clearing the cache and restarting the service?",
                f"This looks like it might be related to a known issue we're tracking. Before I dig deeper, could you confirm if the problem occurs consistently or only under specific conditions?"
            ]
            return random.choice(responses)
            
        elif position == 1 and role == 'user':  # User's first reply
            responses = [
                "Thanks for the quick response! I'm using the latest version, and yes, the issue occurs consistently. Here's a minimal example:\n\n```\n// Example code that reproduces the issue\nfunction test() {\n  // The error happens here\n}\n```\n\nI've checked the logs and found these additional messages: `Warning: Resource limit approaching`",
                "I've tried clearing the cache and restarting, but the issue persists. The problem seems to happen every time I try to run this specific operation, not just occasionally. Let me know what other information would be helpful.",
                "I should clarify that this is happening in our production environment but not in development, which makes me think it might be related to scale or load. The error logs don't show anything beyond what I shared initially."
            ]
            return random.choice(responses)
            
        elif position == 2 and role == 'assistant':  # Second response
            responses = [
                "Thank you for the additional details. Based on what you've shared, I think I understand the issue now. It looks like you're running into a resource limitation problem. Here are a few things you can try:\n\n1. Increase the allocated memory in your configuration\n2. Implement batching to process data in smaller chunks\n3. Add proper error handling for resource constraints\n\nLet me know if any of these help.",
                "I see the problem now. This is likely caused by a regression introduced in the latest release. We have a workaround while we prepare a fix. Can you try the following:\n\n```\n// Workaround code\nfunction patchedVersion() {\n  // This avoids the problematic code path\n}\n```\n\nPlease let me know if this resolves the issue for you.",
                "After looking at your example, I think there's a configuration issue. The error happens because the system is trying to access a resource that isn't properly initialized. Try adding this to your setup:\n\n```\n// Configuration fix\nconfig.preloadResources = true;\nconfig.timeout = 30000;\n```"
            ]
            return random.choice(responses)
            
        else:  # Later in the conversation
            if role == 'user':
                responses = [
                    "I tried implementing your suggestion, but I'm still seeing the issue. The error now happens less frequently, but it's not completely resolved. Is there anything else I should try?",
                    "That worked partially! The main error is gone, but now I'm seeing a different warning in the logs. It says: `Warning: Deprecated method called`. Is this related to the original problem?",
                    "Thanks for the help so far. I implemented your suggestions and it's working much better now. One follow-up question: is there a way to make this solution more efficient? Our system is under heavy load."
                ]
                return random.choice(responses)
            else:  # assistant
                responses = [
                    "Let's try a different approach then. Based on your feedback, it sounds like we're dealing with a deeper issue. Can you try this more comprehensive solution?\n\n```\n// Better solution\nfunction completeOverhaul() {\n  // This addresses both the primary and secondary issues\n}\n```\n\nThis should address both problems you're experiencing.",
                    "Good progress! The warning you're seeing is expected during the transition. It should go away after you update to the latest patch version, which was released yesterday specifically to address this issue chain.",
                    "I'm glad it's working better! For efficiency under heavy load, I recommend implementing the following optimizations:\n\n1. Add a caching layer to reduce redundant processing\n2. Consider scaling horizontally for better distribution\n3. Implement back-pressure mechanisms to prevent overload\n\nThese should help maintain performance even with increased traffic."
                ]
                return random.choice(responses)
    
    def convert_to_qa_format(self, conversations):
        """
        Convert conversations to question-answer format.
        
        Args:
            conversations (list): List of conversations
            
        Returns:
            list: QA pairs
        """
        qa_pairs = []
        
        for conversation in conversations:
            # Process each conversation
            context = []
            
            for i in range(len(conversation['exchanges']) - 1):
                current = conversation['exchanges'][i]
                next_exchange = conversation['exchanges'][i+1]
                
                # Skip if current is not user or next is not assistant
                if current['role'] != 'user' or next_exchange['role'] != 'assistant':
                    continue
                
                # Create QA pair
                qa_pair = {
                    "question": current['content'],
                    "answer": next_exchange['content'],
                    "context": context.copy(),  # Previous exchanges as context
                    "title": conversation['title'],
                    "repo": conversation['repo'],
                    "domain": conversation.get('domain', ''),
                    "url": conversation['url']
                }
                
                qa_pairs.append(qa_pair)
                
                # Update context for next pair
                context.append({
                    "role": current['role'],
                    "content": current['content']
                })
                context.append({
                    "role": next_exchange['role'],
                    "content": next_exchange['content']
                })
        
        return qa_pairs
    
    def save_data(self, conversations, qa_pairs, output_file=None):
        """
        Save processed data to files.
        
        Args:
            conversations (list): Raw conversations
            qa_pairs (list): Processed QA pairs
            output_file (str, optional): Base output file path
            
        Returns:
            tuple: Paths to saved files (conversations, qa_pairs)
        """
        if not output_file:
            base_path = self.output_dir / "github_conversations"
            conversations_file = f"{base_path}_raw.json"
            qa_file = f"{base_path}_qa.json"
        else:
            base_path = Path(output_file).with_suffix('')
            conversations_file = f"{base_path}_raw.json"
            qa_file = f"{base_path}_qa.json"
            
        # Create parent directory if it doesn't exist
        Path(conversations_file).parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save conversations
            with open(conversations_file, 'w', encoding='utf-8') as f:
                json.dump(conversations, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Saved {len(conversations)} conversations to {conversations_file}")
            
            # Save QA pairs
            with open(qa_file, 'w', encoding='utf-8') as f:
                json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Saved {len(qa_pairs)} QA pairs to {qa_file}")
            
            return str(conversations_file), str(qa_file)
            
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            return None, None

def main(sample_size=2000, output_dir="./Datasets", cache_dir=None, github_token=None):
    """
    Process GitHub issues and save to file.
    
    Args:
        sample_size (int): Number of conversations to extract
        output_dir (str): Output directory
        cache_dir (str, optional): Cache directory
        github_token (str, optional): GitHub API token
        
    Returns:
        tuple: Paths to saved files (conversations, qa_pairs)
    """
    processor = GitHubIssueProcessor(
        output_dir=output_dir,
        cache_dir=cache_dir,
        github_token=github_token,
        sample_size=sample_size
    )
    
    conversations = processor.process_repos()
    qa_pairs = processor.convert_to_qa_format(conversations)
    
    conversations_file, qa_file = processor.save_data(
        conversations, 
        qa_pairs,
        os.path.join(output_dir, "github_conversations.json")
    )
    
    return qa_file

if __name__ == "__main__":
    main()
