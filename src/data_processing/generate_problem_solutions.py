"""
Generate problem-solution pairs for Theta AI training.
Creates Q&A pairs for common technical issues and their solutions.
"""

import json
import random
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_domain_issues(domain):
    """Return domain-specific issues"""
    issues = {
        "programming": ["syntax error", "runtime error", "compile error", "memory leak", 
                       "null pointer exception", "undefined variable", "import error", 
                       "type error", "index out of range", "stack overflow"],
        
        "web development": ["CORS error", "responsive layout issue", "browser compatibility problem", 
                           "slow loading speed", "API connection failure", "JavaScript error", 
                           "CSS not applying", "form submission error", "authentication failure",
                           "caching issue"],
        
        "system administration": ["disk space error", "permission denied", "service not starting", 
                                 "network connectivity issue", "high CPU usage", "memory leak", 
                                 "log file errors", "backup failure", "security vulnerability",
                                 "slow system performance"],
        
        "database": ["connection timeout", "query performance issue", "deadlock", "data corruption", 
                    "replication lag", "index fragmentation", "backup failure", "permission error",
                    "constraint violation", "schema migration problem"],
                    
        "networking": ["connection timeout", "DNS resolution failure", "packet loss", 
                      "high latency", "firewall blocking", "IP conflict", "route not found",
                      "certificate error", "VPN connection issue", "bandwidth limitation"]
    }
    return issues.get(domain, ["generic issue"])

def get_domain_actions(domain):
    """Return domain-specific actions"""
    actions = {
        "programming": ["compile the code", "run the application", "import libraries", 
                       "handle exceptions", "debug the function", "optimize performance",
                       "refactor the class", "unit test the module", "parse input data",
                       "implement an interface"],
        
        "web development": ["load the page", "submit the form", "update the DOM", 
                           "fetch API data", "authenticate users", "deploy the site",
                           "implement responsive design", "optimize images", 
                           "set up a CDN", "configure server-side rendering"],
        
        "system administration": ["configure the server", "install updates", 
                                 "manage user permissions", "set up monitoring", 
                                 "perform backups", "migrate services", "analyze logs",
                                 "secure the system", "allocate resources", "schedule jobs"],
        
        "database": ["query the database", "optimize indexes", "migrate schemas", 
                    "set up replication", "perform backups", "manage connections",
                    "implement transactions", "scale horizontally", "shard data",
                    "tune performance"],
                    
        "networking": ["configure the router", "set up VPN", "secure wireless networks", 
                      "troubleshoot connectivity", "monitor network traffic", 
                      "implement QoS", "set up firewalls", "design network topology",
                      "configure DNS", "establish VLANs"]
    }
    return actions.get(domain, ["perform basic operations"])

def generate_problem_solutions(count=1000):
    """
    Generate problem-solution pairs for technical issues.
    
    Args:
        count: Number of problem-solution pairs to generate
        
    Returns:
        list: Generated problem-solution pairs
    """
    domains = ["programming", "web development", "system administration", "database", "networking"]
    
    problem_templates = [
        "I'm getting a {} error when trying to {}.",
        "My {} keeps crashing when I {}.",
        "Why does my {} fail to {} properly?",
        "How do I fix {} issues with {}?",
        "I can't get {} to work when {}.",
        "What causes {} problems when attempting to {}?",
        "How can I resolve {} errors that occur during {}?",
        "I need help with {} issues that happen while {}.",
        "What's the solution for {} failures during {}?"
    ]
    
    problems = []
    
    # Generate equal number for each domain
    domain_count = count // len(domains)
    
    for domain in domains:
        domain_specific_issues = get_domain_issues(domain)
        domain_specific_actions = get_domain_actions(domain)
        
        for _ in range(domain_count):
            issue = random.choice(domain_specific_issues)
            action = random.choice(domain_specific_actions)
            
            template = random.choice(problem_templates)
            question = template.format(issue, action)
            
            # Create a solution based on the domain and issue
            if domain == "programming":
                answer = f"""When dealing with {issue} while trying to {action}, follow these troubleshooting steps:

1. **Check your syntax and error messages**: The most common cause of {issue} is incorrect syntax or missing dependencies. Look at the error message carefully as it often points to the exact line and problem.

2. **Verify your environment**: Make sure your development environment is properly configured. Check versions of languages, libraries, and frameworks to ensure compatibility.

3. **Isolate the problem**: Create a minimal reproducible example to identify exactly what's causing the issue. Remove irrelevant code until you have the smallest piece that still shows the problem.

4. **Use debugging tools**: Employ IDE debugging features, logging, or print statements to trace execution flow and variable values.

5. **Search for known issues**: Check documentation and communities like Stack Overflow for similar problems. This particular error often occurs when {random.choice(['there are scope issues', 'dependencies are misconfigured', 'types are mismatched', 'resources aren\'t properly managed'])}.

The most likely solution is to {random.choice(['update your dependencies', 'check your variable scope', 'implement proper error handling', 'review your logic flow', 'correctly initialize your variables'])}.

Would you like me to provide a code example showing how to properly {action} while avoiding this error?"""
            
            elif domain == "web development":
                answer = f"""To resolve {issue} issues when trying to {action}, follow these steps:

1. **Check browser console**: Open your browser developer tools (F12) and look for error messages in the console. This will often give you specific details about what's failing.

2. **Verify cross-browser compatibility**: Test your site on multiple browsers to see if the issue is browser-specific. {issue} problems often vary across different browsers.

3. **Inspect network requests**: Use the Network tab in developer tools to verify that all resources are loading correctly and identify any failed requests.

4. **Validate your HTML/CSS/JavaScript**: Use validation tools to ensure your code follows standards and best practices.

5. **Check for responsive design issues**: If you're having display problems, test your site at different viewport sizes using the responsive design mode in developer tools.

Common solutions include {random.choice(['updating your JavaScript code to handle browser differences', 'implementing proper CORS headers on your server', 'optimizing resource loading sequence', 'fixing CSS specificity issues'])}.

For {issue} specifically, you may need to {random.choice(['add polyfills for older browsers', 'implement proper error handling in your AJAX calls', 'optimize your CSS selectors', 'fix your media queries for better responsiveness'])}.

Would you like me to provide a code snippet showing how to properly {action} while avoiding {issue} problems?"""
            
            else:
                # Generic solution format for other domains
                answer = f"""When experiencing {issue} while trying to {action}, follow this troubleshooting approach:

1. **Identify the exact error**: Look for specific error messages in logs or system output that can help pinpoint the cause of the {issue}.

2. **Check system status**: Verify that all required services and dependencies are running properly and have sufficient resources.

3. **Review recent changes**: Consider any recent updates, configurations, or environmental changes that might have triggered this issue.

4. **Isolate the problem**: Determine if the issue is isolated to a specific component or if it affects the entire system.

5. **Apply standard fixes**: For {issue} in {domain}, common solutions include {random.choice(['updating configuration files', 'restarting services', 'clearing caches', 'applying patches', 'checking permissions'])}.

The most effective solution typically involves {random.choice(['properly configuring your environment', 'implementing best practices for resource management', 'following security protocols', 'ensuring consistent configuration across systems'])}.

Have you tried {random.choice(['checking the logs for specific error codes', 'verifying network connectivity', 'testing with minimal configuration', 'updating to the latest version'])}? This often resolves many {issue} problems when {action}.

Would you like more specific guidance based on your environment?"""
            
            problems.append({
                "question": question,
                "answer": answer
            })
    
    # Add any remaining count
    remaining = count - (domain_count * len(domains))
    for _ in range(remaining):
        domain = random.choice(domains)
        issue = random.choice(get_domain_issues(domain))
        action = random.choice(get_domain_actions(domain))
        template = random.choice(problem_templates)
        question = template.format(issue, action)
        
        # Generic answer for remaining items
        answer = f"""To resolve {issue} when {action}, follow these steps:

1. **Diagnose the specific issue**: Carefully examine error logs and messages to understand exactly what's happening.

2. **Check documentation**: Refer to official documentation for known issues related to {issue} when {action}.

3. **Verify prerequisites**: Ensure all requirements and dependencies are properly installed and configured.

4. **Test in isolation**: Try to reproduce the issue in a simplified environment to eliminate other variables.

5. **Apply recommended solutions**: For this specific problem, try {random.choice(['updating to the latest version', 'checking configuration settings', 'verifying permissions', 'restarting the service'])}.

If the issue persists, consider {random.choice(['reaching out to community forums', 'checking GitHub issues', 'consulting with specialists', 'implementing a workaround'])}.

Would you like me to suggest additional troubleshooting steps specific to your environment?"""
        
        problems.append({
            "question": question,
            "answer": answer
        })
    
    # Shuffle the results for variety
    random.shuffle(problems)
    
    return problems

def save_problem_solutions(count=1000):
    """
    Generate and save problem-solution pairs.
    
    Args:
        count: Number of problem-solution pairs to generate
        
    Returns:
        str: Path to the saved dataset
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    datasets_dir = project_root / "Datasets"
    datasets_dir.mkdir(exist_ok=True)
    
    # Generate problem-solutions
    logger.info(f"Generating {count} problem-solution pairs...")
    problems = generate_problem_solutions(count)
    
    # Save problem-solutions
    problems_path = datasets_dir / "problem_solutions.json"
    with open(problems_path, 'w', encoding='utf-8') as f:
        json.dump(problems, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Created {len(problems)} problem-solution pairs at {problems_path}")
    return str(problems_path)

def main():
    """Main function to generate and save problem-solution pairs."""
    return save_problem_solutions(1000)

if __name__ == "__main__":
    main()
