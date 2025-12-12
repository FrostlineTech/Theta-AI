"""
Generate tutorial-style content for Theta AI training.
Creates step-by-step guides for various technical topics.
"""

import json
import random
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_python_tutorials(count=100):
    """Generate Python tutorial Q&A pairs"""
    python_topics = [
        "basic Python syntax", "Python functions", "Python classes", "file handling in Python",
        "Python web development with Flask", "Python data analysis with pandas",
        "Python testing", "Python packages", "Python virtual environments",
        "asynchronous programming in Python", "Python decorators", "Python context managers",
        "Python for data science", "Django web development", "Python APIs"
    ]
    
    tutorials = []
    
    for _ in range(count):
        topic = random.choice(python_topics)
        
        question = f"How do I learn {topic}? Please provide a step-by-step guide."
        
        answer = f"# Step-by-Step Guide to Learning {topic}\n\n"
        
        if "basic Python syntax" in topic:
            answer += """1. **Install Python**: Start by downloading and installing Python from python.org. Be sure to check the "Add to PATH" option during installation.

2. **Set up an IDE or Text Editor**: Install an editor like VSCode, PyCharm, or Sublime Text to write your code efficiently.

3. **Learn Basic Types**: Understand Python's fundamental data types: integers, floats, strings, booleans, lists, tuples, dictionaries, and sets.

4. **Master Control Flow**: Learn about if-else statements, for loops, while loops, and how to use logical operators.

5. **Practice with Simple Programs**: Write programs that calculate basic math, manipulate strings, or process user input.

6. **Explore Built-in Functions**: Familiarize yourself with Python's built-in functions like print(), len(), range(), and type().

7. **Understand List Comprehensions**: Learn this powerful Python feature for creating lists in a concise way.

8. **Study Error Handling**: Learn to use try-except blocks to handle exceptions gracefully.

Would you like me to elaborate on any specific step or provide simple code examples to help you get started?"""
            
        elif "Python web development with Flask" in topic:
            answer += """1. **Install Flask**: Begin by installing Flask using pip: `pip install flask`.

2. **Create Your First Flask App**: Write a simple "Hello World" application to understand the basic structure.
   ```python
   from flask import Flask
   app = Flask(__name__)
   
   @app.route('/')
   def hello_world():
       return 'Hello, World!'
   
   if __name__ == '__main__':
       app.run(debug=True)
   ```

3. **Learn About Routes**: Understand how Flask's routing system works to map URLs to Python functions.

4. **Templates with Jinja2**: Learn to use Flask's template engine to create dynamic HTML pages.

5. **Handle Forms**: Implement form handling to collect and process user input.

6. **Connect to Databases**: Learn to integrate databases like SQLite or PostgreSQL using Flask-SQLAlchemy.

7. **Implement User Authentication**: Add login functionality with Flask-Login.

8. **Structure Larger Applications**: Learn the Blueprint system for organizing larger Flask applications.

9. **Deploy Your Application**: Learn to deploy Flask apps using platforms like Heroku, PythonAnywhere, or AWS.

This learning path takes you from basics to deployment. Would you like me to expand on any specific area of Flask development?"""
            
        elif "Python for data science" in topic:
            answer += """1. **Set Up Your Environment**: Install Python and key libraries: NumPy, pandas, Matplotlib, scikit-learn, and Jupyter Notebooks.

2. **Master Data Manipulation**: Learn to load, clean, transform, and merge data using pandas DataFrames.

3. **Visualize Data**: Practice creating insightful visualizations with Matplotlib and Seaborn.

4. **Statistical Analysis**: Learn basic statistical concepts and how to apply them using SciPy and StatsModels.

5. **Machine Learning Foundations**: Understand fundamental ML concepts like supervised vs. unsupervised learning, training/testing splits, and model evaluation.

6. **Implement Basic ML Models**: Start with simple algorithms like linear regression, logistic regression, and k-means clustering using scikit-learn.

7. **Feature Engineering**: Learn techniques for creating and selecting features to improve model performance.

8. **Advanced ML and Deep Learning**: Explore more complex models with libraries like TensorFlow or PyTorch.

9. **Build End-to-End Projects**: Apply your skills to complete data science projects from data collection to insights or predictions.

Would you like a sample data analysis code snippet to get started with pandas?"""
        
        else:
            # Generate generic tutorial steps for other topics
            steps = random.randint(5, 10)
            for step in range(1, steps + 1):
                answer += f"{step}. **{random.choice(['Learn', 'Understand', 'Master', 'Practice', 'Study', 'Implement'])} {topic.split(' ')[0].title()} {random.choice(['Fundamentals', 'Basics', 'Concepts', 'Principles', 'Elements', 'Components'])}" + f" {step}**: "
                answer += f"{random.choice(['This step involves', 'Here you will learn about', 'Focus on understanding', 'Explore how to use'])} key aspects of {topic} including important {random.choice(['techniques', 'methods', 'approaches', 'principles', 'concepts'])}.\n\n"
            
            answer += f"\nFollowing these steps will help you build a solid foundation in {topic}. Would you like me to provide more specific details on any of these steps?"
        
        tutorials.append({
            "question": question,
            "answer": answer
        })
    
    return tutorials

def generate_web_development_tutorials(count=100):
    """Generate web development tutorial Q&A pairs"""
    web_topics = [
        "HTML and CSS basics", "JavaScript fundamentals", "responsive web design",
        "CSS frameworks like Bootstrap", "modern JavaScript frameworks", "React basics",
        "Angular development", "Vue.js applications", "web accessibility",
        "progressive web apps", "serverless web applications", "API integration",
        "front-end performance optimization", "CSS animations", "web security"
    ]
    
    tutorials = []
    
    for _ in range(count):
        topic = random.choice(web_topics)
        
        question = f"I want to learn {topic}. Can you provide a detailed tutorial?"
        
        answer = f"# Complete Tutorial: {topic}\n\n"
        
        if "HTML and CSS basics" in topic:
            answer += """1. **Set Up Your Development Environment**:
   - Install a text editor like VSCode, Sublime Text, or Atom
   - Set up browser developer tools (Chrome DevTools or Firefox Developer Tools)

2. **Learn HTML Structure**:
   - Start with the basic HTML document structure (DOCTYPE, html, head, body)
   - Understand semantic HTML elements (header, footer, nav, section, article)
   - Practice creating forms, tables, and lists

3. **Master CSS Fundamentals**:
   - Learn about selectors, properties, and values
   - Understand the CSS box model (margin, border, padding, content)
   - Master CSS layout techniques (flexbox and grid)
   - Learn about responsive design with media queries

4. **Build Simple Projects**:
   - Create a personal profile page
   - Build a simple multi-page website
   - Implement a responsive layout from scratch

5. **Learn CSS Preprocessing**:
   - Explore SASS or LESS to enhance your CSS workflow
   - Understand variables, mixins, and nesting

Would you like me to provide HTML and CSS code examples to get started?"""
            
        elif "React basics" in topic:
            answer += """1. **Set Up Your React Environment**:
   - Install Node.js and npm
   - Create a new React app with Create React App: `npx create-react-app my-app`
   - Understand the project structure and available scripts

2. **Learn React Fundamentals**:
   - Understand components and JSX syntax
   - Learn about props for passing data between components
   - Master state management with useState hook
   - Implement event handling in React

3. **Component Lifecycle and Effects**:
   - Learn about the useEffect hook for side effects
   - Understand component mounting, updating, and unmounting
   - Practice cleaning up effects to prevent memory leaks

4. **React Router**:
   - Set up client-side routing with React Router
   - Create routes with parameters and nested routes
   - Implement navigation guards and redirects

5. **State Management**:
   - Start with React Context for simple state management
   - Learn about reducers with useReducer
   - Consider Redux or other state management libraries for complex apps

6. **Building a Complete Application**:
   - Create a small project like a todo list or weather app
   - Implement CRUD operations with an API
   - Add styling with CSS-in-JS or a component library

Here's a simple React component example to get started:

```jsx
import { useState } from 'react';

function Counter() {
  const [count, setCount] = useState(0);
  
  return (
    <div>
      <h1>Count: {count}</h1>
      <button onClick={() => setCount(count + 1)}>Increment</button>
      <button onClick={() => setCount(count - 1)}>Decrement</button>
    </div>
  );
}

export default Counter;
```

Would you like me to expand on any specific aspect of React development?"""
        
        else:
            # Generate generic web development tutorial steps
            steps = random.randint(5, 9)
            for step in range(1, steps + 1):
                answer += f"{step}. **{random.choice(['Learn', 'Master', 'Understand', 'Implement', 'Practice'])} {topic.split(' ')[0].title()} {random.choice(['Core Concepts', 'Fundamentals', 'Key Principles', 'Best Practices'])}**:\n"
                answer += f"   - {random.choice(['Start by', 'Begin with', 'Focus on'])} understanding the {random.choice(['basic', 'essential', 'fundamental', 'core'])} components\n"
                answer += f"   - {random.choice(['Practice', 'Implement', 'Work with', 'Experiment with'])} different {random.choice(['approaches', 'techniques', 'methods', 'strategies'])}\n"
                answer += f"   - {random.choice(['Build', 'Create', 'Develop', 'Design'])} a simple project to apply your knowledge\n\n"
            
            answer += f"\nBy following these steps, you'll develop a strong foundation in {topic}. Would you like me to provide more specific code examples or resources?"
        
        tutorials.append({
            "question": question,
            "answer": answer
        })
    
    return tutorials

def generate_cybersecurity_tutorials(count=100):
    """Generate cybersecurity tutorial Q&A pairs"""
    security_topics = [
        "network security basics", "penetration testing", "security auditing",
        "encryption and cryptography", "secure coding practices", "malware analysis",
        "incident response", "security frameworks", "security compliance",
        "web application security", "wireless security", "security monitoring",
        "DevSecOps", "cloud security", "mobile security"
    ]
    
    tutorials = []
    
    for _ in range(count):
        topic = random.choice(security_topics)
        
        question = f"How do I implement {topic}? Please provide a comprehensive guide."
        
        answer = f"# Comprehensive Guide to {topic}\n\n"
        
        if "web application security" in topic:
            answer += """1. **Understand the OWASP Top 10**:
   - Familiarize yourself with the most critical web application security risks
   - Study each vulnerability type, including injection flaws, broken authentication, and XSS

2. **Implement Secure Authentication**:
   - Use strong password policies and multi-factor authentication
   - Implement proper session management with secure cookies
   - Consider OAuth 2.0 or OpenID Connect for federated authentication

3. **Secure Data Transmission**:
   - Always use HTTPS with proper TLS configuration
   - Implement HTTP security headers (Content-Security-Policy, X-XSS-Protection, etc.)
   - Use certificate pinning for mobile applications

4. **Input Validation and Sanitization**:
   - Validate all user inputs on both client and server sides
   - Implement context-specific output encoding
   - Use parameterized queries to prevent SQL injection

5. **Access Control Implementation**:
   - Follow the principle of least privilege
   - Implement proper authorization checks at the resource level
   - Use role-based access control (RBAC) or attribute-based access control (ABAC)

6. **Security Testing**:
   - Perform regular vulnerability scanning with tools like OWASP ZAP or Burp Suite
   - Conduct penetration testing for critical applications
   - Implement continuous security testing in your CI/CD pipeline

7. **Security Monitoring and Incident Response**:
   - Set up logging for security events
   - Implement intrusion detection/prevention systems
   - Create an incident response plan specific to web applications

This guide covers the essentials of web application security. Would you like me to elaborate on any specific area?"""
            
        elif "encryption and cryptography" in topic:
            answer += """1. **Understand Cryptographic Concepts**:
   - Learn the difference between encryption, hashing, and encoding
   - Understand symmetric vs. asymmetric encryption
   - Study cryptographic protocols like TLS/SSL, PGP, and SSH

2. **Implement Symmetric Encryption**:
   - Choose a secure algorithm like AES-256 for data encryption
   - Properly manage encryption keys using a key management system
   - Example in Python using the cryptography library:
     ```python
     from cryptography.fernet import Fernet
     key = Fernet.generate_key()
     cipher = Fernet(key)
     encrypted_data = cipher.encrypt(b"sensitive data")
     decrypted_data = cipher.decrypt(encrypted_data)
     ```

3. **Apply Public Key Cryptography**:
   - Implement RSA or ECC for asymmetric encryption
   - Use digital signatures to verify data integrity and authenticity
   - Set up a public key infrastructure (PKI) for larger systems

4. **Secure Password Storage**:
   - Never store passwords in plaintext
   - Use strong adaptive hashing algorithms like bcrypt, Argon2, or PBKDF2
   - Add unique salts to each password hash to prevent rainbow table attacks

5. **Transport Layer Security**:
   - Configure web servers with modern TLS protocols (TLS 1.2+)
   - Select secure cipher suites and parameters
   - Regularly audit and update your TLS configuration

6. **End-to-End Encryption**:
   - Implement E2EE for messaging or file sharing applications
   - Consider the Signal Protocol as a proven E2EE solution
   - Ensure proper key verification mechanisms

7. **Cryptography Best Practices**:
   - Never implement your own cryptographic algorithms
   - Regularly update cryptographic libraries and protocols
   - Plan for cryptographic agility to adapt to future threats

Would you like more detailed implementation guidance for any specific aspect of encryption or cryptography?"""
        
        else:
            # Generate generic cybersecurity tutorial steps
            steps = random.randint(6, 10)
            for step in range(1, steps + 1):
                answer += f"{step}. **{random.choice(['Implement', 'Establish', 'Develop', 'Set up'])} {random.choice(['Robust', 'Effective', 'Comprehensive', 'Strong'])} {topic.split(' ')[0].title()} {random.choice(['Controls', 'Measures', 'Protocols', 'Standards'])}**:\n"
                answer += f"   - {random.choice(['Conduct', 'Perform', 'Execute', 'Complete'])} a thorough {random.choice(['assessment', 'analysis', 'evaluation', 'review'])} of your current environment\n"
                answer += f"   - {random.choice(['Identify', 'Determine', 'Recognize', 'Pinpoint'])} potential {random.choice(['vulnerabilities', 'weaknesses', 'risks', 'threats'])}\n"
                answer += f"   - {random.choice(['Implement', 'Deploy', 'Establish', 'Institute'])} appropriate {random.choice(['controls', 'safeguards', 'countermeasures', 'protections'])}\n\n"
            
            answer += f"\nImplementing {topic} is an ongoing process that requires regular updates and assessments. Would you like me to provide more specific recommendations for your environment?"
        
        tutorials.append({
            "question": question,
            "answer": answer
        })
    
    return tutorials

def generate_devops_tutorials(count=100):
    """Generate DevOps tutorial Q&A pairs"""
    devops_topics = [
        "CI/CD pipelines", "Docker containers", "Kubernetes orchestration",
        "Infrastructure as Code", "Terraform", "Ansible automation",
        "monitoring and observability", "GitOps workflow", "microservices architecture",
        "site reliability engineering", "DevOps metrics", "chaos engineering",
        "cloud-native applications", "serverless architecture", "container security"
    ]
    
    tutorials = []
    
    for _ in range(count):
        topic = random.choice(devops_topics)
        
        question = f"How do I set up {topic}? Please provide a detailed guide."
        
        answer = f"# Setting Up {topic}: A Comprehensive Guide\n\n"
        
        if "CI/CD pipelines" in topic:
            answer += """1. **Choose Your CI/CD Tools**:
   - Select a CI/CD platform (Jenkins, GitLab CI, GitHub Actions, CircleCI, etc.)
   - Identify version control system integration needs
   - Determine deployment targets (servers, Kubernetes, cloud platforms)

2. **Set Up Version Control Integration**:
   - Configure webhooks to trigger pipelines on code changes
   - Establish branch protection rules for main/production branches
   - Implement a branching strategy (GitFlow, trunk-based development, etc.)

3. **Create Basic Pipeline Stages**:
   - Build: Compile code and create artifacts
   - Test: Run unit tests, integration tests, and code quality checks
   - Deploy: Push artifacts to staging/production environments
   - Example GitHub Actions workflow:
     ```yaml
     name: CI/CD Pipeline
     on:
       push:
         branches: [ main ]
     jobs:
       build-and-test:
         runs-on: ubuntu-latest
         steps:
           - uses: actions/checkout@v2
           - name: Set up environment
             run: npm install
           - name: Run tests
             run: npm test
           - name: Build
             run: npm run build
           - name: Deploy
             if: success()
             run: ./deploy.sh
     ```

4. **Implement Automated Testing**:
   - Add unit tests, integration tests, and end-to-end tests
   - Configure code quality tools (linters, formatters, static analyzers)
   - Implement test coverage requirements

5. **Set Up Deployment Strategies**:
   - Configure blue-green deployments or canary releases
   - Implement rollback mechanisms for failed deployments
   - Add deployment approvals for production environments

6. **Add Monitoring and Notifications**:
   - Integrate pipeline status with chat tools (Slack, Microsoft Teams)
   - Set up failure alerts and performance metrics
   - Create dashboards for pipeline health and deployment frequency

7. **Optimize Pipeline Performance**:
   - Implement caching strategies for dependencies
   - Use parallel jobs for independent tasks
   - Configure self-hosted runners for specific needs

Would you like me to provide a more specific CI/CD pipeline example for a particular technology stack?"""
            
        elif "Infrastructure as Code" in topic:
            answer += """1. **Select an Infrastructure as Code (IaC) Tool**:
   - Choose between declarative tools (Terraform, CloudFormation, ARM templates)
   - Or imperative tools (Ansible, Chef, Puppet)
   - Consider your cloud provider's native offerings

2. **Set Up Your Environment**:
   - Install the chosen IaC tool (e.g., Terraform)
   - Configure provider authentication
   - Set up a state management backend (e.g., S3 bucket for Terraform state)

3. **Define Your Infrastructure**:
   - Create configuration files to describe your resources
   - Example Terraform configuration for an AWS VPC:
     ```hcl
     provider "aws" {
       region = "us-west-2"
     }
     
     resource "aws_vpc" "main" {
       cidr_block = "10.0.0.0/16"
       tags = {
         Name = "MainVPC"
       }
     }
     
     resource "aws_subnet" "public" {
       vpc_id     = aws_vpc.main.id
       cidr_block = "10.0.1.0/24"
       tags = {
         Name = "Public Subnet"
       }
     }
     ```

4. **Implement Modularity and Reusability**:
   - Structure your code into modules for reusability
   - Parameterize configurations for different environments
   - Use variables and outputs to enhance flexibility

5. **Establish Deployment Workflow**:
   - Implement plan and apply stages in your pipeline
   - Set up approval processes for infrastructure changes
   - Create a consistent workflow: plan → review → apply

6. **Add Validation and Testing**:
   - Implement static code analysis for IaC files
   - Create automated tests for infrastructure
   - Use tools like Terratest or kitchen-terraform for infrastructure testing

7. **Set Up Proper Security Practices**:
   - Manage secrets securely (avoid hardcoding)
   - Implement least privilege access
   - Regularly audit your infrastructure code

8. **Implement Drift Detection**:
   - Set up processes to detect manual changes to infrastructure
   - Configure automated drift remediation where appropriate

Would you like me to provide more specific examples or guidance for a particular cloud provider?"""
        
        else:
            # Generate generic DevOps tutorial steps
            steps = random.randint(6, 10)
            for step in range(1, steps + 1):
                answer += f"{step}. **{random.choice(['Set Up', 'Configure', 'Implement', 'Establish'])} {random.choice(['Your', 'The', 'A Complete', 'An Effective'])} {topic.split(' ')[0].title()} {random.choice(['Environment', 'Infrastructure', 'Platform', 'System'])}**:\n"
                answer += f"   - {random.choice(['Install and configure', 'Set up', 'Prepare', 'Deploy'])} the necessary {random.choice(['tools', 'components', 'services', 'resources'])}\n"
                answer += f"   - {random.choice(['Define', 'Establish', 'Create', 'Document'])} your {random.choice(['workflow', 'architecture', 'processes', 'policies'])}\n"
                answer += f"   - {random.choice(['Implement', 'Add', 'Configure', 'Enable'])} automated {random.choice(['testing', 'monitoring', 'deployment', 'scaling'])}\n\n"
            
            answer += f"\nBy following these steps, you'll have a fully functional {topic} setup. Would you like me to provide specific code examples or configurations for your environment?"
        
        tutorials.append({
            "question": question,
            "answer": answer
        })
    
    return tutorials

def generate_tutorials(count=1000):
    """
    Generate tutorial-style content for various technical topics.
    
    Args:
        count: Number of tutorials to generate
        
    Returns:
        list: Generated tutorial Q&A pairs
    """
    # Distribute count across different categories
    category_count = count // 4  # 4 categories
    remaining = count - (category_count * 3)  # Ensure we generate exactly 'count' tutorials
    
    # Generate tutorials for each category
    python_tutorials = generate_python_tutorials(category_count)
    web_tutorials = generate_web_development_tutorials(category_count)
    security_tutorials = generate_cybersecurity_tutorials(category_count)
    devops_tutorials = generate_devops_tutorials(remaining)
    
    # Combine all tutorials
    all_tutorials = python_tutorials + web_tutorials + security_tutorials + devops_tutorials
    
    # Shuffle the combined list
    random.shuffle(all_tutorials)
    
    return all_tutorials

def save_tutorials(count=1000):
    """
    Generate and save tutorial content.
    
    Args:
        count: Number of tutorials to generate
        
    Returns:
        str: Path to the saved dataset
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    datasets_dir = project_root / "Datasets"
    datasets_dir.mkdir(exist_ok=True)
    
    # Generate tutorials
    logger.info(f"Generating {count} tutorial examples...")
    tutorials = generate_tutorials(count)
    
    # Save tutorials
    tutorials_path = datasets_dir / "tutorials.json"
    with open(tutorials_path, 'w', encoding='utf-8') as f:
        json.dump(tutorials, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Created {len(tutorials)} tutorial examples at {tutorials_path}")
    return str(tutorials_path)

def main():
    """Main function to generate and save tutorials."""
    return save_tutorials(1000)

if __name__ == "__main__":
    main()
