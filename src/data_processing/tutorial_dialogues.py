"""
Technical Tutorial Dialogues Generator for Theta AI.

This module converts technical tutorials into conversational dialogues
for training Theta AI to provide step-by-step guidance.
"""

import os
import re
import json
import logging
import random
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Tutorial topics
TOPICS = {
    "web_development": [
        "Setting up a React project", "Building a REST API with Express", 
        "Creating a responsive website", "Implementing authentication with JWT",
        "Optimizing website performance", "Developing a single-page application"
    ],
    "devops": [
        "Setting up CI/CD with GitHub Actions", "Deploying to Kubernetes", 
        "Configuring Nginx as a reverse proxy", "Setting up Docker containers",
        "Implementing infrastructure as code with Terraform", "Monitoring with Prometheus and Grafana"
    ],
    "data_science": [
        "Data cleaning with pandas", "Building machine learning models with scikit-learn",
        "Creating data visualizations with matplotlib", "Setting up a Jupyter notebook environment",
        "Feature engineering techniques", "Implementing natural language processing"
    ],
    "cybersecurity": [
        "Setting up a secure server", "Implementing proper authentication", 
        "Conducting a security audit", "Configuring a firewall",
        "Implementing encryption", "Setting up intrusion detection"
    ]
}

class TutorialDialogueConverter:
    """
    Converts tutorials into conversational dialogues.
    """
    
    def __init__(self, output_dir, sample_size=1000):
        """
        Initialize the converter.
        
        Args:
            output_dir: Directory to save processed data
            sample_size: Number of tutorial dialogues to generate
        """
        self.output_dir = Path(output_dir) if output_dir else Path(".")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_size = sample_size
        
    def generate_tutorial_dialogues(self):
        """
        Generate tutorial dialogues.
        
        Returns:
            list: Generated tutorial dialogues
        """
        dialogues = []
        
        # Generate dialogues for each domain
        topics_per_domain = self.sample_size // len(TOPICS)
        
        for domain, topics in TOPICS.items():
            for _ in range(topics_per_domain):
                # Select a topic
                topic = random.choice(topics)
                
                # Generate a tutorial dialogue
                dialogue = self._generate_dialogue(topic, domain)
                dialogues.append(dialogue)
        
        # If we need more dialogues, generate additional ones
        remaining = self.sample_size - len(dialogues)
        if remaining > 0:
            for _ in range(remaining):
                domain = random.choice(list(TOPICS.keys()))
                topic = random.choice(TOPICS[domain])
                dialogue = self._generate_dialogue(topic, domain)
                dialogues.append(dialogue)
        
        logger.info(f"Generated {len(dialogues)} tutorial dialogues")
        return dialogues
    
    def _generate_dialogue(self, topic, domain):
        """
        Generate a tutorial dialogue for a topic.
        
        Args:
            topic (str): Topic for the tutorial
            domain (str): Domain of the tutorial
            
        Returns:
            dict: Tutorial dialogue
        """
        # Generate steps for the tutorial
        steps = self._generate_tutorial_steps(topic, domain)
        
        # Create dialogue exchanges
        exchanges = []
        
        # Initial question
        exchanges.append({
            "role": "user",
            "content": f"Can you guide me through {topic}?"
        })
        
        # Introduction response
        exchanges.append({
            "role": "assistant",
            "content": f"I'd be happy to guide you through {topic}! This is a {len(steps)}-step process. Let's start with the basics."
        })
        
        # Add exchanges for each step
        for i, step in enumerate(steps, 1):
            # User asking about the step
            if i == 1:
                exchanges.append({
                    "role": "user",
                    "content": "Great, what's the first step?"
                })
            else:
                exchanges.append({
                    "role": "user",
                    "content": random.choice([
                        f"What's step {i}?",
                        "What's next?",
                        "And after that?",
                        "Got it. What should I do next?",
                        "That makes sense. How do I continue?"
                    ])
                })
            
            # Assistant explaining the step
            exchanges.append({
                "role": "assistant",
                "content": step
            })
            
            # Add clarification exchanges occasionally
            if random.random() < 0.3:  # 30% chance
                clarification_question = self._generate_clarification_question(step, domain)
                exchanges.append({
                    "role": "user",
                    "content": clarification_question
                })
                
                clarification_answer = self._generate_clarification_answer(clarification_question, step)
                exchanges.append({
                    "role": "assistant",
                    "content": clarification_answer
                })
        
        # Final exchange
        exchanges.append({
            "role": "user",
            "content": random.choice([
                "That's all the steps?",
                "Is that it? Did we cover everything?",
                "Great! Anything else I should know?",
                "Thanks for the tutorial. Any final tips?"
            ])
        })
        
        exchanges.append({
            "role": "assistant",
            "content": f"Yes, that covers all the steps for {topic}! To recap, we:\n\n" + 
                      "\n".join([f"- Step {i+1}: {step.split('.')[0]}" for i, step in enumerate(steps)]) + 
                      f"\n\nNow you should have a working {topic.lower()} setup. Let me know if you have any questions!"
        })
        
        # Create dialogue object
        dialogue = {
            "topic": topic,
            "domain": domain,
            "num_steps": len(steps),
            "exchanges": exchanges
        }
        
        return dialogue
    
    def _generate_tutorial_steps(self, topic, domain):
        """Generate steps for a tutorial"""
        num_steps = random.randint(4, 8)  # Tutorials have 4-8 steps
        steps = []
        
        if domain == "web_development":
            if "React" in topic:
                steps = [
                    "First, ensure you have Node.js installed. You can check by running `node -v` in your terminal.",
                    "Next, create a new React application by running `npx create-react-app my-app` where 'my-app' is your project name.",
                    "Once installation is complete, navigate to your project directory with `cd my-app`.",
                    "Start the development server with `npm start`. This will open your app in the browser.",
                    "Now you can begin editing the src/App.js file to customize your application.",
                    "To add additional dependencies, use `npm install package-name` or `yarn add package-name`.",
                    "For production builds, run `npm run build` which will create an optimized build in the 'build' folder.",
                    "Finally, you can deploy your application to various hosting platforms like Netlify, Vercel, or GitHub Pages."
                ]
            elif "API" in topic:
                steps = [
                    "Start by creating a new project folder and initializing it with `npm init -y`.",
                    "Install Express by running `npm install express body-parser cors`.",
                    "Create an index.js file and set up the basic Express server:\n```javascript\nconst express = require('express');\nconst app = express();\napp.listen(3000, () => console.log('Server running'));\n```",
                    "Add middleware for JSON parsing and CORS:\n```javascript\napp.use(express.json());\napp.use(require('cors')());\n```",
                    "Define your API routes:\n```javascript\napp.get('/api/items', (req, res) => {\n  // Return items\n});\n```",
                    "Implement CRUD operations for your resources with appropriate HTTP methods.",
                    "Add error handling middleware to catch and process errors.",
                    "Finally, test your API with tools like Postman or curl."
                ]
            else:  # Generic web development
                steps = [
                    "First, create your project structure with HTML, CSS, and JavaScript files.",
                    "Set up the HTML structure with proper semantic elements.",
                    "Create a responsive layout using CSS Grid or Flexbox.",
                    "Add styling to make your website visually appealing.",
                    "Implement JavaScript functionality for interactive elements.",
                    "Test your website on different devices and browsers.",
                    "Optimize performance by minimizing resource sizes and reducing HTTP requests.",
                    "Deploy your website to a hosting provider."
                ]
                
        elif domain == "devops":
            if "CI/CD" in topic:
                steps = [
                    "Start by creating a `.github/workflows` directory in your repository.",
                    "Create a YAML file for your workflow, e.g., `main.yml`.",
                    "Define the events that trigger your workflow, such as push or pull request:\n```yaml\non:\n  push:\n    branches: [ main ]\n  pull_request:\n    branches: [ main ]\n```",
                    "Configure the jobs and steps for your workflow:\n```yaml\njobs:\n  build:\n    runs-on: ubuntu-latest\n    steps:\n    - uses: actions/checkout@v2\n```",
                    "Add steps for building and testing your code.",
                    "Configure deployment steps based on your target environment.",
                    "Add environment secrets for sensitive information.",
                    "Test your workflow by pushing changes to your repository."
                ]
            elif "Kubernetes" in topic:
                steps = [
                    "First, make sure you have kubectl installed and configured to access your cluster.",
                    "Create a deployment YAML file describing your application:\n```yaml\napiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: my-app\nspec:\n  replicas: 3\n```",
                    "Apply the deployment to your cluster: `kubectl apply -f deployment.yaml`",
                    "Create a service to expose your application: `kubectl expose deployment my-app --port=80`",
                    "Verify the deployment and service are running: `kubectl get pods,svc`",
                    "Configure ingress or load balancer for external access.",
                    "Set up monitoring and logging for your application.",
                    "Implement rolling updates for zero-downtime deployments."
                ]
            else:  # Generic DevOps
                steps = [
                    "First, define your infrastructure requirements and architecture.",
                    "Set up version control for your infrastructure code.",
                    "Implement infrastructure as code using tools like Terraform or CloudFormation.",
                    "Create CI/CD pipelines for automated testing and deployment.",
                    "Configure monitoring and alerting for your infrastructure.",
                    "Implement logging and tracing for better observability.",
                    "Set up backup and disaster recovery procedures.",
                    "Document your infrastructure and processes."
                ]
                
        elif domain == "data_science":
            if "pandas" in topic:
                steps = [
                    "First, install pandas using `pip install pandas numpy`.",
                    "Import pandas and load your dataset:\n```python\nimport pandas as pd\ndf = pd.read_csv('data.csv')\n```",
                    "Inspect your data with `df.head()`, `df.info()`, and `df.describe()`.",
                    "Handle missing values: `df.fillna(method='ffill')` or `df.dropna()`.",
                    "Filter and clean your data: `df = df[(df.column > value) & (df.other_column != value)]`.",
                    "Perform data transformations: `df['new_column'] = df['old_column'].apply(function)`.",
                    "Group and aggregate data: `df.groupby('category').agg({'value': 'mean'})`.",
                    "Export your cleaned data: `df.to_csv('cleaned_data.csv', index=False)`."
                ]
            elif "machine learning" in topic:
                steps = [
                    "First, prepare your data by splitting it into features and target: `X, y = data.drop('target', axis=1), data['target']`.",
                    "Split data into training and testing sets: `from sklearn.model_selection import train_test_split; X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)`.",
                    "Select an appropriate model: `from sklearn.ensemble import RandomForestClassifier; model = RandomForestClassifier()`.",
                    "Train your model: `model.fit(X_train, y_train)`.",
                    "Evaluate performance: `from sklearn.metrics import accuracy_score; accuracy_score(y_test, model.predict(X_test))`.",
                    "Tune hyperparameters using grid search or random search.",
                    "Validate your model using cross-validation.",
                    "Deploy your model for predictions."
                ]
            else:  # Generic data science
                steps = [
                    "First, understand the problem and define your objectives.",
                    "Collect and gather relevant data from various sources.",
                    "Clean and preprocess the data to handle missing values and outliers.",
                    "Explore the data through visualization and statistical analysis.",
                    "Feature engineering to create meaningful variables for modeling.",
                    "Build and train models on your prepared dataset.",
                    "Evaluate and compare different models using appropriate metrics.",
                    "Interpret results and communicate findings."
                ]
                
        elif domain == "cybersecurity":
            if "secure server" in topic.lower():
                steps = [
                    "Begin by updating your server: `sudo apt update && sudo apt upgrade -y` (for Ubuntu/Debian).",
                    "Configure a firewall using UFW: `sudo ufw allow 22/tcp` and `sudo ufw enable`.",
                    "Disable root login and set up SSH key authentication in `/etc/ssh/sshd_config`.",
                    "Install and configure fail2ban to prevent brute force attacks.",
                    "Set up automatic security updates.",
                    "Implement proper file permissions and ownership.",
                    "Install and configure intrusion detection system like AIDE or Tripwire.",
                    "Regularly audit your server security and monitor logs."
                ]
            elif "authentication" in topic.lower():
                steps = [
                    "Start by defining your authentication requirements and user model.",
                    "Implement secure password storage using bcrypt or Argon2.",
                    "Set up multi-factor authentication for sensitive operations.",
                    "Create secure session management with proper timeout and renewal.",
                    "Implement proper CSRF protection for forms and API calls.",
                    "Set up secure password reset functionality.",
                    "Add account lockout after failed login attempts.",
                    "Regularly audit authentication logs for suspicious activities."
                ]
            else:  # Generic cybersecurity
                steps = [
                    "First, conduct a security assessment to identify potential vulnerabilities.",
                    "Implement network security measures like firewalls and segmentation.",
                    "Secure all endpoints with proper authentication and authorization.",
                    "Encrypt sensitive data both at rest and in transit.",
                    "Develop and enforce security policies and procedures.",
                    "Train users on security awareness and best practices.",
                    "Implement monitoring and incident response procedures.",
                    "Regularly test and update your security measures."
                ]
                
        else:
            # Generic tutorial steps
            steps = [
                f"First, you need to set up the environment for {topic}.",
                f"Next, configure the basic settings for {topic}.",
                f"Install the necessary dependencies for {topic}.",
                f"Create the initial structure for {topic}.",
                f"Implement the core functionality for {topic}.",
                f"Test your implementation of {topic}.",
                f"Optimize and refine your {topic} implementation.",
                f"Deploy or publish your completed {topic} project."
            ]
        
        # Trim to the desired number of steps
        return steps[:num_steps]
    
    def _generate_clarification_question(self, step, domain):
        """Generate a clarification question based on the step"""
        # Extract keywords from step
        words = re.findall(r'\b\w+\b', step)
        keywords = [word for word in words if len(word) > 4 and word.lower() not in ['first', 'start', 'begin', 'create', 'with', 'using', 'would', 'should', 'could']]
        
        if not keywords:
            keywords = ["this step", "that", "the process"]
        
        keyword = random.choice(keywords)
        
        templates = [
            f"Could you explain more about {keyword}?",
            f"What happens if {keyword} doesn't work?",
            f"Is there an alternative to {keyword}?",
            f"Do I need to customize {keyword} for my specific project?",
            f"How can I troubleshoot issues with {keyword}?",
            f"Is there documentation I should reference for {keyword}?",
            f"Can you provide more details about the {keyword} part?",
            f"What if my system doesn't support {keyword}?",
            f"Are there any prerequisites for the {keyword} step?"
        ]
        
        return random.choice(templates)
    
    def _generate_clarification_answer(self, question, step):
        """Generate an answer to a clarification question"""
        if "explain more" in question.lower() or "more details" in question.lower():
            return f"Sure! {step} This step is crucial because it establishes the foundation for the rest of the process. Make sure you complete it carefully before moving on."
            
        elif "doesn't work" in question.lower() or "troubleshoot" in question.lower():
            return f"If you encounter issues, first check for error messages. Common problems include missing dependencies, configuration errors, or network issues. You can troubleshoot by verifying each component, checking logs, and ensuring all prerequisites are met. If problems persist, check the official documentation or community forums."
            
        elif "alternative" in question.lower():
            return f"Yes, there are alternatives. While the approach I described is the most common, you could also use a different tool or method depending on your specific needs. The key is to achieve the same outcome, which is to properly set up this component of the project."
            
        elif "customize" in question.lower() or "specific" in question.lower():
            return f"Yes, you may need to customize this for your specific project. Consider your requirements, scale, and environment. The basic principles remain the same, but parameters, configurations, or implementations might need adjustments based on your unique situation."
            
        elif "documentation" in question.lower():
            return f"Yes, I recommend referring to the official documentation for more detailed information. It contains comprehensive guides, API references, and examples that can help you implement this step correctly and understand all available options."
            
        elif "prerequisites" in question.lower():
            return f"For this step, you'll need to have completed the previous steps successfully. Additionally, ensure you have sufficient permissions on your system, the necessary dependencies installed, and a stable environment. This will prevent common issues during implementation."
            
        else:
            return f"That's a good question. This step is quite flexible and can be adapted to different situations. The most important thing is to understand the underlying concept and apply it appropriately to your specific use case. Let me know if you need more specific guidance as you implement it."
    
    def convert_to_qa_format(self, dialogues):
        """
        Convert dialogues to QA format.
        
        Args:
            dialogues (list): List of dialogues
            
        Returns:
            list: QA pairs
        """
        qa_pairs = []
        
        for dialogue in dialogues:
            # Process each dialogue
            exchanges = dialogue["exchanges"]
            context = []
            
            for i in range(0, len(exchanges), 2):
                if i + 1 >= len(exchanges):  # Skip if no assistant response
                    continue
                    
                user_msg = exchanges[i]["content"]
                assistant_msg = exchanges[i+1]["content"]
                
                # Create QA pair
                qa_pair = {
                    "question": user_msg,
                    "answer": assistant_msg,
                    "context": context.copy(),
                    "topic": dialogue["topic"],
                    "domain": dialogue["domain"]
                }
                
                qa_pairs.append(qa_pair)
                
                # Update context for next exchange
                context.append({
                    "role": "user",
                    "content": user_msg
                })
                context.append({
                    "role": "assistant",
                    "content": assistant_msg
                })
        
        return qa_pairs
    
    def save_data(self, dialogues, qa_pairs):
        """
        Save processed data to files.
        
        Args:
            dialogues (list): Raw dialogues
            qa_pairs (list): Processed QA pairs
            
        Returns:
            tuple: Paths to saved files
        """
        # Create output paths
        dialogues_path = self.output_dir / "tutorial_dialogues.json"
        qa_path = self.output_dir / "tutorial_qa.json"
        
        # Save dialogues
        with open(dialogues_path, 'w', encoding='utf-8') as f:
            json.dump(dialogues, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Saved {len(dialogues)} dialogues to {dialogues_path}")
        
        # Save QA pairs
        with open(qa_path, 'w', encoding='utf-8') as f:
            json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Saved {len(qa_pairs)} QA pairs to {qa_path}")
        
        return str(dialogues_path), str(qa_path)

def main(sample_size=1000, output_dir="./Datasets"):
    """
    Generate tutorial dialogues and save to files.
    
    Args:
        sample_size (int): Number of tutorial dialogues to generate
        output_dir (str): Output directory
        
    Returns:
        str: Path to saved QA file
    """
    converter = TutorialDialogueConverter(output_dir=output_dir, sample_size=sample_size)
    dialogues = converter.generate_tutorial_dialogues()
    qa_pairs = converter.convert_to_qa_format(dialogues)
    _, qa_path = converter.save_data(dialogues, qa_pairs)
    
    return qa_path

if __name__ == "__main__":
    main()
