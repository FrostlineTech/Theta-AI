"""
Common Facts Database for Theta AI.
Contains factual information for common questions.
"""

import re

class FactsDatabase:
    """
    Database of common facts to answer basic knowledge questions.
    Helps to reduce hallucinations and provide consistent answers.
    """
    
    def __init__(self):
        """Initialize the facts database."""
        # General knowledge facts
        self.general_facts = {
            # Earth and Space
            "earth": "Earth is the third planet from the Sun and the only astronomical object known to harbor life.",
            "sun": "The Sun is the star at the center of our Solar System. It's a nearly perfect sphere of hot plasma heated by nuclear fusion reactions.",
            "moon": "The Moon is Earth's only natural satellite. It's about 1/4 the diameter of Earth and is the fifth largest satellite in the Solar System.",
            "mars": "Mars is the fourth planet from the Sun and the second-smallest planet in the Solar System. It's often called the 'Red Planet' due to its reddish appearance.",
            "solar system": "The Solar System consists of the Sun and the objects that orbit it, including eight planets, their moons, asteroids, comets, and other celestial bodies.",
            
            # Elements and Matter
            "water": "Water (H2O) is a transparent, tasteless, odorless, and nearly colorless chemical substance that is essential for all known forms of life.",
            "oxygen": "Oxygen is a chemical element with symbol O and atomic number 8. It's essential for human respiration and makes up about 21% of Earth's atmosphere.",
            "carbon": "Carbon is a chemical element with symbol C and atomic number 6. It's the basis for all known life on Earth and forms more compounds than any other element.",
            "hydrogen": "Hydrogen is a chemical element with symbol H and atomic number 1. It's the lightest and most abundant element in the universe.",
            
            # Technology
            "internet": "The Internet is a global system of interconnected computer networks that use standardized communication protocols to link devices worldwide.",
            "computer": "A computer is a machine that can be programmed to carry out sequences of arithmetic or logical operations automatically.",
            "artificial intelligence": "Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans.",
            "machine learning": "Machine learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed.",
            "software": "Software is a collection of instructions and data that tell a computer how to work, in contrast to physical hardware from which the system is built.",
            
            # Biology
            "dna": "DNA (deoxyribonucleic acid) is a molecule composed of two polynucleotide chains that coil around each other to form a double helix carrying genetic instructions.",
            "cell": "The cell is the basic structural and functional unit of all living organisms. Cells are the smallest unit of life that can replicate independently.",
            "protein": "Proteins are large biomolecules consisting of one or more long chains of amino acids. They perform a vast array of functions within organisms.",
            "evolution": "Evolution is change in the heritable characteristics of biological populations over successive generations. These changes allow species to adapt to their environment.",
            
            # Mathematics
            "pi": "Pi (π) is a mathematical constant approximately equal to 3.14159. It represents the ratio of a circle's circumference to its diameter.",
            "fibonacci": "The Fibonacci sequence is a series of numbers where each number is the sum of the two preceding ones, usually starting with 0 and 1 (0, 1, 1, 2, 3, 5, 8, 13, ...).",
            "prime number": "A prime number is a natural number greater than 1 that is not a product of two smaller natural numbers.",
            "algorithm": "An algorithm is a finite sequence of well-defined, computer-implementable instructions, typically to solve a class of problems or to perform a computation.",
            
            # History
            "world war ii": "World War II (1939-1945) was a global war that involved the majority of the world's nations, including all of the great powers, forming opposing military alliances.",
            "industrial revolution": "The Industrial Revolution was the transition to new manufacturing processes in Europe and the United States, in the period from about 1760 to sometime between 1820 and 1840.",
            "cold war": "The Cold War (1947-1991) was a period of geopolitical tension between the United States and the Soviet Union and their respective allies.",
            
            # Concepts
            "time": "Time is the indefinite continued progress of existence and events that occur in an apparently irreversible succession from the past, through the present, to the future.",
            "energy": "Energy is the quantitative property that must be transferred to an object in order to perform work on, or to heat, the object.",
            "gravity": "Gravity is a natural phenomenon by which all things with mass or energy are attracted to one another.",
            "money": "Money is any item or verifiable record that is generally accepted as payment for goods and services and repayment of debts in a particular country or socio-economic context.",
        }
        
        # Technology facts
        self.tech_facts = {
            # Programming Languages
            "python": "Python is a high-level, interpreted programming language known for its readability and versatility. It's widely used in web development, data science, artificial intelligence, and automation.",
            "javascript": "JavaScript is a programming language that enables interactive web pages and is an essential part of web applications. It runs on the client side of the web.",
            "java": "Java is a class-based, object-oriented programming language designed to have as few implementation dependencies as possible. It's used for enterprise-scale applications and Android development.",
            "c++": "C++ is a general-purpose programming language created as an extension of the C programming language. It's used for system/software development, game development, and resource-intensive applications.",
            
            # Cybersecurity
            "encryption": "Encryption is the process of encoding information so that only authorized parties can access it, protecting sensitive data from unauthorized access or theft.",
            "firewall": "A firewall is a network security device that monitors incoming and outgoing network traffic and decides whether to allow or block specific traffic based on a defined set of security rules.",
            "phishing": "Phishing is a cybercrime where targets are contacted by email, telephone, or text message by someone posing as a legitimate institution to lure them into providing sensitive data.",
            "malware": "Malware (malicious software) is any software intentionally designed to cause damage to a computer, server, client, or computer network.",
            "ransomware": "Ransomware is a type of malware that threatens to publish the victim's data or perpetually block access to it unless a ransom is paid.",
            
            # Web Technologies
            "html": "HTML (Hypertext Markup Language) is the standard markup language for creating web pages and web applications. It structures content on the web.",
            "css": "CSS (Cascading Style Sheets) is a style sheet language used for describing the presentation of a document written in HTML or XML.",
            "http": "HTTP (Hypertext Transfer Protocol) is the foundation of data communication for the World Wide Web, used for transmitting hypermedia documents.",
            "rest api": "REST API (Representational State Transfer Application Programming Interface) is an architectural style for designing networked applications, commonly used for web services.",
            
            # Software Development
            "git": "Git is a distributed version control system for tracking changes in source code during software development.",
            "agile": "Agile is an approach to software development that emphasizes incremental delivery, team collaboration, continual planning, and continual learning.",
            "devops": "DevOps is a set of practices that combines software development and IT operations to shorten the development lifecycle and provide continuous delivery of high-quality software.",
            "tdd": "TDD (Test-Driven Development) is a software development process that relies on the repetition of a very short development cycle: requirements are turned into test cases, then software is improved to pass the tests.",
            
            # Cloud Computing
            "cloud computing": "Cloud computing is the on-demand availability of computer system resources, especially data storage and computing power, without direct active management by the user.",
            "aws": "AWS (Amazon Web Services) is a comprehensive and widely used cloud platform offering over 200 fully featured services from data centers globally.",
            "azure": "Microsoft Azure is a cloud computing service created by Microsoft for building, testing, deploying, and managing applications and services through Microsoft-managed data centers.",
            "kubernetes": "Kubernetes is an open-source container orchestration system for automating software deployment, scaling, and management.",
        }
        
        # Cybersecurity facts
        self.cybersecurity_facts = {
            "zero day": "A zero-day vulnerability is a software security flaw that is unknown to those who should be interested in mitigating it, including the vendor of the target software.",
            "penetration testing": "Penetration testing (pen testing) is an authorized simulated cyberattack on a computer system, performed to evaluate the security of the system.",
            "social engineering": "Social engineering is the psychological manipulation of people into performing actions or divulging confidential information.",
            "two-factor authentication": "Two-factor authentication (2FA) is a security process in which users provide two different authentication factors to verify themselves, providing an additional layer of security.",
            "vpn": "A Virtual Private Network (VPN) extends a private network across a public network, enabling users to send and receive data across shared or public networks as if their devices were directly connected to the private network.",
            "ids": "An Intrusion Detection System (IDS) is a device or software application that monitors a network or systems for malicious activity or policy violations.",
            "xss": "Cross-site scripting (XSS) is a type of security vulnerability typically found in web applications that allows attackers to inject client-side scripts into web pages viewed by other users.",
            "csrf": "Cross-Site Request Forgery (CSRF) is an attack that forces authenticated users to submit a request to a web application against which they are currently authenticated.",
            "sql injection": "SQL injection is a code injection technique, used to attack data-driven applications, in which malicious SQL statements are inserted into an entry field for execution.",
            "ddos": "A Distributed Denial of Service (DDoS) attack is a malicious attempt to disrupt normal traffic of a targeted server, service or network by overwhelming the target with a flood of Internet traffic."
        }
        
        # Add identity facts about Theta AI
        self.identity_facts = {
            "birthday": "I was created on March 10, 2025. That's my official 'birthday'.",
            "age": "I was created on March 10, 2025, which makes me fairly new as an AI assistant.",
            "creation date": "I was created on March 10, 2025 by the Frostline Solutions team.",
            "name": "My name is Theta AI. I was named after the Greek letter θ (theta), which is often used in mathematics and engineering to represent variables and angles.",
            "creator": "I was developed by Frostline Solutions, a team specializing in cybersecurity and AI development.",
            "purpose": "I was designed to assist with cybersecurity, software development, and technical questions, providing helpful and accurate information to users.",
            "capabilities": "I can answer questions about cybersecurity, software development, and technical topics. I can also perform basic calculations, maintain conversation context, and provide facts on various subjects.",
            "limitations": "While I aim to be helpful, I have limitations. I don't have access to real-time data, can't browse the internet, and my knowledge has a cutoff date. I also can't perform actions outside my designed capabilities."
        }
        
        # Combine all fact dictionaries
        self.facts = {**self.general_facts, **self.tech_facts, **self.cybersecurity_facts, **self.identity_facts}
    
    def get_fact(self, topic):
        """
        Get a fact about a specific topic.
        
        Args:
            topic (str): Topic to retrieve fact about
            
        Returns:
            str or None: Fact about the topic or None if not found
        """
        # Normalize topic
        topic = topic.lower().strip()
        
        # Direct lookup
        if topic in self.facts:
            return self.facts[topic]
            
        # Try to find partial matches
        for key, value in self.facts.items():
            if key in topic or topic in key:
                return value
                
        return None
        
    def is_factual_question(self, question):
        """
        Check if the question is asking for factual information.
        
        Args:
            question (str): Question to check
            
        Returns:
            bool: True if it's a factual question
        """
        question = question.lower().strip()
        
        # Check for question patterns
        factual_patterns = [
            "what is", "what are", "tell me about", "describe", 
            "explain", "who is", "who are", "where is", "where are",
            "when did", "when is", "why is", "why did", "how does",
            "definition of", "meaning of"
        ]
        
        if any(pattern in question for pattern in factual_patterns):
            return True
            
        return False
        
    def extract_topic_from_question(self, question):
        """
        Extract the main topic from a question.
        
        Args:
            question (str): Question to extract topic from
            
        Returns:
            str or None: Extracted topic or None
        """
        question = question.lower().strip()
        
        # Try to extract topic based on common question patterns
        patterns = [
            r"what is (a |an |the )?(.*?)(\?|$|\.)",
            r"what are (a |an |the )?(.*?)(\?|$|\.)",
            r"tell me about (a |an |the )?(.*?)(\?|$|\.)",
            r"describe (a |an |the )?(.*?)(\?|$|\.)",
            r"explain (a |an |the )?(.*?)(\?|$|\.)",
            r"who is (a |an |the )?(.*?)(\?|$|\.)",
            r"who are (a |an |the )?(.*?)(\?|$|\.)",
            r"where is (a |an |the )?(.*?)(\?|$|\.)",
            r"where are (a |an |the )?(.*?)(\?|$|\.)"]
        
        for pattern in patterns:
            match = re.search(pattern, question)
            if match:
                topic = match.group(2).strip()
                return topic
                
        # If no pattern matched, try direct lookup in the facts dictionary
        for topic in self.facts.keys():
            if topic in question:
                return topic
                
        return None
        
    def answer_factual_question(self, question):
        """
        Answer a factual question if possible.
        
        Args:
            question (str): Question to answer
            
        Returns:
            str or None: Answer to the question or None if not answerable
        """
        if not self.is_factual_question(question):
            return None
            
        topic = self.extract_topic_from_question(question)
        if not topic:
            return None
            
        fact = self.get_fact(topic)
        if not fact:
            return None
            
        return fact
