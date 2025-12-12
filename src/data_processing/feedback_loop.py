"""
Feedback Loop System for Theta AI Knowledge Improvement

This module implements a feedback loop that captures user interactions,
analyzes model performance, and improves the knowledge base over time.
"""

import json
import logging
import os
import re
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeedbackLoop:
    """
    Implements a feedback loop for continuous knowledge improvement.
    Captures user feedback, identifies knowledge gaps, and enhances training data.
    """
    
    def __init__(self, datasets_dir: Path, db_path: Optional[Path] = None):
        """
        Initialize the feedback loop system.
        
        Args:
            datasets_dir: Path to the datasets directory
            db_path: Path to the SQLite database (if None, uses default path)
        """
        self.datasets_dir = datasets_dir
        
        # Set up database path
        if db_path is None:
            self.db_path = datasets_dir.parent / "database" / "feedback_loop.db"
        else:
            self.db_path = db_path
            
        # Create directory if it doesn't exist
        os.makedirs(self.db_path.parent, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Create feedback-based datasets directory
        self.feedback_dir = datasets_dir / "feedback_based"
        os.makedirs(self.feedback_dir, exist_ok=True)
        
        # Define minimum feedback thresholds
        self.min_positive_rating = 4  # Out of 5
        self.min_feedback_count = 3   # Minimum number of feedback instances
    
    def _init_database(self):
        """Initialize the SQLite database for feedback storage."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create tables if they don't exist
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT,
                question TEXT,
                answer TEXT,
                rating INTEGER,
                comment TEXT,
                timestamp TEXT,
                domain TEXT,
                incorporated BOOLEAN DEFAULT 0
            )
            ''')
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_gaps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT,
                question_pattern TEXT,
                frequency INTEGER,
                domain TEXT,
                last_updated TEXT,
                addressed BOOLEAN DEFAULT 0
            )
            ''')
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS improvement_cycles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cycle_date TEXT,
                feedback_count INTEGER,
                gaps_addressed INTEGER,
                examples_added INTEGER,
                examples_modified INTEGER
            )
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info(f"Initialized feedback loop database at {self.db_path}")
            
        except Exception as e:
            logger.error(f"Error initializing feedback database: {e}")
    
    def record_feedback(self, conversation_id: str, question: str, answer: str, 
                      rating: int, comment: str = "", domain: str = None) -> bool:
        """
        Record user feedback about a conversation exchange.
        
        Args:
            conversation_id: Unique ID for the conversation
            question: User's question
            answer: Theta's answer
            rating: User rating (1-5)
            comment: Optional comment from user
            domain: Optional domain classification
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Infer domain if not provided
            if domain is None:
                domain = self._infer_domain(question, answer)
            
            # Record timestamp
            timestamp = datetime.now().isoformat()
            
            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                """
                INSERT INTO user_feedback 
                (conversation_id, question, answer, rating, comment, timestamp, domain, incorporated)
                VALUES (?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (conversation_id, question, answer, rating, comment, timestamp, domain)
            )
            
            conn.commit()
            conn.close()
            
            logger.info(f"Recorded feedback for conversation {conversation_id} with rating {rating}")
            
            # Check if this is a knowledge gap
            if rating <= 2:  # Low rating indicates potential knowledge gap
                self._analyze_potential_gap(question, answer, domain)
            
            return True
            
        except Exception as e:
            logger.error(f"Error recording feedback: {e}")
            return False
    
    def _infer_domain(self, question: str, answer: str) -> str:
        """
        Infer the domain based on question and answer content.
        
        Args:
            question: The question text
            answer: The answer text
            
        Returns:
            Inferred domain
        """
        # Combine text for analysis
        text = (question + " " + answer).lower()
        
        # Domain keywords (simplified version)
        domain_keywords = {
            "cybersecurity": ["security", "cyber", "attack", "threat", "vulnerability", 
                            "malware", "phishing", "hacker", "breach", "encryption"],
            "programming": ["code", "programming", "function", "class", "variable", 
                          "algorithm", "debug", "compile", "syntax", "developer"],
            "networking": ["network", "router", "switch", "packet", "protocol", 
                         "tcp/ip", "ethernet", "lan", "wan", "firewall"],
            "data_science": ["data", "algorithm", "model", "prediction", "machine learning", 
                           "statistic", "visualization", "dataset", "regression", "classification"],
            "cloud_computing": ["cloud", "aws", "azure", "gcp", "container", 
                              "kubernetes", "docker", "serverless", "iaas", "paas"]
        }
        
        # Count keyword matches for each domain
        scores = {domain: 0 for domain in domain_keywords}
        
        for domain, keywords in domain_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    scores[domain] += text.count(keyword)
        
        # Get domain with highest score, default to "general" if no matches
        if max(scores.values()) > 0:
            return max(scores.items(), key=lambda x: x[1])[0]
        else:
            return "general"
    
    def _analyze_potential_gap(self, question: str, answer: str, domain: str):
        """
        Analyze a potential knowledge gap from a low-rated response.
        
        Args:
            question: The question that wasn't answered well
            answer: The unsatisfactory answer
            domain: The domain of the question
        """
        try:
            # Generate a question pattern (simplified)
            # In a real system, use more advanced NLP techniques
            question_lower = question.lower()
            
            # Remove specific details to create a more general pattern
            pattern = re.sub(r'[0-9]', 'X', question_lower)
            pattern = re.sub(r'[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+', 'FILENAME', pattern)  # Replace filenames
            pattern = re.sub(r'[a-zA-Z0-9_-]+\([^)]*\)', 'FUNCTION', pattern)  # Replace function calls
            
            # Check if this pattern exists
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                """
                SELECT id, frequency FROM knowledge_gaps
                WHERE question_pattern = ? AND domain = ?
                """,
                (pattern, domain)
            )
            
            result = cursor.fetchone()
            
            if result:
                # Update existing gap
                gap_id, frequency = result
                cursor.execute(
                    """
                    UPDATE knowledge_gaps
                    SET frequency = ?, last_updated = ?
                    WHERE id = ?
                    """,
                    (frequency + 1, datetime.now().isoformat(), gap_id)
                )
            else:
                # Create new gap
                cursor.execute(
                    """
                    INSERT INTO knowledge_gaps
                    (topic, question_pattern, frequency, domain, last_updated, addressed)
                    VALUES (?, ?, 1, ?, ?, 0)
                    """,
                    (self._extract_topic(question), pattern, domain, datetime.now().isoformat())
                )
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error analyzing knowledge gap: {e}")
    
    def _extract_topic(self, question: str) -> str:
        """
        Extract a topic from a question using simple heuristics.
        
        Args:
            question: The question to extract topic from
            
        Returns:
            Extracted topic
        """
        # Simple topic extraction (in a real system, use NLP)
        question_lower = question.lower()
        
        # Check for "what is" questions
        what_is_match = re.search(r'what is ([a-z0-9 _-]+)', question_lower)
        if what_is_match:
            return what_is_match.group(1).strip()
        
        # Check for "how to" questions
        how_to_match = re.search(r'how to ([a-z0-9 _-]+)', question_lower)
        if how_to_match:
            return "how to " + how_to_match.group(1).strip()
        
        # If no pattern matches, return first 30 chars
        return question_lower[:30].strip()
    
    def get_knowledge_gaps(self, min_frequency: int = 2, addressed: bool = False) -> List[Dict]:
        """
        Get knowledge gaps matching criteria.
        
        Args:
            min_frequency: Minimum occurrence frequency
            addressed: Whether to include addressed gaps
            
        Returns:
            List of knowledge gaps
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Return rows as dictionaries
            cursor = conn.cursor()
            
            cursor.execute(
                """
                SELECT id, topic, question_pattern, frequency, domain, last_updated, addressed
                FROM knowledge_gaps
                WHERE frequency >= ? AND addressed = ?
                ORDER BY frequency DESC
                """,
                (min_frequency, int(addressed))
            )
            
            gaps = [dict(row) for row in cursor.fetchall()]
            conn.close()
            
            return gaps
            
        except Exception as e:
            logger.error(f"Error getting knowledge gaps: {e}")
            return []
    
    def get_high_rated_responses(self, min_rating: int = 4, limit: int = 100) -> List[Dict]:
        """
        Get highly-rated responses for potential inclusion in training data.
        
        Args:
            min_rating: Minimum rating to include
            limit: Maximum number of responses to return
            
        Returns:
            List of high-rated QA pairs
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute(
                """
                SELECT id, question, answer, rating, domain, timestamp
                FROM user_feedback
                WHERE rating >= ? AND incorporated = 0
                ORDER BY rating DESC, timestamp DESC
                LIMIT ?
                """,
                (min_rating, limit)
            )
            
            responses = [dict(row) for row in cursor.fetchall()]
            conn.close()
            
            return responses
            
        except Exception as e:
            logger.error(f"Error getting high-rated responses: {e}")
            return []
    
    def run_improvement_cycle(self) -> Dict:
        """
        Run a complete knowledge improvement cycle.
        Identifies gaps, generates new training examples, and updates datasets.
        
        Returns:
            Statistics about the improvement cycle
        """
        start_time = time.time()
        logger.info("Starting knowledge improvement cycle")
        
        stats = {
            "gaps_addressed": 0,
            "examples_added": 0,
            "examples_modified": 0,
            "feedback_processed": 0
        }
        
        try:
            # 1. Get knowledge gaps
            gaps = self.get_knowledge_gaps(min_frequency=2, addressed=False)
            logger.info(f"Found {len(gaps)} knowledge gaps to address")
            
            # 2. Get high-rated responses
            good_responses = self.get_high_rated_responses(min_rating=self.min_positive_rating)
            logger.info(f"Found {len(good_responses)} highly-rated responses to incorporate")
            
            # 3. Group by domain
            domain_responses = {}
            for response in good_responses:
                domain = response.get("domain", "general")
                if domain not in domain_responses:
                    domain_responses[domain] = []
                domain_responses[domain].append(response)
            
            # 4. Create or update domain datasets
            for domain, responses in domain_responses.items():
                if not responses:
                    continue
                    
                stats["feedback_processed"] += len(responses)
                added, modified = self._update_domain_dataset(domain, responses)
                stats["examples_added"] += added
                stats["examples_modified"] += modified
                
                # Mark as incorporated
                self._mark_feedback_incorporated([r["id"] for r in responses])
            
            # 5. Address knowledge gaps by creating targeted datasets
            if gaps:
                gaps_addressed = self._address_knowledge_gaps(gaps)
                stats["gaps_addressed"] = gaps_addressed
            
            # 6. Record improvement cycle
            self._record_improvement_cycle(stats)
            
            # Log completion
            duration = time.time() - start_time
            logger.info(f"Knowledge improvement cycle completed in {duration:.1f} seconds")
            logger.info(f"Stats: {stats}")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error in knowledge improvement cycle: {e}")
            return stats
    
    def _update_domain_dataset(self, domain: str, responses: List[Dict]) -> Tuple[int, int]:
        """
        Update a domain-specific dataset with new responses.
        
        Args:
            domain: Domain to update
            responses: List of responses to incorporate
            
        Returns:
            Tuple of (added_count, modified_count)
        """
        # Define dataset path
        dataset_path = self.feedback_dir / f"{domain}_feedback.json"
        
        # Load existing dataset if available
        existing_data = []
        if dataset_path.exists():
            try:
                with open(dataset_path, 'r') as f:
                    existing_data = json.load(f)
            except Exception as e:
                logger.error(f"Error loading existing dataset {dataset_path}: {e}")
        
        # Convert to question-based dictionary for easy lookup
        existing_dict = {item.get("question", ""): item for item in existing_data}
        
        # Track changes
        added_count = 0
        modified_count = 0
        
        # Process new responses
        for response in responses:
            question = response.get("question", "")
            answer = response.get("answer", "")
            
            # Skip if invalid
            if not question or not answer:
                continue
                
            if question in existing_dict:
                # Update existing entry if new answer has higher rating
                existing_rating = existing_dict[question].get("metadata", {}).get("rating", 0)
                new_rating = response.get("rating", 0)
                
                if new_rating > existing_rating:
                    existing_dict[question]["answer"] = answer
                    existing_dict[question]["metadata"] = {
                        "rating": new_rating,
                        "last_updated": datetime.now().isoformat(),
                        "source": "user_feedback"
                    }
                    modified_count += 1
            else:
                # Add new entry
                new_entry = {
                    "question": question,
                    "answer": answer,
                    "domain": domain,
                    "metadata": {
                        "rating": response.get("rating", 5),
                        "created": datetime.now().isoformat(),
                        "source": "user_feedback"
                    }
                }
                existing_dict[question] = new_entry
                added_count += 1
        
        # Convert back to list
        updated_data = list(existing_dict.values())
        
        # Save updated dataset
        with open(dataset_path, 'w') as f:
            json.dump(updated_data, f, indent=2)
            
        logger.info(f"Updated {domain} dataset: {added_count} added, {modified_count} modified")
        return added_count, modified_count
    
    def _mark_feedback_incorporated(self, feedback_ids: List[int]):
        """
        Mark feedback as incorporated into training data.
        
        Args:
            feedback_ids: List of feedback IDs to mark
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for feedback_id in feedback_ids:
                cursor.execute(
                    """
                    UPDATE user_feedback
                    SET incorporated = 1
                    WHERE id = ?
                    """,
                    (feedback_id,)
                )
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error marking feedback as incorporated: {e}")
    
    def _address_knowledge_gaps(self, gaps: List[Dict]) -> int:
        """
        Address identified knowledge gaps.
        
        Args:
            gaps: List of knowledge gaps to address
            
        Returns:
            Number of gaps addressed
        """
        # Group by domain
        domain_gaps = {}
        for gap in gaps:
            domain = gap.get("domain", "general")
            if domain not in domain_gaps:
                domain_gaps[domain] = []
            domain_gaps[domain].append(gap)
        
        # Create gap-focused datasets
        addressed_count = 0
        for domain, domain_gaps_list in domain_gaps.items():
            # Create dataset with gaps to address
            gaps_dataset = []
            for gap in domain_gaps_list:
                # In a real system, this would generate targeted training examples
                # Here we just create placeholder entries
                gaps_dataset.append({
                    "question": f"Example question for gap: {gap['topic']}",
                    "answer": f"This is a placeholder answer for the knowledge gap about {gap['topic']}. " +
                             "In a real implementation, this would be generated or researched content.",
                    "domain": domain,
                    "metadata": {
                        "gap_id": gap["id"],
                        "created": datetime.now().isoformat(),
                        "source": "knowledge_gap"
                    }
                })
            
            # Save dataset
            if gaps_dataset:
                gaps_path = self.feedback_dir / f"{domain}_gaps.json"
                with open(gaps_path, 'w') as f:
                    json.dump(gaps_dataset, f, indent=2)
                
                # Mark gaps as addressed
                self._mark_gaps_addressed([g["id"] for g in domain_gaps_list])
                addressed_count += len(domain_gaps_list)
        
        return addressed_count
    
    def _mark_gaps_addressed(self, gap_ids: List[int]):
        """
        Mark knowledge gaps as addressed.
        
        Args:
            gap_ids: List of gap IDs to mark
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for gap_id in gap_ids:
                cursor.execute(
                    """
                    UPDATE knowledge_gaps
                    SET addressed = 1
                    WHERE id = ?
                    """,
                    (gap_id,)
                )
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error marking gaps as addressed: {e}")
    
    def _record_improvement_cycle(self, stats: Dict):
        """
        Record the improvement cycle statistics.
        
        Args:
            stats: Statistics about the improvement cycle
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                """
                INSERT INTO improvement_cycles
                (cycle_date, feedback_count, gaps_addressed, examples_added, examples_modified)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    datetime.now().isoformat(),
                    stats.get("feedback_processed", 0),
                    stats.get("gaps_addressed", 0),
                    stats.get("examples_added", 0),
                    stats.get("examples_modified", 0)
                )
            )
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error recording improvement cycle: {e}")
    
    def get_improvement_statistics(self, days: int = 30) -> Dict:
        """
        Get statistics about knowledge improvement over time.
        
        Args:
            days: Number of days to look back
            
        Returns:
            Statistics about improvements
        """
        try:
            # Calculate start date
            start_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get improvement cycles
            cursor.execute(
                """
                SELECT * FROM improvement_cycles
                WHERE cycle_date >= ?
                ORDER BY cycle_date ASC
                """,
                (start_date,)
            )
            
            cycles = [dict(row) for row in cursor.fetchall()]
            
            # Get feedback summary
            cursor.execute(
                """
                SELECT COUNT(*) as total,
                       SUM(CASE WHEN rating >= 4 THEN 1 ELSE 0 END) as positive,
                       SUM(CASE WHEN rating <= 2 THEN 1 ELSE 0 END) as negative,
                       AVG(rating) as average_rating
                FROM user_feedback
                WHERE timestamp >= ?
                """,
                (start_date,)
            )
            
            feedback_stats = dict(cursor.fetchone())
            
            # Get knowledge gap summary
            cursor.execute(
                """
                SELECT COUNT(*) as total,
                       SUM(CASE WHEN addressed = 1 THEN 1 ELSE 0 END) as addressed
                FROM knowledge_gaps
                WHERE last_updated >= ?
                """,
                (start_date,)
            )
            
            gap_stats = dict(cursor.fetchone())
            
            conn.close()
            
            # Combine statistics
            stats = {
                "period_days": days,
                "cycles": cycles,
                "feedback": feedback_stats,
                "knowledge_gaps": gap_stats,
                "summary": {
                    "total_cycles": len(cycles),
                    "total_examples_added": sum(c.get("examples_added", 0) for c in cycles),
                    "total_examples_modified": sum(c.get("examples_modified", 0) for c in cycles),
                    "total_gaps_addressed": sum(c.get("gaps_addressed", 0) for c in cycles),
                    "average_rating": feedback_stats.get("average_rating", 0)
                }
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting improvement statistics: {e}")
            return {"error": str(e)}
    
    def generate_combined_feedback_dataset(self) -> Path:
        """
        Generate a combined dataset from all feedback-based datasets.
        
        Returns:
            Path to the combined dataset
        """
        combined_data = []
        
        try:
            # Find all feedback datasets
            feedback_files = list(self.feedback_dir.glob("*_feedback.json"))
            gap_files = list(self.feedback_dir.glob("*_gaps.json"))
            
            all_files = feedback_files + gap_files
            
            # Load and combine all datasets
            for file_path in all_files:
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    if isinstance(data, list):
                        combined_data.extend(data)
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
            
            # Save combined dataset
            combined_path = self.feedback_dir / "combined_feedback.json"
            with open(combined_path, 'w') as f:
                json.dump(combined_data, f, indent=2)
                
            logger.info(f"Created combined feedback dataset with {len(combined_data)} examples")
            return combined_path
            
        except Exception as e:
            logger.error(f"Error generating combined feedback dataset: {e}")
            return None

def main():
    """Main function to test the feedback loop system."""
    # Get project root and datasets directory
    project_root = Path(__file__).resolve().parent.parent.parent
    datasets_dir = project_root / "Datasets"
    
    # Create feedback loop system
    feedback_loop = FeedbackLoop(datasets_dir)
    
    # Add some test feedback
    print("Adding test feedback...")
    feedback_loop.record_feedback(
        conversation_id="test_conv_1",
        question="What is the principle of least privilege in cybersecurity?",
        answer="The principle of least privilege is a security concept that restricts access rights for users, accounts, and computing processes to only those resources absolutely required to perform routine, legitimate activities.",
        rating=5,
        domain="cybersecurity"
    )
    
    feedback_loop.record_feedback(
        conversation_id="test_conv_2",
        question="How do I implement a binary search algorithm in Python?",
        answer="Here's a simple implementation of binary search in Python...",
        rating=4,
        domain="programming"
    )
    
    feedback_loop.record_feedback(
        conversation_id="test_conv_3",
        question="What is the difference between TCP and UDP?",
        answer="I'm not sure about the specific differences.",
        rating=2,
        domain="networking"
    )
    
    # Run improvement cycle
    print("Running improvement cycle...")
    stats = feedback_loop.run_improvement_cycle()
    print(f"Improvement cycle stats: {stats}")
    
    # Generate combined dataset
    print("Generating combined dataset...")
    combined_path = feedback_loop.generate_combined_feedback_dataset()
    print(f"Combined dataset saved to: {combined_path}")

if __name__ == "__main__":
    main()
