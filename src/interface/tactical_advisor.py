"""
Tactical Advisor for Theta AI.

Provides Cortana-style proactive tactical insights and strategic advice.
Instead of just answering questions, Theta can anticipate needs and offer
unsolicited but helpful observations.
"""

import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class TacticalInsight:
    """A tactical observation or recommendation."""
    category: str  # security, performance, architecture, code_quality, etc.
    severity: str  # info, suggestion, warning, critical
    message: str
    action_items: List[str]


class TacticalAdvisor:
    """
    Proactive tactical analysis system for Theta AI.
    
    Analyzes context and provides strategic insights, warnings, and
    recommendations without being asked - like Cortana warning Master Chief.
    """
    
    def __init__(self):
        """Initialize the tactical advisor."""
        # Keywords that trigger different types of tactical analysis
        self.security_triggers = [
            "password", "auth", "login", "credential", "token", "api key",
            "secret", "encrypt", "decrypt", "hash", "sql", "input", "user data",
            "session", "cookie", "cors", "csrf", "xss", "injection"
        ]
        
        self.performance_triggers = [
            "slow", "performance", "optimize", "speed", "memory", "cpu",
            "database", "query", "cache", "load", "scale", "bottleneck"
        ]
        
        self.architecture_triggers = [
            "architecture", "design", "pattern", "structure", "framework",
            "microservice", "monolith", "api", "integration", "dependency"
        ]
        
        self.code_quality_triggers = [
            "refactor", "clean", "maintainable", "readable", "test",
            "debug", "error", "exception", "bug", "code review"
        ]
        
        # Proactive observations by category
        self.security_observations = [
            TacticalInsight(
                category="security",
                severity="warning",
                message="If you're handling user input here, make sure to sanitize it. SQL injection and XSS are still the most common attack vectors.",
                action_items=["Validate input types", "Escape special characters", "Use parameterized queries"]
            ),
            TacticalInsight(
                category="security",
                severity="suggestion",
                message="Consider implementing rate limiting if this endpoint is public-facing. Prevents brute force attempts.",
                action_items=["Add rate limiting middleware", "Log failed attempts", "Consider CAPTCHA for repeated failures"]
            ),
            TacticalInsight(
                category="security",
                severity="warning",
                message="Storing sensitive data? Make sure it's encrypted at rest, not just in transit.",
                action_items=["Use AES-256 for encryption", "Secure key management", "Regular key rotation"]
            ),
            TacticalInsight(
                category="security",
                severity="critical",
                message="Never hardcode credentials. Use environment variables or a secrets manager.",
                action_items=["Move secrets to env vars", "Consider HashiCorp Vault or AWS Secrets Manager", "Add .env to .gitignore"]
            ),
        ]
        
        self.performance_observations = [
            TacticalInsight(
                category="performance",
                severity="suggestion",
                message="Database queries in a loop? That's an N+1 problem waiting to happen. Consider eager loading or batch queries.",
                action_items=["Use JOIN or eager loading", "Batch similar queries", "Add query logging to detect issues"]
            ),
            TacticalInsight(
                category="performance",
                severity="suggestion",
                message="If this data doesn't change often, caching could significantly reduce load.",
                action_items=["Implement Redis or Memcached", "Set appropriate TTL", "Add cache invalidation logic"]
            ),
            TacticalInsight(
                category="performance",
                severity="info",
                message="Consider async processing for heavy operations. Don't make users wait for things that can run in the background.",
                action_items=["Use message queues", "Implement background workers", "Return early with status endpoint"]
            ),
        ]
        
        self.architecture_observations = [
            TacticalInsight(
                category="architecture",
                severity="suggestion",
                message="This is getting complex. Might be worth extracting into a separate service or module to keep things maintainable.",
                action_items=["Identify clear boundaries", "Define clean interfaces", "Document dependencies"]
            ),
            TacticalInsight(
                category="architecture",
                severity="info",
                message="If you're expecting this to scale, design for horizontal scaling now. It's harder to add later.",
                action_items=["Stateless design", "External session storage", "Load balancer friendly"]
            ),
        ]
        
        self.code_quality_observations = [
            TacticalInsight(
                category="code_quality",
                severity="suggestion",
                message="This logic is complex enough that it deserves unit tests. Future you will thank present you.",
                action_items=["Write tests for edge cases", "Aim for 80% coverage on critical paths", "Use descriptive test names"]
            ),
            TacticalInsight(
                category="code_quality",
                severity="info",
                message="Consider adding error handling here. Silent failures are debugging nightmares.",
                action_items=["Add try-catch blocks", "Log errors with context", "Return meaningful error messages"]
            ),
        ]
    
    def analyze_for_insights(self, user_input: str, conversation_history: List[str] = None) -> List[TacticalInsight]:
        """
        Analyze input and conversation context for tactical insights.
        
        Args:
            user_input: Current user message
            conversation_history: Recent conversation messages
            
        Returns:
            List of relevant tactical insights
        """
        insights = []
        input_lower = user_input.lower()
        
        # Check for security concerns
        if any(trigger in input_lower for trigger in self.security_triggers):
            insight = random.choice(self.security_observations)
            insights.append(insight)
        
        # Check for performance concerns
        if any(trigger in input_lower for trigger in self.performance_triggers):
            insight = random.choice(self.performance_observations)
            insights.append(insight)
        
        # Check for architecture discussions
        if any(trigger in input_lower for trigger in self.architecture_triggers):
            insight = random.choice(self.architecture_observations)
            insights.append(insight)
        
        # Check for code quality topics
        if any(trigger in input_lower for trigger in self.code_quality_triggers):
            insight = random.choice(self.code_quality_observations)
            insights.append(insight)
        
        return insights
    
    def get_tactical_interjection(self, topic: str, context: Dict = None) -> Optional[str]:
        """
        Generate a Cortana-style tactical interjection.
        
        Args:
            topic: Current topic being discussed
            context: Additional context
            
        Returns:
            A tactical observation or None
        """
        topic_lower = topic.lower()
        
        # Security-related interjections
        if any(kw in topic_lower for kw in ["password", "auth", "login"]):
            return random.choice([
                "Quick security note: make sure you're hashing passwords with bcrypt or Argon2, not MD5 or SHA-1.",
                "Heads up: if this is authentication logic, consider implementing MFA. Single-factor auth is increasingly risky.",
                "Remember: never store passwords in plaintext. I've seen that mistake take down entire systems."
            ])
        
        if any(kw in topic_lower for kw in ["api", "endpoint", "route"]):
            return random.choice([
                "If this API is public-facing, rate limiting and input validation are non-negotiable.",
                "Consider versioning your API from the start. Breaking changes are painful without it.",
                "Don't forget CORS configuration if this needs to be accessed from browsers."
            ])
        
        if any(kw in topic_lower for kw in ["database", "query", "sql"]):
            return random.choice([
                "Parameterized queries. Always. SQL injection is still one of the most common attack vectors.",
                "If you're doing complex queries, consider adding indexes on frequently filtered columns.",
                "Connection pooling can make a huge difference in database performance under load."
            ])
        
        if any(kw in topic_lower for kw in ["deploy", "production", "release"]):
            return random.choice([
                "Before going live: environment variables for all secrets, logging configured, monitoring in place.",
                "Have a rollback plan. Things go wrong in production - it's not if, it's when.",
                "Consider blue-green deployment or canary releases to minimize risk."
            ])
        
        return None
    
    def should_provide_insight(self, engagement_level: float = 0.5) -> bool:
        """
        Determine if Theta should proactively provide a tactical insight.
        
        Args:
            engagement_level: Current conversation engagement level
            
        Returns:
            True if should provide insight
        """
        # Higher engagement = more likely to share insights
        base_chance = 0.2
        if engagement_level > 0.7:
            base_chance = 0.4
        elif engagement_level > 0.5:
            base_chance = 0.3
            
        return random.random() < base_chance
    
    def format_insight(self, insight: TacticalInsight) -> str:
        """
        Format a tactical insight for presentation.
        
        Args:
            insight: The tactical insight to format
            
        Returns:
            Formatted string
        """
        severity_prefix = {
            "info": "💡",
            "suggestion": "→",
            "warning": "⚠️",
            "critical": "🚨"
        }
        
        prefix = severity_prefix.get(insight.severity, "•")
        
        result = f"{prefix} {insight.message}"
        
        if insight.action_items and insight.severity in ["warning", "critical"]:
            result += "\n\nRecommended actions:"
            for item in insight.action_items:
                result += f"\n  • {item}"
        
        return result


# Singleton instance
_tactical_advisor = None

def get_tactical_advisor() -> TacticalAdvisor:
    """Get the singleton TacticalAdvisor instance."""
    global _tactical_advisor
    if _tactical_advisor is None:
        _tactical_advisor = TacticalAdvisor()
    return _tactical_advisor
