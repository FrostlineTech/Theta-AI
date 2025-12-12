"""
Math expression handler for Theta AI.
Handles basic arithmetic and common math expressions.
"""

import re
import math

class MathHandler:
    """Handler for mathematical expressions and calculations."""
    
    def __init__(self):
        """Initialize the math handler with patterns for common expressions."""
        # Safe math operations (no builtins allowed)
        self.safe_math_ops = {
            '+': lambda a, b: a + b,
            '-': lambda a, b: a - b,
            '*': lambda a, b: a * b,
            '/': lambda a, b: a / b if b != 0 else float('inf'),
            '^': lambda a, b: a ** b,
            'sqrt': lambda a: math.sqrt(a) if a >= 0 else float('nan'),
            'abs': lambda a: abs(a),
            'round': lambda a: round(a)
        }
        
        # Special case patterns
        self.special_patterns = {
            # Basic arithmetic with operators spelled out
            r'(\d+)\s*plus\s*(\d+)': lambda m: float(m.group(1)) + float(m.group(2)),
            r'(\d+)\s*minus\s*(\d+)': lambda m: float(m.group(1)) - float(m.group(2)),
            r'(\d+)\s*times\s*(\d+)': lambda m: float(m.group(1)) * float(m.group(2)),
            r'(\d+)\s*divided\s*by\s*(\d+)': lambda m: float(m.group(1)) / float(m.group(2)) if float(m.group(2)) != 0 else float('inf'),
            
            # Powers
            r'(\d+)\s*squared': lambda m: float(m.group(1)) ** 2,
            r'(\d+)\s*cubed': lambda m: float(m.group(1)) ** 3,
            r'(\d+)\s*to\s*the\s*power\s*of\s*(\d+)': lambda m: float(m.group(1)) ** float(m.group(2)),
            r'(\d+)\s*\^(\d+)': lambda m: float(m.group(1)) ** float(m.group(2)),
            
            # Square roots
            r'square\s*root\s*of\s*(\d+(?:\.\d+)?)': lambda m: math.sqrt(float(m.group(1))) if float(m.group(1)) >= 0 else float('nan'),
            r'sqrt\s*\(?(\d+(?:\.\d+)?)\)?': lambda m: math.sqrt(float(m.group(1))) if float(m.group(1)) >= 0 else float('nan'),
            
            # Percentages
            r'(\d+(?:\.\d+)?)\s*percent\s*of\s*(\d+(?:\.\d+)?)': lambda m: (float(m.group(1)) / 100) * float(m.group(2)),
            r'(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)': lambda m: (float(m.group(1)) / 100) * float(m.group(2))
        }
    
    def is_math_expression(self, text):
        """
        Check if the text is likely a math expression.
        
        Args:
            text (str): Text to check
            
        Returns:
            bool: True if text is likely a math expression
        """
        text = text.lower().strip()
        
        # Check if it's a question about a math expression
        math_question = re.search(r'(what is|calculate|compute|solve|evaluate|find)\s+(.+)', text)
        if math_question:
            text = math_question.group(2).strip()
        
        # Check for basic arithmetic operators
        if re.search(r'\d+\s*[\+\-\*/×÷\^]\s*\d+', text):
            return True
            
        # Check for text-based math expressions
        math_keywords = ['plus', 'minus', 'times', 'divided by', 'squared', 'cubed', 
                         'square root', 'sqrt', 'percent of', 'power']
        for keyword in math_keywords:
            if keyword in text:
                return True
                
        # Check special patterns
        for pattern in self.special_patterns:
            if re.search(pattern, text):
                return True
                
        return False
    
    def evaluate_expression(self, text):
        """
        Evaluate a mathematical expression.
        
        Args:
            text (str): Math expression to evaluate
            
        Returns:
            float or None: Result of the calculation or None if not a valid expression
        """
        # Clean up the text
        text = text.lower().strip()
        
        # Check if it's a question about a math expression
        math_question = re.search(r'(what is|calculate|compute|solve|evaluate|find)\s+(.+)', text)
        if math_question:
            text = math_question.group(2).strip()
            
        # Remove question marks
        text = text.rstrip('?')
        
        # Check special patterns first
        for pattern, evaluator in self.special_patterns.items():
            match = re.search(pattern, text)
            if match:
                try:
                    return evaluator(match)
                except Exception as e:
                    print(f"Error evaluating pattern {pattern}: {e}")
                    return None
        
        # Handle basic arithmetic expressions
        try:
            # Replace text operators with symbols
            text = text.replace('x', '*').replace('×', '*').replace('÷', '/').replace('^', '**')
            
            # Remove non-math characters
            expression = re.sub(r'[^0-9+\-*/().e]', '', text)
            
            # Safety check: only allow basic arithmetic operations
            if any(op not in "0123456789+-*/().e" for op in expression):
                return None
                
            # Parse and evaluate with safe operations only
            result = self._safe_eval(expression)
            return result
            
        except Exception as e:
            print(f"Error evaluating expression: {e}")
            return None
            
    def _safe_eval(self, expr):
        """
        Safely evaluate a math expression without using eval().
        This is a simplified implementation that handles basic arithmetic.
        
        Args:
            expr (str): Math expression string
            
        Returns:
            float: Result of the calculation
        """
        # Very simple implementation - in a real system, use a proper math parser
        # This handles only basic operations with precedence
        
        # Replace ** with ^ for our internal handling
        expr = expr.replace('**', '^')
        
        # Handle parentheses first
        while '(' in expr:
            # Find innermost parentheses
            inner = re.search(r'\(([^()]*)\)', expr)
            if not inner:
                break
            
            # Evaluate the inner expression
            inner_result = self._safe_eval(inner.group(1))
            
            # Replace in original expression
            expr = expr[:inner.start()] + str(inner_result) + expr[inner.end():]
        
        # Handle powers
        while '^' in expr:
            power_match = re.search(r'(\d+(?:\.\d+)?)\^(\d+(?:\.\d+)?)', expr)
            if not power_match:
                break
                
            base = float(power_match.group(1))
            exp = float(power_match.group(2))
            result = base ** exp
            
            expr = expr[:power_match.start()] + str(result) + expr[power_match.end():]
        
        # Handle multiplication and division
        while re.search(r'\d+(?:\.\d+)?\s*[*/]\s*\d+(?:\.\d+)?', expr):
            op_match = re.search(r'(\d+(?:\.\d+)?)\s*([*/])\s*(\d+(?:\.\d+)?)', expr)
            if not op_match:
                break
                
            a = float(op_match.group(1))
            op = op_match.group(2)
            b = float(op_match.group(3))
            
            if op == '*':
                result = a * b
            else:  # Division
                if b == 0:
                    return float('inf')
                result = a / b
                
            expr = expr[:op_match.start()] + str(result) + expr[op_match.end():]
            
        # Handle addition and subtraction
        while re.search(r'\d+(?:\.\d+)?\s*[+-]\s*\d+(?:\.\d+)?', expr):
            op_match = re.search(r'(\d+(?:\.\d+)?)\s*([+-])\s*(\d+(?:\.\d+)?)', expr)
            if not op_match:
                break
                
            a = float(op_match.group(1))
            op = op_match.group(2)
            b = float(op_match.group(3))
            
            if op == '+':
                result = a + b
            else:  # Subtraction
                result = a - b
                
            expr = expr[:op_match.start()] + str(result) + expr[op_match.end():]
        
        # Final result should be just a number
        return float(expr)
    
    def format_result(self, result):
        """
        Format the result for display.
        
        Args:
            result (float): Result to format
            
        Returns:
            str: Formatted result
        """
        if result is None:
            return None
            
        # Check if result is close to an integer
        if abs(result - round(result)) < 1e-10:
            return str(int(result))
        
        # Format to at most 4 decimal places
        return str(round(result, 4)).rstrip('0').rstrip('.')
