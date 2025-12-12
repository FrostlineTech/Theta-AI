"""
Integration module for all inference improvements.

This module ties together all the inference improvements into a single
installation function to be used in production.
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def integrate_all_improvements(theta_instance):
    """
    Integrate all inference improvements into a ThetaInterface instance
    
    Args:
        theta_instance: The ThetaInterface instance to enhance
        
    Returns:
        The enhanced ThetaInterface instance
    """
    if theta_instance is None:
        logger.error("Cannot integrate improvements: No ThetaInterface instance provided")
        return None
    
    # Import all improvement modules
    try:
        from .inference_improvements import install_inference_improvements
        from .response_generation import install_enhanced_response_generation
        
        # Apply improvements in sequence
        logger.info("Installing inference improvements...")
        theta_instance = install_inference_improvements(theta_instance)
        
        logger.info("Installing enhanced response generation...")
        theta_instance = install_enhanced_response_generation(theta_instance)
        
        # Log the installed modules
        improvements = []
        if hasattr(theta_instance, "inference_improvements"):
            improvements.append("Core Inference Improvements")
        if hasattr(theta_instance, "enhanced_response_generation"):
            improvements.append("Enhanced Response Generation")
            
        logger.info(f"Successfully installed {len(improvements)} improvement modules: {', '.join(improvements)}")
        
        # Create installation marker
        theta_instance._has_inference_improvements = True
        theta_instance._inference_improvements_version = "1.0.0"
        
        return theta_instance
        
    except Exception as e:
        logger.error(f"Error integrating inference improvements: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return theta_instance

def verify_integration(theta_instance):
    """
    Verify that inference improvements are properly integrated
    
    Args:
        theta_instance: The ThetaInterface instance to verify
        
    Returns:
        bool: True if improvements are properly integrated, False otherwise
    """
    verification_points = [
        # Basic checks
        hasattr(theta_instance, "_has_inference_improvements"),
        hasattr(theta_instance, "inference_improvements"),
        hasattr(theta_instance, "enhanced_response_generation"),
        
        # Method checks
        hasattr(theta_instance, "_original_generate_response"),
        theta_instance._original_generate_response != theta_instance.generate_response,
        
        # Configuration checks
        hasattr(theta_instance.inference_improvements, "config") if hasattr(theta_instance, "inference_improvements") else False,
    ]
    
    # Calculate percentage of verified points
    verified_count = sum(1 for check in verification_points if check)
    verification_percentage = (verified_count / len(verification_points)) * 100
    
    if verification_percentage >= 80:
        logger.info(f"Inference improvements integration verified: {verification_percentage:.1f}% complete")
        return True
    else:
        logger.warning(f"Inference improvements integration incomplete: only {verification_percentage:.1f}% verified")
        # Log missing components
        for i, check in enumerate(verification_points):
            if not check:
                logger.warning(f"Verification point {i+1} failed")
        return False
