"""
Setup Enhanced Database for Theta AI

This script initializes the database with the enhanced schema required for
knowledge base improvements.
"""

import os
import logging
from pathlib import Path
import sys

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import database modules
from src.database.db_setup import create_tables, check_database_connection
from src.database.db_setup_update import create_feedback_loop_tables, update_tables

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_database():
    """
    Set up the enhanced database schema.
    """
    logger.info("Setting up enhanced database schema for Theta AI")
    
    # Check database connection
    if not check_database_connection():
        logger.error("Database connection failed. Please check your database configuration.")
        return False
    
    # Create feedback loop tables first
    try:
        logger.info("Creating feedback loop tables...")
        create_feedback_loop_tables()
    except Exception as e:
        logger.error(f"Error creating feedback loop tables: {e}")
        return False
    
    # Create or update base tables
    try:
        logger.info("Creating/updating base database tables...")
        create_tables()
    except Exception as e:
        logger.error(f"Error creating/updating base tables: {e}")
        return False
    
    # Update existing tables with new columns
    try:
        logger.info("Updating existing tables with new columns...")
        update_tables()
    except Exception as e:
        logger.error(f"Error updating tables: {e}")
        return False
    
    logger.info("Database setup completed successfully!")
    return True

if __name__ == "__main__":
    print("Setting up enhanced database for Theta AI knowledge base improvements")
    if setup_database():
        print("\nDatabase setup completed successfully!")
        print("The system is now ready for training with enhanced knowledge base features.")
    else:
        print("\nDatabase setup failed. Please check the logs for details.")
