# services/chat/utils.py
"""Utility functions for chat services."""
import logging

logger = logging.getLogger(__name__)

def clean_sql_query(sql: str) -> str:
    """
    Clean SQL query from any markdown or formatting.
    
    Args:
        sql: Raw SQL string
        
    Returns:
        Cleaned SQL string
    """
    # Remove markdown code blocks
    sql = sql.replace("```sql", "").replace("```", "")
    # Remove any leading/trailing whitespace and semicolons
    sql = sql.strip().rstrip(";").strip()
    return sql

def extract_query_from_execute_command(message: str) -> str:
    """
    Extract the actual query from an execute command.
    
    Args:
        message: User message with execute command
        
    Returns:
        Query text without execute keywords
    """
    query_text = message
    execute_keywords = ["execute", "run"]
    
    for keyword in execute_keywords:
        query_text = query_text.replace(keyword, "").strip()
    
    return query_text if query_text else message

def is_simple_confirmation(message: str) -> bool:
    """
    Check if message is a simple confirmation without query content.
    
    Args:
        message: User message
        
    Returns:
        True if simple confirmation, False otherwise
    """
    message_lower = message.lower().strip()
    simple_confirmations = [
        "execute", "run", "yes", "go ahead", "execute it", 
        "run it", "yes execute", "do it", "proceed"
    ]
    return message_lower in simple_confirmations

def count_intent_indicators(message: str, indicators: list) -> int:
    """
    Count how many intent indicators are present in the message.
    
    Args:
        message: User message
        indicators: List of indicator phrases
        
    Returns:
        Count of indicators found
    """
    message_lower = message.lower()
    return sum(1 for indicator in indicators if indicator in message_lower)