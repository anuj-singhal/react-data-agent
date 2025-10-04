# services/chat/intent_classifier.py
"""Intent classification for chat messages."""
import logging
from typing import Optional
from config.settings import settings
from models.chat import ConversationContext
from .utils import count_intent_indicators, is_simple_confirmation, extract_query_from_execute_command

logger = logging.getLogger(__name__)

# Optional OpenAI import
try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    AsyncOpenAI = None

class IntentClassifier:
    """Classifies user message intent in database conversations."""
    
    # Intent indicators for rule-based classification
    FOLLOW_UP_INDICATORS = [
        "these results", "this data", "the above", "from the results",
        "in the table", "which row", "which record", "maximum", "minimum",
        "average", "sum of", "count of", "how many rows", "what's the highest",
        "what's the lowest", "analyze", "tell me about", "explain the data",
        "from these", "in this result", "among these"
    ]
    
    MODIFY_INDICATORS = [
        "modify", "change", "update the query", "add to the query",
        "also include", "instead of", "but with", "adjust", "refine",
        "alter the query", "edit", "revise"
    ]
    
    EXECUTE_KEYWORDS = ["execute", "run", "go ahead"]
    
    def __init__(self):
        self.llm = None
        if OPENAI_AVAILABLE and settings.OPENAI_API_KEY:
            self.llm = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            logger.info("OpenAI client initialized for intent classification")
    
    async def classify_intent(self, message: str, context: ConversationContext) -> str:
        """
        Classify the intent of a user message.
        
        Args:
            message: User message
            context: Conversation context
            
        Returns:
            Intent string: 'follow_up', 'modify_query', 'execute_immediately', or 'new_query'
        """
        message_lower = message.lower()
        
        # Check for execution commands
        if await self._is_execute_intent(message, context):
            return "execute_immediately"
        
        # If no previous results, it's definitely a new query
        if not context.last_result:
            return "new_query"
        
        # Use LLM for better classification if available
        if self.llm:
            try:
                intent = await self._classify_with_llm(message, context)
                logger.info(f"LLM classified intent as: {intent}")
                return intent
            except Exception as e:
                logger.error(f"LLM classification failed: {e}")
        
        # Fall back to rule-based classification
        return self._classify_with_rules(message, context)
    
    async def _is_execute_intent(self, message: str, context: ConversationContext) -> bool:
        """
        Check if message indicates immediate execution intent.
        
        Args:
            message: User message
            context: Conversation context
            
        Returns:
            True if execute intent, False otherwise
        """
        message_lower = message.lower()
        
        # Check for execute keywords
        has_execute = any(keyword in message_lower for keyword in self.EXECUTE_KEYWORDS)
        
        if not has_execute:
            return False
        
        # Check if it's a simple confirmation
        if is_simple_confirmation(message):
            return bool(context.last_sql)  # Only if there's a pending query
        
        # Extract content after execute keywords
        clean_message = extract_query_from_execute_command(message)
        
        # If substantial content remains, it's a new query to execute
        return len(clean_message) > 10
    
    async def _classify_with_llm(self, message: str, context: ConversationContext) -> str:
        """
        Use LLM to classify the intent of the message.
        
        Args:
            message: User message
            context: Conversation context
            
        Returns:
            Intent classification
        """
        # Prepare context information
        has_results = context.last_result is not None
        last_query_info = ""
        
        if context.last_sql:
            last_query_info = f"Last executed SQL: {context.last_sql[:200]}"
            if has_results:
                row_count = context.last_result.get('row_count', len(context.last_result.get('rows', [])))
                columns = context.last_result.get('columns', [])
                last_query_info += f"\nResult: {row_count} rows with columns: {', '.join(columns[:10])}"
        
        prompt = f"""Classify the user's intent for their message in a database conversation.

Context:
- Previous query executed: {has_results}
{last_query_info if last_query_info else '- No previous query'}

User's message: "{message}"

Classify as EXACTLY one of:
1. "follow_up" - User is asking a question about the existing results
2. "modify_query" - User wants to modify/change the last SQL query
3. "new_query" - User is asking for a completely new database query
4. "execute_immediately" - User explicitly wants to execute a query

Important rules:
- If the message references "these results", "this data", "the above", it's likely a follow_up
- If the message asks for analysis or specific values from existing data, it's follow_up
- If the message mentions modifying or changing the query, it's modify_query
- If the message is about completely different data or tables, it's new_query
- Only return the classification word, nothing else

Return ONLY one of: follow_up, modify_query, new_query, execute_immediately"""
        
        response = await self.llm.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a classification assistant. Return only the classification label."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=20
        )
        
        classification = response.choices[0].message.content.strip().lower()
        
        # Validate the classification
        valid_intents = ["follow_up", "modify_query", "new_query", "execute_immediately"]
        if classification in valid_intents:
            return classification
        
        logger.warning(f"Invalid LLM classification: {classification}, defaulting to new_query")
        return "new_query"
    
    def _classify_with_rules(self, message: str, context: ConversationContext) -> str:
        """
        Rule-based classification fallback.
        
        Args:
            message: User message
            context: Conversation context
            
        Returns:
            Intent classification
        """
        message_lower = message.lower()
        
        # Count intent indicators
        follow_up_score = count_intent_indicators(message, self.FOLLOW_UP_INDICATORS)
        modify_score = count_intent_indicators(message, self.MODIFY_INDICATORS)
        
        logger.info(f"Rule-based scores - Follow-up: {follow_up_score}, Modify: {modify_score}")
        
        # Decision logic
        if follow_up_score > modify_score and follow_up_score > 0:
            return "follow_up"
        elif modify_score > 0:
            return "modify_query"
        
        # Check for specific table references (indicates new query)
        new_query_tables = ["user_tables", "all_users", "v$version"]
        if any(table in message_lower for table in new_query_tables):
            return "new_query"
        
        # Default to new query if unclear
        return "new_query"