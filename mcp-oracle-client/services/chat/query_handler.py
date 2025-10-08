# services/chat/query_handler.py
"""Query generation, modification, and execution handling."""
import logging
from typing import Dict, Optional
from config.settings import settings
from models.chat import ConversationContext, ChatResponse
from services.sql_converter import sql_converter
from services.query_executor import query_executor
from services.mcp_client import mcp_client
from .utils import clean_sql_query, extract_query_from_execute_command, is_simple_confirmation

logger = logging.getLogger(__name__)

# Optional OpenAI import
try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    AsyncOpenAI = None

class QueryHandler:
    """Handles query generation, modification, and execution."""
    
    def __init__(self):
        self.llm = None
        if OPENAI_AVAILABLE and settings.OPENAI_API_KEY:
            self.llm = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            logger.info("OpenAI client initialized for query handling")
    
    async def generate_query(self, message: str, intent_dict, rag_context, session_id: str) -> ChatResponse:
        """
        Generate SQL query from natural language.
        
        Args:
            message: User's natural language query
            session_id: Session identifier
            
        Returns:
            ChatResponse with generated SQL
        """
        try:
            # Check MCP connection
            if not mcp_client.connected:
                await mcp_client.reconnect()
            
            # Generate SQL from natural language
            sql_query = await sql_converter.convert_to_sql(message, intent_dict, rag_context, mcp_client.tools)
            sql_query = clean_sql_query(sql_query)
            
            # Create confirmation message
            confirmation_msg = f"""I've generated the following SQL query:

```sql
{sql_query}
```

Would you like to:
1. **Execute** this query as is
2. **Modify** the query (tell me what to change)
3. **Cancel** and start over

Please let me know how you'd like to proceed."""
            
            return ChatResponse(
                message=confirmation_msg,
                sql_query=sql_query,
                requires_confirmation=True,
                session_id=session_id,
                action_type="query_confirmation"
            )
            
        except Exception as e:
            logger.error(f"Query generation failed: {e}")
            return ChatResponse(
                message=f"I encountered an error generating the query: {str(e)}",
                session_id=session_id,
                error=str(e),
                action_type="general"
            )
    
    async def execute_query_immediately(
        self, 
        message: str, 
        context: ConversationContext,
        session_id: str
    ) -> ChatResponse:
        """
        Execute query immediately based on message.
        
        Args:
            message: User message
            context: Conversation context
            session_id: Session identifier
            
        Returns:
            ChatResponse with execution results
        """
        try:
            # Determine if we should use existing SQL or generate new
            if is_simple_confirmation(message) and context.last_sql:
                sql_query = clean_sql_query(context.last_sql)
                logger.info("Executing pending query")
            else:
                # Generate new SQL query from the message
                query_text = extract_query_from_execute_command(message)
                sql_query = await sql_converter.convert_to_sql(query_text, mcp_client.tools)
                sql_query = clean_sql_query(sql_query)
                logger.info(f"Generated and executing new query: {sql_query[:100]}")
            
            # Execute the query
            result = await query_executor.execute_query(sql_query)
            
            # Check for errors
            if isinstance(result, dict) and "error" in result:
                error_msg = f"Query execution failed: {result['error']}"
                return ChatResponse(
                    message=error_msg,
                    sql_query=sql_query,
                    error=result["error"],
                    session_id=session_id,
                    action_type="query_execution"
                )
            
            # Create success message
            row_count = result.get('row_count', len(result.get('rows', [])))
            success_msg = f"""Query executed successfully! 

**SQL Query:**
```sql
{sql_query}
```

**Results:** {row_count} rows returned

You can now:
- Ask questions about these specific results (e.g., "What's the maximum value?", "Which rows have X > Y?")
- Request a different query for new data
- Export the data

What would you like to do next?"""
            
            return ChatResponse(
                message=success_msg,
                sql_query=sql_query,
                result=result,
                session_id=session_id,
                action_type="query_execution"
            )
            
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return ChatResponse(
                message=f"I encountered an error executing the query: {str(e)}",
                session_id=session_id,
                error=str(e),
                action_type="query_execution"
            )
    
    async def modify_query(
        self, 
        message: str, 
        original_sql: str,
        session_id: str
    ) -> ChatResponse:
        """
        Modify an existing SQL query based on user feedback.
        
        Args:
            message: Modification request
            original_sql: Original SQL to modify
            session_id: Session identifier
            
        Returns:
            ChatResponse with modified SQL
        """
        try:
            # Use LLM to modify the query if available
            if self.llm:
                modified_sql = await self._modify_query_with_llm(original_sql, message)
            else:
                # Generate new query as fallback
                modified_sql = await sql_converter.convert_to_sql(message, mcp_client.tools)
            
            modified_sql = clean_sql_query(modified_sql)
            
            # Create confirmation message
            confirmation_msg = f"""I've modified the query based on your feedback:

```sql
{modified_sql}
```

Would you like to:
1. **Execute** this modified query
2. **Modify** it further
3. **Cancel** and start over"""
            
            return ChatResponse(
                message=confirmation_msg,
                sql_query=modified_sql,
                requires_confirmation=True,
                session_id=session_id,
                action_type="query_confirmation"
            )
            
        except Exception as e:
            logger.error(f"Query modification failed: {e}")
            return ChatResponse(
                message=f"I couldn't modify the query: {str(e)}",
                session_id=session_id,
                error=str(e),
                action_type="general"
            )
    
    async def _modify_query_with_llm(self, original_sql: str, modification_request: str) -> str:
        """
        Use LLM to modify SQL query.
        
        Args:
            original_sql: Original SQL query
            modification_request: User's modification request
            
        Returns:
            Modified SQL query
        """
        original_sql = clean_sql_query(original_sql)
        
        prompt = f"""You are an Oracle SQL expert. Modify the following SQL query based on the user's request.

Original SQL:
{original_sql}

User's modification request:
{modification_request}

Important:
- Return ONLY the modified SQL query
- Do NOT include any markdown formatting like ```sql or ```
- Do NOT include any explanations or comments
- Do NOT include semicolons at the end
- Return just the plain SQL text"""
        
        response = await self.llm.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are an Oracle SQL expert. Return only plain SQL queries without any formatting or markdown."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        modified_sql = response.choices[0].message.content.strip()
        return clean_sql_query(modified_sql)