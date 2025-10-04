# services/chat_manager.py
"""Chat and conversation management service."""
import logging
import json
from typing import Dict, Optional, List, Any, Literal
from datetime import datetime
import uuid

from models.chat import (
    ChatMessage, 
    ConversationContext,
    ChatResponse
)
from services.sql_converter import sql_converter
from services.query_executor import query_executor
from services.mcp_client import mcp_client
from config.settings import settings

logger = logging.getLogger(__name__)

# Optional OpenAI import for follow-up questions
try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    AsyncOpenAI = None

class ChatManager:
    """Manages chat conversations and context."""
    
    def __init__(self):
        self.conversations: Dict[str, ConversationContext] = {}
        self.llm = None
        
        if OPENAI_AVAILABLE and settings.OPENAI_API_KEY:
            self.llm = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            logger.info("OpenAI client initialized for chat management")
    
    def create_session(self) -> str:
        """Create a new chat session."""
        session_id = str(uuid.uuid4())
        self.conversations[session_id] = ConversationContext(session_id=session_id)
        logger.info(f"Created new chat session: {session_id}")
        return session_id
    
    def get_context(self, session_id: str) -> Optional[ConversationContext]:
        """Get conversation context for a session."""
        return self.conversations.get(session_id)
    
    def add_message(self, session_id: str, role: str, content: str, metadata: Dict = None):
        """Add a message to the conversation."""
        if session_id not in self.conversations:
            self.conversations[session_id] = ConversationContext(session_id=session_id)
        
        context = self.conversations[session_id]
        message = ChatMessage(role=role, content=content, metadata=metadata)
        context.messages.append(message)
        context.updated_at = datetime.now()
    
    def clean_sql_query(self, sql: str) -> str:
        """Clean SQL query from any markdown or formatting."""
        # Remove markdown code blocks
        sql = sql.replace("```sql", "").replace("```", "")
        # Remove any leading/trailing whitespace and semicolons
        sql = sql.strip().rstrip(";").strip()
        return sql
    
    async def process_message(self, session_id: str, message: str) -> ChatResponse:
        """Process a chat message and determine action."""
        
        # Ensure session exists
        if session_id not in self.conversations:
            self.conversations[session_id] = ConversationContext(session_id=session_id)
        
        context = self.conversations[session_id]
        
        # Add user message to history
        self.add_message(session_id, "user", message)
        
        # Determine message intent with improved classification
        intent = await self._determine_intent_advanced(message, context)
        
        logger.info(f"Determined intent: {intent} for message: {message[:100]}")
        
        if intent == "follow_up":
            # Handle follow-up question on existing results
            return await self._handle_follow_up(session_id, message, context)
        elif intent == "modify_query":
            # User wants to modify the last query
            return await self._handle_query_modification(session_id, message, context)
        elif intent == "execute_immediately":
            # User wants to execute query immediately
            return await self._handle_immediate_execution(session_id, message, context)
        else:
            # Generate query and ask for confirmation
            return await self._handle_query_generation(session_id, message, context)
    
    async def _determine_intent_advanced(self, message: str, context: ConversationContext) -> str:
        """Advanced intent determination using LLM if available, otherwise rule-based."""
        
        message_lower = message.lower()
        
        # Check for execution commands with actual queries
        # "execute show all tables" should generate and execute new query
        # "execute" alone should execute pending query
        execute_keywords = ["execute", "run", "go ahead"]
        has_execute = any(keyword in message_lower for keyword in execute_keywords)
        
        if has_execute:
            # Check if it's just a confirmation (short message with only execute command)
            clean_message = message_lower
            for keyword in execute_keywords:
                clean_message = clean_message.replace(keyword, "").strip()
            
            # If there's substantial content after removing execute keywords, it's a new query
            if len(clean_message) > 10:  # More than just "it" or similar
                return "execute_immediately"
            # If it's just "execute" or "execute it", check for pending query
            elif context.last_sql:
                return "execute_immediately"
        
        # If no previous results, it's definitely a new query
        if not context.last_result:
            return "new_query"
        
        # Use LLM for better classification if available
        if self.llm:
            try:
                intent = await self._classify_with_llm(message, context)
                return intent
            except Exception as e:
                logger.error(f"LLM classification failed: {e}")
                # Fall back to rule-based
        
        # Rule-based classification
        return self._classify_with_rules(message, context)
    
    async def _classify_with_llm(self, message: str, context: ConversationContext) -> str:
        """Use LLM to classify the intent of the message."""
        
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
1. "follow_up" - User is asking a question about the existing results (e.g., "what's the maximum value?", "how many rows have X?", "which record has Y?")
2. "modify_query" - User wants to modify/change the last SQL query (e.g., "add a filter", "include another column", "change the sort order")
3. "new_query" - User is asking for a completely new database query (e.g., "show me different table", "get data about something else")
4. "execute_immediately" - User explicitly wants to execute a query

Important rules:
- If the message references "these results", "this data", "the above", it's likely a follow_up
- If the message asks for analysis, statistics, or specific values from existing data, it's follow_up
- If the message mentions modifying, changing, or adjusting the query, it's modify_query
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
        
        # Default to new_query if classification is invalid
        logger.warning(f"Invalid LLM classification: {classification}, defaulting to new_query")
        return "new_query"
    
    def _classify_with_rules(self, message: str, context: ConversationContext) -> str:
        """Rule-based classification fallback."""
        message_lower = message.lower()
        
        # Strong indicators for follow-up questions
        follow_up_indicators = [
            "these results", "this data", "the above", "from the results",
            "in the table", "which row", "which record", "maximum", "minimum",
            "average", "sum of", "count of", "how many rows", "what's the highest",
            "what's the lowest", "analyze", "tell me about", "explain the data"
        ]
        
        # Strong indicators for query modification
        modify_indicators = [
            "modify", "change", "update the query", "add to the query",
            "also include", "instead of", "but with", "adjust", "refine"
        ]
        
        # Count matches
        follow_up_score = sum(1 for indicator in follow_up_indicators if indicator in message_lower)
        modify_score = sum(1 for indicator in modify_indicators if indicator in message_lower)
        
        # Decision logic
        if follow_up_score > modify_score and follow_up_score > 0:
            return "follow_up"
        elif modify_score > 0:
            return "modify_query"
        
        # Check if it's asking about completely different data
        if any(table_name.lower() in message_lower for table_name in ["user_tables", "all_users", "v$version"]):
            return "new_query"
        
        # Default to new query if unclear
        return "new_query"
    
    async def _handle_query_generation(self, session_id: str, message: str, context: ConversationContext) -> ChatResponse:
        """Generate SQL query and ask for confirmation."""
        try:
            # Check MCP connection
            if not mcp_client.connected:
                await mcp_client.reconnect()
            
            # Generate SQL from natural language
            sql_query = await sql_converter.convert_to_sql(message, mcp_client.tools)
            
            # Clean the SQL query
            sql_query = self.clean_sql_query(sql_query)
            
            # Store generated SQL
            context.last_sql = sql_query
            
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
            
            # Add assistant message to history
            self.add_message(session_id, "assistant", confirmation_msg, 
                           {"sql_query": sql_query, "action": "confirmation"})
            
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
    
    async def _handle_immediate_execution(self, session_id: str, message: str, context: ConversationContext) -> ChatResponse:
        """Handle immediate query execution."""
        try:
            # Check if user is confirming execution of pending query
            message_lower = message.lower()
            
            # Only use last SQL if it's a simple execution confirmation
            if context.last_sql and message_lower in ["execute", "run", "yes", "go ahead", "execute it"]:
                sql_query = self.clean_sql_query(context.last_sql)
            else:
                # For any message with content, generate a new SQL query
                # Remove "execute" keyword to get the actual query
                query_text = message
                for keyword in ["execute", "run"]:
                    query_text = query_text.replace(keyword, "").strip()
                
                # Generate new SQL query from the actual request
                sql_query = await sql_converter.convert_to_sql(query_text if query_text else message, mcp_client.tools)
                sql_query = self.clean_sql_query(sql_query)
            
            # Execute the query
            result = await query_executor.execute_query(sql_query)
            
            # Check for errors
            if isinstance(result, dict) and "error" in result:
                error_msg = f"Query execution failed: {result['error']}"
                self.add_message(session_id, "assistant", error_msg)
                
                return ChatResponse(
                    message=error_msg,
                    sql_query=sql_query,
                    error=result["error"],
                    session_id=session_id,
                    action_type="query_execution"
                )
            
            # Store result in context
            context.last_result = result
            context.last_sql = sql_query
            context.last_query = message
            
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
            
            self.add_message(session_id, "assistant", success_msg, 
                           {"sql_query": sql_query, "result_count": row_count})
            
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
    
    async def _handle_query_modification(self, session_id: str, message: str, context: ConversationContext) -> ChatResponse:
        """Handle query modification requests."""
        if not context.last_sql:
            return await self._handle_query_generation(session_id, message, context)
        
        try:
            # Use LLM to modify the existing query based on user feedback
            if self.llm:
                modified_sql = await self._modify_query_with_llm(context.last_sql, message)
            else:
                # Simple modification without LLM
                modified_sql = await sql_converter.convert_to_sql(message, mcp_client.tools)
            
            # Clean the modified SQL
            modified_sql = self.clean_sql_query(modified_sql)
            
            # Store modified SQL
            context.last_sql = modified_sql
            
            # Create confirmation message
            confirmation_msg = f"""I've modified the query based on your feedback:

```sql
{modified_sql}
```

Would you like to:
1. **Execute** this modified query
2. **Modify** it further
3. **Cancel** and start over"""
            
            self.add_message(session_id, "assistant", confirmation_msg, 
                           {"sql_query": modified_sql, "action": "modification"})
            
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
    
    async def _handle_follow_up(self, session_id: str, question: str, context: ConversationContext) -> ChatResponse:
        """Handle follow-up questions on existing results."""
        if not context.last_result:
            return ChatResponse(
                message="I don't have any previous results to analyze. Please execute a query first.",
                session_id=session_id,
                action_type="follow_up"
            )
        
        try:
            # Use LLM to answer follow-up questions
            if self.llm:
                answer = await self._answer_with_llm(question, context.last_result, context.last_sql)
            else:
                answer = self._answer_without_llm(question, context.last_result)
            
            # Add note about the data source
            answer += f"\n\n*This answer is based on the {context.last_result.get('row_count', len(context.last_result.get('rows', [])))} rows from your last query.*"
            
            self.add_message(session_id, "assistant", answer, 
                           {"action": "follow_up", "question": question})
            
            return ChatResponse(
                message=answer,
                session_id=session_id,
                action_type="follow_up"
            )
            
        except Exception as e:
            logger.error(f"Follow-up question handling failed: {e}")
            return ChatResponse(
                message=f"I couldn't process your follow-up question: {str(e)}",
                session_id=session_id,
                error=str(e),
                action_type="follow_up"
            )
    
    async def _modify_query_with_llm(self, original_sql: str, modification_request: str) -> str:
        """Modify SQL query using LLM."""
        # Clean the original SQL first
        original_sql = self.clean_sql_query(original_sql)
        
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
        # Clean any potential formatting that slipped through
        return self.clean_sql_query(modified_sql)
    
    async def _answer_with_llm(self, question: str, result: Dict[str, Any], last_sql: str = None) -> str:
        """Answer follow-up questions using LLM."""
        # Prepare data summary
        if 'columns' in result and 'rows' in result:
            # Limit data for LLM context
            sample_rows = result['rows'][:100]  # First 100 rows
            data_summary = {
                "columns": result['columns'],
                "total_rows": result.get('row_count', len(result['rows'])),
                "sample_data": sample_rows
            }
        else:
            data_summary = result
        
        prompt = f"""You are a data analyst assistant. Answer the user's question based ONLY on the query results provided.

{f"SQL Query that generated this data: {last_sql}" if last_sql else ""}

Query Results:
{json.dumps(data_summary, indent=2)}

User's Question:
{question}

Important:
- Answer based ONLY on the data provided above
- If the question cannot be answered from this data, clearly state what information is missing
- Provide specific values, counts, or calculations from the data
- Be precise and factual
- Do not make assumptions about data that isn't shown
- Do not use any markdown formatting in your response"""
        
        response = await self.llm.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful data analyst. Answer questions based strictly on the provided data. Provide plain text answers without markdown formatting."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1  # Lower temperature for more factual responses
        )
        
        return response.choices[0].message.content.strip()
    
    def _answer_without_llm(self, question: str, result: Dict[str, Any]) -> str:
        """Answer follow-up questions without LLM (basic analysis)."""
        question_lower = question.lower()
        
        if 'columns' in result and 'rows' in result:
            row_count = len(result['rows'])
            col_count = len(result['columns'])
            columns = result['columns']
            rows = result['rows']
            
            # Basic statistics
            if "how many" in question_lower:
                return f"The query returned {row_count} rows with {col_count} columns."
            elif "columns" in question_lower or "fields" in question_lower:
                return f"The columns are: {', '.join(columns)}"
            elif "first" in question_lower and "row" in question_lower:
                if rows:
                    first_row = dict(zip(columns, rows[0]))
                    return f"First row: {json.dumps(first_row, indent=2)}"
                return "No data available."
            elif "last" in question_lower and "row" in question_lower:
                if rows:
                    last_row = dict(zip(columns, rows[-1]))
                    return f"Last row: {json.dumps(last_row, indent=2)}"
                return "No data available."
            else:
                return f"""The query returned {row_count} rows. 
                
To analyze this data, you can ask specific questions like:
- "What's the maximum/minimum value in [column]?"
- "How many rows have [condition]?"
- "Show me the first/last row"
- "What are the unique values in [column]?"

Or request a new query for different data."""
        
        return "I need more specific information to answer your question. Please provide details about what you'd like to know from the data."
    
    def clear_session(self, session_id: str):
        """Clear a conversation session."""
        if session_id in self.conversations:
            del self.conversations[session_id]
            logger.info(f"Cleared session: {session_id}")

# Singleton instance
chat_manager = ChatManager()