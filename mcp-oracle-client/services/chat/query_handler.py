# services/chat/query_handler.py
"""Query generation, modification, and execution handling."""
import logging
from typing import Dict, Optional
from config.settings import settings
from models.chat import ConversationContext, ChatResponse
#from services.sql_converter_agent.llm_generator import sql_converter_bkp
from services.sql_converter_agent.llm_interface import sql_converter
from services.sql_validator_agent.sql_query_validator import sql_validator
from services.query_executor import query_executor
from services.mcp_client import mcp_client
from services.rag_agent.rag_system import rag_agent
from .utils import clean_sql_query, extract_query_from_execute_command, is_simple_confirmation
from typing import Dict, Any, Optional, List
from datetime import datetime
from colorama import Fore, Style

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
    
    async def _check_needs_decomposition(self, query: str) -> bool:
        """Check if query needs decomposition"""
        try:
            return await self.llm_interface.check_complexity(query)
        except:
            return False
        
    async def _verify_sql_reuse(self, current_query: str, similar_query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ask LLM if the historical SQL can be reused for current query
        """
        print(f"\n{Fore.CYAN}🤔 Verifying if cached SQL can be reused...{Style.RESET_ALL}")
        print(f"  Original: '{similar_query['natural_language'][:60]}...'")
        print(f"  Current:  '{current_query[:60]}...'")
        
        prompt = f"""
        Determine if an existing SQL query can be reused for a new request.
        
        ORIGINAL QUERY: {similar_query['natural_language']}
        ORIGINAL SQL: {similar_query['sql_query']}
        
        NEW QUERY: {current_query}
        
        Analyze if the SQL can be reused as-is or needs modification.
        Consider:
        1. Are the entities (tables, columns) the same?
        2. Are the filters/conditions equivalent?
        3. Is the aggregation/grouping the same?
        4. Is the time period the same or compatible?
        
        Return JSON:
        {{
            "can_reuse": true/false,
            "reason": "explanation",
            "confidence": 0-100
        }}
        """
        
        try:
            response = await sql_converter.analyze_query_similarity(prompt)
            
            # Parse response
            if isinstance(response, dict):
                can_reuse = response.get('can_reuse', False)
                reason = response.get('reason', '')
                confidence = response.get('confidence', 0)
                
                if can_reuse:
                    print(f"{Fore.GREEN}  ✓ SQL can be reused (confidence: {confidence}%){Style.RESET_ALL}")
                else:
                    print(f"{Fore.YELLOW}  ✗ SQL needs regeneration: {reason}{Style.RESET_ALL}")
                
                return response
            else:
                # Default to not reusing if parsing fails
                print(f"{Fore.YELLOW}  ✗ Unable to verify similarity{Style.RESET_ALL}")
                return {
                    "can_reuse": False,
                    "reason": "Unable to determine similarity",
                    "confidence": 0
                }
        except Exception as e:
            logger.error(f"SQL reuse verification failed: {e}")
            print(f"{Fore.RED}  ✗ Verification failed: {e}{Style.RESET_ALL}")
            return {
                "can_reuse": False,
                "reason": str(e),
                "confidence": 0
            }
    
    async def generate_query(self, message: str, session_id: str, intent,
                             p_context) -> ChatResponse:
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
            rag_messages, previous_messages, last_sql = self._get_messages(p_context)

            if(rag_messages):
                message = rag_messages

            # Step 1: Check query history for similar queries
            #cached_result = rag_agent.find_similar_query(message)
            cached_result = rag_agent.check_query_history(message)
            can_reuse = False

            if(cached_result):
                query = cached_result["matched_query"]
                can_reuse = await self._verify_sql_reuse(message, cached_result)

            if can_reuse:
                query = cached_result["matched_query"]
                #print(f"Found cached query with similarity: {cached_result["similarity_score"]:.2f}")
                cached_dict = {
                    'sql': query.sql_query,
                    'source': 'cache',
                    'similarity_score': cached_result["similarity_score"],
                    'query_id': query.id
                }
                sql_query = query.sql_query
                print(cached_dict)

            else:
                # Generate SQL from natural language
                # Step 2: Find relevant tables using RAG
                search_results = rag_agent.search_relevant_tables(message)
                tables = search_results['tables']

                print(f"Identified tables: {tables}")
                # Step 3: Build context for LLM
                context = rag_agent.build_context(tables, message)
                # Step 4: Generate SQL using LLM
                # Check complexity

                needs_decomposition = await self._check_needs_decomposition(message)
                
                if not needs_decomposition:
                    # Simple query
                    # result = await self._process_simple_with_validation(
                    #     enhanced_query, query, context, None
                    # )
                    print("Simple Query Processing...")
                else:
                    # Complex query
                    # result = await self._process_complex_with_validation(
                    #     enhanced_query, query, context, suggestions
                    # )
                    print("Complex Query Processing...")

                sql_query = await sql_converter.generate_sql(context, intent,  previous_messages, last_sql)
                sql_query = clean_sql_query(sql_query)

                syntax_validator = await sql_validator.validate_sql(message, sql_query, context)
                
                # sql_validation = SQLValidator(context)
                # overall_validation_result = await sql_validation.validate_sql(message, sql_query)

                # Add query to query_history if validated
                rag_agent.add_validated_query_to_history(
                        nl_query=message,
                        sql_query=sql_query,
                        validation_result=None,
                        overall_confidence=None,
                        variations=[]
                    )
                # sql_query = await sql_converter.convert_to_sql(message, intent_dict, mcp_client.tools)
                
            
            # Create confirmation message
            confirmation_msg = f"""I've generated/cached the following SQL query:

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
    
    def _get_messages(self, context):
        prompt_parts = []
        rag_parts = []
        sep = "\n\n"
        sep_2 = " also "
        for message in context.messages:
            role = message.role
            content = message.content
            if(role == "user"):
                prompt_parts.append(f"{role}: {content}")
                rag_parts.append(content)

        previous_messages = sep.join(prompt_parts)
        rag_messages = sep_2.join(rag_parts)
        
        return rag_messages, previous_messages, context.last_sql

    async def modify_query(
        self, 
        message: str, 
        original_sql: str,
        session_id: str,
        previous_messages,
        rag_context
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
                modified_sql = await self._modify_query_with_llm(original_sql, message, previous_messages)
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
    
    async def _modify_query_with_llm(self, original_sql: str, modification_request: str, previous_messages) -> str:
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

User's last requests and responses with last created SQL's:
{previous_messages}



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