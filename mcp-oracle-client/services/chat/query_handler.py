"""
Enhanced Query Handler - Integrates complex query processing and self-correction
This is the updated version of services/chat/query_handler.py
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from colorama import Fore, Style

from models.chat import ChatResponse
from services.rag_agent.rag_system import rag_agent
from services.sql_converter_agent.llm_interface import sql_converter
from services.sql_validator_agent.sql_query_validator import sql_validator
from services.query_executor import query_executor
from .utils import clean_sql_query

# Import new components
from services.sql_converter_agent.complex_query_processor import ComplexQueryProcessor
from services.sql_validator_agent.self_correction_loop import LLMConversationManager

logger = logging.getLogger(__name__)

class QueryHandler:
    """
    Enhanced query handler with complex query processing and self-correction
    Handles both new_query and modify_query intents
    """
    
    def __init__(self):
        self.rag_agent = rag_agent
        self.sql_converter = sql_converter
        self.sql_validator = sql_validator
        self.query_executor = query_executor
        
        # Initialize new components
        self.complex_processor = ComplexQueryProcessor(sql_converter, rag_agent)
        self.llm_conversation = LLMConversationManager(sql_converter, sql_validator)
        
        # Configuration flags
        self.enable_complex_processing = False
        self.enable_self_correction = True
        self.confidence_threshold = 0.95  # For caching
        
    async def generate_query(self, message: str, session_id: str, intent: str, 
                            context: Any, rag_message: Optional[str] = None) -> ChatResponse:
        """
        Generate SQL query with complex processing and self-correction
        Handles both new_query and modify_query intents
        """
        try:
            # Extract previous messages and SQL for modify intent
            previous_messages = None
            last_sql = None
            
            if intent == "modify_query" and context:
                # Build previous messages context for modification
                previous_messages = self._build_previous_messages(context)
                last_sql = getattr(context, 'last_sql', None)
                
                print(f"\n{Fore.YELLOW}🔄 Modify Query Intent Detected{Style.RESET_ALL}")
                if last_sql:
                    print(f"  Previous SQL: {last_sql[:100]}...")
            
            # Step 3: Check query history for similar queries (only for new queries)
            print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
            print("Step 3 - Check query history for similar queries (only for new queries)")
            print(f"{Fore.CYAN}📋 Processing Query ({intent}): {message}...{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
            
            if intent == "new_query":
                cached_result = rag_agent.check_query_history(message)
                
                if cached_result and cached_result.get('similarity_score', 0) >= 0.95:
                    similarity_score = cached_result.get('similarity_score', 0)
                    print(f"\n{Fore.GREEN}✅ Cache Hit! Similarity: {similarity_score:.2%}{Style.RESET_ALL}")
                    # Ask LLM if the historical SQL can be reused
                    # similar_query = cached_result.get('sql_query', '')
                    print("Ask LLM if the historical SQL can be reused")
                    can_reuse = await self._verify_sql_reuse(message, cached_result)
                    
                    # For very high similarity, reuse directly
                    if similarity_score >= 0.95 and can_reuse['can_reuse']:
                        print("Very high similarity, reuse directly")
                        return self._create_cached_response(cached_result, session_id)
                    else:
                        # For 90-95% similarity, revalidate
                        print(f"{Fore.YELLOW}⚠️ Revalidating cached query{Style.RESET_ALL}")
            
            # Step 4: Find relevant tables using RAG
            print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
            print(f"\n{Fore.BLUE}🔍 Step 4: Finding relevant tables...{Style.RESET_ALL}")
            print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
            
            # For modify query, combine current and previous messages for better table discovery
            search_query = rag_message or message
            if intent == "modify_query" and previous_messages:
                search_query = f"{previous_messages} {message}"
            
            search_results = rag_agent.search_relevant_tables(search_query)
            tables = search_results['tables']
            
            if not tables:
                logger.warning("No relevant tables found")
                return ChatResponse(
                    message="I couldn't find relevant tables for your query. Could you provide more details?",
                    session_id=session_id,
                    error="No relevant tables found",
                    action_type="general"
                )
            
            print(f"  📊 Found {len(tables)} relevant tables: {', '.join(tables)}")
            
            # Step 5: Build context for LLM using RAG
            print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
            print(f"\n{Fore.BLUE}📝 Step 5: Building context with RAG...{Style.RESET_ALL}")
            print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
            llm_context = rag_agent.build_context(tables, message)
            
            # Step 6: Check if query is complex
            print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
            print(f"\n{Fore.BLUE}🔬 Step 6: Analyzing query complexity...{Style.RESET_ALL}")
            print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
            
            is_complex = False
            if self.enable_complex_processing:
                # For modify queries, check complexity of the modification request
                check_message = message if intent == "new_query" else f"{previous_messages} {message}"
                is_complex = await self.complex_processor.is_complex_query(check_message, llm_context)
                needs_decomposition = await self._check_needs_decomposition(check_message)
                if needs_decomposition and is_complex:
                    is_complex = True
                else:
                    is_complex = False
                print(f"  Query Type: {'Complex' if is_complex else 'Simple'}")
                print(f"  Intent: {intent}")
            
            # Step 7: Process based on complexity and intent
            if is_complex:
                print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
                print(f"\n{Fore.MAGENTA}🔧 Step 7: Processing complex query...{Style.RESET_ALL}")
                print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
                sql_query = await self._process_complex_query(
                    message, llm_context, intent, previous_messages, last_sql
                )
            else:
                print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
                print(f"\n{Fore.BLUE}⚡ Step 7: Processing simple query...{Style.RESET_ALL}")
                print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
                sql_query = await self._process_simple_query(
                    message, llm_context, intent, previous_messages, last_sql
                )
            
            if not sql_query:
                return ChatResponse(
                    message="Failed to generate SQL query. Please try rephrasing.",
                    session_id=session_id,
                    error="SQL generation failed",
                    action_type="general"
                )
            print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
            print(f"\n{Fore.GREEN}✅ Step 8: Generated Initial SQL:{Style.RESET_ALL}")
            print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
            print(f"  {sql_query}...")
            sql_query = clean_sql_query(sql_query)
            # Step 9: Validate and Self-Correct
            print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
            print(f"\n{Fore.BLUE}🔄 Step 9: Validation and Self-Correction...{Style.RESET_ALL}")
            print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")

            if self.enable_self_correction and not is_complex:
                correction_result = await self.llm_conversation.improve_query_through_conversation(
                    nl_query=message,
                    initial_sql=sql_query,
                    context=llm_context,
                    is_complex=is_complex
                )
                
                final_sql = correction_result['final_sql']
                final_confidence = correction_result['confidence']
                needs_correction = correction_result['needs_correction']
                final_sql = clean_sql_query(final_sql)

                if needs_correction:
                    print(f"  {Fore.YELLOW}✨ Applied {correction_result.get('attempts', 0)} correction(s){Style.RESET_ALL}")
                    print(f"  {Fore.GREEN}Confidence: {correction_result.get('initial_confidence', 0):.0%} → {final_confidence:.0%}{Style.RESET_ALL}")
                else:
                    print(f"  {Fore.GREEN}✅ No correction needed. Confidence: {final_confidence:.0%}{Style.RESET_ALL}")
            else:
                # Fallback to standard validation
                all_validations = await sql_validator.validate_sql(message, sql_query, llm_context)
                final_sql = sql_query
                final_confidence = all_validations.overall_confidence
                correction_result = None
                final_sql = clean_sql_query(final_sql)
            
            # Step 8: Add to query history if high confidence (only for new queries)
            if final_confidence >= self.confidence_threshold and intent == "new_query":
                print(f"\n{Fore.GREEN}💾 Step 8: Caching high-confidence query{Style.RESET_ALL}")
                
                validation_scores = {
                    'syntax': 1.0,
                    'schema': 1.0,
                    'semantic': final_confidence,
                    'completeness': final_confidence
                }
                
                rag_agent.add_validated_query_to_history(
                    nl_query=message,
                    sql_query=final_sql,
                    validation_result=validation_scores,
                    overall_confidence=final_confidence * 100,
                    variations=[rag_message] if rag_message else []
                )
            
            # Step 9: Create confirmation message
            print(f"\n{Fore.GREEN}✅ Step 9: Creating response...{Style.RESET_ALL}")
            
            return self._create_confirmation_response(
                final_sql, 
                session_id, 
                final_confidence,
                correction_result,
                intent
            )
            
        except Exception as e:
            logger.error(f"Query generation failed: {e}", exc_info=True)
            return ChatResponse(
                message=f"An error occurred while processing your query: {str(e)}",
                session_id=session_id,
                error=str(e),
                action_type="general"
            )
    
    async def _check_needs_decomposition(self, query: str) -> bool:
        """Check if query needs decomposition"""
        try:
            return await sql_converter.check_complexity(query)
        except:
            return False

    async def _process_complex_query(self, query: str, context: Dict[str, Any], 
                                intent: str, previous_messages: Optional[str] = None,
                                last_sql: Optional[str] = None) -> str:
        """
        Process complex query using the robust processor
        """
        try:
            # Use the more direct and reliable approach
            sql_query = await self.complex_processor.process_complex_query(
                query, context, intent, previous_messages, last_sql
            )
            
            # Log success
            logger.info(f"Complex query processed successfully: {sql_query[:100]}...")
            
            return sql_query
            
        except Exception as e:
            logger.error(f"Complex query processing failed: {e}")
            # Fallback to direct generation
            return await self._generate_sql_with_intent(
                context, intent, previous_messages, last_sql
            )

    async def _process_simple_query(self, query: str, context: Dict[str, Any], 
                                   intent: str, previous_messages: Optional[str] = None,
                                   last_sql: Optional[str] = None) -> str:
        """
        Process simple query directly
        Handles both new and modify intents
        """
        return await self._generate_sql_with_intent(
            context, intent, previous_messages, last_sql
        )
    
    async def _generate_sql_with_intent(self, context: Dict[str, Any], intent: str,
                                       previous_messages: Optional[str] = None,
                                       last_sql: Optional[str] = None) -> str:
        """
        Generate SQL based on intent (new_query or modify_query)
        This calls the appropriate method in sql_converter
        """
        if intent == "modify_query" and last_sql:
            # Use the modify SQL generation method
            return await sql_converter.generate_sql(
                context,
                intent="modify_query",
                previous_message=previous_messages,
                last_sql=last_sql
            )
        else:
            # Use the standard SQL generation method for new queries
            return await sql_converter.generate_sql(
                context,
                intent="new_query"
            )
    
    def _build_previous_messages(self, context: Any) -> str:
        """
        Build a string of previous messages from context for modify queries
        """
        if not context or not hasattr(context, 'messages'):
            return ""
        
        messages = []
        for msg in context.messages:
            if msg.role == "user":
                messages.append(f"User: {msg.content}")
        
        return " | ".join(messages[-3:])  # Use last 3 user messages for context
    
    async def _verify_sql_reuse(self, current_query: str, similar_query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ask LLM if the historical SQL can be reused for current query
        """
        print(f"\n{Fore.CYAN}🤔 Verifying if cached SQL can be reused...{Style.RESET_ALL}")
        print(f"  Original: '{similar_query['natural_language'][:60]}...'")
        print(f"  Current:  '{current_query[:60]}...'")
        
        prompt = f"""
        Determine if an existing SQL query can be reused for a new request.
        
        ORIGINAL NATURAL LANGUAGE QUERY: {similar_query['natural_language']}
        ORIGINAL GENERATED SQL: {similar_query['sql_query']}
        
        NEW NATURAL LANGUAGE QUERY: {current_query}
        
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

    def _create_cached_response(self, cached_result: Dict[str, Any], 
                               session_id: str) -> ChatResponse:
        """
        Create response from cached result
        """
        sql_query = cached_result.get('sql_query', '')
        confidence = cached_result.get('overall_confidence', 0) / 100
        
        message = f"""✨ **Found matching query in history!**

**Confidence:** {confidence:.0%} (from cache)

I'll execute this validated SQL query for you:

```sql
{sql_query}
```

Would you like to execute this query?"""
        
        return ChatResponse(
            message=message,
            sql_query=sql_query,
            requires_confirmation=True,
            session_id=session_id,
            action_type="query_confirmation"
        )
    
    def _create_confirmation_response(self, sql_query: str, session_id: str,
                                     confidence: float,
                                     correction_result: Optional[Dict] = None,
                                     intent: str = "new_query") -> ChatResponse:
        """
        Create confirmation response with confidence and correction info
        """
        # Determine confidence level and emoji
        if confidence >= 0.95:
            confidence_emoji = "🟢"
            confidence_text = "Very High"
        elif confidence >= 0.90:
            confidence_emoji = "🟢"
            confidence_text = "High"
        elif confidence >= 0.85:
            confidence_emoji = "🟡"
            confidence_text = "Good"
        elif confidence >= 0.80:
            confidence_emoji = "🟡"
            confidence_text = "Moderate"
        else:
            confidence_emoji = "🟠"
            confidence_text = "Low (Review Recommended)"
        
        message_parts = []
        
        # Add intent-specific header
        if intent == "modify_query":
            message_parts.append("I've modified your SQL query based on your request.\n")
        else:
            message_parts.append("I've generated a SQL query for your request.\n")
        
        # Add correction info if available
        if correction_result and correction_result.get('needs_correction'):
            attempts = correction_result.get('attempts', 0)
            initial_conf = correction_result.get('initial_confidence', 0)
            improvement = correction_result.get('improvement', 0)
            
            if attempts > 0:
                message_parts.append(
                    f"\n✨ **Query Improved:** Applied {attempts} automatic correction(s)"
                )
                if improvement > 0:
                    message_parts.append(
                        f"\n📈 **Confidence Increased:** {initial_conf:.0%} → {confidence:.0%} "
                        f"(+{improvement:.0%})"
                    )
        
        message_parts.append(
            f"\n\n**Validation Confidence:** {confidence_emoji} {confidence_text} ({confidence:.0%})"
        )
        
        # Add recommendation if available
        if correction_result and correction_result.get('recommendation'):
            message_parts.append(f"\n💡 **Recommendation:** {correction_result['recommendation']}")
        
        # Add warning for low confidence
        if confidence < 0.85:
            message_parts.append(
                f"\n\n⚠️ **Note:** Confidence is below recommended level. "
                f"Please review the query carefully before execution."
            )
        
        message_parts.extend([
            f"\n\n**{'Modified' if intent == 'modify_query' else 'Generated'} SQL:**",
            f"```sql",
            sql_query,
            "```",
            "\nWould you like to execute this query?"
        ])
        
        return ChatResponse(
            message="\n".join(message_parts),
            sql_query=sql_query,
            requires_confirmation=True,
            session_id=session_id,
            action_type="query_confirmation"
        )