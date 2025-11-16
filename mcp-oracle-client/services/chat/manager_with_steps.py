# services/chat/manager_with_steps.py
"""Extended chat manager that emits processing steps."""
import logging
from typing import Dict, List, Any
from datetime import datetime
from models.chat import ChatResponse, ProcessingStep
from services.chat.manager_with_intent_rag import ChatManagerWithIntent

logger = logging.getLogger(__name__)

class ChatManagerWithSteps(ChatManagerWithIntent):
    """Extended ChatManager that tracks and emits processing steps."""
    
    async def process_message_with_steps(self, session_id: str, message: str, 
                                         steps_store: Dict[str, List]) -> ChatResponse:
        """Process message while tracking steps."""
        
        def add_step(step_num: int, name: str, status: str, message: str, details: Dict = None):
            """Helper to add a processing step."""
            step = ProcessingStep(
                step_number=step_num,
                step_name=name,
                status=status,
                message=message,
                details=details or {},
                timestamp=datetime.now()
            )
            if session_id in steps_store:
                steps_store[session_id].append(step.dict())
        
        # Get the context
        context = self.context_manager.ensure_session(session_id)
        
        # Add user message to history
        self.context_manager.add_message(session_id, "user", message)
        
        # Step 0: NL Classification
        add_step(0, "NL Classification", "processing", "Determining if query is data-related...")
        try:
            classify_result = await self.nl_classify_agent.classify_query(message)
            if classify_result.classification != "process":
                add_step(0, "NL Classification", "completed", 
                        f"Query classified as: {classify_result.classification}", 
                        {"classification": classify_result.classification})
                return ChatResponse(
                    message=classify_result.answer + classify_result.message,
                    session_id=session_id,
                    action_type=classify_result.action_type,
                    processing_steps=steps_store.get(session_id, [])
                )
            add_step(0, "NL Classification", "completed", "Query is data-related ✓")
        except Exception as e:
            add_step(0, "NL Classification", "error", f"Error: {str(e)}")
            logger.error(f"NL Classification failed: {e}")

        # Step 1: Intent Classification
        add_step(1, "Intent Classification", "processing", "Analyzing query intent...")
        try:
            intent = await self.intent_classifier.classify_intent(message, context)
            add_step(1, "Intent Classification", "completed", 
                    f"Intent identified: {intent}", {"intent": intent})
        except Exception as e:
            add_step(1, "Intent Classification", "error", f"Error: {str(e)}")
            intent = "new_query"

        # Step 2: Semantic Intent Extraction (for new queries)
        semantic_intent = None
        if intent in ["new_query", "execute_immediately"]:
            add_step(2, "Semantic Analysis", "processing", "Extracting semantic intent...")
            try:
                semantic_intent = await self.intent_agent.extract_intent(
                    message,
                    {"last_sql": context.last_sql} if context.last_sql else None
                )
                intent_summary = {
                    "primary": semantic_intent.primary_intent,
                    "metrics": semantic_intent.metrics,
                    "dimensions": semantic_intent.dimensions
                }
                add_step(2, "Semantic Analysis", "completed", 
                        f"Semantic intent extracted", intent_summary)
            except Exception as e:
                add_step(2, "Semantic Analysis", "error", f"Error: {str(e)}")
                logger.warning(f"Semantic intent extraction failed: {e}")

        # Step 3: Query History Check (for new queries)
        if intent == "new_query":
            add_step(3, "Query History", "processing", "Checking query history for similar queries...")
            try:
                from services.rag_agent.rag_system import rag_agent
                cached_result = rag_agent.check_query_history(message)
                
                if cached_result and cached_result.get('similarity_score', 0) >= 0.95:
                    similarity = cached_result.get('similarity_score', 0)
                    add_step(3, "Query History", "completed", 
                            f"Found similar query (similarity: {similarity:.2%})", 
                            {"similarity": similarity, "cached": True})
                else:
                    add_step(3, "Query History", "completed", "No matching queries in history")
            except Exception as e:
                add_step(3, "Query History", "completed", "Skipped history check")

        # Step 4: RAG - Finding Relevant Tables
        if intent != "follow_up":
            add_step(4, "RAG Search", "processing", "Finding relevant tables and columns...")
            try:
                from services.rag_agent.rag_system import rag_agent
                
                # Build search query based on intent
                search_query = message
                if intent == "modify_query" and context.messages:
                    # Build context from previous messages
                    prev_messages = " ".join([m.content for m in context.messages if m.role == "user"])
                    search_query = f"{prev_messages} {message}"
                
                search_results = rag_agent.search_relevant_tables(search_query)
                tables = search_results['tables']
                
                add_step(4, "RAG Search", "completed", 
                        f"Found {len(tables)} relevant tables: {', '.join(tables[:5])}", 
                        {"tables": tables, "count": len(tables)})
            except Exception as e:
                add_step(4, "RAG Search", "error", f"Error: {str(e)}")
                tables = []

        # Continue with the original routing logic
        if intent == "follow_up":
            add_step(5, "Processing", "processing", "Analyzing follow-up question...")
            response = await self._handle_follow_up(session_id, message, context)
            add_step(5, "Processing", "completed", "Follow-up question answered ✓")
        elif intent == "modify_query":
            add_step(5, "Query Modification", "processing", "Modifying existing query...")
            response = await self._handle_new_query_with_intent_rag(session_id, message, intent, context)
            add_step(5, "Query Modification", "completed", "Query modified successfully ✓")
        elif intent == "execute_immediately":
            add_step(5, "Query Execution", "processing", "Executing query immediately...")
            response = await self._handle_execute_immediately(session_id, message, context)
            add_step(5, "Query Execution", "completed", "Query executed ✓")
        else:  # new_query
            # Clear session for new query
            if intent == "new_query":
                self.context_manager.clear_session(session_id)
                context = self.context_manager.ensure_session(session_id)
                self.context_manager.add_message(session_id, "user", message)
            
            add_step(5, "SQL Generation", "processing", "Generating SQL query...")
            if semantic_intent:
                response = await self._handle_new_query_with_intent_rag(
                    session_id, message, intent, context, semantic_intent
                )
            else:
                response = await self._handle_new_query(session_id, message, intent, context)
            add_step(5, "SQL Generation", "completed", "SQL query generated ✓")

        # Add processing steps to response
        response.processing_steps = steps_store.get(session_id, [])
        
        # Add assistant response to history
        self.context_manager.add_message(
            session_id, 
            "assistant", 
            response.message,
            {"action": response.action_type, "sql_query": response.sql_query}
        )
        
        # Update context if needed
        if response.result and response.sql_query:
            self.context_manager.update_last_query(
                session_id,
                response.sql_query,
                response.result
            )
        elif response.sql_query and response.requires_confirmation:
            context.last_sql = response.sql_query
        
        return response