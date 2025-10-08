# services/chat/manager_with_intent.py
"""Modified chat manager with minimal Intent Agent integration."""
import logging
from typing import Optional
from models.chat import ChatResponse, ConversationContext
from services.chat.manager import ChatManager
from services.chat.intent_agent import IntentAgent, SemanticIntent
from services.sql_converter import sql_converter
from services.query_executor import query_executor
from services.mcp_client import mcp_client
from services.rag_agent.graph_context_retriever import graph_rag_retriever

logger = logging.getLogger(__name__)


class ChatManagerWithIntent(ChatManager):
    """
    Extended ChatManager that adds Intent Agent capabilities
    while preserving all existing functionality.
    """
    
    def __init__(self):
        """Initialize with Intent Agent added to existing components."""
        super().__init__()
        self.intent_agent = IntentAgent()
        logger.info("Chat manager initialized with Intent Agent extension")
    
    async def process_message(self, session_id: str, message: str) -> ChatResponse:
        """
        Process message with optional semantic intent extraction.
        Falls back to original behavior if intent extraction fails.
        """
        # Get the context
        context = self.context_manager.ensure_session(session_id)
        
        # Add user message to history (as before)
        self.context_manager.add_message(session_id, "user", message)
        
        # Classify intent (existing functionality)
        intent = await self.intent_classifier.classify_intent(message, context)
        logger.info(f"Processing message with intent: {intent}")
        
        # Try to extract semantic intent for new queries only
        semantic_intent = None
        if intent in ["new_query", "execute_immediately"]:
            try:
                semantic_intent = await self.intent_agent.extract_intent(
                    message,
                    {"last_sql": context.last_sql} if context.last_sql else None
                )
                logger.info(f"Semantic intent extracted: {semantic_intent.to_dict()}")
                
                # Store in context for reference
                if hasattr(context, '__dict__'):
                    context.__dict__['last_semantic_intent'] = semantic_intent.to_dict()
                    
            except Exception as e:
                logger.warning(f"Semantic intent extraction failed, continuing normally: {e}")
                # Continue with normal flow
        
        # Use the original routing logic
        if intent == "follow_up":
            response = await self._handle_follow_up(session_id, message, context)
        elif intent == "modify_query":
            response = await self._handle_modify_query(session_id, message, context)
        elif intent == "execute_immediately":
            response = await self._handle_execute_immediately(session_id, message, context)
        else:  # new_query
            # Try enhanced query generation if we have semantic intent
            if semantic_intent:
                response = await self._handle_new_query_with_intent(
                    session_id, message, context, semantic_intent
                )
            else:
                response = await self._handle_new_query(session_id, message, context)
        
        # Add assistant response to history (as before)
        self.context_manager.add_message(
            session_id, 
            "assistant", 
            response.message,
            {"action": response.action_type, "sql_query": response.sql_query}
        )
        
        # Update context if query was executed (as before)
        if response.result and response.sql_query:
            self.context_manager.update_last_query(
                session_id,
                response.sql_query,
                response.result
            )
        elif response.sql_query and response.requires_confirmation:
            context.last_sql = response.sql_query
        
        return response
    
    # async def _handle_new_query_with_intent(
    #     self, 
    #     session_id: str, 
    #     message: str, 
    #     context: ConversationContext,
    #     semantic_intent: SemanticIntent
    # ) -> ChatResponse:
    #     """
    #     Handle new query with semantic intent information.
    #     Falls back to regular handling if enhanced processing fails.
    #     """
    #     try:
    #         # Log the intent for better SQL generation context
    #         intent_info = f"Intent: {semantic_intent.primary_intent}"
    #         if semantic_intent.metrics:
    #             intent_info += f", Metrics: {semantic_intent.metrics}"
    #         if semantic_intent.dimensions:
    #             intent_info += f", Dimensions: {semantic_intent.dimensions}"
    #         logger.info(f"Generating SQL with {intent_info}")
            
    #         # Use the regular query handler but with intent context
    #         response = await self.query_handler.generate_query(message, session_id)
            
    #         # Enhance the confirmation message with intent understanding
    #         if response.requires_confirmation and semantic_intent:
    #             intent_summary = self._build_intent_summary(semantic_intent)
    #             if intent_summary:
    #                 # Insert intent summary before SQL
    #                 original_msg = response.message
    #                 sql_start = original_msg.find("```sql")
    #                 if sql_start > 0:
    #                     response.message = (
    #                         original_msg[:sql_start] + 
    #                         f"\n{intent_summary}\n\n" + 
    #                         original_msg[sql_start:]
    #                     )
            
    #         return response
            
    #     except Exception as e:
    #         logger.error(f"Enhanced query generation failed, using standard: {e}")
    #         return await self._handle_new_query(session_id, message, context)
    
    async def _handle_new_query_with_intent(self, session_id, message, context, semantic_intent):
        try:
            # Get RAG context from intent
            rag_context = None
            if semantic_intent:
                intent_dict = semantic_intent.to_dict()
                rag_context = await graph_rag_retriever.retrieve_context(intent_dict, k=3, depth=1)
                logger.info(f"RAG retrieved tables: {rag_context.primary_tables}")
            
            # Continue with existing SQL generation
            response = await self.query_handler.generate_query(message, intent_dict ,rag_context, session_id)
            
            # Optional: Add context info to confirmation message
            # if response.requires_confirmation and rag_context:
            #     # Add tables being used to the message
            #     tables_info = f"\n**Using Tables:** {', '.join(rag_context.primary_tables)}\n"
            #     response.message = response.message.replace("```sql", f"{tables_info}\n```sql")
            
            #Enhance the confirmation message with intent understanding
            if response.requires_confirmation and semantic_intent:
                intent_summary = self._build_intent_summary(semantic_intent)
                if intent_summary:
                    # Insert intent summary before SQL
                    original_msg = response.message
                    sql_start = original_msg.find("```sql")
                    if sql_start > 0:
                        response.message = (
                            original_msg[:sql_start] + 
                            f"\n{intent_summary}\n\n" + 
                            original_msg[sql_start:]
                        )
            

            return response
            
        except Exception as e:
            logger.error(f"RAG context failed: {e}")
            # Fallback to existing flow
            return await self._handle_new_query(session_id, message, context)

    def _build_intent_summary(self, intent: SemanticIntent) -> str:
        """Build a brief summary of what was understood from the query."""
        parts = []
        
        if intent.primary_intent:
            parts.append(f"**Understood:** {intent.primary_intent.replace('_', ' ').title()} query")
        
        if intent.metrics:
            parts.append(f"**Analyzing:** {', '.join(intent.metrics)}")
        
        if intent.dimensions:
            parts.append(f"**Grouped by:** {', '.join(intent.dimensions)}")
        
        if intent.filters:
            parts.append(f"**With filters:** {', '.join(intent.filters[:3])}")  # Limit to 3
        
        return "\n".join(parts) if parts else ""
    
    def get_last_semantic_intent(self, session_id: str) -> Optional[dict]:
        """Get the last semantic intent for a session."""
        context = self.get_context(session_id)
        if context and hasattr(context, '__dict__'):
            return context.__dict__.get('last_semantic_intent')
        return None


# ===== SIMPLE INTEGRATION IN EXISTING ROUTERS =====
# routers/chat_enhanced.py (or modify existing chat.py)

"""Enhanced chat endpoints with Intent Agent."""
import logging
from fastapi import APIRouter, HTTPException
from models.chat import ChatRequest, ChatResponse, QueryConfirmationRequest, FollowUpRequest
from services.chat.manager_with_intent import ChatManagerWithIntent
from services.query_executor import query_executor

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["Chat"])

# Use the enhanced manager
chat_manager = ChatManagerWithIntent()

# All existing endpoints remain EXACTLY the same
@router.post("/start")
async def start_chat():
    """Start a new chat session."""
    session_id = chat_manager.create_session()
    return {
        "session_id": session_id,
        "message": "Chat session started. How can I help you with your database queries?"
    }

@router.post("/message", response_model=ChatResponse)
async def send_message(request: ChatRequest):
    """Send a message in the chat."""
    try:
        response = await chat_manager.process_message(
            request.session_id,
            request.message
        )
        return response
    except Exception as e:
        logger.error(f"Chat message processing failed: {e}")
        return ChatResponse(
            message=f"I encountered an error: {str(e)}",
            session_id=request.session_id,
            error=str(e),
            action_type="general"
        )

@router.post("/confirm")
async def confirm_query(request: QueryConfirmationRequest):
    """Confirm and execute a generated query - UNCHANGED."""
    try:
        context = chat_manager.get_context(request.session_id)
        if not context:
            raise HTTPException(status_code=404, detail="Session not found")
        
        if request.confirmed:
            # Execute the query
            result = await query_executor.execute_query(request.sql_query)
            
            # Store result in context
            context.last_result = result
            context.last_sql = request.sql_query
            
            # Check for errors
            if isinstance(result, dict) and "error" in result:
                return {
                    "success": False,
                    "error": result["error"],
                    "message": f"Query execution failed: {result['error']}"
                }
            
            row_count = result.get('row_count', len(result.get('rows', [])))
            
            return {
                "success": True,
                "result": result,
                "message": f"Query executed successfully! {row_count} rows returned."
            }
        else:
            # User wants to modify
            if request.user_feedback:
                response = await chat_manager.process_message(
                    request.session_id,
                    request.user_feedback
                )
                return {
                    "success": True,
                    "message": response.message,
                    "sql_query": response.sql_query,
                    "requires_confirmation": response.requires_confirmation
                }
            else:
                return {
                    "success": True,
                    "message": "Query cancelled. Please provide a new query or modification."
                }
                
    except Exception as e:
        logger.error(f"Query confirmation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/followup")
async def follow_up_question(request: FollowUpRequest):
    """Ask a follow-up question about the last results - UNCHANGED."""
    try:
        context = chat_manager.get_context(request.session_id)
        if not context:
            raise HTTPException(status_code=404, detail="Session not found")
        
        if not context.last_result:
            return {
                "success": False,
                "message": "No previous results to analyze. Please execute a query first."
            }
        
        response = await chat_manager.process_message(
            request.session_id,
            request.question
        )
        
        return {
            "success": True,
            "message": response.message,
            "action_type": response.action_type
        }
        
    except Exception as e:
        logger.error(f"Follow-up question failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history/{session_id}")
async def get_chat_history(session_id: str):
    """Get chat history for a session - UNCHANGED."""
    context = chat_manager.get_context(session_id)
    if not context:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session_id,
        "messages": [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat()
            }
            for msg in context.messages
        ],
        "last_sql": context.last_sql,
        "has_results": context.last_result is not None
    }

# NEW ENDPOINT - Only addition
@router.get("/intent/{session_id}")
async def get_semantic_intent(session_id: str):
    """Get the last semantic intent extracted for a session."""
    semantic_intent = chat_manager.get_last_semantic_intent(session_id)
    context = chat_manager.get_context(session_id)
    
    if not context:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session_id,
        "semantic_intent": semantic_intent,
        "last_sql": context.last_sql if context else None
    }

# All other endpoints remain unchanged...