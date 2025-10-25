# services/chat/manager.py
"""Main chat manager orchestrating all chat operations."""
import logging
from typing import Optional
from models.chat import ChatResponse, ConversationContext
from .context_manager import ContextManager
from .intent_classifier import IntentClassifier
from .query_handler import QueryHandler
from .followup_handler import FollowUpHandler
from services.rag_agent.rag_system import rag_agent

logger = logging.getLogger(__name__)

class ChatManager:
    """Main orchestrator for chat conversations."""
    
    def __init__(self):
        """Initialize chat manager with all sub-components."""
        self.context_manager = ContextManager()
        self.intent_classifier = IntentClassifier()
        self.query_handler = QueryHandler()
        self.followup_handler = FollowUpHandler()
        logger.info("Chat manager initialized with all components")
    
    def create_session(self) -> str:
        """
        Create a new chat session.
        
        Returns:
            Session ID
        """
        return self.context_manager.create_session()
    
    def get_context(self, session_id: str) -> Optional[ConversationContext]:
        """
        Get conversation context for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            ConversationContext or None
        """
        return self.context_manager.get_context(session_id)
    
    def clear_session(self, session_id: str):
        """
        Clear a conversation session.
        
        Args:
            session_id: Session identifier
        """
        self.context_manager.clear_session(session_id)
    
    async def process_message(self, session_id: str, message: str) -> ChatResponse:
        """
        Process a chat message and determine appropriate action.
        
        Args:
            session_id: Session identifier
            message: User message
            
        Returns:
            ChatResponse with appropriate action
        """
        # Ensure session exists
        context = self.context_manager.ensure_session(session_id)
        
        # Add user message to history
        self.context_manager.add_message(session_id, "user", message)
        
        # Classify intent
        intent = await self.intent_classifier.classify_intent(message, context)
        logger.info(f"Processing message with intent: {intent}")
        
        # Route to appropriate handler
        if intent == "follow_up":
            response = await self._handle_follow_up(session_id, message, context)
        elif intent == "modify_query":
            response = await self._handle_modify_query(session_id, message, context)
        elif intent == "execute_immediately":
            response = await self._handle_execute_immediately(session_id, message, context)
        else:  # new_query
            response = await self._handle_new_query(session_id, message, context)
        
        # Add assistant response to history
        self.context_manager.add_message(
            session_id, 
            "assistant", 
            response.message,
            {"action": response.action_type, "sql_query": response.sql_query}
        )
        
        # Update context if query was executed
        if response.result and response.sql_query:
            self.context_manager.update_last_query(
                session_id,
                response.sql_query,
                response.result
            )
        elif response.sql_query and response.requires_confirmation:
            # Store pending SQL
            context.last_sql = response.sql_query
        
        return response
    
    async def _handle_new_query(
        self, 
        session_id: str, 
        message: str, 
        context: ConversationContext
    ) -> ChatResponse:
        """Handle new query generation."""
        return await self.query_handler.generate_query(message, session_id)
    
    async def _handle_execute_immediately(
        self, 
        session_id: str, 
        message: str, 
        context: ConversationContext
    ) -> ChatResponse:
        """Handle immediate query execution."""
        response = await self.query_handler.execute_query_immediately(
            message, context, session_id
        )
        
        # Update context with results
        if response.result and response.sql_query:
            context.last_result = response.result
            context.last_sql = response.sql_query
            context.last_query = message
        
        return response
    
    async def _handle_modify_query(
        self, 
        session_id: str, 
        message: str, 
        context: ConversationContext
    ) -> ChatResponse:
        """Handle query modification."""
        if not context.last_sql:
            # No query to modify, treat as new query
            return await self._handle_new_query(session_id, message, context)
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

        search_results = rag_agent.search_relevant_tables(rag_messages)
        tables = search_results['tables']
        
        print(f"Identified tables: {tables}")
        # Step 3: Build context for LLM
        rag_context = rag_agent.build_context(tables, message)

        response = await self.query_handler.modify_query(
            message, context.last_sql, session_id, previous_messages, rag_context
        )
        
        # Update pending SQL if modified
        if response.sql_query:
            context.last_sql = response.sql_query
        
        return response
    
    async def _handle_follow_up(
        self, 
        session_id: str, 
        message: str, 
        context: ConversationContext
    ) -> ChatResponse:
        """Handle follow-up question."""
        return await self.followup_handler.handle_follow_up(
            message,
            context.last_result,
            context.last_sql,
            session_id
        )
    
    def get_active_sessions_count(self) -> int:
        """Get count of active sessions."""
        return self.context_manager.get_session_count()
    
    def cleanup_old_sessions(self, hours: int = 24):
        """
        Clean up old sessions.
        
        Args:
            hours: Age threshold in hours
        """
        self.context_manager.cleanup_old_sessions(hours)

# Singleton instance
chat_manager = ChatManager()