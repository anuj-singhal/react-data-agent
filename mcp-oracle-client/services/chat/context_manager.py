# services/chat/context_manager.py
"""Conversation context management for chat sessions."""
import logging
import uuid
from datetime import datetime
from typing import Dict, Optional

from models.chat import ChatMessage, ConversationContext

logger = logging.getLogger(__name__)

class ContextManager:
    """Manages conversation contexts and sessions."""
    
    def __init__(self):
        self.conversations: Dict[str, ConversationContext] = {}
    
    def create_session(self) -> str:
        """
        Create a new chat session.
        
        Returns:
            Session ID
        """
        session_id = str(uuid.uuid4())
        self.conversations[session_id] = ConversationContext(session_id=session_id)
        logger.info(f"Created new chat session: {session_id}")
        return session_id
    
    def get_context(self, session_id: str) -> Optional[ConversationContext]:
        """
        Get conversation context for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            ConversationContext or None if not found
        """
        return self.conversations.get(session_id)
    
    def ensure_session(self, session_id: str) -> ConversationContext:
        """
        Ensure a session exists, creating if necessary.
        
        Args:
            session_id: Session identifier
            
        Returns:
            ConversationContext
        """
        if session_id not in self.conversations:
            self.conversations[session_id] = ConversationContext(session_id=session_id)
        return self.conversations[session_id]
    
    def add_message(self, session_id: str, role: str, content: str, metadata: Dict = None):
        """
        Add a message to the conversation.
        
        Args:
            session_id: Session identifier
            role: Message role (user/assistant/system)
            content: Message content
            metadata: Optional metadata
        """
        context = self.ensure_session(session_id)
        message = ChatMessage(role=role, content=content, metadata=metadata)
        context.messages.append(message)
        context.updated_at = datetime.now()
    
    def update_last_query(self, session_id: str, sql: str, result: Dict = None):
        """
        Update the last query and result for a session.
        
        Args:
            session_id: Session identifier
            sql: SQL query
            result: Query result
        """
        context = self.ensure_session(session_id)
        context.last_sql = sql
        context.last_result = result
        context.updated_at = datetime.now()
    
    def clear_session(self, session_id: str):
        """
        Clear a conversation session.
        
        Args:
            session_id: Session identifier
        """
        if session_id in self.conversations:
            del self.conversations[session_id]
            logger.info(f"Cleared session: {session_id}")
    
    def get_session_count(self) -> int:
        """Get the number of active sessions."""
        return len(self.conversations)
    
    def cleanup_old_sessions(self, hours: int = 24):
        """
        Clean up sessions older than specified hours.
        
        Args:
            hours: Age threshold in hours
        """
        now = datetime.now()
        to_remove = []
        
        for session_id, context in self.conversations.items():
            age = (now - context.updated_at).total_seconds() / 3600
            if age > hours:
                to_remove.append(session_id)
        
        for session_id in to_remove:
            self.clear_session(session_id)
        
        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old sessions")