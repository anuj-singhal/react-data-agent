# models/chat.py
"""Chat and conversation models."""
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field
from datetime import datetime

class ChatMessage(BaseModel):
    """Individual chat message."""
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = None

class ChatRequest(BaseModel):
    """Chat request model."""
    message: str = Field(..., min_length=1, description="User message")
    session_id: str = Field(..., description="Session ID for conversation context")
    include_history: bool = Field(default=True, description="Include conversation history")
    
class ChatResponse(BaseModel):
    """Chat response model."""
    message: str
    sql_query: Optional[str] = None
    requires_confirmation: bool = False
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    session_id: str
    action_type: Literal["query_confirmation", "query_execution", "follow_up", "general"]

class QueryConfirmationRequest(BaseModel):
    """Request for query confirmation."""
    session_id: str
    sql_query: str
    user_feedback: Optional[str] = None
    confirmed: bool

class FollowUpRequest(BaseModel):
    """Follow-up question request."""
    session_id: str
    question: str
    
class ConversationContext(BaseModel):
    """Conversation context for a session."""
    session_id: str
    messages: List[ChatMessage] = []
    last_query: Optional[str] = None
    last_result: Optional[Dict[str, Any]] = None
    last_sql: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)