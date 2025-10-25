# routers/chat.py
"""Chat and conversation endpoints."""
import logging
from fastapi import APIRouter, HTTPException
from models.chat import (
    ChatRequest,
    ChatResponse,
    QueryConfirmationRequest,
    FollowUpRequest
)

from pathlib import Path
from typing import Dict
import asyncio

#from services.chat import chat_manager
from services.chat.manager_with_intent_rag import ChatManagerWithIntent
chat_manager = ChatManagerWithIntent()
from services.query_executor import query_executor

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["Chat"])

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
        # Process the message
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
    """Confirm and execute a generated query."""
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
                # Process modification request
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
    """Ask a follow-up question about the last results."""
    try:
        context = chat_manager.get_context(request.session_id)
        if not context:
            raise HTTPException(status_code=404, detail="Session not found")
        
        if not context.last_result:
            return {
                "success": False,
                "message": "No previous results to analyze. Please execute a query first."
            }
        
        # Process follow-up question
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
    """Get chat history for a session."""
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

@router.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear a chat session."""
    chat_manager.clear_session(session_id)
    return {"message": "Session cleared successfully"}

@router.get("/stats")
async def get_chat_stats():
    """Get chat statistics."""
    return {
        "active_sessions": chat_manager.get_active_sessions_count(),
        "message": f"Currently {chat_manager.get_active_sessions_count()} active chat sessions"
    }

@router.post("/cleanup")
async def cleanup_sessions(hours: int = 24):
    """Clean up old sessions."""
    chat_manager.cleanup_old_sessions(hours)
    return {
        "message": f"Cleaned up sessions older than {hours} hours",
        "remaining_sessions": chat_manager.get_active_sessions_count()
    }