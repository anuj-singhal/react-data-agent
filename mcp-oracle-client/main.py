# main.py - Minimal changes for Intent Agent integration
"""FastAPI application with minimal Intent Agent integration."""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings
from config.logging_config import setup_logging
from services.mcp_client import mcp_client
from services.rag_agent.rag_system import rag_agent

# Import routers - NO CHANGES to existing routers needed!
from routers import query, database, health

# For chat router, we have two options:
# Option 1: Import the enhanced chat router (if you created chat_enhanced.py)
# from routers import chat_enhanced as chat

# Option 2: Just update the import in your existing chat.py file
# Change: from services.chat.manager import chat_manager
# To: from services.chat.manager_with_intent import ChatManagerWithIntent as ChatManager
# And instantiate: chat_manager = ChatManager()
from routers import chat

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting MCP Oracle Client API...")
    
    # Connect to MCP server
    connected = await mcp_client.connect()
    if connected:
        logger.info("Successfully connected to MCP Oracle server")
    else:
        logger.error("Failed to connect to MCP Oracle server")
    
    rag_initialize = rag_agent.initialize_rag()

    if rag_initialize:
        logger.info("Successfully Initialized RAG")
    else:
        logger.error("Failed to Initialized RAG")

    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down MCP Oracle Client API...")
    await mcp_client.disconnect()
    logger.info("MCP Oracle Client API shutdown complete")


# Create FastAPI app - NO CHANGES
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=settings.API_DESCRIPTION + " (with Intent Agent)",
    lifespan=lifespan
)

# Configure CORS - NO CHANGES
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.CORS_ALLOW_METHODS,
    allow_headers=settings.CORS_ALLOW_HEADERS,
)

# Include routers - NO CHANGES
app.include_router(chat.router)
app.include_router(query.router)
app.include_router(database.router)
app.include_router(health.router)

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "MCP Oracle Client API",
        "version": settings.API_VERSION,
        "mcp_connected": mcp_client.connected,
        "intent_agent": "enabled"  # Just add this line
    }

if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting server on {settings.HOST}:{settings.PORT}")
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        log_level=settings.LOG_LEVEL
    )