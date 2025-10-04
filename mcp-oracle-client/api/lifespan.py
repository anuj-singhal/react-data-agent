# api/lifespan.py
"""Application lifespan management."""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from services.mcp_client import mcp_client
from config.settings import settings

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown."""
    # Startup
    logger.info("="*50)
    logger.info("Starting MCP Oracle Client API Server...")
    logger.info(f"MCP Server Path: {settings.MCP_SERVER_PATH}")
    logger.info(f"OpenAI Configured: {settings.is_openai_configured()}")
    logger.info("="*50)
    
    # Try to connect to MCP server
    connected = await mcp_client.connect()
    
    if not connected:
        logger.warning("Failed to connect to MCP server on startup")
        logger.warning("Will retry on first request")
    else:
        logger.info("Successfully connected to MCP server")
    
    yield
    
    # Shutdown
    logger.info("Shutting down MCP Oracle Client...")
    await mcp_client.disconnect()