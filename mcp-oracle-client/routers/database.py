# routers/database.py
"""Database information endpoints."""
import logging
from fastapi import APIRouter, Path
from services.mcp_client import mcp_client
from services.query_executor import query_executor

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Database"])

@router.get("/tables")
async def get_tables():
    """Get list of available tables."""
    try:
        if not mcp_client.connected:
            await mcp_client.reconnect()
            if not mcp_client.connected:
                return {"tables": [], "error": "MCP server not connected"}
        
        result = await query_executor.get_tables()
        return result
        
    except Exception as e:
        logger.error(f"Failed to get tables: {e}")
        return {"tables": [], "error": str(e)}

@router.get("/table/{table_name}")
async def get_table_info(
    table_name: str = Path(..., description="Name of the table to get info for")
):
    """Get column information for a specific table."""
    try:
        if not mcp_client.connected:
            await mcp_client.reconnect()
            if not mcp_client.connected:
                return {"columns": [], "error": "MCP server not connected"}
        
        result = await query_executor.get_table_info(table_name)
        return result
        
    except Exception as e:
        logger.error(f"Failed to get table info: {e}")
        return {"columns": [], "error": str(e)}