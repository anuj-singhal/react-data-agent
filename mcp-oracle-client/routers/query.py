# routers/query.py
"""Query execution endpoints."""
import logging
from fastapi import APIRouter, HTTPException
from models.requests import QueryRequest
from models.responses import QueryResponse
from services.mcp_client import mcp_client
from services.sql_converter import sql_converter
from services.query_executor import query_executor

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/query", tags=["Query"])

@router.post("", response_model=QueryResponse)
async def execute_query(request: QueryRequest):
    """Execute a natural language or SQL query."""
    try:
        # Check MCP connection
        if not mcp_client.connected:
            logger.info("MCP not connected, attempting to reconnect...")
            connected = await mcp_client.reconnect()
            if not connected:
                return QueryResponse(
                    natural_query=request.query,
                    sql_query="",
                    result=None,
                    error="MCP server not connected. Please check server configuration."
                )
        
        # Convert natural language to SQL
        sql_query = await sql_converter.convert_to_sql(
            request.query, 
            mcp_client.tools
        )
        logger.info(f"Generated SQL: {sql_query}")
        
        # Execute the SQL query
        result = await query_executor.execute_query(sql_query)
        
        # Check for errors in result
        if isinstance(result, dict) and "error" in result:
            return QueryResponse(
                natural_query=request.query,
                sql_query=sql_query,
                result=None,
                error=result["error"]
            )
        
        # Successful response
        return QueryResponse(
            natural_query=request.query,
            sql_query=sql_query,
            result=result,
            error=None
        )
        
    except Exception as e:
        logger.error(f"Query execution failed: {e}")
        
        # Try to get SQL even if execution failed
        sql_query = ""
        try:
            sql_query = await sql_converter.convert_to_sql(request.query, mcp_client.tools)
        except:
            pass
        
        return QueryResponse(
            natural_query=request.query,
            sql_query=sql_query,
            result=None,
            error=str(e)
        )