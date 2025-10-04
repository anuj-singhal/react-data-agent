# models/responses.py
"""Response models for API endpoints."""
from typing import Dict, Any, Optional, List
from pydantic import BaseModel

class QueryResponse(BaseModel):
    """Response model for query execution."""
    natural_query: str
    sql_query: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    mcp_connected: bool
    available_tools: List[str]
    execution_tool: Optional[str] = None

class TablesResponse(BaseModel):
    """Response model for tables list."""
    tables: List[str]
    count: int
    error: Optional[str] = None

class TableInfoResponse(BaseModel):
    """Response model for table information."""
    table_name: str
    columns: List[Dict[str, Any]]
    column_count: int
    error: Optional[str] = None

class ToolsResponse(BaseModel):
    """Response model for available tools."""
    connected: bool
    tools: List[Dict[str, Any]]
    execution_tool: Optional[str] = None

class ErrorResponse(BaseModel):
    """Generic error response."""
    error: str
    detail: Optional[str] = None