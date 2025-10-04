# routers/health.py
"""Health check endpoints."""
from fastapi import APIRouter
from models.responses import HealthResponse, ToolsResponse
from services.mcp_client import mcp_client

router = APIRouter(prefix="/health", tags=["Health"])

@router.get("", response_model=HealthResponse)
async def health_check():
    """Check the health status of the service."""
    return HealthResponse(
        status="healthy" if mcp_client.connected else "disconnected",
        mcp_connected=mcp_client.connected,
        available_tools=mcp_client.get_tool_names(),
        execution_tool=mcp_client.get_execution_tool_name()
    )

@router.get("/tools", response_model=ToolsResponse)
async def get_tools():
    """Get list of available MCP tools (for debugging)."""
    return ToolsResponse(
        connected=mcp_client.connected,
        tools=[
            {
                "name": tool.name,
                "description": tool.description,
                "is_execution_tool": tool == mcp_client.execution_tool
            }
            for tool in mcp_client.tools
        ],
        execution_tool=mcp_client.get_execution_tool_name()
    )