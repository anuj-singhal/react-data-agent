# services/mcp_client.py
"""MCP Client Manager for Oracle database interactions."""
import logging
import json
import subprocess
from typing import Optional, List, Any, Dict
from contextlib import AsyncExitStack

from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession

from config.settings import settings

logger = logging.getLogger(__name__)

class MCPClientManager:
    """Manages MCP client connections and tool discovery."""
    
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.tools: List[Any] = []
        self.tool_map: Dict[str, Any] = {}
        self.execution_tool = None
        self.exit_stack: Optional[AsyncExitStack] = None
        self.connected = False
        
    async def connect(self) -> bool:
        """Connect to MCP Oracle server via stdio and discover tools."""
        try:
            logger.info(f"Attempting to connect to MCP server: {settings.MCP_SERVER_PATH}")
            
            # Determine Python command
            python_cmd = self._get_python_command()
            if not python_cmd:
                return False
            
            # Create exit stack for proper cleanup
            self.exit_stack = AsyncExitStack()
            
            server_params = StdioServerParameters(
                command=python_cmd,
                args=[settings.MCP_SERVER_PATH]
            )
            
            # Start stdio transport
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            stdio, write = stdio_transport
            
            # Create and initialize session
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(stdio, write)
            )
            
            # Initialize MCP session
            await self.session.initialize()
            logger.info("MCP session initialized")
            
            # Discover available tools
            await self._discover_tools()
            
            self.connected = True
            return True
            
        except FileNotFoundError:
            logger.error(f"MCP server script not found: {settings.MCP_SERVER_PATH}")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from MCP server."""
        try:
            if self.exit_stack:
                await self.exit_stack.aclose()
                self.exit_stack = None
            self.session = None
            self.connected = False
            self.tools = []
            self.tool_map = {}
            self.execution_tool = None
            logger.info("Disconnected from MCP server")
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")
    
    async def reconnect(self) -> bool:
        """Reconnect to MCP server."""
        await self.disconnect()
        return await self.connect()
    
    async def _discover_tools(self):
        """Discover and categorize available tools."""
        tools_response = await self.session.list_tools()
        self.tools = tools_response.tools if tools_response else []
        
        # Build tool map and identify execution tool
        for tool in self.tools:
            self.tool_map[tool.name] = tool
            logger.info(f"Discovered tool: {tool.name} - {tool.description}")
            
            # Identify execution tool dynamically
            tool_name_lower = tool.name.lower()
            tool_desc_lower = (tool.description or "").lower()
            
            # Look for execution-related keywords
            if any(keyword in tool_name_lower for keyword in ['execute', 'query', 'run']):
                if not self.execution_tool or 'execute' in tool_name_lower:
                    self.execution_tool = tool
                    logger.info(f"Identified execution tool: {tool.name}")
            elif any(keyword in tool_desc_lower for keyword in ['execute', 'run sql', 'select query']):
                if not self.execution_tool:
                    self.execution_tool = tool
                    logger.info(f"Identified execution tool from description: {tool.name}")
        
        tool_names = [t.name for t in self.tools]
        logger.info(f"Connected to MCP server. Available tools: {tool_names}")
        if self.execution_tool:
            logger.info(f"Primary execution tool: {self.execution_tool.name}")
    
    def _get_python_command(self) -> Optional[str]:
        """Determine the correct Python command."""
        for cmd in ["python", "python3"]:
            try:
                subprocess.run([cmd, "--version"], check=True, capture_output=True)
                return cmd
            except:
                continue
        logger.error("Python executable not found")
        return None
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a specific tool with arguments."""
        if not self.session or not self.connected:
            raise Exception("MCP session not connected")
        
        logger.info(f"Calling tool '{tool_name}' with arguments: {arguments}")
        return await self.session.call_tool(tool_name, arguments)
    
    def get_tool_names(self) -> List[str]:
        """Get list of available tool names."""
        return [tool.name for tool in self.tools]
    
    def get_execution_tool_name(self) -> Optional[str]:
        """Get the name of the execution tool."""
        return self.execution_tool.name if self.execution_tool else None

# Singleton instance
mcp_client = MCPClientManager()