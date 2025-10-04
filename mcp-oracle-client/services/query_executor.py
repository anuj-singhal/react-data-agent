# services/query_executor.py
"""Query execution service."""
import json
import logging
from typing import Dict, Any, Optional

from services.mcp_client import mcp_client

logger = logging.getLogger(__name__)

class QueryExecutor:
    """Handles SQL query execution through MCP."""
    
    async def execute_query(self, sql_query: str) -> Dict[str, Any]:
        """Execute SQL query using MCP tool."""
        if not mcp_client.session or not mcp_client.connected:
            raise Exception("MCP session not connected")
        
        try:
            # Get execution tool
            execution_tool = await self._get_execution_tool()
            if not execution_tool:
                raise Exception("No execution tool available")
            
            # Determine parameter name
            param_name = self._get_query_parameter_name(execution_tool)
            
            logger.info(f"Executing SQL via tool '{execution_tool.name}' with parameter '{param_name}': {sql_query[:100]}...")
            
            # Call the tool with the SQL query
            result = await mcp_client.call_tool(
                execution_tool.name,
                {param_name: sql_query}
            )
            
            # Parse and return the result
            return self._parse_result(result)
                
        except Exception as e:
            logger.error(f"Query execution error: {e}")
            return {"error": str(e)}
    
    async def _get_execution_tool(self):
        """Get the execution tool from MCP client."""
        if mcp_client.execution_tool:
            return mcp_client.execution_tool
        
        # Try to find execution tool
        for tool in mcp_client.tools:
            if 'execute' in tool.name.lower():
                mcp_client.execution_tool = tool
                return tool
        
        # Use first available tool as fallback
        if mcp_client.tools:
            return mcp_client.tools[0]
        
        return None
    
    def _get_query_parameter_name(self, tool) -> str:
        """Determine the correct parameter name for the SQL query."""
        param_name = 'query'  # Default
        
        # Check if tool has input schema
        if hasattr(tool, 'inputSchema'):
            schema = tool.inputSchema
            if isinstance(schema, dict) and 'properties' in schema:
                properties = schema['properties']
                # Look for query-related parameter names
                for prop in properties.keys():
                    if any(keyword in prop.lower() for keyword in ['query', 'sql', 'statement']):
                        param_name = prop
                        break
                else:
                    # Use first property if no match found
                    if properties:
                        param_name = list(properties.keys())[0]
        
        return param_name
    
    def _parse_result(self, result) -> Dict[str, Any]:
        """Parse the MCP tool result."""
        if not result or not result.content:
            return {"error": "No result returned from query"}
        
        content = result.content[0] if isinstance(result.content, list) else result.content
        
        # Try to extract text content
        if hasattr(content, 'text'):
            try:
                parsed = json.loads(content.text)
                
                # Check for error in parsed result
                if isinstance(parsed, dict) and 'error' in parsed:
                    logger.error(f"Query execution error: {parsed['error']}")
                    return parsed
                
                # Successful result
                if isinstance(parsed, dict) and ('rows' in parsed or 'columns' in parsed):
                    logger.info(f"Query returned {parsed.get('row_count', len(parsed.get('rows', [])))} rows")
                    return parsed
                
                # Alternative format - wrap in standard format
                if isinstance(parsed, list):
                    return self._format_list_result(parsed)
                
                return parsed
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                return {"error": f"Invalid response format: {str(e)}"}
        
        # Try direct dictionary access
        if isinstance(content, dict):
            return content
        
        # Fallback - return as string
        return {"result": str(content)}
    
    def _format_list_result(self, data: list) -> Dict[str, Any]:
        """Format list result into standard table format."""
        if not data:
            return {"columns": [], "rows": [], "row_count": 0}
        
        if isinstance(data[0], dict):
            # List of records - convert to rows/columns format
            columns = list(data[0].keys())
            rows = [[row.get(col) for col in columns] for row in data]
            return {
                "columns": columns,
                "rows": rows,
                "row_count": len(rows)
            }
        
        # Simple list - create single column
        return {
            "columns": ["value"],
            "rows": [[item] for item in data],
            "row_count": len(data)
        }
    
    async def get_tables(self) -> Dict[str, Any]:
        """Get list of database tables."""
        # Look for a tables tool
        tables_tool = None
        for tool in mcp_client.tools:
            if 'table' in tool.name.lower() and 'get' in tool.name.lower():
                tables_tool = tool
                break
        
        if tables_tool:
            result = await mcp_client.call_tool(tables_tool.name, {})
            return self._parse_result(result)
        else:
            # Fallback to execute tool with SQL query
            return await self.execute_query("SELECT table_name FROM user_tables ORDER BY table_name")
    
    async def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get column information for a specific table."""
        # Look for a table info tool
        info_tool = None
        for tool in mcp_client.tools:
            if 'table' in tool.name.lower() and 'info' in tool.name.lower():
                info_tool = tool
                break
        
        if info_tool:
            # Determine parameter name
            param_name = self._get_table_parameter_name(info_tool)
            result = await mcp_client.call_tool(info_tool.name, {param_name: table_name})
            return self._parse_result(result)
        else:
            # Fallback to SQL query
            sql = f"""
                SELECT column_name, data_type, nullable, data_length
                FROM user_tab_columns
                WHERE table_name = UPPER('{table_name}')
                ORDER BY column_id
            """
            return await self.execute_query(sql)
    
    def _get_table_parameter_name(self, tool) -> str:
        """Determine the parameter name for table operations."""
        if hasattr(tool, 'inputSchema'):
            schema = tool.inputSchema
            if isinstance(schema, dict) and 'properties' in schema:
                if 'table_name' in schema['properties']:
                    return 'table_name'
                elif 'table' in schema['properties']:
                    return 'table'
                elif schema['properties']:
                    return list(schema['properties'].keys())[0]
        return 'table_name'

# Singleton instance
query_executor = QueryExecutor()