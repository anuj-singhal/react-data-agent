"""
Oracle MCP Server - Generic execution tool
This server provides a flexible execute tool that can run any SELECT query.
"""

from typing import Any, Dict, List
import oracledb
from mcp.server.fastmcp import FastMCP
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Oracle connection configuration
DB_CONFIG = {
    "user": "C##agenticai",
    "password": "agenticai",
    "dsn": "localhost:1521/orcl"
}

mcp = FastMCP("Oracle MCP Server - Enhanced")

@mcp.tool(description="Execute a SELECT SQL query on the Oracle database and return structured results")
async def execute(query: str) -> Dict[str, Any]:
    """
    Execute a SQL query and return structured results.
    
    Args:
        query: SQL SELECT query to execute
        
    Returns:
        Dictionary with columns and rows, or error message
    """
    try:
        # Validate query is read-only
        query_upper = query.strip().upper()
        # if not query_upper.startswith('SELECT'):
        #     return {"error": "Only SELECT queries are allowed in read-only mode"}
        
        # Check for potentially harmful operations
        forbidden_keywords = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER', 'TRUNCATE', 'MERGE']
        if any(keyword in query_upper for keyword in forbidden_keywords):
            return {"error": "Query contains forbidden operations for read-only access"}
        
        # Connect and execute
        logger.info(f"Executing query: {query[:100]}...")
        conn = oracledb.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        # Set transaction to read-only
        cur.execute("SET TRANSACTION READ ONLY")
        
        # Execute the query
        cur.execute(query)
        
        if cur.description:
            # Get column names
            cols: List[str] = [d[0] for d in cur.description]
            
            # Fetch all rows
            rows: List[List[Any]] = [list(r) for r in cur.fetchall()]
            
            logger.info(f"Query returned {len(rows)} rows")
            
            cur.close()
            conn.close()
            
            return {
                "columns": cols,
                "rows": rows,
                "row_count": len(rows)
            }
        else:
            cur.close()
            conn.close()
            return {"error": "Query did not return any results"}
            
    except oracledb.Error as e:
        logger.error(f"Database error: {e}")
        return {"error": f"Database error: {str(e)}"}
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return {"error": f"Unexpected error: {str(e)}"}

@mcp.tool(description="Get list of tables in the database")
async def get_tables() -> Dict[str, Any]:
    """
    Get list of all tables accessible to the user.
    
    Returns:
        Dictionary with list of table names
    """
    try:
        conn = oracledb.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        # Query to get user tables
        cur.execute("""
            SELECT table_name 
            FROM user_tables 
            ORDER BY table_name
        """)
        
        tables = [row[0] for row in cur.fetchall()]
        
        cur.close()
        conn.close()
        
        return {
            "tables": tables,
            "count": len(tables)
        }
        
    except Exception as e:
        logger.error(f"Error getting tables: {e}")
        return {"error": f"Error getting tables: {str(e)}"}

@mcp.tool(description="Get column information for a specific table")
async def get_table_info(table_name: str) -> Dict[str, Any]:
    """
    Get column information for a specific table.
    
    Args:
        table_name: Name of the table
        
    Returns:
        Dictionary with column information
    """
    try:
        conn = oracledb.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        # Query to get column information
        cur.execute("""
            SELECT column_name, data_type, nullable, data_length
            FROM user_tab_columns
            WHERE table_name = UPPER(:table_name)
            ORDER BY column_id
        """, table_name=table_name)
        
        columns = []
        for row in cur.fetchall():
            columns.append({
                "name": row[0],
                "type": row[1],
                "nullable": row[2] == 'Y',
                "length": row[3]
            })
        
        cur.close()
        conn.close()
        
        if columns:
            return {
                "table": table_name,
                "columns": columns,
                "column_count": len(columns)
            }
        else:
            return {"error": f"Table '{table_name}' not found or has no columns"}
            
    except Exception as e:
        logger.error(f"Error getting table info: {e}")
        return {"error": f"Error getting table info: {str(e)}"}

@mcp.tool(description="Test database connection")
async def test_connection() -> Dict[str, Any]:
    """
    Test the database connection.
    
    Returns:
        Dictionary with connection status
    """
    try:
        conn = oracledb.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        # Get database version
        cur.execute("SELECT banner FROM v$version WHERE ROWNUM = 1")
        version = cur.fetchone()[0]
        
        # Get current user
        cur.execute("SELECT USER FROM dual")
        user = cur.fetchone()[0]
        
        cur.close()
        conn.close()
        
        return {
            "status": "Connected",
            "user": user,
            "version": version
        }
        
    except Exception as e:
        logger.error(f"Connection test failed: {e}")
        return {
            "status": "Failed",
            "error": str(e)
        }

if __name__ == "__main__":
    logger.info("Starting Oracle MCP Server...")
    mcp.run()