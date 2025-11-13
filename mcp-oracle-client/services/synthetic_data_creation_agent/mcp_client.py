"""
Improved MCP Client for Synthetic Data Generation
Enhanced with better tool descriptions and few-shot examples for LLM
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI
import pandas as pd
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SyntheticDataClient:
    """Improved MCP Client for Synthetic Data Generation."""
    
    def __init__(self, api_key: str = None, server_script_path: str = "synthetic_duckdb_server_improved.py"):
        """Initialize the client with OpenAI API key and server path."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.openai_client = OpenAI(api_key=self.api_key)
        self.server_script = server_script_path
        self.command = "python"
        self.tools = {}
        
    async def get_available_tools(self):
        """Get list of available tools from the server."""
        exit_stack = AsyncExitStack()
        server_params = StdioServerParameters(
            command=self.command,
            args=[self.server_script]
        )
        
        try:
            # Start stdio transport and session
            stdio_transport = await exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            stdio, write = stdio_transport
            session = await exit_stack.enter_async_context(
                ClientSession(stdio, write)
            )
            
            await session.initialize()
            
            # Get available tools
            tools_response = await session.list_tools()
            for tool in tools_response.tools:
                self.tools[tool.name] = tool
            
            print(f"Available tools: {list(self.tools.keys())}")
            return self.tools
            
        finally:
            await exit_stack.aclose()
    
    async def execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Any:
        """
        Execute a specific tool via MCP server.
        
        Args:
            tool_name: Name of the tool to execute
            tool_args: Arguments for the tool
            
        Returns:
            Tool execution result or None if failed
        """
        exit_stack = AsyncExitStack()
        server_params = StdioServerParameters(
            command=self.command, 
            args=[self.server_script]
        )
        
        try:
            # Start stdio transport and session
            stdio_transport = await exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            stdio, write = stdio_transport
            session = await exit_stack.enter_async_context(
                ClientSession(stdio, write)
            )
            
            await session.initialize()
            
            logger.info(f"Executing tool: {tool_name} with args: {tool_args}")
            
            # Execute tool via MCP
            result = await session.call_tool(tool_name, tool_args)
            
            # Parse result based on content type
            if result.content and len(result.content) > 0:
                content = result.content[0]
                
                # Handle text content
                if hasattr(content, "text"):
                    text_result = content.text
                    
                    # Try to parse as JSON if it looks like JSON
                    if text_result.strip().startswith('{') or text_result.strip().startswith('['):
                        try:
                            structured = json.loads(text_result)
                            logger.info(f"Tool {tool_name} returned structured data")
                            return structured
                        except json.JSONDecodeError:
                            pass
                    
                    logger.info(f"Tool {tool_name} executed successfully")
                    return text_result
                
                # Handle other content types
                else:
                    logger.info(f"Tool {tool_name} returned non-text content")
                    return content
            
            return None
            
        except Exception as e:
            logger.error(f"MCP tool execution failed: {e}")
            raise
            
        finally:
            await exit_stack.aclose()
    
    def create_plan_with_llm(self, user_request: str, metadata_path: str) -> List[Dict[str, Any]]:
        """Use GPT-4o-mini to create an execution plan for synthetic data generation."""
        
        # Read metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Create detailed tool descriptions
        tool_descriptions = """
        Available MCP Server Tools for Synthetic Data Generation:
        
        1. list_tables()
           - Lists all tables currently in the DuckDB database
           - No arguments required
           - Returns: String with comma-separated table names
        
        2. describe_table(table_name: str)
           - Describes a specific table's structure and row count
           - Arguments: table_name (string) - name of table to describe
           - Returns: String with column details and total row count
        
        3. load_metadata(metadata_path: str, create_tables: bool = True)
           - Loads metadata JSON and optionally creates tables based on schema
           - Arguments: 
             - metadata_path: path to metadata JSON file
             - create_tables: whether to create tables if missing (default: True)
           - Returns: Status message about loading and table creation
        
        4. create_sample_tables(metadata_path: str, rows_per_table: int = 100, use_existing_schema: bool = True)
           - Creates and populates tables with realistic sample data
           - Arguments:
             - metadata_path: path to metadata JSON
             - rows_per_table: number of sample rows per table (default: 100)
             - use_existing_schema: whether to use existing tables (default: True)
           - Returns: Status message about populated tables
        
        5. generate_single_table_synthetic(source_table_name: str, target_table_name: str = None, 
                                         num_rows: int = None, with_relationships: bool = False, 
                                         metadata_path: str = None)
           - Generates synthetic data for a single table, optionally with related tables
           - Arguments:
             - source_table_name: source table to generate from
             - target_table_name: name for synthetic table (optional)
             - num_rows: number of rows to generate (optional)
             - with_relationships: whether to generate related child tables (default: False)
             - metadata_path: path to metadata for relationships (optional)
           - Returns: Status message about generation
           - Use for: Generating specific number of records (e.g., "10 clients")
        
        6. generate_multi_table_synthetic(metadata_path: str, ensure_tables_exist: bool = True)
           - Generates synthetic data for ALL tables defined in metadata
           - Arguments:
             - metadata_path: path to metadata JSON
             - ensure_tables_exist: create missing tables first (default: True)
           - Returns: Status message about all generated tables
           - Use for: Generating synthetic data for entire database schema
        
        7. evaluate_synthetic_quality(real_table: str, synthetic_table: str, metadata_path: str = None)
           - Evaluates quality of synthetic data against real data
           - Arguments:
             - real_table: name of real data table
             - synthetic_table: name of synthetic data table
             - metadata_path: optional metadata path for better evaluation
           - Returns: Dictionary with quality scores
        
        8. export_synthetic_data(table_name: str, output_path: str, format: str = "csv")
           - Exports synthetic data to file
           - Arguments:
             - table_name: table to export
             - output_path: file path for output
             - format: export format (csv, parquet, json)
           - Returns: Export status message
        """
        
        few_shot_examples = """
        Few-Shot Examples for Plan Generation:
        
        EXAMPLE 1: "Create sample tables for wealth management portfolio data, generate synthetic data maintaining relationships for all tables, evaluate the quality, and provide a summary. Generate 500 rows per table."
        
        Breakdown:
        - "Create sample tables" → First check if tables exist, create if missing
        - "generate synthetic data...for all tables" → Use generate_multi_table_synthetic
        - "maintaining relationships" → The multi-table tool handles relationships
        - "evaluate the quality" → Use evaluate_synthetic_quality for main tables
        - "500 rows per table" → Specify rows_per_table=500 in create_sample_tables
        
        Plan:
        1. load_metadata to ensure schema is loaded
        2. list_tables to check what exists
        3. create_sample_tables with 500 rows per table
        4. generate_multi_table_synthetic for all tables
        5. evaluate_synthetic_quality for key tables (clients, portfolios)
        6. Generate summary from results
        
        EXAMPLE 2: "Generate synthetic data for 10 clients with their related data, evaluate the quality, and provide a summary."
        
        Breakdown:
        - "10 clients" → Specific number for single table
        - "with their related data" → Use with_relationships=True
        - This is single table generation that cascades to child tables
        
        Plan:
        1. Check if clients table exists using describe_table
        2. generate_single_table_synthetic with source_table_name="clients", num_rows=10, with_relationships=True
        3. evaluate_synthetic_quality for clients and clients_synthetic
        4. list_tables to show all synthetic tables created
        5. Generate summary
        
        EXAMPLE 3: "Generate synthetic data for all tables in the metadata"
        
        Breakdown:
        - "all tables" → Use multi-table generation
        - No specific row count → Use defaults
        
        Plan:
        1. list_tables to check what exists
        2. create_sample_tables with 500 rows per table
        3. generate_multi_table_synthetic with metadata_path, ensure_tables_exist=True
        4. list_tables to verify all synthetic tables
        5. Generate summary
        
        KEY DECISION RULES:
        - If user specifies a NUMBER with a specific TABLE → check the table exists if not then create sample table and use generate_single_table_synthetic
        - If user wants ALL TABLES or mentions "maintaining relationships" for multiple tables → check the table exists if not then create sample table and use generate_multi_table_synthetic
        - If user mentions "create tables" or "sample data" → use create_sample_tables first
        - Always evaluate quality if user mentions "quality" or "evaluate"
        - The multi-table tool automatically creates missing tables when ensure_tables_exist=True
        """
        
        # Create prompt for the LLM
        system_prompt = f"""You are a synthetic data generation assistant using an MCP server. 
        Create a step-by-step execution plan based on the user's request.
        
        {tool_descriptions}
        
        {few_shot_examples}
        
        Return a JSON object with a "steps" array, where each step has:
        {{
            "step": step_number,
            "description": "clear description of what this step does",
            "tool": "exact_tool_name",
            "arguments": {{argument_dict}}
        }}
        
        Be precise with tool names and arguments. Match the exact parameter names."""
        
        user_prompt = f"""
        User request: {user_request}
        
        Metadata structure:
        Tables: {list(metadata['tables'].keys())}
        Relationships: {len(metadata.get('relationships', []))} relationships defined
        Table relationships: {', '.join([f"{r['child_table_name']} -> {r['parent_table_name']}" for r in metadata.get('relationships', [])])}
        
        Metadata path to use: {metadata_path}
        
        Create a practical execution plan. Consider:
        1. Whether to generate single table or multiple tables
        2. Whether relationships need to be maintained
        3. Whether tables need to be created first
        4. Whether to evaluate quality
        
        Return as JSON with a "steps" array.
        """
        
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        plan_json = response.choices[0].message.content
        plan = json.loads(plan_json)
        
        # Extract steps array
        if isinstance(plan, dict) and 'steps' in plan:
            return plan['steps']
        elif isinstance(plan, list):
            return plan
        else:
            # Wrap in steps array if needed
            return [plan] if isinstance(plan, dict) else []
    
    async def execute_plan(self, plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute the plan step by step."""
        results = []
        
        for step in plan:
            print(f"\n{'='*60}")
            print(f"Step {step.get('step', '?')}: {step.get('description', 'Executing...')}")
            print(f"Tool: {step.get('tool')}")
            print(f"Arguments: {step.get('arguments', {})}")
            
            try:
                # Execute the tool
                result = await self.execute_tool(
                    step.get('tool'),
                    step.get('arguments', {})
                )
                
                # Format result for display
                if isinstance(result, dict):
                    print(f"Result: {json.dumps(result, indent=2)[:500]}...")
                elif isinstance(result, str) and len(result) > 500:
                    print(f"Result: {result[:500]}...")
                else:
                    print(f"Result: {result}")
                
                results.append({
                    "step": step.get('step'),
                    "description": step.get('description'),
                    "tool": step.get('tool'),
                    "status": "success",
                    "result": result if not isinstance(result, dict) else json.dumps(result)
                })
                
                # Add small delay between steps for visibility
                await asyncio.sleep(0.5)
                
            except Exception as e:
                print(f"Error: {str(e)}")
                results.append({
                    "step": step.get('step'),
                    "description": step.get('description'),
                    "tool": step.get('tool'),
                    "status": "error",
                    "error": str(e)
                })
        
        return {"execution_results": results}
    
    def generate_summary(self, execution_results: Dict[str, Any]) -> str:
        """Use GPT-4o-mini to generate a summary of the execution results."""
        
        system_prompt = """You are a synthetic data generation assistant.
        Summarize the execution results in a clear, concise manner.
        
        Focus on:
        1. What was successfully generated (tables, row counts)
        2. Quality metrics if evaluated (scores, insights)
        3. Any errors or issues
        4. Practical next steps
        
        Be specific about numbers and results."""
        
        user_prompt = f"""
        Execution results:
        {json.dumps(execution_results, indent=2)}
        
        Please provide a comprehensive summary of the synthetic data generation process.
        Include:
        1. What was accomplished (be specific about tables and row counts)
        2. Quality metrics (if evaluated, include specific scores)
        3. Any issues encountered
        4. Recommendations for next steps
        
        Keep it concise but informative.
        """
        
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.5
        )
        
        return response.choices[0].message.content


async def main():
    """Main function to demonstrate the improved client usage."""
    
    # Initialize client
    print("="*70)
    print("SYNTHETIC DATA GENERATION MCP CLIENT")
    print("="*70)
    
    # Server and metadata paths
    server_path = "synthetic_duckdb_server.py"
    metadata_path = "wealth_management_metadata.json"
    
    try:
        client = SyntheticDataClient(server_script_path=server_path)
        
        # Get available tools
        print(f"\nConnecting to MCP Server: {server_path}")
        await client.get_available_tools()
        
        # Show example requests
        print("\n" + "="*70)
        print("EXAMPLE REQUESTS:")
        print("-"*70)
        print("1. Generate 10 clients with related data:")
        print("   'Generate synthetic data for 10 clients with their portfolios and holdings'")
        print("\n2. Generate all tables:")
        print("   'Create sample tables and generate synthetic data for all wealth management tables'")
        print("\n3. Full pipeline:")
        print("   'Create 500 rows of sample data per table, generate synthetic versions, and evaluate quality'")
        print("="*70)
        
        # Get user request
        user_request = input("\nWhat would you like to do with synthetic data?\n> ")
        
        if not user_request.strip():
            # Default request
            user_request = "Generate synthetic data for 10 clients with their related portfolios and holdings, then evaluate the quality"
            print(f"\nUsing example request: {user_request}")
        
        # Create execution plan
        print("\n" + "="*70)
        print("CREATING EXECUTION PLAN")
        print("-"*70)
        plan = client.create_plan_with_llm(user_request, metadata_path)
        
        print("\nGenerated Plan:")
        for i, step in enumerate(plan, 1):
            print(f"  {i}. {step.get('description', 'No description')}")
            print(f"     Tool: {step.get('tool')}")
            args = step.get('arguments', {})
            if args:
                print(f"     Args: {', '.join([f'{k}={v}' for k, v in args.items()])}")
        
        # Confirm execution
        confirm = input("\nProceed with execution? (y/n): ")
        if confirm.lower() != 'y':
            print("Execution cancelled.")
            return
        
        # Execute plan
        print("\n" + "="*70)
        print("EXECUTING PLAN")
        print("-"*70)
        results = await client.execute_plan(plan)
        
        # Generate summary
        print("\n" + "="*70)
        print("GENERATING SUMMARY")
        print("-"*70)
        summary = client.generate_summary(results)
        
        print("\n" + "="*70)
        print("FINAL SUMMARY")
        print("="*70)
        print(summary)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"synthetic_data_results_{timestamp}.json"
        
        # Prepare serializable results
        serializable_results = {
            "request": user_request,
            "plan": plan,
            "results": results,
            "summary": summary,
            "timestamp": timestamp
        }
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n✅ Results saved to: {results_file}")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the async main function
    asyncio.run(main())