# services/sql_converter.py
"""Natural Language to SQL conversion service."""
import logging
from typing import Optional, List

from config.settings import settings
from data.schemas import DATA_DICTIONARY, SQL_PATTERNS

logger = logging.getLogger(__name__)

# Optional OpenAI import
try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    AsyncOpenAI = None

class SQLConverter:
    """Converts natural language queries to SQL."""
    
    def __init__(self):
        self.llm = None
        if OPENAI_AVAILABLE and settings.OPENAI_API_KEY:
            self.llm = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            logger.info("OpenAI client initialized for SQL conversion")
        else:
            logger.warning("OpenAI not configured, using fallback SQL conversion")
    
    def clean_sql_query(self, sql: str) -> str:
        """Clean SQL query from any markdown or formatting."""
        # Remove markdown code blocks
        sql = sql.replace("```sql", "").replace("```", "")
        # Remove any leading/trailing whitespace and semicolons
        sql = sql.strip().rstrip(";").strip()
        return sql
    
    async def convert_to_sql(self, natural_language_query: str, tools: List = None) -> str:
        """Convert natural language to SQL query."""
        # Try LLM conversion first if available
        if self.llm:
            try:
                sql = await self._convert_with_llm(natural_language_query, tools)
                if sql and sql != "INVALID_QUERY":
                    return self.clean_sql_query(sql)
            except Exception as e:
                logger.error(f"LLM conversion error: {e}")
        
        # Fallback to pattern matching
        sql = self._convert_with_patterns(natural_language_query)
        return self.clean_sql_query(sql)
    
    async def _convert_with_llm(self, query: str, tools: List = None) -> str:
        """Convert using OpenAI LLM."""
        # Build table descriptions
        table_descriptions = self._build_table_descriptions()
        tool_descriptions = self._build_tool_descriptions(tools) if tools else ""
        
        prompt = f"""You are an Oracle SQL expert. Database schema:
{table_descriptions}

{tool_descriptions}

User task: {query}

Important rules:
- Generate a valid Oracle SELECT query
- Use only existing tables and columns from the schema above
- Return ONLY the SQL query text
- Do NOT include any markdown formatting like ```sql or ```
- Do NOT include any explanations, comments, or additional text
- Do NOT include semicolons at the end
- Return just the plain SQL statement
- If the query cannot be generated, return exactly: INVALID_QUERY"""
        
        response = await self.llm.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are an Oracle SQL expert. Return only plain SQL queries without any formatting, markdown, or explanations."},
                {"role": "user", "content": prompt}
            ],
            temperature=settings.OPENAI_TEMPERATURE,
            max_tokens=settings.OPENAI_MAX_TOKENS
        )
        
        sql = response.choices[0].message.content.strip()
        
        # Double-check and clean the SQL
        sql = self.clean_sql_query(sql)
        
        logger.info(f"LLM generated SQL: {sql}")
        return sql
    
    def _convert_with_patterns(self, query: str) -> str:
        """Convert using pattern matching (fallback)."""
        query_lower = query.lower()
        
        logger.info(f"Using fallback SQL conversion for: {query_lower}")
        
        # Check common patterns
        for pattern, sql in SQL_PATTERNS.items():
            if pattern in query_lower:
                return sql
        
        # Check if already SQL
        if query_lower.strip().startswith("select"):
            return query.replace(";", "").strip()
        
        # Handle table-specific queries
        return self._handle_table_queries(query_lower)
    
    def _handle_table_queries(self, query_lower: str) -> str:
        """Handle queries related to specific tables."""
        # Default queries for the UAE banks table
        if "banks" in query_lower or "financial" in query_lower:
            if "all" in query_lower:
                return "SELECT * FROM uae_banks_financial_data WHERE ROWNUM <= 20"
            elif "profit" in query_lower:
                if "top" in query_lower:
                    return "SELECT * FROM (SELECT bank, ytd_profit FROM uae_banks_financial_data ORDER BY ytd_profit DESC) WHERE ROWNUM <= 5"
                else:
                    return "SELECT bank, ytd_profit FROM uae_banks_financial_data ORDER BY ytd_profit DESC"
            elif "deposits" in query_lower:
                return "SELECT bank, deposits FROM uae_banks_financial_data WHERE deposits > 10000"
            elif "average" in query_lower and "nim" in query_lower:
                return "SELECT AVG(nim) as average_nim FROM uae_banks_financial_data"
        
        # Default fallback
        logger.warning(f"Could not parse query: {query_lower}")
        return "SELECT table_name FROM user_tables"
    
    def _build_table_descriptions(self) -> str:
        """Build table descriptions for LLM prompt."""
        descriptions = []
        for table, info in DATA_DICTIONARY.items():
            cols_desc = ', '.join(f'{col} ({desc})' for col, desc in info['columns'].items())
            descriptions.append(f"{table}: {info['description']}. Columns: {cols_desc}")
        return "\n".join(descriptions)
    
    def _build_tool_descriptions(self, tools: List) -> str:
        """Build tool descriptions for LLM prompt."""
        if not tools:
            return ""
        
        descriptions = ["Available MCP tools:"]
        for tool in tools:
            descriptions.append(f"- {tool.name}: {tool.description}")
        return "\n".join(descriptions)

# Singleton instance
sql_converter = SQLConverter()