"""
LLM Query Generator - Handles SQL generation using LLMs
"""

import os
from typing import Dict, Any, Optional
try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    AsyncOpenAI = None


class LLMQueryGenerator:
    """Handles SQL query generation using various LLMs"""
    
    def __init__(self, provider: str = "openai", api_key: Optional[str] = None, model: Optional[str] = None):
        self.provider = provider.lower()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if self.provider == "openai":
            self.model = model or "gpt-4o-mini"
            if self.api_key:
                self.client = AsyncOpenAI(api_key=self.api_key)
            else:
                print("Warning: No OpenAI API key provided")
                self.client = None
        else:
            self.model = None
            self.client = None
    
    async def generate_sql(self, context: Dict[str, Any], intent, previous_message = None, last_sql = None) -> str:
        """Generate SQL based on context"""
        if self.provider == "openai" and self.client:
            if(intent == "modify"):
                return await self._generate_openai_modify_sql(context, previous_message, last_sql)
            else:
                return await self._generate_openai_sql(context)
        else:
            return await self._generate_heuristic_sql(context)
    
    async def _generate_openai_modify_sql(self, context: Dict[str, Any], previous_message = None, last_sql = None) -> str:
        """Generate SQL using OpenAI"""
        prompt = self._build_modify_prompt(context, previous_message, last_sql)
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert SQL developer. Generate clean, efficient modified SQL query"
                        "based on the previous user request and modified request and provided database schema"
                        "Return only the SQL query without explanations or markdown."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            sql = response.choices[0].message.content.strip()
            sql = sql.replace("```sql", "").replace("```", "").strip()
            return sql
            
        except Exception as e:
            print(f"Error with OpenAI: {e}")
            return self._generate_heuristic_sql(context)

    async def _generate_openai_sql(self, context: Dict[str, Any]) -> str:
        """Generate SQL using OpenAI"""
        prompt = self._build_prompt(context)
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert SQL developer. Generate clean, efficient SQL queries. Return only the SQL query without explanations or markdown."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            sql = response.choices[0].message.content.strip()
            sql = sql.replace("```sql", "").replace("```", "").strip()
            return sql
            
        except Exception as e:
            print(f"Error with OpenAI: {e}")
            return self._generate_heuristic_sql(context)
    

    async def _generate_heuristic_sql(self, context: Dict[str, Any]) -> str:
        """Generate SQL using heuristics when LLM is not available"""
        tables = list(context['tables'].keys())
        user_query_lower = context['user_query'].lower()
        
        if not tables:
            return "-- No relevant tables found"
        
        # Determine query type
        needs_aggregation = any(word in user_query_lower for word in 
                               ['average', 'sum', 'count', 'total', 'max', 'min'])
        needs_join = len(tables) > 1 and context.get('relationships')
        
        # Build SQL
        if needs_join and len(tables) >= 2:
            # Generate JOIN query
            sql = "SELECT "
            
            if 'financial' in user_query_lower:
                sql += "b.BANK_NAME, fp.*"
                from_clause = "FINANCIAL_PERFORMANCE fp JOIN BANKS b ON fp.BANK_ID = b.BANK_ID"
            elif 'market' in user_query_lower:
                sql += "b.BANK_NAME, md.*"
                from_clause = "MARKET_DATA md JOIN BANKS b ON md.BANK_ID = b.BANK_ID"
            else:
                sql += "*"
                from_clause = f"{tables[0]} t1"
                if len(tables) > 1 and context['relationships']:
                    rel = context['relationships'][0]
                    from_clause += f" JOIN {tables[1]} t2 ON {rel['from']} = {rel['to']}"
            
            sql += f"\nFROM {from_clause}"
            
        else:
            # Simple query
            table = tables[0]
            sql = f"SELECT * FROM {table}"
        
        # Add WHERE clause
        sql = self._add_where_conditions(sql, user_query_lower)
        
        # Add GROUP BY if needed
        if needs_aggregation:
            sql += "\nGROUP BY 1"
        
        sql += "\nORDER BY 1"
        
        return sql
    
    async def _add_where_conditions(self, sql: str, query_lower: str) -> str:
        """Add WHERE conditions based on query"""
        conditions = []
        
        # Year conditions
        for year in ['2020', '2021', '2022', '2023', '2024']:
            if year in query_lower:
                conditions.append(f"YEAR = {year}")
                break
        
        # Quarter conditions
        if 'q1' in query_lower or 'quarter 1' in query_lower:
            conditions.append("QUARTER = 1")
        elif 'q2' in query_lower or 'quarter 2' in query_lower:
            conditions.append("QUARTER = 2")
        elif 'q3' in query_lower or 'quarter 3' in query_lower:
            conditions.append("QUARTER = 3")
        elif 'q4' in query_lower or 'quarter 4' in query_lower:
            conditions.append("QUARTER = 4")
        
        # Bank names
        bank_names = {'mashreq': 'Mashreq', 'adcb': 'ADCB', 'dib': 'DIB', 'enbd': 'ENBD', 'fab': 'FAB'}
        for bank_key, bank_value in bank_names.items():
            if bank_key in query_lower:
                conditions.append(f"BANK_NAME = '{bank_value}'")
                break
        
        # Threshold conditions
        import re
        numbers = re.findall(r'\d+\.?\d*', query_lower)
        
        if numbers and ('above' in query_lower or 'greater' in query_lower or '>' in query_lower):
            value = numbers[0]
            if 'npl' in query_lower:
                conditions.append(f"NPL_RATIO > {value}")
            elif 'cet1' in query_lower:
                conditions.append(f"CET1 > {value}")
            elif 'share' in query_lower or 'price' in query_lower:
                conditions.append(f"SHARE_PRICE > {value}")
        
        if numbers and ('below' in query_lower or 'less' in query_lower or '<' in query_lower):
            value = numbers[0]
            if 'npl' in query_lower:
                conditions.append(f"NPL_RATIO < {value}")
            elif 'pe' in query_lower:
                conditions.append(f"PE_RATIO < {value}")
        
        if conditions:
            sql += "\nWHERE " + " AND ".join(conditions)
        
        return sql
    
    def _build_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for LLM"""
        prompt = f"""Generate SQL for: {context['user_query']}

DATABASE SCHEMA:
"""
        
        # Add table information
        for table_name, table_info in context['tables'].items():
            prompt += f"\n\nTABLE: {table_name}"
            prompt += f"\nDescription: {table_info['description']}"
            prompt += "\nColumns:"
            
            for col in table_info['columns']:
                pk = " [PK]" if col.get('is_pk') else ""
                prompt += f"\n  - {col['name']} ({col['type']}){pk}: {col['description']}"
                
                if 'samples' in col:
                    prompt += f"\n    Examples: {', '.join(str(s) for s in col['samples'])}"
        
        # Add relationships
        if context['relationships']:
            prompt += "\n\nRELATIONSHIPS:"
            for rel in context['relationships']:
                prompt += f"\n- {rel['from']} -> {rel['to']} ({rel['type']})"
        
        # Add Business Rules
        if context['business_rules']:
            prompt += "\n\BUSINESS RULES:"
            for rul in context['business_rules']:
                if(rul["active"]):
                    prompt += f"\n- Rule Name: {rul['name']}"
                    prompt += f"\n- Rule to Apply : {rul['rule']}"

        # Add SQL Rules
        if context['sql_rules']:
            prompt += "\n\SQL GENERATION RULES:"
            for rul in context['sql_rules']:
                if(rul["active"]):
                    prompt += f"\n- Rule Name: {rul['name']}"
                    prompt += f"\n- Rule to Apply : {rul['rule']}"

        prompt += "\n\nGenerate only the SQL query:"
        
        return prompt
    
    
    def _build_modify_prompt(self, context: Dict[str, Any], previous_messages, last_sql) -> str:
        """Build prompt for LLM"""
        prompt = f"""You are an Oracle SQL expert. Modify the following SQL query based on the user's request.

        Original SQL:
        {last_sql}

        User's modification request:
        {context["user_query"]}

        User's last requests which generated above Original SQL:
        {previous_messages}

        DATABASE SCHEMA:
        """

        # Add table information
        for table_name, table_info in context['tables'].items():
            prompt += f"\n\nTABLE: {table_name}"
            prompt += f"\nDescription: {table_info['description']}"
            prompt += "\nColumns:"
            
            for col in table_info['columns']:
                pk = " [PK]" if col.get('is_pk') else ""
                prompt += f"\n  - {col['name']} ({col['type']}){pk}: {col['description']}"
                
                if 'samples' in col:
                    prompt += f"\n    Examples: {', '.join(str(s) for s in col['samples'])}"
        
        # Add relationships
        if context['relationships']:
            prompt += "\n\nRELATIONSHIPS:"
            for rel in context['relationships']:
                prompt += f"\n- {rel['from']} -> {rel['to']} ({rel['type']})"

        prompt += f""" 
        Important:
        - Return ONLY the modified SQL query
        - Do NOT include any markdown formatting like ```sql or ```
        - Do NOT include any explanations or comments
        - Do NOT include semicolons at the end
        - Return just the plain SQL text"""
        
        return prompt
    

    async def classify_query_type(self, sql: str) -> str:
        """Classify the type of SQL query"""
        sql_upper = sql.upper()
        
        if 'JOIN' in sql_upper:
            if 'GROUP BY' in sql_upper:
                return 'join_aggregation'
            return 'join_filter'
        elif 'GROUP BY' in sql_upper:
            return 'aggregation'
        elif 'WHERE' in sql_upper:
            return 'filter'
        else:
            return 'simple_select'
        
# Singleton instance
sql_converter = LLMQueryGenerator()