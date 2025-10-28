"""
LLM Interface for Agentic SQL Generation
Unified interface for different LLM providers
"""

import json
import re
import logging
from typing import Dict, Any, List, Optional
from openai import AsyncOpenAI
import os

logger = logging.getLogger(__name__)


class AgenticLLMInterface:
    """Unified interface for LLM providers used in agentic SQL generation"""
    
    def __init__(self, provider: str = "openai", api_key: Optional[str] = None, model: Optional[str] = None):
        self.provider = provider.lower()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if self.provider == "openai":
            self.model = model or "gpt-4o-mini"
            if self.api_key:
                self.client = AsyncOpenAI(api_key=self.api_key)
            else:
                logger.warning("No OpenAI API key provided")
                self.client = None
        else:
            self.model = None
            self.client = None
    
    async def check_complexity(self, query: str) -> bool:
        """
        Check if a query is complex and needs decomposition
        """
        if self.provider == "openai" and self.client:
            prompt = f"""
            Analyze if this SQL query request is complex and needs decomposition into multiple steps.
            
            Query: {query}
            
            A query is COMPLEX if it needs:
            - Multiple aggregations at different levels
            - Data from 3+ tables with complex joins
            - Multiple CTEs or subqueries
            - Comparison of multiple time periods
            - Ranking combined with other operations
            - Complex calculations with multiple steps
            
            Return ONLY: true or false
            """
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a SQL complexity analyzer. Return only 'true' or 'false'."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=10
                )
                
                result = response.choices[0].message.content.strip().lower()
                return result == "true"
                
            except Exception as e:
                logger.error(f"Complexity check failed: {e}")
                return False
        else:
            # Simple heuristic
            complex_indicators = ['rank', 'compare', 'trend', 'correlation', 'multiple', 'analyze', 'comprehensive']
            return any(indicator in query.lower() for indicator in complex_indicators)
    
    async def analyze_query_similarity(self, prompt: str) -> Dict[str, Any]:
        """
        Analyze if a historical SQL can be reused for current query
        """
        if self.provider == "openai" and self.client:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are a SQL similarity analyzer. Return only valid JSON."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=200
                )
                
                content = response.choices[0].message.content.strip()
                
                # Try to parse JSON
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    # Extract JSON from response if wrapped in markdown
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group())
                    else:
                        return {
                            "can_reuse": False,
                            "reason": "Failed to parse response",
                            "confidence": 0
                        }
                        
            except Exception as e:
                logger.error(f"Query similarity analysis failed: {e}")
                return {
                    "can_reuse": False,
                    "reason": str(e),
                    "confidence": 0
                }
        else:
            # Fallback: simple keyword matching
            return {
                "can_reuse": False,
                "reason": "LLM not available for similarity analysis",
                "confidence": 0
            }
    
    async def generate_sql(self, context: Dict[str, Any], intent, previous_message = None, last_sql = None) -> str:
        """Generate SQL based on context"""
        if self.provider == "openai" and self.client:
            if(intent == "modify_query"):
                return await self._generate_openai_modify_sql(context, previous_message, last_sql)
            else:
                return await self._generate_openai_sql(context)
            
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

    async def validate_syntax(self, sql_query:str) -> str:
        """Generate SQL using OpenAI"""
        prompt = self._create_syntax_validation_prompt(sql_query)
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert SQL validator specialized in different SQL Validation. Always respond with valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            result = response.choices[0].message.content.strip()
            result = result.replace("```json", "").replace("```", "").strip()
            result = json.loads(result)
            return result
            
        except Exception as e:
            print(f"Error with OpenAI: {e}")
            return None

    
    def _create_syntax_validation_prompt(self, sql_query: str) -> str:
        """Create prompt for syntax validation with banking examples"""
        prompt = f"""
            You are an expert SQL syntax validator for banking systems. Analyze the given SQL query for syntax correctness.

            ## SYNTAX_VALIDATION

            ### Few-Shot Examples Using Banking Tables:

            Example 1:
            Input SQL: SELECT * FORM BANKS WHERE BANK_NAME = 'ADCB'
            Valid: False
            Confidence: 0.95
            Issues: ["SQL keyword 'FORM' is incorrect, should be 'FROM'"]
            Suggestions: ["Replace 'FORM' with 'FROM'"]
            Corrected SQL: SELECT * FROM BANKS WHERE BANK_NAME = 'ADCB'

            Example 2:
            Input SQL: SELECT b.BANK_NAME, fp.YTD_PROFIT FROM BANKS b JOIN FINANCIAL_PERFORMANCE fp ON b.BANK_ID = fp.BANK_ID WHERE fp.YEAR = 2024 AND fp.QUARTER = 4
            Valid: True
            Confidence: 0.98
            Issues: []
            Suggestions: []
            Analysis: Proper JOIN syntax with correct table aliases and WHERE conditions

            Example 3:
            Input SQL: SELECT BANK_NAME, SUM(YTD_PROFIT) AS total_profit FROM BANKS b JOIN FINANCIAL_PERFORMANCE fp ON b.BANK_ID = fp.BANK_ID GROUP BY b.BANK_NAME HAVING SUM(YTD_PROFIT) > 1000
            Valid: True
            Confidence: 0.97
            Issues: []
            Suggestions: ["Consider using table alias consistently for BANK_NAME in SELECT"]
            Analysis: Valid GROUP BY with HAVING clause for aggregation

            Example 4:
            Input SQL: UPDATE FINANCIAL_PERFORMANCE SET NPL_RATIO = 3.5 WHERE BANK_ID = 1 AND YEAR = 2024 AND QUARTER = 4
            Valid: True
            Confidence: 0.96
            Issues: []
            Suggestions: []
            Analysis: Valid UPDATE statement with compound WHERE condition

            Example 5:
            Input SQL: SELECT BANK_NAME, AVG(NIM) OVER (PARTITION BY YEAR ORDER BY QUARTER) as avg_nim FROM BANKS b JOIN FINANCIAL_PERFORMANCE fp USING(BANK_ID)
            Valid: True
            Confidence: 0.94
            Issues: []
            Suggestions: ["Window function syntax is correct"]
            Analysis: Valid window function with PARTITION BY and ORDER BY

            ### Now validate this SQL query:
            Input SQL: {sql_query}

            Provide response in JSON format:
            {{
                "valid": boolean,
                "confidence": float (0-1),
                "issues": [list of syntax issues],
                "suggestions": [list of improvement suggestions],
                "corrected_sql": "corrected query if issues found",
                "analysis": "detailed analysis"
            }}
            """
        return prompt
    

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
    
    async def generate_sql_bkp(self, query: str, schema_context: str) -> str:
        """
        Generate SQL for a simple query
        """
        if self.provider == "openai" and self.client:
            prompt = f"""
            Generate SQL for this request:
            {query}
            
            Available Schema:
            {schema_context}
            
            Return only the SQL query, no explanations.
            """
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert SQL developer. Generate clean, efficient SQL queries."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=500
                )
                
                sql = response.choices[0].message.content.strip()
                # Remove markdown code blocks if present
                sql = re.sub(r'```sql?\n?', '', sql)
                sql = sql.replace('```', '')
                
                return sql
                
            except Exception as e:
                logger.error(f"SQL generation failed: {e}")
                return f"-- Error: {e}\nSELECT * FROM banks LIMIT 10"
        else:
            # Fallback to simple SQL
            return "SELECT * FROM banks LIMIT 10"
    
    async def decompose_query(self, query: str, schema_context: str) -> Dict[str, Any]:
        """
        Decompose complex query into tasks
        """
        if self.provider == "openai" and self.client:
            prompt = f"""
            Decompose this complex SQL request into smaller tasks.
            
            Request: {query}
            
            Schema:
            {schema_context}
            
            Return a JSON object with:
            {{
                "tasks": [
                    {{
                        "task_id": "T1",
                        "description": "task description",
                        "sql_query": "SELECT ...",
                        "dependencies": [],
                        "execution_type": "sequential"
                    }}
                ],
                "execution_order": [["T1"], ["T2", "T3"]],
                "synthesis_instructions": "how to combine results"
            }}
            """
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a SQL query decomposer. Return only valid JSON."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=1000
                )
                
                content = response.choices[0].message.content.strip()
                
                # Parse JSON
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    # Extract JSON from markdown
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group())
                    else:
                        raise ValueError("Failed to parse decomposition response")
                        
            except Exception as e:
                logger.error(f"Query decomposition failed: {e}")
                # Return fallback single task
                return {
                    "tasks": [
                        {
                            "task_id": "T1",
                            "description": "Execute full query",
                            "sql_query": f"-- {query}\nSELECT * FROM banks LIMIT 10",
                            "dependencies": [],
                            "execution_type": "sequential"
                        }
                    ],
                    "execution_order": [["T1"]],
                    "synthesis_instructions": "Return results as is"
                }
        else:
            # Fallback
            return {
                "tasks": [
                    {
                        "task_id": "T1",
                        "description": "Execute query",
                        "sql_query": "SELECT * FROM banks LIMIT 10",
                        "dependencies": [],
                        "execution_type": "sequential"
                    }
                ],
                "execution_order": [["T1"]],
                "synthesis_instructions": "Return as is"
            }
    
    async def synthesize_to_sql(self, original_query: str, tasks_description: str, 
                                synthesis_instructions: str, execution_order: List[List[str]]) -> str:
        """
        Synthesize multiple tasks into final SQL
        """
        if self.provider == "openai" and self.client:
            prompt = f"""
            Combine these SQL tasks into a single, efficient SQL query.
            
            Original Request: {original_query}
            
            Tasks and SQL:
            {tasks_description}
            
            Synthesis Instructions: {synthesis_instructions}
            
            Create a single SQL query that accomplishes all tasks efficiently.
            Use CTEs or subqueries as needed.
            
            Return only the final SQL query.
            """
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a SQL synthesis expert. Combine multiple queries efficiently."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=1000
                )
                
                sql = response.choices[0].message.content.strip()
                sql = re.sub(r'```sql?\n?', '', sql)
                sql = sql.replace('```', '')
                
                return sql
                
            except Exception as e:
                logger.error(f"SQL synthesis failed: {e}")
                return f"-- Synthesis failed: {e}\nSELECT * FROM banks LIMIT 10"
        else:
            return "-- No LLM available for synthesis\nSELECT * FROM banks LIMIT 10"
    
    async def modify_sql_with_context(self, task_description: str, 
                                      original_sql: str, dependency_context: str) -> str:
        """
        Modify SQL based on dependency results
        """
        if self.provider == "openai" and self.client:
            prompt = f"""
            Modify this SQL based on dependency results.
            
            Task: {task_description}
            Original SQL: {original_sql}
            
            Dependency Results:
            {dependency_context}
            
            Return only the modified SQL.
            """
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "Modify SQL based on context. Return only SQL."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=500
                )
                
                sql = response.choices[0].message.content.strip()
                sql = re.sub(r'```sql?\n?', '', sql)
                sql = sql.replace('```', '')
                
                return sql
                
            except Exception as e:
                logger.error(f"SQL modification failed: {e}")
                return original_sql
        else:
            return original_sql
    
    async def synthesize_results(self, original_query: str, 
                                 synthesis_instructions: str, results_summary: str) -> Dict[str, Any]:
        """
        Synthesize execution results into final answer
        """
        if self.provider == "openai" and self.client:
            prompt = f"""
            Synthesize these query results into a final answer.
            
            Original Question: {original_query}
            Instructions: {synthesis_instructions}
            
            Results:
            {results_summary}
            
            Return a JSON with:
            {{
                "answer": "main answer",
                "key_insights": ["insight1", "insight2"],
                "recommendations": ["rec1", "rec2"],
                "summary_metrics": {{"metric": value}},
                "final_sql": "the final SQL if applicable"
            }}
            """
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "Synthesize query results. Return valid JSON."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=800
                )
                
                content = response.choices[0].message.content.strip()
                
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group())
                    else:
                        return {
                            "answer": "Results processed",
                            "key_insights": [],
                            "recommendations": [],
                            "summary_metrics": {},
                            "final_sql": None
                        }
                        
            except Exception as e:
                logger.error(f"Results synthesis failed: {e}")
                return {
                    "answer": "Query executed",
                    "key_insights": [],
                    "recommendations": [],
                    "summary_metrics": {},
                    "final_sql": None
                }
        else:
            return {
                "answer": "Query completed",
                "key_insights": ["Data retrieved"],
                "recommendations": [],
                "summary_metrics": {},
                "final_sql": None
            }
        
    async def generate_thought(self, user_input: str, context: Dict[str, Any]) -> str:
        """
        Generate ReAct thought for next action
        
        Args:
            user_input: User's query
            context: Current conversation context
            
        Returns:
            Formatted thought, action, and action input
        """
        if self.provider == "openai" and self.client:
            prompt = f"""
Given the user input and context, think step-by-step about what action to take.

User Input: {user_input}

Context:
- Has Previous SQL: {context.get('last_sql') is not None}
- Has Active Results: {context.get('has_active_data', False)}
- Recent Queries: {context.get('recent_queries', [])}

Format your response EXACTLY as:
Thought: [Your reasoning about what the user wants and what to do]
Action: [One of: GENERATE_SQL, MODIFY_SQL, EXECUTE_SQL, ANALYZE_RESULTS, ANSWER_FOLLOWUP]
Action Input: [The input needed for the action]

Think step by step about:
1. What is the user asking for?
2. Do we have context from previous queries?
3. What action would best address their need?
"""
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a ReAct SQL agent. Reason about the next action to take."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=300
                )
                
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                logger.error(f"Thought generation failed: {e}")
                # Fallback response
                if context.get('last_sql'):
                    return f"Thought: User might want to modify existing query\nAction: MODIFY_SQL\nAction Input: {user_input}"
                else:
                    return f"Thought: Generate new SQL query\nAction: GENERATE_SQL\nAction Input: {user_input}"
        else:
            # Heuristic fallback
            if context.get('last_sql') and any(word in user_input.lower() for word in ['add', 'modify', 'change']):
                return f"Thought: Modify existing query\nAction: MODIFY_SQL\nAction Input: {user_input}"
            elif context.get('has_active_data') and '?' in user_input:
                return f"Thought: Answer question about results\nAction: ANSWER_FOLLOWUP\nAction Input: {user_input}"
            else:
                return f"Thought: Generate new SQL\nAction: GENERATE_SQL\nAction Input: {user_input}"
    
    async def modify_sql_with_instruction(self, current_sql: str, 
                                         instruction: str, schema: str) -> str:
        """
        Modify existing SQL based on user instruction
        
        Args:
            current_sql: The current SQL query
            instruction: Modification instruction from user
            schema: Schema context
            
        Returns:
            Modified SQL query
        """
        if self.provider == "openai" and self.client:
            prompt = f"""
Modify the following SQL query based on the user's instruction.

Current SQL:
{current_sql}

User's Modification Request:
{instruction}

Available Schema:
{schema}

Rules:
1. Keep the original query structure when possible
2. Only modify what the user requested
3. Ensure the modified query is valid
4. Add comments to show what was changed
5. Return ONLY the modified SQL query

Examples:
- If user says "add column X", add X to SELECT clause
- If user says "filter by Y", add Y to WHERE clause
- If user says "sort by Z", add/modify ORDER BY clause
"""
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a SQL expert. Modify SQL queries based on instructions while preserving the original intent."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=500
                )
                
                sql = response.choices[0].message.content.strip()
                
                # Clean SQL from markdown
                sql = re.sub(r'```sql?\n?', '', sql)
                sql = sql.replace('```', '')
                
                return sql
                
            except Exception as e:
                logger.error(f"SQL modification failed: {e}")
                return current_sql
        else:
            # Simple heuristic modification
            instruction_lower = instruction.lower()
            
            if 'add' in instruction_lower:
                # Try to add to SELECT clause
                return current_sql.replace('SELECT', f'SELECT /* Added: {instruction} */', 1)
            elif 'filter' in instruction_lower or 'where' in instruction_lower:
                # Add to WHERE clause
                if 'WHERE' in current_sql.upper():
                    return current_sql.replace('WHERE', f'WHERE /* {instruction} */ ', 1)
                else:
                    return current_sql.replace('FROM', f'FROM /* Filter: {instruction} */\nWHERE ', 1)
            else:
                return current_sql + f"\n-- Modification requested: {instruction}"
    
    async def analyze_results_with_question(self, question: str, 
                                           results_summary: Dict[str, Any]) -> str:
        """
        Analyze query results to answer a specific question
        
        Args:
            question: User's question about the results
            results_summary: Summary of query results
            
        Returns:
            Analysis answering the question
        """
        if self.provider == "openai" and self.client:
            prompt = f"""
Analyze these query results to answer the user's question.

Question: {question}

Results Summary:
- Columns: {', '.join(results_summary.get('columns', [])[:10])}
- Row Count: {results_summary.get('row_count', 0)}
- Sample Data (first 3 rows): {results_summary.get('sample_rows', [])[:3]}

Provide a concise, direct answer to the question based on the data.
Focus on:
1. Directly answering what was asked
2. Using specific numbers/values from the data
3. Explaining any patterns or insights
"""
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a data analyst. Answer questions based on query results concisely and accurately."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=300
                )
                
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                logger.error(f"Results analysis failed: {e}")
                return self._fallback_analysis(question, results_summary)
        else:
            return self._fallback_analysis(question, results_summary)
    
    def _fallback_analysis(self, question: str, results_summary: Dict[str, Any]) -> str:
        """Simple fallback analysis when LLM is not available"""
        row_count = results_summary.get('row_count', 0)
        columns = results_summary.get('columns', [])
        
        if 'why' in question.lower():
            return f"Based on the {row_count} rows of data, this appears to be related to the values in columns: {', '.join(columns[:3])}"
        elif 'how many' in question.lower():
            return f"The query returned {row_count} rows"
        elif 'what' in question.lower():
            return f"The data shows {row_count} records with the following attributes: {', '.join(columns[:5])}"
        else:
            return f"The query returned {row_count} rows with {len(columns)} columns of data"        
        
# Singleton instance
sql_converter = AgenticLLMInterface()        