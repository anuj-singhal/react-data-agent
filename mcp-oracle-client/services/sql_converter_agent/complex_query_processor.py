"""
Robust Complex Query Processor - Handles complex SQL query generation
Focuses on reliable SQL output rather than strict decomposition adherence
"""

import logging
import json
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)

class TaskType(Enum):
    FILTER = "filter"
    AGGREGATION = "aggregation"
    JOIN = "join"
    RANKING = "ranking"
    TIME_SERIES = "time_series"
    CALCULATION = "calculation"
    ANALYSIS = "analysis"

@dataclass
class QueryTask:
    """Represents a single task in a decomposed query"""
    task_id: str
    task_type: TaskType
    description: str
    tables: List[str]
    columns: List[str]
    conditions: List[str]
    sql_query: Optional[str] = None
    dependencies: List[str] = None
    result: Optional[Dict[str, Any]] = None

@dataclass
class DecomposedQuery:
    """Represents a decomposed complex query"""
    original_query: str
    tasks: List[QueryTask]
    execution_order: List[str]
    synthesis_plan: str
    final_sql: Optional[str] = None

class ComplexQueryProcessor:
    """Robust complex query processor that focuses on reliable SQL generation"""
    
    def __init__(self, llm_interface, rag_agent):
        self.llm_interface = llm_interface
        self.rag_agent = rag_agent
        self.max_attempts = 3  # Number of attempts for operations
        
    async def is_complex_query(self, query: str, context: Dict[str, Any]) -> bool:
        """
        Determine if a query is complex and needs special processing
        """
        # Simple heuristic approach for reliability
        query_lower = query.lower()
        
        # Check for complex indicators
        complex_indicators = [
            ('multi-step', ['steps', 'step 1', 'step 2', 'analyze', 'comprehensive', 'first', 'then', 'finally']),
            ('comparison', ['compare', 'versus', 'vs', 'difference', 'correlation']),
            ('time series', ['trend', 'over time', 'quarter-over-quarter', 'year-over-year', 'month-over-month']),
            ('multiple metrics', ['ratio', 'rate', 'growth', 'performance', 'metrics', 'kpis']),
            ('aggregations', ['average', 'total', 'sum', 'count', 'min', 'max']),
            ('ranking', ['rank', 'top', 'bottom', 'best', 'worst']),
        ]
        
        # Count matches
        complexity_score = 0
        for indicator_type, patterns in complex_indicators:
            if any(pattern in query_lower for pattern in patterns):
                complexity_score += 1
        
        # Check for numeric references
        numbers = re.findall(r'\d+\.', query_lower)
        if numbers:
            complexity_score += 1
        
        # Check for task lists
        list_patterns = [r'\d+\.\s+[A-Z]', r'-\s+[A-Z]', r'\n\s*\d+\.']
        if any(re.search(pattern, query) for pattern in list_patterns):
            complexity_score += 2
        
        # Check number of tables
        num_tables = len(context.get('tables', {}))
        if num_tables >= 3:
            complexity_score += 1
        
        # Check query length - long queries are often complex
        if len(query.split()) > 40:
            complexity_score += 1
        
        logger.info(f"Query complexity score: {complexity_score}, Is complex: {complexity_score >= 3}")
        return complexity_score >= 3
    
    async def process_complex_query(self, query: str, context: Dict[str, Any], 
                                   intent: str = "new_query", 
                                   previous_messages: Optional[str] = None,
                                   last_sql: Optional[str] = None) -> str:
        """
        Main entry point for complex query processing
        Returns a single SQL query
        """
        logger.info("Starting robust complex query processing...")
        
        # Multi-approach strategy for reliable results
        approaches = [
            self._approach_direct_with_plan,
            self._approach_cte_based,
            self._approach_decompose_synthesize
        ]
        
        errors = []
        
        # Try each approach until one succeeds
        for i, approach in enumerate(approaches):
            try:
                logger.info(f"Trying approach {i+1}/{len(approaches)}: {approach.__name__}")
                sql_query = await approach(query, context, intent, previous_messages, last_sql)
                
                # Validate the query
                if sql_query and "SELECT" in sql_query.upper():
                    logger.info(f"Approach {i+1} succeeded")
                    return sql_query
                else:
                    logger.warning(f"Approach {i+1} returned invalid SQL")
                    errors.append(f"Approach {i+1} returned invalid SQL")
            except Exception as e:
                logger.error(f"Error in approach {i+1}: {str(e)}")
                errors.append(f"Approach {i+1} error: {str(e)}")
        
        # If all approaches fail, use a simple fallback
        logger.warning("All complex query approaches failed. Using fallback.")
        return await self._fallback_generation(query, context, intent, previous_messages, last_sql, 
                                              errors=errors)
    
    async def _approach_direct_with_plan(self, query: str, context: Dict[str, Any],
                                       intent: str, previous_messages: Optional[str],
                                       last_sql: Optional[str]) -> str:
        """
        Approach 1: Direct generation with explicit planning
        """
        # First, create a detailed plan
        plan_prompt = f"""
You are an expert SQL developer. First plan, then generate a complex SQL query.

Natural Language Query:
{query}

Database Schema:
{self._format_schema_summary(context)}

STEP 1: Create a detailed plan for your SQL query. Think about:
1. What tables and joins you'll need
2. What calculations and aggregations are required
3. How to structure the query with appropriate CTEs, window functions, etc.
4. Any specific SQL techniques needed (window functions, subqueries, etc.)

OUTPUT YOUR PLAN:
"""
        
        try:
            # Generate plan
            plan_response = await self.llm_interface.client.chat.completions.create(
                model=self.llm_interface.model,
                messages=[
                    {"role": "system", "content": "You are an expert SQL planner. Create a detailed plan."},
                    {"role": "user", "content": plan_prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            plan = plan_response.choices[0].message.content.strip()
            logger.info(f"Generated SQL plan: {plan[:200]}...")
            
            # Now generate SQL using the plan
            sql_prompt = f"""
You are an expert SQL developer. Generate a complex SQL query following this plan.

Natural Language Query:
{query}

Your SQL Plan:
{plan}

Database Schema:
{self._format_schema_context(context)}

Now, write a single, efficient SQL query that implements this plan. The SQL should:
1. Use CTEs for readability and modularity
2. Use appropriate window functions for ranking and comparisons
3. Handle all requirements from the original query
4. Be optimized and well-structured
5. Return only final sql query in this format ```sql <SQL Query> ```
6. Do not include any Explanations:


FINAL SQL QUERY:
"""
            
            # Generate SQL from plan
            sql_response = await self.llm_interface.client.chat.completions.create(
                model=self.llm_interface.model,
                messages=[
                    {"role": "system", "content": "You are an expert SQL developer. Generate efficient and correct SQL following the provided plan."},
                    {"role": "user", "content": sql_prompt}
                ],
                temperature=0.1,
                max_tokens=1500
            )
            
            sql = sql_response.choices[0].message.content.strip()
            
            # Clean the SQL
            sql = sql.replace("```sql", "").replace("```", "").strip()
            
            return sql
            
        except Exception as e:
            logger.error(f"Error in direct with plan approach: {e}")
            raise
    
    async def _approach_cte_based(self, query: str, context: Dict[str, Any],
                                intent: str, previous_messages: Optional[str],
                                last_sql: Optional[str]) -> str:
        """
        Approach 2: CTE-based decomposition
        """
        # Identify main analysis steps
        analysis_steps_prompt = f"""
You are an expert in SQL query planning. Break down this complex analysis into distinct steps:

Natural Language Query:
{query}

Identify 3-5 key analysis steps, each representing a CTE in a final SQL query.
Each step should be something that can be calculated with SQL.
Format each step as a numbered item.
Keep your response concise - just the numbered steps.
"""
        
        try:
            # Get analysis steps
            steps_response = await self.llm_interface.client.chat.completions.create(
                model=self.llm_interface.model,
                messages=[
                    {"role": "system", "content": "You are an SQL analysis expert. Break complex queries into steps."},
                    {"role": "user", "content": analysis_steps_prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            steps = steps_response.choices[0].message.content.strip()
            logger.info(f"Analysis steps: {steps}")
            
            # Now generate CTE-based SQL
            cte_prompt = f"""
You are an expert SQL developer. Create a complex SQL query using Common Table Expressions (CTEs).

Natural Language Query:
{query}

Analysis Steps:
{steps}

Database Schema:
{self._format_schema_context(context)}

Create a SQL query with the following structure:
1. Each analysis step should be its own CTE
2. The CTEs should build on each other as needed
3. The final query should combine or use the CTEs to produce the final result
4. Use appropriate SQL techniques (window functions, subqueries, etc.)
5. Ensure all tables and columns referenced exist in the schema

FINAL SQL QUERY:
"""
            
            # Generate SQL
            sql_response = await self.llm_interface.client.chat.completions.create(
                model=self.llm_interface.model,
                messages=[
                    {"role": "system", "content": "You are an expert SQL developer focused on CTE-based queries."},
                    {"role": "user", "content": cte_prompt}
                ],
                temperature=0.1,
                max_tokens=1500
            )
            
            sql = sql_response.choices[0].message.content.strip()
            
            # Clean the SQL
            sql = sql.replace("```sql", "").replace("```", "").strip()
            
            return sql
            
        except Exception as e:
            logger.error(f"Error in CTE-based approach: {e}")
            raise
    
    async def _approach_decompose_synthesize(self, query: str, context: Dict[str, Any],
                                          intent: str, previous_messages: Optional[str],
                                          last_sql: Optional[str]) -> str:
        """
        Approach 3: Traditional decompose and synthesize
        """
        try:
            # Decompose the query
            decomposed = await self.decompose_query(query, context)
            
            if not decomposed.tasks:
                raise ValueError("Decomposition failed to identify tasks")
            
            # Generate SQL for high-priority tasks only
            if len(decomposed.tasks) > 3:
                # Focus on key tasks for better reliability
                key_tasks = decomposed.tasks[:3]  # First 3 tasks
                logger.info(f"Focusing on {len(key_tasks)} key tasks out of {len(decomposed.tasks)} total")
                simplified_decomposed = DecomposedQuery(
                    original_query=decomposed.original_query,
                    tasks=key_tasks,
                    execution_order=decomposed.execution_order[:3],
                    synthesis_plan=decomposed.synthesis_plan,
                    final_sql=None
                )
                decomposed = simplified_decomposed
            
            # Generate SQL for tasks
            await self._generate_task_queries_robust(decomposed, context)
            
            # Synthesize final SQL
            final_sql = await self._synthesize_queries_robust(
                decomposed, query, context, intent, previous_messages, last_sql
            )
            
            return final_sql
            
        except Exception as e:
            logger.error(f"Error in decompose-synthesize approach: {e}")
            raise
    
    async def decompose_query(self, query: str, context: Dict[str, Any]) -> DecomposedQuery:
        """
        Decompose a complex query into tasks
        """
        logger.info("Starting query decomposition...")
        
        # Build decomposition prompt
        prompt = f"""
You are an expert SQL query decomposer. Break down this complex query into manageable tasks:

COMPLEX QUERY: {query}

AVAILABLE TABLES:
{self._format_schema_summary(context)}

Please decompose into tasks following this JSON structure:
{{
    "tasks": [
        {{
            "task_id": "T1",
            "task_type": "filter|aggregation|join|ranking|time_series|calculation|analysis",
            "description": "Clear description of what this task does",
            "tables": ["table1", "table2"],
            "columns": ["col1", "col2"],
            "conditions": ["condition1", "condition2"],
            "dependencies": []
        }}
    ],
    "execution_order": ["T1", "T2", "T3"],
    "synthesis_plan": "Description of how to combine the tasks"
}}

IMPORTANT GUIDELINES:
1. Each task should be simple and focused on one operation
2. Tasks should be executable in the specified order
3. Keep to 3-5 tasks maximum for better reliability
4. Identify dependencies between tasks
5. The synthesis plan should explain how to combine results
"""
        
        for attempt in range(self.max_attempts):
            try:
                response = await self.llm_interface.client.chat.completions.create(
                    model=self.llm_interface.model,
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are an expert SQL query decomposer. Break down complex queries into manageable tasks."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=1500,
                    response_format={"type": "json_object"}
                )
                
                decomposition_text = response.choices[0].message.content.strip()
                
                # Parse the decomposition
                decomposition_data = json.loads(decomposition_text)
                
                # Create QueryTask objects
                tasks = []
                for task_data in decomposition_data.get('tasks', []):
                    # Ensure each task has the required fields
                    task_type_str = task_data.get('task_type', 'analysis').lower()
                    # Convert to TaskType enum or use ANALYSIS as default
                    try:
                        task_type = TaskType(task_type_str)
                    except ValueError:
                        task_type = TaskType.ANALYSIS
                        
                    task = QueryTask(
                        task_id=task_data.get('task_id', f"T{len(tasks)+1}"),
                        task_type=task_type,
                        description=task_data.get('description', ''),
                        tables=task_data.get('tables', []),
                        columns=task_data.get('columns', []),
                        conditions=task_data.get('conditions', []),
                        dependencies=task_data.get('dependencies', [])
                    )
                    tasks.append(task)
                
                # If no tasks were created, retry
                if not tasks:
                    if attempt < self.max_attempts - 1:
                        logger.warning(f"No tasks in decomposition (attempt {attempt+1}). Retrying...")
                        continue
                    else:
                        logger.warning("No tasks in decomposition after all attempts")
                        return DecomposedQuery(
                            original_query=query,
                            tasks=[],
                            execution_order=[],
                            synthesis_plan="Failed to decompose query",
                            final_sql=None
                        )
                
                logger.info(f"Query decomposed into {len(tasks)} tasks")
                
                return DecomposedQuery(
                    original_query=query,
                    tasks=tasks,
                    execution_order=decomposition_data.get('execution_order', [t.task_id for t in tasks]),
                    synthesis_plan=decomposition_data.get('synthesis_plan', ''),
                    final_sql=None
                )
                
            except json.JSONDecodeError:
                logger.error(f"Failed to parse decomposition JSON (attempt {attempt+1})")
                if attempt < self.max_attempts - 1:
                    continue
            except Exception as e:
                logger.error(f"Decomposition error (attempt {attempt+1}): {e}")
                if attempt < self.max_attempts - 1:
                    continue
        
        # If we get here, all attempts failed
        logger.warning("All decomposition attempts failed")
        return DecomposedQuery(
            original_query=query,
            tasks=[],
            execution_order=[],
            synthesis_plan="Decomposition failed",
            final_sql=None
        )
    
    async def _generate_task_queries_robust(self, decomposed_query: DecomposedQuery, 
                                         context: Dict[str, Any]) -> DecomposedQuery:
        """
        Generate SQL for each task with robust error handling
        """
        logger.info(f"Generating SQL for {len(decomposed_query.tasks)} tasks...")
        
        # Ensure we have table context
        if not context.get('tables'):
            logger.warning("No tables in context")
            return decomposed_query
        
        for task in decomposed_query.tasks:
            # Make multiple attempts for each task
            for attempt in range(self.max_attempts):
                try:
                    # Build comprehensive task context
                    task_context = self._build_robust_task_context(task, context)
                    
                    # Generate SQL using template specific to task type
                    prompt = self._create_task_sql_prompt(task, task_context)
                    
                    response = await self.llm_interface.client.chat.completions.create(
                        model=self.llm_interface.model,
                        messages=[
                            {
                                "role": "system", 
                                "content": f"You are an expert SQL developer specializing in {task.task_type.value} queries. Generate clean, efficient SQL."
                            },
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.1,
                        max_tokens=800
                    )
                    
                    sql = response.choices[0].message.content.strip()
                    
                    # Clean the SQL
                    sql = sql.replace("```sql", "").replace("```", "").strip()
                    
                    # Validate SQL is non-empty and starts with SELECT
                    if sql and "SELECT" in sql.upper():
                        task.sql_query = sql
                        logger.info(f"Generated SQL for task {task.task_id} (attempt {attempt+1})")
                        break
                    else:
                        logger.warning(f"Invalid SQL for task {task.task_id} (attempt {attempt+1})")
                        if attempt < self.max_attempts - 1:
                            continue
                except Exception as e:
                    logger.error(f"Failed to generate SQL for task {task.task_id} (attempt {attempt+1}): {str(e)}")
                    if attempt < self.max_attempts - 1:
                        continue
            
            # If all attempts failed, add a placeholder
            if not task.sql_query:
                task.sql_query = f"-- Failed to generate SQL for task {task.task_id}"
                logger.warning(f"All attempts failed for task {task.task_id}")
        
        return decomposed_query
    
    async def _synthesize_queries_robust(self, decomposed_query: DecomposedQuery,
                                       original_query: str,
                                       context: Dict[str, Any],
                                       intent: str,
                                       previous_messages: Optional[str],
                                       last_sql: Optional[str]) -> str:
        """
        Synthesize all task queries into a single final query with improved robustness
        """
        logger.info("Synthesizing task queries into final SQL...")
        
        # Check if any tasks have valid SQL
        valid_tasks = [t for t in decomposed_query.tasks 
                      if t.sql_query and not t.sql_query.startswith('--')]
        
        if not valid_tasks:
            logger.warning("No valid task SQL to synthesize")
            return await self._fallback_generation(original_query, context, intent, 
                                                 previous_messages, last_sql)
        
        # Try different synthesis approaches
        synthesis_approaches = [
            self._synthesize_with_ctes,
            self._synthesize_with_instructions,
            self._synthesize_direct
        ]
        
        for i, approach in enumerate(synthesis_approaches):
            try:
                logger.info(f"Trying synthesis approach {i+1}/{len(synthesis_approaches)}")
                sql = await approach(decomposed_query, original_query, context)
                
                if sql and "SELECT" in sql.upper():
                    logger.info(f"Synthesis successful with approach {i+1}")
                    return sql
                else:
                    logger.warning(f"Synthesis approach {i+1} returned invalid SQL")
            except Exception as e:
                logger.error(f"Error in synthesis approach {i+1}: {str(e)}")
        
        # If all synthesis approaches fail, try a direct generation
        logger.warning("All synthesis approaches failed, using fallback")
        return await self._fallback_generation(original_query, context, intent, 
                                             previous_messages, last_sql)
    
    async def _synthesize_with_ctes(self, decomposed_query: DecomposedQuery,
                                 original_query: str, context: Dict[str, Any]) -> str:
        """
        Synthesize with explicit CTE structure
        """
        # Create task details for prompt
        task_details = []
        for task in decomposed_query.tasks:
            if task.sql_query and not task.sql_query.startswith('--'):
                task_details.append(f"""
Task {task.task_id}: {task.description}
Task Type: {task.task_type.value}
SQL:
{task.sql_query}
""")
        
        task_sql = "\n".join(task_details)
        
        prompt = f"""
You are an expert SQL developer. Combine these SQL queries into one efficient query using CTEs.

Original Request: {original_query}

Task Queries:
{task_sql}

Synthesis Instructions:
1. Create a WITH clause with a CTE for each task
2. Use meaningful CTE names based on the task descriptions
3. Ensure proper reference between CTEs based on dependencies
4. Create a final SELECT statement that uses the CTEs to answer the original request

Return ONLY the final combined SQL query with no explanations.
"""
        
        response = await self.llm_interface.client.chat.completions.create(
            model=self.llm_interface.model,
            messages=[
                {
                    "role": "system", 
                    "content": "You are an expert SQL synthesizer. Combine multiple queries into one efficient query using CTEs."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=1500
        )
        
        sql = response.choices[0].message.content.strip()
        
        # Clean the SQL
        sql = sql.replace("```sql", "").replace("```", "").strip()
        
        return sql
    
    async def _synthesize_with_instructions(self, decomposed_query: DecomposedQuery,
                                        original_query: str, context: Dict[str, Any]) -> str:
        """
        Synthesize with detailed instructions
        """
        # Create task details for prompt
        task_details = []
        for task in decomposed_query.tasks:
            if task.sql_query and not task.sql_query.startswith('--'):
                task_details.append(f"""
Task {task.task_id}: {task.description}
SQL:
{task.sql_query}
""")
        
        task_sql = "\n".join(task_details)
        
        prompt = f"""
You are an expert SQL developer. Create a single comprehensive SQL query that answers this request.

Original Request: {original_query}

Here are component queries that address parts of the request:
{task_sql}

Using the Database Schema:
{self._format_schema_summary(context)}

Synthesis Instructions:
1. Create a single SQL query that addresses ALL requirements in the original request
2. Use CTEs (WITH clauses) to organize the query
3. Use window functions, joins, and subqueries as needed
4. The query should produce a clear, complete result that addresses the original request
5. Use insights from the component queries but create your own optimized solution
6. Focus on correctness and completeness

Return ONLY the final SQL query with no explanations.
"""
        
        response = await self.llm_interface.client.chat.completions.create(
            model=self.llm_interface.model,
            messages=[
                {
                    "role": "system", 
                    "content": "You are an expert SQL synthesizer. Create comprehensive SQL queries from requirements."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=1500
        )
        
        sql = response.choices[0].message.content.strip()
        
        # Clean the SQL
        sql = sql.replace("```sql", "").replace("```", "").strip()
        
        return sql
    
    async def _synthesize_direct(self, decomposed_query: DecomposedQuery,
                             original_query: str, context: Dict[str, Any]) -> str:
        """
        Direct synthesis without detailed instructions
        """
        prompt = f"""
Generate a single, comprehensive SQL query that fully addresses this request:

{original_query}

Using this database schema:
{self._format_schema_summary(context)}

The query should:
1. Use WITH clauses (CTEs) for modularity
2. Use window functions for any ranking or comparative analysis
3. Include all tables and joins needed for the analysis
4. Produce a complete result that addresses all aspects of the request

Return ONLY the SQL query with no explanations.
"""
        
        response = await self.llm_interface.client.chat.completions.create(
            model=self.llm_interface.model,
            messages=[
                {
                    "role": "system", 
                    "content": "You are an expert SQL developer. Generate complete, efficient SQL queries."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=1500
        )
        
        sql = response.choices[0].message.content.strip()
        
        # Clean the SQL
        sql = sql.replace("```sql", "").replace("```", "").strip()
        
        return sql
    
    async def _fallback_generation(self, query: str, context: Dict[str, Any], 
                               intent: str, previous_messages: Optional[str],
                               last_sql: Optional[str], errors: List[str] = None) -> str:
        """
        Fallback to direct SQL generation with schema context
        """
        logger.info("Using direct SQL generation fallback")
        
        # Create a comprehensive prompt with all context
        prompt = f"""
You must create a SINGLE, COMPREHENSIVE SQL query for this request:

{query}

Database schema:
{self._format_schema_context(context)}

Your task:
1. Create a single SQL query that fully addresses the request
2. Use WITH clauses (CTEs) to organize and modularize the query
3. Use appropriate SQL techniques (window functions, subqueries, JOIN operations)
4. Include all required tables and conditions
5. Produce a complete result

Previous approaches failed with these errors:
{json.dumps(errors or [], indent=2)}

Return ONLY the SQL query with no explanations or comments.
"""
        
        try:
            response = await self.llm_interface.client.chat.completions.create(
                model=self.llm_interface.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert SQL developer. Generate complete, correct SQL queries with no explanations."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1500
            )
            
            sql = response.choices[0].message.content.strip()
            
            # Clean the SQL
            sql = sql.replace("```sql", "").replace("```", "").strip()
            
            if sql and "SELECT" in sql.upper():
                return sql
            else:
                logger.warning("Fallback generation returned invalid SQL")
                # Last resort - extremely simple query
                return f"SELECT * FROM {list(context.get('tables', {}).keys())[0]} LIMIT 10"
            
        except Exception as e:
            logger.error(f"Error in fallback generation: {str(e)}")
            # Absolute last resort
            return f"SELECT * FROM {list(context.get('tables', {}).keys())[0]} LIMIT 10"
    
    def _build_robust_task_context(self, task: QueryTask, full_context: Dict[str, Any]) -> Dict[str, Any]:
        """Build comprehensive context for a task with all necessary attributes"""
        # Create a base context with all key attributes
        task_context = {
            'user_query': task.description,
            'tables': {},
            'relationships': [],
            'business_rules': full_context.get('business_rules', []),
            'sql_rules': full_context.get('sql_rules', []),
            'sample_values': full_context.get('sample_values', {})
        }
        
        # Include tables mentioned in task.tables
        tables_to_include = set(task.tables)
        
        # If no tables specified, include all tables
        if not tables_to_include:
            tables_to_include = set(full_context.get('tables', {}).keys())
        
        # Add related tables based on relationships
        for rel in full_context.get('relationships', []):
            from_table = rel.get('from', '').split('.')[0] if '.' in rel.get('from', '') else ''
            to_table = rel.get('to', '').split('.')[0] if '.' in rel.get('to', '') else ''
            
            if from_table in tables_to_include:
                tables_to_include.add(to_table)
            if to_table in tables_to_include:
                tables_to_include.add(from_table)
        
        # Include all specified tables
        for table_name in tables_to_include:
            if table_name in full_context.get('tables', {}):
                task_context['tables'][table_name] = full_context['tables'][table_name]
        
        # Include relevant relationships
        for rel in full_context.get('relationships', []):
            from_table = rel.get('from', '').split('.')[0] if '.' in rel.get('from', '') else ''
            to_table = rel.get('to', '').split('.')[0] if '.' in rel.get('to', '') else ''
            
            if from_table in tables_to_include and to_table in tables_to_include:
                task_context['relationships'].append(rel)
        
        # Copy any other attributes from full context
        for key, value in full_context.items():
            if key not in task_context:
                task_context[key] = value
        
        return task_context
    
    def _create_task_sql_prompt(self, task: QueryTask, context: Dict[str, Any]) -> str:
        """Create a specialized prompt based on task type"""
        base_prompt = f"""
Generate SQL for this task:

Task Description: {task.description}
Task Type: {task.task_type.value}

Database Schema:
{self._format_schema_context(context)}

"""
        
        # Add task-specific guidance based on type
        if task.task_type == TaskType.FILTER:
            base_prompt += """
Guidance for Filter Task:
- Create a query that filters data based on specific conditions
- Ensure appropriate WHERE clauses are used
- Focus on accurate filter conditions
"""
        elif task.task_type == TaskType.AGGREGATION:
            base_prompt += """
Guidance for Aggregation Task:
- Use appropriate aggregation functions (SUM, AVG, COUNT, etc.)
- Include proper GROUP BY clauses
- Consider HAVING clauses for filtered aggregations
"""
        elif task.task_type == TaskType.JOIN:
            base_prompt += """
Guidance for Join Task:
- Use appropriate JOIN types (INNER, LEFT, RIGHT)
- Ensure proper join conditions
- Include all necessary tables
"""
        elif task.task_type == TaskType.RANKING:
            base_prompt += """
Guidance for Ranking Task:
- Use window functions (ROW_NUMBER, RANK, DENSE_RANK)
- Include appropriate PARTITION BY and ORDER BY clauses
- Consider filters for top/bottom N results
"""
        elif task.task_type == TaskType.TIME_SERIES:
            base_prompt += """
Guidance for Time Series Task:
- Use window functions for period-over-period comparisons
- Consider LAG/LEAD functions for sequential comparisons
- Ensure proper time period filtering and grouping
"""
        elif task.task_type == TaskType.CALCULATION:
            base_prompt += """
Guidance for Calculation Task:
- Implement the required mathematical operations
- Consider NULL handling with COALESCE or NULLIF
- Use CASE statements for conditional logic if needed
"""
        else:  # Analysis or other types
            base_prompt += """
Guidance for Analysis Task:
- Create a comprehensive query that addresses the task
- Use appropriate SQL techniques (CTEs, window functions, etc.)
- Focus on producing clear, accurate results
"""
        
        # Add final instructions
        base_prompt += """
Return ONLY the SQL query with no explanations or comments.
"""
        
        return base_prompt
    
    def _format_schema_context(self, context: Dict[str, Any]) -> str:
        """Format full schema context for prompts"""
        schema_text = ""
        
        # Add table information
        for table_name, table_info in context.get('tables', {}).items():
            schema_text += f"\nTABLE: {table_name}"
            schema_text += f"\nDescription: {table_info.get('description', '')}"
            schema_text += "\nColumns:"
            
            for col in table_info.get('columns', []):
                pk_text = " [PK]" if col.get('is_pk') else ""
                schema_text += f"\n  - {col.get('name', '')} ({col.get('type', '')}){pk_text}: {col.get('description', '')}"
                
                if 'samples' in col:
                    samples = col.get('samples', [])
                    if samples:
                        schema_text += f" (Examples: {', '.join([str(s) for s in samples[:3]])})"
        
        # Add relationships
        if context.get('relationships'):
            schema_text += "\n\nRELATIONSHIPS:"
            for rel in context.get('relationships', []):
                schema_text += f"\n- {rel.get('from', '')} -> {rel.get('to', '')} ({rel.get('type', '')})"
        
        # Add business rules if present
        if context.get('business_rules'):
            schema_text += "\n\nBUSINESS RULES:"
            for rule in context.get('business_rules', []):
                if rule.get('active'):
                    schema_text += f"\n- {rule.get('name', '')}: {rule.get('rule', '')}"
        
        # Add SQL rules if present
        if context.get('sql_rules'):
            schema_text += "\n\nSQL GENERATION RULES:"
            for rule in context.get('sql_rules', []):
                if rule.get('active'):
                    schema_text += f"\n- {rule.get('name', '')}: {rule.get('rule', '')}"
        
        return schema_text
    
    def _format_schema_summary(self, context: Dict[str, Any]) -> str:
        """Format a concise schema summary for prompts"""
        summary = ""
        
        # Add table summaries
        tables = context.get('tables', {})
        for table_name, table_info in tables.items():
            summary += f"\nTABLE: {table_name} - {table_info.get('description', '')}"
            
            # Add key columns
            columns = table_info.get('columns', [])
            if columns:
                column_names = [col.get('name', '') for col in columns if col.get('is_pk') or 'id' in col.get('name', '').lower()]
                other_columns = [col.get('name', '') for col in columns if col.get('name', '') not in column_names]
                
                # Include primary keys and a few other columns
                column_names.extend(other_columns[:5])
                column_str = ', '.join(column_names)
                
                summary += f"\n  Columns: {column_str}"
                
                # Add '...' if there are more columns
                if len(columns) > len(column_names):
                    summary += f", ... ({len(columns) - len(column_names)} more)"
        
        # Add relationship summary
        relationships = context.get('relationships', [])
        if relationships:
            summary += "\n\nKey Relationships:"
            
            # Limit to a few important relationships
            for i, rel in enumerate(relationships[:5]):
                summary += f"\n- {rel.get('from', '')} -> {rel.get('to', '')}"
            
            if len(relationships) > 5:
                summary += f"\n- ... ({len(relationships) - 5} more relationships)"
        
        return summary