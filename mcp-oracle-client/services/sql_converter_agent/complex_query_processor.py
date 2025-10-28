"""
Complex Query Processor - Handles decomposition and synthesis of complex queries
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)

class TaskType(Enum):
    FILTER = "filter"
    AGGREGATION = "aggregation"
    JOIN = "join"
    RANKING = "ranking"
    TIME_SERIES = "time_series"
    CALCULATION = "calculation"

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
    """Handles complex query decomposition and synthesis"""
    
    def __init__(self, llm_interface, rag_agent):
        self.llm_interface = llm_interface
        self.rag_agent = rag_agent
        
    async def is_complex_query(self, query: str, context: Dict[str, Any]) -> bool:
        """
        Determine if a query is complex and needs decomposition
        """
        query_lower = query.lower()
        num_tables = len(context.get('tables', {}))
        
        # Complex patterns that require decomposition
        complex_patterns = [
            ('window function', ['over', 'partition by', 'row_number', 'rank']),
            ('multiple aggregations', ['sum', 'avg', 'count', 'max', 'min']),
            ('time series', ['year over year', 'quarter over quarter', 'trend']),
            ('complex joins', num_tables > 3),
            ('subqueries', ['exists', 'in (select', 'with']),
            ('ranking', ['top', 'bottom', 'rank', 'dense_rank']),
            ('pivot', ['pivot', 'unpivot', 'crosstab'])
        ]
        
        complexity_score = 0
        
        # Check for complex patterns
        for pattern_name, indicators in complex_patterns[:7]:
            if isinstance(indicators, list):
                if any(ind in query_lower for ind in indicators):
                    complexity_score += 1
                    logger.debug(f"Found complex pattern: {pattern_name}")
            elif isinstance(indicators, bool) and indicators:
                complexity_score += 1
                logger.debug(f"Found complex pattern: {pattern_name}")
        
        # Check multiple aggregations
        agg_functions = ['sum', 'avg', 'count', 'max', 'min', 'stddev', 'variance']
        agg_count = sum(1 for func in agg_functions if func in query_lower)
        if agg_count >= 3:
            complexity_score += 2
            
        # Check for multiple conditions
        condition_words = ['where', 'having', 'and', 'or', 'between', 'in']
        condition_count = sum(1 for word in condition_words if word in query_lower)
        if condition_count >= 5:
            complexity_score += 1
            
        # Consider complex if score >= 3
        is_complex = complexity_score >= 3
        
        logger.info(f"Query complexity score: {complexity_score}, Is complex: {is_complex}")
        return is_complex
    
    async def decompose_query(self, query: str, context: Dict[str, Any]) -> DecomposedQuery:
        """
        Decompose a complex query into smaller manageable tasks
        """
        logger.info("Starting query decomposition...")
        
        # Build decomposition prompt
        prompt = self._build_decomposition_prompt(query, context)
        
        # Get decomposition from LLM
        try:
            response = await self.llm_interface.client.chat.completions.create(
                model=self.llm_interface.model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert SQL query decomposer. Break down complex queries into simple, manageable tasks.
                        Each task should be a simple SQL operation that can be executed independently or with minimal dependencies.
                        Return a JSON structure with tasks and execution plan."""
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1500
            )
            
            decomposition_text = response.choices[0].message.content.strip()
            
            # Parse the decomposition
            decomposed = self._parse_decomposition(decomposition_text, query, context)
            
            logger.info(f"Query decomposed into {len(decomposed.tasks)} tasks")
            return decomposed
            
        except Exception as e:
            logger.error(f"Decomposition failed: {e}")
            # Return empty decomposition on failure
            return DecomposedQuery(
                original_query=query,
                tasks=[],
                execution_order=[],
                synthesis_plan="Direct execution - decomposition failed",
                final_sql=None
            )
    
    async def generate_task_queries(self, decomposed_query: DecomposedQuery, 
                                   context: Dict[str, Any], intent: str = "new_query",
                                   previous_messages: str = None, last_sql: str = None) -> DecomposedQuery:
        """
        Generate SQL for each task in the decomposed query
        Handles both new_query and modify_query intents
        """
        logger.info(f"Generating SQL for {len(decomposed_query.tasks)} tasks (intent: {intent})...")
        
        for task in decomposed_query.tasks:
            # Build task-specific context from the full RAG context
            task_context = self._build_task_context(task, context)
            
            # Add the task description as user_query for the LLM
            task_context['user_query'] = task.description
            
            try:
                if intent == "modify_query" and last_sql:
                    # For modify intent, pass the previous context
                    sql = await self.llm_interface.generate_sql(
                        task_context,
                        intent="modify_query",
                        previous_message=previous_messages,
                        last_sql=last_sql
                    )
                else:
                    # For new query intent
                    sql = await self.llm_interface.generate_sql(
                        task_context,
                        intent="new_query"
                    )
                
                task.sql_query = sql.strip()
                logger.debug(f"Generated SQL for task {task.task_id}: {task.sql_query[:100]}...")
                
            except Exception as e:
                logger.error(f"Failed to generate SQL for task {task.task_id}: {e}")
                task.sql_query = f"-- Failed to generate SQL for task {task.task_id}"
        
        return decomposed_query
    
    async def synthesize_queries(self, decomposed_query: DecomposedQuery, 
                                context: Dict[str, Any], intent: str = "new_query",
                                previous_messages: str = None, last_sql: str = None) -> str:
        """
        Synthesize all task queries into a single final query
        Handles both new_query and modify_query intents
        """
        logger.info(f"Synthesizing task queries into final SQL (intent: {intent})...")
        
        # Build synthesis prompt
        prompt = self._build_synthesis_prompt(decomposed_query, context)
        
        # Add modify context if needed
        if intent == "modify_query" and last_sql:
            prompt += f"\n\nPrevious SQL that needs modification:\n{last_sql}"
            prompt += f"\n\nModification context:\n{previous_messages}"
        
        try:
            response = await self.llm_interface.client.chat.completions.create(
                model=self.llm_interface.model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert SQL synthesizer. Combine multiple simple SQL queries into one efficient query.
                        Use CTEs, subqueries, or joins as appropriate. Ensure the final query is optimized and correct."""
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            final_sql = response.choices[0].message.content.strip()
            
            # Clean the SQL
            final_sql = final_sql.replace("```sql", "").replace("```", "").strip()
            
            decomposed_query.final_sql = final_sql
            logger.info("Successfully synthesized final SQL query")
            
            return final_sql
            
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            # Fallback: try to combine tasks manually
            return self._fallback_synthesis(decomposed_query)
    
    def _build_decomposition_prompt(self, query: str, context: Dict[str, Any]) -> str:
        """Build prompt for query decomposition"""
        
        # Get table and column info
        tables_info = []
        for table_name, table_data in context.get('tables', {}).items():
            columns = [col['name'] for col in table_data.get('columns', [])]
            tables_info.append(f"  - {table_name}: {', '.join(columns[:5])}...")
        
        prompt = f"""
        Decompose this complex query into simple, executable tasks:
        
        Query: {query}
        
        Available Tables:
        {chr(10).join(tables_info)}
        
        Relationships:
        {json.dumps(context.get('relationships', []), indent=2)}
        
        Please decompose into tasks following this JSON structure:
        {{
            "tasks": [
                {{
                    "task_id": "T1",
                    "task_type": "filter|aggregation|join|ranking|time_series|calculation",
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
        
        Guidelines:
        1. Each task should be simple and focused on one operation
        2. Identify dependencies between tasks
        3. Tasks should be executable in the specified order
        4. The synthesis plan should explain how to combine results
        """
        
        return prompt
    
    def _parse_decomposition(self, decomposition_text: str, query: str, 
                           context: Dict[str, Any]) -> DecomposedQuery:
        """Parse the LLM decomposition response"""
        
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', decomposition_text, re.DOTALL)
            if json_match:
                decomposition_data = json.loads(json_match.group())
            else:
                decomposition_data = json.loads(decomposition_text)
            
            # Create QueryTask objects
            tasks = []
            for task_data in decomposition_data.get('tasks', []):
                task = QueryTask(
                    task_id=task_data.get('task_id', f"T{len(tasks)+1}"),
                    task_type=TaskType(task_data.get('task_type', 'filter').lower()),
                    description=task_data.get('description', ''),
                    tables=task_data.get('tables', []),
                    columns=task_data.get('columns', []),
                    conditions=task_data.get('conditions', []),
                    dependencies=task_data.get('dependencies', [])
                )
                tasks.append(task)
            
            return DecomposedQuery(
                original_query=query,
                tasks=tasks,
                execution_order=decomposition_data.get('execution_order', []),
                synthesis_plan=decomposition_data.get('synthesis_plan', ''),
                final_sql=None
            )
            
        except Exception as e:
            logger.error(f"Failed to parse decomposition: {e}")
            return DecomposedQuery(
                original_query=query,
                tasks=[],
                execution_order=[],
                synthesis_plan="Parse failed",
                final_sql=None
            )
    
    def _build_task_context(self, task: QueryTask, full_context: Dict[str, Any]) -> Dict[str, Any]:
        """Build context specific to a task"""
        
        task_context = {
            'user_query': task.description,
            'tables': {},
            'relationships': []
        }
        
        # Include only relevant tables
        for table_name in task.tables:
            if table_name in full_context.get('tables', {}):
                task_context['tables'][table_name] = full_context['tables'][table_name]
        
        # Include relevant relationships
        for rel in full_context.get('relationships', []):
            if any(table in task.tables for table in [rel.get('from', '').split('.')[0], 
                                                       rel.get('to', '').split('.')[0]]):
                task_context['relationships'].append(rel)
        
        return task_context
    
    def _build_synthesis_prompt(self, decomposed_query: DecomposedQuery, 
                               context: Dict[str, Any]) -> str:
        """Build prompt for query synthesis"""
        
        # Collect task SQLs
        task_sqls = []
        for task in decomposed_query.tasks:
            if task.sql_query:
                task_sqls.append(f"""
                Task {task.task_id} ({task.task_type.value}):
                Description: {task.description}
                SQL: {task.sql_query}
                """)
        
        prompt = f"""
        Synthesize these task queries into a single, efficient SQL query:
        
        Original Request: {decomposed_query.original_query}
        
        Task Queries:
        {chr(10).join(task_sqls)}
        
        Synthesis Plan: {decomposed_query.synthesis_plan}
        
        Execution Order: {' -> '.join(decomposed_query.execution_order)}
        
        Guidelines:
        1. Combine the tasks into one efficient query
        2. Use CTEs (WITH clauses) for complex intermediate results
        3. Ensure proper join conditions and aggregations
        4. Optimize for performance
        5. Maintain the logical flow from the original request
        
        Return only the final SQL query.
        """
        
        return prompt
    
    def _fallback_synthesis(self, decomposed_query: DecomposedQuery) -> str:
        """Fallback synthesis when LLM synthesis fails"""
        
        if not decomposed_query.tasks:
            return "-- No tasks to synthesize"
        
        # Try to combine tasks with UNION or CTEs
        if len(decomposed_query.tasks) == 1:
            return decomposed_query.tasks[0].sql_query or "-- Single task SQL failed"
        
        # Build a simple CTE structure
        cte_parts = []
        for i, task in enumerate(decomposed_query.tasks):
            if task.sql_query:
                cte_parts.append(f"task_{task.task_id} AS ({task.sql_query})")
        
        if cte_parts:
            return f"""
            WITH {','.join(cte_parts)}
            SELECT * FROM task_{decomposed_query.tasks[-1].task_id}
            """
        
        return "-- Fallback synthesis failed"